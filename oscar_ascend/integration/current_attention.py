# Archive #55-#69/#126/#129-#142: native causal current-chunk FIA must return
# exact BF16 output and a complete LSE while CV keeps INT2 history/window.
# D.4 four questions: (1) this replaces only the current part of `fia`;
# (2) the failed route spent 6.5s dequantizing historical INT2; its separate
# FIA phase was ~7ms at 16K and 18.5-18.9ms at 32K, not a current-only timing;
# (3) no historical KV is materialized, only current BF16 Q/K/V enters FIA;
# (4) target NPU oracle, graph and paired speed remain independent gates.
"""Exact native current-chunk partial for OSCAR's three-source LSE merge."""

from __future__ import annotations

import math

import torch


class CurrentAttentionError(RuntimeError):
    pass


_PREFILL_STATES = frozenset({"PrefillNoCache", "ChunkedPrefill", "PrefillCacheHit"})
_DECODE_STATES = frozenset({"DecodeOnly", "SpecDecoding"})
_MASKS: dict[torch.device, torch.Tensor] = {}


def use_native_current(metadata) -> bool:
    """Use native current only in eager main-model prefill or mixed batches.

    The native runner's host AttentionState describes the scheduling stage.
    First and subsequent MTP draft metadata are explicitly marked is_draft;
    graph capture/replay records and reuses the existing complete CV route.
    No decision depends on request length, NPU tensor values, or a failed op.
    """
    # #142: native MTP warmup is labelled ChunkedPrefill but all its slots
    # are -1. The runner scope marks both warmup and capture explicitly;
    # preserve CV's full padding path without disabling the real-slot guard.
    if metadata.capture_origin or metadata.is_draft or metadata.dummy_origin:
        return False
    state = metadata.attn_state
    if type(state).__name__ != "AscendAttentionState" or not isinstance(getattr(state, "name", None), str):
        raise CurrentAttentionError("native AscendAttentionState is required for main-model attention")
    if state.name in _PREFILL_STATES:
        return True
    if state.name in _DECODE_STATES:
        return False
    raise CurrentAttentionError(f"unrecognized native Ascend attention state {state.name!r}")


def current_cumulative_lengths(metadata, active_tokens: int) -> list[int]:
    """Read only the runner's already-CPU qstart mirror; drop dummy padding."""
    starts = metadata.query_start_loc_cpu
    if (starts is None or not isinstance(starts, torch.Tensor)
            or starts.device.type != "cpu" or starts.dtype not in (torch.int32, torch.int64)
            or starts.ndim != 1 or starts.numel() < metadata.num_reqs + 1):
        raise CurrentAttentionError("native CPU query-start mirror is missing or invalid")
    if type(active_tokens) is not int or active_tokens <= 0:
        raise CurrentAttentionError("current FIA requires a positive exact active-token count")
    values = starts[:metadata.num_reqs + 1].tolist()
    if values[0] != 0 or any(after < before for before, after in zip(values, values[1:])):
        raise CurrentAttentionError("native CPU query starts are malformed")
    if values[-1] < active_tokens or active_tokens not in values:
        raise CurrentAttentionError("native CPU query starts do not end at the actual token count")
    cumulative: list[int] = []
    for value in values[1:]:
        if value > active_tokens:
            break  # The native synthetic FIA row covers only padding tokens.
        if value > (cumulative[-1] if cumulative else 0):
            cumulative.append(value)
    if not cumulative or cumulative[-1] != active_tokens:
        raise CurrentAttentionError("current FIA cumulative lengths omit active tokens")
    return cumulative


def suppress_current_source_tasks(tasks: torch.Tensor, tokens: int,
                                  kv_heads: int, splits: int) -> None:
    """Make source2 empty on device, preserving every task's error and status.

    Task ABI from attention_tasks.cpp: columns 3/4 are KV begin/end, 7 is
    source kind, 10 is metadata error. Alter only a healthy source2 end. CV's
    own empty-range path writes zero/-inf partials and both AIV status words;
    bad/padded tasks still enter their original validation paths (#67/#126).
    """
    if (type(tokens) is not int or type(kv_heads) is not int or type(splits) is not int
            or tokens <= 0 or kv_heads <= 0 or splits <= 0 or
            tasks.dtype != torch.int64 or tasks.device.type != "npu" or
            tuple(tasks.shape) != (tokens * kv_heads * 3 * splits, 16)
            or not tasks.is_contiguous()):
        raise CurrentAttentionError("source2 task tensor does not match the signed CV ABI")
    _empty_current_source_ranges(tasks, tokens, kv_heads, splits)


def _empty_current_source_ranges(tasks: torch.Tensor, tokens: int,
                                 kv_heads: int, splits: int) -> None:
    """Tensor-only task rewrite, also exercised on CPU for ABI regression."""
    source = tasks.view(tokens, kv_heads, 3, splits, 16)[:, :, 2]
    begin, end, errors = source[..., 3], source[..., 4], source[..., 10]
    end.copy_(torch.where((errors == 0) & (end >= begin), begin, end))


def _causal_mask(device: torch.device) -> torch.Tensor:
    if device.type != "npu":
        raise CurrentAttentionError("native current FIA requires an NPU")
    mask = _MASKS.get(device)
    if mask is None:
        # Native attention_mask.py:53-79, same fixed 2048x2048 int8 mask and
        # sparse mode 3 as attention_cp.py:1017-1032. Stable across eager calls.
        mask = torch.triu(torch.ones((2048, 2048), dtype=torch.int8, device=device), diagonal=1)
        _MASKS[device] = mask
    return mask


def guard_current_slots(ops, status: torch.Tensor, slots: torch.Tensor,
                        active_tokens: int) -> None:
    """Trap on any internal invalid slot before native FIA executes on stream.

    The pre-existing status_guard is a same-stream AscendC trap, with no host
    reduction or D2H readback. Its temporary status buffer is overwritten by
    the later LSE merge. Trailing graph padding is excluded by active_tokens.
    """
    if (type(active_tokens) is not int or active_tokens <= 0 or
            status.device.type != "npu" or slots.device != status.device or
            status.dtype != torch.int32 or slots.dtype not in (torch.int32, torch.int64) or
            status.ndim != 2 or status.shape[0] < active_tokens or
            slots.ndim != 1 or slots.shape[0] < active_tokens or
            not status.is_contiguous()):
        raise CurrentAttentionError("current-slot guard requires fixed NPU status/slot buffers")
    temporary = _populate_slot_error_status(status, slots, active_tokens)
    empty = status[:0].view(-1)
    ops.status_guard(temporary.view(-1), empty, empty, empty)


def _populate_slot_error_status(status: torch.Tensor, slots: torch.Tensor,
                                active_tokens: int) -> torch.Tensor:
    """Generate only the active-token error map (CPU exercised, NPU executed)."""
    temporary = status[:active_tokens]
    temporary.copy_((slots[:active_tokens] < 0).to(torch.int32)[:, None])
    return temporary


def native_current_partial(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                           cumulative: list[int] | tuple[int, ...], *, heads: int, kv_heads: int,
                           scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Call the pinned native causal TND FIA on exact current BF16 Q/K/V."""
    tokens = query.shape[0]
    dim = query.shape[-1]
    if (tokens <= 0 or type(heads) is not int or type(kv_heads) is not int or
            heads <= 0 or kv_heads <= 0 or heads % kv_heads or
            type(scale) not in (float, int) or not math.isfinite(scale) or scale <= 0 or
            not isinstance(cumulative, (list, tuple)) or not cumulative or
            any(type(x) is not int or x <= 0 for x in cumulative) or
            any(b <= a for a, b in zip(cumulative, cumulative[1:])) or cumulative[-1] != tokens):
        raise CurrentAttentionError("native current FIA geometry or cumulative lengths are invalid")
    for tensor, expected in ((query, (tokens, heads, dim)),
                             (key, (tokens, kv_heads, dim)),
                             (value, (tokens, kv_heads, dim))):
        if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected or
                tensor.dtype != torch.bfloat16 or tensor.device != query.device or
                tensor.device.type != "npu" or not tensor.is_contiguous()):
            raise CurrentAttentionError("native current FIA requires contiguous same-NPU BF16 Q/K/V")
    output, lse = torch.ops.npu.npu_fused_infer_attention_score(
        query, key, value, num_heads=heads, num_key_value_heads=kv_heads,
        input_layout="TND", atten_mask=_causal_mask(query.device), scale=float(scale),
        sparse_mode=3, antiquant_mode=0, antiquant_scale=None,
        softmax_lse_flag=True, actual_seq_lengths_kv=cumulative,
        actual_seq_lengths=cumulative)
    if (not isinstance(output, torch.Tensor) or not isinstance(lse, torch.Tensor) or
            tuple(output.shape) != (tokens, heads, dim) or
            tuple(lse.shape) != (tokens, heads, 1) or
            output.device != query.device or lse.device != query.device or
            output.dtype != torch.bfloat16 or
            lse.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
        raise CurrentAttentionError("native current FIA returned invalid output/LSE shape, dtype or device")
    return output, lse


def write_current_partial(partial: torch.Tensor, partial_lse: torch.Tensor,
                          output: torch.Tensor, lse: torch.Tensor,
                          active_tokens: int, source_splits: int,
                          slots: torch.Tensor | None = None) -> None:
    """Fill source2 split0; leave other source2 splits at CV's zero/-inf."""
    if (type(active_tokens) is not int or type(source_splits) is not int or
            active_tokens <= 0 or source_splits <= 0 or
            partial.dtype != torch.float32 or partial_lse.dtype != torch.float32 or
            partial.device.type != "npu" or partial_lse.device != partial.device or
            partial.ndim != 4 or partial_lse.ndim != 3 or
            partial.shape[:3] != (partial_lse.shape[0], partial_lse.shape[1], 3 * source_splits) or
            active_tokens > partial.shape[0] or
            tuple(output.shape) != (active_tokens, partial.shape[1], partial.shape[3]) or
            tuple(lse.shape) != (active_tokens, partial.shape[1], 1) or
            output.device != partial.device or lse.device != partial.device):
        raise CurrentAttentionError("native current partial does not match OSCAR merge workspace")
    current = 2 * source_splits
    dst, dst_lse = partial[:active_tokens, :, current], partial_lse[:active_tokens, :, current]
    dst.copy_(output)
    dst_lse.copy_(lse.squeeze(-1))
    if slots is not None:
        if (slots.device != partial.device or slots.ndim != 1 or
                slots.shape[0] < active_tokens or slots.dtype not in (torch.int32, torch.int64)):
            raise CurrentAttentionError("native current partial received an invalid slot mask")
        invalid = slots[:active_tokens] < 0
        dst.masked_fill_(invalid[:, None, None], 0.0)
        dst_lse.masked_fill_(invalid[:, None], float("-inf"))
