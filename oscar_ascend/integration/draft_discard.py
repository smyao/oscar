"""Archive #27/#34/#36/#37-49/#157/#158/#161: logical cache and discard semantics.

Native model_runner_v1.py:1125-1147 publishes discard_request_mask in current
input_batch order before proposing; llm_base_proposer.py:1535-1666 retains that
order with one token/request on later steps. Only this explicit mask permits
padding. A slot/table error is never evidence that a live request was discarded.
Copies occur in metadata construction, before eager execution or graph replay;
fixed per-draft destinations preserve captured addresses. No device readback,
block-table scan, KV data operation, or per-layer tensor work is introduced.

#161: later cache slots use seq_len - rejected + draft_index - 1, independently
of model RoPE coordinates. The first pass already materializes logical KV.
This fixes a reproducible offset-coordinate defect, not a claim that the
latest asynchronously reported target Trap has been uniquely attributed.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch

from .runtime_api import OscarReadinessError


@dataclass(frozen=True)
class _DraftScope:
    proposer: object
    mask: object
    requests: int
    dummy: bool
    accepted_end: object = None
    invalid_counts: object = None


_scope = ContextVar("oscar_native_draft_discard", default=None)


@contextmanager
def native_draft_scope(proposer, *, dummy=False, common=None, rejected_tokens=None):
    """Keep each native proposal's mask/order within its synchronous call.

    The mask is the native fixed device buffer, not a cached derived mask.
    Metadata construction enqueues the copy/mask on the native current stream,
    before the proposal runs and before the next input preparation updates it.
    Dummy construction never consumes stale real-request mask contents.
    """
    runner = proposer.runner
    requests = 0 if dummy else runner.input_batch.num_reqs
    mask = None if dummy else getattr(getattr(runner, "discard_request_mask", None), "gpu", None)
    if not dummy:
        if (type(requests) is not int or requests < 0 or not torch.is_tensor(mask)
                or mask.dtype != torch.bool or mask.ndim != 1 or mask.numel() < requests):
            raise OscarReadinessError("native MTP discard mask does not match the current input batch")
        if getattr(proposer, "pcp_size", 1) != 1 or getattr(proposer, "dcp_size", 1) != 1:
            raise OscarReadinessError("native MTP discard bridge requires request order without context parallelism")
    accepted_end = invalid_counts = None
    if not dummy and common is not None:
        # #161: cache indices are logical token offsets, not mRoPE coordinates.
        # The first pass writes [seq_len - qlen, seq_len); rejected rows stay
        # materialized, but later queries resume at seq_len - rejected.
        lens = common.seq_lens[:requests]
        starts = common.query_start_loc[:requests + 1]
        if (lens.shape != (requests,) or starts.shape != (requests + 1,)
                or lens.device != mask.device or starts.device != mask.device):
            raise OscarReadinessError("native MTP logical cache metadata does not match request rows")
        rejected = torch.zeros_like(lens) if rejected_tokens is None else rejected_tokens
        if (not torch.is_tensor(rejected) or rejected.shape != (requests,)
                or rejected.device != lens.device or rejected.dtype not in (torch.int32, torch.int64)):
            raise OscarReadinessError("native MTP rejected counts do not match request rows")
        qlen = starts[1:] - starts[:-1]
        invalid_counts = ((rejected < 0) | (rejected >= qlen) | (lens < qlen))
        accepted_end = lens.to(torch.int64) - rejected.to(torch.int64)
    token = _scope.set(_DraftScope(proposer, mask, requests, dummy, accepted_end, invalid_counts))
    try:
        yield
    finally:
        _scope.reset(token)


def later_draft_slot_copy(common, *, draft_index=None, capture=False, require_scope=False):
    """Return a fixed OSCAR slot copy with only native-discarded rows padded.

    Capture builders do not receive a draft_index. Identify their source by
    its native persistent slot group address, which is also used by replay.
    First draft and main-model slots always retain their original address.
    All live and trailing-padding bytes survive, even when they are malformed;
    prepare/status_guard remains responsible for rejecting those bytes.
    """
    scope = _scope.get()
    source = common.slot_mapping
    if require_scope and scope is None and draft_index != 0 and not capture:
        raise OscarReadinessError(
            "OSCAR native MTP discard bridge missing: later-draft metadata was built "
            "outside the hooked proposer scope")
    if scope is None or draft_index == 0:
        return source
    groups = scope.proposer.slot_mapping_group
    index = next((i for i, group in enumerate(groups)
                  if source.data_ptr() == group.data_ptr()), None)
    if index is None or (draft_index is not None and index != draft_index):
        raise OscarReadinessError("later MTP slots are not the native draft group's fixed buffer")
    if index == 0:
        return source
    native = groups[index]
    if (source.ndim != 1 or source.stride() != (1,)
            or source.dtype not in (torch.int32, torch.int64)
            or source.dtype != native.dtype or source.device != native.device
            or source.numel() > native.numel()):
        raise OscarReadinessError("later MTP slot view does not match its native allocation")
    if not scope.dummy and not capture:
        if (common.max_query_len != 1 or common.num_actual_tokens != scope.requests
                or common.num_reqs < scope.requests or source.numel() < scope.requests
                or scope.mask.device != source.device):
            raise OscarReadinessError("later MTP discard rows do not match native one-token request order")
    copies = getattr(scope.proposer, "_oscar_discard_slot_copies", None)
    if copies is None:
        copies = {}
        scope.proposer._oscar_discard_slot_copies = copies
    entry = copies.get(index)
    if entry is None:
        entry = (native, torch.empty_like(native))
        copies[index] = entry
    original, destination = entry
    if (original.data_ptr() != native.data_ptr() or original.shape != native.shape
            or original.dtype != native.dtype or original.device != native.device):
        raise OscarReadinessError("native MTP slot allocation changed after OSCAR binding")
    result = destination[:source.numel()]
    result.copy_(source)
    if not scope.dummy and not capture:
        if scope.accepted_end is not None:
            # Native later slots use clamped_positions (RoPE) to index the
            # virtual128 table. FULL KV is indexed by logical sequence order.
            # Keep native padding/discard decisions and all model positions.
            size = scope.requests
            table = common.block_table_tensor[:size]
            limit = scope.proposer.max_model_len
            if table.ndim != 2 or table.shape[0] != size or table.shape[1] == 0:
                raise OscarReadinessError("native MTP logical cache table is empty or malformed")
            position = scope.accepted_end + index - 1
            column = torch.div(position, 128, rounding_mode="floor")
            bad = (scope.invalid_counts | (position < 0) | (position >= limit)
                   | (column < 0) | (column >= table.shape[1]))
            # Clamp only the speculative gather address, never accept a bad
            # logical index: bad live rows get length=0 below and still trap.
            block = table.gather(1, column.clamp(0, table.shape[1] - 1)[:, None]).reshape(-1)
            logical_slot = block.to(torch.int64) * 128 + position.remainder(128)
            bad = bad | (block < 0) | (logical_slot > torch.iinfo(source.dtype).max)
            live = (source[:size] >= 0) & ~scope.mask[:size]
            safe_slot = torch.where(bad, torch.zeros_like(logical_slot), logical_slot)
            result[:size].copy_(torch.where(live, safe_slot.to(source.dtype), source[:size]))
            errors = getattr(scope.proposer, "_oscar_logical_slot_errors", None)
            if errors is None:
                errors = {}
                scope.proposer._oscar_logical_slot_errors = errors
            errors[index] = live & bad
        result[:scope.requests].masked_fill_(scope.mask[:scope.requests], -1)
        if not getattr(scope.proposer, "_oscar_discard_bridge_logged", False):
            # Host-only execution evidence; the native mask stays on device.
            # Enqueue is not a claim that device execution has completed.
            print("[oscar] MTP_DISCARD_BRIDGE "
                  f"proposer={type(scope.proposer).__name__} method=mtp "
                  f"requests={scope.requests} mask_capacity={scope.mask.numel()} "
                  f"draft_index={index} source=native.discard_request_mask binding=queued "
                  f"cache_position={'logical_rejection_boundary' if scope.accepted_end is not None else 'native_slot'}", flush=True)
            scope.proposer._oscar_discard_bridge_logged = True
    return result


def later_draft_lengths_copy(common, slots, *, draft_index):
    """Keep valid accepted positions visible when native MTP wraps lens to 1.

    Native llm_base_proposer.py:1608 wraps its optimistic (rejected rows kept)
    length when it exceeds max_model_len, while padding the slot only when
    the accepted position itself exceeds the limit. Those conditions differ.
    The original first-pass device length plus draft_index identifies this
    explicit wrap. Restore only its scanning upper bound, never infer discard
    or accepted position from a bad slot or an mRoPE coordinate. Each proposal
    refreshes the fixed destination before executing its later draft model.
    """
    scope = _scope.get()
    source = common.seq_lens
    if scope is None or scope.dummy or draft_index == 0:
        return source
    proposer = scope.proposer
    limit = getattr(proposer, "max_model_len", None)
    groups = getattr(proposer, "seq_lens_group", None)
    if (type(limit) is not int or limit <= 0 or groups is None
            or draft_index >= len(groups)):
        raise OscarReadinessError("native MTP model limit or fixed length groups are missing")
    base, native = groups[0], groups[draft_index]
    if (source.data_ptr() != native.data_ptr() or source.ndim != 1 or source.dtype != torch.int32
            or source.device != slots.device or source.numel() < scope.requests
            or base.dtype != source.dtype or base.device != source.device
            or base.numel() < scope.requests):
        raise OscarReadinessError("native MTP length groups do not match this proposal")
    copies = getattr(proposer, "_oscar_later_length_copies", None)
    if copies is None:
        copies = {}
        proposer._oscar_later_length_copies = copies
    entry = copies.get(draft_index)
    if entry is None:
        entry = (native, torch.empty_like(native))
        copies[draft_index] = entry
    original, destination = entry
    if (original.data_ptr() != native.data_ptr() or original.shape != native.shape
            or original.dtype != native.dtype or original.device != native.device):
        raise OscarReadinessError("native MTP length allocation changed after OSCAR binding")
    result = destination[:source.numel()]
    result.copy_(source)
    if scope.accepted_end is not None:
        errors = getattr(proposer, "_oscar_logical_slot_errors", {}).get(draft_index)
        if errors is None:
            raise OscarReadinessError("logical MTP slots must be prepared before their lengths")
        # A corrupt live row remains live with impossible metadata. Frozen
        # prepare/guard reject it; it is never disguised as discarded padding.
        result[:scope.requests].masked_fill_(errors, 0)
    # Native NPUModelRunner._build_attention_metadata publishes the maximum
    # CPU optimistic sequence length (or the full model limit for capture).
    # It bounds the corrected first-pass device lengths. Its scalar survives
    # native shallow-copy later metadata unchanged. Preserve the destination
    # address but avoid extra device predicates when this bound proves no wrap.
    upper = getattr(common, "max_seq_len", 0)
    if type(upper) is int and 0 < upper and upper + draft_index <= limit:
        return result
    size = scope.requests
    before_wrap = base[:size] + draft_index
    wrapped_live = ((source[:size] == 1) & (before_wrap > limit)
                    & (slots[:size] >= 0) & ~scope.mask[:size])
    result[:size].copy_(torch.where(wrapped_live, before_wrap.clamp(max=limit), result[:size]))
    if scope.accepted_end is not None:
        result[:size].masked_fill_(errors, 0)
    return result
