# Archive #55–69/#126/#129–140: native FIA tiling, causal mask, LSE shape,
# output precision, live progress and device-time attribution must be tested on
# the target NPU. This probe gates the current-chunk production primitive;
# only the independent offline oracle below materializes diagnostic history.
# D.4 four questions: (1) phase=fia, for the exact current-chunk source only;
# (2) the failed implementation spent 6499.8–6655.1 ms dequantizing history;
# its old FIA phase took 18.5–18.9 ms at 32K and ~7 ms at 16K, not a
# current-only timing;
# (3) native causal FIA receives only current BF16 K/V; a separate offline
# oracle checks its output/LSE and the merge with packed-history/window parts;
# (4) those D.4 times are reference order only—this script reports measured
# 16K device-event time and never claims a performance or service pass.
"""Fail-closed target-NPU experiment for native current-chunk FIA and LSE."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

from .phase import atomic_json
from .plog import attach_plog


class NativeCurrentFIAProbeError(RuntimeError):
    pass


def _cumulative_lengths(lengths: tuple[int, ...] | list[int]) -> list[int]:
    if not lengths or any(type(length) is not int or length <= 0 for length in lengths):
        raise ValueError("each native TND request must have a positive integer current-chunk length")
    cumulative = []
    total = 0
    for length in lengths:
        total += length
        cumulative.append(total)
    return cumulative


def _check_shapes(output_shape, lse_shape, tokens: int, heads: int, dim: int) -> None:
    # Archive #58/#60–62: a placeholder scalar/empty LSE is not a result.
    if tuple(output_shape) != (tokens, heads, dim):
        raise NativeCurrentFIAProbeError(
            f"native FIA output shape {tuple(output_shape)} != {(tokens, heads, dim)}")
    if tuple(lse_shape) != (tokens, heads, 1):
        raise NativeCurrentFIAProbeError(
            f"native FIA LSE shape {tuple(lse_shape)} != {(tokens, heads, 1)}")


def _target_geometry(target: dict) -> tuple[int, int, int]:
    model_path = Path(target["model"]) / "config.json"
    model = json.loads(model_path.read_text())
    text = model.get("text_config", model)
    tp = target["tensor_parallel_size"]
    heads = text.get("num_attention_heads")
    kv_heads = text.get("num_key_value_heads")
    dim = text.get("head_dim")
    hidden = text.get("hidden_size")
    if (type(tp) is not int or tp <= 0 or type(heads) is not int or heads <= 0
            or type(kv_heads) is not int or kv_heads <= 0):
        raise NativeCurrentFIAProbeError("target model must declare integral TP, attention and KV head counts")
    if dim is None:
        if type(hidden) is not int or hidden % heads:
            raise NativeCurrentFIAProbeError("target model must declare a valid head_dim or hidden_size")
        dim = hidden // heads
    if (type(dim) is not int or dim not in (64, 128, 256) or heads % tp or kv_heads % tp
            or heads < kv_heads or (heads // tp) % (kv_heads // tp)):
        raise NativeCurrentFIAProbeError("target head geometry is incompatible with OSCAR TP/CV")
    return heads // tp, kv_heads // tp, dim


def _select_target_npu(target: dict) -> None:
    devices = target.get("devices")
    if (not isinstance(devices, list) or not devices
            or any(type(device) is not int or device < 0 for device in devices)):
        raise NativeCurrentFIAProbeError("configs/target.json must select physical NPU devices")
    expected = ",".join(map(str, devices))
    selected = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if selected is not None and selected != expected:
        raise NativeCurrentFIAProbeError(
            f"ASCEND_RT_VISIBLE_DEVICES={selected!r} differs from target {expected!r}")
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = expected


def _call_fia(torch, query, key, value, cumulative, mask, scale, heads, kv_heads):
    # Exercise the actual production helper, including its native mask and
    # output/LSE shape checks; the oracle stays independent below.
    from oscar_ascend.integration.current_attention import native_current_partial
    # Production stores this once per native step as an immutable tuple.
    # Exercise that exact Python binding type, not only a list microprobe.
    return native_current_partial(query, key, value, tuple(cumulative),
                                  heads=heads, kv_heads=kv_heads, scale=scale)


def _validate_result(torch, output, lse, tokens, heads, dim, device) -> None:
    if not isinstance(output, torch.Tensor) or not isinstance(lse, torch.Tensor):
        raise NativeCurrentFIAProbeError("native FIA did not return output and LSE tensors")
    _check_shapes(output.shape, lse.shape, tokens, heads, dim)
    if output.device != device or lse.device != device:
        raise NativeCurrentFIAProbeError("native FIA result is not on the selected NPU")
    if output.dtype != torch.bfloat16 or lse.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NativeCurrentFIAProbeError(
            f"unexpected FIA output/LSE dtypes: {output.dtype}, {lse.dtype}")
    if not bool(torch.isfinite(output).all()) or not bool(torch.isfinite(lse).all()):
        raise NativeCurrentFIAProbeError("native FIA returned non-finite output or LSE")


def _assert_oracle(torch, output, lse, expected, tolerance: dict) -> dict:
    atol, rtol = tolerance["atol"], tolerance["rtol"]
    actual_output = output.float().cpu()
    actual_lse = lse.float().cpu().squeeze(-1)
    reference_output = expected.output.float().cpu()
    reference_lse = expected.lse.float().cpu()
    torch.testing.assert_close(actual_output, reference_output, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual_lse, reference_lse, atol=atol, rtol=rtol)
    return {"max_output_abs": float((actual_output - reference_output).abs().max()),
            "max_lse_abs": float((actual_lse - reference_lse).abs().max())}


def _small_case(torch, reference_attention, device, mask, lengths, heads, kv_heads, dim, scale,
                tolerance, seed):
    total = sum(lengths)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    query_cpu = torch.randn((total, heads, dim), generator=generator).to(torch.bfloat16)
    key_cpu = torch.randn((total, kv_heads, dim), generator=generator).to(torch.bfloat16)
    value_cpu = torch.randn((total, kv_heads, dim), generator=generator).to(torch.bfloat16)
    cumulative = _cumulative_lengths(lengths)
    output, lse = _call_fia(torch, query_cpu.to(device), key_cpu.to(device),
                            value_cpu.to(device), cumulative, mask, scale, heads, kv_heads)
    torch.npu.synchronize()
    _validate_result(torch, output, lse, total, heads, dim, device)
    pieces = []
    start = 0
    for stop in cumulative:
        pieces.append(reference_attention(query_cpu[start:stop], key_cpu[start:stop],
                                          value_cpu[start:stop], scale=scale, causal=True))
        start = stop
    from oscar_ascend.ops.reference import AttentionResult
    expected = AttentionResult(torch.cat([part.output for part in pieces]),
                               torch.cat([part.lse for part in pieces]))
    errors = _assert_oracle(torch, output, lse, expected, tolerance)
    return {"lengths": list(lengths), "tokens": total, "output_shape": list(output.shape),
            "lse_shape": list(lse.shape), "oracle": "passed", **errors}


def _task_rewrite_case(torch, device, heads, kv_heads):
    """Check the production strided source2 write and same-stream slot guard."""
    from oscar_ascend.integration.current_attention import (
        guard_current_slots, suppress_current_source_tasks)
    from oscar_ascend.ops.loader import require_capabilities
    require_capabilities({"status_guard"})
    tokens, splits = 7, 3
    cpu = torch.zeros((tokens, kv_heads, 3, splits, 16), dtype=torch.int64)
    cpu[..., 1] = 1
    cpu[..., 3] = 11
    cpu[..., 4] = 27
    cpu[..., 7] = torch.arange(3).view(1, 1, 3, 1)
    cpu[0, :, 2, 0, 10] = 5
    cpu[1, :, 2, 0, 4] = 10
    cpu[2, :, 2, 0, 1] = -1
    expected = cpu.clone()
    for token in range(tokens):
        for head in range(kv_heads):
            for split in range(splits):
                row = expected[token, head, 2, split]
                if row[10] == 0 and row[4] >= row[3]:
                    row[4] = row[3]
    tasks = cpu.view(-1, 16).to(device)
    suppress_current_source_tasks(tasks, tokens, kv_heads, splits)
    slots = torch.arange(tokens + 2, dtype=torch.int64, device=device)
    slots[tokens:] = -1
    status = torch.full((tokens + 2, heads), -99, dtype=torch.int32, device=device)
    guard_current_slots(torch.ops.oscar_ascend_ops, status, slots, tokens)
    torch.npu.synchronize()
    torch.testing.assert_close(tasks.cpu().view_as(cpu), expected, atol=0, rtol=0)
    torch.testing.assert_close(status[:tokens].cpu(), torch.zeros(tokens, heads, dtype=torch.int32),
                               atol=0, rtol=0)
    torch.testing.assert_close(status[tokens:].cpu(), torch.full((2, heads), -99, dtype=torch.int32),
                               atol=0, rtol=0)
    return {"source2_range_rewrite": "passed", "metadata_error_preserved": True,
            "slot_guard": "passed", "padding_excluded": True}


def _mixed_merge_case(torch, reference_attention, device, mask, heads, kv_heads, dim,
                      scale, tolerance):
    """Run the actual three-source device pipeline against an offline oracle.

    Each request owns disjoint, permuted physical pages. Only the independent
    expected-output calculation materializes history on CPU (#71/D.4).
    """
    from oscar_ascend.ops.reference import encode_kv, decode_kv
    from oscar_ascend.ops.loader import require_capabilities
    from oscar_ascend.integration.current_attention import (
        guard_current_slots, suppress_current_source_tasks,
        write_current_partial)

    require_capabilities({"prepare_attention_tasks_out", "attention_cv_out",
                          "merge_lse_out", "status_guard"})
    ops = torch.ops.oscar_ascend_ops
    lengths, contexts = (1, 17, 129), (17, 511, 2049)
    page_assignments = ((2,), (7, 0), (4, 1, 6, 3, 5))
    block_tokens, blocks, sink, recent, speculative = 512, 8, 64, 256, 3
    splits, cores, prefix = 3, 2, 64
    if sorted(page for request in page_assignments for page in request) != list(range(blocks)):
        raise NativeCurrentFIAProbeError("mixed oracle physical pages are not disjoint")
    generator = torch.Generator(device="cpu").manual_seed(46776)
    tokens = sum(lengths)
    q = torch.randn((tokens, heads, dim), generator=generator).to(torch.bfloat16)
    ck = torch.randn((tokens, kv_heads, dim), generator=generator).to(torch.bfloat16)
    cv = torch.randn((tokens, kv_heads, dim), generator=generator).to(torch.bfloat16)
    rotation_k = torch.ones((1, 1), dtype=torch.float32)
    while rotation_k.shape[0] < dim:
        rotation_k = torch.cat((torch.cat((rotation_k, rotation_k), 1),
                                torch.cat((rotation_k, -rotation_k), 1)), 0) / math.sqrt(2)
    rotation_v = rotation_k.flip(1).contiguous()
    slot_bytes = dim // 2 + 8
    stride = block_tokens * slot_bytes * kv_heads
    raw = torch.full((prefix + blocks * stride,), 0xA5, dtype=torch.uint8)
    window_rows = sink + recent + speculative
    window_k = torch.full((blocks, window_rows, kv_heads, dim), float("nan"), dtype=torch.bfloat16)
    window_v = window_k.clone()
    window_tags = torch.full((blocks, window_rows), -1, dtype=torch.int64)
    columns = math.ceil(max(context + length for context, length in zip(contexts, lengths)) / 128)
    table = torch.full((len(lengths) + 1, columns), -1, dtype=torch.int32)
    slots_cpu = torch.full((tokens + 3,), -1, dtype=torch.int64)
    starts_cpu = [0]
    lens_cpu = []
    expected_outputs, expected_lses = [], []
    begin = 0
    for request, (length, context, pages) in enumerate(zip(lengths, contexts, page_assignments)):
        if len(pages) != math.ceil((context + length) / block_tokens):
            raise NativeCurrentFIAProbeError("mixed oracle request page budget is wrong")
        for logical_page, physical in enumerate(pages):
            for quarter in range(block_tokens // 128):
                column = logical_page * (block_tokens // 128) + quarter
                if column < columns:
                    table[request, column] = physical * (block_tokens // 128) + quarter
        old_k = torch.randn((context, kv_heads, dim), generator=generator).to(torch.bfloat16)
        old_v = torch.randn((context, kv_heads, dim), generator=generator).to(torch.bfloat16)
        packed = encode_kv(old_k.float() @ rotation_k, old_v.float() @ rotation_v)
        decoded_k, decoded_v = decode_kv(packed, dim)
        restored_k, restored_v = decoded_k @ rotation_k.T, decoded_v @ rotation_v.T
        for position in range(context):
            physical, inpage = pages[position // block_tokens], position % block_tokens
            address = prefix + physical * stride + inpage * slot_bytes * kv_heads
            raw[address:address + slot_bytes * kv_heads] = packed[position].reshape(-1)
            row = position if position < sink else sink + inpage % (recent + speculative)
            window_k[physical, row] = old_k[position]
            window_v[physical, row] = old_v[position]
            window_tags[physical, row] = inpage
        for local in range(length):
            position = context + local
            physical, inpage = pages[position // block_tokens], position % block_tokens
            slots_cpu[begin + local] = physical * block_tokens + inpage
        old_positions = torch.arange(context)
        for local in range(length):
            index = begin + local
            cut = min(context, max(sink, context + local + 1 - recent))
            compressed = (old_positions >= sink) & (old_positions < cut)
            exact_k = torch.where(compressed[:, None, None], restored_k, old_k.float())
            exact_v = torch.where(compressed[:, None, None], restored_v, old_v.float())
            dense_k = torch.cat((exact_k, ck[begin:index + 1].float()))
            dense_v = torch.cat((exact_v, cv[begin:index + 1].float()))
            dense = reference_attention(q[index:index+1], dense_k, dense_v,
                                        scale=scale, causal=False)
            expected_outputs.append(dense.output[0])
            expected_lses.append(dense.lse[0])
        begin += length
        starts_cpu.append(begin)
        lens_cpu.append(context + length)
    if begin != tokens or bool(torch.any(slots_cpu[:tokens] < 0)):
        raise NativeCurrentFIAProbeError("mixed oracle left an active current slot unassigned")
    padded = tokens + 3
    starts_cpu.append(padded)  # Native FIA's synthetic terminal padding row.
    lens_cpu.append(1)
    q_padded_cpu = torch.cat((q, torch.zeros((3, heads, dim), dtype=torch.bfloat16)))
    q_padded = q_padded_cpu.to(device)
    k_padded = torch.cat((ck, torch.zeros((3, kv_heads, dim), dtype=torch.bfloat16))).to(device)
    v_padded = torch.cat((cv, torch.zeros((3, kv_heads, dim), dtype=torch.bfloat16))).to(device)
    q_rot = (q_padded_cpu.float() @ rotation_k).to(device)
    starts = torch.tensor(starts_cpu, dtype=torch.int32, device=device)
    lens = torch.tensor(lens_cpu, dtype=torch.int32, device=device)
    slots = slots_cpu.to(device)
    tasks = torch.empty((padded * kv_heads * 3 * splits, 16), dtype=torch.int64, device=device)
    positions = torch.empty((padded,), dtype=torch.int64, device=device)
    partial = torch.full((padded, heads, 3 * splits, dim), float("nan"), dtype=torch.float32, device=device)
    lse = torch.full((padded, heads, 3 * splits), float("nan"), dtype=torch.float32, device=device)
    cv_status = torch.full((tasks.shape[0], 2), -99, dtype=torch.int32, device=device)
    workspace = torch.empty(cores * (384 * dim + 8192) * 4, dtype=torch.uint8, device=device)
    ops.prepare_attention_tasks_out(starts, lens, slots, tasks, positions,
                                    heads, kv_heads, sink, recent, splits)
    suppress_current_source_tasks(tasks, padded, kv_heads, splits)
    ops.attention_cv_out(q_padded, q_rot, k_padded, v_padded, rotation_v.to(device),
        raw.to(device), table.to(device), window_k.to(device), window_v.to(device),
        window_tags.to(device), tasks, partial, lse, cv_status, workspace,
        block_tokens, blocks, prefix, stride, sink, recent, speculative,
        splits, scale, cores)
    guard_status = torch.full((padded, heads), -99, dtype=torch.int32, device=device)
    guard_current_slots(ops, guard_status, slots, tokens)
    native_output, native_lse = _call_fia(torch, q_padded[:tokens], k_padded[:tokens],
        v_padded[:tokens], _cumulative_lengths(lengths), mask, scale, heads, kv_heads)
    _validate_result(torch, native_output, native_lse, tokens, heads, dim, device)
    write_current_partial(partial, lse, native_output, native_lse, tokens, splits)
    output = torch.empty((padded * heads, dim), dtype=torch.float32, device=device)
    output_lse = torch.empty(padded * heads, dtype=torch.float32, device=device)
    status = torch.full((padded * heads,), -1, dtype=torch.int32, device=device)
    ops.merge_lse_out(partial.view(padded * heads, 3 * splits, dim),
        lse.view(padded * heads, 3 * splits), output, output_lse, status)
    torch.npu.synchronize()
    torch.testing.assert_close(cv_status.cpu(), torch.zeros_like(cv_status.cpu()), atol=0, rtol=0)
    torch.testing.assert_close(guard_status[:tokens].cpu(),
                               torch.zeros(tokens, heads, dtype=torch.int32), atol=0, rtol=0)
    torch.testing.assert_close(status.cpu(), torch.zeros(padded * heads, dtype=torch.int32),
                               atol=0, rtol=0)
    expected_positions = torch.cat([torch.arange(context, context + length)
                                    for context, length in zip(contexts, lengths)])
    torch.testing.assert_close(positions[:tokens].cpu(), expected_positions, atol=0, rtol=0)
    torch.testing.assert_close(positions[tokens:].cpu(), torch.full((3,), -1, dtype=torch.int64),
                               atol=0, rtol=0)
    from oscar_ascend.ops.reference import AttentionResult
    expected = AttentionResult(torch.stack(expected_outputs), torch.stack(expected_lses))
    output = output.view(padded, heads, dim)
    output_lse = output_lse.view(padded, heads, 1)
    errors = _assert_oracle(torch, output[:tokens], output_lse[:tokens], expected, tolerance)
    torch.testing.assert_close(output[tokens:].cpu(), torch.zeros(3, heads, dim), atol=0, rtol=0)
    if not bool(torch.isneginf(output_lse[tokens:]).all()):
        raise NativeCurrentFIAProbeError("current-source padding did not preserve empty LSE")
    return {"lengths": list(lengths), "contexts": list(contexts), "tokens": tokens,
            "rotation": "distinct_orthonormal_hadamard_K_V", "history": "production_INT2_CV",
            "window": "exact_BF16_moving_cut", "prepare": "production_NPU_prepare_attention_tasks_out",
            "source2": "production_NPU_suppress_current_source_tasks",
            "history_window": "production_NPU_attention_cv_out",
            "current": "production_native_current_partial",
            "merge": "production_NPU_merge_lse_out",
            "physical_pages": [list(pages) for pages in page_assignments],
            "source_splits": splits,
            "padding_rows": 3, "oracle": "passed", **errors}


def _long_case(torch, reference_attention, device, mask, heads, kv_heads, dim, scale,
               tolerance, warmup, repeats):
    tokens = 16384
    generator = torch.Generator(device="cpu").manual_seed(46775)
    query_cpu = torch.randn((tokens, heads, dim), generator=generator).to(torch.bfloat16)
    key_cpu = torch.randn((tokens, kv_heads, dim), generator=generator).to(torch.bfloat16)
    value_cpu = torch.randn((tokens, kv_heads, dim), generator=generator).to(torch.bfloat16)
    query, key, value = query_cpu.to(device), key_cpu.to(device), value_cpu.to(device)
    cumulative = [tokens]
    output, lse = _call_fia(torch, query, key, value, cumulative, mask, scale, heads, kv_heads)
    torch.npu.synchronize()
    _validate_result(torch, output, lse, tokens, heads, dim, device)
    # Full 16K dense scores would be prohibitively large. These rows cross the
    # native 2048 mask-tile boundary and include the final causal position.
    sample = [0, 1, 2047, 2048, 8191, 16383]
    expected = reference_attention(query_cpu[sample], key_cpu, value_cpu, scale=scale,
                                   causal=True, query_positions=torch.tensor(sample))
    errors = _assert_oracle(torch, output[sample], lse[sample], expected, tolerance)
    for _ in range(warmup):
        output, lse = _call_fia(torch, query, key, value, cumulative, mask, scale, heads, kv_heads)
        torch.npu.synchronize()
        _validate_result(torch, output, lse, tokens, heads, dim, device)
    milliseconds = []
    stream = torch.npu.current_stream()
    for _ in range(repeats):
        begin = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        begin.record(stream)
        output, lse = _call_fia(torch, query, key, value, cumulative, mask, scale, heads, kv_heads)
        end.record(stream)
        torch.npu.synchronize()
        duration = float(begin.elapsed_time(end))
        if not math.isfinite(duration) or duration <= 0:
            raise NativeCurrentFIAProbeError(f"invalid native FIA event duration {duration!r} ms")
        milliseconds.append(duration)
    _validate_result(torch, output, lse, tokens, heads, dim, device)
    return {"tokens": tokens, "output_shape": list(output.shape), "lse_shape": list(lse.shape),
            "sampled_query_rows": sample, "sampled_oracle": "passed", **errors,
            "device_event_ms": milliseconds, "device_event_median_ms": statistics.median(milliseconds),
            "device_event_scope": "isolated native FIA current chunk only; no model, CV, or service"}


def probe(target_path: Path, acceptance_path: Path) -> dict:
    target = json.loads(target_path.read_text())
    acceptance = json.loads(acceptance_path.read_text())
    if acceptance.get("frozen_before_measurement") is not True:
        raise NativeCurrentFIAProbeError("acceptance thresholds must be frozen before measuring")
    tolerance = acceptance["fused_attention"]
    if any(type(tolerance.get(name)) not in (int, float) or tolerance[name] <= 0 for name in ("atol", "rtol")):
        raise NativeCurrentFIAProbeError("frozen fused_attention tolerances are invalid")
    _select_target_npu(target)  # Must precede torch/torch_npu import.
    import torch
    try:
        import torch_npu  # noqa: F401  # Register the native NPU operator namespace.
    except (ImportError, OSError) as error:
        raise NativeCurrentFIAProbeError("torch_npu is required; no CPU operator substitute is allowed") from error
    if not torch.npu.is_available():
        raise NativeCurrentFIAProbeError("a real target NPU is required; no CPU operator substitute is allowed")
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    heads, kv_heads, dim = _target_geometry(target)
    if not hasattr(torch.ops.npu, "npu_fused_infer_attention_score"):
        raise NativeCurrentFIAProbeError("native npu_fused_infer_attention_score is unavailable")
    from oscar_ascend.ops.reference import attention as reference_attention
    # Native attention_mask.py:53–79: int8 upper triangle, 1 means masked.
    mask = None  # The production helper owns the native mask for this device.
    scale = dim ** -0.5
    cases = [_small_case(torch, reference_attention, device, mask, lengths, heads, kv_heads,
                         dim, scale, tolerance, seed=46774 + index)
             for index, lengths in enumerate(((1,), (17,), (17, 33), (127, 130)))]
    performance = acceptance["performance"]
    warmup, repeats = performance["warmup"], performance["repeats"]
    if type(warmup) is not int or warmup < 0 or type(repeats) is not int or repeats < 1:
        raise NativeCurrentFIAProbeError("frozen timing repetition counts are invalid")
    long_case = _long_case(torch, reference_attention, device, mask, heads, kv_heads,
                           dim, scale, tolerance, warmup, repeats)
    merged_case = _mixed_merge_case(torch, reference_attention, device, mask, heads, kv_heads,
                                   dim, scale, tolerance)
    task_case = _task_rewrite_case(torch, device, heads, kv_heads)
    return {"status": "current_partial_probe_passed", "scope": "native_current_and_three_source_merge",
            "device": str(device), "device_name": torch.npu.get_device_name(0),
            "target_devices": target["devices"], "head_geometry": {"query": heads, "kv": kv_heads, "dim": dim},
            "causal_mask": "native 2048x2048 int8 upper triangle", "softmax_lse_flag": True,
            "frozen_tolerance": tolerance, "small_cases": cases, "long_case": long_case,
            "mixed_history_window_current_merge": merged_case,
            "production_task_contract": task_case,
            "device_completion": "integrated_current_cv_merge_completed",
            "current_source_replaced": True, "history_window_cv_preserved": True,
            "production_route_modified": True, "graph_capture": "not_run", "graph_replay": "not_run",
            "mtp": "not_run", "full_service_acceptance": "not_run", "performance_acceptance": "not_run"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=Path("configs/target.json"))
    parser.add_argument("--acceptance", type=Path, default=Path("configs/acceptance.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    started = time.time()
    try:
        report = probe(args.target, args.acceptance)
        code = 0
    except Exception as error:
        traceback.print_exc()
        report = {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                  "device_completion": "not_established", "oscar_cv_replaced": False,
                  "production_route_modified": False, "graph_capture": "not_run", "graph_replay": "not_run",
                  "full_service_acceptance": "not_run", "performance_acceptance": "not_run"}
        attach_plog(report, started_at=started, owned_pids={os.getpid()})
        code = 1
    report["elapsed_seconds"] = time.time() - started
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
