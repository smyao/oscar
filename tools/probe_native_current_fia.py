# Archive #55–69/#126/#129–135: native FIA tiling, causal mask, LSE shape,
# output precision, live progress and device-time attribution must be tested on
# the target NPU. This isolated experiment never participates in service routing.
# D.4 four questions: (1) phase=fia, for the exact current-chunk source only;
# (2) the failed implementation spent 6499.8–6655.1 ms dequantizing history,
# while native FIA took 18.5–18.9 ms at 32K and ~7 ms at 16K;
# (3) this probe passes only current BF16 K/V to native causal FIA, creates no
# historical tensor, and checks output plus LSE against the independent oracle;
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
    # Native vllm-ascend attention/context_parallel/attention_cp.py:1017–1032.
    # Block table is absent because every K/V passed here is in this chunk.
    return torch.ops.npu.npu_fused_infer_attention_score(
        query, key, value, num_heads=heads, num_key_value_heads=kv_heads,
        input_layout="TND", atten_mask=mask, scale=scale, sparse_mode=3,
        antiquant_mode=0, antiquant_scale=None, softmax_lse_flag=True,
        actual_seq_lengths_kv=cumulative, actual_seq_lengths=cumulative)


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
    mask = torch.triu(torch.ones((2048, 2048), dtype=torch.int8, device=device), diagonal=1)
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
    return {"status": "isolated_probe_passed", "scope": "native_current_chunk_fia_only",
            "device": str(device), "device_name": torch.npu.get_device_name(0),
            "target_devices": target["devices"], "head_geometry": {"query": heads, "kv": kv_heads, "dim": dim},
            "causal_mask": "native 2048x2048 int8 upper triangle", "softmax_lse_flag": True,
            "frozen_tolerance": tolerance, "small_cases": cases, "long_case": long_case,
            "device_completion": "isolated_fia_completed", "oscar_cv_replaced": False,
            "production_route_modified": False, "graph_capture": "not_run", "graph_replay": "not_run",
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
