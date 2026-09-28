# Archive #130/#139/#143/#148-151 and startup D.4: measure the actual
# current-native mixed attention path, never a full-history BF16 restoration
# or a CV source2 substitute. This is an operator diagnostic, not AISBench.
"""One-NPU mixed FULL-attention diagnostic with a matched decode cohort."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import traceback

from . import probe_history_reuse as reuse
from .phase import atomic_json

ROOT = Path(__file__).resolve().parents[1]
DECODE_REQUESTS = 31
DECODE_Q = 4
PREFILL_Q = 16260
MIXED_TOKENS = DECODE_REQUESTS * DECODE_Q + PREFILL_Q  # 16384
DECODE_PADDED_TOKENS = 128
CONTEXT_CYCLE = (20000, 23000, 27000, 30000)
REPEATS, WARMUP = 5, 2


class MixedAttentionProbeError(RuntimeError):
    pass


def shape_plan() -> tuple[reuse.Shape, reuse.Shape]:
    contexts = tuple(CONTEXT_CYCLE[i % len(CONTEXT_CYCLE)]
                     for i in range(DECODE_REQUESTS))
    qlens = (DECODE_Q,) * DECODE_REQUESTS + (PREFILL_Q,)
    return tuple(reuse.Shape(
        "mixed_cold_first_chunk" if context == 0 else "mixed_history_30k",
        qlens, contexts + (context,), 256, 1, 1,
        expect_clusters=context > 0)
        for context in (0, 13740))


def _logical_request_hash(request: int, q, old_k, old_v, ck, cv) -> bytes:
    # _fingerprint copies contiguous CPU tensor bytes directly; no NumPy or
    # per-byte Python loop. Physical page assignment is intentionally absent.
    digest = reuse._fingerprint({"q": q.contiguous(), "old_k": old_k.contiguous(),
                                 "old_v": old_v.contiguous(),
                                 "ck": ck.contiguous(), "cv": cv.contiguous()})
    return f"{request}:{digest}\n".encode()


def make_mixed_fixture(torch, spec: reuse.Shape) -> dict:
    if (len(spec.qlens) != DECODE_REQUESTS + 1 or
            spec.qlens != (DECODE_Q,) * DECODE_REQUESTS + (PREFILL_Q,) or
            spec.contexts[-1] not in (0, 13740) or
            spec.splits != 1 or sum(spec.qlens) != MIXED_TOKENS):
        raise MixedAttentionProbeError("mixed fixture does not reproduce N16384/S1")
    logical = hashlib.sha256()
    def observe(request, _start, q, old_k, old_v, ck, cv):
        if request < DECODE_REQUESTS:
            logical.update(_logical_request_hash(request, q, old_k, old_v, ck, cv))
    fixture = reuse.make_fixture(torch, spec, on_request=observe)
    fixture["logical_decode_input_sha256"] = logical.hexdigest()
    return fixture


def derive_decode_fixture(torch, mixed: dict, *, requests: int = DECODE_REQUESTS,
                          padded_tokens: int = DECODE_PADDED_TOKENS,
                          splits: int = 3) -> dict:
    """Share logical Q/K/V and cache bytes; mask only the four graph tail slots."""
    source: reuse.Shape = mixed["spec"]
    if (not 0 < requests < len(source.qlens) or
            any(length != DECODE_Q for length in source.qlens[:requests]) or
            padded_tokens < requests * DECODE_Q or
            padded_tokens > mixed["tokens"] or splits <= 0):
        raise MixedAttentionProbeError("invalid derived pure-decode geometry")
    actual = requests * DECODE_Q
    spec = reuse.Shape(
        f"decode_from_{source.name}_n{padded_tokens}_s{splits}", source.qlens[:requests],
        source.contexts[:requests], source.dim, source.kv_heads, splits,
        expect_clusters=False, padded_tokens=padded_tokens,
        recent_tokens=source.recent_tokens)
    cpu = dict(mixed["cpu"])
    for key in ("q", "qr", "ck", "cv"):
        cpu[key] = cpu[key][:padded_tokens].contiguous().clone()
    cpu["slots"] = torch.cat((mixed["cpu"]["slots"][:actual].clone(),
                              torch.full((padded_tokens - actual,), -1,
                                         dtype=torch.int64)))
    cpu["starts"] = mixed["cpu"]["starts"][:requests + 1].contiguous().clone()
    cpu["lens"] = mixed["cpu"]["lens"][:requests].contiguous().clone()
    cpu["table"] = mixed["cpu"]["table"][:requests].contiguous().clone()
    if (cpu["starts"][-1].item() != actual or
            not bool((cpu["slots"][:actual] >= 0).all()) or
            not bool((cpu["slots"][actual:] == -1).all())):
        raise MixedAttentionProbeError("pure decode did not preserve 31 valid requests plus padding")
    expected = {index: value for index, value in mixed["expected"].items()
                if index < actual}
    if len(expected) != actual:
        raise MixedAttentionProbeError("pure decode is missing a logical oracle row")
    return {"spec": spec, "cpu": cpu, "expected": expected,
            "input_sha256": reuse._fingerprint(cpu),
            "logical_decode_input_sha256": mixed["logical_decode_input_sha256"],
            "blocks": mixed["blocks"], "stride": mixed["stride"],
            "tokens": padded_tokens, "actual_tokens": actual,
            "page_assignments": mixed["page_assignments"][:requests],
            "scale": mixed["scale"],
            "source_fixture_sha256": mixed["input_sha256"]}


def _allocate(torch, fixture: dict, device, cores: int, *, mixed: bool) -> dict:
    spec: reuse.Shape = fixture["spec"]
    n, h, d = fixture["tokens"], spec.heads, spec.dim
    cv = reuse._allocate(torch, fixture, device, cores, candidate=mixed)
    return {"cv": cv,
            "output": torch.empty((n * h, d), dtype=torch.float32, device=device),
            "output_lse": torch.empty((n * h,), dtype=torch.float32, device=device),
            "merge_status": torch.empty((n * h,), dtype=torch.int32, device=device),
            "rotation_status": torch.empty((n, h), dtype=torch.int32, device=device)}


def _prepare_rotation(torch, ops, fixture: dict, tensors: dict,
                      buffers: dict, cores: int) -> dict:
    """Execute real prepare/rotate, but exclude both from measured intervals."""
    cv = buffers["cv"]
    preparation = reuse._prepare(torch, ops, tensors, fixture, cv)
    rk = reuse._hadamard(torch, fixture["spec"].dim).T.contiguous().to(tensors["q"].device)
    rotated = torch.empty_like(tensors["qr"])
    buffers["rotation_status"].fill_(-99)
    # Target FULL layers use rotation_method=hadamard. Keep that production
    # branch even though rotation is intentionally outside the Event window.
    ops.rotate_out(tensors["q"], rk, rotated,
                   buffers["rotation_status"], True, tensors["slots"])
    torch.npu.synchronize()
    if not bool((buffers["rotation_status"].cpu() == 0).all()):
        raise MixedAttentionProbeError("production rotate_out left a status error")
    tensors["qr"] = rotated
    return {**preparation,
            "rotation": "real_hadamard_rotate_out_outside_Event_interval",
            "task_prepare": "real_prepare_attention_tasks_out_outside_Event_interval"}


def _oracle(torch, fixture: dict, buffers: dict, acceptance: dict) -> dict:
    spec: reuse.Shape = fixture["spec"]
    n, h, d = fixture["tokens"], spec.heads, spec.dim
    sampled = sorted(fixture["expected"])
    if not sampled:
        raise MixedAttentionProbeError("mixed attention has no independent oracle samples")
    got = buffers["output"].view(n, h, d)[sampled].cpu()
    got_lse = buffers["output_lse"].view(n, h)[sampled].cpu()
    expected = torch.stack([fixture["expected"][index][0] for index in sampled])
    expected_lse = torch.stack([fixture["expected"][index][1] for index in sampled])
    torch.testing.assert_close(got, expected, **acceptance["fused_attention"])
    torch.testing.assert_close(got_lse, expected_lse,
                               **acceptance["fused_attention"])
    cv_status = buffers["cv"]["status"].cpu()
    merge_status = buffers["merge_status"].cpu()
    if not bool((cv_status == 0).all()) or not bool((merge_status == 0).all()):
        raise MixedAttentionProbeError("CV or merge left an error/unwritten status")
    if fixture["actual_tokens"] < n:
        actual = fixture["actual_tokens"]
        partial = buffers["cv"]["partial"][actual:]
        lse = buffers["cv"]["lse"][actual:]
        if not bool((partial == 0).all()) or not bool(torch.isneginf(lse).all()):
            raise MixedAttentionProbeError("padded decode partial/LSE was not zero/-inf")
    return {"status": "passed", "sampled_queries": len(sampled),
            "max_output_abs": float((got - expected).abs().max()),
            "max_lse_abs": float((got_lse - expected_lse).abs().max()),
            "cv_status": "all_zero", "merge_status": "all_zero",
            "padding_partial_lse": "zero_negative_inf" if fixture["actual_tokens"] < n
                                   else "not_applicable"}


def _current_source_written(torch, fixture: dict, buffers: dict,
                            current_output, current_lse) -> None:
    spec: reuse.Shape = fixture["spec"]
    source2 = 2 * spec.splits
    partial = buffers["cv"]["partial"][:, :, source2]
    lse = buffers["cv"]["lse"][:, :, source2]
    if (not torch.equal(partial, current_output.float()) or
            not torch.equal(lse, current_lse.squeeze(-1).float())):
        raise MixedAttentionProbeError("native current partial was not written into source2 split0")
    if spec.splits > 1:
        tail = buffers["cv"]["partial"][:, :, source2 + 1:]
        tail_lse = buffers["cv"]["lse"][:, :, source2 + 1:]
        if not bool((tail == 0).all()) or not bool(torch.isneginf(tail_lse).all()):
            raise MixedAttentionProbeError("CV did not preserve empty later current splits")


def _verify_suppressed_cv(torch, ops, fixture: dict, tensors: dict,
                          buffers: dict, pristine_tasks, cores: int) -> dict:
    """A separate synchronized preflight, never inside the 2+5 Event window."""
    from oscar_ascend.integration.current_attention import suppress_current_source_tasks
    spec: reuse.Shape = fixture["spec"]
    n, splits = fixture["tokens"], spec.splits
    cv = buffers["cv"]
    before_error = pristine_tasks.view(n, spec.kv_heads, 3, splits, 16)[..., 10].clone()
    cv["tasks"].copy_(pristine_tasks)
    reuse._poison(torch, cv)
    suppress_current_source_tasks(cv["tasks"], n, spec.kv_heads, splits)
    from .probe_fast_unpack import _launch as launch_fast
    launch_fast(ops, tensors, fixture, cv, cores, "c4", fast=True)
    torch.npu.synchronize()
    reuse._check_status(torch, cv, invalid=False)
    source2 = cv["partial"][:, :, 2 * splits:3 * splits]
    source2_lse = cv["lse"][:, :, 2 * splits:3 * splits]
    if not bool((source2 == 0).all()) or not bool(torch.isneginf(source2_lse).all()):
        raise MixedAttentionProbeError("suppressed source2 CV was not exact zero/-inf")
    after_error = cv["tasks"].view(n, spec.kv_heads, 3, splits, 16)[..., 10]
    if not torch.equal(after_error, before_error):
        raise MixedAttentionProbeError("source2 suppression changed task metadata errors")
    return {"status": "passed", "scope": "separate_synchronized_preflight_not_timed",
            "source2_partial": "all_zero", "source2_lse": "all_negative_inf",
            "cv_status": "all_zero", "task_error_preserved": True}


def _run_once(torch, ops, fixture: dict, tensors: dict, buffers: dict,
              cores: int, *, mixed: bool, pristine_tasks=None,
              cumulative: tuple[int, ...] | None = None) -> dict:
    """One same-stream Event envelope; no per-phase synchronize or profiler."""
    from oscar_ascend.integration.current_attention import (
        guard_current_slots, native_current_partial,
        suppress_current_source_tasks, write_current_partial)
    spec: reuse.Shape = fixture["spec"]
    n, h, d, splits = fixture["tokens"], spec.heads, spec.dim, spec.splits
    cv = buffers["cv"]
    if mixed:
        if pristine_tasks is None:
            raise MixedAttentionProbeError("mixed tasks lack the unsuppressed source2 snapshot")
        if (cumulative is None or len(cumulative) != len(spec.qlens) or
                cumulative[-1] != n or any(after - before != length
                for before, after, length in zip((0, *cumulative[:-1]),
                                                  cumulative, spec.qlens))):
            raise MixedAttentionProbeError("native TND cumulative lengths differ from current qstarts")
        cv["tasks"].copy_(pristine_tasks)
    reuse._poison(torch, cv)
    buffers["output"].fill_(float("nan"))
    buffers["output_lse"].fill_(float("nan"))
    buffers["merge_status"].fill_(-99)
    stream = torch.npu.current_stream()
    def stamp():
        event = torch.npu.Event(enable_timing=True)
        event.record(stream)
        return event
    start = stamp()
    if mixed:
        suppress_current_source_tasks(cv["tasks"], n, spec.kv_heads, splits)
        after_suppress = stamp()
        from .probe_fast_unpack import _launch as launch_fast
        launch_fast(ops, tensors, fixture, cv, cores, "c4", fast=True)
    else:
        after_suppress = start
        from .probe_fast_unpack import _launch as launch_fast
        launch_fast(ops, tensors, fixture, cv, cores, "base", fast=True)
    after_cv = stamp()
    current_output = current_lse = None
    if mixed:
        guard_current_slots(ops, buffers["merge_status"].view(n, h),
                            tensors["slots"], fixture["actual_tokens"])
        after_guard = stamp()
        current_output, current_lse = native_current_partial(
            tensors["q"], tensors["ck"], tensors["cv"], cumulative,
            heads=h, kv_heads=spec.kv_heads, scale=fixture["scale"])
        write_current_partial(cv["partial"], cv["lse"], current_output,
                              current_lse, fixture["actual_tokens"], splits)
        after_current = stamp()
    else:
        after_guard = after_current = after_cv
    ops.merge_lse_out(cv["partial"].view(n * h, 3 * splits, d),
                      cv["lse"].view(n * h, 3 * splits),
                      buffers["output"], buffers["output_lse"],
                      buffers["merge_status"])
    end = stamp()
    end.synchronize()
    if torch.npu.current_stream() != stream:
        raise MixedAttentionProbeError("attention pipeline changed NPU stream")
    if mixed:
        _current_source_written(torch, fixture, buffers, current_output, current_lse)
        source2 = cv["tasks"].view(n, spec.kv_heads, 3, splits, 16)[:, :, 2]
        begin, end_range, errors = source2[..., 3], source2[..., 4], source2[..., 10]
        if not bool((end_range[errors == 0] == begin[errors == 0]).all()):
            raise MixedAttentionProbeError("mixed source2 task range was not suppressed")
    durations = {"suppress_ms": start.elapsed_time(after_suppress) if mixed else 0.0,
                 "cv_ms": after_suppress.elapsed_time(after_cv),
                 "guard_ms": after_cv.elapsed_time(after_guard) if mixed else 0.0,
                 "current_ms": after_guard.elapsed_time(after_current) if mixed else 0.0,
                 "merge_ms": after_current.elapsed_time(end),
                 "total_ms": start.elapsed_time(end)}
    if any(not math.isfinite(value) or value < 0 for value in durations.values()):
        raise MixedAttentionProbeError("invalid same-stream NPU Event duration")
    return durations


def measure_case(torch, ops, fixture: dict, device, cores: int,
                 acceptance: dict, *, mixed: bool) -> dict:
    """2 warmups and 5 normal Event repeats; per-phase markers never sync."""
    from oscar_ascend.ops.cv_dispatch import (
        FAST_CLUSTER4_CV_OP, FAST_CV_OP, select_cv_op)
    spec: reuse.Shape = fixture["spec"]
    selected = select_cv_op(4, spec.heads, spec.kv_heads, fixture["tokens"],
                            max(spec.qlens), q1_draft=False, fast_unpack=True)
    expected_op = FAST_CLUSTER4_CV_OP if mixed else FAST_CV_OP
    if selected != expected_op:
        raise MixedAttentionProbeError("production candidate route differs from measured fast symbol")
    tensors = {name: tensor.to(device) for name, tensor in fixture["cpu"].items()}
    buffers = _allocate(torch, fixture, device, cores, mixed=mixed)
    prepared = _prepare_rotation(torch, ops, fixture, tensors, buffers, cores)
    pristine = buffers["cv"]["tasks"].clone() if mixed else None
    suppress_preflight = (_verify_suppressed_cv(
        torch, ops, fixture, tensors, buffers, pristine, cores) if mixed else None)
    cumulative = (tuple(int(value) for value in fixture["cpu"]["starts"][1:].tolist())
                  if mixed else None)
    if mixed and (len(cumulative) != 32 or cumulative[-2:] != (124, 16384)):
        raise MixedAttentionProbeError("mixed current FIA must use 31 q4 + one q16260 TND lengths")
    metrics: list[dict] = []
    oracle = None
    for index in range(WARMUP + REPEATS):
        row = _run_once(torch, ops, fixture, tensors, buffers, cores,
                        mixed=mixed, pristine_tasks=pristine,
                        cumulative=cumulative)
        oracle = _oracle(torch, fixture, buffers, acceptance)
        if index >= WARMUP:
            metrics.append(row)
    if len(metrics) != REPEATS or oracle is None:
        raise MixedAttentionProbeError("mixed diagnostic did not finish its 2+5 Event schedule")
    from .probe_fast_unpack import _check_one
    cluster_stats = _check_one(torch, fixture, buffers["cv"],
                               invalid=False, expected_stats=prepared) if mixed else None
    medians = {name: statistics.median(sample[name] for sample in metrics)
               for name in ("suppress_ms", "cv_ms", "guard_ms", "current_ms",
                            "merge_ms", "total_ms")}
    row = {"case": spec.name, "status": "observed", "accuracy": "passed",
           "requests": len(spec.qlens), "total_n": fixture["tokens"],
           "actual_tokens": fixture["actual_tokens"],
           "prompt_q": PREFILL_Q if mixed else 0,
           "decode_q": DECODE_REQUESTS * DECODE_Q,
           "prefill_context": spec.contexts[-1] if mixed else None,
           "decode_contexts": list(spec.contexts[:DECODE_REQUESTS]),
           "splits": spec.splits,
           "operator": selected,
           "current_source": "native_TND_FIA_plus_write" if mixed else "CV_source2",
           "native_current_allocates_output_lse_each_call": mixed,
           "input_sha256": fixture["input_sha256"],
           "logical_decode_input_sha256": fixture["logical_decode_input_sha256"],
           "source_fixture_sha256": fixture.get("source_fixture_sha256"),
           "task_sha256": prepared["task_sha256"],
           "preparation": {"task_prepare": prepared["task_prepare"],
                           "rotation": prepared["rotation"],
                           "outside_timing": True},
           "suppressed_cv_preflight": suppress_preflight,
           "cumulative_current_qstarts": list(cumulative) if cumulative else None,
           "cluster_stats": cluster_stats,
           "oracle": oracle,
           "event_samples_ms": metrics, "median_ms": medians,
           "event_scope": "same_NPU_stream_main_FULL_attention_only_excludes_MTP_draft0_model_GDN_MLP_sampling_store",
           "median_rule": "phase_medians_are_independent_do_not_add_to_total",
           "warmup": WARMUP, "repeats": REPEATS}
    del tensors, buffers, fixture
    gc.collect()
    torch.npu.empty_cache()
    return row


def probe(config_path: Path, acceptance_path: Path) -> dict:
    target = json.loads(config_path.read_text())
    acceptance = json.loads(acceptance_path.read_text())
    policy = acceptance.get("performance", {})
    if (acceptance.get("frozen_before_measurement") is not True or
            policy.get("warmup") != WARMUP or policy.get("repeats") != REPEATS or
            policy.get("statistic") != "median"):
        raise MixedAttentionProbeError("frozen 2+5 diagnostic policy is required")
    from .probe_native_current_fia import _select_target_npu, _target_geometry
    _select_target_npu(target)
    if _target_geometry(target) != (6, 1, 256):
        raise MixedAttentionProbeError("target TP head geometry differs from Hq6/Hkv1/D256")
    if target.get("rotation_method") != "hadamard":
        raise MixedAttentionProbeError("synthetic rotation differs from target hadamard route")
    import torch
    import torch_npu  # noqa: F401 - actual Ascend NPU only
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.npu.set_device(0)
    from .build_ops import normalize_soc
    if (not torch.npu.is_available() or
            normalize_soc(torch.npu.get_device_name(0)) != target["soc_version"]):
        raise MixedAttentionProbeError("selected target NPU/SOC is unavailable")
    from oscar_ascend.ops.loader import require_capabilities, validate_build_artifacts
    manifest_path = ROOT / "build/ascendc/build_manifest.json"
    manifest = validate_build_artifacts(manifest_path)
    require_capabilities({"prepare_attention_tasks_out", "rotate_out",
                          "attention_cv_fast_out", "attention_cv_fast_cluster4_out",
                          "merge_lse_out", "status_guard"}, manifest_path)
    ops = torch.ops.oscar_ascend_ops
    cores = reuse._core_count(torch, target)
    if cores != 20:
        raise MixedAttentionProbeError("target A2 geometry requires 20 Cube cores")
    device = torch.device("npu:0")
    results = []
    decode_hash = None
    for index, spec in enumerate(shape_plan()):
        mixed = make_mixed_fixture(torch, spec)
        if decode_hash is None:
            decode_hash = mixed["logical_decode_input_sha256"]
        elif mixed["logical_decode_input_sha256"] != decode_hash:
            raise MixedAttentionProbeError("cold/warm logical decode Q/K/V/history changed")
        row = measure_case(torch, ops, mixed, device, cores, acceptance, mixed=True)
        results.append(row)
        _print_case(row)
        if index == 1:
            # Both decode controls derive from this SAME warm mixed fixture.
            # N128/S3 versus N16384/S1 changes only padding and split count.
            for n, splits in ((128, 3), (16384, 1)):
                pure = derive_decode_fixture(torch, mixed,
                                             padded_tokens=n, splits=splits)
                if pure["logical_decode_input_sha256"] != decode_hash:
                    raise MixedAttentionProbeError("pure decode lost the common logical cohort")
                control = measure_case(torch, ops, pure, device, cores,
                                       acceptance, mixed=False)
                results.append(control)
                _print_case(control)
        del mixed
        gc.collect()
    if len(results) != 4 or any(row["accuracy"] != "passed" for row in results):
        raise MixedAttentionProbeError("mixed attention diagnostic lacks four accurate shapes")
    return {"status": "observed", "accuracy": "passed",
            "performance_acceptance": "not_run_operator_diagnostic_only",
            "full_model_acceptance": "not_run", "scope": "one_NPU_attention_pipeline_only",
            "note": "pure_decode_has_different_batch_composition; main_FULL_excludes_MTP_draft0_and_other_model_work",
            "logical_decode_input_sha256": decode_hash,
            "artifact_signature": manifest["signature"],
            "artifact_sha256": manifest["sha256"],
            "device": str(device), "device_name": torch.npu.get_device_name(0),
            "physical_devices": target["devices"], "cases": results}


def _print_case(row: dict) -> None:
    med = row["median_ms"]
    print("[oscar] PERF_MIXED_ATTENTION " + json.dumps({
        "case": row["case"], "requests": row["requests"],
        "total_n": row["total_n"], "prompt_q": row["prompt_q"],
        "decode_q": row["decode_q"], "prefill_context": row["prefill_context"],
        "splits": row["splits"], "operator": row["operator"],
        "cv_ms": med["cv_ms"], "current_ms": med["current_ms"],
        "merge_ms": med["merge_ms"], "total_ms": med["total_ms"],
        "oracle": row["oracle"]["status"]}, sort_keys=True), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/target.json")
    parser.add_argument("--acceptance", type=Path,
                        default=ROOT / "configs/acceptance.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.name in {"mixed-attention.json", "status.json"}:
        parser.error("#151: mixed result path must differ from phase state JSON")
    try:
        report = probe(args.config, args.acceptance)
        atomic_json(args.output, report)
        print("[oscar] PERF_MIXED_RESULT " + json.dumps({
            "status": report["status"], "accuracy": report["accuracy"],
            "cases": len(report["cases"]), "report": str(args.output)},
            sort_keys=True), flush=True)
        return 0
    except Exception as error:
        atomic_json(args.output, {"status": "failed", "accuracy": "not_established",
                                 "error_type": type(error).__name__,
                                 "error": str(error),
                                 "device_completion": "not_established"})
        traceback.print_exc()
        print("[oscar] PERF_MIXED_RESULT " + json.dumps({
            "status": "failed", "error": str(error), "report": str(args.output)},
            sort_keys=True), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
