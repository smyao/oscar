# Archive #126/#129/#143/#148-150 and startup D.4: isolate full-history
# attention without restoring compressed KV in production or mistaking a
# standalone operator result for native whole-model graph performance.
"""Model-free target-NPU q4 decode: native BF16 FIA versus OSCAR INT2 CV."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import traceback

from .phase import atomic_json


ROOT = Path(__file__).resolve().parents[1]
CONTEXTS = (20000, 23000, 27000, 30000) * 8
QUERY_LENGTHS = (4,) * 32
NATIVE_BLOCK_SIZE = 128
NATIVE_MASK_SIZE = 2048
PROFILE_FIELDS = (
    "valid_tasks", "kv_units", "kv_rows", "task_span_ticks", "q_stage",
    "packed_dma", "unpack_total", "precise_load", "kv_publish",
    "aiv_qk_wait", "softmax_p", "aiv_pv_wait", "pv_acc", "aic_kv_wait",
    "aic_qk_compute", "aic_p_wait", "aic_pv_compute", "aic_acc_wait",
    "aic_rv_compute", "aiv_finalize", "unpack_bits", "unpack_meta",
    "rotation_wait", "actor_compute_span_ticks",
)
ENGINES = ("AIC", "AIV0", "AIV1")
SOURCES = ("history", "window", "current", "total")


class DecodeHotpathError(RuntimeError):
    pass


def _tensor_bytes(tensor) -> bytes:
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise DecodeHotpathError("logical input hash requires contiguous CPU tensors")
    return ctypes.string_at(tensor.data_ptr(), tensor.numel() * tensor.element_size())


class NativeBF16Builder:
    """Observe the fe0 fixture's original logical tensors before quantization."""

    def __init__(self, torch, spec, *, block_size: int = NATIVE_BLOCK_SIZE):
        if block_size != 128 or spec.kv_heads != 1 or spec.heads != 6:
            raise DecodeHotpathError("native diagnostic requires signed TP4 Hq6/Hkv1/128-block geometry")
        self.torch = torch
        self.spec = spec
        self.block_size = block_size
        self.page_counts = tuple(math.ceil((context + length) / block_size)
                                 for context, length in zip(spec.contexts, spec.qlens))
        self.pages = list(range(sum(self.page_counts)))
        random.Random(46817 + sum(spec.qlens)).shuffle(self.pages)
        self.table = torch.zeros((len(spec.qlens), max(self.page_counts)), dtype=torch.int32)
        blocks = len(self.pages)
        self.key = torch.zeros((blocks, block_size, spec.kv_heads, spec.dim), dtype=torch.bfloat16)
        self.value = torch.zeros_like(self.key)
        self.query = torch.empty((sum(spec.qlens), spec.heads, spec.dim), dtype=torch.bfloat16)
        self.expected_output = torch.empty((sum(spec.qlens), spec.heads, spec.dim), dtype=torch.float32)
        self.q_cumulative = []
        self.kv_lengths = []
        self._next_request = 0
        self._next_page = 0
        self.logical_sha256 = hashlib.sha256()

    def on_request(self, request, query_begin, query, old_k, old_v, current_k, current_v):
        from oscar_ascend.ops.reference import attention

        torch = self.torch
        spec = self.spec
        if request != self._next_request or query_begin != sum(spec.qlens[:request]):
            raise DecodeHotpathError("fe0 fixture request order changed")
        context, length = spec.contexts[request], spec.qlens[request]
        if (tuple(query.shape) != (length, spec.heads, spec.dim) or
                tuple(old_k.shape) != (context, spec.kv_heads, spec.dim) or
                tuple(old_v.shape) != tuple(old_k.shape) or
                tuple(current_k.shape) != (length, spec.kv_heads, spec.dim) or
                tuple(current_v.shape) != tuple(current_k.shape)):
            raise DecodeHotpathError("native observer received changed logical geometry")
        for tensor in (query, old_k, old_v, current_k, current_v):
            self.logical_sha256.update(_tensor_bytes(tensor.contiguous()))
        self.query[query_begin:query_begin + length] = query
        full_k = torch.cat((old_k, current_k))
        full_v = torch.cat((old_v, current_v))
        pages = self.pages[self._next_page:self._next_page + self.page_counts[request]]
        self._next_page += len(pages)
        for logical, physical in enumerate(pages):
            begin = logical * self.block_size
            end = min(begin + self.block_size, context + length)
            self.key[physical, :end - begin] = full_k[begin:end]
            self.value[physical, :end - begin] = full_v[begin:end]
            self.table[request, logical] = physical
        for local in range(length):
            index = query_begin + local
            reference = attention(query[local:local + 1], full_k[:context + local + 1],
                                  full_v[:context + local + 1], scale=spec.dim ** -0.5,
                                  causal=False)
            self.expected_output[index] = reference.output[0]
        self.q_cumulative.append(query_begin + length)
        self.kv_lengths.append(context + length)
        self._next_request += 1

    def result(self) -> dict:
        if self._next_request != len(self.spec.qlens) or self._next_page != len(self.pages):
            raise DecodeHotpathError("native observer omitted a request or physical page")
        if sorted(self.pages) != list(range(len(self.pages))):
            raise DecodeHotpathError("native BF16 pages are not disjoint")
        return {"query": self.query, "key": self.key, "value": self.value,
                "block_table": self.table, "q_cumulative": self.q_cumulative,
                "kv_lengths": self.kv_lengths, "expected_output": self.expected_output,
                "logical_input_sha256": self.logical_sha256.hexdigest(),
                "page_counts": self.page_counts,
                "table_scope": "minimal_legal_synthetic_width_not_service_stride"}


def build_paired_fixture(torch, spec):
    from .probe_history_reuse import make_fixture

    native = NativeBF16Builder(torch, spec)
    oscar = make_fixture(torch, spec, on_request=native.on_request)
    return oscar, native.result()


def native_kwargs(torch, native: dict, device, scale: float) -> tuple[dict, object]:
    """Mirror attention_v1.py full_graph_fia's BF16 TND .out arguments."""
    if not math.isfinite(scale) or scale <= 0:
        raise DecodeHotpathError("native FIA scale is invalid")
    mask = torch.triu(torch.ones((NATIVE_MASK_SIZE, NATIVE_MASK_SIZE),
                                dtype=torch.int8, device=device), diagonal=1)
    query = native["query"].to(device)
    key = native["key"].to(device).view(-1, NATIVE_BLOCK_SIZE, native["key"].shape[-1])
    value = native["value"].to(device).view(-1, NATIVE_BLOCK_SIZE, native["value"].shape[-1])
    table = native["block_table"].to(device)
    kwargs = dict(query=query, key=key, value=value, atten_mask=mask,
                  block_table=table, input_layout="TND", block_size=NATIVE_BLOCK_SIZE,
                  actual_seq_lengths=native["q_cumulative"],
                  actual_seq_lengths_kv=native["kv_lengths"],
                  num_key_value_heads=1, num_heads=6, scale=scale, sparse_mode=3,
                  pre_tokens=2147483647, next_tokens=2147483647)
    return kwargs, mask


def _validate_native(torch, observed, native: dict, tolerance: dict) -> dict:
    expected = native["expected_output"]
    if (tuple(observed.shape) != tuple(expected.shape) or observed.dtype != torch.bfloat16 or
            observed.device.type != "npu"):
        raise DecodeHotpathError("native BF16 FIA output shape, dtype or device differs from graph path")
    actual = observed.float().cpu()
    torch.testing.assert_close(actual, expected, **tolerance)
    return {"status": "passed", "queries": expected.shape[0],
            "max_output_abs": float((actual - expected).abs().max()),
            "lse": "not_produced_by_native_graph_path"}


def _profile_summary(values: list, cores: int) -> dict:
    if len(values) != cores or any(len(core) != 3 for core in values):
        raise DecodeHotpathError("profile core/engine geometry is incomplete")
    critical = {}
    for core, engines in enumerate(values):
        for engine, sources in enumerate(engines):
            if len(sources) != 4 or any(len(vector) != len(PROFILE_FIELDS) for vector in sources):
                raise DecodeHotpathError("profile source/field geometry is incomplete")
            if any(type(value) is not int or value < 0 for vector in sources for value in vector):
                raise DecodeHotpathError("profile contains invalid raw counters")
            for field in range(23):
                if sources[3][field] != sum(sources[source][field] for source in range(3)):
                    raise DecodeHotpathError("profile total source does not equal separate sources")
            if any(sources[source][23] != 0 for source in range(3)):
                raise DecodeHotpathError("profile whole span must belong only to total source")
            if sources[3][23] <= 0:
                raise DecodeHotpathError(f"profile actor counters were not completed: core={core} engine={engine}")
    for source in range(3):
        source_actors = {}
        for role, engines in (("AIC", (0,)), ("AIV", (1, 2))):
            choices = [(values[core][engine][source][3], core, engine)
                       for core in range(cores) for engine in engines
                       if values[core][engine][source][0] > 0]
            if not choices:
                raise DecodeHotpathError(f"profile has no live {SOURCES[source]} {role} actor")
            span, core, engine = max(choices)
            if span <= 0:
                raise DecodeHotpathError(f"profile has no completed {SOURCES[source]} {role} task span")
            vector = values[core][engine][source]
            source_actors[role] = {"core": core, "engine": ENGINES[engine],
                "task_span_raw_ticks": span,
                "fields_raw_ticks_or_counts": dict(zip(PROFILE_FIELDS, vector)),
                "same_actor_fraction_of_task_span": {
                    PROFILE_FIELDS[field]: vector[field] / span for field in range(4, 23)},
                "scope": "one_actor_one_source_raw_SYS_CNT_no_cross_core_sum"}
        critical[SOURCES[source]] = source_actors
    whole = max((values[core][engine][3][23], core, engine)
                for core in range(cores) for engine in range(3))
    if whole[0] <= 0:
        raise DecodeHotpathError("profile did not record any completed actor compute span")
    return {"critical_source_actors": critical,
            "critical_compute_actor": {"raw_ticks": whole[0], "core": whole[1],
                                     "engine": ENGINES[whole[2]]},
            "field_contract": "20+21 nested in 6; 22 nested in rotation/final; field23 is one actor Init+Process without flush, not kernel wall; no cross-core sum"}


def _aligned_profile(torch, device, cores: int):
    elements = cores * 3 * 4 * len(PROFILE_FIELDS)
    storage = torch.empty(elements + 8, dtype=torch.int64, device=device)
    offset = ((-storage.data_ptr()) % 64) // 8
    profile = storage[offset:offset + elements].view(cores, 3, 4, len(PROFILE_FIELDS))
    if profile.data_ptr() % 64 or not profile.is_contiguous():
        raise DecodeHotpathError("profile buffer is not 64-byte aligned")
    return profile, storage


def _profile_call(ops, tensors: dict, fixture: dict, buffers: dict, profile, cores: int):
    from .probe_history_reuse import BLOCK_TOKENS, PREFIX, SINK, SPECULATIVE

    spec = fixture["spec"]
    args = (tensors["q"], tensors["qr"], tensors["ck"], tensors["cv"], tensors["rv"],
            tensors["raw"], tensors["table"], tensors["wk"], tensors["wv"], tensors["tags"],
            buffers["tasks"], buffers["partial"], buffers["lse"], buffers["status"],
            buffers["workspace"], profile, BLOCK_TOKENS, fixture["blocks"], PREFIX,
            fixture["stride"], SINK, spec.recent_tokens, SPECULATIVE, spec.splits,
            fixture["scale"], cores)
    ops.attention_cv_profile_out(*args)


def _timed_pairs(torch, torch_npu, ops, native_kwargs_: dict, native_out,
                 native_lse, native_workspace, tensors: dict, fixture: dict,
                 buffers: dict, merge_out, merge_lse, merge_status, cores: int,
                 warmup: int, repeats: int, native_cpu: dict,
                 tolerance: dict, reference_bits: dict) -> dict:
    from .probe_history_reuse import _assert_same_bits, _check_status, _launch

    stream = torch.npu.current_stream()
    n, h, d, s = fixture["tokens"], fixture["spec"].heads, fixture["spec"].dim, 3 * fixture["spec"].splits
    times = {"native_fia": [], "oscar_cv": [], "oscar_cv_merge": []}
    for iteration in range(warmup + repeats):
        order = ("native", "oscar") if iteration < warmup or (iteration - warmup) % 2 == 0 else ("oscar", "native")
        for kind in order:
            begin = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            if kind == "native":
                begin.record(stream)
                torch_npu.npu_fused_infer_attention_score.out(
                    **native_kwargs_, workspace=native_workspace,
                    out=[native_out, native_lse])
                end.record(stream)
                end.synchronize()
                if iteration >= warmup:
                    times["native_fia"].append(float(begin.elapsed_time(end)))
                    _validate_native(torch, native_out, native_cpu, tolerance)
                continue
            buffers["partial"].fill_(float("nan"))
            buffers["lse"].fill_(float("nan"))
            buffers["status"].fill_(-99)
            middle = torch.npu.Event(enable_timing=True)
            begin.record(stream)
            _launch(ops, tensors, fixture, buffers, cores, candidate=False)
            middle.record(stream)
            ops.merge_lse_out(buffers["partial"].view(n * h, s, d),
                              buffers["lse"].view(n * h, s), merge_out,
                              merge_lse, merge_status)
            end.record(stream)
            end.synchronize()
            if iteration >= warmup:
                times["oscar_cv"].append(float(begin.elapsed_time(middle)))
                times["oscar_cv_merge"].append(float(begin.elapsed_time(end)))
                _check_status(torch, buffers, invalid=False)
                _assert_same_bits(torch, reference_bits, buffers, "fe0 timed q4")
                if not bool((merge_status.cpu() == 0).all()):
                    raise DecodeHotpathError("timed OSCAR merge reported nonzero status")
    if any(len(values) != repeats or any(not math.isfinite(ms) or ms <= 0 for ms in values)
           for values in times.values()):
        raise DecodeHotpathError("NPU Event timing was incomplete or invalid")
    return {name: {"device_event_ms": values, "median_ms": statistics.median(values),
                   "warmup": warmup, "repeats": repeats}
            for name, values in times.items()}


def probe(config_path: Path, acceptance_path: Path) -> dict:
    from .probe_history_reuse import (Shape, _active_device, _allocate, _assert_same_bits,
        _check_status, _core_count, _launch, _merge_and_oracle, _poison, _prepare, _snapshot)

    target = json.loads(config_path.read_text())
    acceptance = json.loads(acceptance_path.read_text())
    _active_device(target)
    perf = acceptance["performance"]
    if (acceptance.get("frozen_before_measurement") is not True or
            perf.get("warmup") != 2 or perf.get("repeats") != 5 or
            perf.get("statistic") != "median"):
        raise DecodeHotpathError("frozen 2+5 median acceptance policy is required")
    import torch
    import torch_npu  # noqa: F401 - fail closed without target NPU
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.npu.set_device(0)
    from .build_ops import normalize_soc
    if not torch.npu.is_available() or normalize_soc(torch.npu.get_device_name(0)) != target["soc_version"]:
        raise DecodeHotpathError("selected target NPU/SOC is unavailable")
    from .probe_native_current_fia import _target_geometry
    if _target_geometry(target) != (6, 1, 256):
        raise DecodeHotpathError("target model TP4 head geometry differs from Hq6/Hkv1/D256")
    from oscar_ascend.runtime import WorkspaceGeometry
    from oscar_ascend.ops.loader import require_capabilities, validate_build_artifacts
    manifest_path = ROOT / "build/ascendc/build_manifest.json"
    manifest = validate_build_artifacts(manifest_path)
    require_capabilities({"prepare_attention_tasks_out", "attention_cv_out",
                          "attention_cv_profile_out", "merge_lse_out"}, manifest_path)
    ops = torch.ops.oscar_ascend_ops
    device = torch.device("npu:0")
    cores = _core_count(torch, target)
    if cores != 20:
        raise DecodeHotpathError(f"signed q4 diagnostic expects 20 Cube cores, got {cores}")
    capacity = max(int(target["max_num_batched_tokens"]),
                   max(target.get("compilation_config", {}).get("cudagraph_capture_sizes", ()), default=0))
    geometry = WorkspaceGeometry(capacity, 6, 1, 256,
                                 int(target.get("attention_splits", 1)), cores)
    if geometry.splits_for_tokens(128) != 3:
        raise DecodeHotpathError("production q4 N128 source split count differs from S3")
    spec = Shape("native_decode_q4_32", QUERY_LENGTHS, CONTEXTS, 256, 1, 3,
                 False, speed_gate=False)
    fixture, native_cpu = build_paired_fixture(torch, spec)
    tensors = {name: tensor.to(device) for name, tensor in fixture["cpu"].items()}
    buffers = _allocate(torch, fixture, device, cores, candidate=False)
    preparation = _prepare(torch, ops, tensors, fixture, buffers)
    _poison(torch, buffers)
    _launch(ops, tensors, fixture, buffers, cores, candidate=False)
    torch.npu.synchronize()
    _check_status(torch, buffers, invalid=False)
    oscar_oracle = _merge_and_oracle(torch, ops, fixture, buffers, acceptance["fused_attention"])
    baseline_bits = _snapshot(buffers)

    kwargs, mask = native_kwargs(torch, native_cpu, device, fixture["scale"])
    workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(**kwargs)
    native_out = torch.empty_like(kwargs["query"])
    native_lse = torch.empty((1,), dtype=kwargs["query"].dtype, device=device)
    torch_npu.npu_fused_infer_attention_score.out(
        **kwargs, workspace=workspace, out=[native_out, native_lse])
    torch.npu.synchronize()
    native_oracle = _validate_native(torch, native_out, native_cpu, acceptance["fused_attention"])

    n, h, d, s = fixture["tokens"], spec.heads, spec.dim, 3 * spec.splits
    merge_out = torch.empty((n * h, d), dtype=torch.float32, device=device)
    merge_lse = torch.empty((n * h,), dtype=torch.float32, device=device)
    merge_status = torch.empty((n * h,), dtype=torch.int32, device=device)
    torch.npu.synchronize()
    timings = _timed_pairs(torch, torch_npu, ops, kwargs, native_out, native_lse,
                           workspace, tensors, fixture, buffers, merge_out,
                           merge_lse, merge_status, cores, perf["warmup"], perf["repeats"],
                           native_cpu, acceptance["fused_attention"], baseline_bits)
    _check_status(torch, buffers, invalid=False)
    _assert_same_bits(torch, baseline_bits, buffers, "fe0 timed q4")
    if not bool((merge_status.cpu() == 0).all()):
        raise DecodeHotpathError("timed OSCAR merge reported nonzero status")
    _validate_native(torch, native_out, native_cpu, acceptance["fused_attention"])

    profile_buffers = _allocate(torch, fixture, device, cores, candidate=False)
    profile_buffers["tasks"] = buffers["tasks"]
    profile, profile_storage = _aligned_profile(torch, device, cores)
    # The independent profile symbol has its own first-launch registration.
    # Warm it before reporting instrumentation overhead against a warm CV.
    for _ in range(perf["warmup"]):
        profile.zero_()
        _poison(torch, profile_buffers)
        _profile_call(ops, tensors, fixture, profile_buffers, profile, cores)
        torch.npu.synchronize()
        _check_status(torch, profile_buffers, invalid=False)
        _assert_same_bits(torch, baseline_bits, profile_buffers, "profile warmup versus fe0 q4")
    profile.zero_()
    _poison(torch, profile_buffers)
    stream = torch.npu.current_stream()
    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    begin.record(stream)
    _profile_call(ops, tensors, fixture, profile_buffers, profile, cores)
    end.record(stream)
    end.synchronize()
    profile_event_ms = float(begin.elapsed_time(end))
    _check_status(torch, profile_buffers, invalid=False)
    _assert_same_bits(torch, baseline_bits, profile_buffers, "profile versus fe0 q4")
    raw_profile = profile.cpu().tolist()
    summary = _profile_summary(raw_profile, cores)
    if not math.isfinite(profile_event_ms) or profile_event_ms <= 0:
        raise DecodeHotpathError("profile NPU Event was invalid")
    del profile_storage, mask  # Keep storage and mask alive through all launches.
    return {"status": "observed", "scope": "model_free_operator_eager_q4_not_whole_model_graph",
            "native_oracle": "passed", "oscar_oracle": "passed",
            "profile_parity": "bitwise_passed", "profile_observed": True,
            "artifact_signature": manifest["signature"], "artifact_sha256": manifest["sha256"],
            "device": str(device), "device_name": torch.npu.get_device_name(0),
            "physical_devices": target["devices"], "cube_cores": cores,
            "shape": {"requests": 32, "query_lengths": QUERY_LENGTHS,
                      "contexts": CONTEXTS, "tokens": 128, "query_heads": 6,
                      "kv_heads": 1, "head_dim": 256, "oscar_source_splits": 3,
                      "native_block_size": 128,
                      "native_table_columns": native_cpu["block_table"].shape[1],
                      "native_table_scope": native_cpu["table_scope"],
                      "native_physical_pages": native_cpu["key"].shape[0]},
            "logical_input_sha256": native_cpu["logical_input_sha256"],
            "oscar_fixture_sha256": fixture["input_sha256"],
            "oscar_task_sha256": preparation["task_sha256"],
            "accuracy": {"native_bf16": native_oracle,
                         "oscar_int2": {key: value for key, value in oscar_oracle.items()
                                        if key not in ("output", "lse")}},
            "timings": {**timings,
                        "scope": "NPU_Event_eager_only; native_FIA_vs_OSCAR_CV_plus_merge_have_distinct_math",
                        "native_cache_fill_included": False,
                        "oscar_quantize_prepare_store_included": False,
                        "per_measured_repeat_accuracy_checks": perf["repeats"]},
            "profile": {"profiling_only": True, "debug_instrumented": True,
                        "warmup": perf["warmup"], "event_samples": 1,
                        "extra_local_completion_fences": ["FIX_S", "MTE2_S", "V_S", "MTE3_S"],
                        "overlap_changed_by_instrumentation": True,
                        "field_names": PROFILE_FIELDS, "engines": ENGINES, "sources": SOURCES,
                        "raw_SYS_CNT": raw_profile, **summary,
                        "event_ms": profile_event_ms,
                        "over_normal_cv_median": profile_event_ms / timings["oscar_cv"]["median_ms"],
                        "timing_scope": "instrumented_kernel_diagnostic_not_speed_gate"}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/target.json")
    parser.add_argument("--acceptance", type=Path, default=ROOT / "configs/acceptance.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = probe(args.config, args.acceptance)
        atomic_json(args.output, report)
        timing = report["timings"]
        print("[oscar] PERF_Q4_COMPARE " + json.dumps({
            "status": report["status"], "native_fia_ms": timing["native_fia"]["median_ms"],
            "oscar_cv_ms": timing["oscar_cv"]["median_ms"],
            "oscar_cv_merge_ms": timing["oscar_cv_merge"]["median_ms"],
            "scope": report["scope"], "speed_gate": "diagnostic_not_service_acceptance"},
            sort_keys=True), flush=True)
        actors = report["profile"]["critical_source_actors"]
        for source in ("history", "window", "current"):
            selected = {}
            for role in ("AIC", "AIV"):
                item = actors[source][role]
                fields = item["fields_raw_ticks_or_counts"]
                ranked = sorted(((name, fields[name]) for name in PROFILE_FIELDS[4:20]
                                 if fields[name] > 0), key=lambda pair: (-pair[1], pair[0]))
                top = dict(ranked[:6])
                for nested in ("unpack_bits", "unpack_meta"):
                    if fields[nested] > 0:
                        top[nested] = fields[nested]
                if fields["rotation_wait"] > 0:
                    top["rotation_wait"] = fields["rotation_wait"]
                selected[role] = {"core": item["core"], "engine": item["engine"],
                                  "task_span_raw_ticks": item["task_span_raw_ticks"],
                                  "same_actor_top_stages_raw_ticks": top}
            line = {"source": source, "actors": selected,
                    "nested_subitems": "bits/meta belong to unpack_total; AIV rotation_wait belongs to finalize; AIC rotation_wait is separate",
                    "scope": "same_actor_source_raw_ticks_profile_only"}
            if source == "history":
                line["profile_event_ms"] = report["profile"]["event_ms"]
                line["over_normal_cv_median"] = report["profile"]["over_normal_cv_median"]
            print("[oscar] PERF_Q4_PROFILE " + json.dumps(line, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        failure = {"status": "failed", "error": f"{type(error).__name__}: {error}",
                   "traceback": traceback.format_exc(),
                   "scope": "model_free_operator_eager_q4_not_whole_model_graph"}
        atomic_json(args.output, failure)
        print("[oscar] PERF_Q4_COMPARE " + json.dumps({"status": "failed",
            "error": failure["error"], "report": str(args.output)}, sort_keys=True),
            file=sys.stderr, flush=True)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
