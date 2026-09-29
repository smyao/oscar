# Archive #126/#129/#140-153 and startup D.4: isolated old-C4/C16 versus
# bounded B4 fused INT2 attention. Never restore full history (D.4: 6.5s/card).
# Precision, old/new bitwise, graph and same-input 2+5 Event speed all gate.
"""Short real-NPU B4 history probe; no model or AISBench workload."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
import math
from pathlib import Path
import statistics
import traceback

from . import probe_fast_unpack as fast
from . import probe_history_reuse as reuse
from . import probe_mixed_attention as mixed
from . import probe_mixed_optimization as opt
from .phase import atomic_json

ROOT = Path(__file__).resolve().parents[1]
FAST_C4 = opt.OLD_C4
FAST_C16 = opt.NEW_C16
B4 = "attention_cv_batched4_out"
WARMUP, REPEATS = 2, 5


class BatchedHistoryProbeError(RuntimeError):
    pass


def b4_workspace_per_core(dim: int) -> int:
    """Mirror csrc/include/oscar_batched4_experimental.h:11-16."""
    if dim not in (64, 128, 256):
        raise BatchedHistoryProbeError("unsupported B4 dimension")
    m, b = 128, 256
    return ((3 * 4 * m + 2 * b) * dim + 4 * m * b + 3 * 4 * m) * 4


def case_plan() -> tuple[reuse.Shape, ...]:
    warm = mixed.shape_plan()[1]
    long = opt.shape_plan()[2]
    continuation = reuse.Shape("long_continuation_20k", (4096,), (16000,),
                               256, 1, 1, True)
    return continuation, warm, long


def _old_symbol(spec: reuse.Shape, tokens: int) -> str:
    from oscar_ascend.ops.cv_dispatch import select_cv_op
    selected = select_cv_op(16, spec.heads, spec.kv_heads, tokens,
                            max(spec.qlens), fast_unpack=True, mixed_cv=True)
    if selected not in {FAST_C4, FAST_C16}:
        raise BatchedHistoryProbeError(f"{spec.name}: old production route {selected} is not C4/C16")
    return selected


def _proxy(ops, selected: str):
    return opt._RedirectOps(ops, FAST_C4, selected)


def _buffers(torch, fixture: dict, device, cores: int, old_name: str):
    old = mixed._allocate(torch, fixture, device, cores, mixed=True)
    new = mixed._allocate(torch, fixture, device, cores, mixed=True)
    if old_name == FAST_C16:
        old["cv"]["workspace"] = torch.empty(
            cores * opt._c16_workspace_per_core(fixture["spec"].dim),
            dtype=torch.uint8, device=device)
    new["cv"]["workspace"] = torch.empty(
        cores * b4_workspace_per_core(fixture["spec"].dim),
        dtype=torch.uint8, device=device)
    new["cv"]["tasks"] = old["cv"]["tasks"]
    new["cv"]["positions"] = old["cv"]["positions"]
    return old, new


def _stats(torch, cv: dict, prepared: dict, spec: reuse.Shape,
           symbol: str) -> dict:
    if symbol == FAST_C16:
        return opt._check_c16_stats(torch, cv, prepared["source0_leaders"],
                                    require_clusters=spec.expect_clusters)
    return reuse._check_cluster_stats(cv["cluster_stats"].cpu(),
        expect_clusters=spec.expect_clusters,
        expected_source0_leaders=prepared["source0_leaders"])


def _verdict(samples: dict, old_name: str, limit: float) -> dict:
    if old_name == B4 or limit != 1.0 or any(len(samples[k]) != REPEATS
                                            for k in ("old", "new")):
        raise BatchedHistoryProbeError("B4 requires distinct operators and strict 2+5 policy")
    old = {key: statistics.median(row[key] for row in samples["old"])
           for key in ("cv_ms", "total_ms")}
    new = {key: statistics.median(row[key] for row in samples["new"])
           for key in ("cv_ms", "total_ms")}
    if any(not math.isfinite(x) or x <= 0 for x in (*old.values(), *new.values())):
        raise BatchedHistoryProbeError("invalid B4 device Event median")
    ratios = {key: new[key] / old[key] for key in old}
    return {"old": old, "new": new, "ratios": ratios,
            "status": "passed" if all(value <= limit for value in ratios.values()) else "failed",
            "ratio_scope": "distinct_actual_old_C4_or_C16_vs_B4",
            "warmup": WARMUP, "repeats": REPEATS,
            "order": "AB_BA_alternating",
            "scope": "same_NPU_stream_CV_and_full_attention_pipeline_not_model"}


def _run_case(torch, ops, fixture: dict, device, cores: int,
              acceptance: dict) -> dict:
    spec = fixture["spec"]
    old_name = _old_symbol(spec, fixture["tokens"])
    tensors = {name: value.to(device) for name, value in fixture["cpu"].items()}
    old, new = _buffers(torch, fixture, device, cores, old_name)
    prepared = mixed._prepare_rotation(torch, ops, fixture, tensors, old, cores)
    pristine = old["cv"]["tasks"].clone()
    cumulative = tuple(int(value) for value in fixture["cpu"]["starts"][1:].tolist())
    old_ops, new_ops = _proxy(ops, old_name), _proxy(ops, B4)
    for selected_ops, buffers in ((old_ops, old), (new_ops, new)):
        mixed._verify_suppressed_cv(torch, selected_ops, fixture, tensors,
                                     buffers, pristine, cores)

    def run(selected_ops, buffers):
        return mixed._run_once(torch, selected_ops, fixture, tensors, buffers,
                               cores, mixed=True, pristine_tasks=pristine,
                               cumulative=cumulative)

    run(old_ops, old)
    old_oracle = mixed._oracle(torch, fixture, old, acceptance)
    reference = opt._reference_bits(old)
    old_stats = _stats(torch, old["cv"], prepared, spec, old_name)
    run(new_ops, new)
    new_oracle = mixed._oracle(torch, fixture, new, acceptance)
    opt._check_reference_bits(torch, reference, new, spec.name + " B4")
    new_stats = _stats(torch, new["cv"], prepared, spec, B4)
    samples = {"old": [], "new": []}
    for index in range(WARMUP + REPEATS):
        order = ("old", "new") if index < WARMUP or (index - WARMUP) % 2 == 0 \
            else ("new", "old")
        for label in order:
            selected_ops, buffers = ((old_ops, old) if label == "old"
                                     else (new_ops, new))
            row = run(selected_ops, buffers)
            mixed._oracle(torch, fixture, buffers, acceptance)
            opt._check_reference_bits(torch, reference, buffers,
                                      spec.name + " " + label + " repeat")
            if index >= WARMUP:
                samples[label].append(row)
    performance = _verdict(samples, old_name,
                           acceptance["performance"]["max_latency_ratio"])
    result = {"case": spec.name, "status": performance["status"],
              "precision": "passed", "old_operator": old_name,
              "new_operator": B4, "input_sha256": fixture["input_sha256"],
              "shape": {"requests": len(spec.qlens), "total_n": fixture["tokens"],
                        "max_query_len": max(spec.qlens), "max_context": max(spec.contexts),
                        "splits": spec.splits},
              "task_sha256": prepared["task_sha256"],
              "oracle": {"old": old_oracle, "new": new_oracle},
              "partial_lse_status_merged": "bitwise_passed",
              "stats": {"old": old_stats, "new": new_stats},
              "performance": performance, "event_samples_ms": samples,
              "source2_policy": "suppressed_current_plus_native_FIA",
              "scope": "one_MAIN_FULL_attention_only_not_MTP_GDN_or_AISBench"}
    del tensors, old, new
    gc.collect();torch.npu.empty_cache()
    return result


def _active_b4_cluster(tasks_cpu, fixture: dict) -> tuple[int, int]:
    spec = fixture["spec"]
    if spec.kv_heads != 1 or spec.splits != 1 or len(spec.qlens) != 1:
        raise BatchedHistoryProbeError("B4 graph fault needs one request/head/split")
    tile = 128 // (spec.heads // spec.kv_heads)
    width = 4 * tile
    rows = tasks_cpu.view(fixture["tokens"], 1, 3, 1, 16)[:, 0, 0, 0]
    for anchor in range(0, fixture["tokens"] - width + 1, width):
        members = [rows[anchor + j * tile].tolist() for j in range(4)]
        first = members[0]
        if first[1] != tile or first[3] < reuse.SINK or first[4] != spec.contexts[0]:
            continue
        if all(row[0] == anchor + j * tile and row[1] == tile and
               row[3:6] == first[3:6] and row[8:11] == first[8:11]
               for j, row in enumerate(members)):
            return anchor, first[3] + min(16, first[4] - first[3] - 1)
    raise BatchedHistoryProbeError("small graph fixture has no active C4/B4 cluster")


def _graph_case(torch, ops, device, cores: int, acceptance: dict) -> dict:
    spec = reuse.Shape("b4_active_graph", (1024,), (641,), 256, 1, 1,
                       True, recent_tokens=32)
    fixture = reuse.make_fixture(torch, spec)
    tensors = {name: value.to(device) for name, value in fixture["cpu"].items()}
    old, new = _buffers(torch, fixture, device, cores, FAST_C4)
    prepared = reuse._prepare(torch, ops, tensors, fixture, old["cv"])
    old_ops, new_ops = _proxy(ops, FAST_C4), _proxy(ops, B4)

    def launch(selected_ops, buffers):
        reuse._poison(torch, buffers["cv"])
        fast._launch(selected_ops, tensors, fixture, buffers["cv"], cores,
                     "c4", fast=True)
        torch.npu.synchronize()
        reuse._check_status(torch, buffers["cv"], invalid=False)

    launch(old_ops, old)
    reference = reuse._snapshot(old["cv"])
    old_merge = reuse._merge_and_oracle(torch, ops, fixture, old["cv"],
                                        acceptance["fused_attention"])
    launch(new_ops, new)
    reuse._assert_same_bits(torch, reference, new["cv"], "B4 graph eager")
    new_merge = reuse._merge_and_oracle(torch, ops, fixture, new["cv"],
                                        acceptance["fused_attention"])
    for key in ("output", "lse"):
        if not reuse._bitwise_identical(torch, old_merge[key], new_merge[key]):
            raise BatchedHistoryProbeError(f"B4 graph eager merged {key} differs")
    _stats(torch, new["cv"], prepared, spec, B4)
    anchor, position = _active_b4_cluster(old["cv"]["tasks"].cpu(), fixture)
    original_raw, original_qr = tensors["raw"].clone(), tensors["qr"].clone()
    faults = []
    try:
        for label in ("shared_live_scale", "one_group_QR_nan"):
            tensors["raw"].copy_(original_raw);tensors["qr"].copy_(original_qr)
            if label == "shared_live_scale":
                raw = fixture["cpu"]["raw"].clone()
                fast._write_half_bits(fixture, raw, position, "k", "scale", 0x7e01)
                tensors["raw"].copy_(raw.to(device))
            else:
                tensors["qr"][anchor + 2 * 21, 0, 0] = float("nan")
            statuses = []
            for selected_ops, buffers in ((old_ops, old), (new_ops, new)):
                reuse._poison(torch, buffers["cv"])
                fast._launch(selected_ops, tensors, fixture, buffers["cv"], cores,
                             "c4", fast=True)
                torch.npu.synchronize()
                statuses.append(buffers["cv"]["status"].cpu().clone())
            if not torch.equal(*statuses) or not bool((statuses[0] != 0).any()):
                raise BatchedHistoryProbeError(f"{label}: B4 final status differs or misses fault")
            faults.append({"case": label, "status": "passed", "final_status": "bitwise"})
    finally:
        tensors["raw"].copy_(original_raw);tensors["qr"].copy_(original_qr)
        torch.npu.synchronize()

    graph_old, graph_new = torch.npu.NPUGraph(), torch.npu.NPUGraph()
    reuse._poison(torch, old["cv"]);reuse._poison(torch, new["cv"])
    torch.npu.synchronize()
    with torch.npu.graph(graph_old, capture_error_mode="thread_local",
                         auto_dispatch_capture=True):
        fast._launch(old_ops, tensors, fixture, old["cv"], cores, "c4", fast=True)
    with torch.npu.graph(graph_new, capture_error_mode="thread_local",
                         auto_dispatch_capture=True):
        fast._launch(new_ops, tensors, fixture, new["cv"], cores, "c4", fast=True)
    torch.npu.synchronize()
    original_q, original_qr = tensors["q"].clone(), tensors["qr"].clone()
    try:
        for changed in (False, True):
            if changed:
                tensors["q"].copy_(-original_q);tensors["qr"].copy_(-original_qr)
            reuse._poison(torch, old["cv"]);reuse._poison(torch, new["cv"])
            graph_old.replay();graph_new.replay();torch.npu.synchronize()
            reuse._check_status(torch, old["cv"], invalid=False)
            reuse._check_status(torch, new["cv"], invalid=False)
            observed = reuse._snapshot(old["cv"])
            reuse._assert_same_bits(torch, observed, new["cv"],
                                    "B4 changed-Q graph replay")
            if not changed:
                reuse._assert_same_bits(torch, reference, old["cv"],
                                        "B4 initial graph replay")
            elif all(reuse._bitwise_identical(torch, reference[key], observed[key])
                     for key in ("partial", "lse")):
                raise BatchedHistoryProbeError("B4 graph replay ignored changed Q")
            for selected_ops, buffers in ((old_ops, old), (new_ops, new)):
                launch(selected_ops, buffers)
                reuse._assert_same_bits(torch, observed, buffers["cv"],
                                        "B4 changed graph versus eager")
            _stats(torch, new["cv"], prepared, spec, B4)
    finally:
        tensors["q"].copy_(original_q);tensors["qr"].copy_(original_qr)
        torch.npu.synchronize()
    result = {"case": spec.name, "status": "passed", "clusters":
              _stats(torch, new["cv"], prepared, spec, B4)["eligible_clusters"],
              "input_sha256": fixture["input_sha256"],
              "graph_capture": "passed", "graph_replay": "passed",
              "changed_Q_same_addresses": True, "faults": faults,
              "scope": "standalone_B4_operator_graph_not_full_service"}
    del tensors, old, new, fixture
    gc.collect();torch.npu.empty_cache()
    return result


def probe(config_path: Path, acceptance_path: Path) -> dict:
    target = json.loads(config_path.read_text())
    acceptance = json.loads(acceptance_path.read_text())
    policy = acceptance.get("performance", {})
    if (acceptance.get("frozen_before_measurement") is not True or
            (policy.get("warmup"), policy.get("repeats"), policy.get("statistic"),
             policy.get("max_latency_ratio")) != (WARMUP, REPEATS, "median", 1.0)):
        raise BatchedHistoryProbeError("frozen 2+5 median precision/speed policy required")
    from .probe_native_current_fia import _select_target_npu, _target_geometry
    _select_target_npu(target)
    if _target_geometry(target) != (6, 1, 256) or target.get("rotation_method") != "hadamard":
        raise BatchedHistoryProbeError("target Hq6/Hkv1/D256 or rotation changed")
    import torch
    import torch_npu  # noqa: F401 - actual target NPU only
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.npu.set_device(0)
    from .build_ops import normalize_soc
    if (not torch.npu.is_available() or
            normalize_soc(torch.npu.get_device_name(0)) != target["soc_version"]):
        raise BatchedHistoryProbeError("selected target NPU/SOC unavailable")
    from oscar_ascend.ops.loader import require_capabilities, validate_build_artifacts
    manifest_path = ROOT / "build/ascendc/build_manifest.json"
    manifest = validate_build_artifacts(manifest_path)
    require_capabilities({"prepare_attention_tasks_out", "rotate_out", FAST_C4,
                          FAST_C16, B4, "merge_lse_out", "status_guard"}, manifest_path)
    ops = torch.ops.oscar_ascend_ops
    cores = reuse._core_count(torch, target)
    if cores != 20:
        raise BatchedHistoryProbeError("target A2 geometry requires 20 Cube cores")
    device = torch.device("npu:0")
    rows = []
    for spec in case_plan():
        if spec.name == "mixed_history_30k":
            fixture = mixed.make_mixed_fixture(torch, spec)
        elif spec.name == "mixed_history_65k":
            fixture = opt._long_fixture(torch, spec)
        else:
            fixture = reuse.make_fixture(torch, spec)
        row = _run_case(torch, ops, fixture, device, cores, acceptance)
        rows.append(row)
        perf = row["performance"]
        print("[oscar] PERF_BATCHED_HISTORY " + json.dumps({
            "case": row["case"], "old_operator": row["old_operator"],
            "new_operator": B4, "old_cv_ms": perf["old"]["cv_ms"],
            "new_cv_ms": perf["new"]["cv_ms"],
            "old_total_ms": perf["old"]["total_ms"],
            "new_total_ms": perf["new"]["total_ms"],
            "cv_ratio": perf["ratios"]["cv_ms"],
            "total_ratio": perf["ratios"]["total_ms"],
            "precision": row["precision"], "status": row["status"]},
            sort_keys=True), flush=True)
        del fixture;gc.collect()
        if row["status"] != "passed":
            break
    graph = (_graph_case(torch, ops, device, cores, acceptance)
             if len(rows) == 3 and all(row["status"] == "passed" for row in rows)
             else None)
    passed = len(rows) == 3 and all(row["status"] == "passed" for row in rows) and \
        graph is not None and graph["graph_capture"] == graph["graph_replay"] == "passed"
    return {"status": "passed" if passed else "failed",
            "precision": "passed" if all(row["precision"] == "passed" for row in rows)
            else "failed", "graph_capture": "passed" if passed else "not_established",
            "graph_replay": "passed" if passed else "not_established",
            "performance": "passed" if passed else "failed",
            "artifact_signature": manifest["signature"],
            "artifact_sha256": manifest["sha256"],
            "device": str(device), "physical_devices": target["devices"],
            "cases": rows, "graph": graph,
            "full_model_quality": "not_run", "full_model_performance": "not_run",
            "first_error": next((f"{row['case']}: strict CV/total ratio>1"
                                 for row in rows if row["status"] != "passed"),
                                None if passed else "required graph or cases missing")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/target.json")
    parser.add_argument("--acceptance", type=Path, default=ROOT / "configs/acceptance.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.name in {"batched-history.json", "status.json"}:
        parser.error("#136/#151: report path must differ from phase ledger")
    try:
        report = probe(args.config, args.acceptance)
        atomic_json(args.output, report)
        print("[oscar] PERF_BATCHED_HISTORY_RESULT " + json.dumps({
            "status": report["status"], "precision": report["precision"],
            "graph_capture": report["graph_capture"],
            "graph_replay": report["graph_replay"],
            "performance": report["performance"],
            "cases": len(report["cases"]), "first_error": report["first_error"],
            "report": str(args.output)}, sort_keys=True), flush=True)
        return 0 if report["status"] == "passed" else 2
    except Exception as error:
        atomic_json(args.output, {"status": "failed", "precision": "not_established",
                                  "graph_capture": "not_established",
                                  "graph_replay": "not_established",
                                  "performance": "not_established",
                                  "first_error": str(error),
                                  "error_type": type(error).__name__})
        traceback.print_exc()
        print("[oscar] PERF_BATCHED_HISTORY_RESULT " + json.dumps({
            "status": "failed", "first_error": str(error),
            "report": str(args.output)}, sort_keys=True), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
