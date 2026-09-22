"""Auditable paired NPU evidence comparison; never manufactures measurements.

Archive #70/#71: compare real device phases/ranks; host launch time is not
device completion. #72: create output parents. #73: historical baseline
numbers cannot substitute for a matching current native run. Start §10:
freeze policy, compare every case and require correctness/route evidence.
Only Python stdlib is imported. Hashes prove file binding, not honesty of a
measurement producer; the original traces remain subject to human audit.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any


DEFAULT_POLICY = Path(__file__).resolve().parents[1] / "configs" / "acceptance.json"
DEFAULT_TARGET = Path(__file__).resolve().parents[1] / "configs" / "target.json"
CASE_FIELDS = ("input_tokens", "concurrency", "q_len")
LATENCY_METRICS = ("ttft", "tpot", "itl_p95", "e2e")
THROUGHPUT_METRICS = ("prompt", "generation")
SOFTWARE_FIELDS = ("vllm_commit", "vllm_ascend_commit", "torch", "torch_npu",
                   "cann", "driver", "firmware", "compiler")
NPU_CHECKS = ("build", "load", "device_completion", "graph_capture", "graph_replay", "tp4", "mtp")
ACCURACY_CHECKS = ("pack_unpack", "store_dequant", "fused_attention", "prefill_window", "model_quality")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _bad_constant(value: str):
    raise ValueError(f"nonstandard non-finite JSON number: {value}")


def read_json(path: str | Path) -> Any:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), parse_constant=_bad_constant,
                      object_pairs_hook=unique_pairs)


def _issue(items: list, code: str, detail: str, status: str = "failed") -> None:
    items.append({"code": code, "detail": detail, "status": status})


def _status(issues: list[dict]) -> str:
    return "failed" if any(item["status"] == "failed" for item in issues) else "not_run" if issues else "passed"


def _positive_number(value: Any, *, zero: bool = False) -> bool:
    return (type(value) in (int, float) and math.isfinite(value)
            and (value >= 0 if zero else value > 0))


def _case_key(case: dict) -> tuple[int, int, int]:
    if not isinstance(case, dict):
        raise ValueError("each case must be a mapping")
    key = tuple(case.get(field) for field in CASE_FIELDS)
    if any(type(value) is not int or value <= 0 for value in key):
        raise ValueError("case requires positive integer input_tokens/concurrency/q_len")
    return key


def required_cases(policy: dict) -> list[tuple[int, int, int]]:
    axes = [policy.get("required_input_lengths"), policy.get("required_concurrency"),
            policy.get("required_q_lens", [1, 2, 3, 4])]
    for axis in axes:
        if (not isinstance(axis, list) or not axis or any(type(v) is not int or v <= 0 for v in axis)
                or len(set(axis)) != len(axis)):
            raise ValueError("required case axes must be nonempty unique positive integers")
    return list(itertools.product(*axes))


def _policy_issues(policy: dict) -> list[dict]:
    if not isinstance(policy, dict) or not isinstance(policy.get("performance"), dict):
        raise ValueError("policy and policy.performance must be mappings")
    issues = []
    perf = policy.get("performance", {})
    if policy.get("frozen_before_measurement") is not True:
        _issue(issues, "policy_not_frozen", "acceptance policy must be frozen before collecting samples")
    for key, value in (("repeats", 5), ("warmup", 2), ("statistic", "median"),
                       ("per_case_required", True), ("noise_allowance", 0.0)):
        if perf.get(key) != value:
            _issue(issues, "unsupported_policy", f"performance.{key} must equal {value!r}")
    if (not _positive_number(perf.get("max_latency_ratio"))
            or not _positive_number(perf.get("min_throughput_ratio"))):
        _issue(issues, "invalid_threshold", "positive finite latency/throughput ratios are required")
    quality = policy.get("model_quality", {})
    if not isinstance(quality, dict):
        raise ValueError("policy.model_quality must be a mapping")
    if (quality.get("status") != "frozen_before_measurement"
            or any(not _positive_number(quality.get(k), zero=True)
                   for k in ("logits_tolerance", "task_metric_tolerance", "mtp_acceptance_tolerance"))):
        _issue(issues, "requires_premeasurement_definition",
               "model quality thresholds remain undefined; no overall acceptance is possible", "not_run")
    return issues


def _validate_pair(pair: Any, issues: list, target: dict | None = None) -> None:
    if not isinstance(pair, dict):
        _issue(issues, "missing_pair_identity", "pair identity must be a mapping")
        return
    for key in ("model_fingerprint", "dataset_sha256", "workload_sha256"):
        if not isinstance(pair.get(key), str) or not SHA256.fullmatch(pair[key]):
            _issue(issues, "invalid_fingerprint", f"pair.{key} must be a full lowercase SHA256")
    software = pair.get("software")
    if not isinstance(software, dict) or any(not isinstance(software.get(k), str) or not software[k]
                                             for k in SOFTWARE_FIELDS):
        _issue(issues, "software_incomplete", "all pinned software/driver/firmware/compiler fields are required")
    devices = pair.get("devices")
    if (not isinstance(devices, list) or len(devices) != 4
            or any(type(item) is not int or item < 0 for item in devices)
            or len(set(devices)) != 4 or pair.get("tp") != 4):
        _issue(issues, "target_hardware_mismatch", "record four unique nonnegative physical device IDs and TP=4")
    if target is not None and target.get("devices") is not None and devices != target["devices"]:
        _issue(issues, "selected_devices_mismatch", "paired devices differ from the explicitly selected target devices")
    if not isinstance(pair.get("npu_model"), str) or not pair["npu_model"]:
        _issue(issues, "missing_npu_model", "record the actual NPU model")
    mtp = pair.get("mtp", {})
    if not isinstance(mtp, dict) or mtp.get("method") != "qwen3_5_mtp" or mtp.get("num_speculative_tokens") != 3:
        _issue(issues, "mtp_mismatch", "target requires qwen3_5_mtp with 3 speculative tokens")
    if pair.get("graph") != "FULL_DECODE_ONLY" or pair.get("draft_graph_scope") != "eager":
        _issue(issues, "graph_mismatch", "record target FULL_DECODE_ONLY and native eager draft scope")
    workload = pair.get("workload")
    required = ("output_tokens", "sampling", "arrival", "prefix_cache", "max_model_len", "max_num_seqs", "async_scheduling")
    if not isinstance(workload, dict) or any(key not in workload for key in required):
        _issue(issues, "workload_incomplete", "record output, sampling, arrival, prefix state and engine limits")
    elif canonical_sha256(workload) != pair.get("workload_sha256"):
        _issue(issues, "workload_hash_mismatch", "workload fingerprint does not match its recorded content")


def _load_evidence(run: dict, case: dict, kind: str, root: Path, policy_hash: str,
                   issues: list) -> dict | None:
    label = f"{run['kind']}:{kind}"
    manifest = case.get("evidence")
    pointer = manifest.get(kind) if isinstance(manifest, dict) else None
    if not isinstance(pointer, dict) or not pointer.get("path") or not isinstance(pointer.get("sha256"), str):
        _issue(issues, "missing_evidence", f"{label}: path and SHA256 are required", "not_run")
        return None
    if not SHA256.fullmatch(pointer["sha256"]):
        _issue(issues, "invalid_evidence_hash", f"{label}: invalid SHA256")
        return None
    if not isinstance(pointer["path"], str):
        _issue(issues, "invalid_evidence_path", f"{label}: path must be a string")
        return None
    path = root / pointer["path"]
    try:
        raw = path.read_bytes()
    except OSError as error:
        _issue(issues, "missing_evidence_file", f"{label}: {error}", "not_run")
        return None
    if hashlib.sha256(raw).hexdigest() != pointer["sha256"]:
        _issue(issues, "evidence_hash_mismatch", f"{label}: file digest differs: {path}")
        return None
    try:
        evidence = read_json(path)
    except (OSError, ValueError) as error:
        _issue(issues, "invalid_evidence_json", f"{label}: {error}")
        return None
    if not isinstance(evidence, dict):
        _issue(issues, "invalid_evidence_schema", f"{label}: expected JSON mapping")
        return None
    bindings = {"schema_version": 1, "evidence_kind": kind, "run_id": run["run_id"],
                "measurement_type": "target_npu", "acceptance_sha256": policy_hash,
                "case": {key: case[key] for key in CASE_FIELDS},
                "pair_sha256": canonical_sha256(run["pair"])}
    for key, value in bindings.items():
        if evidence.get(key) != value:
            _issue(issues, "evidence_binding_mismatch", f"{label}: field {key} is not bound to this run")
    if evidence.get("status") != "passed":
        _issue(issues, "evidence_not_passed", f"{label}: status={evidence.get('status')!r}",
               "failed" if evidence.get("status") == "failed" else "not_run")
    return evidence


def _case_evidence(run: dict, case: dict, root: Path, policy_hash: str, issues: list) -> None:
    npu = _load_evidence(run, case, "npu", root, policy_hash, issues)
    if npu is not None and (not isinstance(npu.get("checks"), dict) or any(npu["checks"].get(k) != "passed" for k in NPU_CHECKS)):
        failed = isinstance(npu.get("checks"), dict) and "failed" in npu["checks"].values()
        _issue(issues, "npu_validation_incomplete", f"{run['kind']}: build/load/completion/graph/TP/MTP evidence required", "failed" if failed else "not_run")
    accuracy = _load_evidence(run, case, "accuracy", root, policy_hash, issues)
    required = ACCURACY_CHECKS if run["kind"] == "oscar" else ("native_output",)
    if accuracy is not None and (not isinstance(accuracy.get("checks"), dict) or any(accuracy["checks"].get(k) != "passed" for k in required)):
        failed = isinstance(accuracy.get("checks"), dict) and "failed" in accuracy["checks"].values()
        _issue(issues, "accuracy_incomplete", f"{run['kind']}: missing passed accuracy checks {required}", "failed" if failed else "not_run")
    route = _load_evidence(run, case, "route", root, policy_hash, issues)
    if route is not None:
        if route.get("backend") != run["kind"]:
            _issue(issues, "wrong_route", f"{run['kind']}: backend={route.get('backend')!r}")
        if route.get("ranks") != [0, 1, 2, 3] or route.get("all_full_layers") is not True:
            _issue(issues, "route_coverage_incomplete", f"{run['kind']}: all ranks and FULL layers must be evidenced", "not_run")
        if route.get("gdn_native") is not True:
            _issue(issues, "gdn_route_violation", f"{run['kind']}: GDN native route not established")
        if run["kind"] == "native" and route.get("oscar_plugin_disabled") is not True:
            _issue(issues, "native_is_not_baseline", "native run must prove OSCAR plugin disabled")
    if run["kind"] == "oscar":
        memory = _load_evidence(run, case, "memory", root, policy_hash, issues)
        if memory is not None:
            for field in ("full_history_bf16_restore_bytes", "full_history_bf16_shadow_bytes"):
                value = memory.get(field)
                if type(value) is not int or value != 0:
                    _issue(issues, "full_history_bf16_violation", f"{field} must be measured as integer zero; got {value!r}")


def _samples(case: dict, label: str, issues: list) -> list | None:
    samples, warmups = case.get("samples"), case.get("warmup")
    if not isinstance(warmups, list) or len(warmups) != 2 or any(w.get("status") != "completed" for w in warmups if isinstance(w, dict)) or any(not isinstance(w, dict) for w in warmups):
        _issue(issues, "warmup_incomplete", f"{label}: exactly two completed warmup records required", "not_run")
    if not isinstance(samples, list) or len(samples) != 5:
        _issue(issues, "repeats_incomplete", f"{label}: exactly five measurement records required", "not_run")
        return None
    if any(not isinstance(s, dict) for s in samples):
        _issue(issues, "invalid_sample", f"{label}: samples must be mappings")
        return None
    if [s.get("repeat") for s in samples] != list(range(5)):
        _issue(issues, "repeat_identity", f"{label}: repeat ids must be [0,1,2,3,4]")
    for sample in samples:
        if (any(type(sample.get(field)) is not int for field in ("completed_requests", "failed_requests", "timeouts"))
                or sample.get("completed_requests") != case["concurrency"]
                or sample.get("failed_requests") != 0 or sample.get("timeouts") != 0):
            _issue(issues, "request_failures", f"{label}: incomplete requests, failures or timeouts")
        for group, names in (("latency_ms", LATENCY_METRICS), ("throughput_tps", THROUGHPUT_METRICS)):
            metrics = sample.get(group)
            if not isinstance(metrics, dict) or any(not _positive_number(metrics.get(name)) for name in names):
                _issue(issues, "invalid_metrics", f"{label}: every {group} metric must be finite and positive")
        phases = sample.get("phase_ms")
        if not isinstance(phases, dict) or not phases:
            _issue(issues, "missing_phases", f"{label}: device phase timings required", "not_run")
        elif any(not isinstance(values, list) or len(values) != 4
                 or any(not _positive_number(value, zero=True) for value in values) for values in phases.values()):
            _issue(issues, "invalid_phase_metrics", f"{label}: each device phase needs four nonnegative finite rank times")
    return samples


def _metric(native: list[float], oscar: list[float], limit: float, higher: bool = False) -> dict:
    baseline, candidate = statistics.median(native), statistics.median(oscar)
    ratio = candidate / baseline if baseline != 0 else (1.0 if candidate == 0 else None)
    if ratio is not None and not math.isfinite(ratio):
        ratio = None
    passed = ratio is not None and (ratio >= limit if higher else ratio <= limit)
    def summary(values):
        ordered = sorted(values)
        return {"median": statistics.median(values), "p95_nearest_rank": ordered[math.ceil(.95 * len(ordered)) - 1],
                "min": min(values), "max": max(values), "samples": values}
    return {"status": "passed" if passed else "failed", "native": summary(native), "oscar": summary(oscar),
            "ratio": ratio, "limit": limit, "direction": "at_least" if higher else "at_most"}


def compare_runs(native: dict, oscar: dict, policy: dict, *, native_root: str | Path = ".",
                 oscar_root: str | Path = ".", target: dict | None = None) -> dict:
    """Compare complete JSON reports and their local evidence artifacts."""
    if target is not None and not isinstance(target, dict):
        raise ValueError("target policy must be a mapping when supplied")
    issues = _policy_issues(policy)
    policy_hash = canonical_sha256(policy)
    required = required_cases(policy)
    for kind, run in (("native", native), ("oscar", oscar)):
        if not isinstance(run, dict):
            raise ValueError(f"{kind} report must be a mapping")
        if run.get("schema_version") != 1 or run.get("kind") != kind or not isinstance(run.get("run_id"), str) or not run["run_id"]:
            _issue(issues, "invalid_run_schema", f"{kind}: schema_version/kind/run_id missing or incorrect")
        if run.get("measurement_type") != "target_npu":
            _issue(issues, "npu_run_missing", f"{kind}: measurement_type is not target_npu", "not_run")
        if run.get("acceptance_sha256") != policy_hash:
            _issue(issues, "policy_hash_mismatch", f"{kind}: recorded policy fingerprint does not match frozen config")
        _validate_pair(run.get("pair"), issues, target)
    if native.get("pair") != oscar.get("pair"):
        _issue(issues, "pair_mismatch", "model/software/devices/TP/MTP/graph/workload/dataset identity differs")
    case_maps = []
    for kind, run in (("native", native), ("oscar", oscar)):
        mapping = {}
        if not isinstance(run.get("cases"), list):
            _issue(issues, "cases_missing", f"{kind}: cases array missing", "not_run")
        else:
            for case in run["cases"]:
                key = _case_key(case)
                if key in mapping:
                    _issue(issues, "duplicate_case", f"{kind}: {key}")
                mapping[key] = case
        case_maps.append(mapping)
    results = []
    structural = issues
    for key in sorted(set(required) | set(case_maps[0]) | set(case_maps[1])):
        local = []
        result = {**dict(zip(CASE_FIELDS, key)), "issues": local, "metrics": {}, "phase_ms": {}}
        results.append(result)
        left, right = case_maps[0].get(key), case_maps[1].get(key)
        if left is None or right is None:
            _issue(local, "missing_case", f"native={left is not None}, oscar={right is not None}", "not_run")
        elif structural:
            _issue(local, "run_validation_failed", "resolve report-level provenance errors before timing comparison", _status(structural))
        else:
            if not isinstance(left.get("input_sha256"), str) or not SHA256.fullmatch(left["input_sha256"]) or left.get("input_sha256") != right.get("input_sha256"):
                _issue(local, "case_input_mismatch", "actual tokenized input SHA256 must match")
            _case_evidence(native, left, Path(native_root), policy_hash, local)
            _case_evidence(oscar, right, Path(oscar_root), policy_hash, local)
            ns = _samples(left, "native", local)
            os = _samples(right, "oscar", local)
            if not local:
                perf = policy["performance"]
                for group, names, higher in (("latency_ms", LATENCY_METRICS, False), ("throughput_tps", THROUGHPUT_METRICS, True)):
                    result["metrics"][group] = {}
                    for name in names:
                        metric = _metric([s[group][name] for s in ns], [s[group][name] for s in os],
                                         perf["min_throughput_ratio" if higher else "max_latency_ratio"], higher)
                        result["metrics"][group][name] = metric
                        if metric["status"] == "failed":
                            _issue(local, "performance_regression", f"{group}.{name} ratio={metric['ratio']}")
                phases = set().union(*(set(s["phase_ms"]) for s in ns + os))
                for phase in sorted(phases):
                    native_def = native.get("phase_definitions")
                    oscar_def = oscar.get("phase_definitions")
                    nd = native_def.get(phase) if isinstance(native_def, dict) else None
                    od = oscar_def.get(phase) if isinstance(oscar_def, dict) else None
                    if (any(phase not in s["phase_ms"] for s in ns + os) or not isinstance(nd, dict) or not nd or nd != od
                            or nd.get("clock") != "device" or not isinstance(nd.get("scope"), str) or not nd["scope"]
                            or not isinstance(nd.get("definition_sha256"), str)
                            or not SHA256.fullmatch(nd["definition_sha256"])):
                        result["phase_ms"][phase] = {"status": "needs_baseline"}
                        _issue(local, "needs_baseline", f"phase {phase}: equivalent device scope/baseline missing", "not_run")
                        continue
                    ranks = [_metric([s["phase_ms"][phase][rank] for s in ns],
                                     [s["phase_ms"][phase][rank] for s in os], perf["max_latency_ratio"])
                             for rank in range(4)]
                    result["phase_ms"][phase] = {"status": "passed" if all(r["status"] == "passed" for r in ranks) else "failed", "ranks": ranks}
                    if any(rank["status"] == "failed" for rank in ranks):
                        _issue(local, "phase_regression", f"phase {phase}: at least one TP rank regressed")
        result["status"] = _status(local)
    all_issues = issues + [issue for case in results for issue in case["issues"]]
    return {"schema_version": 1, "status": _status(all_issues), "scope": "paired target-NPU evidence audit",
            "acceptance_sha256": policy_hash, "native_run_id": native.get("run_id"), "oscar_run_id": oscar.get("run_id"),
            "target_sha256": canonical_sha256(target) if target is not None else None,
            "native_report_sha256": canonical_sha256(native), "oscar_report_sha256": canonical_sha256(oscar),
            "issues": issues, "required_cases": len(required), "cases": results,
            "evidence_limit": "File hashes establish binding/integrity, not independent authenticity of producer claims."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--oscar", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = compare_runs(read_json(args.native), read_json(args.oscar), read_json(args.acceptance),
                              native_root=args.native.parent, oscar_root=args.oscar.parent, target=read_json(args.target))
    except (OSError, ValueError, TypeError, KeyError) as error:
        report = {"schema_version": 1, "status": "failed", "issues": [{"code": "invalid_input", "detail": str(error)}]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"{report['status']}: {args.output}")
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
