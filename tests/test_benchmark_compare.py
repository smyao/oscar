"""Synthetic temporary fixtures test rejection logic, never NPU measurements.

Archive #70/#71: do not hide slow phases/ranks with aggregate throughput;
#72: output parents; #73 and start §10: paired provenance and required cases.
"""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.compare import (
    ACCURACY_CHECKS, DEFAULT_POLICY, LATENCY_METRICS, NPU_CHECKS, THROUGHPUT_METRICS,
    canonical_sha256, compare_runs, main, read_json, required_cases,
)


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def bundle(tmp_path):
    """All files are test fixtures; no measured-results directory is written."""
    policy = read_json(DEFAULT_POLICY)
    policy.update(required_input_lengths=[16384, 32768], required_concurrency=[1, 4], required_q_lens=[1, 4])
    policy["model_quality"] = {"status": "frozen_before_measurement", "logits_tolerance": .01,
                               "task_metric_tolerance": .01, "mtp_acceptance_tolerance": .01}
    workload = {"output_tokens": 32, "sampling": {"temperature": 0}, "arrival": "closed_loop",
                "prefix_cache": "cold", "max_model_len": 262144, "max_num_seqs": 128, "async_scheduling": True}
    pair = {"model_fingerprint": _sha("synthetic model"), "dataset_sha256": _sha("synthetic data"),
            "workload_sha256": canonical_sha256(workload), "workload": workload,
            "software": {k: "synthetic" for k in ("vllm_commit", "vllm_ascend_commit", "torch", "torch_npu", "cann", "driver", "firmware", "compiler")},
            "devices": [4, 5, 6, 7], "npu_model": "synthetic Ascend910B4", "tp": 4,
            "mtp": {"method": "qwen3_5_mtp", "num_speculative_tokens": 3}, "graph": "FULL_DECODE_ONLY", "draft_graph_scope": "eager"}
    runs = []
    for kind in ("native", "oscar"):
        run = {"schema_version": 1, "kind": kind, "run_id": f"synthetic-unit-test-{kind}",
               "measurement_type": "target_npu", "acceptance_sha256": canonical_sha256(policy), "pair": copy.deepcopy(pair),
               "phase_definitions": {"attention": {"clock": "device", "scope": "same whole FULL attention operation",
                                                      "definition_sha256": _sha("fixture phase definition")}}, "cases": []}
        for i, key in enumerate(required_cases(policy)):
            case = dict(zip(("input_tokens", "concurrency", "q_len"), key))
            case.update(input_sha256=_sha(str(key)), warmup=[{"status": "completed"} for _ in range(2)],
                        samples=[{"repeat": r, "completed_requests": key[1], "failed_requests": 0, "timeouts": 0,
                                  "latency_ms": {name: 10.0 if kind == "native" else 9.0 for name in LATENCY_METRICS},
                                  "throughput_tps": {name: 100.0 if kind == "native" else 110.0 for name in THROUGHPUT_METRICS},
                                  "phase_ms": {"attention": [2.0 if kind == "native" else 1.9] * 4}} for r in range(5)],
                        evidence={})
            for evidence_kind in (("npu", "accuracy", "route", "memory") if kind == "oscar" else ("npu", "accuracy", "route")):
                evidence = {"schema_version": 1, "evidence_kind": evidence_kind, "run_id": run["run_id"],
                            "measurement_type": "target_npu", "acceptance_sha256": run["acceptance_sha256"],
                            "case": dict(zip(("input_tokens", "concurrency", "q_len"), key)),
                            "pair_sha256": canonical_sha256(pair), "status": "passed"}
                if evidence_kind == "npu":
                    evidence["checks"] = dict.fromkeys(NPU_CHECKS, "passed")
                elif evidence_kind == "accuracy":
                    evidence["checks"] = dict.fromkeys(ACCURACY_CHECKS if kind == "oscar" else ("native_output",), "passed")
                elif evidence_kind == "route":
                    evidence.update(backend=kind, ranks=[0, 1, 2, 3], all_full_layers=True,
                                    gdn_native=True, oscar_plugin_disabled=kind == "native")
                elif evidence_kind == "memory":
                    evidence.update(full_history_bf16_restore_bytes=0, full_history_bf16_shadow_bytes=0)
                path = tmp_path / f"fixture-{kind}-{i}-{evidence_kind}.json"
                path.write_text(json.dumps(evidence))
                case["evidence"][evidence_kind] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            run["cases"].append(case)
        runs.append(run)
    return {"native": runs[0], "oscar": runs[1], "policy": policy, "root": tmp_path}


def evaluate(bundle):
    return compare_runs(bundle["native"], bundle["oscar"], bundle["policy"],
                        native_root=bundle["root"], oscar_root=bundle["root"])


def change_evidence(bundle, kind, mutation, *, rehash=True):
    pointer = bundle["oscar"]["cases"][0]["evidence"][kind]
    path = bundle["root"] / pointer["path"]
    content = read_json(path)
    mutation(content)
    path.write_text(json.dumps(content))
    if rehash:
        pointer["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def test_complete_synthetic_pair_exercises_pass_and_per_rank_statistics(bundle):
    result = evaluate(bundle)
    assert result["status"] == "passed"
    assert result["required_cases"] == 8
    assert all(case["status"] == "passed" for case in result["cases"])
    metric = result["cases"][0]["metrics"]["latency_ms"]["ttft"]
    assert metric["ratio"] == .9
    assert metric["native"]["p95_nearest_rank"] == 10
    assert len(result["cases"][0]["phase_ms"]["attention"]["ranks"]) == 4


@pytest.mark.parametrize("field", ["model_fingerprint", "software", "devices", "tp", "mtp", "graph", "workload", "dataset_sha256"])
def test_unpaired_runs_cannot_pass(bundle, field):
    bundle["oscar"]["pair"][field] = "different"
    result = evaluate(bundle)
    assert result["status"] == "failed"
    assert any(issue["code"] == "pair_mismatch" for issue in result["issues"])


def test_missing_case_is_not_run_even_when_all_other_cases_are_faster(bundle):
    bundle["oscar"]["cases"].pop()
    result = evaluate(bundle)
    assert result["status"] == "not_run"
    assert any(case["status"] == "not_run" for case in result["cases"])


def test_single_slow_case_cannot_be_hidden_by_global_average(bundle):
    for sample in bundle["oscar"]["cases"][0]["samples"]:
        sample["latency_ms"]["ttft"] = 11
    result = evaluate(bundle)
    assert result["status"] == "failed"
    assert result["cases"][0]["status"] == "failed"
    assert all(case["status"] == "passed" for case in result["cases"][1:])


def test_slow_rank_cannot_be_hidden_by_other_ranks(bundle):
    for sample in bundle["oscar"]["cases"][0]["samples"]:
        sample["phase_ms"]["attention"][2] = 6500
    result = evaluate(bundle)
    assert result["status"] == "failed"
    assert result["cases"][0]["phase_ms"]["attention"]["ranks"][2]["ratio"] == 3250


def test_extreme_finite_ratio_fails_without_nonstandard_json(bundle):
    for sample in bundle["native"]["cases"][0]["samples"]:
        sample["latency_ms"]["ttft"] = 1e-308
    for sample in bundle["oscar"]["cases"][0]["samples"]:
        sample["latency_ms"]["ttft"] = 1e308
    result = evaluate(bundle)
    assert result["status"] == "failed"
    assert result["cases"][0]["metrics"]["latency_ms"]["ttft"]["ratio"] is None
    json.dumps(result, allow_nan=False)


def test_new_phase_without_native_equivalent_requires_baseline(bundle):
    for sample in bundle["oscar"]["cases"][0]["samples"]:
        sample["phase_ms"]["dequant"] = [0.5] * 4
    result = evaluate(bundle)
    assert result["status"] == "not_run"
    assert result["cases"][0]["phase_ms"]["dequant"]["status"] == "needs_baseline"


@pytest.mark.parametrize("kind", ["npu", "accuracy", "route", "memory"])
def test_missing_evidence_never_passes(bundle, kind):
    del bundle["oscar"]["cases"][0]["evidence"][kind]
    result = evaluate(bundle)
    assert result["status"] == "not_run"
    assert result["cases"][0]["metrics"] == {}


def test_accuracy_failure_prevents_any_timing_comparison(bundle):
    change_evidence(bundle, "accuracy", lambda ev: ev.update(status="failed"))
    result = evaluate(bundle)
    assert result["status"] == "failed"
    assert result["cases"][0]["metrics"] == {}


def test_native_route_falsely_labeled_oscar_fails(bundle):
    change_evidence(bundle, "route", lambda ev: ev.update(backend="native"))
    assert evaluate(bundle)["status"] == "failed"


@pytest.mark.parametrize("field", ["full_history_bf16_restore_bytes", "full_history_bf16_shadow_bytes"])
def test_full_history_restoration_or_shadow_is_a_failure(bundle, field):
    change_evidence(bundle, "memory", lambda ev: ev.update({field: 1}))
    assert evaluate(bundle)["status"] == "failed"


def test_evidence_file_hash_is_actually_verified(bundle):
    change_evidence(bundle, "memory", lambda ev: ev.update(full_history_bf16_restore_bytes=1), rehash=False)
    result = evaluate(bundle)
    assert result["status"] == "failed"
    assert any(i["code"] == "evidence_hash_mismatch" for i in result["cases"][0]["issues"])


def test_evidence_from_another_case_cannot_be_reused(bundle):
    change_evidence(bundle, "npu", lambda ev: ev.update(case={"input_tokens": 1, "concurrency": 1, "q_len": 1}))
    assert evaluate(bundle)["status"] == "failed"


@pytest.mark.parametrize("mutation", [
    lambda case: case["samples"].pop(),
    lambda case: case["warmup"].pop(),
])
def test_missing_repeat_or_warmup_is_not_run(bundle, mutation):
    mutation(bundle["oscar"]["cases"][0])
    assert evaluate(bundle)["status"] == "not_run"


def test_unfrozen_threshold_cannot_pass(bundle):
    bundle["policy"]["performance"]["max_latency_ratio"] = 100
    result = evaluate(bundle)
    assert result["status"] == "failed"
    assert any(i["code"] == "policy_hash_mismatch" for i in result["issues"])


def test_cpu_measurements_cannot_pass_npu_gate(bundle):
    bundle["oscar"]["measurement_type"] = "cpu_test"
    assert evaluate(bundle)["status"] == "not_run"


def test_current_quality_policy_is_explicitly_not_ready(bundle):
    from benchmarks.compare import _policy_issues
    issues = _policy_issues(read_json(DEFAULT_POLICY))
    assert any(i["code"] == "requires_premeasurement_definition" and i["status"] == "not_run" for i in issues)


def test_device_ownership_is_not_assumed_from_historical_numbers():
    from benchmarks.compare import _validate_pair
    # Device checks alone must accept another explicitly recorded TP4 selection.
    issues = []
    _validate_pair({"devices": [0, 1, 2, 3], "tp": 4}, issues, target={"devices": None})
    assert not any(i["code"] in ("target_hardware_mismatch", "selected_devices_mismatch") for i in issues)
    issues = []
    _validate_pair({"devices": [0, 1, 2, 3], "tp": 4}, issues, target={"devices": [4, 5, 6, 7]})
    assert any(i["code"] == "selected_devices_mismatch" for i in issues)


@pytest.mark.parametrize("devices", [[1, 1, 2, 3], [0, 1, 2], [-1, 0, 1, 2], [True, 1, 2, 3]])
def test_invalid_physical_device_lists_are_rejected(devices):
    from benchmarks.compare import _validate_pair
    issues = []
    _validate_pair({"devices": devices, "tp": 4}, issues)
    assert any(i["code"] == "target_hardware_mismatch" for i in issues)


def test_invalid_metrics_do_not_become_ratios(bundle):
    bundle["oscar"]["cases"][0]["samples"][0]["latency_ms"]["e2e"] = float("nan")
    with pytest.raises(ValueError):
        evaluate(bundle)  # canonical output fingerprints also reject non-finite JSON.


def test_cli_creates_output_parent_and_returns_nonzero_for_missing_reports(tmp_path):
    destination = tmp_path / "nested" / "comparison.json"
    code = main(["--native", str(tmp_path / "missing1.json"), "--oscar", str(tmp_path / "missing2.json"), "--output", str(destination)])
    assert code == 2
    assert read_json(destination)["status"] == "failed"


def test_cli_rejects_duplicate_json_keys(tmp_path):
    report = tmp_path / "bad.json"
    report.write_text('{"schema_version": 1, "schema_version": 2}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        read_json(report)
