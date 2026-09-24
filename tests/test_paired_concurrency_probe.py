"""Archive #70-#73/#85/#94/#95/#125/#126/#129-#140: signed build and NPU gate before pairing."""

import json
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from benchmarks.compare import canonical_sha256
from tools import paired_concurrency_probe as paired


def _report(variant, *, latency=1.0, throughput=1.0, tpot=10.0):
    requests = []
    for index, length in enumerate((20000, 23000, 27000, 30000)):
        requests.append({"request_id": f"synthetic-{index}-{length}",
                         "status": "completed", "prompt_sha256": f"prompt-{index}",
                         "prompt_tokens": length, "ttft_ms": 100.0 * latency,
                         "tpot_ms": tpot * latency, "e2e_ms": 200.0 * latency,
                         "cache_salt_sha256": canonical_sha256(f"salt-{index}")})
    sample = {"repeat": 0, "status": "completed", "completed_requests": 4,
              "failed_requests": 0, "timeouts": 0, "requests": requests,
              "throughput_tps": {"prompt": 5000.0 * throughput,
                                 "generation": 50.0 * throughput},
              "latency_ms": {name: {"p50": value * latency, "p95": value * latency}
                             for name, value in (("ttft_ms", 100), ("tpot_ms", tpot),
                                                 ("e2e_ms", 200))}}
    manifest = [{"id": row["request_id"], "prompt_tokens": row["prompt_tokens"],
                 "prompt_sha256": row["prompt_sha256"]} for row in requests]
    identity = {"target_config_sha256": "same-config", "model_fingerprint": "same-model",
                "manifest": manifest, "output_tokens": 64,
                "arrival": "simultaneous_barrier", "cache_salt": "distinct deterministic per request"}
    return {"mode": "synthetic_mixed", "variant": variant, "status": "measured",
            "synthetic_mixed": {"status": "measured", "pair_sha256": canonical_sha256(identity),
                                "pair_identity": identity,
                                "prompt_lengths": [20000, 23000, 27000, 30000],
                                "prompt_manifest": manifest, "sample": sample}}


def _acceptance():
    return {"frozen_before_measurement": True,
            "performance": {"max_latency_ratio": 1.0,
                            "min_throughput_ratio": 1.0}}


def test_native_current_gate_requires_three_source_oracle_and_release(tmp_path, monkeypatch):
    report = {"status": "current_partial_probe_passed", "small_cases": [{"oracle": "passed"}],
              "long_case": {"sampled_oracle": "passed", "device_event_median_ms": 10},
              "mixed_history_window_current_merge": {"oracle": "passed",
                "history_window": "production_NPU_attention_cv_out",
                "prepare": "production_NPU_prepare_attention_tasks_out",
                "source2": "production_NPU_suppress_current_source_tasks"},
              "production_task_contract": {"source2_range_rewrite": "passed", "slot_guard": "passed",
                                           "metadata_error_preserved": True, "padding_excluded": True}}
    released = {"status": "passed"}

    def phase(_name, command, **kwargs):
        destination = command[command.index("--output") + 1]
        Path(destination).write_text(json.dumps(report))
        return SimpleNamespace(returncode=0, cleanup_complete=True, log="current-fia.log")

    from pathlib import Path
    monkeypatch.setattr(paired, "run_phase", phase)
    monkeypatch.setattr(paired, "_observe_resources",
                        lambda *_args, before=None: released if before is not None else {"snapshot": True})
    config = {"devices": [0, 1, 2, 3]}
    args = (tmp_path / "target.json", config, tmp_path / "acceptance.json", tmp_path)
    assert paired.ensure_native_current_attention(*args)["status"] == "passed"
    report["mixed_history_window_current_merge"]["oracle"] = "not_run"
    with pytest.raises(paired.OperatorGateError, match="report invalid"):
        paired.ensure_native_current_attention(*args)
    report["mixed_history_window_current_merge"]["oracle"] = "passed"
    report["production_task_contract"]["metadata_error_preserved"] = False
    with pytest.raises(paired.OperatorGateError, match="report invalid"):
        paired.ensure_native_current_attention(*args)
    report["production_task_contract"]["metadata_error_preserved"] = True
    released["status"] = "failed"
    with pytest.raises(paired.OperatorGateError, match="oracle/release failed"):
        paired.ensure_native_current_attention(*args)


def test_paired_comparison_requires_every_ratio_and_exact_workload():
    native = _report("native")
    oscar = _report("oscar", latency=.8, throughput=1.2)
    result = paired.compare_synthetic_reports(native, oscar, _acceptance())
    assert result["status"] == "passed"
    assert result["performance_acceptance"] == "not_run"
    assert len(result["batches"][0]["requests"]) == 4
    oscar["synthetic_mixed"]["sample"]["requests"][2]["ttft_ms"] = 101.0
    result = paired.compare_synthetic_reports(native, oscar, _acceptance())
    assert result["status"] == "failed"
    assert any("ttft_ms" in issue for issue in result["issues"])
    oscar["synthetic_mixed"]["prompt_manifest"][0]["prompt_sha256"] = "different"
    assert paired.compare_synthetic_reports(native, oscar, _acceptance())["status"] == "failed"


def test_zero_tpot_fails_closed_and_full_service_warmup_is_disclosed():
    native = _report("native", tpot=0.0)
    oscar = _report("oscar", tpot=0.0)
    full_service = {"status": "passed", "server": {"variant": "oscar"},
                    "performance": oscar["synthetic_mixed"]}
    result = paired.compare_synthetic_reports(native, full_service, _acceptance())
    assert result["warmup_pairing"] == "unpaired"
    assert result["performance_acceptance"] == "not_run"
    assert result["status"] == "needs_evidence"
    assert any("tpot_unresolved_SSE_burst" in issue for issue in result["issues"])


def test_native_failure_stops_before_oscar_and_preserves_child_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(paired, "ensure_native_current_attention", lambda *_args: {"status": "passed"})
    config_path = tmp_path / "target.json"
    acceptance_path = tmp_path / "acceptance.json"
    config_path.write_text(json.dumps({"devices": [0, 1, 2, 3]}))
    acceptance_path.write_text(json.dumps(_acceptance()))
    variants = []

    def variant(name, *_args):
        variants.append(name)
        return {"status": "failed", "returncode": 7, "probe_error": "device failure",
                "log": str(tmp_path / "native.log"), "report": str(tmp_path / "native.json"),
                "owned_server_cleanup_complete": True, "runner_cleanup_complete": True}

    monkeypatch.setattr(paired, "_run_variant", variant)
    monkeypatch.setattr(paired, "ensure_current_operators",
                        lambda *_args, **_kwargs: {"status": "passed", "build": "reused",
                                        "accuracy": "reused_prior_evidence"})
    monkeypatch.setattr(paired, "_observe_resources",
                        lambda *_args, before=None: ({"status": "passed"} if before else
                                                     {"devices": [0, 1, 2, 3], "memory": []}))
    report = paired.run_paired(config_path, output=tmp_path / "paired.json",
                               log_dir=tmp_path / "logs", acceptance_path=acceptance_path)
    assert variants == ["native"]
    assert report["status"] == "failed" and report["exit_code"] == 7
    assert report["oscar"] == "not_run"
    assert json.loads((tmp_path / "paired.json").read_text())["exit_code"] == 7


def test_native_only_requires_release_before_reporting_success(tmp_path, monkeypatch):
    monkeypatch.setattr(paired, "ensure_native_current_attention", lambda *_args: {"status": "passed"})
    config_path = tmp_path / "target.json"
    acceptance_path = tmp_path / "acceptance.json"
    config_path.write_text(json.dumps({"devices": [0, 1, 2, 3]}))
    acceptance_path.write_text(json.dumps(_acceptance()))
    monkeypatch.setattr(paired, "_run_variant", lambda name, *_args: {
        "status": "passed", "returncode": 0, "probe_error": None,
        "log": str(tmp_path / "native.log"), "report": str(tmp_path / "native.json"),
        "owned_server_cleanup_complete": True, "runner_cleanup_complete": True})
    monkeypatch.setattr(paired, "ensure_current_operators",
                        lambda *_args, **_kwargs: {"status": "passed", "build": "reused",
                                        "accuracy": "reused_prior_evidence"})
    monkeypatch.setattr(paired, "_observe_resources",
                        lambda *_args, before=None: ({"status": "failed", "reason": "memory held"}
                                                     if before else {"devices": [0, 1, 2, 3], "memory": []}))
    report = paired.run_paired(config_path, output=tmp_path / "paired.json",
                               log_dir=tmp_path / "logs", acceptance_path=acceptance_path,
                               native_only=True)
    assert report["status"] == "failed"
    assert report["native"]["resource_release"] == "failed"
    assert report["oscar"] == "not_run"


def test_one_call_runs_native_then_oscar_and_writes_paired_ratios(tmp_path, monkeypatch):
    monkeypatch.setattr(paired, "ensure_native_current_attention", lambda *_args: {"status": "passed"})
    config_path = tmp_path / "target.json"
    acceptance_path = tmp_path / "acceptance.json"
    config_path.write_text(json.dumps({"devices": [0, 1, 2, 3]}))
    acceptance_path.write_text(json.dumps(_acceptance()))
    order = []

    def variant(name, _config_path, _config, directory, policy):
        assert policy == acceptance_path.resolve()
        order.append(name)
        directory.mkdir(parents=True)
        path = directory / "report.json"
        path.write_text(json.dumps(_report(name, latency=.8 if name == "oscar" else 1,
                                           throughput=1.2 if name == "oscar" else 1)))
        return {"status": "passed", "returncode": 0, "probe_error": None,
                "log": str(directory / "console.log"), "report": str(path),
                "owned_server_cleanup_complete": True, "runner_cleanup_complete": True}

    monkeypatch.setattr(paired, "_run_variant", variant)
    monkeypatch.setattr(paired, "ensure_current_operators",
                        lambda *_args, **_kwargs: {"status": "passed", "build": "reused",
                                        "accuracy": "reused_prior_evidence"})
    monkeypatch.setattr(paired, "_observe_resources",
                        lambda *_args, before=None: ({"status": "passed"} if before else
                                                     {"devices": [0, 1, 2, 3], "memory": []}))
    report = paired.run_paired(config_path, output=tmp_path / "paired.json",
                               log_dir=tmp_path / "logs", acceptance_path=acceptance_path)
    assert order == ["native", "oscar"]
    assert report["status"] == "passed" and report["exit_code"] == 0
    assert report["comparison"]["status"] == "passed"
    assert report["comparison"]["warmup_pairing"] == "fresh_service_before_batch"
    assert report["comparison"]["performance_acceptance"] == "not_run"
    assert (tmp_path / "logs/comparison.json").exists()


def _write_real_npu_junit(path, *, skip_cv=False):
    suite = ET.Element("testsuite")
    for index in range(paired.CV_NPU_MIN_CASES):
        case = ET.SubElement(suite, "testcase", name=f"test_npu_cv_oracle[{index}]")
        if skip_cv and index == 0:
            ET.SubElement(case, "skipped")
    for logical in range(4):
        for index in range(26):
            case = ET.SubElement(suite, "testcase",
                                 name=f"test_rotation_ascendc_real_npu[{logical}-{index}]")
            props = ET.SubElement(case, "properties")
            ET.SubElement(props, "property", name="physical_device", value=str(logical))
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(path)


def test_fast_probe_rebuilds_drift_runs_npu_gate_and_reuses_exact_prior_evidence(tmp_path, monkeypatch):
    # #85/#126: the real build owns source/CANN/SOC drift detection. Here its
    # returned manifest changes only when the fake csrc tree changes, while the
    # paired runner must execute and inspect the frozen NPU gate before service.
    from tools import build_ops
    monkeypatch.setattr(paired, "ROOT", tmp_path)
    monkeypatch.setattr(build_ops, "reusable_build", lambda _path, _signature:
                        json.loads((tmp_path / "reports/build.json").read_text()))
    for name in ("csrc/attention_cv.cpp", "oscar_ascend/runtime.py",
                 "tests/test_cv_contracts.py", "tests/test_rotation_npu.py",
                 "tools/generate_rotation_cpu_cases.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("initial")
    config_path = tmp_path / "configs/target.json"
    config_path.parent.mkdir()
    config = {"devices": [0, 1, 2, 3], "soc_version": "ascend910b4",
              "phase_timeout_seconds": 30, "shutdown_timeout_seconds": 1}
    config_path.write_text(json.dumps(config))
    acceptance = _acceptance()
    phases, resource_calls = [], []
    signatures = {}

    def phase(name, command, *, log_dir, env, **_kwargs):
        phases.append(name)
        assert env["ASCEND_RT_VISIBLE_DEVICES"] == "0,1,2,3"
        assert env["OSCAR_RUN_NPU_TESTS"] == "1"
        assert env["OSCAR_TARGET_CONFIG"] == str(config_path.resolve())
        if name == "operator-build":
            source = paired.file_fingerprint(tmp_path / "csrc")
            source_key = paired.canonical_sha256(source)
            reused = source_key in signatures
            if source_key not in signatures:
                signatures[source_key] = chr(ord("a") + len(signatures)) * 64
            manifest = {"build": "passed", "signature": signatures[source_key],
                        "reused": reused,
                        "sha256": {"operator.so": "signed-bits"},
                        "configuration": {"source": source, "soc": "ascend910b4"}}
            path = tmp_path / "reports/build.json"
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(manifest))
        else:
            assert name == "operator-cv-npu"
            assert "--junitxml=" + str(log_dir / "cv-npu.xml") in command
            _write_real_npu_junit(log_dir / "cv-npu.xml")
        return SimpleNamespace(returncode=0, cleanup_complete=True,
                               log=str(log_dir / f"{name}.log"))

    def observe(_config, _directory, *, before=None):
        resource_calls.append("release" if before else "before")
        return {"status": "passed"} if before else {"devices": [0, 1, 2, 3], "memory": []}

    monkeypatch.setattr(paired, "run_phase", phase)
    monkeypatch.setattr(paired, "_observe_resources", observe)
    first = paired.ensure_current_operators(config_path, config, acceptance, tmp_path / "run1")
    assert first["build"] == "rebuilt" and first["accuracy"] == "fresh_device_completion"
    assert first["source_sha256"] == first["identity"]["source_sha256"]
    assert first["resource_release"] == "passed"
    assert first["cv_cases"] == 21 and first["rotation_cases"] == 104
    assert phases == ["operator-build", "operator-cv-npu"]
    second = paired.ensure_current_operators(config_path, config, acceptance, tmp_path / "run2")
    assert second["build"] == "reused" and second["accuracy"] == "reused_prior_evidence"
    assert phases == ["operator-build", "operator-cv-npu", "operator-build"]
    assert resource_calls == ["before", "release"]

    # Full install invokes the same one-command runner with --require-fresh-npu:
    # build reuse remains valid, but old NPU evidence must not satisfy this run.
    forced = paired.ensure_current_operators(config_path, config, acceptance,
                                              tmp_path / "run-full", require_fresh_npu=True)
    assert forced["build"] == "reused" and forced["accuracy"] == "fresh_device_completion"
    assert phases[-2:] == ["operator-build", "operator-cv-npu"]
    assert resource_calls == ["before", "release", "before", "release"]

    (tmp_path / "csrc/attention_cv.cpp").write_text("changed kernel")
    third = paired.ensure_current_operators(config_path, config, acceptance, tmp_path / "run3")
    assert third["build"] == "rebuilt" and third["accuracy"] == "fresh_device_completion"
    assert third["build_signature"] != first["build_signature"]
    assert phases[-2:] == ["operator-build", "operator-cv-npu"]
    assert resource_calls == ["before", "release", "before", "release", "before", "release"]


def test_real_npu_junit_rejects_skips_and_missing_cards(tmp_path):
    path = tmp_path / "cv.xml"
    _write_real_npu_junit(path, skip_cv=True)
    with pytest.raises(RuntimeError, match="skipped"):
        paired._real_npu_junit(path, [0, 1, 2, 3])
    _write_real_npu_junit(path)
    with pytest.raises(RuntimeError, match="all selected physical cards"):
        paired._real_npu_junit(path, [4, 5, 6, 7])


def test_cli_threads_fresh_npu_gate_to_default_native_baseline(tmp_path, monkeypatch):
    received = {}

    def run(config_path, **options):
        received.update(config_path=config_path, **options)
        return {"exit_code": 0}

    monkeypatch.setattr(paired, "run_paired", run)
    assert paired.main(["--config", str(tmp_path / "target.json"),
                        "--acceptance", str(tmp_path / "acceptance.json"),
                        "--log-dir", str(tmp_path / "logs"),
                        "--native-only", "--require-fresh-npu"]) == 0
    assert received["native_only"] is True
    assert received["require_fresh_npu"] is True


def test_operator_gate_failure_blocks_both_variants_and_keeps_phase_rc(tmp_path, monkeypatch):
    config_path = tmp_path / "target.json"
    acceptance_path = tmp_path / "acceptance.json"
    config_path.write_text(json.dumps({"devices": [0, 1, 2, 3]}))
    acceptance_path.write_text(json.dumps(_acceptance()))
    monkeypatch.setattr(paired, "ensure_current_operators", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        paired.OperatorGateError("operator-cv-npu", "NPU oracle mismatch; log=cv.log",
                                 returncode=7, evidence={"status": "failed", "accuracy": "failed"})))
    monkeypatch.setattr(paired, "_observe_resources",
                        lambda *_args, **_kwargs: pytest.fail("resource/variant phase started after NPU gate failure"))
    monkeypatch.setattr(paired, "_run_variant",
                        lambda *_args, **_kwargs: pytest.fail("native or OSCAR started after NPU gate failure"))
    report = paired.run_paired(config_path, output=tmp_path / "paired.json",
                               log_dir=tmp_path / "logs", acceptance_path=acceptance_path)
    assert report["status"] == "failed" and report["exit_code"] == 7
    assert report["failed_phase"] == "operator-cv-npu"
    assert report["native"] == "not_run" and report["oscar"] == "not_run"
    assert json.loads((tmp_path / "paired.json").read_text())["operator_gate"]["accuracy"] == "failed"


@pytest.mark.parametrize("probe_rc,release_status,failed_phase,expected_rc", [
    (7, "passed", "operator-cv-npu", 7),
    (0, "failed", "operator-cv-npu-release", 1),
])
def test_new_device_gate_preserves_probe_rc_and_requires_npu_release(
        tmp_path, monkeypatch, probe_rc, release_status, failed_phase, expected_rc):
    # #125/#126 and H15: a failed oracle or retained selected-card memory
    # cannot create reusable accuracy evidence, even if pytest itself exits 0.
    from tools import build_ops
    monkeypatch.setattr(paired, "ROOT", tmp_path)
    for name in ("csrc/attention_cv.cpp", "oscar_ascend/runtime.py",
                 "tests/test_cv_contracts.py", "tests/test_rotation_npu.py",
                 "tools/generate_rotation_cpu_cases.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source")
    config_path = tmp_path / "target.json"
    config = {"devices": [0, 1, 2, 3], "soc_version": "ascend910b4"}
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(build_ops, "reusable_build", lambda _path, _signature:
                        json.loads((tmp_path / "reports/build.json").read_text()))

    def phase(name, _command, *, log_dir, **_kwargs):
        if name == "operator-build":
            manifest = {"build": "passed", "signature": "a" * 64, "reused": False,
                        "sha256": {"operator.so": "signed-bits"},
                        "configuration": {"source": paired.file_fingerprint(tmp_path / "csrc"),
                                          "soc": "ascend910b4"}}
            path = tmp_path / "reports/build.json"
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(manifest))
        elif probe_rc == 0:
            _write_real_npu_junit(log_dir / "cv-npu.xml")
        return SimpleNamespace(returncode=0 if name == "operator-build" else probe_rc,
                               cleanup_complete=True, log=str(log_dir / f"{name}.log"))

    monkeypatch.setattr(paired, "run_phase", phase)
    monkeypatch.setattr(paired, "_observe_resources",
                        lambda _config, _directory, *, before=None:
                        {"status": release_status} if before else {"devices": [0, 1, 2, 3], "memory": []})
    with pytest.raises(paired.OperatorGateError) as caught:
        paired.ensure_current_operators(config_path, config, _acceptance(), tmp_path / "run")
    assert caught.value.phase == failed_phase and caught.value.returncode == expected_rc
    assert not (tmp_path / "build/ascendc/oscar_fast_probe_cv_gate.json").exists()
    assert json.loads((tmp_path / "run/operator-gate.json").read_text())["status"] == "failed"
