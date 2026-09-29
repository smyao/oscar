# 档案 #94/#95/#125/#133/#140-145/#150：一键观察不发推理请求，fe0 pin与
# candidate/native标签不能虚构已配对的用户负载，图内残差保持missing。
import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from tools import observe_serve
from tools.phase import PhaseResult


def test_fe0_kernel_pin_and_candidate_flag():
    baseline = observe_serve._source_identity("baseline", {"experimental_history_reuse": False})
    assert baseline["fe0_production_kernels_match"] is True
    with pytest.raises(RuntimeError, match="requires experimental_history_reuse"):
        observe_serve._source_identity("candidate", {"experimental_history_reuse": False})
    assert observe_serve._source_identity("candidate", {"experimental_history_reuse": True})["variant"] == "candidate"


def test_step_summary_keeps_rank_buckets_and_missing_graph_separate(tmp_path):
    trace = tmp_path / "trace"
    trace.mkdir()
    rows = [
        {"t": "oscar-passive-step", "step_id": "one", "rank": 0, "bucket": "prefill",
         "status": "pending"},
        {"t": "oscar-passive-step", "step_id": "one", "rank": 0, "bucket": "prefill",
         "status": "measured", "step_device_ms": 10.0, "oscar_union_ms": 7.0,
         "residual_ms": 3.0, "scheduled_prompt_tokens": 512,
         "scopes": [{"phase": "target_forward", "duration_ms": 9.0},
                    {"phase": "fia", "duration_ms": 7.0}]},
        {"t": "oscar-passive-step", "step_id": "two", "rank": 1, "bucket": "decode",
         "status": "missing", "missing_reason": "graph_replay_has_no_python_oscar_phase_breakdown"},
    ]
    (trace / "passive-step-123.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    report = observe_serve._step_summary(trace, tmp_path / "summary.json", "baseline")
    assert report["by_rank_bucket"]["0"]["prefill"]["residual_p50_ms"] == 3.0
    assert report["by_rank_bucket"]["1"]["decode"]["missing"] == 1
    assert len(report["samples"]) == 2


def test_graph_only_step_keeps_whole_graph_time_but_no_attention_residual(tmp_path):
    trace = tmp_path / "trace"
    trace.mkdir()
    record = {"t": "oscar-passive-step", "step_id": "graph-step", "rank": 2,
        "bucket": "decode", "status": "missing", "step_device_ms": 12.0,
        "attention_union_ms": None, "oscar_union_ms": None, "residual_ms": None,
        "missing_reason": "graph_replay_has_no_python_oscar_phase_breakdown",
        "scopes": [{"phase": "target_forward", "duration_ms": 11.0},
                   {"phase": "graph_replay", "duration_ms": 10.0}]}
    (trace / "passive-step-321.jsonl").write_text(json.dumps(record) + "\n")
    report = observe_serve._step_summary(trace, tmp_path / "summary.json", "baseline")
    row = report["by_rank_bucket"]["2"]["decode"]
    assert report["status"] == row["coverage"] == "graph_opaque"
    assert row["step_p50_ms"] == 12.0
    assert row["scopes"]["graph_replay"]["p50_ms"] == 10.0
    assert row["residual_p50_ms"] is None
    assert "decode=rank2:step12.00ms" in observe_serve._headline(report)
    assert "residualmissing/graph10.0" in observe_serve._headline(report)


def test_plan_makes_no_requests(capsys):
    assert observe_serve.main(["--plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["variant"] == "baseline"
    assert plan["inference_requests_generated"] == 0
    assert "managed_service_health" in plan["phases"]


def test_rear_card_plan_is_limited_to_devices_four_through_seven(capsys):
    assert observe_serve.main(["--plan", "--variant", "candidate", "--rear-cards",
                               "--probe-only", "--diagnose-mixed"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["devices"] == [4, 5, 6, 7]
    assert plan["port"] == 7878 and plan["placement"] == "rear"


def test_candidate_gate_requires_signed_artifact_graph_and_precision(monkeypatch, tmp_path):
    from oscar_ascend.ops import loader
    config = json.loads((observe_serve.ROOT / "configs/target.json").read_text())
    config_path = tmp_path / "effective-target.json"
    config_path.write_text(json.dumps({**config, "experimental_history_reuse": True}))
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    status = {"phases": []}
    manifest = {"signature": "abc", "sha256": {"extension": "def"}}
    monkeypatch.setattr(loader, "validate_build_artifacts", lambda *_args: manifest)
    monkeypatch.setattr(observe_serve, "read_npu_resources", lambda *args, **kwargs: {})
    monkeypatch.setattr(observe_serve, "wait_for_release", lambda *args, **kwargs: {"status": "passed"})
    allowed = True
    def phase(name, command, *, log_dir, **kwargs):
        assert name == "history-reuse-npu"
        assert command[command.index("--config") + 1] == str(config_path)
        report = {"status": "passed", "candidate_evaluation_allowed": allowed,
            "default_route": "fe0", "graph_capture": "passed", "graph_replay": "passed",
            "artifact_signature": "abc", "artifact_sha256": manifest["sha256"],
            "q1_schedule_gate": "passed", "q1_precision": "passed", "q1_performance": "passed",
            "q1_graph_capture": "passed", "q1_graph_replay": "passed",
            "reference_commit": "fe0e925e7ef78bfb64217a300031502fc4a7b7bc",
            "production_promotion": "blocked_pending_full_model_quality_and_service_performance",
            "fe0_source_sha256": {
                "csrc/kernels/attention_cv.cpp": observe_serve.FE0_KERNEL_SHA256["attention_cv.cpp"],
                "csrc/kernels/oscar_common.h": observe_serve.FE0_KERNEL_SHA256["oscar_common.h"]}}
        (log_dir / "history-reuse.json").write_text(json.dumps(report))
        return PhaseResult(name, command, 0, 1.0, False, False, str(log_dir / "gate.log"), True)
    monkeypatch.setattr(observe_serve, "run_phase", phase)
    observe_serve._candidate_gate(config_path, config, {}, log_dir, status)
    assert status["candidate_gate"]["candidate_evaluation_allowed"] is True
    allowed = False
    with pytest.raises(RuntimeError, match="lacks exact signed artifact"):
        observe_serve._candidate_gate(config_path, config, {}, log_dir, status)


@pytest.mark.parametrize("diagnose_mixed", [False, True])
def test_probe_only_finishes_after_operator_gate_without_model(monkeypatch, tmp_path, diagnose_mixed):
    config = observe_serve.ROOT / "configs/target.json"
    seen = []
    monkeypatch.setattr(observe_serve, "_preflight", lambda *args: seen.append("preflight"))
    monkeypatch.setattr(observe_serve, "_candidate_gate", lambda *args: seen.append("candidate-gates"))
    monkeypatch.setattr(observe_serve, "_fast_unpack_gate", lambda *args: seen.append("fast-unpack-gate"))
    monkeypatch.setattr(observe_serve, "_q4_diagnostic", lambda *args: seen.append("q4-diagnostic"))
    monkeypatch.setattr(observe_serve, "_mixed_diagnostic", lambda *args: seen.append("mixed-diagnostic"))
    monkeypatch.setattr(observe_serve, "managed_server", lambda *args, **kwargs:
                        pytest.fail("probe-only must not launch a model or AISBench"))
    monkeypatch.setattr(observe_serve, "_terminal", lambda *args, **kwargs: None)
    logs = tmp_path / "probe"
    assert observe_serve.run(config, logs, "candidate", probe_only=True,
                             diagnose_mixed=diagnose_mixed, rear_cards=True) == 0
    assert seen == ["preflight", "candidate-gates", "fast-unpack-gate"] + (
        ["mixed-diagnostic"] if diagnose_mixed else [])
    status = json.loads((logs / "status.json").read_text())
    assert status["service_started"] is False
    assert status["performance_acceptance"] == "operator_only_not_end_to_end"
    effective = json.loads((logs / "effective-target.json").read_text())
    assert effective["experimental_fast_unpack"] is True
    assert effective["experimental_weighted_q4"] is True
    assert effective["devices"] == [4, 5, 6, 7] and effective["port"] == 7878


def test_q4_diagnostic_records_gap_without_claiming_performance_pass(monkeypatch, tmp_path):
    from oscar_ascend.ops import loader
    config = json.loads((observe_serve.ROOT / "configs/target.json").read_text())
    manifest = {"signature": "q4-signature", "sha256": {"extension": "q4-binary"}}
    monkeypatch.setattr(loader, "validate_build_artifacts", lambda *args: manifest)
    monkeypatch.setattr(observe_serve, "read_npu_resources", lambda *args, **kwargs: {})
    monkeypatch.setattr(observe_serve, "wait_for_release", lambda *args, **kwargs: {"status": "passed"})
    report = {"status": "observed", "native_oracle": "passed", "oscar_oracle": "passed",
              "profile_parity": "bitwise_passed", "profile_observed": True,
              "artifact_signature": manifest["signature"], "artifact_sha256": manifest["sha256"],
              "oscar_over_native": 15.0}
    def phase(name, command, **kwargs):
        assert name == "q4-hotpath" and "tools.probe_decode_hotpath" in command
        Path(command[command.index("--output") + 1]).write_text(json.dumps(report))
    monkeypatch.setattr(observe_serve, "_phase", phase)
    status = {}
    observe_serve._q4_diagnostic(observe_serve.ROOT / "configs/target.json", config, {}, tmp_path, status)
    assert status["q4_diagnostic"]["performance_acceptance"] == "not_established"
    report["profile_parity"] = "failed"
    with pytest.raises(RuntimeError, match="lacks signed"):
        observe_serve._q4_diagnostic(observe_serve.ROOT / "configs/target.json", config, {}, tmp_path, {})


def test_fast_unpack_gate_requires_exact_new_npu_evidence(monkeypatch, tmp_path):
    from oscar_ascend.ops import loader
    config = json.loads((observe_serve.ROOT / "configs/target.json").read_text())
    manifest = {"signature": "fast-signature", "sha256": {"extension": "fast-binary"}}
    monkeypatch.setattr(loader, "validate_build_artifacts", lambda *args: manifest)
    monkeypatch.setattr(observe_serve, "read_npu_resources", lambda *args, **kwargs: {})
    monkeypatch.setattr(observe_serve, "wait_for_release", lambda *args, **kwargs: {"status": "passed"})
    report = {key: "passed" for key in ("status", "precision", "graph_capture",
                                          "graph_replay", "performance",
                                          "p0_performance", "q4_split_scan_gate")}
    report.update(artifact_signature=manifest["signature"], artifact_sha256=manifest["sha256"])
    def phase(name, command, **kwargs):
        assert name == "fast-unpack" and "tools.probe_fast_unpack" in command
        output = Path(command[command.index("--output") + 1])
        assert output.name != name + ".json"
        output.write_text(json.dumps(report))
    monkeypatch.setattr(observe_serve, "_phase", phase)
    status = {}
    observe_serve._fast_unpack_gate(observe_serve.ROOT / "configs/target.json", config, {}, tmp_path, status)
    assert status["fast_unpack_gate"]["full_service_performance"] == "not_established"
    for key in ("precision", "graph_capture", "graph_replay", "performance",
                "p0_performance", "q4_split_scan_gate", "artifact_signature"):
        saved = report[key]
        report[key] = "failed"
        with pytest.raises(RuntimeError, match=key):
            observe_serve._fast_unpack_gate(observe_serve.ROOT / "configs/target.json", config, {}, tmp_path, {})
        report[key] = saved


def test_real_phase_status_cannot_overwrite_q4_measurement(monkeypatch, tmp_path):
    # #151: use the real subprocess/phase writer, not the previous _phase mock
    # that failed to expose its reserved <phase>.json output path.
    import sys
    from tools.phase import run_phase as real_run_phase
    from oscar_ascend.ops import loader
    config = json.loads((observe_serve.ROOT / "configs/target.json").read_text())
    manifest = {"signature": "probe-signature", "sha256": {"extension": "probe-binary"}}
    report = {"status": "observed", "native_oracle": "passed", "oscar_oracle": "passed",
              "profile_parity": "bitwise_passed", "profile_observed": True,
              "artifact_signature": manifest["signature"], "artifact_sha256": manifest["sha256"]}
    monkeypatch.setattr(loader, "validate_build_artifacts", lambda *args: manifest)
    monkeypatch.setattr(observe_serve, "read_npu_resources", lambda *args, **kwargs: {})
    monkeypatch.setattr(observe_serve, "wait_for_release", lambda *args, **kwargs: {"status": "passed"})
    def runner(name, command, **kwargs):
        output = command[command.index("--output") + 1]
        child = [sys.executable, "-c", "import pathlib,sys;pathlib.Path(sys.argv[1]).write_text(sys.argv[2])",
                 output, json.dumps(report)]
        return real_run_phase(name, child, **kwargs)
    monkeypatch.setattr(observe_serve, "run_phase", runner)
    status = {"phases": []}
    observe_serve._q4_diagnostic(observe_serve.ROOT / "configs/target.json", config, {}, tmp_path, status)
    assert json.loads((tmp_path / "q4-hotpath-report.json").read_text()) == report
    phase = json.loads((tmp_path / "q4-hotpath.json").read_text())
    assert phase["phase"] == "q4-hotpath" and phase["returncode"] == 0
    assert status["q4_diagnostic"]["status"] == "observed"


def test_q4_child_failure_is_not_hidden_by_resource_observation_error(monkeypatch, tmp_path):
    config = json.loads((observe_serve.ROOT / "configs/target.json").read_text())
    monkeypatch.setattr(observe_serve, "read_npu_resources", lambda *args, **kwargs: {})
    def fail(*args, **kwargs):
        error = RuntimeError("q4 child failed")
        error.returncode = 2
        error.phase = "q4-hotpath"
        raise error
    monkeypatch.setattr(observe_serve, "_phase", fail)
    monkeypatch.setattr(observe_serve, "wait_for_release", lambda *args, **kwargs:
                        (_ for _ in ()).throw(RuntimeError("resource read failed")))
    status = {}
    with pytest.raises(RuntimeError, match="q4 child failed") as error:
        observe_serve._q4_diagnostic(observe_serve.ROOT / "configs/target.json", config, {}, tmp_path, status)
    assert error.value.returncode == 2
    assert status["q4_resource_release"]["status"] == "failed"


def test_mixed_diagnostic_retains_real_child_report_and_requires_signed_oracle(monkeypatch, tmp_path):
    import sys
    from tools.phase import run_phase as real_run_phase
    from oscar_ascend.ops import loader
    config = json.loads((observe_serve.ROOT / "configs/target.json").read_text())
    manifest = {"signature": "mixed-signature", "sha256": {"extension": "mixed-binary"}}
    report = {"status": "observed", "accuracy": "passed",
              "artifact_signature": manifest["signature"], "artifact_sha256": manifest["sha256"]}
    monkeypatch.setattr(loader, "validate_build_artifacts", lambda *args: manifest)
    monkeypatch.setattr(observe_serve, "read_npu_resources", lambda *args, **kwargs: {})
    monkeypatch.setattr(observe_serve, "wait_for_release", lambda *args, **kwargs: {"status": "passed"})
    def runner(name, command, **kwargs):
        assert name == "mixed-attention" and "tools.probe_mixed_attention" in command
        output = command[command.index("--output") + 1]
        child = [sys.executable, "-c", "import pathlib,sys;pathlib.Path(sys.argv[1]).write_text(sys.argv[2])",
                 output, json.dumps(report)]
        return real_run_phase(name, child, **kwargs)
    monkeypatch.setattr(observe_serve, "run_phase", runner)
    status = {"phases": []}
    observe_serve._mixed_diagnostic(observe_serve.ROOT / "configs/target.json", config, {}, tmp_path, status)
    assert json.loads((tmp_path / "mixed-attention-report.json").read_text()) == report
    assert json.loads((tmp_path / "mixed-attention.json").read_text())["returncode"] == 0
    assert status["mixed_diagnostic"]["performance_acceptance"] == "not_established"
    report["accuracy"] = "failed"
    with pytest.raises(RuntimeError, match="accuracy"):
        observe_serve._mixed_diagnostic(observe_serve.ROOT / "configs/target.json", config, {}, tmp_path, status)


def test_mixed_child_failure_survives_resource_observation_error(monkeypatch, tmp_path):
    config = json.loads((observe_serve.ROOT / "configs/target.json").read_text())
    monkeypatch.setattr(observe_serve, "read_npu_resources", lambda *args, **kwargs: {})
    def fail(*args, **kwargs):
        error = RuntimeError("mixed child failed")
        error.returncode = 2
        error.phase = "mixed-attention"
        raise error
    monkeypatch.setattr(observe_serve, "_phase", fail)
    monkeypatch.setattr(observe_serve, "wait_for_release", lambda *args, **kwargs:
                        (_ for _ in ()).throw(RuntimeError("resource read failed")))
    status = {}
    with pytest.raises(RuntimeError, match="mixed child failed") as error:
        observe_serve._mixed_diagnostic(observe_serve.ROOT / "configs/target.json", config, {}, tmp_path, status)
    assert error.value.returncode == 2
    assert status["mixed_resource_release"]["status"] == "failed"


def test_candidate_effective_config_is_one_click_and_gate_blocks_service(monkeypatch, tmp_path):
    source = json.loads((observe_serve.ROOT / "configs/target.json").read_text())
    source["experimental_history_reuse"] = False
    config_path = tmp_path / "target.json"
    config_path.write_text(json.dumps(source))
    logs = tmp_path / "logs"
    observed = []
    monkeypatch.setattr(observe_serve, "_preflight", lambda path, *_args: observed.append(path))
    def gate(path, *_args):
        observed.append(path)
        raise RuntimeError("candidate graph gate rejected")
    monkeypatch.setattr(observe_serve, "_candidate_gate", gate)
    monkeypatch.setattr(observe_serve, "managed_server", lambda *_args, **kwargs:
                        pytest.fail("candidate service must not start after gate failure"))
    monkeypatch.setattr(observe_serve, "_terminal", lambda *args, **kwargs: None)
    assert observe_serve.run(config_path, logs, "candidate") == 1
    assert len(observed) == 2 and observed[0] == observed[1] == logs / "effective-target.json"
    assert json.loads((logs / "effective-target.json").read_text())["experimental_history_reuse"] is True
    assert json.loads(config_path.read_text())["experimental_history_reuse"] is False
    status = json.loads((logs / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["target_config"]["original_sha256"] != status["target_config"]["effective_sha256"]


def test_candidate_child_rc2_reaches_outer_status_without_serving(monkeypatch, tmp_path):
    config = json.loads((observe_serve.ROOT / "configs/target.json").read_text())
    config_path = tmp_path / "target.json"
    config_path.write_text(json.dumps(config))
    logs = tmp_path / "logs"
    monkeypatch.setattr(observe_serve, "_preflight", lambda *_args: None)
    monkeypatch.setattr(observe_serve, "read_npu_resources", lambda *args, **kwargs: {})
    monkeypatch.setattr(observe_serve, "wait_for_release", lambda *args, **kwargs: {"status": "passed"})
    monkeypatch.setattr(observe_serve, "run_phase", lambda name, command, *, log_dir, **kwargs:
        PhaseResult(name, command, 2, 1.0, False, False, str(log_dir / "gate.log"), True))
    monkeypatch.setattr(observe_serve, "managed_server", lambda *_args, **kwargs:
                        pytest.fail("candidate service must not start after rc2"))
    monkeypatch.setattr(observe_serve, "_terminal", lambda *args, **kwargs: None)
    assert observe_serve.run(config_path, logs, "candidate") == 2
    state = json.loads((logs / "status.json").read_text())
    assert state["failed_phase"] == "history-reuse-npu"
    assert state["returncode"] == 2
    assert state["candidate_resource_release"]["status"] == "passed"


def test_managed_service_only_observes_external_load(monkeypatch, tmp_path):
    from benchmarks import passive
    config = json.loads((observe_serve.ROOT / "configs/target.json").read_text())
    config["experimental_history_reuse"] = False
    config_path = tmp_path / "target.json"
    config_path.write_text(json.dumps(config))
    logs = tmp_path / "logs"
    calls = []

    class Server:
        base_url = "http://127.0.0.1:8989"
        trace_dir = tmp_path / "trace"
        def check_alive(self):
            raise KeyboardInterrupt("test stop after READY")

    @contextmanager
    def managed(_config, _path, *, log_dir, lifecycle, native):
        calls.append(("managed", native))
        Server.trace_dir.mkdir()
        lifecycle["trace_dir"] = str(Server.trace_dir)
        yield Server()
        # This record appears only after the server context exits. The final
        # summary must be reread after cleanup, not frozen while it is live.
        (Server.trace_dir / "passive-step-99.jsonl").write_text(json.dumps({
            "t": "oscar-passive-step", "step_id": "late", "rank": 0,
            "bucket": "prefill", "status": "measured", "step_device_ms": 5.0,
            "attention_union_ms": 2.0, "oscar_union_ms": 2.0,
            "residual_ms": 3.0, "scopes": []}) + "\n")
        lifecycle["cleanup_complete"] = True

    def passive_only(_url, *, stop, ready, output, on_window_end, **kwargs):
        calls.append(("passive", kwargs["variant"]))
        ready.set()
        stop.wait(1)
        return {"status": "not_run", "windows": [], "performance_acceptance": "not_run"}

    monkeypatch.setattr(observe_serve, "_preflight", lambda *args: calls.append(("preflight",)))
    monkeypatch.setattr(observe_serve, "managed_server", managed)
    monkeypatch.setattr(observe_serve, "read_npu_resources", lambda *args, **kwargs: {"status": "observed"})
    monkeypatch.setattr(observe_serve, "wait_for_release", lambda *args, **kwargs: {"status": "passed"})
    monkeypatch.setattr(observe_serve, "_terminal", lambda *args, **kwargs: None)
    monkeypatch.setattr(passive, "observe", passive_only)
    assert observe_serve.run(config_path, logs, "baseline") == 0
    assert ("managed", False) in calls and ("passive", "oscar") in calls
    assert json.loads((logs / "status.json").read_text())["status"] == "stopped"
    assert json.loads((logs / "summary.json").read_text())["status"] == "observed"
    assert json.loads((logs / "status.json").read_text())["step_evidence"]["samples"][0]["step_id"] == "late"


def test_cleanup_resource_error_preserves_server_rc_and_late_summary(monkeypatch, tmp_path):
    from benchmarks import passive
    config_path = tmp_path / "target.json"
    config_path.write_text((observe_serve.ROOT / "configs/target.json").read_text())
    logs = tmp_path / "logs"
    trace = tmp_path / "trace"
    class ServerFailure(RuntimeError):
        returncode = 7
        phase = "serve"
    class Server:
        base_url = "http://127.0.0.1:8989"
        trace_dir = trace
        def check_alive(self):
            raise KeyboardInterrupt("test stop")
    @contextmanager
    def managed(_config, _path, *, lifecycle, **kwargs):
        trace.mkdir()
        lifecycle["trace_dir"] = str(trace)
        yield Server()
        (trace / "passive-step-1.jsonl").write_text(json.dumps({
            "t": "oscar-passive-step", "step_id": "late", "rank": 0,
            "bucket": "prefill", "status": "measured", "step_device_ms": 4.0,
            "attention_union_ms": 1.0, "oscar_union_ms": 1.0,
            "residual_ms": 3.0, "scopes": []}) + "\n")
        lifecycle["cleanup_complete"] = True
        raise ServerFailure("owned server rc7")
    def observe(_url, *, stop, ready, **kwargs):
        ready.set(); stop.wait(1)
        return {"status": "not_run", "windows": []}
    monkeypatch.setattr(observe_serve, "_preflight", lambda *args: None)
    monkeypatch.setattr(observe_serve, "managed_server", managed)
    monkeypatch.setattr(observe_serve, "read_npu_resources", lambda *args, **kwargs: {})
    monkeypatch.setattr(observe_serve, "wait_for_release", lambda *args, **kwargs:
                        (_ for _ in ()).throw(OSError("resource observer unavailable")))
    monkeypatch.setattr(observe_serve, "_terminal", lambda *args, **kwargs: None)
    monkeypatch.setattr(passive, "observe", observe)
    assert observe_serve.run(config_path, logs, "baseline") == 7
    state = json.loads((logs / "status.json").read_text())
    assert state["failed_phase"] == "serve" and state["returncode"] == 7
    assert state["resource_release"]["status"] == "failed"
    assert state["step_evidence"]["samples"][0]["step_id"] == "late"
