# 档案 #70–73/#94–95/#125–126/#131–140：一键先原生后OSCAR，真实CV门只执行一次且失败阻断服务。
# 本次用户指令：保留探针，取消真机环境检查；附录 F 指定本任务设备与 SOC。
"""Exercise deployment orchestration without installing or opening an NPU."""
import json
import os
from dataclasses import asdict
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

from tools import deploy, service_probe
from tools.phase import PhaseResult
from tools.target_cli import target_env


ROOT = Path(__file__).resolve().parents[1]
PROBES = ("probe-ops", "native-synthetic", "service-probe")
AUDITS = {"environment", "runtime-readiness", "binary-readiness", "native-integrity"}


def _configure(monkeypatch, tmp_path, *, devices=None):
    config = json.loads((ROOT / "configs/target.json").read_text())
    if devices is not None:
        config["devices"] = devices
    path = tmp_path / "target.json"
    path.write_text(json.dumps(config))
    logs = tmp_path / "logs"
    monkeypatch.setattr(sys, "argv", ["deploy", "--config", str(path), "--log-dir", str(logs)])
    return config, path, logs


def _phase_result(name, command, log_dir, code=0):
    return PhaseResult(name, command, code, 0.01, False, False,
                       str(log_dir / f"{name}.log"), True)


def _native_evidence(log_dir):
    child = log_dir / "native-synthetic" / "native"
    child.mkdir(parents=True, exist_ok=True)
    (child / "report.json").write_text(json.dumps({"status": "measured", "mode": "synthetic_mixed", "variant": "native"}))
    (log_dir / "native-synthetic-report.json").write_text(json.dumps({
        "status": "passed", "operator_gate": {
            "status": "passed", "build": "reused", "accuracy": "fresh_device_completion",
            "resource_release": "passed"},
        "native": {"status": "passed", "returncode": 0, "resource_release": "passed",
                   "owned_server_cleanup_complete": True, "runner_cleanup_complete": True}}))


def test_checked_in_target_has_current_task_devices_and_build_soc():
    # Appendix F declares 0–3 / 910B4 for this checkout; another checkout's
    # remembered allocation or a parent shell's mask must not take precedence.
    config = json.loads((ROOT / "configs/target.json").read_text())
    assert config["devices"] == [0, 1, 2, 3]
    assert config["soc_version"] == "ascend910b4"
    env = target_env(config, {"ASCEND_RT_VISIBLE_DEVICES": "4,5,6,7"})
    assert env["ASCEND_RT_VISIBLE_DEVICES"] == "0,1,2,3"


def test_install_build_probes_and_formal_serve_share_configured_devices(monkeypatch, tmp_path):
    config, config_path, logs = _configure(monkeypatch, tmp_path, devices=[8, 9, 10, 11])
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4,5,6,7")
    monkeypatch.setenv("OSCAR_TARGET_CONFIG", "/some/other/task.json")
    monkeypatch.setenv("VLLM_PLUGINS", "another_plugin")
    calls = []
    served = []

    def phase(name, command, *, cwd, log_dir, env, **kwargs):
        calls.append((name, command))
        assert cwd == ROOT
        assert env["ASCEND_RT_VISIBLE_DEVICES"] == "8,9,10,11"
        assert env["OSCAR_TARGET_CONFIG"] == str(config_path)
        assert env["OSCAR_RUN_NPU_TESTS"] == "1"
        assert env["OSCAR_TERMINAL_LOG_MODE"] == "compact"
        assert kwargs["heartbeat"] == 60
        assert {"ascend", "oscar_ascend", "another_plugin"} <= set(env["VLLM_PLUGINS"].split(","))
        if name == "native-synthetic":
            _native_evidence(log_dir)
        if name == "service-probe":
            (log_dir / "service-probe-report.json").write_text(json.dumps({"status": "passed", "resource_release": "passed", "performance": {"status": "measured"}}))
            # The phase ledger writes <name>.json after the phase exits; it must
            # not clobber the probe report the full-service gate reads (#136).
            (log_dir / "service-probe.json").write_text(json.dumps(asdict(_phase_result(name, command, log_dir))))
        return _phase_result(name, command, log_dir)

    def serve():
        served.append(list(sys.argv))
        assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "8,9,10,11"
        assert os.environ["OSCAR_TARGET_CONFIG"] == str(config_path)
        assert "--serve" in sys.argv
        assert sys.argv[sys.argv.index("--config") + 1] == str(config_path)
        return 0

    monkeypatch.setattr(deploy, "run_phase", phase)
    monkeypatch.setattr(deploy, "_compare_performance", lambda native, oscar: {"status": "passed", "performance_acceptance": "not_run"})
    monkeypatch.setattr(deploy, "_performance_lines", lambda *args, **kwargs: ["PERF_VERDICT client_ratio_gate=met acceptance=not_established"])
    monkeypatch.setattr(service_probe, "main", serve)
    with patch.dict(os.environ):
        original_env = os.environ.copy()
        original_argv = sys.argv
        assert deploy.main() == 0
        assert os.environ == original_env
        assert sys.argv == original_argv
    names = [name for name, _ in calls]
    assert set(names).isdisjoint(AUDITS)
    assert [name for name in names if name in PROBES] == list(PROBES)
    assert "probe-cv" not in names  # native preflight owns the sole CV/rotation gate.
    assert names.index("install-plugin") < names.index("build-ops") < names.index("probe-ops")
    assert names.index("prepare-rotations") < names.index("native-synthetic") < names.index("service-probe")
    commands = dict(calls)
    build = commands["build-ops"]
    assert build[build.index("--soc") + 1] == config["soc_version"]
    install = commands["install-plugin"]
    assert Path(install[install.index("-e") + 1]) == ROOT
    assert "--no-deps" in install
    native = commands["native-synthetic"]
    assert "--native-only" in native
    assert "--require-fresh-npu" in native
    assert native[native.index("--config") + 1] == str(config_path)
    assert (logs / "paired-performance-report.json").is_file()
    assert len(served) == 1
    final_status = json.loads((logs / "status.json").read_text())
    assert len(final_status["phases"]) == len(calls)
    assert final_status["paired_performance"]["performance_acceptance"] == "not_run"


@pytest.mark.parametrize("selected", ["build-ops", "probe-cv"])
def test_explicit_operator_diagnostic_remains_available_without_formal_service(monkeypatch, tmp_path, selected):
    config, config_path, logs = _configure(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["deploy", "--config", str(config_path),
                                      "--log-dir", str(logs), "--only", selected])
    calls = []

    def phase(name, command, *, log_dir, **kwargs):
        calls.append((name, command))
        return _phase_result(name, command, log_dir)

    monkeypatch.setattr(deploy, "run_phase", phase)
    monkeypatch.setattr(service_probe, "main", lambda: pytest.fail("formal serve must not start"))
    assert deploy.main() == 0
    assert len(calls) == 1 and calls[0][0] == selected
    command = calls[0][1]
    if selected == "build-ops":
        assert command[command.index("--soc") + 1] == config["soc_version"]
    else:
        assert str(ROOT / "tests/test_cv_contracts.py") in command
        assert str(ROOT / "tests/test_rotation_npu.py") in command
    assert json.loads((logs / "status.json").read_text())["status"] == "selected_phase_passed"


def test_only_phase_missing_from_plan_fails_instead_of_silent_pass(monkeypatch, tmp_path):
    _, config_path, logs = _configure(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["deploy", "--config", str(config_path),
                                      "--log-dir", str(logs), "--only", "build-ops"])
    monkeypatch.setattr(deploy, "plan", lambda *_args: [])
    assert deploy.main() == 1
    state = json.loads((logs / "status.json").read_text())
    assert state["status"] == "failed" and state["failed_phase"] == "build-ops"


@pytest.mark.parametrize("failure", PROBES)
def test_any_probe_failure_stops_formal_serve_and_keeps_original_code(monkeypatch, tmp_path, failure):
    _, _, logs = _configure(monkeypatch, tmp_path)
    calls = []

    def phase(name, command, *, log_dir, **kwargs):
        calls.append(name)
        if name == "native-synthetic" and name != failure:
            _native_evidence(log_dir)
        return _phase_result(name, command, log_dir, 19 if name == failure else 0)

    def serve():
        pytest.fail("formal service must not start after a failed probe")

    monkeypatch.setattr(deploy, "run_phase", phase)
    monkeypatch.setattr(service_probe, "main", serve)
    assert deploy.main() == 19
    assert calls[-1] == failure
    assert set(calls).isdisjoint(AUDITS)
    status = json.loads((logs / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["failed_phase"] == failure
    assert status["phases"][-1]["returncode"] == 19


def test_resource_release_failure_blocks_formal_serve_even_when_probe_exits_zero(monkeypatch, tmp_path):
    _, _, logs = _configure(monkeypatch, tmp_path)

    def phase(name, command, *, log_dir, **kwargs):
        if name == "native-synthetic":
            _native_evidence(log_dir)
        if name == "service-probe":
            (log_dir / "service-probe-report.json").write_text(json.dumps({"status": "passed", "resource_release": "failed"}))
            (log_dir / "service-probe.json").write_text(json.dumps(asdict(_phase_result(name, command, log_dir))))
        return _phase_result(name, command, log_dir)

    def serve():
        pytest.fail("formal service must not start before owned resources are released")

    monkeypatch.setattr(deploy, "run_phase", phase)
    monkeypatch.setattr(service_probe, "main", serve)
    assert deploy.main() != 0
    status = json.loads((logs / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["failed_phase"] == "full-service-probe"


def test_native_report_missing_blocks_oscar_and_formal_service(monkeypatch, tmp_path):
    _, _, logs = _configure(monkeypatch, tmp_path)
    _native_evidence(logs)  # Prior --log-dir run must not satisfy this one (#136).
    calls = []

    def phase(name, command, *, log_dir, **kwargs):
        calls.append(name)
        return _phase_result(name, command, log_dir)

    monkeypatch.setattr(deploy, "run_phase", phase)
    monkeypatch.setattr(service_probe, "main", lambda: pytest.fail("formal serve must not start"))
    assert deploy.main() != 0
    assert calls[-1] == "native-synthetic"
    status = json.loads((logs / "status.json").read_text())
    assert status["failed_phase"] == "native-synthetic"


def test_empty_stage_plan_cannot_start_formal_service(monkeypatch, tmp_path):
    _, _, logs = _configure(monkeypatch, tmp_path)
    monkeypatch.setattr(deploy, "plan", lambda *_args: [])
    monkeypatch.setattr(service_probe, "main", lambda: pytest.fail("formal serve must not start"))
    assert deploy.main() == 1
    status = json.loads((logs / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["failed_phase"] == "full-service-probe"
    assert "native-synthetic" in status["error"]


def test_formal_service_exception_restores_environment_and_argv(monkeypatch, tmp_path):
    _, _, logs = _configure(monkeypatch, tmp_path)

    def phase(name, command, *, log_dir, **kwargs):
        if name == "native-synthetic":
            _native_evidence(log_dir)
        if name == "service-probe":
            (log_dir / "service-probe-report.json").write_text(json.dumps({
                "status": "passed", "resource_release": "passed", "performance": {"status": "measured"}}))
        return _phase_result(name, command, log_dir)

    monkeypatch.setattr(deploy, "run_phase", phase)
    monkeypatch.setattr(deploy, "_compare_performance", lambda native, oscar: {"status": "passed"})
    monkeypatch.setattr(deploy, "_performance_lines", lambda *args, **kwargs: [])
    monkeypatch.setattr(service_probe, "main", lambda: (_ for _ in ()).throw(RuntimeError("serve failed")))
    original_env, original_argv = os.environ.copy(), sys.argv
    with pytest.raises(RuntimeError, match="serve failed"):
        deploy.main()
    assert os.environ == original_env
    assert sys.argv == original_argv
    assert json.loads((logs / "status.json").read_text())["failed_phase"] == "serve"


def test_formal_service_nonzero_code_is_recorded_without_losing_original_code(monkeypatch, tmp_path):
    _, _, logs = _configure(monkeypatch, tmp_path)

    def phase(name, command, *, log_dir, **kwargs):
        if name == "native-synthetic":
            _native_evidence(log_dir)
        if name == "service-probe":
            (log_dir / "service-probe-report.json").write_text(json.dumps({
                "status": "passed", "resource_release": "passed", "performance": {"status": "measured"}}))
        return _phase_result(name, command, log_dir)

    monkeypatch.setattr(deploy, "run_phase", phase)
    monkeypatch.setattr(deploy, "_compare_performance", lambda native, oscar: {"status": "passed"})
    monkeypatch.setattr(deploy, "_performance_lines", lambda *args, **kwargs: [])
    monkeypatch.setattr(service_probe, "main", lambda: 27)
    assert deploy.main() == 27
    status = json.loads((logs / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["failed_phase"] == "serve"
    assert status["service_returncode"] == 27


def test_native_npu_release_failure_blocks_oscar(monkeypatch, tmp_path):
    _, _, logs = _configure(monkeypatch, tmp_path)
    calls = []

    def phase(name, command, *, log_dir, **kwargs):
        calls.append(name)
        if name == "native-synthetic":
            _native_evidence(log_dir)
            wrapper = json.loads((log_dir / "native-synthetic-report.json").read_text())
            wrapper["native"]["resource_release"] = "failed"
            (log_dir / "native-synthetic-report.json").write_text(json.dumps(wrapper))
        return _phase_result(name, command, log_dir)

    monkeypatch.setattr(deploy, "run_phase", phase)
    monkeypatch.setattr(service_probe, "main", lambda: pytest.fail("formal serve must not start"))
    assert deploy.main() != 0
    assert calls[-1] == "native-synthetic"
    assert json.loads((logs / "status.json").read_text())["failed_phase"] == "native-synthetic"


def test_one_click_rejects_reused_cv_evidence(monkeypatch, tmp_path):
    _, _, logs = _configure(monkeypatch, tmp_path)
    calls = []

    def phase(name, command, *, log_dir, **kwargs):
        calls.append(name)
        if name == "native-synthetic":
            _native_evidence(log_dir)
            path = log_dir / "native-synthetic-report.json"
            wrapper = json.loads(path.read_text())
            wrapper["operator_gate"]["accuracy"] = "reused_prior_evidence"
            path.write_text(json.dumps(wrapper))
        return _phase_result(name, command, log_dir)

    monkeypatch.setattr(deploy, "run_phase", phase)
    monkeypatch.setattr(service_probe, "main", lambda: pytest.fail("formal serve must not start"))
    assert deploy.main() != 0
    assert calls[-1] == "native-synthetic"
    assert json.loads((logs / "status.json").read_text())["failed_phase"] == "native-synthetic"


def test_paired_regression_blocks_formal_service_after_real_probes(monkeypatch, tmp_path):
    _, _, logs = _configure(monkeypatch, tmp_path)

    def phase(name, command, *, log_dir, **kwargs):
        if name == "native-synthetic":
            _native_evidence(log_dir)
        if name == "service-probe":
            (log_dir / "service-probe-report.json").write_text(json.dumps({"status": "passed", "resource_release": "passed", "performance": {"status": "measured"}}))
        return _phase_result(name, command, log_dir)

    monkeypatch.setattr(deploy, "run_phase", phase)
    monkeypatch.setattr(deploy, "_compare_performance", lambda native, oscar: {"status": "failed", "issues": ["synthetic-0 TTFT slower"]})
    monkeypatch.setattr(deploy, "_performance_lines", lambda *args, **kwargs: ["PERF_VERDICT client_ratio_gate=regressed acceptance=not_established"])
    monkeypatch.setattr(service_probe, "main", lambda: pytest.fail("formal serve must not start"))
    assert deploy.main() != 0
    status = json.loads((logs / "status.json").read_text())
    assert status["failed_phase"] == "paired-performance"
    assert json.loads((logs / "paired-performance-report.json").read_text())["status"] == "failed"
