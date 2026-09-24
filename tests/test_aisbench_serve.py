# 档案 #94/#95/#125/#140-147：实验服务只跳过性能配对，数值失败仍阻断；
# 终端READY必须等实际短请求与被动observer就绪，完整日志/退出码留存。
"""Host-only orchestration contracts for the user-owned AISBench service."""
import json
import os
from pathlib import Path
import time

from tools import aisbench_serve
from tools.phase import PhaseResult, atomic_json


ROOT = Path(__file__).resolve().parents[1]


def _target(tmp_path):
    config = json.loads((ROOT / "configs/target.json").read_text())
    path = tmp_path / "target.json"
    path.write_text(json.dumps(config))
    return path, config


def _fake_phase(name, command, *, log_dir, **kwargs):
    if name == "probe-ops":
        atomic_json(log_dir / "operators.json", {"status": "primitive_probe_passed",
            "cases": [{"device_completion": "passed"}]})
    return PhaseResult(name, command, 0, 0.01, False, False,
                       str(log_dir / f"{name}.log"), True)


def test_plan_and_environment_exclude_diagnostic_and_paired_work(monkeypatch, tmp_path):
    path, config = _target(tmp_path)
    for key in aisbench_serve._DIAGNOSTIC_ENV:
        monkeypatch.setenv(key, "1")
    actual, env = aisbench_serve._config(path)
    assert actual == config
    assert env["ASCEND_RT_VISIBLE_DEVICES"] == "4,5,6,7"
    assert env["OSCAR_TERMINAL_LOG_MODE"] == "compact"
    assert env["OSCAR_TARGET_CONFIG"] == str(path)
    assert not set(aisbench_serve._DIAGNOSTIC_ENV) & set(env)
    stages = aisbench_serve.phase_plan(path, tmp_path / "logs")
    assert [name for name, _ in stages] == ["build-dependencies", "install-plugin",
                                             "probe-ops", "prepare-rotations"]
    assert not any("paired_concurrency_probe" in " ".join(command)
                   or "probe_cv_hotshape" in " ".join(command)
                   or "--native" in command for _, command in stages)


def test_numeric_gate_failure_blocks_serve_and_preserves_code(monkeypatch, tmp_path):
    path, _ = _target(tmp_path)
    logs = tmp_path / "logs"
    calls = []

    def phase(name, command, **kwargs):
        calls.append(name)
        return _fake_phase(name, command, **kwargs)

    def accuracy(*_args):
        raise aisbench_serve.PhaseFailure("operator-cv-npu", 17, "frozen NPU oracle failed")

    monkeypatch.setattr(aisbench_serve, "run_phase", phase)
    monkeypatch.setattr(aisbench_serve, "_accuracy_gates", accuracy)
    monkeypatch.setattr(aisbench_serve, "_serve", lambda *_args: (_ for _ in ()).throw(
        AssertionError("service must not start after numerical failure")))
    assert aisbench_serve.run(path, logs) == 17
    assert calls == ["build-dependencies", "install-plugin"]
    status = json.loads((logs / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["failed_phase"] == "operator-cv-npu"
    assert status["returncode"] == 17
    assert status["performance_acceptance"] == "not_run"
    assert "frozen NPU oracle failed" in (logs / "operator-cv-npu-gate.log").read_text()


def test_ready_only_after_observer_then_clean_stop(monkeypatch, tmp_path):
    path, _ = _target(tmp_path)
    logs = tmp_path / "logs"
    printed = []
    phases = []
    monkeypatch.setenv("OSCAR_DEBUG_SYNC", "1")
    monkeypatch.setenv("OSCAR_PROFILER", "1")
    prior_env = os.environ.copy()

    def phase(name, command, **kwargs):
        phases.append(name)
        return _fake_phase(name, command, **kwargs)

    def accuracy(_path, _config, _logs, status):
        status["operator_gate"] = {"status": "passed", "accuracy": "reused_prior_evidence"}
        status["native_current_gate"] = {"status": "passed"}

    def service(_path, directory):
        assert os.environ["OSCAR_TERMINAL_LOG_MODE"] == "compact"
        assert "OSCAR_DEBUG_SYNC" not in os.environ
        assert "OSCAR_PROFILER" not in os.environ
        atomic_json(directory / "serve.json", {"status": "serving", "server": {"status": "healthy"},
            "external_observation": {"status": "recording"}})
        for _ in range(20):
            if (directory / "ready.json").exists():
                break
            time.sleep(0.05)
        assert (directory / "ready.json").exists()
        return {"status": "stopped", "resource_release": "passed",
                "server": {"cleanup_complete": True}}

    monkeypatch.setattr(aisbench_serve, "run_phase", phase)
    monkeypatch.setattr(aisbench_serve, "_accuracy_gates", accuracy)
    monkeypatch.setattr(aisbench_serve, "_serve", service)
    monkeypatch.setattr(aisbench_serve, "_foreground_line", printed.append)
    assert aisbench_serve.run(path, logs) == 0
    assert os.environ == prior_env
    assert phases == ["build-dependencies", "install-plugin", "probe-ops", "prepare-rotations"]
    ready = json.loads((logs / "ready.json").read_text())
    assert ready["url"] == "http://127.0.0.1:9595/v1/chat/completions"
    assert ready["model"] == "qwen3.5"
    assert ready["performance_acceptance"] == "not_run"
    assert len([line for line in printed if "AISBENCH_READY" in line]) == 1
    status = json.loads((logs / "status.json").read_text())
    assert status["status"] == "stopped"
    assert status["resource_release"] == "passed"
    assert status["server_cleanup_complete"] is True
    assert status["performance_acceptance"] == "not_run"


def test_unexpected_gate_traceback_is_live_and_durable(monkeypatch, tmp_path, capsys):
    path, _ = _target(tmp_path)
    logs = tmp_path / "logs"
    monkeypatch.setattr(aisbench_serve, "run_phase", _fake_phase)
    monkeypatch.setattr(aisbench_serve, "_accuracy_gates",
                        lambda *_args: (_ for _ in ()).throw(ValueError("unexpected precision gate crash")))
    assert aisbench_serve.run(path, logs) == 1
    assert "Traceback (most recent call last)" in capsys.readouterr().err
    assert "unexpected precision gate crash" in (logs / "operator-accuracy-gate.log").read_text()
