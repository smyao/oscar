# 档案 #74–76/#94–95/#117：环境审计不阻塞；真实探针失败保留退出码，安装使用工程绝对路径。
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
PROBES = ("probe-ops", "probe-cv", "service-probe")
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
        assert {"ascend", "oscar_ascend", "another_plugin"} <= set(env["VLLM_PLUGINS"].split(","))
        if name == "service-probe":
            (log_dir / "service-probe-report.json").write_text(json.dumps({"status": "passed", "resource_release": "passed"}))
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
    monkeypatch.setattr(service_probe, "main", serve)
    with patch.dict(os.environ):
        assert deploy.main() == 0
    names = [name for name, _ in calls]
    assert set(names).isdisjoint(AUDITS)
    assert [name for name in names if name in PROBES] == list(PROBES)
    assert names.index("install-plugin") < names.index("build-ops") < names.index("probe-ops")
    assert names.index("prepare-rotations") < names.index("service-probe")
    commands = dict(calls)
    build = commands["build-ops"]
    assert build[build.index("--soc") + 1] == config["soc_version"]
    install = commands["install-plugin"]
    assert Path(install[install.index("-e") + 1]) == ROOT
    assert "--no-deps" in install
    assert len(served) == 1
    assert len(json.loads((logs / "status.json").read_text())["phases"]) == len(calls)


@pytest.mark.parametrize("failure", PROBES)
def test_any_probe_failure_stops_formal_serve_and_keeps_original_code(monkeypatch, tmp_path, failure):
    _, _, logs = _configure(monkeypatch, tmp_path)
    calls = []

    def phase(name, command, *, log_dir, **kwargs):
        calls.append(name)
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
