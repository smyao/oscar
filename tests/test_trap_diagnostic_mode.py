"""Archive #157-159: optional eager attribution is not a performance run."""
import json
from pathlib import Path

import pytest

from tools import install_serve, observe_serve, service_probe, target_cli
from tools.phase import PhaseResult


def config():
    return json.loads((target_cli.ROOT / "configs/target.json").read_text())


@pytest.mark.parametrize("enabled", [False, True])
def test_shared_target_argv_env_and_default_invariants(enabled):
    selected = config()
    if enabled:
        selected["diagnose_guard"] = True
    argv = target_cli.serve_argv(selected)
    env = target_cli.target_env(selected, base={})
    assert ("--enforce-eager" in argv) is enabled
    assert env.get("OSCAR_GUARD_HOST_SYNC") == ("1" if enabled else None)
    assert "--enable-prefix-caching" in argv
    assert "--max-num-batched-tokens" not in argv and "--scheduler-cls" not in argv
    assert env["ASCEND_RT_VISIBLE_DEVICES"] == "0,1,2,3"
    assert argv[argv.index("--port") + 1] == "8989"
    assert json.loads(argv[argv.index("--speculative_config") + 1])["enforce_eager"] is True


@pytest.mark.parametrize("invalid", ["true", 1, None])
def test_diagnostic_config_is_strictly_typed(invalid):
    selected = {**config(), "diagnose_guard": invalid}
    for helper in (target_cli.serve_argv, target_cli.target_env):
        with pytest.raises(ValueError, match="diagnose_guard"):
            helper(selected)


@pytest.mark.parametrize("entry", [install_serve, observe_serve])
def test_both_optional_plans_select_same_service_mode(entry, tmp_path, capsys):
    assert entry.main(["--variant", "candidate", "--diagnose-trap", "--plan",
                       "--log-dir", str(tmp_path / "uncreated")]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["diagnose_trap"] is True
    assert "--enforce-eager" in plan["vllm_argv"]
    assert plan["runtime_selection"]["token_budget_source"] == "native"
    assert not (tmp_path / "uncreated").exists()


def test_direct_install_exec_receives_diagnostic_config_and_env(monkeypatch, tmp_path):
    source = tmp_path / "target.json"
    source.write_text(json.dumps(config()))
    original = source.read_bytes()
    seen = []

    def build(name, command, *, env, **kwargs):
        selected = json.loads(Path(env["OSCAR_TARGET_CONFIG"]).read_text())
        assert selected["diagnose_guard"] is True
        assert env["OSCAR_GUARD_HOST_SYNC"] == "1"
        seen.append(name)
        return PhaseResult(name, command, 0, 0.0, False, False, str(tmp_path / name), True)

    class Served(Exception):
        pass

    def launch(command, *, env, **kwargs):
        selected = json.loads(Path(command[-1]).read_text())
        assert env["OSCAR_TARGET_CONFIG"] == command[-1]
        assert env["OSCAR_GUARD_HOST_SYNC"] == "1"
        assert "--enforce-eager" in target_cli.serve_argv(selected)
        assert selected["experimental_mdl_decode"] is True
        assert seen == list(install_serve.BUILD_PHASES)
        raise Served

    monkeypatch.setattr(install_serve, "run_phase", build)
    monkeypatch.setattr(install_serve, "run_service", launch)
    with pytest.raises(Served):
        install_serve.main(["--config", str(source), "--log-dir", str(tmp_path / "run"),
                            "--variant", "candidate", "--diagnose-trap"])
    assert source.read_bytes() == original
    status = json.loads((tmp_path / "run/status.json").read_text())
    assert status["performance_acceptance"] == "invalid_diagnostic"


def test_performance_gate_rejects_environment_or_saved_diagnostic_config(monkeypatch):
    for key in ("OSCAR_DEBUG_SYNC", "OSCAR_TIMING", "OSCAR_PROFILER", "OSCAR_GUARD_HOST_SYNC"):
        monkeypatch.delenv(key, raising=False)
    service_probe._require_uninstrumented_synthetic(config())
    with pytest.raises(service_probe.ServiceProbeError, match="diagnose_guard"):
        service_probe._require_uninstrumented_synthetic({**config(), "diagnose_guard": True})
    monkeypatch.setenv("OSCAR_GUARD_HOST_SYNC", "1")
    with pytest.raises(service_probe.ServiceProbeError, match="OSCAR_GUARD_HOST_SYNC"):
        service_probe._require_uninstrumented_synthetic(config())


def test_existing_service_debug_modes_still_reach_service_preparation(monkeypatch, tmp_path):
    for key in ("OSCAR_DEBUG_SYNC", "OSCAR_TIMING", "OSCAR_PROFILER"):
        monkeypatch.setenv(key, "1")
    monkeypatch.delenv("OSCAR_GUARD_HOST_SYNC", raising=False)
    source = tmp_path / "target.json"
    source.write_text(json.dumps(config()))
    reached = []

    def stop_before_device(selected):
        reached.append(selected)
        raise RuntimeError("host stop after legacy debug configuration admitted")

    monkeypatch.setattr(service_probe, "target_env", stop_before_device)
    monkeypatch.setattr(service_probe, "attach_plog", lambda *args, **kwargs: None)
    report = service_probe.run_service(source, output=tmp_path / "report.json", log_dir=tmp_path / "run")
    assert len(reached) == 1
    assert "legacy debug configuration admitted" in report["error"]
    # Paired measurement remains stricter than the older diagnostic service.
    with pytest.raises(service_probe.ServiceProbeError, match="OSCAR_DEBUG_SYNC"):
        service_probe._require_uninstrumented_synthetic(config())
