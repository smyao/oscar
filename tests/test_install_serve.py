"""Archive #94/#123/#125/#148/#150: direct install/serve must use the selected route and devices.

Host orchestration tests only; no model, NPU, installation or probes run here.
"""
import json

import pytest

from tools import install_serve
from tools.phase import PhaseResult


def config_file(tmp_path, enabled=False):
    config = json.loads((install_serve.ROOT / "configs/target.json").read_text())
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"]}))
    config["model"] = str(model)
    config["experimental_history_reuse"] = enabled
    path = tmp_path / "target.json"
    path.write_text(json.dumps(config))
    return path


@pytest.mark.parametrize("rear_cards", [False, True])
def test_candidate_plan_is_read_only_and_has_no_probes(tmp_path, capsys, rear_cards):
    config = config_file(tmp_path)
    logs = tmp_path / "planned"
    assert install_serve.main(["--config", str(config), "--log-dir", str(logs),
                               "--variant", "candidate", "--plan"] +
                              (["--rear-cards"] if rear_cards else [])) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["variant"] == "candidate"
    assert plan["optimizations"]["whole_prefill"] is False
    assert all(plan["optimizations"][name] for name in (
        "history_cluster4", "later_mtp_q1", "fast_unpack", "mixed_cv", "striped_cache",
        "current_only", "first_mtp_current_fia", "compact_later_mtp", "mixed_decode_split", "decode_bundle"))
    assert plan["runtime_selection"]["token_budget_source"] == "explicit"
    assert plan["runtime_selection"]["max_num_batched_tokens"] == 16384
    assert plan["runtime_selection"]["scheduler_cls"] is None
    assert [name for name, _ in plan["stages"]] == list(install_serve.BUILD_PHASES)
    assert plan["probes"] == "none" and not logs.exists()
    assert plan["serve"][-1] == str(logs / "effective-target.json")
    assert plan["stages"][-1][1][-1] == plan["serve"][-1]
    assert plan["target_devices"] == [4, 5, 6, 7]
    assert plan["port"] == 7878


@pytest.mark.parametrize("variant,original,enabled", [
    ("candidate", False, True), ("baseline", True, False), (None, True, True), (None, False, True),
])
@pytest.mark.parametrize("rear_cards", [False, True])
def test_all_phases_and_service_share_selected_config_without_mutating_original(
        monkeypatch, tmp_path, variant, original, enabled, rear_cards):
    path = config_file(tmp_path, original)
    before = path.read_bytes()
    logs = tmp_path / "run"
    devices = [4, 5, 6, 7]
    port = 7878
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "8,9,10,11")
    seen = []
    def run_phase(name, command, *, env, **kwargs):
        seen.append(name)
        selected = json.loads(open(env["OSCAR_TARGET_CONFIG"]).read())
        expected_timeout = (86400 if name == "prepare-rotations" and
                            selected.get("rotation_refinement") == "optr" else
                            selected["phase_timeout_seconds"])
        assert kwargs["timeout"] == expected_timeout
        assert selected.get("rotation_refinement") == (None if variant == "baseline" else "optr")
        assert selected["experimental_history_reuse"] is enabled
        assert selected.get("experimental_fast_unpack", False) is (variant != "baseline")
        assert selected.get("experimental_mixed_cv", False) is (variant != "baseline")
        assert selected.get("experimental_striped_cache", False) is (variant != "baseline")
        assert selected["use_native_token_budget"] is False
        assert selected["max_num_batched_tokens"] == 16384 and "scheduler_cls" not in selected
        assert not selected.get("experimental_whole_prefill", False)
        assert not selected.get("experimental_bounded_whole_prefill", False)
        assert env["OSCAR_ENABLED"] == "1"
        assert env["ASCEND_RT_VISIBLE_DEVICES"] == ",".join(map(str, devices))
        assert selected["devices"] == devices and selected["port"] == port
        if "--config" in command:
            assert command[command.index("--config") + 1] == env["OSCAR_TARGET_CONFIG"]
        return PhaseResult(name, command, 0, 0.0, False, False, str(logs / name), True)
    def launch(command, *, env, **kwargs):
        from tools.target_cli import serve_argv
        assert seen == list(install_serve.BUILD_PHASES)
        assert command[-1] == env["OSCAR_TARGET_CONFIG"]
        selected = json.loads(open(command[-1]).read())
        assert selected.get("rotation_refinement") == (None if variant == "baseline" else "optr")
        assert selected["experimental_history_reuse"] is enabled
        assert selected.get("experimental_striped_cache", False) is (variant != "baseline")
        assert selected["devices"] == devices
        service_args = serve_argv(selected)
        assert service_args[service_args.index("--max-num-batched-tokens") + 1] == "16384"
        assert "--scheduler-cls" not in service_args
        assert service_args[service_args.index("--port") + 1] == str(port)
        assert kwargs['grace'] == selected['shutdown_timeout_seconds']
        assert 'timeout' not in kwargs
        return PhaseResult('serve', command, 0, 0., False, False, str(logs/'serve.log'), True)
    monkeypatch.setattr(install_serve, "run_phase", run_phase)
    monkeypatch.setattr(install_serve, "run_service", launch)
    args = ["--config", str(path), "--log-dir", str(logs)]
    if variant:
        args += ["--variant", variant]
    if rear_cards:
        args += ["--rear-cards"]
    assert install_serve.main(args) == 0
    assert path.read_bytes() == before
    status = json.loads((logs / "status.json").read_text())
    assert status["variant"] == ("candidate" if enabled else "baseline")
    assert status["probes"] == "none"
    assert status['status'] == 'stopped' and status['service_returncode'] == 0
    assert status['cleanup_complete'] is True


def test_build_failure_preserves_rc_and_prevents_exec(monkeypatch, tmp_path):
    path = config_file(tmp_path)
    monkeypatch.setattr(install_serve, "run_phase", lambda name, command, **kwargs:
        PhaseResult(name, command, 7, 0.0, False, False, "failure.log", True))
    monkeypatch.setattr(install_serve, "run_service", lambda *args, **kwargs:
                        pytest.fail("must not start service after failed installation"))
    logs = tmp_path / "failed"
    assert install_serve.main(["--config", str(path), "--log-dir", str(logs),
                               "--variant", "candidate"]) == 7
    assert json.loads((logs / "status.json").read_text())["status"] == "failed"


def test_device_lease_covers_build_service_and_failure_diagnostics(monkeypatch, tmp_path):
    from tools.formal_service import device_lease
    path = config_file(tmp_path)
    logs = tmp_path/'run'
    monkeypatch.setattr(install_serve, 'ROOT', tmp_path)
    events = []
    def locked():
        with pytest.raises(RuntimeError, match='already has an active direct launcher'):
            with device_lease(tmp_path, [4,5,6,7], tmp_path/'duplicate'):
                pytest.fail('must retain lease through teardown')
    def phase(name, command, **kwargs):
        locked(); events.append(name)
        return PhaseResult(name, command, 0, 0., False, False, str(logs/name), True)
    def launch(command, **kwargs):
        locked(); events.append('serve')
        return PhaseResult('serve', command, 1, 0., False, False, str(logs/'serve.log'), True)
    def diagnose(*args, **kwargs):
        locked(); events.append('diagnose')
        raise OSError('diagnostic unavailable')
    monkeypatch.setattr(install_serve, 'run_phase', phase)
    monkeypatch.setattr(install_serve, 'run_service', launch)
    monkeypatch.setattr(install_serve, 'memory_failure_diagnostics', diagnose)
    assert install_serve.main(['--config',str(path),'--log-dir',str(logs)]) == 1
    assert events == [*install_serve.BUILD_PHASES, 'serve', 'diagnose']
    state = json.loads((logs/'status.json').read_text())
    assert state['status'] == 'failed' and state['failed_phase'] == 'serve'
    assert state['service_returncode'] == 1 and state['cleanup_complete']
    assert 'diagnostic unavailable' in state['diagnostic_error']
    with device_lease(tmp_path, [4,5,6,7], tmp_path/'next'):
        pass


def test_active_direct_launcher_blocks_duplicate_before_build(monkeypatch, tmp_path):
    from tools.formal_service import device_lease
    path = config_file(tmp_path)
    logs = tmp_path/'duplicate'
    monkeypatch.setattr(install_serve, 'ROOT', tmp_path)
    monkeypatch.setattr(install_serve, 'run_phase', lambda *args, **kwargs: pytest.fail('duplicate must not build/load'))
    with device_lease(tmp_path, [4,5,6,7], tmp_path/'active'):
        with pytest.raises(RuntimeError, match='physical NPU 4 already has an active direct launcher'):
            install_serve.main(['--config',str(path),'--log-dir',str(logs)])
    state = json.loads((logs/'status.json').read_text())
    assert state['status'] == 'failed' and not state['phases']
