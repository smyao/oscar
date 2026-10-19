"""Archive #94/#125/#148/#150: direct install/serve must use the selected route.

Host orchestration tests only; no model, NPU, installation or probes run here.
"""
import json

import pytest

from tools import install_serve
from tools.phase import PhaseResult


def config_file(tmp_path, enabled=False):
    config = json.loads((install_serve.ROOT / "configs/target.json").read_text())
    # Keep an explicit alternate config to exercise --rear-cards overrides;
    # the checked-in default is covered by test_serving_variants/deploy_launcher.
    config.update(devices=[0, 1, 2, 3], port=8989, use_native_token_budget=True)
    config.pop("max_num_batched_tokens", None)
    if enabled:
        config["experimental_history_reuse"] = True
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
    assert plan["runtime_selection"]["token_budget_source"] == "native"
    assert plan["runtime_selection"]["max_num_batched_tokens"] is None
    assert plan["runtime_selection"]["scheduler_cls"] is None
    assert [name for name, _ in plan["stages"]] == list(install_serve.BUILD_PHASES)
    assert plan["probes"] == "none" and not logs.exists()
    assert plan["serve"][-1] == str(logs / "effective-target.json")
    assert plan["stages"][-1][1][-1] == plan["serve"][-1]
    assert plan["target_devices"] == ([4, 5, 6, 7] if rear_cards else [0, 1, 2, 3])
    assert plan["port"] == (7878 if rear_cards else 8989)


@pytest.mark.parametrize("variant,original,enabled", [
    ("candidate", False, True), ("baseline", True, False), (None, True, True), (None, False, True),
])
@pytest.mark.parametrize("rear_cards", [False, True])
def test_all_phases_and_supervised_serve_share_selected_config_without_mutating_original(
        monkeypatch, tmp_path, variant, original, enabled, rear_cards):
    path = config_file(tmp_path, original)
    before = path.read_bytes()
    logs = tmp_path / "run"
    devices = [4, 5, 6, 7] if rear_cards else [0, 1, 2, 3]
    port = 7878 if rear_cards else 8989
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "8,9,10,11")
    seen = []
    def run_phase(name, command, *, env, **kwargs):
        seen.append(name)
        selected = json.loads(open(env["OSCAR_TARGET_CONFIG"]).read())
        assert selected["experimental_history_reuse"] is enabled
        assert selected.get("experimental_fast_unpack", False) is True
        assert selected.get("experimental_mixed_cv", False) is True
        assert selected.get("experimental_striped_cache", False) is True
        assert selected["use_native_token_budget"] is True
        assert "max_num_batched_tokens" not in selected and "scheduler_cls" not in selected
        assert not selected.get("experimental_whole_prefill", False)
        assert not selected.get("experimental_bounded_whole_prefill", False)
        assert env["OSCAR_ENABLED"] == "1"
        assert env["ASCEND_RT_VISIBLE_DEVICES"] == ",".join(map(str, devices))
        assert selected["devices"] == devices and selected["port"] == port
        if "--config" in command:
            assert command[command.index("--config") + 1] == env["OSCAR_TARGET_CONFIG"]
        return PhaseResult(name, command, 0, 0.0, False, False, str(logs / name), True)
    class Served(Exception):
        pass
    def launch(config, config_path, *, log_dir, env):
        command = ["--config", str(config_path)]
        from tools.target_cli import serve_argv
        assert seen == list(install_serve.BUILD_PHASES)
        assert command[-1] == env["OSCAR_TARGET_CONFIG"]
        selected = json.loads(open(command[-1]).read())
        assert selected["experimental_history_reuse"] is enabled
        assert selected.get("experimental_striped_cache", False) is True
        assert selected["devices"] == devices
        service_args = serve_argv(selected)
        assert "--max-num-batched-tokens" not in service_args
        assert "--scheduler-cls" not in service_args
        assert service_args[service_args.index("--port") + 1] == str(port)
        raise Served
    monkeypatch.setattr(install_serve, "run_phase", run_phase)
    monkeypatch.setattr(install_serve, "run_service", launch)
    args = ["--config", str(path), "--log-dir", str(logs)]
    if variant:
        args += ["--variant", variant]
    if rear_cards:
        args += ["--rear-cards"]
    if variant == "baseline":
        with pytest.raises(ValueError, match="retired"):
            install_serve.main(args)
        assert not seen and path.read_bytes() == before
        return
    with pytest.raises(Served):
        install_serve.main(args)
    assert path.read_bytes() == before
    status = json.loads((logs / "status.json").read_text())
    assert status["variant"] == ("candidate" if enabled else "baseline")
    assert status["probes"] == "none"


def test_build_failure_preserves_rc_and_prevents_serve(monkeypatch, tmp_path):
    path = config_file(tmp_path)
    monkeypatch.setattr(install_serve, "run_phase", lambda name, command, **kwargs:
        PhaseResult(name, command, 7, 0.0, False, False, "failure.log", True))
    monkeypatch.setattr(install_serve, "run_service", lambda *args, **kwargs:
                        pytest.fail("must not start service after failed installation"))
    logs = tmp_path / "failed"
    assert install_serve.main(["--config", str(path), "--log-dir", str(logs),
                               "--variant", "candidate"]) == 7
    assert json.loads((logs / "status.json").read_text())["status"] == "failed"
