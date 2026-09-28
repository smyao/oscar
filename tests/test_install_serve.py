"""Archive #94/#125/#148/#150: direct install/serve must use the selected route.

Host orchestration tests only; no model, NPU, installation or probes run here.
"""
import json

import pytest

from tools import install_serve
from tools.phase import PhaseResult


def config_file(tmp_path, enabled=False):
    config = json.loads((install_serve.ROOT / "configs/target.json").read_text())
    config["experimental_history_reuse"] = enabled
    path = tmp_path / "target.json"
    path.write_text(json.dumps(config))
    return path


def test_candidate_plan_is_read_only_and_has_no_probes(tmp_path, capsys):
    config = config_file(tmp_path)
    logs = tmp_path / "planned"
    assert install_serve.main(["--config", str(config), "--log-dir", str(logs),
                               "--variant", "candidate", "--plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["variant"] == "candidate"
    assert all(plan["optimizations"].values())
    assert [name for name, _ in plan["stages"]] == list(install_serve.BUILD_PHASES)
    assert plan["probes"] == "none" and not logs.exists()
    assert plan["serve"][-1] == str(logs / "effective-target.json")
    assert plan["stages"][-1][1][-1] == plan["serve"][-1]


@pytest.mark.parametrize("variant,original,enabled", [
    ("candidate", False, True), ("baseline", True, False), (None, True, True), (None, False, False),
])
def test_all_phases_and_exec_share_selected_config_without_mutating_original(
        monkeypatch, tmp_path, variant, original, enabled):
    path = config_file(tmp_path, original)
    before = path.read_bytes()
    logs = tmp_path / "run"
    seen = []
    def run_phase(name, command, *, env, **kwargs):
        seen.append(name)
        selected = json.loads(open(env["OSCAR_TARGET_CONFIG"]).read())
        assert selected["experimental_history_reuse"] is enabled
        assert env["OSCAR_ENABLED"] == "1"
        assert env["ASCEND_RT_VISIBLE_DEVICES"] == "0,1,2,3"
        if "--config" in command:
            assert command[command.index("--config") + 1] == env["OSCAR_TARGET_CONFIG"]
        return PhaseResult(name, command, 0, 0.0, False, False, str(logs / name), True)
    class Served(Exception):
        pass
    def launch(executable, command, env):
        assert seen == list(install_serve.BUILD_PHASES)
        assert command[-1] == env["OSCAR_TARGET_CONFIG"]
        assert json.loads(open(command[-1]).read())["experimental_history_reuse"] is enabled
        raise Served
    monkeypatch.setattr(install_serve, "run_phase", run_phase)
    monkeypatch.setattr(install_serve.os, "execvpe", launch)
    args = ["--config", str(path), "--log-dir", str(logs)]
    if variant:
        args += ["--variant", variant]
    with pytest.raises(Served):
        install_serve.main(args)
    assert path.read_bytes() == before
    status = json.loads((logs / "status.json").read_text())
    assert status["variant"] == ("candidate" if enabled else "baseline")
    assert status["probes"] == "none"


def test_build_failure_preserves_rc_and_prevents_exec(monkeypatch, tmp_path):
    path = config_file(tmp_path)
    monkeypatch.setattr(install_serve, "run_phase", lambda name, command, **kwargs:
        PhaseResult(name, command, 7, 0.0, False, False, "failure.log", True))
    monkeypatch.setattr(install_serve.os, "execvpe", lambda *args:
                        pytest.fail("must not start service after failed installation"))
    logs = tmp_path / "failed"
    assert install_serve.main(["--config", str(path), "--log-dir", str(logs),
                               "--variant", "candidate"]) == 7
    assert json.loads((logs / "status.json").read_text())["status"] == "failed"
