"""Archive #94/#95/#125/#156/#157: local replay failures must stop the run."""
import json

import pytest

from tools import local_simulation
from tools.phase import PhaseResult


@pytest.mark.parametrize("fault,expected", [("child_exit", 7), ("cleanup", 1),
    ("missing_junit", 1), ("junit_failure", 1), ("only_skips", 1)])
def test_local_replay_preserves_failure_and_stops(tmp_path, monkeypatch, fault, expected):
    calls = []

    def phase(name, command, *, cwd, log_dir, timeout, env):
        calls.append(name)
        if fault != "missing_junit":
            (log_dir / f"{name}-junit.xml").write_text(
                f'<testsuite tests="1" skipped="{int(fault == "only_skips")}" '
                f'failures="{int(fault == "junit_failure")}" errors="0"/>')
        return PhaseResult(name, command, 7 if fault == "child_exit" else 0, 0.01,
                           False, False, str(log_dir / f"{name}.log"), fault != "cleanup")

    monkeypatch.setattr(local_simulation, "run_phase", phase)
    assert local_simulation.run(tmp_path, 3) == expected
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["exit_code"] == expected and report["status"] == "failed"
    assert calls == ["contracts"]
    assert "NPU_completion" in report["not_verified"]
