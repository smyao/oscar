# Archive #129: stall collection observes only a selected local run; it never sends signals.
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import inspect_run


def test_inspect_collects_existing_run_without_device_or_signal_calls(tmp_path, monkeypatch):
    run = tmp_path / "logs/20260922T053137.509986Z"
    server = run / "service-probe"
    trace = server / "trace-test"
    trace.mkdir(parents=True)
    (server / "server_lifecycle.json").write_text(json.dumps({"pid": 123, "owned_pids": [123, 456]}))
    (server / "server.log").write_text("128 completed\n16384 waiting\n")
    (trace / "phase-456.json").write_text(json.dumps({"pid": 456, "phase": "fia", "state": "waiting_for_device"}))
    (trace / "worker-456.jsonl").write_text('{"event":"attention_dispatched"}\n')
    calls = []
    monkeypatch.setattr(inspect_run.OwnedProcessGroup, "refresh", lambda self, **kwargs: calls.append("owned group"))
    monkeypatch.setattr(inspect_run.subprocess, "run", lambda command, **kwargs:
        (calls.append(command) or SimpleNamespace(stdout="456 123 123 S 00:05 python\n")))
    def plog(report, **kwargs):
        assert kwargs["owned_pids"] == {123, 456}
        report["cann_plog"] = {"status": "fixture"}
    monkeypatch.setattr(inspect_run, "attach_plog", plog)
    report = inspect_run.inspect(run, root=tmp_path)
    assert report["read_only"] and report["device_completion"] == "not_established"
    assert report["worker_progress"][0]["state"] == "waiting_for_device"
    assert "16384 waiting" in report["server_log_tail"]
    assert calls[1][0] == "ps" and calls[1][-1] == "123,456"


def test_inspect_rejects_other_project_paths(tmp_path):
    with pytest.raises(ValueError, match="this project's logs"):
        inspect_run.inspect(tmp_path, root=tmp_path)


def test_tail_is_bounded(tmp_path):
    path = tmp_path / "server.log"
    path.write_text("old" * 100 + "LAST")
    assert inspect_run.tail(path, 4) == "LAST"
