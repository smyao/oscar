"""Archive #130/#131/#159: one diagnostic request, explicit run, bounded deadline."""
import json

import pytest

from tools import single_request_probe as probe


def test_default_plan_does_not_send_requests_or_create_files(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(probe, "isolated_http", lambda *args: pytest.fail("plan cannot send HTTP"))
    output = tmp_path / "unused"
    assert probe.main(["--output", str(output)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["generation_requests"] == 1 and plan["prompt_tokens"] == 24616
    assert plan["output_tokens"] == 16 and not output.exists()


def test_run_sends_one_exact_cold_request_and_records_completion(monkeypatch, tmp_path):
    calls = []
    def http(directory, name, task):
        calls.append((name, task))
        assert 0 < task["timeout"] <= 180
        if name == "tokenize":
            return {"body": {"tokens": [3, 5, 8]}}
        assert task["stream"] and task["payload"]["ignore_eos"]
        assert len(task["payload"]["prompt"]) == 24616
        assert task["payload"]["max_tokens"] == 16 and task["payload"]["cache_salt"]
        return {"status": "completed", "sse_done": True}
    monkeypatch.setattr(probe, "isolated_http", http)
    out = tmp_path / "request"
    assert probe.main(["--run", "--output", str(out)]) == 0
    assert [name for name, _ in calls] == ["tokenize", "request"]
    assert json.loads((out / "summary.json").read_text())["status"] == "completed"


def test_failure_keeps_report_and_never_retries(monkeypatch, tmp_path):
    calls = []
    def http(directory, name, task):
        calls.append(name)
        if name == "tokenize":
            return {"body": {"tokens": [3]}}
        raise RuntimeError("service stream error: EngineDeadError")
    monkeypatch.setattr(probe, "isolated_http", http)
    out = tmp_path / "failed"
    assert probe.main(["--run", "--output", str(out)]) == 1
    report = json.loads((out / "summary.json").read_text())
    assert report["status"] == "failed" and "EngineDeadError" in report["error"]
    assert calls == ["tokenize", "request"]
