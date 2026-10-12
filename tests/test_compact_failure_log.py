import json

from tools.compact_failure_log import compact


def test_first_fault_survives_noisy_tail_and_size_cap(tmp_path):
    (tmp_path / "service").mkdir()
    (tmp_path / "service/server.log").write_text(
        'OSCAR_FIRST_ERROR {"code":5,"pid":123}\n' + 'noise\n' * 10000 + 'RuntimeError: 507035\n')
    text = compact(tmp_path)
    assert '"code":5' in text and '507035' in text
    assert len(text) <= 6000
    assert 'service/server.log:1:' in text


def test_report_status_and_plog_without_request_content(tmp_path):
    (tmp_path / "service-report.json").write_text(json.dumps({
        "status": "failed", "error": "request timeout", "resource_release": "passed",
        "requests": [{"prompt": "SECRET_PAYLOAD"}],
        "cann_plog": {"excerpts": [{"path": "/plog/rank2", "text": "device error 507035"}]}}))
    text = compact(tmp_path)
    assert 'request timeout' in text and '/plog/rank2' in text
    assert 'SECRET_PAYLOAD' not in text


def test_incomplete_run_is_not_reported_as_success(tmp_path):
    (tmp_path / "diagnostic-report.json").write_text('{broken')
    text = compact(tmp_path, 2000)
    assert 'cannot read' in text and 'missing' in text
    assert 'NOT a pass' in text and len(text) <= 2000


def test_repeated_status_and_layout_are_preserved(tmp_path):
    (tmp_path / "service-report.json").write_text(json.dumps({"status": "passed", "resource_release": "passed"}))
    (tmp_path / "diagnostic-report.json").write_text(json.dumps({"status": "passed"}))
    trace = tmp_path / "service/trace-abc"
    trace.mkdir(parents=True)
    (trace / "worker-42.jsonl").write_text(json.dumps({"event": "cache_layout", "rank": None, "pid": 42}) + '\n')
    (tmp_path / "terminal.log").write_text('OSCAR_FIRST_ERROR {"context":{"layer":"synthetic"},"code":7}\n')
    text = compact(tmp_path)
    assert 'diagnostic-report.json.status: "passed"' in text
    assert 'resource_release: "passed"' in text
    assert '"rank": null' in text and 'worker-42.jsonl' in text
    assert '"code":7' not in text
