"""Archive #94/#95/#130/#133/#156: bounded read-only evidence, no NPU claims."""
import hashlib
import json
import os

import pytest

from tools import collect_performance_evidence as evidence
from tools import collect_startup_evidence as startup


def test_middle_engine_spec_and_native_budget_are_preserved(tmp_path):
    path = tmp_path / "server.log"
    signals = [
        "Chunked prefill is enabled with max_num_batched_tokens=2048.",
        "Engine 000: Avg prompt throughput: 55 tokens/s, Running: 2 reqs, Waiting: 30 reqs",
        "SpecDecoding metrics: Mean acceptance length: 1.2",
        "│ Benchmark Duration │ 1800 │",
    ]
    path.write_text("noise\n" * 200 + "\n".join(signals) + "\n" + "noise\n" * 200)
    result = evidence.scan(path)
    assert result["lines"] == 404
    for name in ("engine", "spec", "capacity", "benchmark"):
        assert result["log_signals"][name]["matched"] >= 1
    assert all(signal in json.dumps(result, ensure_ascii=False) for signal in signals)


def test_throttled_samples_bounded_and_no_fabricated_step_duration(tmp_path):
    path = tmp_path / "worker-123.jsonl"
    path.write_text("\n".join(json.dumps({"event": "prefix_chunk", "computed_tokens": i,
        "scheduled_tokens": 768, "proposed_tokens": 2048, "api_key": "SECRET"}) for i in range(1000)))
    result = evidence.scan(path)
    sample = result["events"]["prefix_chunk"]
    assert sample["matched"] == 1000 and sample["omitted"] == 984
    assert sample["first_then_last"][-1]["computed_tokens"] == 999
    assert result["event_sample_counts"] == {"prefix_chunk": 1000}
    assert "SECRET" not in json.dumps(result)
    assert "step_count" not in result and "device_time" not in result


def test_huge_line_and_truncated_json_are_explicit(tmp_path):
    path = tmp_path / "worker-1.jsonl"
    raw = b"x" * (2 * 1024 * 1024) + b'\n{"event":"prefix_chunk",\n'
    path.write_bytes(raw)
    result = evidence.scan(path)
    assert result["sha256_of_scanned_bytes"] == hashlib.sha256(raw).hexdigest()
    assert result["oversized_lines_prefix_only"] == result["malformed_json_lines"] == 1
    assert result["lines"] == 2


def test_default_discovery_reads_only_selected_run_and_keeps_missing_console_explicit(tmp_path, monkeypatch):
    for name, mtime in (("old", 10), ("new", 20)):
        run = tmp_path / "logs" / name
        (run / "trace").mkdir(parents=True)
        (run / "status.json").write_text('{"status":"serving"}')
        os.utime(run / "status.json", (mtime, mtime))
        (run / "effective-target.json").write_text('{"devices":[4,5,6,7],"port":7878,"secret":"hidden"}')
        (run / "trace" / "worker-123.jsonl").write_text(json.dumps({"event":"attention_progress",
            "cv_operator": name, "requests":32}))
    (tmp_path / "logs" / "new" / "private.txt").write_text("NEVER_READ")
    monkeypatch.setattr(evidence, "ROOT", tmp_path)
    monkeypatch.setattr(evidence, "snapshot", lambda *_: pytest.fail("offline must not execute commands"))
    output = tmp_path / "key.txt"
    assert evidence.main(["--output", str(output)]) == 0
    report = output.read_text()
    assert '"cv_operator": "new"' in report and '"cv_operator": "old"' not in report
    assert "hidden" not in report and "NEVER_READ" not in report
    assert '"console_files_found": []' in report
    assert output.stat().st_size <= 256 * 1024
    with pytest.raises(SystemExit):
        evidence.main(["--output", str(output)])


def test_empty_run_is_failure_with_evidence_file(tmp_path):
    output = tmp_path / "key.txt"
    assert evidence.main(["--run-dir", str(tmp_path), "--output", str(output)]) == 1
    assert '"input_files": []' in output.read_text()


def test_snapshot_env_and_fd_links_are_read_without_opening_output(tmp_path):
    proc = tmp_path / "123"
    (proc / "fd").mkdir(parents=True)
    (proc / "environ").write_bytes(b"OSCAR_GUARD_HOST_SYNC=1\0OSCAR_DEBUG_SYNC=1\0API_KEY=SECRET\0")
    (proc / "fd" / "1").symlink_to("/dev/pts/10")
    (proc / "fd" / "2").symlink_to("/missing/console.log")
    result = startup.proc_identity(123, tmp_path)
    assert result["stdout"] == "/dev/pts/10" and result["stderr"] == "/missing/console.log"
    assert result["environment_allowlist"] == {"OSCAR_GUARD_HOST_SYNC":"1", "OSCAR_DEBUG_SYNC":"1"}
    assert "SECRET" not in json.dumps(result)
