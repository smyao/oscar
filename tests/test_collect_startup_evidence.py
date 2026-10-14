"""Archive #94/#95/#123/#158: synthetic bounded evidence, never an NPU pass."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from tools import collect_startup_evidence as evidence


ERROR = ("(Worker pid=112676) ValueError: Free memory on device (0.64/29.49 GiB) "
         "on startup is less than desired GPU memory utilization (0.9, 26.54 GiB).")


def test_stream_keeps_early_cause_before_late_cascade_and_counts_all_input(tmp_path):
    source = tmp_path / "console.log"
    raw = ("startup\r\n\x1b[31m" + ERROR + "\x1b[0m\r\n" + "heartbeat\n" * 20000 +
           "RuntimeError: Engine core initialization failed\n").encode()
    source.write_bytes(raw)
    report = evidence.scan_log(source)
    assert report["sha256"] == hashlib.sha256(raw).hexdigest()
    assert report["bytes_scanned"] == len(raw)
    assert report["lines"] == 20003 and report["first_fault_line"] == 2
    assert report["pids_seen"] == [112676]
    assert report["memory_gates"] == [{"line": 2, "pid": "112676", "free_gib": 0.64,
                                       "total_gib": 29.49, "utilization": 0.9, "required_gib": 26.54}]
    assert any(ERROR in row for row in report["first_fault_context"])
    assert "Engine core" in report["tail"][-1]
    assert len(report["first_fault_context"]) <= 112
    assert "\x1b" not in json.dumps(report, ensure_ascii=False)


def test_oversized_line_is_drained_with_exact_hash_and_next_line_number(tmp_path):
    source = tmp_path / "long.log"
    raw = b"x" * (2 * 1024 * 1024) + b"\n" + ERROR.encode() + b"\n"
    source.write_bytes(raw)
    report = evidence.scan_log(source)
    assert report["lines"] == 2 and report["first_fault_line"] == 2
    assert report["oversized_lines_prefix_only"] == 1
    assert report["sha256"] == hashlib.sha256(raw).hexdigest()
    assert report["bytes_scanned"] == len(raw)


def test_proc_reads_only_allowed_environment_and_records_namespace(tmp_path):
    proc = tmp_path / "123"
    proc.mkdir()
    (proc / "environ").write_bytes(b"API_KEY=hidden\0ASCEND_RT_VISIBLE_DEVICES=4,5,6,7\0")
    (proc / "comm").write_text("VLLM::Worker\n")
    (proc / "status").write_text("Pid:\t123\nNSpid:\t987 123\nVmRSS:\t100\n")
    (proc / "cwd").symlink_to(tmp_path, target_is_directory=True)
    result = evidence.proc_identity(123, tmp_path)
    assert result["environment_allowlist"] == {"ASCEND_RT_VISIBLE_DEVICES": "4,5,6,7"}
    assert result["namespace_status"] == ["Pid:\t123", "NSpid:\t987 123"]
    assert "hidden" not in json.dumps(result)


@pytest.mark.parametrize("failure", [FileNotFoundError("npu-smi absent"), subprocess.TimeoutExpired("ps", 10)])
def test_snapshot_command_failures_are_explicit(monkeypatch, failure):
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(evidence.subprocess, "run", fail)
    result = evidence.command_snapshot(["npu-smi", "info"])
    assert "error" in result and "returncode" not in result


def test_offline_cli_does_not_run_commands_and_never_overwrites(tmp_path, monkeypatch):
    log, output = tmp_path / "console.log", tmp_path / "key.txt"
    log.write_text(ERROR)
    monkeypatch.setattr(evidence, "command_snapshot", lambda *a, **k: pytest.fail("offline must not run tools"))
    assert evidence.main(["--log", str(log), "--output", str(output)]) == 0
    assert ERROR in output.read_text()
    assert output.stat().st_size <= 256 * 1024
    with pytest.raises(SystemExit):
        evidence.main(["--log", str(log), "--output", str(output)])
    missing_output = tmp_path / "missing.txt"
    assert evidence.main(["--log", str(tmp_path / "missing.log"), "--output", str(missing_output)]) == 1
    assert "read_error" in missing_output.read_text()


def test_global_cap_and_metadata_allowlist(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "effective-target.json").write_text(json.dumps({"devices": [4, 5, 6, 7], "port": 7878, "api_key": "hidden"}))
    log = tmp_path / "large.log"
    log.write_text((ERROR + "x" * 10000 + "\n") * 100)
    output = tmp_path / "small.txt"
    assert evidence.main(["--log", str(log)] * 8 + ["--run-dir", str(run), "--output", str(output)]) == 0
    text = output.read_text()
    assert "hidden" not in text and "TRUNCATED" in text
    assert output.stat().st_size <= 256 * 1024


def test_successful_slow_warmup_in_middle_survives_without_errors(tmp_path):
    log = tmp_path / "warmup.log"
    stages = [
        "(Worker_TP0 pid=101) torch.compile took 230.25 s in total",
        "(Worker_TP1 pid=102) torch.compile took 231.50 s in total",
        "(Worker_TP0 pid=101) Initial profiling/warmup run took 1.20 s",
        "Capturing CUDA graphs (decode, FULL): 100%|34/34",
        "(Worker_TP0 pid=101) Graph capturing finished in 115 secs, took 1.00 GiB",
        "(EngineCore pid=100) init engine (profile, create kv cache, warmup model) took 360.00 s (compilation: 230.25 s)",
        "Start warmup, run with concurrency: 1",
        "Warmup finished Total Count: 1",
    ]
    log.write_text("noise\n" * 50 + "\n".join(stages) + "\n" + "noise\n" * 50)
    report = evidence.scan_log(log)
    assert report["first_fault_line"] is None
    assert report["startup_markers_matched"] == len(stages)
    assert [d["seconds"] for d in report["startup_durations"]] == [230.25, 231.5, 1.2, 115, 360]
    assert [d["pid"] for d in report["startup_durations"][:2]] == ["101", "102"]
    output = tmp_path / "warmup-key.txt"
    assert evidence.main(["--log", str(log), "--output", str(output)]) == 0
    assert all(stage in output.read_text() for stage in stages)


def test_many_capture_markers_preserve_completion_and_report_omission(tmp_path):
    log = tmp_path / "progress.log"
    log.write_text("Capturing CUDA graphs: progress\n" * 100 + "Graph capturing finished in 300 secs\n")
    report = evidence.scan_log(log)
    assert len(report["startup_markers"]) == 48
    assert report["startup_markers_omitted"] == 53
    assert "finished in 300 secs" in report["startup_markers"][-1]
