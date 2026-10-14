"""Archive #94/#95/#118/#125/#157: retain primary errors in bounded excerpts."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools import collect_error_bundle as bundle


def scan(text):
    raw = text.encode()
    return bundle.scan(io.BytesIO(raw), size=len(raw))


def test_repeated_api_stack_does_not_hide_worker_terminal_exception():
    api = "(APIServer pid=1) ERROR 10-14 02:48:23 [serving.py:932] "
    repeated = (api + '  File "async_llm.py", line 579, in generate\n'
                + api + "    raise output\n") * 600
    text = repeated + """(Worker_TP2 pid=22) ERROR 10-14 02:48:22 [worker.py:10] Traceback (most recent call last):
(Worker_TP2 pid=22) ERROR 10-14 02:48:22 [worker.py:10]   File "guard_diag.py", line 41, in run_status_guard
(Worker_TP2 pid=22) ERROR 10-14 02:48:22 [worker.py:10] OscarReadinessError: site=forward family=store
""" + repeated
    result = scan(text)
    output = bundle.make_report([("console.log", result)], root="logs", omitted=0, notices=[], max_bytes=49152)
    assert "family=store" in output
    assert output.index("worker_or_device") < output.index("FIRST api CONTEXT")
    assert "repeated" in output and len(output.encode()) <= 49152
    assert "API errors alone cannot identify" not in output
    assert result["error_line_counts"]["worker_or_device"] == 3


def test_long_traceback_keeps_exception_after_context_limit():
    text = "Traceback (most recent call last):\n" + '  File "frame.py", line 3\n' * 400
    text += "ValueError: important final reason\n"
    result = scan(text)
    assert result["windows"]["other"]["left"] == 0
    output = bundle.make_report([("server.log", result)], root="logs", omitted=0, notices=[], max_bytes=16384)
    assert "important final reason" in output and "L402:" in output


def test_api_only_excerpt_explicitly_reports_no_primary_evidence():
    result = scan("(APIServer pid=1) ERROR 10-14 02:48:23 [x.py:1] EngineDeadError: EngineCore encountered an issue\n")
    output = bundle.make_report([("terminal", result)], root="logs", omitted=0, notices=[], max_bytes=16384)
    assert "API errors alone cannot identify the root cause" in output
    assert "EngineDeadError" in output


def test_normal_hccl_timeout_info_is_not_a_fault():
    result = scan("(Worker_TP0 pid=2) INFO 10-14 01:00:00 [platform.py:1277] "
                  "The timeout interval of the HCCL operator is 1836s. "
                  "Set VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000\n")
    assert result["error_line_counts"] == {}


def test_recovered_compile_warning_does_not_displace_real_worker_fault():
    result = scan("(Worker_TP0 pid=2) WARNING 10-14 06:07:49 [decorators.py:321] Original traceback:\n"
                  "(Worker_TP0 pid=2) INFO 10-14 06:09:24 [monitor.py:53] torch.compile took 94.13 s in total\n"
                  "(Worker_TP0 pid=2) ERROR 10-14 06:16:30 [multiproc_executor.py:987] WorkerProc hit an exception.\n")
    assert result["error_line_counts"] == {"worker_or_device": 1}
    assert result["windows"]["worker_or_device"]["first_error_line"] == 3


@pytest.mark.parametrize("stamp", [1791948488.5070946, 1791948660.507023])
def test_actual_dispatch_timestamps_do_not_fabricate_device_faults(stamp):
    # User's 20261014T032237 run: both rows were falsely labeled exceptions.
    result = scan(json.dumps({"event": "attention_dispatched", "wall_time": stamp,
        "pid": 507035, "rank": 2, "cv_operator": "attention_cv_mdl_decode_out",
        "device_completion": "not_observed_here"}) + "\n")
    assert result["error_line_counts"] == {} and result["detail_matches"] == 0
    report = bundle.make_report([("worker.jsonl", result)], root="logs", omitted=0,
                                notices=[], max_bytes=49152)
    assert "No Worker/device or EngineCore error found" in report
    assert "EXCEPTION DETAILS" not in report


@pytest.mark.parametrize("record", [
    {"returncode": 0, "exit_code": None, "error": None, "failed_phase": None},
    {"status": "passed", "returncode": "0", "elapsed_seconds": 1.507035},
    {"phases": [{"returncode": 0}, {"exit_code": 0}]},
])
def test_successful_json_results_are_not_exceptions(record):
    for text in (json.dumps(record), json.dumps(record, indent=2), "RESULT " + json.dumps(record)):
        result = scan(text + "\n")
        assert result["error_line_counts"] == {} and result["detail_matches"] == 0


@pytest.mark.parametrize("record", [
    {"returncode": -9}, {"exit_code": "2"}, {"status": "failed"},
    {"level": "ERROR", "message": "worker exited"},
    {"error": "device stopped"}, {"server": {"returncode": 7}},
])
def test_real_structured_failures_still_keep_context_and_details(record):
    for text in (json.dumps(record), json.dumps(record, indent=2), "RESULT " + json.dumps(record)):
        result = scan(text + "\n")
        assert result["error_line_counts"] and result["detail_matches"]


@pytest.mark.parametrize("text", ["ACL error code is 507035", "error code: 507035",
                                    "ErrCode=507035", "aivec error", "timeout or trap error",
                                    "RuntimeError: synchronize failed with 507035"])
def test_plain_device_faults_remain_primary_evidence(text):
    result = scan(text + "\n")
    assert result["error_line_counts"] == {"worker_or_device": 1}
    assert result["detail_matches"] == 1


def test_plain_fractional_timestamps_are_not_device_faults():
    result = scan("elapsed=1.507035 timestamp=1791948660.507023 pid=507035\n")
    assert result["error_line_counts"] == {} and not result["detail_matches"]


def test_stream_snapshot_and_oversized_line_hash():
    raw = b"x" * (2 * 1024 * 1024) + b"\nRuntimeError: retained\n"
    result = bundle.scan(io.BytesIO(raw + b"new appended data\n"), size=len(raw))
    assert result["bytes_scanned"] == len(raw)
    assert result["sha256_of_scanned_bytes"] == hashlib.sha256(raw).hexdigest()
    assert result["lines"] == 2 and result["oversized_lines_prefix_only"] == 1
    assert result["details"][0][1][1] == "RuntimeError: retained"


def test_discovery_limits_recent_files_and_does_not_follow_symlinks(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    for i in range(6):
        path = logs / f"{i}.log"
        path.write_text("test")
        os.utime(path, (i + 1, i + 1))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.log").write_text("not selected")
    (logs / "linked").symlink_to(outside, target_is_directory=True)
    (logs / "linked.log").symlink_to(outside / "secret.log")
    (logs / "error-bundle-old.txt").write_text("old report")
    paths, omitted, errors = bundle.discover(logs, (logs / "5.log").resolve(), 2)
    assert [p.name for p in paths] == ["4.log", "3.log"]
    assert omitted == 3 and errors == []


def test_many_chinese_errors_obey_global_byte_limit_and_keep_source_labels():
    reports = []
    for i in range(32):
        text = "".join(f"(Worker_TP{i % 4} pid={i}) ERROR 10-14 02:48:22 [x.py:1] "
                       f"RuntimeError: failure{j}原因" + "中文" * 700 + "\n" for j in range(40))
        reports.append((f"logs/run-{i}/server.log", scan(text)))
    output = bundle.make_report(reports, root="logs", omitted=10, notices=[], max_bytes=49152)
    assert len(output.encode()) <= 49152
    assert "omitted" in output and "logs/run-0/server.log" in output
    assert output.endswith("original logs unchanged.\n")


def test_standalone_cli_pasted_terminal_plus_directory(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "worker.log").write_text("(Worker_TP0 pid=2) RuntimeError: worker reason\n")
    output = tmp_path / "key.txt"
    result = subprocess.run([sys.executable, str(Path(bundle.__file__).resolve()),
        "--terminal", "-", "--logs-dir", str(logs), "--output", str(output)],
        input="(APIServer pid=1) EngineDeadError: engine failed\n", text=True,
        capture_output=True, cwd=tmp_path, timeout=10)
    assert result.returncode == 0, result.stderr
    report = output.read_text()
    assert "worker reason" in report and "engine failed" in report
    assert "<pasted-terminal>" in report and "Ctrl-D" in result.stderr
    assert str(output) in result.stdout and output.stat().st_size <= 49152


def test_read_failures_overwrite_and_fifo_are_explicit(tmp_path, capsys):
    missing = tmp_path / "missing.log"
    output = tmp_path / "key.txt"
    assert bundle.main(["--logs-dir", str(tmp_path), "--terminal", str(missing), "--output", str(output)]) == 1
    assert "READ_ERROR" in output.read_text()
    messages = capsys.readouterr()
    assert str(missing) in messages.err and "READ_ERROR" in messages.err
    assert "--terminal -" in messages.err
    assert "terminal file NOT collected" in output.read_text()
    with pytest.raises(SystemExit):
        bundle.main(["--output", str(output)])
    pipe = tmp_path / "pipe"
    os.mkfifo(pipe)
    other = tmp_path / "other.txt"
    assert bundle.main(["--logs-dir", str(tmp_path / "no-dir"), "--terminal", str(pipe), "--output", str(other)]) == 1
    assert "regular saved file" in other.read_text()


def test_duplicate_terminal_and_discovered_path_is_read_once(tmp_path):
    log = tmp_path / "terminal.log"
    log.write_text("RuntimeError: one\n")
    output = tmp_path / "out.txt"
    assert bundle.main(["--logs-dir", str(tmp_path), "--terminal", str(log), "--output", str(output)]) == 0
    assert "selected_sources=1" in output.read_text()
