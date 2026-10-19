# Archive #94/#95/#125/#135/#157/#165/#166/#167: fatal engine evidence must not
# become a successful service outcome merely because the API process exits 0.
# Native evidence: launcher.terminate_if_errored and the guarded failure logs
# in multiproc_executor.WorkerProc/monitor_workers and engine.core.run_engine_core.
"""Real host subprocesses exercise fatal outcomes; no NPU or HTTP requests."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import install_serve, phase, serve_process


ROOT = Path(__file__).resolve().parents[1]
WORKER_DEATH = (
    "(EngineCore pid=488884) ERROR 10-19 06:39:32 [multiproc_executor.py:284] "
    "Worker proc VllmWorker-2 died unexpectedly, shutting down executor."
)
ENGINE_FATAL = (
    "(EngineCore pid=488884) ERROR 10-19 06:39:39 [core.py:1197] "
    "EngineCore encountered a fatal error."
)


def settings():
    config = json.loads((ROOT / "configs/target.json").read_text())
    config["shutdown_timeout_seconds"] = .15
    return config


def run_fixture(monkeypatch, tmp_path, source):
    original = subprocess.Popen

    def spawn(command, **kwargs):
        assert command[1:3] == ["-m", "tools.target_cli"]
        return original([sys.executable, "-c", source], **kwargs)

    monkeypatch.setattr(phase.subprocess, "Popen", spawn)
    return serve_process.run_service(settings(), ROOT / "configs/target.json", log_dir=tmp_path)


@pytest.mark.parametrize("message", [
    WORKER_DEATH,
    ENGINE_FATAL,
    "ERROR 10-19 06:39:32 [multiproc_executor.py:888] WorkerProc failed to start.",
    "(Worker_TP0 pid=488945) ERROR 10-19 06:39:32 [multiproc_executor.py:894] WorkerProc failed.",
    "(EngineCore_DP0 pid=123) ERROR 10-19 06:39:32 [core.py:1195] EngineCore failed to start.",
    "\x1b[31m" + WORKER_DEATH + "\x1b[0m",
    "(Worker_TP0 pid=488945) (Worker_TP1 pid=488966) " + WORKER_DEATH,
])
def test_native_fatal_with_zero_process_exit_is_failed_before_any_passed_output(
        monkeypatch, tmp_path, capsys, message):
    result = run_fixture(monkeypatch, tmp_path, f"print({message!r});raise SystemExit(0)")
    assert result.returncode != 0
    assert result.process_returncode == 0 and result.cleanup_complete
    assert result.failure["kind"] == "native_service_fatal"
    assert result.failure["message"] in message
    assert result.failure["log_byte_offset"] >= 0
    ledger = json.loads((tmp_path / "serve.json").read_text())
    assert ledger["returncode"] != 0 and ledger["process_returncode"] == 0
    output = capsys.readouterr().out
    assert "[oscar] FAILED phase=serve" in output
    assert "[oscar] PASSED phase=serve" not in output


@pytest.mark.parametrize("message", [
    "ERROR 10-19 06:39:32 [serving_chat.py:250] User request failed validation",
    "Traceback (most recent call last):\nValueError: invalid user request",
    "(Worker_TP0 pid=123) EOFError\nUserWarning: resource_tracker: 124 leaked semaphore objects",
    "INFO 10-19 06:39:32 [multiproc_executor.py:284] Worker proc VllmWorker-2 died unexpectedly, shutting down executor.",
    "ERROR 10-19 06:39:32 [request.py:284] Worker proc VllmWorker-2 died unexpectedly, shutting down executor.",
    "ERROR 10-19 06:39:32 [multiproc_executor.py:284] prompt='Worker proc VllmWorker-2 died unexpectedly, shutting down executor.'",
    "ERROR 10-19 06:39:32 [multiproc_executor.py:987] Error executing method 'execute_model'. This might cause deadlock in distributed execution.",
    "WARNING 10-19 06:39:32 [multiproc_executor.py:438] [shutdown] Executor: workers still running after grace period; sending SIGTERM count=4",
])
def test_request_errors_and_normal_shutdown_noise_are_not_engine_fatal(monkeypatch, tmp_path, message):
    result = run_fixture(monkeypatch, tmp_path, f"print({message!r})")
    assert result.returncode == result.process_returncode == 0
    assert result.failure is None and result.cleanup_complete


def test_nonzero_process_exit_is_kept_alongside_native_fatal_evidence(monkeypatch, tmp_path):
    result = run_fixture(monkeypatch, tmp_path, f"print({WORKER_DEATH!r});raise SystemExit(7)")
    assert result.returncode == result.process_returncode == 7
    assert result.failure["kind"] == "native_service_fatal"


def test_supervisor_cleanup_generated_worker_death_is_excluded(monkeypatch, tmp_path):
    # The API leader returns 0, while its descendant holds the log fd. Only
    # our ensuing TERM makes the descendant print the fatal-looking line.
    ready = tmp_path / "ready"
    source = (
        "import os,signal,time\nfrom pathlib import Path\n"
        "if os.fork()==0:\n"
        " def stop(signum,frame):\n"
        f"  print({WORKER_DEATH!r},flush=True)\n"
        "  os._exit(0)\n"
        " signal.signal(signal.SIGTERM,stop)\n"
        f" Path({str(ready)!r}).write_text('ready')\n"
        " time.sleep(30)\n"
        " os._exit(0)\n"
        f"while not Path({str(ready)!r}).exists(): time.sleep(.01)\n"
        "os._exit(0)\n"
    )
    result = run_fixture(monkeypatch, tmp_path, source)
    assert result.returncode == result.process_returncode == 0 and result.cleanup_complete
    assert WORKER_DEATH in Path(result.log).read_text()
    assert result.failure is None


def test_supervisor_signal_keeps_signal_exit_and_ignores_cleanup_fatal(monkeypatch, tmp_path):
    source = (
        "import os,signal,time\n"
        "def stop(signum,frame):\n"
        f" print({WORKER_DEATH!r},flush=True)\n"
        " raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM,stop)\n"
        "os.kill(os.getppid(),signal.SIGINT)\ntime.sleep(30)\n"
    )
    result = run_fixture(monkeypatch, tmp_path, source)
    assert result.returncode == 130 and result.process_returncode == 0
    assert result.interrupted and result.cleanup_complete and result.failure is None
    assert WORKER_DEATH in Path(result.log).read_text()


def test_plain_build_phase_does_not_apply_service_log_policy(tmp_path):
    result = phase.run_phase("build-fixture", [sys.executable, "-c", f"print({WORKER_DEATH!r})"],
        cwd=ROOT, log_dir=tmp_path, timeout=3, grace=.1)
    assert result.returncode == result.process_returncode == 0 and result.failure is None


@pytest.mark.parametrize("entry", ["install", "direct"])
def test_entrypoint_records_failed_status_with_original_zero_process_exit(monkeypatch, tmp_path, entry):
    original = subprocess.Popen
    monkeypatch.setattr(phase.subprocess, "Popen", lambda command, **kwargs:
        original([sys.executable, "-c", f"print({WORKER_DEATH!r})"], **kwargs))
    monkeypatch.setattr(install_serve, "run_phase", lambda name, command, **kwargs:
        phase.PhaseResult(name, command, 0, .01, False, False, str(tmp_path / name), True))
    if entry == "install":
        code = install_serve.main(["--variant", "candidate", "--log-dir", str(tmp_path)])
    else:
        code = serve_process.main(["--log-dir", str(tmp_path)])
    assert code != 0
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["service"]["returncode"] == 1
    assert status["service"]["process_returncode"] == 0
    assert status["service"]["failure"]["kind"] == "native_service_fatal"


def test_classifier_io_failure_is_not_success_and_cleanup_still_runs(monkeypatch, tmp_path, capsys):
    def cannot_read(*args):
        raise OSError("injected service log read failure")

    monkeypatch.setattr(serve_process, "classify_service_failure", cannot_read)
    result = run_fixture(monkeypatch, tmp_path, "raise SystemExit(0)")
    assert result.returncode == 1 and result.process_returncode == 0
    assert result.cleanup_complete and result.failure["kind"] == "failure_classification_error"
    assert "injected service log read failure" in result.failure["error"]
    output = capsys.readouterr().out
    assert "[oscar] PASSED phase=serve" not in output
    assert "injected service log read failure" in output


def test_actual_target_attachment_has_the_native_worker_death_evidence():
    source = ROOT / "reports/target_runtime_trap_20261019.txt"
    raw = source.read_bytes()
    failure = serve_process.classify_service_failure(source, len(raw))
    assert failure["kind"] == "native_service_fatal"
    assert failure["source"] == "multiproc_executor.py"
    assert failure["message"] == "Worker proc VllmWorker-2 died unexpectedly, shutting down executor."
    assert raw[failure["log_byte_offset"]:].startswith(WORKER_DEATH.encode())


def test_streaming_classifier_handles_large_log_and_oversized_request_line(tmp_path, monkeypatch):
    source = tmp_path / "large.log"
    # More than the startup diagnostic's 8 MiB limit; this verdict must find a
    # fatal at the end of a long run with constant-sized read requests.
    with source.open("wb") as stream:
        for _ in range(160):
            stream.write(b"INFO " + b"x" * 65530 + b"\n")
        stream.write(b"request " + b"x" * 65528 + WORKER_DEATH.encode() + b"\n")
        expected_offset = stream.tell()
        stream.write(ENGINE_FATAL.encode())  # complete message without newline
    original_open = Path.open
    reads = []

    class BoundedReader:
        def __enter__(self):
            self.stream = original_open(source, "rb")
            return self

        def __exit__(self, *args):
            self.stream.close()

        def readline(self, size):
            assert 0 < size <= 65536
            reads.append(size)
            return self.stream.readline(size)

    monkeypatch.setattr(Path, "open", lambda path, *args, **kwargs: BoundedReader())
    failure = serve_process.classify_service_failure(source, source.stat().st_size)
    assert len(reads) > 160
    assert failure["message"] == "EngineCore encountered a fatal error."
    assert failure["log_byte_offset"] == expected_offset


def test_fixed_byte_limit_excludes_appended_fatal_and_truncation_fails(tmp_path):
    source = tmp_path / "serve.log"
    prefix = b"INFO 10-19 06:39:32 [launcher.py:127] normal shutdown\n"
    source.write_bytes(prefix + WORKER_DEATH.encode() + b"\n")
    assert serve_process.classify_service_failure(source, len(prefix)) is None
    source.write_bytes(prefix)
    with pytest.raises(OSError, match="shortened"):
        serve_process.classify_service_failure(source, source.stat().st_size + 1)


@pytest.mark.parametrize("classification_error", [False, True])
def test_compact_phase_exposes_service_failure_marker_and_actual_process_code(tmp_path, capsys, classification_error):
    def unavailable(*args):
        raise OSError("injected classifier failure")

    classifier = unavailable if classification_error else serve_process.classify_service_failure
    result = phase.run_phase("serve", [sys.executable, "-c", f"print({WORKER_DEATH!r})"],
        cwd=ROOT, log_dir=tmp_path, timeout=3, grace=.1,
        env={"OSCAR_TERMINAL_LOG_MODE": "compact"}, classify_failure=classifier)
    output = capsys.readouterr().out
    assert result.returncode == 1 and result.process_returncode == 0
    assert "SERVICE_FAILURE " in output and "process_rc=0" in output
    assert "[oscar] PASSED phase=serve" not in output
