# Archive #94/#95/#125/#135/#136/#166/#167: every owned service entrance must
# fail on internal fatal/early exit/cleanup failure, with live durable evidence.
# Host-only process fixtures bypass HTTP/device work explicitly; no NPU or
# model completion is inferred from these lifecycle contract tests.
"""Fault-inject the shared probe/observe managed-service lifecycle, without sockets."""
import json
from contextlib import contextmanager
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from tools import phase, service_probe


ROOT = Path(__file__).resolve().parents[1]
FATAL = ("(EngineCore pid=123) ERROR 10-19 06:39:32 [multiproc_executor.py:284] "
         "Worker proc VllmWorker-2 died unexpectedly, shutting down executor.")


@pytest.fixture
def managed_fixture(monkeypatch, tmp_path):
    config = json.loads((ROOT / "configs/target.json").read_text())
    config.update(shutdown_timeout_seconds=.15, service_startup_timeout_seconds=3)
    path = tmp_path / "target.json"
    path.write_text(json.dumps(config))
    ready = tmp_path / "ready"
    finish = tmp_path / "finish"

    # Deliberately isolate the process-supervision contract from health, HTTP,
    # process inventory and NPU behavior. The actual child and signals are real.
    monkeypatch.setattr(service_probe, "_assert_port_free", lambda *_: None)

    def health(*args, **kwargs):
        deadline = time.monotonic() + 2
        while not ready.exists():
            if time.monotonic() >= deadline:
                raise AssertionError("host child did not start")
            time.sleep(.005)
        return "OK"

    monkeypatch.setattr(service_probe, "_http", health)
    monkeypatch.setattr(service_probe.OwnedProcessGroup, "refresh", lambda *args, **kwargs: None)

    def command(*, code=0, fatal=False, cleanup_message=False, fatal_start=False):
        source = (
            "import os,signal,time\nfrom pathlib import Path\n"
            "def stop(signum,frame):\n"
            f" print({FATAL if cleanup_message else 'normal child shutdown'!r},flush=True)\n"
            " raise SystemExit(0)\n"
            "signal.signal(signal.SIGTERM,stop)\n"
            + (f"print({FATAL!r},flush=True)\n" if fatal_start else "") +
            f"Path({str(ready)!r}).write_text(str(os.getpid()))\n"
            f"while not Path({str(finish)!r}).exists(): time.sleep(.005)\n"
            f"print({FATAL if fatal else 'normal child exit'!r},flush=True)\n"
            f"raise SystemExit({code})\n"
        )
        return [sys.executable, "-c", source]

    return config, path, ready, finish, command


def test_managed_service_does_not_accept_fatal_after_last_successful_check(managed_fixture, tmp_path):
    config, path, ready, finish, command = managed_fixture
    lifecycle = {}
    with pytest.raises(service_probe.ServiceProbeError, match="fatal"):
        with service_probe.managed_server(config, path, log_dir=tmp_path / "server",
                lifecycle=lifecycle, command=command(fatal=True)) as server:
            server.check_alive()
            finish.touch()
            assert server.process.wait(timeout=2) == 0
    assert lifecycle["status"] == "failed" and lifecycle["exit_code"] == 0
    assert lifecycle["failure"]["kind"] == "native_service_fatal"
    assert lifecycle["cleanup_complete"] and not phase.group_exists(int(ready.read_text()))


@pytest.mark.parametrize("code", [0, 7])
def test_api_exit_after_last_check_is_unexpected_even_without_fatal_record(managed_fixture, tmp_path, code):
    config, path, _, finish, command = managed_fixture
    lifecycle = {}
    with pytest.raises(service_probe.ServiceProbeError, match="exited before cleanup"):
        with service_probe.managed_server(config, path, log_dir=tmp_path / "server",
                lifecycle=lifecycle, command=command(code=code)) as server:
            finish.touch()
            server.process.wait(timeout=2)
    assert lifecycle["status"] == "failed" and lifecycle["exit_code"] == code
    assert lifecycle["cleanup_complete"]


def test_cleanup_failure_cannot_silently_leave_successful_context(managed_fixture, monkeypatch, tmp_path):
    config, path, ready, _, command = managed_fixture
    cleanup = service_probe.cleanup_group

    def failed_receipt(process, grace):
        assert cleanup(process, grace)
        return False  # represent a real failure of the cleanup contract

    monkeypatch.setattr(service_probe, "cleanup_group", failed_receipt)
    lifecycle = {}
    with pytest.raises(service_probe.ServiceProbeError, match="cleanup"):
        with service_probe.managed_server(config, path, log_dir=tmp_path / "server",
                lifecycle=lifecycle, command=command()):
            pass
    assert lifecycle["status"] == "failed" and not lifecycle["cleanup_complete"]
    assert not phase.group_exists(int(ready.read_text()))


def test_fatal_created_only_by_our_cleanup_does_not_fail_successful_context(managed_fixture, tmp_path):
    config, path, _, _, command = managed_fixture
    lifecycle = {}
    with service_probe.managed_server(config, path, log_dir=tmp_path / "server",
            lifecycle=lifecycle, command=command(cleanup_message=True)):
        pass
    assert lifecycle["status"] == "finished" and lifecycle["cleanup_complete"]
    assert lifecycle.get("failure") is None
    assert FATAL in Path(lifecycle["log"]).read_text()


def test_original_body_exception_survives_cleanup_failure(managed_fixture, monkeypatch, tmp_path):
    config, path, _, _, command = managed_fixture
    cleanup = service_probe.cleanup_group

    def failed_receipt(process, grace):
        assert cleanup(process, grace)
        return False

    monkeypatch.setattr(service_probe, "cleanup_group", failed_receipt)
    original = ValueError("original request processing failure")
    lifecycle = {}
    with pytest.raises(ValueError) as raised:
        with service_probe.managed_server(config, path, log_dir=tmp_path / "server",
                lifecycle=lifecycle, command=command()):
            raise original
    assert raised.value is original
    assert lifecycle["status"] == "failed" and not lifecycle["cleanup_complete"]


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_managed_context_handles_termination_and_restores_caller_handlers(managed_fixture, tmp_path, signum):
    config, path, ready, _, command = managed_fixture
    before = signal.getsignal(signum)
    lifecycle = {}
    with pytest.raises(KeyboardInterrupt, match=f"signal {signum}"):
        with service_probe.managed_server(config, path, log_dir=tmp_path / "server",
                lifecycle=lifecycle, command=command()):
            handler = signal.getsignal(signum)
            # This assertion protects the test runner when run against the old
            # code, which left TERM/HUP as process-terminating default actions.
            assert callable(handler) and handler is not before
            os.kill(os.getpid(), signum)
    assert lifecycle["status"] == "interrupted" and lifecycle["cleanup_complete"]
    assert signal.getsignal(signum) is before
    assert not phase.group_exists(int(ready.read_text()))


def test_synthetic_report_cannot_measure_through_failed_server_cleanup(managed_fixture, monkeypatch, tmp_path):
    config, path, _, _, command = managed_fixture
    cleanup = service_probe.cleanup_group

    def failed_receipt(process, grace):
        assert cleanup(process, grace)
        return False

    monkeypatch.setattr(service_probe, "cleanup_group", failed_receipt)
    monkeypatch.setattr(service_probe, "run_synthetic_mixed", lambda *args, **kwargs:
        {"status": "measured", "window": [0, 1]})
    report = service_probe.run_synthetic_only(path, output=tmp_path / "report.json",
        log_dir=tmp_path / "server", command=command(), tokenizer_factory=lambda _: object(), native=True)
    assert report["status"] == "failed" and not report["server"]["cleanup_complete"]


def test_precleanup_classifier_read_error_fails_closed(managed_fixture, monkeypatch, tmp_path):
    config, path, ready, _, command = managed_fixture

    def unreadable(*args):
        raise OSError("injected service log read failure")

    monkeypatch.setattr(service_probe, "classify_service_failure", unreadable)
    lifecycle = {}
    with pytest.raises(service_probe.ServiceProbeError, match="failure_classification_error"):
        with service_probe.managed_server(config, path, log_dir=tmp_path / "server",
                lifecycle=lifecycle, command=command()):
            pass
    assert lifecycle["status"] == "failed"
    assert lifecycle["failure"]["kind"] == "failure_classification_error"
    assert lifecycle["process_returncode_before_cleanup"] is None
    assert lifecycle["process_returncode"] == lifecycle["exit_code"] == 0
    assert lifecycle["cleanup_complete"] and not phase.group_exists(int(ready.read_text()))


def test_native_fatal_is_failure_even_while_api_process_remains_alive(managed_fixture, tmp_path):
    config, path, ready, _, command = managed_fixture
    lifecycle = {}
    with pytest.raises(service_probe.ServiceProbeError, match="native_service_fatal"):
        with service_probe.managed_server(config, path, log_dir=tmp_path / "server",
                lifecycle=lifecycle, command=command(fatal_start=True)) as server:
            assert server.process.poll() is None
    assert lifecycle["process_returncode_before_cleanup"] is None
    assert lifecycle["status"] == "failed" and lifecycle["cleanup_complete"]
    assert not phase.group_exists(int(ready.read_text()))


def test_original_exception_survives_precleanup_fatal(managed_fixture, tmp_path):
    config, path, _, finish, command = managed_fixture
    original = ValueError("original request error")
    lifecycle = {}
    with pytest.raises(ValueError) as raised:
        with service_probe.managed_server(config, path, log_dir=tmp_path / "server",
                lifecycle=lifecycle, command=command(fatal=True)) as server:
            finish.touch()
            server.process.wait(timeout=2)
            raise original
    assert raised.value is original
    assert lifecycle["failure"]["kind"] == "native_service_fatal"
    assert lifecycle["error"] == "ValueError: original request error"
    assert lifecycle["cleanup_complete"]


def test_repeated_cleanup_signals_do_not_interrupt_owned_cleanup(managed_fixture, monkeypatch, tmp_path):
    config, path, ready, _, command = managed_fixture
    cleanup = service_probe.cleanup_group
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}

    def interrupted_cleanup(process, grace):
        for sig in before:
            assert signal.getsignal(sig) == signal.SIG_IGN
            os.kill(os.getpid(), sig)
        return cleanup(process, grace)

    monkeypatch.setattr(service_probe, "cleanup_group", interrupted_cleanup)
    lifecycle = {}
    with service_probe.managed_server(config, path, log_dir=tmp_path / "server",
            lifecycle=lifecycle, command=command()):
        pass
    assert lifecycle["status"] == "finished" and lifecycle["cleanup_complete"]
    assert all(signal.getsignal(sig) is handler for sig, handler in before.items())
    assert not phase.group_exists(int(ready.read_text()))


@pytest.mark.parametrize("entry", ["phase", "managed_server"])
@pytest.mark.parametrize("when", ["before_child", "after_child", "repeated_after_child"])
def test_signal_during_popen_return_cannot_lose_owned_child(managed_fixture, monkeypatch, tmp_path, entry, when):
    config, path, _, _, command = managed_fixture
    real_popen = subprocess.Popen
    children = []

    def launch_and_interrupt(*args, **kwargs):
        if when == "before_child":
            os.kill(os.getpid(), signal.SIGTERM)
        child = real_popen(*args, **kwargs)
        children.append(child)
        # This real signal arrives after the OS child exists but before the
        # caller has assigned Popen's result to its owned-process variable.
        if when != "before_child":
            os.kill(os.getpid(), signal.SIGTERM)
        if when == "repeated_after_child":
            os.kill(os.getpid(), signal.SIGINT)
            os.kill(os.getpid(), signal.SIGHUP)
        return child

    monkeypatch.setattr(subprocess, "Popen", launch_and_interrupt)
    try:
        if entry == "phase":
            result = phase.run_phase("launch-interrupt", command(), cwd=ROOT,
                log_dir=tmp_path / "phase", timeout=3, grace=.15)
            assert result.interrupted and result.returncode == 128 + signal.SIGTERM
            assert result.cleanup_complete
        else:
            lifecycle = {}
            with pytest.raises(KeyboardInterrupt, match=f"signal {signal.SIGTERM}"):
                with service_probe.managed_server(config, path, log_dir=tmp_path / "server",
                        lifecycle=lifecycle, command=command()):
                    pytest.fail("interrupted launch must not yield a service")
            assert lifecycle["status"] == "interrupted" and lifecycle["cleanup_complete"]
        assert len(children) == 1 and children[0].poll() is not None
        assert not phase.group_exists(children[0].pid)
    finally:
        # A regression must not leak the real fault-injection child itself.
        for child in children:
            phase.cleanup_group(child, .15)


@pytest.mark.parametrize("entry", ["phase", "managed_server"])
@pytest.mark.parametrize("with_signal", [False, True])
def test_popen_failure_preserves_original_error_and_pending_signal(managed_fixture, monkeypatch, tmp_path, entry, with_signal):
    config, path, _, _, command = managed_fixture
    original = OSError("injected process creation failure")

    def fail_launch(*args, **kwargs):
        if with_signal:
            os.kill(os.getpid(), signal.SIGTERM)
        raise original

    monkeypatch.setattr(subprocess, "Popen", fail_launch)
    if entry == "phase":
        result = phase.run_phase("launch-error", command(), cwd=ROOT,
            log_dir=tmp_path / "phase", timeout=3, grace=.15)
        assert result.returncode == (128 + signal.SIGTERM if with_signal else 127)
        assert result.interrupted is with_signal
        assert result.process_returncode is None and result.cleanup_complete
        assert str(original) in Path(result.log).read_text()
    else:
        lifecycle = {}
        with pytest.raises(KeyboardInterrupt if with_signal else OSError) as raised:
            with service_probe.managed_server(config, path, log_dir=tmp_path / "server",
                    lifecycle=lifecycle, command=command()):
                pytest.fail("failed creation must not yield a service")
        assert (raised.value.__cause__ if with_signal else raised.value) is original
        assert lifecycle["launch_error"] == f"OSError: {original}"
        assert lifecycle["status"] == ("interrupted" if with_signal else "failed")
        assert lifecycle["process_returncode"] is None and lifecycle["cleanup_complete"]


def test_synthetic_interrupt_cannot_hide_existing_native_fatal(managed_fixture, monkeypatch, tmp_path):
    config, path, _, finish, command = managed_fixture

    def fatal_then_interrupt(server, *args, **kwargs):
        finish.touch()
        assert server.process.wait(timeout=2) == 0
        raise KeyboardInterrupt("user interrupted after worker failure")

    monkeypatch.setattr(service_probe, "run_synthetic_mixed", fatal_then_interrupt)
    report = service_probe.run_synthetic_only(path, output=tmp_path / "report.json",
        log_dir=tmp_path / "server", command=command(fatal=True), tokenizer_factory=lambda _: object(), native=True)
    assert report["status"] == "failed"
    assert report["server"]["failure"]["kind"] == "native_service_fatal"
    assert report["server"]["process_returncode"] == 0 and report["server"]["cleanup_complete"]


@pytest.mark.parametrize("classification_error", [False, True])
@pytest.mark.parametrize("secondary_failure", [False, True])
def test_phase_failure_record_write_error_cannot_skip_owned_cleanup(managed_fixture, monkeypatch, tmp_path,
        classification_error, secondary_failure):
    _, _, ready, _, command = managed_fixture
    real_log, real_cleanup = phase.live_log, phase.cleanup_group
    cleaned_children = []
    launched_children = []
    real_popen = subprocess.Popen
    original = OSError("injected SERVICE_FAILURE write failure")
    secondary = OSError("injected CLEANUP_ERROR write failure")

    class FaultyWriter:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def write(self, value):
            if value.startswith("SERVICE_FAILURE "):
                raise original
            if secondary_failure and value.startswith("CLEANUP_ERROR "):
                raise secondary
            return self.stream.write(value)

    @contextmanager
    def failing_log(*args, **kwargs):
        with real_log(*args, **kwargs) as stream:
            yield FaultyWriter(stream)

    def observe_cleanup(process, grace):
        cleaned_children.append(process)
        return real_cleanup(process, grace)

    def observed_launch(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        launched_children.append(child)
        return child

    monkeypatch.setattr(phase, "live_log", failing_log)
    monkeypatch.setattr(phase, "cleanup_group", observe_cleanup)
    monkeypatch.setattr(subprocess, "Popen", observed_launch)

    def unreadable(*args):
        raise OSError("injected classification read failure")

    def run():
        return phase.run_phase("failure-write", command(fatal_start=not classification_error), cwd=ROOT,
            log_dir=tmp_path / "phase", timeout=.2, grace=.15,
            classify_failure=unreadable if classification_error else service_probe.classify_service_failure)

    result = None
    try:
        if secondary_failure:
            with pytest.raises(OSError) as raised:
                run()
            assert raised.value is original and raised.value.__cause__ is secondary
        else:
            result = run()
            assert result.timed_out and result.returncode == 124
            assert result.failure["kind"] == ("failure_classification_error" if classification_error
                                               else "native_service_fatal")
            assert result.cleanup_complete
            assert str(original) in Path(result.log).read_text()
        assert len(cleaned_children) == 1, "failure logging bypassed owned-process cleanup"
        assert cleaned_children[0].poll() == 0
        if result is not None:
            assert result.process_returncode == cleaned_children[0].returncode
        assert not phase.group_exists(int(ready.read_text()))
    finally:
        # Baseline failure must not leak this test's own child. No unrelated
        # PID can enter this list or this fixture's ready file.
        for child in launched_children:
            real_cleanup(child, .15)


def test_signal_while_installing_cleanup_handlers_cannot_skip_cleanup(managed_fixture, monkeypatch, tmp_path):
    _, _, ready, _, command = managed_fixture
    real_signal = signal.signal
    injected = False

    def interrupt_before_ignore(signum, handler):
        nonlocal injected
        if handler == signal.SIG_IGN and not injected:
            injected = True
            assert callable(signal.getsignal(signal.SIGHUP))
            os.kill(os.getpid(), signal.SIGHUP)
        return real_signal(signum, handler)

    monkeypatch.setattr(signal, "signal", interrupt_before_ignore)
    result = phase.run_phase("cleanup-signal", command(), cwd=ROOT,
        log_dir=tmp_path / "phase", timeout=.2, grace=.15)
    assert injected and result.timed_out and result.returncode == 124
    assert result.cleanup_complete and result.process_returncode == 0
    assert not phase.group_exists(int(ready.read_text()))


@pytest.mark.parametrize("cleanup_outcome", ["complete", "incomplete", "raises"])
def test_phase_body_exception_survives_failure_logging_and_cleanup_faults(managed_fixture, monkeypatch, tmp_path,
        cleanup_outcome):
    _, _, ready, _, command = managed_fixture
    real_log, real_cleanup, real_sleep = phase.live_log, phase.cleanup_group, time.sleep
    original = ValueError("original phase supervision error")
    injected = False
    cleanup_calls = []

    class FaultyWriter:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def write(self, value):
            if value.startswith("SERVICE_FAILURE "):
                raise OSError("secondary failure-record write error")
            return self.stream.write(value)

    @contextmanager
    def failing_log(*args, **kwargs):
        with real_log(*args, **kwargs) as stream:
            yield FaultyWriter(stream)

    def body_error(seconds):
        nonlocal injected
        if not injected:
            injected = True
            deadline = time.monotonic() + 2
            while not ready.exists() and time.monotonic() < deadline:
                real_sleep(.005)
            assert ready.exists()
            raise original
        real_sleep(seconds)

    def observed_cleanup(process, grace):
        cleanup_calls.append(process)
        assert real_cleanup(process, grace)
        if cleanup_outcome == "raises":
            raise PermissionError("injected cleanup receipt error")
        return cleanup_outcome == "complete"

    monkeypatch.setattr(phase, "live_log", failing_log)
    monkeypatch.setattr(phase, "cleanup_group", observed_cleanup)
    monkeypatch.setattr(time, "sleep", body_error)
    with pytest.raises(ValueError) as raised:
        phase.run_phase("body-error", command(fatal_start=True), cwd=ROOT,
            log_dir=tmp_path / "phase", timeout=3, grace=.15,
            classify_failure=service_probe.classify_service_failure)
    assert raised.value is original
    assert len(cleanup_calls) == 1 and cleanup_calls[0].poll() == 0
    assert not phase.group_exists(int(ready.read_text()))
    recorded = json.loads((tmp_path / "phase/body-error.json").read_text())
    assert recorded["returncode"] == 127 and recorded["process_returncode"] == 0
    assert recorded["failure"]["kind"] == "native_service_fatal"
    assert recorded["cleanup_complete"] is (cleanup_outcome == "complete")
