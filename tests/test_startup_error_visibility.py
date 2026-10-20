# Archive #94/#95/#125/#133: real child-process startup failures must be
# visible and clean up owned children even when the API-server parent lives.
# Native multiproc_executor.py:987 catches worker RPC errors and continues;
# health retries alone therefore cannot detect this failure mode. Host tests
# deliberately use no NPU and do not establish target startup acceptance.
import json
import sys
import time
from pathlib import Path

import pytest

from tools import service_probe
from tools.phase import group_exists


ROOT = Path(__file__).resolve().parents[1]


def child_log(path):
    # The launch receipt embeds the program text. Only actual child lines can
    # make the injected health check ready, exactly as a real server would.
    if not path.exists():
        return ""
    return "\n".join(line for line in path.read_text().splitlines()
                     if not line.startswith("START owned_service "))


def configured(monkeypatch):
    config = json.loads((ROOT / "configs/target.json").read_text())
    config.update(service_startup_timeout_seconds=30, shutdown_timeout_seconds=0.5)
    # Actual owned child processes are exercised; HTTP is a host-only seam so
    # this fault test does not require a listening network or a target device.
    monkeypatch.setattr(service_probe, "_assert_port_free", lambda *args: None)
    return config


@pytest.mark.parametrize("health_returns_ok", [False, True])
def test_worker_rpc_failure_stops_live_parent_and_owned_children(
        tmp_path, monkeypatch, capsys, health_returns_ok):
    config = configured(monkeypatch)
    log = tmp_path / "server" / "server.log"
    prefix = "(Worker_TP1 pid=178292) ERROR 10-20 00:19:49 [multiproc_executor.py:987] "
    code = "\n".join([
        "import subprocess, sys, time",
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])",
        f"print({(prefix + 'WorkerProc hit an exception.')!r}, flush=True)",
        f"print({(prefix + 'Traceback (most recent call last):')!r}, flush=True)",
        f"print({(prefix + '  File \"<string>\", line 113, in kernel')!r}, flush=True)",
        f"print({(prefix + 'ValueError: too many values to unpack (expected 20)')!r}, flush=True)",
        "time.sleep(30)",
    ])

    def health(*args, **kwargs):
        if health_returns_ok and "ValueError:" in child_log(log):
            return ""
        raise OSError("Connection refused")

    monkeypatch.setattr(service_probe, "_http", health)
    lifecycle = {}
    start = time.monotonic()
    with pytest.raises(service_probe.ServiceProbeError, match="too many values to unpack"):
        with service_probe.managed_server(config, ROOT / "configs/target.json",
                log_dir=log.parent, lifecycle=lifecycle,
                command=[sys.executable, "-c", code], quiet=True):
            pytest.fail("worker startup failure must not become a healthy service")
    assert time.monotonic() - start < 5, "must not wait for the 30-second startup timeout"
    assert lifecycle["status"] == "failed" and lifecycle["cleanup_complete"] is True
    assert not group_exists(lifecycle["pid"])
    assert len(lifecycle["owned_pids"]) >= 2
    persisted = json.loads((log.parent / "server_lifecycle.json").read_text())
    assert persisted["startup_failure"]["cause"] == "ValueError: too many values to unpack (expected 20)"
    output = capsys.readouterr()
    assert "STARTUP_ERROR_EXCERPT_BEGIN" in output.err
    assert "ValueError: too many values to unpack (expected 20)" in output.err
    assert "CLEANUP_END owned_server complete=True" in output.out
    assert "Traceback (most recent call last)" in log.read_text()


def test_recoverable_error_warning_and_background_traceback_do_not_abort(
        tmp_path, monkeypatch, capsys):
    config = configured(monkeypatch)
    log = tmp_path / "server" / "server.log"
    chatter = "\n".join([
        "WARNING optional feature is unavailable",
        "ERROR optional telemetry exporter is unavailable; continuing",
        "Exception in thread TBE-cache-worker:",
        "Traceback (most recent call last):",
        "ConnectionError: optional cache endpoint unavailable",
        "READY after recoverable initialization error",
    ])

    def health(*args, **kwargs):
        if "READY after" in child_log(log):
            return ""
        raise OSError("Connection refused")

    monkeypatch.setattr(service_probe, "_http", health)
    lifecycle = {}
    with service_probe.managed_server(config, ROOT / "configs/target.json",
            log_dir=log.parent, lifecycle=lifecycle, quiet=True,
            command=[sys.executable, "-c", f"import time; print({chatter!r}, flush=True); time.sleep(30)"]):
        assert lifecycle["status"] == "healthy"
        assert "startup_failure" not in lifecycle
    output = capsys.readouterr()
    assert chatter not in output.out and "STARTUP_FAILED" not in output.err
    assert chatter in log.read_text() and lifecycle["cleanup_complete"] is True


def test_quiet_startup_progress_shows_host_phase_without_child_spam(
        tmp_path, monkeypatch, capsys):
    config = configured(monkeypatch)
    monkeypatch.setattr(service_probe, "QUIET_STARTUP_HEARTBEAT_SECONDS", 0.1)
    log = tmp_path / "server" / "server.log"
    code = "\n".join([
        "import json, os, pathlib, time",
        "trace = pathlib.Path(os.environ['OSCAR_TRACE_DIR'])",
        "(trace / 'phase-test.json').write_text(json.dumps({'pid': os.getpid(), 'rank': 0, 'state': 'entered', 'phase': 'model_load', 'layer': 'layers.3', 'wall_time': time.time()}))",
        "print('INFO model weights loaded; compiling graph', flush=True)",
        "time.sleep(0.7)",
        "print('READY_TEST', flush=True)",
        "time.sleep(30)",
    ])

    def health(*args, **kwargs):
        if "READY_TEST" in child_log(log):
            return ""
        raise OSError("Connection refused")

    monkeypatch.setattr(service_probe, "_http", health)
    lifecycle = {}
    with service_probe.managed_server(config, ROOT / "configs/target.json",
            log_dir=log.parent, lifecycle=lifecycle, quiet=True,
            command=[sys.executable, "-c", code]):
        assert lifecycle["status"] == "healthy"
    output = capsys.readouterr().out
    waits = [line for line in output.splitlines() if "waiting for service health" in line]
    latest = [line for line in output.splitlines() if "startup latest=" in line]
    assert waits and len(latest) == len(waits)
    assert all("r0:" in line and "layers.3" in line for line in waits)
    assert any("compiling graph" in line for line in latest)
    assert lifecycle["startup_worker_progress"][0]["phase"] == "model_load"


def test_failure_excerpt_is_bounded_but_keeps_terminal_cause(tmp_path, monkeypatch, capsys):
    log = tmp_path / "server.log"
    log.write_text("WorkerProc hit an exception.\n" +
                   ("  File \"long_stack.py\", line 123, in call " + "x" * 1000 + "\n") * 400 +
                   "ValueError: too many values to unpack (expected 20)\n")
    observer = service_probe._QuietStartupLog(log)
    observer.refresh()
    monkeypatch.setattr(service_probe, "_STARTUP_TRACE_DRAIN_SECONDS", 0)
    lifecycle = {}
    with pytest.raises(service_probe.ServiceProbeError, match="expected 20"):
        observer.check(lifecycle)
    excerpt = lifecycle["startup_failure"]["excerpt"]
    assert len(excerpt) <= 64
    assert len(("\n".join(excerpt) + "\n").encode()) <= 16 * 1024
    output = capsys.readouterr().err
    assert len(output.splitlines()) <= 67 and len(output.encode()) <= 18 * 1024
    assert "ValueError: too many values to unpack (expected 20)" in output
