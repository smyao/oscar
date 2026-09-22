# Archive #35/#75/#94/#95/#117/#120: errors must be visible before exit,
# without losing logs, original exit codes, bounded cleanup or fixed cwd.
"""Real subprocess handshakes prove terminal output is not delayed to exit."""
import io
import json
import os
from pathlib import Path
import socket
import sys

import pytest

from tools import phase, service_probe


ROOT = Path(__file__).resolve().parents[1]


class AcknowledgingTerminal(io.StringIO):
    """Release a blocked child only after its output reaches the terminal."""

    def __init__(self, descriptor, messages):
        super().__init__()
        self.descriptor = descriptor
        self.remaining = list(messages)
        self.observed = []

    def write(self, text):
        count = super().write(text)
        while self.remaining and self.remaining[0] in self.getvalue():
            self.observed.append(self.remaining.pop(0))
            os.write(self.descriptor, b"x")
        return count


@pytest.fixture
def acknowledgment(tmp_path):
    path = tmp_path / "terminal-ack"
    os.mkfifo(path)
    descriptor = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    try:
        yield path, descriptor
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("nested", [False, True])
def test_phase_forwards_stdout_stderr_and_partial_line_before_child_exit(
    monkeypatch, tmp_path, acknowledgment, nested
):
    fifo, descriptor = acknowledgment
    # There are deliberately no flush() calls: the runner must set unbuffered
    # Python output even when the caller supplies a buffered environment.
    program = tmp_path / "blocked_child.py"
    program.write_text(
        "import os, sys\n"
        "print('OUT-ready')\n"
        "sys.stderr.write('ERR-partial')\n"
        "with open(sys.argv[1], 'rb', buffering=0) as ack:\n"
        "    assert ack.read(1) == b'x'\n"
        "print('\\nACK-received')\n"
        "raise SystemExit(7)\n"
    )
    sink = AcknowledgingTerminal(descriptor, ["\nOUT-ready\nERR-partial"])
    monkeypatch.setattr(sys, "stdout", sink)
    command = [sys.executable, str(program), str(fifo)]
    if nested:
        command = [sys.executable, "-m", "tools.phase", "--name", "inner",
                   "--log-dir", str(tmp_path / "inner"), "--timeout", "4", "--", *command]
    environment = {**os.environ, "PYTHONUNBUFFERED": "0"}
    result = phase.run_phase("outer", command, cwd=ROOT, log_dir=tmp_path,
                             timeout=6, grace=.2, env=environment)
    assert result.returncode == 7 and not result.timed_out and result.cleanup_complete
    assert sink.observed == ["\nOUT-ready\nERR-partial"]
    assert "ACK-received" in sink.getvalue()
    assert "PYTHONUNBUFFERED" not in environment or environment["PYTHONUNBUFFERED"] == "0"
    disk = Path(result.log).read_text()
    assert "OUT-ready\nERR-partial\nACK-received" in disk
    assert '"returncode": 7' in disk
    if nested:
        assert "OUT-ready\nERR-partial\nACK-received" in (tmp_path / "inner/inner.log").read_text()


def test_timeout_drains_signal_handler_output_and_keeps_final_status(tmp_path, capsys):
    program = tmp_path / "wait_for_timeout.py"
    program.write_text(
        "import os, signal\n"
        "def stop(signum, frame):\n"
        "    os.write(2, b'TERM-final-partial')\n"
        "    raise SystemExit(9)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "print('waiting-for-timeout')\n"
        "signal.pause()\n"
    )
    result = phase.run_phase("timeout", [sys.executable, str(program)], cwd=ROOT,
                             log_dir=tmp_path, timeout=1, grace=.5)
    terminal = capsys.readouterr().out
    assert result.returncode == 124 and result.timed_out and result.cleanup_complete
    for marker in ("waiting-for-timeout", "TIMEOUT phase=timeout", "TERM-final-partial", "RESULT "):
        assert marker in terminal and marker in Path(result.log).read_text()
    assert json.loads((tmp_path / "timeout.json").read_text())["timed_out"]


def test_spawn_error_is_visible_and_logged(tmp_path, capsys):
    result = phase.run_phase("missing", [str(tmp_path / "missing")], cwd=ROOT,
                             log_dir=tmp_path, timeout=1)
    terminal = capsys.readouterr().out
    assert result.returncode == 127 and result.cleanup_complete
    assert "EXEC_ERROR phase=missing" in terminal
    assert "EXEC_ERROR phase=missing" in Path(result.log).read_text()


def test_cleanup_error_is_visible_without_erasing_original_exit_code(monkeypatch, tmp_path, capsys):
    def failed_cleanup(process, grace):
        assert process.poll() == 7
        raise PermissionError("injected cleanup failure")

    monkeypatch.setattr(phase, "cleanup_group", failed_cleanup)
    result = phase.run_phase("failed", [sys.executable, "-c", "raise SystemExit(7)"],
                             cwd=ROOT, log_dir=tmp_path, timeout=3)
    assert result.returncode == 7 and not result.cleanup_complete
    assert "CLEANUP_ERROR phase=failed: PermissionError" in capsys.readouterr().out
    assert json.loads((tmp_path / "failed.json").read_text())["cleanup_complete"] is False


def test_service_startup_and_ongoing_output_reach_terminal_before_response(
    monkeypatch, tmp_path, acknowledgment
):
    fifo, descriptor = acknowledgment
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = json.loads((ROOT / "configs/target.json").read_text())
    config.update(host="127.0.0.1", port=port, service_startup_timeout_seconds=5,
                  shutdown_timeout_seconds=.5)
    program = tmp_path / "blocked_server.py"
    program.write_text(
        "import sys\n"
        "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
        "ack = open(sys.argv[2], 'rb', buffering=0)\n"
        "print('SERVER-startup')\n"
        "assert ack.read(1) == b'x'\n"
        "class Handler(BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        if self.path == '/trigger':\n"
        "            sys.stderr.write('\\nSERVER-active-partial')\n"
        "            assert ack.read(1) == b'x'\n"
        "        self.send_response(200)\n"
        "        self.send_header('Content-Length', '0')\n"
        "        self.end_headers()\n"
        "HTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()\n"
    )
    sink = AcknowledgingTerminal(descriptor, ["\nSERVER-startup\n", "\nSERVER-active-partial"])
    monkeypatch.setattr(sys, "stdout", sink)
    lifecycle = {}
    with service_probe.managed_server(
        config, ROOT / "configs/target.json", log_dir=tmp_path / "service",
        lifecycle=lifecycle, command=[sys.executable, str(program), str(port), str(fifo)]
    ) as server:
        assert sink.observed == ["\nSERVER-startup\n"]
        assert service_probe._http(server.base_url + "/trigger", timeout=3) == ""
        assert sink.observed == ["\nSERVER-startup\n", "\nSERVER-active-partial"]
    disk = Path(lifecycle["log"]).read_text()
    assert "SERVER-startup\n" in disk and "SERVER-active-partial" in disk
    assert "SERVICE_RESULT " in sink.getvalue() and lifecycle["cleanup_complete"]


def test_terminal_line_uses_one_descriptor_write_and_falls_back(monkeypatch, capsys):
    class DescriptorTerminal(io.StringIO):
        def __init__(self, descriptor):
            super().__init__()
            self.descriptor = descriptor

        def fileno(self):
            return self.descriptor

    read_end, write_end = os.pipe()
    try:
        sink = DescriptorTerminal(write_end)
        monkeypatch.setattr(sys, "stdout", sink)
        phase.terminal_line("whole-line")
        os.close(write_end)
        write_end = -1
        assert os.read(read_end, 4096) == b"whole-line\n"
        assert sink.getvalue() == ""
    finally:
        os.close(read_end)
        if write_end >= 0:
            os.close(write_end)
    monkeypatch.undo()
    phase.terminal_line("fallback-line")
    assert "fallback-line\n" in capsys.readouterr().out


def test_large_output_and_split_unicode_are_preserved(tmp_path, capsys):
    program = tmp_path / "many_bytes.py"
    program.write_text(
        "import os\n"
        "os.write(1, b'x' * 200000)\n"
        "os.write(1, b'\\xe4')\n"
        "os.write(2, b'\\xb8\\xad\\n')\n"
    )
    result = phase.run_phase("bulk", [sys.executable, str(program)], cwd=ROOT,
                             log_dir=tmp_path, timeout=3)
    assert result.returncode == 0
    expected = "x" * 200000 + "中\n"
    assert expected in capsys.readouterr().out and expected in Path(result.log).read_text()
