# Archive #35/#75/#94/#95/#117/#120/#125/#135: errors must be visible before exit,
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


def test_compact_terminal_keeps_summary_and_traceback_while_log_stays_complete(
    tmp_path, capsys
):
    # Archive #94/#95/#125: concise terminal output must still expose the
    # actual failure, while the durable phase log preserves routine chatter.
    child = ("print('INFO routine startup')\n"
             "print('[oscar] PERF_NATIVE complete=4/4 ttft_p50_ms=10')\n"
             "raise RuntimeError('device phase failed')\n")
    result = phase.run_phase("compact", [sys.executable, "-c", child],
                             cwd=ROOT, log_dir=tmp_path, timeout=5,
                             env={**os.environ, "OSCAR_TERMINAL_LOG_MODE": "compact"})
    terminal = capsys.readouterr().out
    disk = Path(result.log).read_text()
    assert result.returncode != 0
    assert "INFO routine startup" not in terminal
    assert "[oscar] PERF_NATIVE complete=4/4" in terminal
    assert "Traceback (most recent call last)" in terminal
    assert "RuntimeError: device phase failed" in terminal
    assert "INFO routine startup" in disk
    assert "RuntimeError: device phase failed" in disk


@pytest.mark.parametrize("variant", ["native_interleaved", "oscar_single"])
def test_compact_filters_only_successful_tbe_shutdown_eof_and_startup_chatter(
    tmp_path, capsys, variant
):
    # Archive #94/#95/#125: preserve the complete service evidence on disk.
    # The native trace is deliberately interleaved across TP ranks, as in the
    # user-provided 2026-09-23 paired performance run.
    if variant == "native_interleaved":
        traceback_lines = [
            "(Worker_TP3 pid=84152) Exception in thread Thread-2:",
            "(Worker_TP2 pid=84092) Exception in thread Thread-2:",
            "(Worker_TP3 pid=84152) Traceback (most recent call last):",
            "(Worker_TP2 pid=84092) Traceback (most recent call last):",
            '(Worker_TP3 pid=84152)   File "/usr/local/python3.12.13/lib/python3.12/threading.py", line 1075, in _bootstrap_inner',
            '(Worker_TP2 pid=84092)   File "/usr/local/python3.12.13/lib/python3.12/threading.py", line 1075, in _bootstrap_inner',
            '(Worker_TP3 pid=84152)   File "/usr/local/Ascend/cann-9.1.0/python/site-packages/tbe/common/repository_manager/utils/multiprocess_util.py", line 68, in run',
            '(Worker_TP2 pid=84092)   File "/usr/local/Ascend/cann-9.1.0/python/site-packages/tbe/common/repository_manager/utils/multiprocess_util.py", line 68, in run',
            "(Worker_TP3 pid=84152)     raise EOFError",
            "(Worker_TP2 pid=84092)     raise EOFError",
            "(Worker_TP3 pid=84152) EOFErrorbuf = self._recv(4)",
            "(Worker_TP2 pid=84092) EOFError",
        ]
    else:
        traceback_lines = [
            "(Worker_TP0 pid=89131) Exception in thread Thread-3:",
            "(Worker_TP0 pid=89131) Traceback (most recent call last):",
            '(Worker_TP0 pid=89131)   File "/usr/local/python3.12.13/lib/python3.12/threading.py", line 1075, in _bootstrap_inner',
            '(Worker_TP0 pid=89131)   File "/usr/local/Ascend/cann-9.1.0/python/site-packages/tbe/common/repository_manager/utils/multiprocess_util.py", line 68, in run',
            "(Worker_TP0 pid=89131)     raise EOFError",
            "(Worker_TP0 pid=89131) EOFError",
        ]
    lines = [
        "(APIServer pid=83886) INFO 09-23 05:14:21 [platform.py:1277] The timeout interval of the HCCL operator is 1836s. Timeout in seconds for execute_model RPC calls in multiprocessing must be greater than 1836s, Set VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000",
        "Capturing CUDA graphs (decode, FULL):  44%|████▍     | 15/34 [00:12<00:13,  1.42it/s]STARTUP_WAIT elapsed=286.0 health=<urlopen error [Errno 111] Connection refused>",
        "[oscar] SYNTHETIC_MIXED done status=measured completed=4/4 failed=0",
        "(APIServer pid=83886) INFO 09-23 05:19:56 [launcher.py:116] [shutdown] API server: stopping engine client mode=abort timeout=0s",
        "(EngineCore pid=83948) INFO 09-23 05:19:56 [core.py:1297] [shutdown] EngineCore: start mode=abort timeout=0s",
        "(APIServer pid=83886) INFO 09-23 05:19:56 [core_client.py:652] [shutdown] MPClient: start timeout=0s",
        *traceback_lines,
        "SERVICE_RESULT " + json.dumps({"status": "finished", "cleanup_complete": True, "exit_code": 0}),
    ]
    path = tmp_path / f"{variant}.log"
    with phase.live_log(path, mode="compact") as stream:
        stream.write("\n".join(lines) + "\n")
    terminal = capsys.readouterr().out
    assert "[oscar] SYNTHETIC_MIXED done" in terminal
    assert "HCCL operator" not in terminal
    assert "STARTUP_WAIT" not in terminal
    assert "[shutdown]" not in terminal
    assert "Exception in thread" not in terminal
    assert "EOFError" not in terminal
    assert path.read_text() == "\n".join(lines) + "\n"


def test_compact_replays_unexpected_worker_traceback_before_service_result():
    # Archive #125: an actual worker failure must not wait for shutdown or
    # disappear just because its first frames resemble the TBE EOF case.
    compact = phase._CompactTerminalFilter()
    compact.feed("(EngineCore pid=1) INFO 09-23 05:19:56 [core.py:1297] [shutdown] EngineCore: start mode=abort timeout=0s")
    assert not compact.feed("(Worker_TP0 pid=2) Exception in thread Thread-3:")
    assert not compact.feed("(Worker_TP0 pid=2) Traceback (most recent call last):")
    visible = compact.feed("(Worker_TP0 pid=2) RuntimeError: device execution failed")
    assert "Exception in thread" in "\n".join(visible)
    assert "RuntimeError: device execution failed" in "\n".join(visible)


@pytest.mark.parametrize("result", [
    {"status": "failed", "cleanup_complete": True, "exit_code": 0},
    {"status": "finished", "cleanup_complete": False, "exit_code": 0},
    {"status": "finished", "cleanup_complete": True, "exit_code": 1},
])
def test_compact_replays_tbe_eof_when_shutdown_did_not_succeed(result):
    compact = phase._CompactTerminalFilter()
    compact.feed("(EngineCore pid=1) INFO 09-23 05:19:56 [core.py:1297] [shutdown] EngineCore: start mode=abort timeout=0s")
    for line in (
        "(Worker_TP0 pid=2) Exception in thread Thread-3:",
        "(Worker_TP0 pid=2) Traceback (most recent call last):",
        '(Worker_TP0 pid=2)   File "/usr/local/Ascend/cann-9.1.0/python/site-packages/tbe/common/repository_manager/utils/multiprocess_util.py", line 68, in run',
        "(Worker_TP0 pid=2)     raise EOFError",
        "(Worker_TP0 pid=2) EOFError",
    ):
        assert compact.feed(line) == []
    visible = compact.feed("SERVICE_RESULT " + json.dumps(result))
    assert "Exception in thread" in "\n".join(visible)
    assert "EOFError" in "\n".join(visible)


def test_compact_replays_incomplete_tbe_trace_even_after_successful_shutdown():
    compact = phase._CompactTerminalFilter()
    compact.feed("(EngineCore pid=1) INFO 09-23 05:19:56 [core.py:1297] [shutdown] EngineCore: start mode=abort timeout=0s")
    compact.feed("(Worker_TP0 pid=2) Exception in thread Thread-3:")
    compact.feed("(Worker_TP0 pid=2) Traceback (most recent call last):")
    compact.feed('(Worker_TP0 pid=2)   File "/usr/local/Ascend/cann-9.1.0/python/site-packages/tbe/common/repository_manager/utils/multiprocess_util.py", line 68, in run')
    visible = compact.feed("SERVICE_RESULT " + json.dumps(
        {"status": "finished", "cleanup_complete": True, "exit_code": 0}))
    assert "Traceback (most recent call last)" in "\n".join(visible)


def test_compact_keeps_unexpected_startup_and_hccl_errors():
    state = {}
    assert phase._compact_should_mirror(
        "STARTUP_WAIT elapsed=286.0 health=HTTP 500 from /health", state)
    assert phase._compact_should_mirror(
        "Capturing CUDA graphs ERROR STARTUP_WAIT elapsed=286.0 health=<urlopen error [Errno 111] Connection refused>", state)
    assert phase._compact_should_mirror(
        "(Worker_TP0 pid=2) ERROR 09-23 05:22:59 [platform.py:1277] HCCL timeout", state)
    assert phase._compact_should_mirror("SERVICE_ERROR ServiceProbeError: startup failed", state)
    assert phase._compact_should_mirror("[oscar] CLEANUP_END owned_server complete=False exit=1", state)


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
