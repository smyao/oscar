# 档案 #35/#51/#52/#75/#94/#95/#101/#116/#117/#120/#125/#135：有界进程组、保留原始退出码与完整日志；仅精确折叠成功关闭期噪声。
# #166/#167: the native process exit code is separate from a service failure verdict;
# service-specific evidence is classified before any supervisor cleanup signal.
"""Run a phase in an owned process group, always recording its outcome."""
from __future__ import annotations

import codecs
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from typing import Callable


@dataclass(frozen=True)
class PhaseResult:
    phase: str
    command: list[str]
    returncode: int
    elapsed_seconds: float
    timed_out: bool
    interrupted: bool
    log: str
    cleanup_complete: bool
    process_returncode: int | None = None
    failure: dict[str, object] | None = None


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def terminal_line(line: str, *, stderr: bool = False) -> None:
    """Write one whole line in a single write() call.

    The deploy parent and the service-probe child are separate processes that
    share one terminal; multi-call prints can split a line between them. Falls
    back to print when the stream has no real descriptor (test capture).
    """
    stream = sys.stderr if stderr else sys.stdout
    try:
        descriptor = stream.fileno()
    except (AttributeError, OSError):
        print(line, file=stream, flush=True)
        return
    view = memoryview((line + "\n").encode(errors="replace"))
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("terminal write returned 0 bytes")
        view = view[written:]


_COMPACT_ALERT = re.compile(
    r"\b(?:ERROR|FATAL|FAILED|RuntimeError|AssertionError|EngineDeadError|"
    r"Segfault|CLEANUP_ERROR|EXEC_ERROR|TIMEOUT)\b|fatal error:|^E\s+",
    re.IGNORECASE,
)
_TRACEBACK_END = re.compile(r"\b[A-Za-z_]*(?:Error|Exception):|\bKeyboardInterrupt\b")
_HCCL_TIMEOUT_INFO = re.compile(
    r"^\([^)]* pid=\d+\) INFO \d\d-\d\d \d\d:\d\d:\d\d \[platform\.py:\d+\] "
    r"The timeout interval of the HCCL operator is \d+s\. Timeout in seconds for "
    r"execute_model RPC calls in multiprocessing must be greater than \d+s, "
    r"Set VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=\d+$"
)
_STARTUP_CONNECTION_REFUSED = re.compile(
    r"^(?:Capturing CUDA graphs[^\n]*?)?STARTUP_WAIT elapsed=\d+(?:\.\d+)? "
    r"health=<urlopen error \[Errno 111\] Connection refused>$"
)
_SHUTDOWN_TIMEOUT_INFO = re.compile(
    r"^\([^)]* pid=\d+\) INFO \d\d-\d\d \d\d:\d\d:\d\d "
    r"\[[^]]+\] \[shutdown\] (?:API server: stopping engine client mode=abort "
    r"timeout=0s|EngineCore: start mode=abort timeout=0s|MPClient: start timeout=0s)$"
)
_TBE_THREAD_START = re.compile(r"^\(Worker_TP\d+ pid=\d+\) Exception in thread Thread-\d+:")
_TBE_WORKER_LINE = re.compile(r"^\(Worker_TP\d+ pid=\d+\) ")
_TBE_REPOSITORY_FRAME = "tbe/common/repository_manager/utils/multiprocess_util.py"
_TBE_MAX_LINES = 128
_TBE_MAX_SECONDS = 2.0


def _compact_should_mirror(line: str, state: dict[str, int]) -> bool:
    """Keep a small terminal stream while surfacing failures immediately.

    Archive #94/#95/#125: full tracebacks remain in the phase log, and a
    traceback is mirrored through its terminal exception line in real time.
    Normal INFO/build chatter is saved to disk without flooding the terminal.
    """
    if _HCCL_TIMEOUT_INFO.fullmatch(line):
        return False
    if (_STARTUP_CONNECTION_REFUSED.fullmatch(line)
            and not _COMPACT_ALERT.search(line.split("STARTUP_WAIT", 1)[0])):
        return False
    if _SHUTDOWN_TIMEOUT_INFO.fullmatch(line):
        return False
    if "STARTUP_WAIT " in line:
        return True
    if line.startswith(("SERVICE_ERROR ", "SERVICE_FAILURE ", "[oscar] CLEANUP_END owned_server complete=False")):
        return True
    if line.startswith(("START phase=", "START owned_service",
                        "RESULT ", "SERVICE_RESULT ")):
        return False
    if "Traceback (most recent call last)" in line or "Exception in thread" in line:
        state["traceback_left"] = 96
        return True
    if state.get("traceback_left", 0) > 0:
        state["traceback_left"] -= 1
        if _TRACEBACK_END.search(line):
            state["traceback_left"] = 0
        return True
    if line.startswith(("[oscar] PERF_", "[oscar] FAILED", "[oscar] REQUEST_ERROR",
                        "[oscar] SYNTHETIC_MIXED done", "[oscar-observe] OBSERVER_READY")):
        return True
    # #158: one host-only bridge receipt per worker may carry its native
    # Worker_TP prefix. Preserve that exact marker without mirroring INFO spam.
    if "[oscar] MTP_DISCARD_BRIDGE " in line:
        return True
    return bool(_COMPACT_ALERT.search(line))


class _CompactTerminalFilter:
    """Delay only a candidate TBE shutdown traceback until service exit is known.

    Archive #94/#95/#125 requires an unexpected traceback to remain visible.
    Therefore an incomplete, long, failed-shutdown or non-EOF traceback is
    replayed, while the durable log is never filtered. The small time/line cap
    also keeps an unexpected worker exception from being hidden indefinitely.
    """

    def __init__(self) -> None:
        self.state: dict[str, int] = {}
        self.shutdown_seen = False
        self.candidate: list[str] = []
        self.candidate_started = 0.0

    def _flush_candidate(self) -> list[str]:
        lines = self.candidate
        self.candidate = []
        if lines:
            self.state["traceback_left"] = 96
        return lines

    @staticmethod
    def _unexpected_candidate_error(line: str) -> bool:
        # EOFError is the one expected terminal exception. Another error on
        # the same interleaved line must still end buffering immediately.
        without_eof = line.replace("EOFError", "")
        return bool(_COMPACT_ALERT.search(without_eof) or _TRACEBACK_END.search(without_eof))

    def _candidate_is_known_tbe_eof(self) -> bool:
        if not self.candidate:
            return False
        body = "\n".join(self.candidate)
        starts = sum(bool(_TBE_THREAD_START.match(line)) for line in self.candidate)
        traces = body.count("Traceback (most recent call last):")
        frames = body.count(_TBE_REPOSITORY_FRAME)
        final_eofs = body.count("EOFError") - body.count("raise EOFError")
        return (starts > 0 and starts == traces == frames
                and final_eofs == starts
                and not any(self._unexpected_candidate_error(line)
                            for line in self.candidate))

    @staticmethod
    def _service_finished(line: str) -> bool:
        try:
            result = json.loads(line.removeprefix("SERVICE_RESULT "))
        except (ValueError, TypeError):
            return False
        return (result.get("status") == "finished"
                and result.get("cleanup_complete") is True
                and result.get("exit_code") == 0)

    def feed(self, line: str) -> list[str]:
        forwarded = self.expire()
        if "[shutdown]" in line:
            self.shutdown_seen = True
        if self.candidate and line.startswith("SERVICE_RESULT "):
            if not (self._service_finished(line) and self._candidate_is_known_tbe_eof()):
                forwarded.extend(self._flush_candidate())
            else:
                self.candidate = []
        elif self.candidate and _TBE_WORKER_LINE.match(line):
            self.candidate.append(line)
            if (len(self.candidate) > _TBE_MAX_LINES
                    or self._unexpected_candidate_error(line)):
                forwarded.extend(self._flush_candidate())
            return forwarded
        elif not self.candidate and self.shutdown_seen and _TBE_THREAD_START.match(line):
            self.candidate = [line]
            self.candidate_started = time.monotonic()
            return forwarded
        if _compact_should_mirror(line, self.state):
            forwarded.append(line)
        return forwarded

    def expire(self) -> list[str]:
        if (self.candidate and not self._candidate_is_known_tbe_eof()
                and time.monotonic() - self.candidate_started >= _TBE_MAX_SECONDS):
            return self._flush_candidate()
        return []

    def finish(self) -> list[str]:
        return self._flush_candidate()


@contextmanager
def live_log(path: Path, *, mode: str | None = None):
    """Keep a durable child log and mirror full or compact terminal output.

    Children inherit a regular file, never a pipe that can fill during cleanup
    or while the service supervisor is making a blocking HTTP request. A reader
    with its own file offset mirrors chunks. Nested phases naturally forward
    their output through the same outer log. Compact mode mirrors critical
    lines and tracebacks as they arrive, except a bounded candidate TBE EOF
    during graceful shutdown; the phase log remains complete.
    """
    stopped = threading.Event()
    failures = []
    terminal = sys.stdout
    mode = os.environ.get("OSCAR_TERMINAL_LOG_MODE", "full") if mode is None else mode
    if mode not in {"full", "compact"}:
        raise ValueError(f"invalid OSCAR_TERMINAL_LOG_MODE={mode!r}")
    with path.open("w", buffering=1, encoding="utf-8") as stream, path.open("rb") as reader:
        def mirror():
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            pending = ""
            compact_filter = _CompactTerminalFilter()

            def forward(decoded: str, *, final: bool = False) -> None:
                nonlocal pending
                if mode == "full":
                    if decoded:
                        terminal.write(decoded)
                        terminal.flush()
                    return
                pending += decoded
                while "\n" in pending:
                    line, pending = pending.split("\n", 1)
                    for visible in compact_filter.feed(line):
                        terminal_line(visible)
                if final and pending:
                    for visible in compact_filter.feed(pending):
                        terminal_line(visible)
                    pending = ""
                if final:
                    for visible in compact_filter.finish():
                        terminal_line(visible)

            try:
                while True:
                    chunk = reader.read(65536)
                    if chunk:
                        forward(decoder.decode(chunk))
                    elif stopped.is_set():
                        forward(decoder.decode(b"", final=True), final=True)
                        return
                    else:
                        if mode == "compact":
                            for visible in compact_filter.expire():
                                terminal_line(visible)
                        stopped.wait(0.05)
            except Exception as error:
                failures.append(error)

        worker = threading.Thread(target=mirror, name=f"oscar-log-{path.stem}", daemon=True)
        worker.start()
        try:
            yield stream
        finally:
            stream.flush()
            stopped.set()
            # Do not let a disconnected or blocked output consumer hang cleanup.
            worker.join(timeout=5)
            if worker.is_alive():
                raise RuntimeError(f"terminal log forwarding did not finish; full log={path}")
            if failures:
                raise RuntimeError(f"terminal log forwarding failed; full log={path}: {failures[0]}") from failures[0]


def group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def _reap_exiting_child(proc: subprocess.Popen, error: PermissionError) -> None:
    # Local Darwin evidence: killpg can return EPERM for a dying/zombie-only
    # owned group before waitpid(WNOHANG) can reap its leader. Reap only our
    # direct child, with a finite bound; EPERM itself never means "gone".
    if proc.poll() is None:
        try:
            proc.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            raise error


def _owned_group_exists(proc: subprocess.Popen) -> bool:
    try:
        return group_exists(proc.pid)
    except PermissionError as error:
        _reap_exiting_child(proc, error)
        return group_exists(proc.pid)  # one fresh probe; persistent EPERM fails


def _signal_owned_group(proc: subprocess.Popen, signum: int) -> bool:
    try:
        os.killpg(proc.pid, signum)
        return True
    except ProcessLookupError:
        return False
    except PermissionError as error:
        _reap_exiting_child(proc, error)
        if group_exists(proc.pid):
            raise error
        return False  # disappearance was established by ESRCH, not EPERM


def _wait_owned_group_exit(proc: subprocess.Popen, seconds: float) -> bool:
    """Require ESRCH even across Darwin's transient zombie-only group EPERM.

    A killed orphan can outlive the reaped leader by a small interval. The
    direct child's waitpid cannot reap that orphan. Retry observation within
    the existing shutdown bound; persistent denial still fails explicitly.
    """
    deadline = time.monotonic() + seconds
    while True:
        denied = None
        proc.poll()
        try:
            if not _owned_group_exists(proc):
                return True
        except PermissionError as error:
            denied = error
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if denied is not None:
                raise denied
            return False
        time.sleep(min(0.05, remaining))


def cleanup_group(proc: subprocess.Popen, grace: float) -> bool:
    """Only touch the session created by this runner; never match process names."""
    if _owned_group_exists(proc) and not _signal_owned_group(proc, signal.SIGTERM):
        return True
    if _wait_owned_group_exit(proc, grace):
        return True
    if _owned_group_exists(proc) and not _signal_owned_group(proc, signal.SIGKILL):
        return True
    try:
        proc.wait(timeout=max(grace, 1))
    except subprocess.TimeoutExpired:
        return False
    return _wait_owned_group_exit(proc, max(grace, 1))


def run_phase(name: str, command: list[str], *, cwd: Path, log_dir: Path,
              timeout: float | None, env: dict[str, str] | None = None,
              grace: float = 5, heartbeat: float = 15,
              classify_failure: Callable[[Path, int], dict[str, object] | None] | None = None) -> PhaseResult:
    # None is explicit: foreground serving has no lifetime deadline. Its
    # shutdown remains bounded by grace, exactly like installation phases.
    if (not command or (timeout is not None and (not math.isfinite(timeout) or timeout <= 0))
            or not math.isfinite(grace) or grace < 0
            or not math.isfinite(heartbeat) or heartbeat < 0):
        raise ValueError("phase requires a command, positive timeout (or None), and finite nonnegative grace/heartbeat")
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / f"{name}.log"
    started = time.monotonic()
    expired = interrupted = False
    proc = None
    cleaned = True
    rc = 127
    old_signals = {}
    interrupt_signal = None
    launching = False
    cleaning = False
    failure = None

    def on_signal(signum, _frame):
        nonlocal interrupt_signal
        if cleaning:
            # Cover the small interval before every OS handler is SIG_IGN.
            return
        if interrupt_signal is None:
            interrupt_signal = signum
        if launching:
            # Popen may have created the OS child before returning its handle.
            # Unwind only after ownership is assigned; do not mask signals in
            # the child or falsely report that no process needed cleanup.
            return
        raise KeyboardInterrupt(f"signal {signum}")

    environment = dict(os.environ if env is None else env)
    environment["PYTHONUNBUFFERED"] = "1"
    with live_log(logfile, mode=environment.get("OSCAR_TERMINAL_LOG_MODE")) as stream:
        stream.write(f"START phase={name} cwd={cwd.resolve()} command={json.dumps(command)}\n")
        try:
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                old_signals[sig] = signal.signal(sig, on_signal)
            launching = True
            launch_error = None
            try:
                proc = subprocess.Popen(command, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                                        stdout=stream, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                stream.write(f"OWNED_PROCESS phase={name} pid={proc.pid} pgid={proc.pid}\n")
            except BaseException as error:
                launch_error = error
                raise
            finally:
                launching = False
                if interrupt_signal is not None:
                    if launch_error is not None:
                        stream.write(f"EXEC_ERROR phase={name}: {type(launch_error).__name__}: {launch_error}\n")
                    raise KeyboardInterrupt(f"signal {interrupt_signal}") from launch_error
            next_heartbeat = started + heartbeat
            while proc.poll() is None:
                now = time.monotonic()
                if timeout is not None and now - started >= timeout:
                    expired = True
                    rc = 124
                    stream.write(f"TIMEOUT phase={name} limit={timeout}s\n")
                    break
                if heartbeat and now >= next_heartbeat:
                    terminal_line(f"[oscar] phase={name} running seconds={now-started:.1f} log={logfile}")
                    next_heartbeat = now + heartbeat
                time.sleep(0.1 if timeout is None else min(0.1, max(timeout-(now-started), 0.001)))
            if not expired:
                rc = int(proc.returncode)
        except KeyboardInterrupt as exc:
            interrupted = True
            rc = 128 + (interrupt_signal or signal.SIGINT)
            stream.write(f"INTERRUPTED phase={name}: {exc}\n")
        except OSError as exc:
            stream.write(f"EXEC_ERROR phase={name}: {exc}\n")
        finally:
            cleaning = True
            try:
                # Repeated Ctrl-C/TERM/HUP must not abort bounded cleanup and
                # strand workers after the owned leader has already exited.
                for sig in old_signals:
                    signal.signal(sig, signal.SIG_IGN)
                if proc is not None and classify_failure is not None:
                    try:
                        stream.flush()
                        # Freeze evidence before our TERM/KILL can generate
                        # worker death logs. A service may outlive the API
                        # leader, so EOF or post-cleanup log size is unsafe.
                        observed_bytes = os.fstat(stream.fileno()).st_size
                        failure = classify_failure(logfile, observed_bytes)
                    except Exception as error:
                        failure = {"kind": "failure_classification_error",
                                   "error": f"{type(error).__name__}: {error}"}
                    if failure is not None:
                        if rc == 0:
                            rc = 1
                        stream.write("SERVICE_FAILURE " + json.dumps(failure) + "\n")
            except BaseException as exc:
                if rc == 0:
                    rc = 125
                try:
                    stream.write(f"CLEANUP_ERROR phase={name}: {type(exc).__name__}: {exc}\n")
                except BaseException as log_error:
                    raise exc from log_error
            finally:
                # Failure classification and its logging are fallible. Neither
                # may skip cleanup, even when recording the first error fails.
                try:
                    if proc is not None:
                        cleaned = cleanup_group(proc, grace)
                except BaseException as exc:
                    cleaned = False
                    stream.write(f"CLEANUP_ERROR phase={name}: {type(exc).__name__}: {exc}\n")
                finally:
                    for sig, handler in old_signals.items():
                        signal.signal(sig, handler)
            if not cleaned and rc == 0:
                rc = 125
            result = PhaseResult(name, command, rc, round(time.monotonic()-started, 6),
                                 expired, interrupted, str(logfile.resolve()), cleaned,
                                 None if proc is None else proc.poll(), failure)
            stream.write("RESULT " + json.dumps(asdict(result)) + "\n")
            atomic_json(log_dir / f"{name}.json", asdict(result))
    state = "PASSED" if rc == 0 else "FAILED"
    process_code = f" process_rc={result.process_returncode}" if classify_failure is not None else ""
    terminal_line(f"[oscar] {state} phase={name} rc={rc}{process_code} log={logfile}")
    return result


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    return run_phase(args.name, command, cwd=args.cwd, log_dir=args.log_dir,
                     timeout=args.timeout).returncode


if __name__ == "__main__":
    sys.exit(main())
