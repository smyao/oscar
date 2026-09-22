# 档案 #35/#51/#52/#75/#94/#95/#101/#116/#117/#120：有界进程组、保留原始退出码、独立日志和固定 cwd。
"""Run a phase in an owned process group, always recording its outcome."""
from __future__ import annotations

import codecs
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass


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


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def live_log(path: Path):
    """Keep a durable child log while mirroring new bytes to the terminal.

    Children inherit a regular file, never a pipe that can fill during cleanup
    or while the service supervisor is making a blocking HTTP request. A reader
    with its own file offset mirrors chunks, including incomplete lines. Nested
    phases naturally forward their output through the same outer log.
    """
    stopped = threading.Event()
    failures = []
    terminal = sys.stdout
    with path.open("w", buffering=1, encoding="utf-8") as stream, path.open("rb") as reader:
        def mirror():
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            try:
                while True:
                    chunk = reader.read(65536)
                    if chunk:
                        text = decoder.decode(chunk)
                        if text:
                            terminal.write(text)
                            terminal.flush()
                    elif stopped.is_set():
                        tail = decoder.decode(b"", final=True)
                        if tail:
                            terminal.write(tail)
                            terminal.flush()
                        return
                    else:
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


def cleanup_group(proc: subprocess.Popen, grace: float) -> bool:
    """Only touch the session created by this runner; never match process names."""
    if _owned_group_exists(proc) and not _signal_owned_group(proc, signal.SIGTERM):
        return True
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        proc.poll()  # reap the direct child before checking the group
        if not _owned_group_exists(proc):
            return True
        time.sleep(0.05)
    if _owned_group_exists(proc) and not _signal_owned_group(proc, signal.SIGKILL):
        return True
    try:
        proc.wait(timeout=max(grace, 1))
    except subprocess.TimeoutExpired:
        return False
    deadline = time.monotonic() + max(grace, 1)
    while _owned_group_exists(proc) and time.monotonic() < deadline:
        time.sleep(0.05)
    return not _owned_group_exists(proc)


def run_phase(name: str, command: list[str], *, cwd: Path, log_dir: Path,
              timeout: float, env: dict[str, str] | None = None,
              grace: float = 5, heartbeat: float = 15) -> PhaseResult:
    if not command or timeout <= 0 or grace < 0:
        raise ValueError("phase requires a command, positive timeout and nonnegative grace")
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / f"{name}.log"
    started = time.monotonic()
    expired = interrupted = False
    proc = None
    cleaned = True
    rc = 127
    old_signals = {}

    def on_signal(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")

    environment = dict(os.environ if env is None else env)
    environment["PYTHONUNBUFFERED"] = "1"
    with live_log(logfile) as stream:
        stream.write(f"START phase={name} cwd={cwd.resolve()} command={json.dumps(command)}\n")
        try:
            for sig in (signal.SIGTERM, signal.SIGINT):
                old_signals[sig] = signal.signal(sig, on_signal)
            proc = subprocess.Popen(command, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                                    stdout=stream, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            next_heartbeat = started + heartbeat
            while proc.poll() is None:
                now = time.monotonic()
                if now - started >= timeout:
                    expired = True
                    rc = 124
                    stream.write(f"TIMEOUT phase={name} limit={timeout}s\n")
                    break
                if now >= next_heartbeat:
                    print(f"[oscar] phase={name} running seconds={now-started:.1f} log={logfile}", flush=True)
                    next_heartbeat = now + heartbeat
                time.sleep(min(0.1, max(timeout-(now-started), 0.001)))
            if not expired:
                rc = int(proc.returncode)
        except KeyboardInterrupt as exc:
            interrupted = True
            rc = 130
            stream.write(f"INTERRUPTED phase={name}: {exc}\n")
        except OSError as exc:
            stream.write(f"EXEC_ERROR phase={name}: {exc}\n")
        finally:
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
                                 expired, interrupted, str(logfile.resolve()), cleaned)
            stream.write("RESULT " + json.dumps(asdict(result)) + "\n")
            atomic_json(log_dir / f"{name}.json", asdict(result))
    state = "PASSED" if rc == 0 else "FAILED"
    print(f"[oscar] {state} phase={name} rc={rc} log={logfile}", flush=True)
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
