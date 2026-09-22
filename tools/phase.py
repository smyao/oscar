# 档案 #35/#51/#52/#75/#94/#95/#101/#116/#117/#120：有界进程组、保留原始退出码、独立日志和固定 cwd。
"""Run a phase in an owned process group, always recording its outcome."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
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


def group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def cleanup_group(proc: subprocess.Popen, grace: float) -> bool:
    """Only touch the session created by this runner; never match process names."""
    if group_exists(proc.pid):
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return True
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        proc.poll()  # reap the direct child before checking the group
        if not group_exists(proc.pid):
            return True
        time.sleep(0.05)
    if group_exists(proc.pid):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return True
    try:
        proc.wait(timeout=max(grace, 1))
    except subprocess.TimeoutExpired:
        return False
    deadline = time.monotonic() + max(grace, 1)
    while group_exists(proc.pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    return not group_exists(proc.pid)


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

    with logfile.open("w", buffering=1) as stream:
        stream.write(f"START phase={name} cwd={cwd.resolve()} command={json.dumps(command)}\n")
        try:
            for sig in (signal.SIGTERM, signal.SIGINT):
                old_signals[sig] = signal.signal(sig, on_signal)
            proc = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
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
            if proc is not None:
                cleaned = cleanup_group(proc, grace)
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
