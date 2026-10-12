#!/usr/bin/env python3
"""Run one bounded CAModel arm inside the VM, independently of an SSH pipe.

Archive #94/#95: persist the real return code and both logs even on failure.
Archive #129/#132: a hard wall deadline and owned-process cleanup are separate
from simulated device timing. This tool never retries or interprets accuracy.

Launch this supervisor with LD_PRELOAD absent. Pass simulator preload settings
only in --env-json, which is applied to the child. The caller may detach this
script using Popen(start_new_session=True) with stdin=DEVNULL and regular files
for stdout/stderr, then poll --control-path and its supervisor_pid.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import traceback


LOCK_PATH = Path("/tmp/oscar-cann-validation.lock")
MAX_TIMEOUT = 900.0
CLEANUP_GRACE = 12.0


def atomic_json(path: Path, value: dict) -> None:
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # A group transitioning to zombie-only can produce EPERM on macOS.
        # It is not evidence of absence: keep waiting and reap our child.
        return True


def signal_group(pgid: int, signum: int, actions: list) -> None:
    try:
        os.killpg(pgid, signum)
        actions.append(signal.Signals(signum).name)
    except ProcessLookupError:
        pass


def cleanup(child: subprocess.Popen, actions: list) -> bool:
    """Only the separately-created child process group is eligible for signals."""
    if group_alive(child.pid):
        signal_group(child.pid, signal.SIGTERM, actions)
        # The collector owns a separate SDK session: TERM lets it forward the
        # signal, bound the runtime stop to 5 s and finish FIFO draining in 5 s.
        # These 12 s are cleanup only, never additional computation budget.
        grace = time.monotonic() + CLEANUP_GRACE
        while time.monotonic() < grace:
            child.poll()  # Reap our direct child, including after natural exit.
            if not group_alive(child.pid):
                break
            time.sleep(0.02)
    if group_alive(child.pid):
        signal_group(child.pid, signal.SIGKILL, actions)
    try:
        child.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        return False
    end = time.monotonic() + 1.0
    while group_alive(child.pid) and time.monotonic() < end:
        time.sleep(0.02)
    return not group_alive(child.pid)


def run(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    control = Path(args.control_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    control.parent.mkdir(parents=True, exist_ok=True)
    # A repeated launch cannot silently replace an earlier arm's evidence.
    with control.open("x") as initial:
        initial.write("{}\n")
    started = time.monotonic()
    state = {
        "schema_version": 1, "status": "starting", "supervisor_pid": os.getpid(),
        "child_pid": None, "child_pgid": None, "command": args.command,
        "cwd": str(Path(args.cwd).resolve()), "timeout_seconds": args.timeout,
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stdout_path": str(output / "stdout.log"),
        "stderr_path": str(output / "stderr.log"),
        "lock_path": str(LOCK_PATH), "lock_acquired": False,
        "timed_out": False, "interrupt_signal": None,
        "child_returncode": None, "cleanup_complete": None,
        "cleanup_grace_seconds": CLEANUP_GRACE,
        "cleanup_signals": [], "supervisor_returncode": None,
    }
    atomic_json(control, state)
    child = None
    lock = None
    code = 2

    def interrupted(signum, _frame):
        if state["interrupt_signal"] is None:
            state["interrupt_signal"] = signum

    previous = {sig: signal.signal(sig, interrupted)
                for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        if os.environ.get("LD_PRELOAD"):
            raise RuntimeError("supervisor must start without LD_PRELOAD; use child --env-json")
        if not math.isfinite(args.timeout) or not 0 < args.timeout <= MAX_TIMEOUT:
            raise ValueError("timeout must be finite, positive, and at most 900 seconds")
        if not args.command:
            raise ValueError("provide exactly one command after --")
        child_env = dict(os.environ)
        changes = json.loads(Path(args.env_json).read_text()) if args.env_json else {}
        if not isinstance(changes, dict):
            raise ValueError("env-json must contain an object of string or null values")
        for key, value in changes.items():
            if not isinstance(key, str) or not key or "=" in key or "\0" in key:
                raise ValueError("invalid environment variable name")
            if value is None:
                child_env.pop(key, None)
            elif isinstance(value, str) and "\0" not in value:
                child_env[key] = value
            else:
                raise ValueError("environment values must be strings or null")
        state["child_env_override_keys"] = sorted(changes)
        lock = LOCK_PATH.open("a+")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state["lock_acquired"] = True
        with (output / "stdout.log").open("xb", buffering=0) as stdout, \
             (output / "stderr.log").open("xb", buffering=0) as stderr:
            if state["interrupt_signal"] is None:
                child = subprocess.Popen(
                    args.command, cwd=state["cwd"], env=child_env,
                    stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                    start_new_session=True, close_fds=True,
                )
                state.update(status="running", child_pid=child.pid, child_pgid=child.pid)
                child_started = time.monotonic()
                state["child_started_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
                atomic_json(control, state)
                while child.poll() is None and state["interrupt_signal"] is None:
                    if time.monotonic() - child_started >= args.timeout:
                        state["timed_out"] = True
                        break
                    time.sleep(min(0.05, max(0.001, args.timeout -
                                           (time.monotonic() - child_started))))
                state["child_elapsed_seconds"] = time.monotonic() - child_started
            code = 0
    except BaseException as error:
        state["error"] = f"{type(error).__name__}: {error}"
        state["traceback"] = traceback.format_exc()
        print(state["traceback"], file=sys.stderr, flush=True)
    finally:
        if child is not None:
            try:
                state["cleanup_complete"] = cleanup(child, state["cleanup_signals"])
                state["child_returncode"] = child.returncode
            except BaseException as error:
                state["cleanup_complete"] = False
                state["cleanup_error"] = f"{type(error).__name__}: {error}"
                state["cleanup_traceback"] = traceback.format_exc()
        else:
            state["cleanup_complete"] = True
        if state["interrupt_signal"] is not None:
            code = 128 + state["interrupt_signal"]
        elif state["timed_out"]:
            code = 124
        elif not state["cleanup_complete"] or "error" in state:
            code = 2
        elif state["child_returncode"] is not None:
            rc = state["child_returncode"]
            code = rc if rc >= 0 else 128 - rc
        state.update(status="completed" if code == 0 else "failed",
                     supervisor_returncode=code,
                     elapsed_seconds=time.monotonic() - started,
                     finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        atomic_json(control, state)
        if lock is not None:
            lock.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Directory for stdout.log and stderr.log")
    parser.add_argument("--control-path", required=True, help="Exclusive atomic status JSON")
    parser.add_argument("--timeout", type=float, default=MAX_TIMEOUT)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--env-json", help="JSON file with child-only environment overrides")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
