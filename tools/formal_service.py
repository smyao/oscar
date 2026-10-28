"""Archive G21/G22/#94/#95/#120/#174: own direct-service resources until exit.

No model/health probe, NPU context, inference hook, or service time limit.
Only the newly created session is signalled; an old PID or a process name is
never authority to kill a process. Device leases are local to this checkout.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from .phase import PhaseResult, atomic_json, cleanup_group, live_log, run_phase, terminal_line


@contextmanager
def device_lease(cwd: Path, devices: list[int], log_dir: Path):
    """Prevent overlapping direct launches without terminating a live owner."""
    if not devices or any(type(n) is not int or n < 0 for n in devices) or len(set(devices)) != len(devices):
        raise ValueError("device lease requires distinct explicit physical devices")
    directory = cwd / "logs" / ".direct-serve-locks"
    directory.mkdir(parents=True, exist_ok=True)
    handles = []
    try:
        for device in sorted(devices):
            path = directory / f"device-{device}.lock"
            handle = path.open("a+")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                handle.seek(0)
                owner = handle.read(4096)
                handle.close()
                raise RuntimeError(f"physical NPU {device} already has an active direct launcher "
                                   f"in this checkout; owner={owner!r}; lock={path}") from error
            except BaseException:
                handle.close()
                raise
            handles.append(handle)
            handle.seek(0)
            handle.truncate()
            json.dump({"pid": os.getpid(), "devices": devices, "log_dir": str(log_dir),
                       "scope": "direct launcher; no NPU ownership claim"}, handle)
            handle.flush()
        yield
    finally:
        # Do not unlink: another process may already be waiting on this inode.
        for handle in reversed(handles):
            handle.close()


def memory_failure_diagnostics(result: PhaseResult, config: dict, *, cwd: Path, log_dir: Path, env: dict):
    """Keep the native memory failure; collect bounded read-only evidence once."""
    if result.returncode == 0:
        return
    with Path(result.log).open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - 1024 * 1024))
        tail = stream.read().decode(errors="replace")
    failures = [line for line in tail.splitlines() if "Free memory on device (" in line
                and "is less than desired GPU memory utilization" in line]
    if not failures:
        return
    report = {"status": "native_startup_memory_rejected", "devices": config["devices"],
              "gpu_memory_utilization": config["gpu_memory_utilization"],
              "native_errors": list(dict.fromkeys(failures))[:16], "service_returncode": result.returncode,
              "ownership": "not_established_by_memory_error", "processes_terminated_by_diagnostic": [],
              "memory_budget_changed": False}
    path = log_dir / "serve-memory-failure.json"
    atomic_json(path, report)
    terminal_line("[oscar] selected NPUs have insufficient free memory before model loading; "
                  "keeping the configured budget and collecting read-only occupancy evidence. "
                  f"devices={config['devices']} report={path}", stderr=True)
    executable = shutil.which("npu-smi", path=env.get("PATH"))
    if executable is None:
        report["occupancy_evidence"] = {"status": "unavailable", "reason": "npu-smi executable not found"}
    else:
        observation = run_phase("serve-memory-diagnostics", [executable, "info"], cwd=cwd,
            log_dir=log_dir, timeout=10, grace=1, env=env, heartbeat=60)
        report["occupancy_evidence"] = asdict(observation)
    atomic_json(path, report)


class _ServiceSignal(BaseException):
    def __init__(self, signum):
        self.signum = signum


def run_service(command: list[str], *, cwd: Path, log_dir: Path, env: dict, grace: float) -> PhaseResult:
    """Wait without polling/timeout; clean the owned session even if API exits."""
    if not command or not math.isfinite(grace) or grace < 0:
        raise ValueError("service requires a command and finite nonnegative cleanup grace")
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / "serve.log"
    started = time.monotonic()
    process = None
    rc, interrupted, cleaned = 127, False, True
    previous = {}
    pending_signal = None
    cleaning = False
    lifecycle = {"status": "starting", "command": command, "timeout": None,
                 "cleanup_scope": "newly_created_process_group", "log": str(logfile)}
    environment = dict(env)
    environment["PYTHONUNBUFFERED"] = "1"

    def on_signal(signum, _frame):
        nonlocal pending_signal
        if cleaning or pending_signal is not None:
            return
        pending_signal = signum
        # Popen may already have forked before returning its handle. Do not
        # unwind that assignment and lose ownership of the just-created child.
        if process is not None:
            raise _ServiceSignal(pending_signal)

    with live_log(logfile, mode=environment.get("OSCAR_TERMINAL_LOG_MODE")) as stream:
        stream.write("START phase=serve command=" + json.dumps(command) + "\n")
        try:
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                previous[sig] = signal.signal(sig, on_signal)
            if pending_signal is not None:
                raise _ServiceSignal(pending_signal)
            process = subprocess.Popen(command, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            lifecycle.update(status="running", pid=process.pid, pgid=process.pid)
            if pending_signal is not None:
                raise _ServiceSignal(pending_signal)
            atomic_json(log_dir / "serve-lifecycle.json", lifecycle)
            # A permanent service has no phase deadline and no periodic NPU,
            # health, /proc or process polling in its supervisor.
            child_returncode = process.wait()
            rc = child_returncode if child_returncode >= 0 else 128 - child_returncode
        except _ServiceSignal as error:
            interrupted = True
            rc = 128 + error.signum
            lifecycle["signal"] = error.signum
        except OSError as error:
            lifecycle["error"] = f"{type(error).__name__}: {error}"
            stream.write("EXEC_ERROR phase=serve: " + lifecycle["error"] + "\n")
        finally:
            cleaning = True
            # A second Ctrl-C must not interrupt bounded owned-group cleanup.
            for sig in previous:
                signal.signal(sig, signal.SIG_IGN)
            try:
                if process is not None:
                    cleaned = cleanup_group(process, grace)
            except BaseException as error:
                cleaned = False
                lifecycle["cleanup_error"] = f"{type(error).__name__}: {error}"
                stream.write("CLEANUP_ERROR phase=serve: " + lifecycle["cleanup_error"] + "\n")
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
            if not cleaned and rc == 0:
                rc = 125
            result = PhaseResult("serve", command, rc, round(time.monotonic() - started, 6),
                                 False, interrupted, str(logfile.resolve()), cleaned)
            lifecycle.update(status="stopped" if rc == 0 else "failed", result=asdict(result),
                             child_returncode=None if process is None else process.poll())
            atomic_json(log_dir / "serve-lifecycle.json", lifecycle)
            atomic_json(log_dir / "serve.json", asdict(result))
            stream.write("RESULT " + json.dumps(asdict(result)) + "\n")
    terminal_line(f"[oscar] formal service exited rc={rc} cleanup_complete={cleaned} log={logfile}")
    return result
