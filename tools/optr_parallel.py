"""Archive #94/#120/#123/#166/#172/#174: bounded offline layer workers.

Only CPU control messages cross pipes. Each spawned process owns one NPU and
one layer at a time; tensors stay in its authenticated on-disk checkpoints.
The parent waits for every worker to exit before a serving PT can be published.
"""
from __future__ import annotations

from collections import deque
import math
import multiprocessing as mp
from multiprocessing.connection import wait
import os
import signal
import threading
import time


def worker_plan(config, layers):
    from tools.target_cli import target_env
    target_env(config)  # Validate this run's four physical cards, never infer them.
    if type(layers) is not int or layers < 0:
        raise ValueError("OptR pending layer count must be nonnegative")
    count = min(len(config["devices"]), layers)
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    return {"workers": count, "devices": config["devices"][:count],
            "cpu_threads_per_worker": min(4, max(1, cores // max(count, 1))),
            "start_method": "spawn", "unit": "complete_layer"}


class _Interrupted(BaseException):
    def __init__(self, signum):
        self.signum = signum


def _worker_entry(target, args):
    # Parent temporarily blocks cancellation while acquiring the process
    # handle. A spawned child must not retain that signal mask.
    if hasattr(signal, "pthread_sigmask"):
        signal.pthread_sigmask(signal.SIG_UNBLOCK, (signal.SIGINT, signal.SIGTERM, signal.SIGHUP))
    target(*args)


def _stop_workers(states, grace=5.):
    # Bound the whole pool's grace period, not five seconds per process.
    for state in states:
        process = state["process"]
        if process.pid is not None and process.is_alive():
            process.terminate()
    deadline = time.monotonic() + grace
    for state in states:
        process = state["process"]
        if process.pid is not None:
            process.join(max(0., deadline - time.monotonic()))
    for state in states:
        process = state["process"]
        if process.pid is not None and process.is_alive():
            process.kill()
    deadline = time.monotonic() + grace
    for state in states:
        process = state["process"]
        if process.pid is not None:
            process.join(max(0., deadline - time.monotonic()))
        state["pipe"].close()
    alive = [s["process"].pid for s in states if s["process"].pid is not None and s["process"].is_alive()]
    if alive:
        raise RuntimeError(f"OptR worker cleanup incomplete: {alive}")


def run_layer_workers(spec, names, *, progress, completed, timeout, worker_target=None):
    """Dynamically schedule complete layers, failing on any individual stall.

    `completed(name)` must authenticate the worker's saved layer. Even after
    it returns, publication remains the caller's job after all workers exit.
    Test targets require an explicitly test-only specification.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("OptR worker timeout must be finite and positive")
    if len(set(names)) != len(names) or any(name not in spec["names"] for name in names):
        raise ValueError("OptR scheduled layers must be unique and belong to this capture")
    if worker_target is not None and spec.get("testing") is not True:
        raise ValueError("injected OptR worker is test-only")
    if worker_target is None:
        from tools.optr_fit_worker import fit_worker
        worker_target = fit_worker
    plan = worker_plan(spec["config"], len(names))
    progress(phase="parallel_fit_plan", **plan)
    if not names:
        return
    pending, states = deque(names), []
    context = mp.get_context("spawn")
    handlers, interrupted = {}, None

    def interrupt(signum, _frame):
        raise _Interrupted(signum)

    def dispatch(state):
        name = pending.popleft() if pending else None
        state.update(layer=name, phase="working" if name else "stopping", last=time.monotonic())
        state["pipe"].send(name)

    try:
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                handlers[signum] = signal.signal(signum, interrupt)
        for index in range(plan["workers"]):
            parent, child = context.Pipe()
            process = context.Process(target=_worker_entry,
                args=(worker_target, (spec, index, plan["cpu_threads_per_worker"], child)), name=f"optr-fit-{index}")
            state = {"process": process, "pipe": parent, "index": index,
                     "phase": "starting", "layer": None, "last": time.monotonic(), "detail": {}}
            states.append(state)  # Keep ownership even if start is interrupted.
            mask = signal.pthread_sigmask(signal.SIG_BLOCK, tuple(handlers)) if handlers else None
            try:
                process.start()
            finally:
                child.close()
                if mask is not None:
                    signal.pthread_sigmask(signal.SIG_SETMASK, mask)
            progress(phase="fit_worker_started", worker=index, physical_device=plan["devices"][index], pid=process.pid)
        active = list(states)
        while active:
            ready = wait([s["pipe"] for s in active] + [s["process"].sentinel for s in active], timeout=.1)
            for state in list(active):
                pipe, process = state["pipe"], state["process"]
                if pipe in ready:
                    try:
                        message = pipe.recv()
                    except EOFError:
                        message = None
                        # EOF may precede Process.exitcode becoming visible.
                        process.join(2.)
                        if state["phase"] != "stopping":
                            raise RuntimeError(f"OptR worker {state['index']} lost its pipe at {state['layer']}; exitcode={process.exitcode}")
                    if message is not None:
                        kind = message.get("kind")
                        if kind == "ready" and state["phase"] == "starting":
                            dispatch(state)
                        elif kind == "progress" and message.get("layer") == state["layer"]:
                            state["last"] = time.monotonic()
                            state["detail"] = message["fields"]
                            progress(**{**message["fields"], "worker": state["index"],
                                        "physical_device": plan["devices"][state["index"]]})
                        elif kind == "done" and state["phase"] == "working" and message.get("layer") == state["layer"]:
                            completed(state["layer"])
                            dispatch(state)
                        else:
                            raise RuntimeError(f"Invalid OptR worker message: {message}")
                if process.exitcode is not None:
                    progress(phase="fit_worker_exited", worker=state["index"], pid=process.pid, returncode=process.exitcode)
                    if process.exitcode != 0 or state["phase"] != "stopping":
                        raise RuntimeError(f"OptR worker {state['index']} failed at {state['layer']}; exitcode={process.exitcode}")
                    process.join()
                    pipe.close()
                    active.remove(state)
                elif time.monotonic() - state["last"] >= timeout:
                    raise TimeoutError(f"OptR worker {state['index']} stalled for {timeout}s at {state['layer']}: {state['detail']}")
    except _Interrupted as error:
        interrupted = error
    finally:
        # Repeated terminal signals must not interrupt owned-child cleanup.
        for signum in handlers:
            signal.signal(signum, signal.SIG_IGN)
        try:
            _stop_workers(states)
            progress(phase="fit_workers_cleanup", cleanup_complete=True,
                     workers=[{"worker": s["index"], "pid": s["process"].pid,
                               "returncode": s["process"].exitcode} for s in states])
        finally:
            for signum, handler in handlers.items():
                signal.signal(signum, handler)
    if interrupted is not None:
        raise SystemExit(128 + interrupted.signum)
