"""Archive #94/#95/#120/#172: bound stalls using real offline work progress.

The launcher still owns the process group and total deadline. This watchdog
terminates only its own preparation process if numerical progress stops; a
periodic log heartbeat never extends the deadline.
"""
from __future__ import annotations

import json
import math
import os
import signal
import threading
import time


class ProgressWatchdog:
    def __init__(self, timeout, *, output=None, on_timeout=None, clock=time.monotonic):
        if isinstance(timeout, bool) or not isinstance(timeout, (float, int)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("OptR progress timeout must be finite and positive")
        self.timeout = timeout
        self.clock = clock
        self.output = output or self._print
        self.on_timeout = on_timeout or self._terminate_self
        self.lock = threading.Lock()
        self.changed = threading.Event()
        self.finished = False
        self.last = {"phase": "offline_fit_start"}
        self.last_time = clock()
        self.thread = threading.Thread(target=self._run, name="optr-progress-watchdog", daemon=True)

    @staticmethod
    def _print(**fields):
        print(json.dumps({"rotation_prepare": "optr", **fields}), flush=True)

    @staticmethod
    def _terminate_self():
        # run_phase reports the true signal return code and cleans all owned
        # descendants. Never kill a PID from a file or another service.
        os.kill(os.getpid(), signal.SIGTERM)

    def progress(self, **fields):
        with self.lock:
            self.last = dict(fields)
            self.last_time = self.clock()
        self.output(**fields)
        self.changed.set()

    def expired(self):
        with self.lock:
            return not self.finished and self.clock() - self.last_time >= self.timeout

    def _run(self):
        while True:
            with self.lock:
                if self.finished:
                    return
                remaining = self.timeout - (self.clock() - self.last_time)
                last = dict(self.last)
            if remaining <= 0:
                with self.lock:
                    if self.finished:
                        return
                    if self.clock() - self.last_time < self.timeout:
                        continue
                    self.finished = True
                    last = dict(self.last)
                try:
                    self.output(phase="FAILED", reason="offline_progress_timeout", timeout_seconds=self.timeout,
                                last_completed_or_started_work=last)
                finally:
                    self.on_timeout()
                return
            self.changed.wait(remaining)
            self.changed.clear()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        with self.lock:
            self.finished = True
        self.changed.set()
        self.thread.join(timeout=2)
