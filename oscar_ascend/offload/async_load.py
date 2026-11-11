"""One worker thread for synchronous offload reads with explicit completion.

Archive #37--49/#127/#128/#157/#163/#171: submitting work never publishes
readiness or releases physical pages. The caller owns immutable plans, producer
events, graph addresses and request generations until a completion is consumed.
Archive #182/#183: SDK failures remain failures, never cache hits or misses.
The callback performs the existing strict SDK read (with its fatal watchdog);
this queue neither times it out nor abandons a possibly active DMA operation.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Hashable
from threading import Condition, Event, Thread, current_thread
from typing import Any


class AsyncLoadError(RuntimeError):
    """A background failure, including successful receipts not yet consumed."""

    def __init__(self, failures, completed):
        self.failures = tuple(failures)
        self.completed = tuple(completed)
        self.key, self.phase, self.cause = self.failures[0]
        super().__init__(
            f"asynchronous KV load {self.phase} failed for request {self.key!r}: "
            f"{type(self.cause).__name__}: {self.cause} "
            f"(background failures={len(self.failures)})")


class AsyncLoadQueue:
    """Serialize reads off the model thread; poll never waits for SDK work.

    ``setup`` runs once, on the lazy worker thread, before its first operation.
    It may select the NPU device; operations must wait for their producer event
    and finish the synchronous SDK read before returning a receipt.

    Keys must include the caller's request/generation identity. A key cannot be
    resubmitted while queued, running, or awaiting receipt consumption. Once
    poll delivers it, identity retention belongs to the caller; this avoids an
    unbounded lifetime history inside the queue. No cancellation is implicit.
    """

    def __init__(self, setup: Callable[[], None] | None = None):
        if setup is not None and not callable(setup):
            raise TypeError("async load setup must be callable")
        self._setup = setup
        self._condition = Condition()
        self._pending: deque[tuple[Hashable, Callable[[], Any]]] = deque()
        self._completed: deque[tuple[Hashable, Any]] = deque()
        self._keys: set[Hashable] = set()
        self._failures: list[tuple[Hashable, str, BaseException]] = []
        self._thread: Thread | None = None
        self._closed = False
        # A Thread.join interrupted by a signal can make is_alive() unreliable
        # as an ownership proof. Only the callback thread's finally may release
        # this fence after a start attempt; an unused queue owns no callbacks.
        self._drained = Event()
        self._drained.set()

    @property
    def closed(self) -> bool:
        """Whether new submissions have been disabled (not a drain receipt)."""
        with self._condition:
            return self._closed

    @property
    def alive(self) -> bool:
        with self._condition:
            return self._thread is not None and self._thread.is_alive()

    @property
    def drained(self) -> bool:
        """Positive proof that no accepted callback can still touch buffers."""
        return self._drained.is_set()

    def _raise_failure(self) -> None:
        if self._failures:
            error = AsyncLoadError(self._failures, self._completed)
            raise error from error.cause

    def submit(self, key: Hashable, operation: Callable[[], Any]) -> None:
        hash(key)  # Validate before changing ownership or creating a thread.
        if not callable(operation):
            raise TypeError("async load operation must be callable")
        with self._condition:
            self._raise_failure()
            if self._closed:
                raise RuntimeError("async load queue is closed")
            if key in self._keys:
                raise ValueError(f"duplicate async load request {key!r}")
            self._keys.add(key)
            self._pending.append((key, operation))
            if self._thread is None:
                thread = Thread(target=self._run, name="oscar-kv-load", daemon=True)
                self._drained.clear()
                try:
                    thread.start()
                except BaseException as exc:
                    self._pending.pop()
                    self._keys.remove(key)
                    self._closed = True
                    self._failures.append((key, "thread start", exc))
                    self._raise_failure()
                self._thread = thread
            self._condition.notify()

    def poll(self) -> list[tuple[Hashable, Any]]:
        with self._condition:
            # Never consume successful receipts while reporting a failure.
            # They remain visible on AsyncLoadError.completed for teardown.
            self._raise_failure()
            completed = list(self._completed)
            self._completed.clear()
            self._keys.difference_update(key for key, _ in completed)
            return completed

    def _run(self) -> None:
        try:
            self._run_reads()
        finally:
            self._drained.set()

    def _run_reads(self) -> None:
        initialized = False
        setup_failure: BaseException | None = None
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending or self._closed)
                if not self._pending:
                    return
                key, operation = self._pending.popleft()
            if not initialized:
                try:
                    if self._setup is not None:
                        self._setup()
                except BaseException as exc:
                    # Even SystemExit in a worker must reach the model thread.
                    setup_failure = exc
                initialized = True
            phase = "setup" if setup_failure is not None else "read"
            try:
                if setup_failure is not None:
                    raise setup_failure
                result = operation()
            except BaseException as exc:
                with self._condition:
                    self._failures.append((key, phase, exc))
            else:
                with self._condition:
                    self._completed.append((key, result))
            finally:
                # Do not retain the last callback's buffers while idle.
                del operation

    def shutdown(self) -> None:
        """Drain every accepted callback before reporting errors or returning.

        No join timeout is used: returning while SDK work may still touch a
        buffer would authorize unsafe unregister/free. Its SDK watchdog bounds
        hung calls by terminating the process, rather than releasing memory.
        """
        with self._condition:
            if self._thread is current_thread():
                raise RuntimeError("async load worker cannot join itself")
            self._closed = True
            thread = self._thread
            self._condition.notify_all()
        if thread is not None:
            thread.join()
            # A prior interrupted join may have marked the Python Thread as
            # stopped while its callback still runs. Require its own fence too.
            self._drained.wait()
        with self._condition:
            self._raise_failure()
