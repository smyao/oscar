"""Host concurrency contracts; no NPU, SDK or performance measurements.

Archive #37--49/#127/#128/#157/#163/#171: pages cannot be released on enqueue
or before an active read finishes. #182/#183: failures are explicit and cannot
produce successful external-prefix completion receipts.
"""
from threading import Event, Thread, get_ident
import time

import pytest

from oscar_ascend.offload.async_load import AsyncLoadError, AsyncLoadQueue


def wait_until(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, "background task did not reach its state"
        Event().wait(0.001)


def test_blocked_read_does_not_block_submit_or_poll():
    entered, release, foreground_done = Event(), Event(), Event()
    queue = AsyncLoadQueue()
    foreground = []

    def read():
        entered.set()
        assert release.wait(5)
        return "first loaded"

    def submit_and_poll():
        queue.submit(("next", 2), lambda: "second loaded")
        foreground.append(queue.poll())
        foreground_done.set()

    queue.submit(("first", 1), read)
    thread = Thread(target=submit_and_poll)
    try:
        assert entered.wait(5)
        thread.start()
        assert foreground_done.wait(5), "submit/poll waited for the blocked SDK read"
        assert foreground == [[]]
    finally:
        release.set()
        if thread.ident is not None:
            thread.join(5)
        queue.shutdown()
    assert queue.poll() == [(("first", 1), "first loaded"), (("next", 2), "second loaded")]
    assert queue.poll() == []


def test_setup_is_lazy_once_and_in_the_same_background_thread():
    calls = []
    main_thread = get_ident()
    queue = AsyncLoadQueue(lambda: calls.append(("setup", get_ident())))
    assert not queue.alive and not calls
    queue.submit("a", lambda: calls.append(("a", get_ident())))
    queue.submit("b", lambda: calls.append(("b", get_ident())))
    queue.shutdown()
    assert [name for name, _ in calls] == ["setup", "a", "b"]
    assert len({ident for _, ident in calls}) == 1
    assert calls[0][1] != main_thread
    assert queue.closed and not queue.alive and queue.drained
    assert queue.poll() == [("a", None), ("b", None)]


def test_poll_delivers_finished_read_while_next_read_is_still_running():
    entered, release = Event(), Event()
    queue = AsyncLoadQueue()

    def blocked():
        entered.set()
        assert release.wait(5)
        return "second bytes"

    queue.submit("first", lambda: "first bytes")
    queue.submit("second", blocked)
    try:
        assert entered.wait(5)  # The first receipt was already published.
        assert queue.poll() == [("first", "first bytes")]
        assert queue.poll() == []
        assert queue._keys == {"second"}
    finally:
        release.set()
        queue.shutdown()
    assert queue.poll() == [("second", "second bytes")]


def test_duplicate_pending_running_and_unconsumed_keys_are_rejected():
    entered, release = Event(), Event()
    queue = AsyncLoadQueue()

    def blocked():
        entered.set()
        assert release.wait(5)
        return 1

    queue.submit("running", blocked)
    try:
        assert entered.wait(5)
        queue.submit("queued", lambda: 2)
        for key in ("running", "queued"):
            with pytest.raises(ValueError, match="duplicate"):
                queue.submit(key, lambda: "duplicate")
        release.set()
        wait_until(lambda: len(queue._completed) == 2)
        with pytest.raises(ValueError, match="duplicate"):
            queue.submit("running", lambda: "duplicate")
        assert queue.poll() == [("running", 1), ("queued", 2)]
        # Delivered receipts do not leave an unbounded identity history.
        assert not queue._keys
        queue.submit("running", lambda: 3)
    finally:
        release.set()
        queue.shutdown()
    assert queue.poll() == [("running", 3)]


@pytest.mark.parametrize("cause", [RuntimeError("SDK failed"), SystemExit("worker exit")])
def test_read_failure_is_sticky_with_request_and_original_cause(cause):
    queue = AsyncLoadQueue()

    def fail():
        raise cause

    queue.submit(("request", 7), fail)
    wait_until(lambda: bool(queue._failures))
    for action in (queue.poll, lambda: queue.submit("new", lambda: None), queue.shutdown):
        with pytest.raises(AsyncLoadError, match="request.*7") as error:
            action()
        assert error.value.key == ("request", 7)
        assert error.value.phase == "read"
        assert error.value.__cause__ is cause
        assert error.value.completed == ()
    assert not queue.alive and queue.drained


def test_setup_failure_never_executes_queued_reads():
    entered, release = Event(), Event()
    cause = RuntimeError("set_device failed")
    calls = []

    def setup():
        entered.set()
        assert release.wait(5)
        raise cause

    queue = AsyncLoadQueue(setup)
    queue.submit("a", lambda: calls.append("a"))
    try:
        assert entered.wait(5)
        queue.submit("b", lambda: calls.append("b"))
        assert queue.poll() == []
    finally:
        release.set()
        with pytest.raises(AsyncLoadError) as error:
            queue.shutdown()
    assert not calls and not queue.alive and queue.drained
    assert error.value.__cause__ is cause
    assert [(key, phase) for key, phase, _ in error.value.failures] == [("a", "setup"), ("b", "setup")]


def test_shutdown_stops_new_submissions_and_drains_queued_reads():
    entered, release, shutdown_done = Event(), Event(), Event()
    queue = AsyncLoadQueue()
    calls = []

    def first():
        entered.set()
        assert release.wait(5)
        calls.append("first")
        return "first"

    def close():
        queue.shutdown()
        shutdown_done.set()

    queue.submit("first", first)
    queue.submit("second", lambda: calls.append("second"))
    thread = Thread(target=close)
    try:
        assert entered.wait(5)
        thread.start()
        wait_until(lambda: queue.closed)
        assert not shutdown_done.is_set()
        with pytest.raises(RuntimeError, match="closed"):
            queue.submit("third", lambda: None)
        assert queue.poll() == []
    finally:
        release.set()
        if thread.ident is not None:
            thread.join(5)
        queue.shutdown()
    assert shutdown_done.is_set() and not queue.alive
    assert calls == ["first", "second"]
    assert queue.poll() == [("first", "first"), ("second", None)]


def test_shutdown_drains_after_failure_and_preserves_successful_receipts():
    entered, release = Event(), Event()
    queue = AsyncLoadQueue()
    cause = ValueError("bad object size")
    calls = []

    def fail():
        entered.set()
        assert release.wait(5)
        raise cause

    queue.submit("bad", fail)
    try:
        assert entered.wait(5)
        queue.submit("already queued", lambda: calls.append("finished"))
    finally:
        release.set()
        with pytest.raises(AsyncLoadError) as error:
            queue.shutdown()
    assert calls == ["finished"] and not queue.alive and queue.drained
    assert error.value.__cause__ is cause
    assert error.value.completed == (("already queued", None),)
    with pytest.raises(AsyncLoadError) as again:
        queue.poll()
    assert again.value.completed == error.value.completed


def test_empty_shutdown_is_idempotent_without_starting_setup():
    queue = AsyncLoadQueue(lambda: pytest.fail("unused queue initialized a device"))
    assert queue.drained
    queue.shutdown()
    queue.shutdown()
    assert queue.closed and not queue.alive and queue.drained and queue.poll() == []
    with pytest.raises(RuntimeError, match="closed"):
        queue.submit("late", lambda: None)


def test_invalid_submission_cannot_start_work():
    queue = AsyncLoadQueue()
    with pytest.raises(TypeError):
        queue.submit([], lambda: None)
    with pytest.raises(TypeError, match="callable"):
        queue.submit("bad", None)
    assert not queue.alive and queue.poll() == []
    queue.shutdown()


def test_thread_start_failure_cannot_leave_unowned_pending_work(monkeypatch):
    from oscar_ascend.offload import async_load

    cause = RuntimeError("cannot create worker thread")

    def fail_start(self):
        raise cause

    monkeypatch.setattr(async_load.Thread, "start", fail_start)
    queue = AsyncLoadQueue()
    with pytest.raises(AsyncLoadError) as error:
        queue.submit("request", lambda: pytest.fail("unstarted work executed"))
    assert error.value.phase == "thread start"
    assert error.value.__cause__ is cause
    assert queue.closed and not queue.alive and not queue._pending and not queue._keys
    # Thread.start may also be interrupted after launching but before returning;
    # do not invent a completion fence in the foreground exception handler.
    assert not queue.drained
    with pytest.raises(AsyncLoadError):
        queue.shutdown()


def test_interrupted_join_and_false_alive_do_not_claim_buffers_are_drained(monkeypatch):
    entered, release = Event(), Event()
    queue = AsyncLoadQueue()

    def blocked():
        entered.set()
        assert release.wait(5)
        return "bytes finished"

    def interrupted_join():
        raise KeyboardInterrupt("shutdown interrupted")

    queue.submit("request", blocked)
    try:
        assert entered.wait(5)
        assert not queue.drained
        # Simulate CPython's signal-interrupted join case without sending a
        # process-wide signal: the thread indicator no longer proves ownership.
        with monkeypatch.context() as patch:
            patch.setattr(queue._thread, "join", interrupted_join)
            patch.setattr(queue._thread, "is_alive", lambda: False)
            with pytest.raises(KeyboardInterrupt, match="shutdown interrupted"):
                queue.shutdown()
            assert queue.closed and not queue.alive and not queue.drained
            assert queue.poll() == []
    finally:
        release.set()
        queue.shutdown()
    assert queue.drained
    assert queue.poll() == [("request", "bytes finished")]


def test_shutdown_requires_callback_fence_even_when_join_returns_early(monkeypatch):
    entered, release, shutdown_done = Event(), Event(), Event()
    queue = AsyncLoadQueue()

    def blocked():
        entered.set()
        assert release.wait(5)

    def close():
        queue.shutdown()
        shutdown_done.set()

    queue.submit("request", blocked)
    closer = Thread(target=close)
    try:
        assert entered.wait(5)
        with monkeypatch.context() as patch:
            patch.setattr(queue._thread, "join", lambda: None)
            patch.setattr(queue._thread, "is_alive", lambda: False)
            closer.start()
            wait_until(lambda: queue.closed)
            assert not queue.drained and not shutdown_done.is_set()
            release.set()
            assert shutdown_done.wait(5)
            closer.join(5)
    finally:
        release.set()
        if closer.ident is not None:
            closer.join(5)
        queue.shutdown()
    assert queue.drained
