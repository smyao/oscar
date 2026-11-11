"""Registered destinations must outlive background reads, including errors."""
from threading import Event, Thread
from types import SimpleNamespace as NS

import pytest

from oscar_ascend.offload.async_load import AsyncLoadQueue, AsyncLoadError
from test_offload_native_bridge import bridge_module, connector_pair


@pytest.mark.parametrize("failure", [False, True])
def test_connector_drains_before_unregister_and_preserves_background_error(
        bridge_module, monkeypatch, failure):
    _, worker, *_ = connector_pair(bridge_module, monkeypatch, "none")
    entered, release, stopping = Event(), Event(), Event()
    order, errors = [], []
    def read():
        entered.set()
        assert release.wait(5)
        order.append("read_done")
        if failure:
            raise ValueError("SDK failed")
    queue = worker._load_queue = AsyncLoadQueue()
    worker.metadata_client = NS(close=lambda: order.append("metadata_close"))
    close = worker.transport.close
    def close_transport():
        assert queue.drained and order[0] == "read_done"
        order.append("unregister")
        close()
    worker.transport.close = close_transport
    queue.submit(("r", 1), read)
    assert entered.wait(5)
    def stop():
        stopping.set()
        try:
            worker.shutdown()
        except BaseException as error:
            errors.append(error)
    thread = Thread(target=stop)
    thread.start()
    try:
        assert stopping.wait(5)
        assert not queue.drained and not order
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert order == ["read_done", "metadata_close", "unregister"]
    assert queue.drained and not worker.transport._owners
    if failure:
        assert len(errors) == 1 and isinstance(errors[0], AsyncLoadError)
        assert isinstance(errors[0].__cause__, ValueError)
    else:
        assert not errors


def test_interrupted_connector_shutdown_retains_registered_owners_until_retry(
        bridge_module, monkeypatch):
    _, worker, *_ = connector_pair(bridge_module, monkeypatch, "none")
    entered, release = Event(), Event()
    closed = []
    worker.metadata_client = NS(close=lambda: closed.append(True))
    queue = worker._load_queue = AsyncLoadQueue()
    def read():
        entered.set()
        assert release.wait(5)
    queue.submit(("r", 1), read)
    assert entered.wait(5)
    original_join = queue._thread.join
    def interrupted_join():
        raise KeyboardInterrupt()
    monkeypatch.setattr(queue._thread, "join", interrupted_join)
    owners = tuple(worker.transport._owners)
    try:
        with pytest.raises(KeyboardInterrupt):
            worker.shutdown()
        assert not queue.drained and not closed
        assert len(worker.transport._owners) == len(owners)
        assert all(a is b for a, b in zip(worker.transport._owners, owners))
        assert not worker.transport._closed
    finally:
        monkeypatch.setattr(queue._thread, "join", original_join)
        release.set()
        worker.shutdown()
    assert queue.drained and closed == [True]
    assert not worker.transport._owners and worker.transport._closed
