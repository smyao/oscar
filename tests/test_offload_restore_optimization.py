"""Restore work moves off the model thread without weakening ownership.

Archive #37-49/#127/#128/#157/#163/#171: restore complete physical FULL and
GDN state before native admission, retaining pages through cancellation.
Archive #182/#183: a scheduler lookup is not an eviction-proof lease; missing
required payloads remain failures, while resident local pages need no read.
CPU byte transfers and pinned native completion seams are functional evidence,
not NPU, graph, service or performance acceptance. No timing thresholds.
"""
from copy import copy
from dataclasses import replace
from threading import Event, get_ident
from types import SimpleNamespace as NS

import pytest

from test_offload_async_bridge import (
    ProducerFence, assert_payload_bytes, empty_step, native_lifecycle,
    published_prefix,
)
from test_offload_cleanup import deliver, saved_tp_swap
from test_offload_native_bridge import bridge_module
from test_offload_native_completion import native_completion
from oscar_ascend.offload.async_load import AsyncLoadError
from oscar_ascend.offload.state import WorkerCheckpointTracker
from oscar_ascend.offload.transport import TransferError


def test_async_object_planning_can_block_without_blocking_model_thread(
        bridge_module, monkeypatch):
    (sender, worker, runner, scheduler, request, pages, store,
     expected, dispatch, _) = published_prefix(bridge_module, monkeypatch, 4)
    plan = dispatch.kv_connector_metadata.restores[0]
    native, _, _ = native_completion(monkeypatch, worker)
    entered, release = Event(), Event()
    model_thread = get_ident()
    original_objects = worker._objects
    threads, reads = [], []
    original_get = store.batch_get_into_layers

    def construct(payload, **kwargs):
        # A synchronous implementation fails immediately, rather than hanging
        # this test while waiting for an event only its caller can release.
        assert get_ident() != model_thread, "restore planning ran on the model thread"
        threads.append(get_ident())
        entered.set()
        assert release.wait(5), "test did not release background object planning"
        return original_objects(payload, **kwargs)

    def read(keys, addresses, lengths, direction):
        reads.extend(keys)
        return original_get(keys, addresses, lengths, direction)

    monkeypatch.setattr(worker, "_objects", construct)
    monkeypatch.setattr(store, "batch_get_into_layers", read)
    monkeypatch.setattr(bridge_module, "expected_keys",
        lambda *args: pytest.fail("async worker rebuilt all-TP expected keys"))
    try:
        pending = native.kv_connector_no_forward(dispatch, None)
        assert entered.wait(5)
        assert threads and not release.is_set() and not reads
        assert not pending.kv_connector_output.finished_recving
        assert not worker._restored_cursors
        # Re-dispatch cannot enqueue another writer for these destination pages.
        worker.metadata = dispatch.kv_connector_metadata
        with pytest.raises(RuntimeError, match="duplicate asynchronous restore"):
            worker.start_load_kv(None)
        release.set()
        worker._load_queue.shutdown()
        completed = native.kv_connector_no_forward(empty_step(sender), None)
        assert completed.kv_connector_output.finished_recving == {"r"}
        assert not worker._loading_request_ids
        assert threads == [threads[0]] and threads[0] != model_thread
        monkeypatch.setattr(worker, "_objects", original_objects)
        assert_payload_bytes(worker, plan, expected)
    finally:
        release.set()
        if worker._load_queue is not None:
            worker._load_queue.shutdown()


def test_same_dispatch_shares_one_producer_event_for_multiple_restores(
        bridge_module, monkeypatch):
    (sender, worker, runner, scheduler, request, pages, store,
     expected, dispatch, _) = published_prefix(bridge_module, monkeypatch, 0)
    first = dispatch.kv_connector_metadata.restores[0]
    second = replace(first, request_id="second", block_ids=(
        (0, 1, 2), (3, 4, 5, 6, 7, 8, 9)))
    dispatch.kv_connector_metadata = replace(dispatch.kv_connector_metadata,
        restores=(first, second))
    fence = ProducerFence()
    records, reads = [], []
    original_get = store.batch_get_into_layers

    def record():
        records.append(get_ident())
        return fence

    def read(keys, addresses, lengths, direction):
        reads.extend(keys)
        return original_get(keys, addresses, lengths, direction)

    monkeypatch.setattr(worker, "_record_producer_event", record)
    monkeypatch.setattr(store, "batch_get_into_layers", read)
    native, _, _ = native_completion(monkeypatch, worker)
    try:
        pending = native.kv_connector_no_forward(dispatch, None)
        assert fence.entered.wait(5)
        assert records == [get_ident()]
        assert not reads and not pending.kv_connector_output.finished_recving
        assert len(worker._async_loads) == 2
        assert worker._loading_request_ids == {"r", "second"}
        fence.release.set()
        worker._load_queue.shutdown()
        completed = native.kv_connector_no_forward(empty_step(sender), None)
        assert completed.kv_connector_output.finished_recving == {"r", "second"}
        assert records == [get_ident()]
        assert_payload_bytes(worker, first, expected)
        assert_payload_bytes(worker, second, expected)
        assert not worker._async_loads
        assert not worker._loading_request_ids
        assert not native.kv_connector_no_forward(empty_step(sender), None).kv_connector_output.finished_recving
    finally:
        fence.release.set()
        if worker._load_queue is not None:
            worker._load_queue.shutdown()


def test_async_partial_local_hit_ignores_evicted_unused_external_full_page(
        bridge_module, monkeypatch):
    (sender, worker, runner, scheduler, request, pages, store,
     expected, dispatch, all_objects) = published_prefix(bridge_module, monkeypatch, 4)
    plan = dispatch.kv_connector_metadata.restores[0]
    status, freed, cached = native_lifecycle(monkeypatch, bridge_module, scheduler, request, pages)
    # Lookup saw a complete joint checkpoint. Its first FULL page is already
    # resident, so later eviction of that external copy must not block restore.
    unused = all_objects[0].key
    del store.objects[unused]
    calls = []
    for method in ("batch_is_exist", "batch_get_key_info", "batch_get_into_layers"):
        original = getattr(store, method)
        def record(keys, *args, method=method, original=original):
            calls.append((method, tuple(keys)))
            return original(keys, *args)
        monkeypatch.setattr(store, method, record)
    native, Aggregator, _ = native_completion(monkeypatch, worker)
    fence = ProducerFence()
    worker._record_producer_event = lambda: fence
    try:
        native.kv_connector_no_forward(dispatch, None)
        assert fence.entered.wait(5)
        fence.release.set()
        worker._load_queue.shutdown()
        completed = native.kv_connector_no_forward(empty_step(sender), None)
        needed = tuple(obj.key for obj in all_objects[1:])
        assert calls == [("batch_get_key_info", needed), ("batch_get_into_layers", needed)]
        assert completed.kv_connector_output.finished_recving == {"r"}
        assert_payload_bytes(worker, replace(plan, local_tokens=0), expected)
        assert unused not in store.objects
        scheduler._update_from_kv_xfer_finished(Aggregator(1).aggregate([completed]).kv_connector_output)
        assert scheduler._try_promote_blocked_waiting_request(request)
        assert request.status == status.WAITING and cached == [("r", 8)] and not freed
    finally:
        fence.release.set()
        if worker._load_queue is not None:
            worker._load_queue.shutdown()


@pytest.mark.parametrize("failure", ["before_info", "after_info", "sdk_status"])
def test_one_tp_rank_failure_cannot_publish_joint_readiness_or_release_private_source(
        bridge_module, monkeypatch, failure):
    (sender, first, runner, scheduler, request, pages,
     snapshot, receipts, store) = saved_tp_swap(bridge_module, monkeypatch)
    deliver(sender, receipts[0].aggregate(receipts[1]))
    sender._init_load_state(True)
    first._init_load_state(True)
    first.rank = 0
    second = copy(first)
    second._init_load_state(True)
    second.rank = 1
    second._acks, second._prefix_acks, second._load_acks = [], [], []
    second.tracker = WorkerCheckpointTracker()
    assert sender.get_num_new_matched_tokens(request, 0) == (8, True)
    pages["r"] = ((0, 12, 13), (10, 11, 12, 13))
    sender.update_state_after_alloc(request, pages["r"], 8)
    request.num_computed_tokens = 8
    status, freed, cached = native_lifecycle(monkeypatch, bridge_module, scheduler, request, pages)
    dispatch = empty_step(sender)
    plan = dispatch.kv_connector_metadata.restores[0]
    first_keys = tuple(obj.key for obj in first._objects(plan, loading=True))
    second_keys = tuple(obj.key for obj in second._objects(plan, loading=True))
    bad_key = second_keys[0]
    original_get = store.batch_get_into_layers
    infos, reads = [], []

    def info(keys):
        infos.append(tuple(keys))
        result = [NS(size=lambda n=len(store.objects.get(key, b"")): n) for key in keys]
        if failure == "after_info" and bad_key in keys:
            del store.objects[bad_key]
        return result

    def read(keys, addresses, lengths, direction):
        reads.append(tuple(keys))
        if bad_key in keys:
            if failure == "sdk_status":
                return [-3001] * len(keys)
            if bad_key not in store.objects:
                return [-3007] * len(keys)
        return original_get(keys, addresses, lengths, direction)

    monkeypatch.setattr(store, "batch_get_key_info", info)
    monkeypatch.setattr(store, "batch_get_into_layers", read)
    if failure == "before_info":
        del store.objects[bad_key]
    workers = (first, second)
    fences = (ProducerFence(), ProducerFence())
    natives = []
    for worker, fence in zip(workers, fences):
        worker._record_producer_event = lambda fence=fence: fence
        native, Aggregator, _ = native_completion(monkeypatch, worker)
        natives.append(native)
    aggregate = Aggregator(2)
    try:
        pending = [native.kv_connector_no_forward(dispatch, None) for native in natives]
        assert all(fence.entered.wait(5) for fence in fences)
        assert not aggregate.aggregate(pending).kv_connector_output.finished_recving
        fences[0].release.set()
        first._load_queue.shutdown()
        waiting = empty_step(sender)
        results = [native.kv_connector_no_forward(waiting, None) for native in natives]
        assert results[0].kv_connector_output.finished_recving == {"r"}
        assert not results[1].kv_connector_output.finished_recving
        combined = aggregate.aggregate(results).kv_connector_output
        scheduler._update_from_kv_xfer_finished(combined)
        assert infos == [first_keys] and reads == [first_keys]
        assert not combined.finished_recving and not combined.finished_sending
        fences[1].release.set()
        with pytest.raises(AsyncLoadError):
            second._load_queue.shutdown()
        with pytest.raises(AsyncLoadError):
            natives[1].kv_connector_no_forward(empty_step(sender), None)
        assert infos == [first_keys, second_keys]
        assert reads == ([first_keys] if failure == "before_info" else [first_keys, second_keys])
        assert not second._load_acks and not second._restored_cursors
        assert not scheduler._try_promote_blocked_waiting_request(request)
        assert request.status == status.WAITING_FOR_REMOTE_KVS
        assert not freed and not cached and not scheduler.finished_recving_kv_req_ids
        record = sender._swap_payloads[snapshot.identity]
        assert set(record.loads) == {0}
        assert all(key in store.objects for key in first_keys)
    finally:
        for fence in fences:
            fence.release.set()
        if first._load_queue is not None:
            first._load_queue.shutdown()
        if second._load_queue is not None:
            with pytest.raises(AsyncLoadError):
                second._load_queue.shutdown()


def test_sync_restore_keeps_joint_precheck_before_any_destination_write(bridge_module, monkeypatch):
    (sender, worker, runner, scheduler, request, pages,
     snapshot, receipts, store) = saved_tp_swap(bridge_module, monkeypatch)
    deliver(sender, receipts[0].aggregate(receipts[1]))
    assert sender.get_num_new_matched_tokens(request, 0) == (8, False)
    sender.update_state_after_alloc(request, ((0, 12, 13), (10, 11, 12, 13)), 8)
    dispatch = empty_step(sender)
    plan = dispatch.kv_connector_metadata.restores[0]
    worker.rank = 1
    del store.objects[worker._objects(plan, loading=True)[0].key]
    worker.rank = 0
    worker.metadata = dispatch.kv_connector_metadata
    monkeypatch.setattr(store, "batch_get_key_info",
        lambda keys: pytest.fail("sync load queried local sizes before rejecting incomplete joint state"))
    monkeypatch.setattr(store, "batch_get_into_layers",
        lambda *args: pytest.fail("sync load wrote a partial joint checkpoint"))
    with pytest.raises(TransferError, match="joint checkpoint was evicted"):
        worker.start_load_kv(None)
    assert not worker._load_acks and worker._load_queue is None
