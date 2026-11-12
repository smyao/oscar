"""Archive #37-49/#127/#128/#157/#163/#171/#172/#174: exact ordered saves.

Shared host counter reads change neither native committed-position arithmetic
nor GDN/page ownership. CPU pointer transfers, controlled events and operation
counts are functional evidence only, not NPU timing or SDK acceptance.
"""
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch

from oscar_ascend.offload import state as state_module
from oscar_ascend.offload.scheduler import OffloadLifecycleError
from oscar_ascend.offload.state import WorkerCheckpointTracker, _WorkerRecord
from oscar_ascend.offload.transport import TransferError
from test_offload_checkpoint_reuse import device_events, execute, make_runner, snapshot
from test_offload_native_bridge import bridge_module
from test_offload_prefix_single_writer import (
    close_cluster, deliver, make_cluster, metadata, native_hashes, save_rank,
)


def count_host_copies(monkeypatch):
    copies = []
    original = torch.Tensor.cpu

    def cpu(tensor, *args, **kwargs):
        copies.append((tuple(tensor.shape), tensor.dtype))
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    return copies


@pytest.mark.parametrize("count", [1, 16, 28])
@pytest.mark.parametrize("mode,speculative", [("none", False), ("align", False), ("none", True), ("align", True)])
def test_batch_reads_only_valid_rows_once_per_needed_column(
        device_events, monkeypatch, count, mode, speculative):
    runner, tracker = make_runner(mode, capacity=64), WorkerCheckpointTracker()
    ids = [str(index) for index in range(count)]
    bases = [2**40 + index for index in range(count)]
    raw = [1 + index % 4 for index in range(count)]
    normalized = [1 + (index + 1) % 4 for index in range(count)]
    execute(tracker, runner, ids, 1, bases=bases, accepted=raw, normalized=normalized,
            speculative=speculative)
    slot = tracker._records[ids[0]].counter_snapshot
    for buffer in slot.buffers:
        buffer[count:] = -999  # Unwritten capacity must never be copied to host.
    copies = count_host_copies(monkeypatch)
    columns = []
    monkeypatch.setattr(state_module, "copy_canonical_mamba",
        lambda runner, blocks, running, accepted, **kwargs: columns.append(accepted) or ())
    results = list(tracker.resolve_many(runner, [snapshot(rid, mode=mode) for rid in ids], clone=False))
    assert [item.committed_tokens for item in results] == [
        base + (accepted if speculative else 4) for base, accepted in zip(bases, raw)]
    expected_columns = normalized if speculative and mode == "align" else raw if speculative else [1] * count
    assert columns == expected_columns
    assert len(copies) == (3 if speculative and mode == "align" else 2)
    assert all(shape == (count,) for shape, _ in copies)
    assert device_events.events[0].syncs == 1
    assert slot.valid_rows == count and slot.references == count
    assert not tracker._free_counters


def test_cpu_mirror_is_independent_of_cpu_alias_and_limited_to_one_call(device_events, monkeypatch):
    runner, tracker = make_runner(), WorkerCheckpointTracker()
    execute(tracker, runner, ["a", "b"], 1, bases=[10, 20], speculative=False)
    monkeypatch.setattr(state_module, "copy_canonical_mamba", lambda *args, **kwargs: ())
    copies = count_host_copies(monkeypatch)
    records = [snapshot("a"), snapshot("b")]
    results = tracker.resolve_many(runner, records, clone=False)
    assert next(results).committed_tokens == 14
    slot = tracker._records["a"].counter_snapshot
    slot.buffers[0][1] = 500
    # Tensor.cpu() aliases a CPU tensor. tolist() must have made independent
    # Python values before the iterator yielded the first completed request.
    assert next(results).committed_tokens == 24
    assert len(copies) == 2
    assert list(tracker.resolve_many(runner, [snapshot("b")], clone=False))[0].committed_tokens == 504
    assert len(copies) == 4  # No persistent positive/value cache across calls.


def test_later_stream_is_not_waited_until_its_request_is_consumed(device_events, monkeypatch):
    runner, tracker = make_runner("align"), WorkerCheckpointTracker()
    execute(tracker, runner, ["first"], 1, bases=[10], speculative=False)
    first = tracker._records["first"]
    device_events.stream = object()
    execute(tracker, runner, ["later"], 2, bases=[20], speculative=False)
    later = tracker._records["later"]
    monkeypatch.setattr(state_module, "copy_canonical_mamba", lambda *args, **kwargs: ())
    copies = count_host_copies(monkeypatch)
    results = tracker.resolve_many(runner, [snapshot("first", mode="align"), snapshot("later", mode="align")])
    assert first.ready_event.syncs == later.ready_event.syncs == 0
    assert next(results).committed_tokens == 14
    assert first.ready_event.syncs == 1 and later.ready_event.syncs == 0
    assert len(copies) == 2
    assert next(results).committed_tokens == 24
    assert later.ready_event.syncs == 1 and len(copies) == 4


@pytest.mark.parametrize("failure", ["generation", "missing", "invalid_accepted", "producer"])
def test_late_failure_preserves_first_result_and_exception_order(device_events, monkeypatch, failure):
    runner, tracker = make_runner("align"), WorkerCheckpointTracker()
    execute(tracker, runner, ["a"], 1, bases=[10], accepted=[2])
    device_events.stream = object()
    execute(tracker, runner, ["b"], 2, bases=[20], accepted=[2])
    later = tracker._records["b"]
    second = snapshot("b", mode="align")
    if failure == "generation":
        second = replace(second, generation=2)
    elif failure == "missing":
        tracker.forget("b")
    elif failure == "invalid_accepted":
        later.accepted.fill_(0)
    else:
        def failed_producer():
            raise RuntimeError("later producer failed")
        later.ready_event.synchronize = failed_producer
    monkeypatch.setattr(state_module, "copy_canonical_mamba", lambda *args, **kwargs: ())
    results = tracker.resolve_many(runner, [snapshot("a", mode="align"), second], clone=False)
    assert next(results).committed_tokens == 12
    assert later.ready_event.syncs == 0
    with pytest.raises((OffloadLifecycleError, RuntimeError)):
        next(results)


def test_legacy_manual_record_uses_original_sync_and_scalar_conversion(monkeypatch):
    tracker, runner = WorkerCheckpointTracker(), make_runner()
    operations = []
    class Scalar:
        def __init__(self, name, value):
            self.name, self.value = name, value
        def item(self):
            operations.append(self.name)
            return self.value
    tracker._records["a"] = _WorkerRecord(1, 1, ((1,),), Scalar("base", 10), Scalar("raw", 2),
        Scalar("next", 2), 4, 3, 0, "none", NS(synchronize=lambda: operations.append("sync")))
    monkeypatch.setattr(state_module, "copy_canonical_mamba", lambda *args, **kwargs: ())
    assert list(tracker.resolve_many(runner, [snapshot("a")]))[0].committed_tokens == 12
    assert operations == ["sync", "base", "raw", "next"]


@pytest.mark.parametrize("row", [-1, 2, True])
def test_invalid_row_cannot_read_unused_capacity(device_events, monkeypatch, row):
    runner, tracker = make_runner(capacity=64), WorkerCheckpointTracker()
    execute(tracker, runner, ["a", "b"], 1)
    tracker._records["a"] = replace(tracker._records["a"], counter_row=row)
    copies = count_host_copies(monkeypatch)
    with pytest.raises(OffloadLifecycleError, match="row or completion"):
        next(tracker.resolve_many(runner, [snapshot("a")]))
    assert not copies and device_events.events[0].syncs == 0


def test_reused_pool_slot_uses_new_completion_identity(device_events, monkeypatch):
    runner, tracker = make_runner(), WorkerCheckpointTracker()
    execute(tracker, runner, ["a", "b"], 1, bases=[10, 20], speculative=False)
    monkeypatch.setattr(state_module, "copy_canonical_mamba", lambda *args, **kwargs: ())
    copies = count_host_copies(monkeypatch)
    results = tracker.resolve_many(runner, [snapshot("a"), snapshot("b")])
    assert next(results).committed_tokens == 14
    slot = tracker._records["b"].counter_snapshot
    tracker.forget("a")
    tracker.forget("b")
    execute(tracker, runner, ["b"], 2, bases=[200], speculative=False)
    assert tracker._records["b"].counter_snapshot is slot
    assert next(results).committed_tokens == 204
    assert [shape for shape, _ in copies] == [(2,), (2,), (1,), (1,)]


def distinct_prefixes(module, monkeypatch):
    cluster = make_cluster(module, monkeypatch)
    cluster.peer.all_token_ids = [900, *cluster.peer.all_token_ids[1:]]
    cluster.peer.block_hashes = native_hashes(monkeypatch, cluster.peer)
    output = metadata(cluster)
    assert len(output.kv_connector_metadata.prefixes) == 2
    return cluster, output


def test_prefix_batch_keeps_ordered_full_tp_publication_with_one_save_barrier(
        bridge_module, device_events, monkeypatch):
    cluster, output = distinct_prefixes(bridge_module, monkeypatch)
    syncs, copies = [], count_host_copies(monkeypatch)
    try:
        for rank, worker in enumerate(cluster.workers):
            worker._synchronize = lambda rank=rank: syncs.append(rank)
            receipts = save_rank(cluster, output, rank)
            assert [item.plan.request_id for item in receipts.prefixes] == ["r", "peer"]
            deliver(cluster, receipts)
            assert len(cluster.store.host_writes) == (0 if rank == 0 else 2)
        assert syncs == [0, 1]
        assert len(copies) == 4  # Two columns, once for each TP worker.
        assert all(event.syncs == 1 for event in device_events.events)
        assert len(cluster.store.data_writes) == 4  # Still synchronous per-request PUT.
        for request in (cluster.request, cluster.peer):
            assert cluster.sender.get_num_new_matched_tokens(request, 0) == (8, False)
    finally:
        close_cluster(cluster)


@pytest.mark.parametrize("failure", ["position", "second_put"])
def test_prefix_batch_late_failure_keeps_first_saved_ack_and_never_publishes_partial_tp(
        bridge_module, device_events, monkeypatch, failure):
    cluster, output = distinct_prefixes(bridge_module, monkeypatch)
    worker = cluster.workers[0]
    syncs = []
    worker._synchronize = lambda: syncs.append("sync")
    original = cluster.store.batch_put_from_layers
    if failure == "position":
        worker.runner.num_computed_tokens[1] = 1  # Second plan expects8; committed worker says9.
    else:
        def fail_second(*args):
            result = original(*args)
            return result if len(cluster.store.data_writes) == 1 else [-1] * len(result)
        monkeypatch.setattr(cluster.store, "batch_put_from_layers", fail_second)
    try:
        with pytest.raises((OffloadLifecycleError, TransferError)):
            save_rank(cluster, output, 0)
        receipts = worker.build_connector_worker_meta()
        assert [item.plan.request_id for item in receipts.prefixes] == ["r"]
        assert syncs == ["sync"]
        assert len(cluster.store.data_writes) == (1 if failure == "position" else 2)
        deliver(cluster, receipts)
        assert not cluster.store.host_writes
    finally:
        close_cluster(cluster)


def test_save_resolved_default_keeps_independent_private_barriers(bridge_module, monkeypatch):
    cluster, output = distinct_prefixes(bridge_module, monkeypatch)
    worker = cluster.workers[0]
    syncs = []
    worker._synchronize = lambda: syncs.append("sync")
    worker.tracker.record_step(worker.runner, output, generations=output.kv_connector_metadata.generations,
                               step_id=output.kv_connector_metadata.step_id)
    try:
        for plan in output.kv_connector_metadata.prefixes:
            state = worker.tracker._records[plan.request_id]
            resolved = worker.tracker.resolve(worker.runner,
                replace(snapshot(plan.request_id, mode="align", generation=plan.generation),
                        num_computed_hint=plan.committed_tokens), clone=False)
            worker._save_resolved(replace(plan, version=1), resolved)
            assert resolved.block_ids == state.block_ids
        assert syncs == ["sync", "sync"]
    finally:
        close_cluster(cluster)


def test_single_prefix_keeps_scalar_path_without_a_batch_mirror(bridge_module, monkeypatch):
    cluster = make_cluster(bridge_module, monkeypatch)
    output = metadata(cluster)
    assert len(output.kv_connector_metadata.prefixes) == 1
    worker = cluster.workers[0]
    copies, items, syncs = count_host_copies(monkeypatch), [], []
    original_item = torch.Tensor.item

    def scalar(tensor, *args, **kwargs):
        items.append(tuple(tensor.shape))
        return original_item(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "item", scalar)
    monkeypatch.setattr(worker.tracker, "resolve_many", lambda *args, **kwargs: pytest.fail("single prefix used batch resolver"))
    monkeypatch.setattr(state_module, "_CounterReadBatch", lambda: pytest.fail("single prefix allocated batch mirror"))
    worker._synchronize = lambda: syncs.append("sync")
    try:
        receipt = save_rank(cluster, output, 0)
        assert len(receipt.prefixes) == 1
        assert items == [(), ()] and not copies
        assert syncs == ["sync"]
    finally:
        close_cluster(cluster)


def test_no_prefix_records_state_without_constructing_batch_save_objects(bridge_module, monkeypatch):
    cluster = make_cluster(bridge_module, monkeypatch)
    output = metadata(cluster)
    output.kv_connector_metadata = replace(output.kv_connector_metadata, prefixes=())
    worker = cluster.workers[0]
    recorded = []
    original = worker.tracker.record_step

    def record(*args, **kwargs):
        recorded.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(worker.tracker, "record_step", record)
    for name in ("resolve", "resolve_many"):
        monkeypatch.setattr(worker.tracker, name, lambda *args, **kwargs: pytest.fail("empty prefix list invoked resolver"))
    monkeypatch.setattr(bridge_module, "RequestSnapshot", lambda *args, **kwargs: pytest.fail("empty prefix list allocated snapshot"))
    monkeypatch.setattr(state_module, "_CounterReadBatch", lambda: pytest.fail("empty prefix list allocated mirror"))
    worker._synchronize = lambda: pytest.fail("empty prefix list added a save barrier")
    try:
        receipt = save_rank(cluster, output, 0)
        assert recorded == [True] and not receipt.prefixes
        assert set(worker.tracker._records) == {"r", "peer"}
        assert not cluster.store.data_writes
    finally:
        close_cluster(cluster)
