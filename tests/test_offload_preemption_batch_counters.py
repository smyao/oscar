"""Ordered private swap saves share only immutable step counter readbacks.

Archive #37-49/#127/#128/#157/#163/#171/#172/#174: preserve accepted GDN
columns, private rollback, per-save producer barriers and before-reuse order.
Actual connector/transport methods copy CPU pointer bytes; controlled event
and read counts establish host behavior, not NPU or performance acceptance.
"""
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch

from oscar_ascend.offload.scheduler import OffloadLifecycleError, RequestSnapshot
from oscar_ascend.offload.transport import TransferError
from test_offload_checkpoint_reuse import device_events
from test_offload_native_bridge import bridge_module, connector_pair


def prepared(module, monkeypatch, count, mode, *, speculative=True):
    sender, worker, runner, tensors, group, store = connector_pair(module, monkeypatch, mode)
    ids = [f"r{i}" for i in range(count)]
    blocks = ((9, 10, 11), (1, 2, 3, 4, 5, 6, 7, 8))
    runner.input_batch.req_ids = ids
    runner.requests = {rid: NS(block_ids=blocks) for rid in ids}
    runner.mamba_state_idx = {rid: 4 for rid in ids}
    runner.num_computed_tokens = torch.full((count,), 6, dtype=torch.int64)
    runner.num_accepted_tokens.gpu = torch.tensor([i % 4 + 1 for i in range(count)], dtype=torch.int32)
    normalized = torch.tensor([(i + 2) % 4 + 1 for i in range(count)], dtype=torch.int32)
    runner._get_mamba_bufs = lambda: NS(postprocess_align=NS(num_accepted_tokens_out=normalized))
    output = NS(num_scheduled_tokens={rid: 4 for rid in ids},
                scheduled_spec_decode_tokens={rid: [91, 92, 93] for rid in ids} if speculative else {})
    worker.tracker.record_step(runner, output, generations={rid: 1 for rid in ids}, step_id=1)
    snapshots = tuple(RequestSnapshot(rid, 1, 1, 0, blocks, 10, 6, 11, "unused", mode) for rid in ids)
    metadata = module.OscarOffloadMetadata("test-session", 2, {}, preemptions=snapshots)
    return NS(sender=sender, worker=worker, runner=runner, store=store, metadata=metadata, output=output)


def read_counts(monkeypatch):
    counts = {"scalar_items": 0, "cpu_columns": []}
    original_item, original_cpu = torch.Tensor.item, torch.Tensor.cpu

    def item(tensor, *args, **kwargs):
        counts["scalar_items"] += 1
        return original_item(tensor, *args, **kwargs)

    def cpu(tensor, *args, **kwargs):
        counts["cpu_columns"].append(tuple(tensor.shape))
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "item", item)
    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    return counts


@pytest.mark.parametrize("count", [1, 16, 28])
@pytest.mark.parametrize("mode", ["none", "align"])
@pytest.mark.parametrize("speculative", [False, True])
def test_actual_private_swap_matches_scalar_bytes_and_receipts_with_bounded_reads(
        bridge_module, device_events, monkeypatch, count, mode, speculative):
    outcomes = []
    for legacy in (True, False):
        case = prepared(bridge_module, monkeypatch, count, mode, speculative=speculative)
        worker = case.worker
        event = worker.tracker._records["r0"].ready_event
        barriers, operations = [], []
        worker._synchronize = lambda: barriers.append("producer")
        if legacy:
            monkeypatch.setattr(worker.tracker, "resolve_many",
                lambda runner, snapshots, **kwargs: (
                    worker.tracker.resolve(runner, snapshot, **kwargs) for snapshot in snapshots))
        elif count == 1:
            monkeypatch.setattr(worker.tracker, "resolve_many",
                lambda *a, **k: pytest.fail("single preemption created a batch read mirror"))
        original_put, original_forget = case.store.batch_put_from_layers, worker.tracker.forget

        def put(keys, *args):
            operations.append(("put", len(worker._acks)))
            return original_put(keys, *args)

        def forget(rid):
            operations.append(("forget", rid, worker._acks[-1].request_id))
            original_forget(rid)

        monkeypatch.setattr(case.store, "batch_put_from_layers", put)
        monkeypatch.setattr(worker.tracker, "forget", forget)
        try:
            with monkeypatch.context() as mp:
                counts = read_counts(mp)
                worker.handle_preemptions(case.metadata)
            assert barriers == ["producer"] * count  # No private barrier was removed.
            assert operations == [event for i in range(count)
                                  for event in (("put", i), ("forget", f"r{i}", f"r{i}"))]
            assert not worker.tracker._records and case.store.put_calls == count
            assert [ack.committed_tokens for ack in worker._acks] == [
                6 + (i % 4 + 1 if speculative else 4) for i in range(count)]
            if legacy or count == 1:
                assert event.syncs == count
                assert counts == {"scalar_items": count * (3 if speculative else 2), "cpu_columns": []}
            else:
                assert event.syncs == 1
                assert counts == {"scalar_items": 0, "cpu_columns": [(count,)] * (
                    3 if speculative and mode == "align" else 2)}
            outcomes.append((dict(case.store.objects), tuple(worker._acks)))
        finally:
            worker.transport.close()
            case.sender.transport.close()
    assert outcomes[0] == outcomes[1]  # FULL packed/windows/tags and canonical GDN bytes.


@pytest.mark.parametrize("failure", ["generation", "missing", "accepted", "producer", "put"])
def test_later_private_failure_keeps_prior_ack_and_owned_remaining_record(
        bridge_module, device_events, monkeypatch, failure):
    case = prepared(bridge_module, monkeypatch, 2, "align")
    worker = case.worker
    # Give the second request a different step/stream event. Batch resolution
    # must not wait on that event until the first request is saved and released.
    first = worker.tracker._records["r0"]
    device_events.stream = object()
    case.runner.input_batch.req_ids = ["r1"]
    worker.tracker.record_step(case.runner, NS(num_scheduled_tokens={"r1": 4},
        scheduled_spec_decode_tokens={"r1": [91, 92, 93]}), generations={"r1": 1}, step_id=2)
    later = worker.tracker._records["r1"]
    original_wait = later.ready_event.synchronize
    waits, barriers, writes = [], [], []

    def wait_later():
        assert [ack.request_id for ack in worker._acks] == ["r0"]
        assert "r0" not in worker.tracker._records
        waits.append("later")
        if failure == "producer":
            raise RuntimeError("later producer failed")
        original_wait()

    later.ready_event.synchronize = wait_later
    worker._synchronize = lambda: barriers.append("producer")
    if failure == "generation":
        case.metadata.preemptions = (case.metadata.preemptions[0],
                                     replace(case.metadata.preemptions[1], generation=2))
    elif failure == "missing":
        worker.tracker.forget("r1")
    elif failure == "accepted":
        later.accepted.fill_(0)
    original_put = case.store.batch_put_from_layers

    def put(keys, addresses, lengths, direction):
        assert "r0" in worker.tracker._records if not writes else "r0" not in worker.tracker._records
        result = original_put(keys, addresses, lengths, direction)
        writes.append(tuple(keys))
        return [-3001] * len(keys) if failure == "put" and len(writes) == 2 else result

    monkeypatch.setattr(case.store, "batch_put_from_layers", put)
    try:
        with pytest.raises((OffloadLifecycleError, RuntimeError, TransferError)):
            worker.handle_preemptions(case.metadata)
        assert first.ready_event.syncs == 1
        assert [ack.request_id for ack in worker._acks] == ["r0"]
        assert "r0" not in worker.tracker._records
        assert ("r1" in worker.tracker._records) is (failure != "missing")
        assert waits == ([] if failure in ("generation", "missing") else ["later"])
        assert barriers == ["producer"] * (2 if failure == "put" else 1)
        assert set(case.store.objects) == set(writes[0])  # Failed second PUT rolls back only itself.
    finally:
        worker.transport.close()
        case.sender.transport.close()


def test_empty_preemptions_do_not_construct_or_read_counter_mirror(bridge_module, monkeypatch):
    case = prepared(bridge_module, monkeypatch, 1, "none")
    case.metadata.preemptions = ()
    for name in ("resolve", "resolve_many"):
        monkeypatch.setattr(case.worker.tracker, name,
                            lambda *a, **k: pytest.fail("empty preemptions invoked resolver"))
    case.worker._synchronize = lambda: pytest.fail("empty preemptions waited on a producer")
    try:
        case.worker.handle_preemptions(case.metadata)
        assert not case.worker._acks and not case.store.objects
        assert "r0" in case.worker.tracker._records
    finally:
        case.worker.transport.close()
        case.sender.transport.close()
