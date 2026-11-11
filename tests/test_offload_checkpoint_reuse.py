"""Archive #37-49/#127/#128/#157/#163/#171: retain exact native checkpoints.

#172/#174 permit allocation/host representation reuse, not changed arithmetic.
CPU byte comparisons and controlled producer events exercise ownership without
NPU timing, native source changes, or optimistic scheduler cursor assumptions.
"""
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from oscar_ascend.offload import state as state_module
from oscar_ascend.offload.scheduler import OffloadLifecycleError, RequestSnapshot
from oscar_ascend.offload.state import WorkerCheckpointTracker


@pytest.fixture
def device_events(monkeypatch):
    control = NS(stream=object(), events=[], allocations=[], recorded_streams=[])

    class Event:
        def __init__(self):
            self.complete = False
            self.queries = self.syncs = 0
            self.recorded_stream = None

        def record(self):
            self.recorded_stream = control.stream
            control.events.append(self)

        def query(self):
            self.queries += 1
            return self.complete

        def synchronize(self):
            self.syncs += 1
            self.complete = True

    original = torch.empty_like

    def allocate(source, *args, **kwargs):
        result = original(source, *args, **kwargs)
        control.allocations.append(result)
        return result

    monkeypatch.setattr(torch.cpu, "Event", Event)
    monkeypatch.setattr(torch.cpu, "current_stream", lambda: control.stream)
    monkeypatch.setattr(torch, "empty_like", allocate)
    monkeypatch.setattr(state_module, "_record_counter_stream",
                        lambda buffers, stream: control.recorded_streams.append((buffers, stream)))
    return control


def make_runner(mode="none", capacity=4):
    normalized = torch.zeros(capacity, dtype=torch.int64)
    return NS(
        device=torch.device("cpu"), cache_config=NS(mamba_cache_mode=mode),
        input_batch=NS(req_ids=[]), requests={}, mamba_state_idx={},
        num_computed_tokens=torch.zeros(capacity, dtype=torch.int64),
        num_accepted_tokens=NS(gpu=torch.zeros(capacity, dtype=torch.int32)),
        normalized=normalized,
        _get_mamba_bufs=lambda: NS(postprocess_align=NS(num_accepted_tokens_out=normalized)),
    )


def execute(tracker, runner, ids, step, *, bases=None, accepted=None,
            normalized=None, generations=None, speculative=True):
    count = len(ids)
    runner.input_batch.req_ids = list(ids)
    runner.num_computed_tokens[:count] = torch.tensor(bases or [step * 10 + i for i in range(count)])
    runner.num_accepted_tokens.gpu[:count] = torch.tensor(accepted or [2] * count)
    runner.normalized[:count] = torch.tensor(normalized or [1] * count)
    for rid in ids:
        runner.requests.setdefault(rid, NS(block_ids=([1, 2], [3, 4, 5, 6])))
        runner.mamba_state_idx.setdefault(rid, 0)
    output = NS(num_scheduled_tokens={rid: 4 for rid in ids},
                scheduled_spec_decode_tokens={rid: [1, 2, 3] for rid in ids} if speculative else {})
    tracker.record_step(runner, output, generations=generations or {rid: 1 for rid in ids}, step_id=step)


def snapshot(rid, *, mode="none", generation=1):
    return RequestSnapshot(rid, generation, 1, 0, (), 9999, 100, 100, "scheduler-hint", mode)


def bytes_of(tensor):
    return tensor.reshape(-1).view(torch.uint8).tolist()


@pytest.mark.parametrize("mode", ["none", "align"])
def test_steady_steps_reuse_two_slabs_preserving_counter_bytes_and_new_events(device_events, mode):
    runner, tracker = make_runner(mode), WorkerCheckpointTracker()
    addresses = set()
    previous_event = None
    for step in range(40):
        execute(tracker, runner, ["a", "b"], step, bases=[2**40 + step, 7 + step],
                accepted=[1, 4], normalized=[4, 1])
        for index, rid in enumerate(("a", "b")):
            record = tracker._records[rid]
            expected = (runner.num_computed_tokens[index], runner.num_accepted_tokens.gpu[index],
                        runner.normalized[index] if mode == "align" else runner.num_accepted_tokens.gpu[index])
            for actual, source in zip((record.base, record.accepted, record.next_accepted), expected):
                assert actual.dtype == source.dtype
                assert bytes_of(actual) == bytes_of(source)
            addresses.add(tuple(buffer.data_ptr() for buffer in record.counter_snapshot.buffers))
        assert record.ready_event is not previous_event
        previous_event = record.ready_event
    assert len(addresses) == 2
    assert len(device_events.allocations) == (6 if mode == "align" else 4)
    assert len(device_events.events) == 40
    assert not any(event.syncs or event.queries for event in device_events.events)


@pytest.mark.parametrize("mode", ["none", "align"])
def test_old_step_records_survive_reorder_native_overwrite_and_many_new_steps(device_events, monkeypatch, mode):
    runner, tracker = make_runner(mode), WorkerCheckpointTracker()
    execute(tracker, runner, ["held", "b"], 1, bases=[97, 300], accepted=[3, 4], normalized=[2, 4])
    held = tracker._records["held"]
    original_bytes = tuple(bytes_of(tensor) for tensor in (held.base, held.accepted, held.next_accepted))
    for step in range(2, 12):
        execute(tracker, runner, ["c", "b"], step)
        assert tracker._records["held"] is held
        assert tuple(bytes_of(tensor) for tensor in (held.base, held.accepted, held.next_accepted)) == original_bytes
        assert held.counter_snapshot not in tracker._free_counters
    seen = []
    monkeypatch.setattr(state_module, "copy_canonical_mamba",
                        lambda runner, blocks, running, accepted, **kwargs: seen.append((blocks, running, accepted)) or ())
    resolved = tracker.resolve(runner, snapshot("held", mode=mode))
    assert resolved.committed_tokens == 100 and resolved.step_id == 1
    assert seen == [(held.block_ids, 0, 2 if mode == "align" else 3)]
    assert held.ready_event.syncs == 1
    assert len(device_events.allocations) == (9 if mode == "align" else 6)


def test_true_prefill_resolve_still_uses_column_one_after_pool_reuse(device_events, monkeypatch):
    runner, tracker = make_runner("align"), WorkerCheckpointTracker()
    for step in range(3):
        execute(tracker, runner, ["r"], step, bases=[97], accepted=[4], normalized=[4], speculative=False)
    accepted_columns = []
    monkeypatch.setattr(state_module, "copy_canonical_mamba",
                        lambda runner, blocks, running, accepted, **kwargs: accepted_columns.append(accepted) or ())
    result = tracker.resolve(runner, snapshot("r", mode="align"))
    assert result.committed_tokens == 101
    assert accepted_columns == [1]


def test_incomplete_other_stream_slot_is_skipped_then_completed_slot_is_recorded(device_events):
    runner, tracker = make_runner(), WorkerCheckpointTracker()
    execute(tracker, runner, ["r"], 1)
    first = tracker._records["r"].counter_snapshot
    device_events.stream = object()
    execute(tracker, runner, ["r"], 2)
    execute(tracker, runner, ["r"], 3)
    assert tracker._records["r"].counter_snapshot is not first
    assert first.ready_event.queries == 1 and not first.ready_event.complete
    assert len(device_events.allocations) == 6
    assert device_events.recorded_streams == []
    for slot in tracker._free_counters:
        slot.ready_event.complete = True
    device_events.stream = object()
    previous_allocations = len(device_events.allocations)
    execute(tracker, runner, ["r"], 4)
    assert len(device_events.allocations) == previous_allocations
    assert device_events.recorded_streams == [(tracker._records["r"].counter_snapshot.buffers, device_events.stream)]
    assert not any(event.syncs for event in device_events.events)


def test_free_cache_is_bounded_across_layouts_and_pending_producers(device_events):
    tracker = WorkerCheckpointTracker()
    retained = []
    for capacity in range(1, 9):
        device_events.stream = object()
        runner = make_runner("align" if capacity % 2 else "none", capacity)
        runner.num_computed_tokens = runner.num_computed_tokens.to(torch.int32 if capacity % 3 else torch.int64)
        execute(tracker, runner, ["r"], capacity)
        record = tracker._records["r"]
        retained.append(record.counter_snapshot)
        assert record.base.dtype == runner.num_computed_tokens.dtype
        assert record.counter_snapshot.buffers[0].numel() == capacity
        assert len(tracker._free_counters) <= 2
    assert len({id(slot) for slot in retained}) == 8
    assert not any(event.syncs for event in device_events.events)
    tracker.clear()
    assert not tracker._records and not tracker._free_counters
    assert all(slot.references == 0 for slot in retained)


def test_forget_releases_only_last_owner_and_same_id_generation_gets_new_values(device_events):
    runner, tracker = make_runner(), WorkerCheckpointTracker()
    execute(tracker, runner, ["a", "b"], 1)
    old = tracker._records["a"]
    slab = old.counter_snapshot
    tracker.forget("a")
    assert slab.references == 1 and slab not in tracker._free_counters
    tracker.forget("absent")
    tracker.forget("b")
    assert slab.references == 0 and tracker._free_counters == [slab]
    execute(tracker, runner, ["a"], 2, bases=[500], generations={"a": 9})
    record = tracker._records["a"]
    assert record.counter_snapshot is slab
    assert record.generation == 9 and record.base.item() == 500
    assert record.ready_event is not old.ready_event
    with pytest.raises(OffloadLifecycleError, match="generation"):
        tracker.resolve(runner, snapshot("a", generation=1))


def test_block_reuse_checks_full_contents_and_keeps_independent_prior_snapshot(device_events, monkeypatch):
    runner, tracker = make_runner(), WorkerCheckpointTracker()
    calls = []
    original = state_module.freeze_block_ids
    monkeypatch.setattr(state_module, "freeze_block_ids", lambda blocks: calls.append(blocks) or original(blocks))
    execute(tracker, runner, ["r"], 1)
    first = tracker._records["r"].block_ids
    calls.clear()
    execute(tracker, runner, ["r"], 2)
    assert tracker._records["r"].block_ids is first and calls == []
    groups = runner.requests["r"].block_ids
    groups[0][0] = 99  # same list object, same length
    execute(tracker, runner, ["r"], 3)
    changed = tracker._records["r"].block_ids
    assert changed == ((99, 2), (3, 4, 5, 6)) and first == ((1, 2), (3, 4, 5, 6))
    assert changed[1] is first[1] and len(calls) == 1
    groups[0].extend([7, 8])
    execute(tracker, runner, ["r"], 4)
    assert tracker._records["r"].block_ids[0] == (99, 2, 7, 8)
    del groups[0][1:]
    execute(tracker, runner, ["r"], 5)
    assert tracker._records["r"].block_ids[0] == (99,)
    runner.requests["r"].block_ids = ([5], [6, 7])
    execute(tracker, runner, ["r"], 6)
    assert tracker._records["r"].block_ids == ((5,), (6, 7))
    runner.requests["r"].block_ids[1][0] = -1
    with pytest.raises(OffloadLifecycleError, match="nonnegative"):
        execute(tracker, runner, ["r"], 7)
    assert tracker._records["r"].step_id == 6


@pytest.mark.parametrize("value", [True, np.int32(1), np.int64(1), "1"])
def test_int_like_blocks_keep_original_conversion_semantics(device_events, value):
    runner, tracker = make_runner(), WorkerCheckpointTracker()
    execute(tracker, runner, ["r"], 1)
    runner.requests["r"].block_ids[0][0] = value
    execute(tracker, runner, ["r"], 2)
    assert tracker._records["r"].block_ids[0] == (1, 2)
    assert all(type(item) is int for item in tracker._records["r"].block_ids[0])
    runner.requests["r"].block_ids[0][0] = complex(1)
    with pytest.raises(TypeError):
        execute(tracker, runner, ["r"], 3)


def test_cross_stream_allocator_registration_uses_each_real_device_buffer():
    calls, stream = [], object()
    buffers = tuple(NS(device=NS(type="npu"), record_stream=lambda actual, i=i: calls.append((i, actual)))
                    for i in range(3))
    state_module._record_counter_stream(buffers, stream)
    assert calls == [(0, stream), (1, stream), (2, stream)]


@pytest.mark.parametrize("groups", [([-1], [complex(1)]), ([complex(1)], [-1])])
def test_invalid_block_conversion_precedes_negative_id_error(device_events, groups):
    runner, tracker = make_runner(), WorkerCheckpointTracker()
    execute(tracker, runner, ["r"], 1)
    runner.requests["r"].block_ids = groups
    with pytest.raises(TypeError):
        execute(tracker, runner, ["r"], 2)
    assert tracker._records["r"].step_id == 1
