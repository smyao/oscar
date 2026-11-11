"""Async restore ownership through pinned native completion/scheduler seams.

Archive #37-49/#127/#128/#157/#163/#171/#182/#183: enqueue is not completion,
empty steps must drain receipts, and cancelled destinations remain owned until
all TP readers finish. These tests move real CPU tensor bytes through the SDK
shaped bridge; they are not NPU, graph, service or performance acceptance.
"""
from copy import copy
from dataclasses import replace
import ctypes
import logging
from threading import Event
from types import MethodType, SimpleNamespace as NS

import pytest

from test_offload_native_bridge import (
    bridge_module, connector_pair, native_postprocess, scheduled, scheduler_for,
)
from test_offload_native_completion import native_completion
from test_offload_cleanup import saved_tp_swap, deliver
from test_runtime_native import definitions
from oscar_ascend.offload.state import WorkerCheckpointTracker
from oscar_ascend.offload.async_load import AsyncLoadError


def empty_step(sender, finished=()):
    output = NS(num_scheduled_tokens={}, scheduled_spec_decode_tokens={},
                finished_req_ids=set(finished), preempted_req_ids=set(),
                total_num_scheduled_tokens=0)
    output.kv_connector_metadata = sender.build_connector_meta(output)
    return output


def native_lifecycle(monkeypatch, module, scheduler, request, pages):
    """Use native cancellation, all-rank completion and waiting promotion."""
    status = NS(RUNNING="running", WAITING="waiting", PREEMPTED="preempted",
                WAITING_FOR_REMOTE_KVS="remote", WAITING_FOR_STREAMING_REQ="streaming",
                FINISHED_ABORTED="aborted", is_finished=lambda value: value == "aborted")
    methods = ("finish_requests", "_free_request", "_free_blocks", "_connector_finished",
               "_update_from_kv_xfer_finished", "_update_waiting_for_remote_kv",
               "_try_promote_blocked_waiting_request")
    native = definitions("references/vllm/vllm/v1/core/sched/scheduler.py",
        "offload_async_native_scheduler", set(),
        {"RequestStatus": status, "SupportsHMA": module.SupportsHMA,
         "logger": logging.getLogger(__name__),
         "remove_all": lambda seq, removed: [item for item in seq if item not in removed]},
        monkeypatch, methods=[("Scheduler", name) for name in methods])
    for name in methods:
        setattr(scheduler, name, MethodType(getattr(native, name), scheduler))
    scheduler.running = []
    scheduler.waiting.remove_requests = lambda requests: None
    scheduler.skipped_waiting = NS(remove_requests=lambda requests: None)
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler.finished_recving_kv_req_ids = set()
    scheduler.failed_recving_kv_req_ids = set()
    scheduler.kv_cache_manager.get_block_ids = lambda rid: pages[rid]
    scheduler.kv_cache_manager.remove_skipped_blocks = lambda **kwargs: None
    freed, cached = [], []
    def free(req):
        freed.append(req.request_id)
        pages.pop(req.request_id)
    scheduler.kv_cache_manager.free = free
    scheduler.kv_cache_manager.cache_blocks = lambda req, count: cached.append((req.request_id, count))
    request.status = status.WAITING_FOR_REMOTE_KVS
    request.client_index = 0
    request.num_tokens = len(request.all_token_ids)
    request.is_finished = lambda: status.is_finished(request.status)
    return status, freed, cached


class ProducerFence:
    """Hold native page-zero completion without blocking the model thread."""
    def __init__(self):
        self.entered, self.release = Event(), Event()

    def synchronize(self):
        self.entered.set()
        assert self.release.wait(5), "test did not release producer page-zero fence"


def assert_payload_bytes(worker, plan, expected):
    for obj in worker._objects(plan, loading=True):
        actual = b"".join(ctypes.string_at(segment.address, segment.length)
                          for segment in obj.segments)
        assert actual == expected[obj.key]


@pytest.mark.parametrize("mode", ["none", "align"])
def test_async_empty_dispatch_completion_then_admission_preserves_full_and_gdn(
        bridge_module, monkeypatch, mode):
    sender, worker, runner, tensors, group, store = connector_pair(bridge_module, monkeypatch, mode)
    scheduler, request, pages = scheduler_for(sender, mode, tokens=10)
    output = scheduled()
    output.kv_connector_metadata = sender.build_connector_meta(output)
    runner.requests["r"] = NS(block_ids=pages["r"], prev_num_draft_len=3)
    runner.num_accepted_tokens.gpu[0] = 2
    if mode == "align":
        native_postprocess(monkeypatch, runner, group, pages["r"][1], 2)
    worker.observe_and_save(runner, None, output)
    sender.scheduler_state.before_preempt(request)
    worker.handle_preemptions(empty_step(sender).kv_connector_metadata)
    deliver(sender, worker.build_connector_worker_meta())
    sender.scheduler_state.after_output(output)
    sender._init_load_state(True)
    worker._init_load_state(True)
    assert sender.get_num_new_matched_tokens(request, 0) == (8, True)
    destination = ((0, 12, 13), (0, 1, 2, 10, 11, 12, 13, 14))
    pages["r"] = destination
    sender.update_state_after_alloc(request, destination, 8)
    request.num_computed_tokens = 8
    statuses, freed, cached = native_lifecycle(monkeypatch, bridge_module, scheduler, request, pages)
    expected = dict(store.objects)
    for tensor in (*tensors[0], *tensors[1:]):
        tensor.zero_()
    runner.requests.clear()
    runner.input_batch.req_ids = []
    runner.input_batch.req_id_to_index = {}
    runner.input_batch.num_accepted_tokens_cpu[:] = 4
    fence = ProducerFence()
    worker._record_producer_event = lambda: fence
    worker._synchronize = lambda: pytest.fail("async submission synchronized the model thread")
    mixin, Aggregator, _ = native_completion(monkeypatch, worker)
    dispatch = empty_step(sender)
    plan = dispatch.kv_connector_metadata.restores[0]
    worker.after_native_update_states(runner, dispatch)  # No row yet; must not reset.
    premature = copy(dispatch)
    premature.num_scheduled_tokens = {"r": 1}
    with pytest.raises(RuntimeError, match="before its async load receipt"):
        worker.after_native_update_states(runner, premature)
    try:
        pending = mixin.kv_connector_no_forward(dispatch, None)
        assert fence.entered.wait(5)
        assert not pending.kv_connector_output.finished_recving
        assert not pending.kv_connector_output.kv_connector_worker_meta.loads
        assert runner.input_batch.num_accepted_tokens_cpu[0] == 4
        assert not worker._restored_cursors
        premature.kv_connector_metadata = replace(dispatch.kv_connector_metadata, restores=())
        with pytest.raises(RuntimeError, match="before its async load receipt"):
            worker.after_native_update_states(runner, premature)
        assert all(tensor.count_nonzero() == 0 for tensor in (*tensors[0], *tensors[1:]))
        fence.release.set()
        worker._load_queue.shutdown()  # Deterministic join; production polls instead.
        completion = mixin.kv_connector_no_forward(empty_step(sender), None)
        assert completion.kv_connector_output.finished_recving == {"r"}
        assert not completion.kv_connector_output.finished_sending
        assert_payload_bytes(worker, plan, expected)
        scheduler._update_from_kv_xfer_finished(Aggregator(1).aggregate([completion]).kv_connector_output)
        assert request.status == statuses.WAITING_FOR_REMOTE_KVS
        assert scheduler._try_promote_blocked_waiting_request(request)
        assert cached == [("r", 8)] and request.status == statuses.WAITING
        assert not freed
        # Native has now created the admitted row. The connector reset must
        # precede the native Mamba preprocessor, not the later forward hook.
        runner.input_batch.req_ids = ["r"]
        runner.input_batch.req_id_to_index = {"r": 0}
        runner.input_batch.prev_req_id_to_index = {"r": 0}
        runner.requests["r"] = NS(block_ids=destination, prev_num_draft_len=3)
        runner.mamba_state_idx["r"] = 7
        admitted = scheduled(1, ())
        admitted.kv_connector_metadata = sender.build_connector_meta(admitted)
        wrong_generation = copy(admitted)
        wrong_generation.kv_connector_metadata = replace(
            admitted.kv_connector_metadata, generations={"r": plan.generation + 1})
        isolated = copy(worker)
        isolated._restored_cursors = dict(worker._restored_cursors)
        with pytest.raises(RuntimeError, match="admission generation changed"):
            isolated.after_native_update_states(runner, wrong_generation)
        worker.after_native_update_states(runner, admitted)
        assert runner.input_batch.num_accepted_tokens_cpu[0] == runner.num_accepted_tokens.np[0] == 1
        assert runner.requests["r"].prev_num_draft_len == 0
        assert "r" not in runner.mamba_state_idx
        assert "r" not in runner.input_batch.prev_req_id_to_index
        assert not worker._restored_cursors
        assert_payload_bytes(worker, plan, expected)
        assert not mixin.kv_connector_no_forward(empty_step(sender), None).kv_connector_output.finished_recving
    finally:
        fence.release.set()
        if worker._load_queue is not None:
            worker._load_queue.shutdown()


@pytest.mark.parametrize("cancel", [None, "before_dispatch", "while_loading"])
def test_async_tp_receipts_span_empty_steps_and_cancel_frees_only_after_last_reader(
        bridge_module, monkeypatch, cancel):
    sender, first, runner, scheduler, request, pages, snapshot, receipts, store = saved_tp_swap(
        bridge_module, monkeypatch)
    deliver(sender, receipts[0].aggregate(receipts[1]))
    sender._init_load_state(True)
    first._init_load_state(True)
    first.rank = 0
    second = copy(first)
    second._init_load_state(True)
    second.rank = 1
    second._load_acks = []
    second._acks = []
    second._prefix_acks = []
    second.tracker = WorkerCheckpointTracker()
    assert sender.get_num_new_matched_tokens(request, 0) == (8, True)
    destination = ((0, 12, 13), (10, 11, 12, 13))
    pages["r"] = destination
    sender.update_state_after_alloc(request, destination, 8)
    request.num_computed_tokens = 8
    status, freed, cached = native_lifecycle(monkeypatch, bridge_module, scheduler, request, pages)
    original = dict(store.objects)
    workers = (first, second)
    fences = (ProducerFence(), ProducerFence())
    natives = []
    for worker, fence in zip(workers, fences):
        worker._record_producer_event = lambda fence=fence: fence
        mixin, Aggregator, _ = native_completion(monkeypatch, worker)
        natives.append(mixin)
    aggregator = Aggregator(2)
    if cancel == "before_dispatch":
        assert scheduler.finish_requests("r", status.FINISHED_ABORTED) == [("r", 0)]
        assert sender._restores and store.objects == original
        assert not freed and "r" in scheduler.requests
    dispatch = empty_step(sender, scheduler.finished_req_ids)
    try:
        results = [mixin.kv_connector_no_forward(dispatch, None) for mixin in natives]
        assert all(fence.entered.wait(5) for fence in fences)
        scheduler._update_from_kv_xfer_finished(aggregator.aggregate(results).kv_connector_output)
        assert not freed and store.objects == original
        if cancel == "while_loading":
            assert scheduler.finish_requests("r", status.FINISHED_ABORTED) == [("r", 0)]
            assert not freed and store.objects == original
        # Cancellation is propagated while readers are still blocked, then
        # each rank completes in a different native empty model step.
        waiting = empty_step(sender, scheduler.finished_req_ids)
        results = [mixin.kv_connector_no_forward(waiting, None) for mixin in natives]
        scheduler._update_from_kv_xfer_finished(aggregator.aggregate(results).kv_connector_output)
        fences[0].release.set()
        first._load_queue.shutdown()
        one_done = empty_step(sender)
        results = [mixin.kv_connector_no_forward(one_done, None) for mixin in natives]
        assert results[0].kv_connector_output.finished_recving == {"r"}
        assert not results[1].kv_connector_output.finished_recving
        combined = aggregator.aggregate(results).kv_connector_output
        assert not combined.finished_recving and not combined.finished_sending
        scheduler._update_from_kv_xfer_finished(combined)
        assert store.objects == original  # Rank one may still inspect all TP keys.
        assert snapshot.identity in sender._swap_payloads
        assert not freed and not scheduler.finished_recving_kv_req_ids
        fences[1].release.set()
        second._load_queue.shutdown()
        last_done = empty_step(sender)
        results = [mixin.kv_connector_no_forward(last_done, None) for mixin in natives]
        assert not results[0].kv_connector_output.finished_recving  # No repeated rank zero ACK.
        assert results[1].kv_connector_output.finished_recving == {"r"}
        combined = aggregator.aggregate(results).kv_connector_output
        assert combined.finished_recving == {"r"} and not combined.finished_sending
        scheduler._update_from_kv_xfer_finished(combined)
        assert not store.objects and not sender._swap_payloads
        if cancel:
            assert freed == ["r"] and "r" not in scheduler.requests and "r" not in pages
            assert all(not worker._restored_cursors for worker in workers)
        else:
            assert not freed and scheduler._try_promote_blocked_waiting_request(request)
            assert cached == [("r", 8)]
        results = [mixin.kv_connector_no_forward(empty_step(sender), None) for mixin in natives]
        combined = aggregator.aggregate(results).kv_connector_output
        assert not combined.finished_recving and not combined.finished_sending
        scheduler._update_from_kv_xfer_finished(combined)
        assert freed == (["r"] if cancel else [])
    finally:
        for fence in fences:
            fence.release.set()
        for worker in workers:
            if worker._load_queue is not None:
                worker._load_queue.shutdown()


def published_prefix(bridge_module, monkeypatch, local_tokens):
    """Publish a genuine joint prefix, retire it, and look it up as a new request."""
    sender, worker, runner, tensors, group, store = connector_pair(bridge_module, monkeypatch, "align")
    scheduler, request, pages = scheduler_for(sender, "align", tokens=12, prompt=12)
    scheduler._oscar_prefix_checkpoint_policy = NS(alignment_tokens=4, terminal_position=lambda count: 8)
    request.num_computed_tokens = 4
    runner.num_computed_tokens[0] = 4
    runner.mamba_state_idx["r"] = 3
    runner.requests["r"] = NS(block_ids=pages["r"], prev_num_draft_len=0)
    output = scheduled(4, ())
    output.kv_connector_metadata = sender.build_connector_meta(output)
    native_postprocess(monkeypatch, runner, group, pages["r"][1], 1, base=4, count=4, spec=0)
    worker.metadata = output.kv_connector_metadata
    worker.wait_for_save()
    worker.observe_and_save(runner, None, output)
    deliver(sender, worker.build_connector_worker_meta())
    sender.scheduler_state.after_output(output)
    expected = dict(store.objects)
    assert expected and not sender._prefix_publications
    sender.request_finished_all_groups(request, pages.pop("r"))
    replacement = NS(**vars(request))
    replacement.num_computed_tokens = local_tokens
    scheduler.requests["r"] = replacement
    sender._init_load_state(True)
    worker._init_load_state(True)
    assert sender.get_num_new_matched_tokens(replacement, local_tokens) == (8 - local_tokens, True)
    destination = ((12, 13, 14), (0, 1, 2, 10, 11, 12, 13))
    pages["r"] = destination
    sender.update_state_after_alloc(replacement, destination, 8 - local_tokens)
    replacement.num_computed_tokens = 8
    for tensor in (*tensors[0], *tensors[1:]):
        tensor.zero_()
    runner.requests.clear()
    runner.input_batch.req_ids = []
    runner.input_batch.req_id_to_index = {}
    dispatch = empty_step(sender)
    plan = dispatch.kv_connector_metadata.restores[0]
    assert plan.version == 0 and plan.local_tokens == local_tokens
    # Model a valid already-resident first FULL page when local prefix hit=4.
    all_objects = worker._objects(replace(plan, local_tokens=0), loading=True)
    if local_tokens:
        local_page = all_objects[0]
        offset = 0
        for segment in local_page.segments:
            ctypes.memmove(segment.address, expected[local_page.key][offset:offset + segment.length], segment.length)
            offset += segment.length
    return sender, worker, runner, scheduler, replacement, pages, store, expected, dispatch, all_objects


@pytest.mark.parametrize("local_tokens", [0, 4])
@pytest.mark.parametrize("reused_id_frame", [False, True])
def test_shared_prefix_async_load_skips_local_full_pages_but_restores_joint_gdn(
        bridge_module, monkeypatch, local_tokens, reused_id_frame):
    (sender, worker, runner, scheduler, request, pages, store,
     expected, dispatch, all_objects) = published_prefix(bridge_module, monkeypatch, local_tokens)
    status, freed, cached = native_lifecycle(monkeypatch, bridge_module, scheduler, request, pages)
    plan = dispatch.kv_connector_metadata.restores[0]
    if reused_id_frame:
        # published_prefix retired the old request and then allocated this
        # replacement generation. Native may carry both in the same frame.
        dispatch.finished_req_ids = {"r"}
        assert plan.generation > 1
        assert not dispatch.kv_connector_metadata.retired_loads
    read_keys = []
    original_get = store.batch_get_into_layers
    def read(keys, addresses, lengths, direction):
        read_keys.extend(keys)
        return original_get(keys, addresses, lengths, direction)
    monkeypatch.setattr(store, "batch_get_into_layers", read)
    fence = ProducerFence()
    worker._record_producer_event = lambda: fence
    mixin, Aggregator, _ = native_completion(monkeypatch, worker)
    try:
        worker.after_native_update_states(runner, dispatch)
        pending = mixin.kv_connector_no_forward(dispatch, None)
        assert fence.entered.wait(5)
        assert not pending.kv_connector_output.finished_recving and not read_keys
        assert not worker._restored_cursors
        assert not worker._cancelled_loads  # Old finished ID must not cancel this generation.
        fence.release.set()
        worker._load_queue.shutdown()
        completed = mixin.kv_connector_no_forward(empty_step(sender), None)
        assert completed.kv_connector_output.finished_recving == {"r"}
        assert not completed.kv_connector_output.kv_connector_worker_meta.loads  # Shared prefix is not private swap.
        assert worker._restored_cursors["r"].generation == plan.generation
        assert read_keys == [obj.key for obj in all_objects[1 if local_tokens else 0:]]
        assert len(read_keys) == (3 if local_tokens else 4)  # FULL page(s), conv and SSM.
        assert_payload_bytes(worker, replace(plan, local_tokens=0), expected)
        assert store.objects == expected  # Reading shared data never deletes it.
        scheduler._update_from_kv_xfer_finished(Aggregator(1).aggregate([completed]).kv_connector_output)
        assert not sender._scheduled_loads
        assert scheduler._try_promote_blocked_waiting_request(request)
        assert cached == [("r", 8)] and request.status == status.WAITING and not freed
        runner.input_batch.req_ids = ["r"]
        runner.input_batch.req_id_to_index = {"r": 0}
        runner.requests["r"] = NS(block_ids=pages["r"], prev_num_draft_len=3)
        admitted = scheduled(4, ())
        admitted.kv_connector_metadata = sender.build_connector_meta(admitted)
        worker.after_native_update_states(runner, admitted)
        assert runner.requests["r"].prev_num_draft_len == 0
        assert not worker._restored_cursors
        assert_payload_bytes(worker, replace(plan, local_tokens=0), expected)
    finally:
        fence.release.set()
        if worker._load_queue is not None:
            worker._load_queue.shutdown()


def test_async_sdk_failure_never_advertises_ready_or_promotes_prefix_request(bridge_module, monkeypatch):
    sender, worker, runner, scheduler, request, pages, store, expected, dispatch, _ = published_prefix(
        bridge_module, monkeypatch, 4)
    status, freed, cached = native_lifecycle(monkeypatch, bridge_module, scheduler, request, pages)
    monkeypatch.setattr(store, "batch_get_into_layers",
                        lambda keys, addresses, lengths, direction: [-3001] * len(keys))
    fence = ProducerFence()
    worker._record_producer_event = lambda: fence
    mixin, _, _ = native_completion(monkeypatch, worker)
    try:
        pending = mixin.kv_connector_no_forward(dispatch, None)
        assert fence.entered.wait(5)
        assert not pending.kv_connector_output.finished_recving
        fence.release.set()
        with pytest.raises(AsyncLoadError, match="failed object statuses"):
            worker._load_queue.shutdown()
        with pytest.raises(AsyncLoadError, match="failed object statuses"):
            mixin.kv_connector_no_forward(empty_step(sender), None)
        assert not worker._restored_cursors and not worker._load_acks
        assert not scheduler._try_promote_blocked_waiting_request(request)
        assert request.status == status.WAITING_FOR_REMOTE_KVS
        assert not freed and not cached and store.objects == expected
    finally:
        fence.release.set()
        if worker._load_queue is not None:
            with pytest.raises(AsyncLoadError):
                worker._load_queue.shutdown()
