"""Archive #37-49/#127/#128/#157/#163/#171: native free and GDN commit witnesses.

Execute pinned native preemption and Mamba copy functions on synthetic CPU
storage. These tests establish lifecycle/byte correctness, not NPU acceptance.
"""
from dataclasses import dataclass, replace
from types import SimpleNamespace as NS

import pytest
import torch

from oscar_ascend.offload.scheduler import (
    OffloadLifecycleError, OffloadSchedulerState, SnapshotAck,
)
from oscar_ascend.offload.state import (
    committed_position, copy_canonical_mamba, restore_canonical_mamba,
    WorkerCheckpointTracker, _WorkerRecord, reset_restored_cursor,
)
from test_runtime_native import definitions


def setup_scheduler(world_size=2, mode="none"):
    request = NS(request_id="r", num_computed_tokens=104, num_prompt_tokens=100,
                 all_token_ids=list(range(101)), spec_token_ids=[7, 8, 9],
                 status="running", num_preemptions=0)
    pages = {"r": ((5, 9), (7, 8, 10, 11))}
    frees = []
    cache = NS(get_blocks=lambda rid: pages[rid],
               free=lambda req: frees.append(pages.pop(req.request_id)))
    waiting = []
    scheduler = NS(requests={"r": request}, kv_cache_manager=cache,
                   encoder_cache_manager=NS(free=lambda req: None),
                   _inflight_prefills=set(), log_stats=False,
                   waiting=NS(prepend_request=lambda req: waiting.insert(0, req)),
                   cache_config=NS(mamba_cache_mode=mode),
                   parallel_config=NS(tensor_parallel_size=world_size, pipeline_parallel_size=1))
    state = OffloadSchedulerState(scheduler)
    return scheduler, state, request, pages, frees


def ack(snapshot, count, rank=0):
    return SnapshotAck(snapshot.request_id, snapshot.generation, snapshot.version,
                       snapshot.epoch, count, rank, f"snapshot/{rank}")


def test_native_preemption_frees_same_step_and_async_ack_waits_for_output(monkeypatch):
    scheduler, state, request, pages, frees = setup_scheduler()
    # Native Request is identity-hashable; namespace is not.
    scheduler._inflight_prefills = NS(discard=lambda req: None)
    native = definitions("references/vllm/vllm/v1/core/sched/scheduler.py",
                         "offload_native_preempt", set(),
                         {"RequestStatus": NS(RUNNING="running", PREEMPTED="preempted")},
                         monkeypatch, methods=[("Scheduler", "_preempt_request")])
    output = NS(num_scheduled_tokens={"r": 4})
    state.begin_step(output)
    snapshot = state.before_preempt(request)
    native._preempt_request(scheduler, request, 0.0)
    assert request.num_computed_tokens == 0
    assert request.spec_token_ids == []
    assert "r" not in pages and frees == [snapshot.block_ids]
    # Native allocator can reuse all old pages immediately. Snapshot metadata
    # still points to the old contents, to be staged before worker zeroing.
    pages["other"] = snapshot.block_ids
    assert state.drain_preemptions() == (snapshot,)
    state.acknowledge(ack(snapshot, 102, 1))
    state.acknowledge(ack(snapshot, 102, 0))
    assert state.match_resume(request, 0) == (None, False)
    # Native async output accepts two tokens including the next uncomputed
    # sampled token. Rejected drafts never enter all_token_ids.
    request.all_token_ids.extend([101, 102])
    state.after_output(output)
    assert state.match_resume(request, 0) == (102, False)
    assert state.match_resume(request, 96) == (6, False)
    plan = state.allocated(request, ((15, 16), (20, 21, 22, 23)))
    assert plan.committed_tokens == 102
    assert plan.payload_keys == ("snapshot/0", "snapshot/1")
    assert plan.block_ids[0] == (15, 16)


def test_no_false_commit_from_hint_and_chunked_prefill_needs_no_extra_rewind():
    _, state, request, _, _ = setup_scheduler(world_size=1)
    request.num_computed_tokens = 64
    request.all_token_ids = list(range(100))
    snapshot = state.before_preempt(request)
    assert state.match_resume(request, 0) == (None, False)
    state.acknowledge(ack(snapshot, 64))
    assert state.match_resume(request, 0) == (64, False)
    # Full prompt completion with one sampled output also restores all prompt
    # KV; the sampled output is the replay token. Do not discard a valid KV.
    request.all_token_ids.append(100)
    newer = state.before_preempt(request)
    state.acknowledge(ack(newer, 100))
    assert state.match_resume(request, 0) == (100, False)


def test_tp_disagreement_and_conflicting_duplicate_fail_closed():
    _, state, request, _, _ = setup_scheduler()
    snapshot = state.before_preempt(request)
    state.acknowledge(ack(snapshot, 100))
    assert state.acknowledge(ack(snapshot, 100))
    with pytest.raises(OffloadLifecycleError, match="TP workers"):
        state.acknowledge(ack(snapshot, 99, 1))
    with pytest.raises(OffloadLifecycleError, match="conflicting duplicate"):
        state.acknowledge(ack(snapshot, 99, 0))


@pytest.mark.parametrize("action", ["cancel", "branch", "reset", "reuse_id"])
def test_generation_fence_rejects_late_snapshot_completion(action):
    scheduler, state, request, _, _ = setup_scheduler(world_size=1)
    snapshot = state.before_preempt(request)
    if action == "cancel":
        state.finish(request.request_id)
    elif action == "branch":
        request.num_computed_tokens = 100
        state.branch(request)
    elif action == "reset":
        state.reset()
    else:
        replacement = NS(**vars(request))
        scheduler.requests["r"] = replacement
        state.observe_request(replacement)
    assert not state.acknowledge(ack(snapshot, 100))
    assert state.match_resume(request, 0) is None


def test_no_saved_gdn_state_can_be_rewound_by_changing_a_token_count():
    _, state, request, _, _ = setup_scheduler(world_size=1)
    snapshot = state.before_preempt(request)
    state.acknowledge(ack(snapshot, len(request.all_token_ids)))
    with pytest.raises(OffloadLifecycleError, match="remaining token"):
        state.match_resume(request, 0)


@pytest.mark.parametrize("accepted,expected", [(1, 101), (2, 102), (3, 103), (4, 104)])
def test_native_mtp_rejection_cursor(accepted, expected):
    assert committed_position(100, 4, 3, accepted) == expected
    assert committed_position(96, 4, 0, 0) == 100  # real q4 prefill (#163)


def native_mamba_runner(monkeypatch):
    native = definitions("references/vllm/vllm/model_executor/layers/mamba/mamba_utils.py",
                         "offload_native_mamba_copy",
                         {"MambaCopySpec", "get_conv_copy_spec", "get_temporal_copy_spec"},
                         {"torch": torch, "dataclass": dataclass,
                          "is_conv_state_dim_first": lambda: False}, monkeypatch)
    # Physical stride is larger than state contents, as in OSCAR shared pages.
    conv_storage = torch.arange(24 * 64, dtype=torch.float32)
    temporal_storage = torch.arange(24 * 64, dtype=torch.float32) + 10000
    conv = conv_storage.as_strided((24, 6, 2), (64, 2, 1))
    temporal = temporal_storage.as_strided((24, 2, 2), (64, 2, 1))
    group = NS(layer_names=["gdn"], kv_cache_spec=NS(block_size=8))
    runner = NS(model=NS(get_mamba_state_copy_func=lambda:
                         (native.get_conv_copy_spec, native.get_temporal_copy_spec)),
                compilation_config=NS(static_forward_context={"gdn": NS(kv_cache=(conv, temporal))}))
    return runner, [(1, group)], conv, temporal


@pytest.mark.parametrize("mode,running,L", [("none", 0, 102), ("align", 1, 10)])
@pytest.mark.parametrize("accepted", [1, 2, 3, 4])
def test_canonical_gdn_roundtrip_uses_native_accepted_slot_without_mutating_live_state(
        monkeypatch, mode, running, L, accepted):
    runner, groups, conv, temporal = native_mamba_runner(monkeypatch)
    blocks = ((1, 2), (3, 5, 7, 9, 11))
    original_conv, original_temporal = conv.clone(), temporal.clone()
    saved = copy_canonical_mamba(runner, blocks, running, accepted, groups=groups)
    assert torch.equal(conv, original_conv) and torch.equal(temporal, original_temporal)
    assert torch.equal(saved[0].values, conv[blocks[1][running], accepted-1:].flatten())
    assert torch.equal(saved[1].values, temporal[blocks[1][running+accepted-1]].flatten())
    # Free/reallocate old page IDs and destroy all source bytes. Scratch must
    # survive; the destination can use entirely different physical pages.
    conv.fill_(-888)
    temporal.fill_(-999)
    destination = ((14, 15), (16, 17, 18, 19, 20))
    restore_canonical_mamba(runner, saved, destination, L, mode, groups=groups)
    logical = 0 if mode == "none" else (L-1)//8
    target = destination[1][logical]
    assert torch.equal(conv[target].flatten()[:saved[0].values.numel()], saved[0].values)
    assert torch.equal(temporal[target].flatten(), saved[1].values)


def test_worker_resolve_uses_corrected_gpu_count_not_scheduler_hint(monkeypatch):
    runner, groups, _, _ = native_mamba_runner(monkeypatch)
    scheduler, state, request, _, _ = setup_scheduler(world_size=1)
    snapshot = state.before_preempt(request)
    tracker = WorkerCheckpointTracker()
    synced = []
    tracker._records["r"] = _WorkerRecord(
        snapshot.generation, 7, ((5, 9), (7, 8, 10, 11)),
        torch.tensor(97), torch.tensor(2), torch.tensor(2), 4, 3, 0, "none",
        NS(synchronize=lambda: synced.append(True)))
    monkeypatch.setattr("oscar_ascend.offload.state._mamba_groups", lambda runner: groups)
    resolved = tracker.resolve(runner, snapshot)
    assert synced == [True]
    assert resolved.committed_tokens == 99  # scheduler hint was optimistic104
    assert resolved.step_id == 7


def test_restored_cursor_cannot_inherit_previous_mtp_accepted_column():
    runner = NS(input_batch=NS(req_id_to_index={"r": 1},
                              num_accepted_tokens_cpu=[4, 4],
                              prev_req_id_to_index={"other": 0, "r": 1}),
                num_accepted_tokens=NS(np=[4, 4]),
                requests={"r": NS(prev_num_draft_len=3)}, mamba_state_idx={"r": 6})
    reset_restored_cursor(runner, "r")
    assert runner.input_batch.num_accepted_tokens_cpu == [4, 1]
    assert runner.num_accepted_tokens.np == [4, 1]
    assert runner.requests["r"].prev_num_draft_len == 0
    assert runner.input_batch.prev_req_id_to_index == {"other": 0}
    assert "r" not in runner.mamba_state_idx


def test_borrowed_canonical_ranges_stay_inside_registered_raw_storage(monkeypatch):
    runner, groups, conv, temporal = native_mamba_runner(monkeypatch)
    blocks = ((1, 2), (3, 5, 7, 9))
    values = copy_canonical_mamba(runner, blocks, 0, 3, groups=groups, clone=False)
    assert values[0].values.data_ptr() == conv[3, 2:].data_ptr()
    assert values[1].values.data_ptr() == temporal[7].data_ptr()
    assert values[0].values.untyped_storage().data_ptr() == conv.untyped_storage().data_ptr()
    assert values[1].values.untyped_storage().data_ptr() == temporal.untyped_storage().data_ptr()


def test_worker_metadata_survives_row_reorder_and_reused_native_buffers(monkeypatch):
    class Event:
        def record(self):
            pass

        def synchronize(self):
            pass

    monkeypatch.setattr(torch.cpu, "Event", Event, raising=False)
    runner, groups, _, _ = native_mamba_runner(monkeypatch)
    runner.device = torch.device("cpu")
    runner.cache_config = NS(mamba_cache_mode="none")
    runner.input_batch = NS(req_ids=["r", "other"])
    runner.num_computed_tokens = torch.tensor([97, 300])
    runner.num_accepted_tokens = NS(gpu=torch.tensor([2, 4]))
    runner.requests = {
        "r": NS(block_ids=((5, 9), (7, 8, 10, 11))),
        "other": NS(block_ids=((2, 3), (4, 5, 6, 7))),
    }
    output = NS(num_scheduled_tokens={"r": 4, "other": 4},
                scheduled_spec_decode_tokens={"r": [1, 2, 3], "other": [1, 2, 3]})
    _, state, request, _, _ = setup_scheduler(world_size=1)
    snapshot = state.before_preempt(request)
    tracker = WorkerCheckpointTracker()
    tracker.record_step(runner, output, generations={"r": snapshot.generation, "other": 2}, step_id=12)
    # r may leave the persistent batch for a turn before it is preempted;
    # its saved metadata cannot alias overwritten native batch buffers.
    runner.input_batch.req_ids = ["other"]
    runner.num_computed_tokens[:] = torch.tensor([400, 500])
    runner.num_accepted_tokens.gpu[:] = 1
    monkeypatch.setattr("oscar_ascend.offload.state._mamba_groups", lambda runner: groups)
    result = tracker.resolve(runner, snapshot)
    assert result.committed_tokens == 99
    assert result.step_id == 12


def test_multiple_inflight_native_outputs_all_finish_before_resume():
    _, state, request, _, _ = setup_scheduler(world_size=1)
    first = NS(num_scheduled_tokens={"r": 4})
    second = NS(num_scheduled_tokens={"r": 4})
    assert state.begin_step(first) != state.begin_step(second)
    snapshot = state.before_preempt(request)
    state.acknowledge(ack(snapshot, 104))
    request.all_token_ids.extend([101, 102, 103, 104])
    state.after_output(second)
    assert state.match_resume(request, 0) == (None, False)
    state.after_output(first)
    assert state.match_resume(request, 0) == (104, False)


@pytest.mark.parametrize("local_tokens", [64, 80])
def test_valid_native_joint_prefix_supersedes_remote_without_shared_page_write(local_tokens):
    _, state, request, _, _ = setup_scheduler(world_size=1, mode="align")
    request.num_computed_tokens = 64
    request.all_token_ids = list(range(100))
    snapshot = state.before_preempt(request)
    state.acknowledge(ack(snapshot, 64))
    assert state.match_resume(request, local_tokens) == (0, False)
    assert state.allocated(request, ((15, 16), (20, 21, 22, 23))) is None
    assert not state.acknowledge(ack(snapshot, 64))


def test_native_streaming_branch_can_preempt_before_its_first_new_forward(monkeypatch):
    scheduler, state, request, _, _ = setup_scheduler(world_size=1)
    request.num_computed_tokens = 100
    request.num_prompt_tokens = 95
    request.prompt_token_ids = list(range(95))
    request._all_token_ids = request.all_token_ids
    request._output_token_ids = list(range(95, 101))
    request.update_block_hashes = lambda: None
    request.status = "stream_wait"
    scheduler.num_waiting_for_streaming_input = 1
    old_generation = state.observe_request(request)
    runner, groups, _, _ = native_mamba_runner(monkeypatch)
    runner.requests = {"r": NS(prompt_token_ids=list(range(95)),
                               output_token_ids=[95, 96, -1, -1, -1, -1])}
    tracker = WorkerCheckpointTracker()
    tracker._records["r"] = _WorkerRecord(old_generation, 4, ((5, 9), (7, 8, 10, 11)),
        torch.tensor(96), torch.tensor(4), torch.tensor(4), 4, 3, 0, "none", NS(synchronize=lambda: None))
    generation = state.branch(request)
    branch = state.drain_branches()
    native = definitions("references/vllm/vllm/v1/core/sched/scheduler.py", "offload_native_stream_branch", set(),
        {"RequestStatus": NS(WAITING_FOR_STREAMING_REQ="stream_wait", WAITING="waiting")}, monkeypatch,
        methods=[("Scheduler", "_update_request_as_session")])
    native._update_request_as_session(scheduler, request,
        NS(mm_features=[], prompt_token_ids=[201, 202], arrival_time=1.0, sampling_params=None))
    assert request.all_token_ids == [*range(100), 201, 202]
    assert request.num_prompt_tokens == 102
    assert scheduler.num_waiting_for_streaming_input == 0
    snapshot = state.before_preempt(request)
    tracker.apply_branches(runner, branch)
    tracker.apply_branches(runner, branch)  # retry/duplicate cannot shift state
    monkeypatch.setattr("oscar_ascend.offload.state._mamba_groups", lambda runner: groups)
    result = tracker.resolve(runner, snapshot)
    assert result.committed_tokens == 100 and snapshot.generation == generation
    assert tracker._records["r"].generation != old_generation


def test_streaming_branch_rejects_stale_cursor_and_corrupted_prefix():
    _, state, request, _, _ = setup_scheduler(world_size=1)
    request.num_computed_tokens = 100
    old = state.observe_request(request)
    state.branch(request)
    branch = state.drain_branches()[0]
    tracker = WorkerCheckpointTracker()
    tracker._records["r"] = _WorkerRecord(old, 2, ((5, 9), (7, 8, 10, 11)),
        torch.tensor(96), torch.tensor(3), torch.tensor(3), 4, 3, 0, "none", NS(synchronize=lambda: None))
    runner = NS(requests={})
    with pytest.raises(OffloadLifecycleError, match="different native GDN cursor"):
        tracker.apply_branches(runner, (branch,))
    with pytest.raises(OffloadLifecycleError, match="invalid streaming branch"):
        tracker.apply_branches(runner, (replace(branch, retained_prefix_digest="wrong"),))


@pytest.mark.parametrize("successful", [False, True])
def test_normal_prefix_cache_reset_preserves_live_generations(successful):
    from oscar_ascend.offload.hooks import resetting_cache
    scheduler, state, request, _, _ = setup_scheduler(world_size=1)
    scheduler.connector = NS(_oscar_offload=True, scheduler_state=state)
    snapshot = state.before_preempt(request)
    epoch = state.epoch
    with resetting_cache(scheduler) as complete:
        assert not state.suppress_preemption
        complete(successful)
    assert state.epoch == epoch
    assert state.acknowledge(ack(snapshot, 100))
    assert state.match_resume(request, 0) == (100, False)
    assert not state.suppress_preemption
