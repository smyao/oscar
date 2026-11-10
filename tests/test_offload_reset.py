"""Archive #37-49/#127/#128/#171: prefix invalidation is not active-KV loss."""
from types import SimpleNamespace as NS

import pytest

from oscar_ascend.offload.hooks import resetting_cache
from test_offload_lifecycle import setup_scheduler, ack


@pytest.mark.parametrize("successful", [False, True])
@pytest.mark.parametrize("previous_suppression", [False, True])
def test_forced_reset_commits_snapshot_invalidation_only_after_native_success(successful, previous_suppression):
    scheduler, state, request, _, _ = setup_scheduler(world_size=1)
    scheduler.connector = NS(_oscar_offload=True, scheduler_state=state)
    snapshot = state.before_preempt(request)
    epoch = state.epoch
    state.suppress_preemption = previous_suppression
    with resetting_cache(scheduler, reset_running_requests=True) as complete:
        assert state.suppress_preemption
        assert state.before_preempt(request) is None
        assert state.epoch == epoch  # No invalidation ahead of native success.
        complete(successful)
    assert state.suppress_preemption is previous_suppression
    assert state.epoch == epoch + int(successful)
    if successful:
        assert not state.acknowledge(ack(snapshot, 100))
        assert state.match_resume(request, 0) is None
    else:
        assert state.acknowledge(ack(snapshot, 100))
        assert state.match_resume(request, 0) == (100, False)


@pytest.mark.parametrize("force", [False, True])
def test_reset_exception_preserves_recoverable_snapshot_and_propagates(force):
    scheduler, state, request, _, _ = setup_scheduler(world_size=1)
    scheduler.connector = NS(_oscar_offload=True, scheduler_state=state)
    snapshot = state.before_preempt(request)
    epoch = state.epoch
    with pytest.raises(RuntimeError, match="native reset failed"):
        with resetting_cache(scheduler, reset_running_requests=force):
            raise RuntimeError("native reset failed")
    assert state.epoch == epoch
    assert not state.suppress_preemption
    assert state.acknowledge(ack(snapshot, 100))
    assert state.match_resume(request, 0) == (100, False)


def test_reset_without_oscar_connector_does_not_add_side_effects():
    scheduler = NS(connector=NS(_oscar_offload=False))
    with resetting_cache(scheduler, reset_running_requests=True) as complete:
        complete(True)
    assert vars(scheduler) == {"connector": scheduler.connector}


@pytest.mark.parametrize("call_args,call_kwargs,force", [
    ((), {}, False), ((True,), {}, True), ((), {"reset_running_requests": True}, True),
    ((False, True), {}, False),
])
@pytest.mark.parametrize("successful", [False, True])
def test_plugin_wrapper_preserves_native_reset_signature_and_commits_result(monkeypatch, call_args, call_kwargs, force, successful):
    from oscar_ascend import plugin

    original_scheduler, state, request, _, _ = setup_scheduler(world_size=1)
    snapshot = state.before_preempt(request)
    native_calls = []

    class Scheduler:
        def _preempt_request(self, request, *args, **kwargs):
            pass

        def update_from_output(self, output):
            pass

        def reset_prefix_cache(self, reset_running_requests=False, reset_connector=False):
            native_calls.append((reset_running_requests, reset_connector))
            assert state.suppress_preemption is force
            return successful

    monkeypatch.setattr(plugin, "_patch", lambda owner, name, builder:
                        setattr(owner, name, builder(getattr(owner, name))))
    plugin._patch_offload_scheduler(NS(Scheduler=Scheduler))
    scheduler = Scheduler()
    scheduler.connector = NS(_oscar_offload=True, scheduler_state=state)
    assert scheduler.reset_prefix_cache(*call_args, **call_kwargs) is successful
    assert native_calls == [(force, bool(len(call_args) == 2 and call_args[1]))]
    assert not state.suppress_preemption
    assert state.acknowledge(ack(snapshot, 100)) is not (force and successful)


from test_offload_native_bridge import (
    bridge_module, connector_pair, scheduler_for, scheduled, native_postprocess,
)


def prefix_checkpoint(bridge_module, monkeypatch):
    sender, worker, runner, tensors, group, store = connector_pair(bridge_module, monkeypatch, "align")
    scheduler, request, pages = scheduler_for(sender, "align", tokens=12, prompt=12)
    scheduler._oscar_prefix_checkpoint_policy = NS(alignment_tokens=4, terminal_position=lambda n: 8)
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
    receipts = worker.build_connector_worker_meta()
    return sender, worker, scheduler, request, runner, output, receipts, store


def test_external_prefix_reset_hides_old_view_without_deleting_payload_or_active_snapshot(bridge_module, monkeypatch):
    from oscar_ascend.offload.scheduler import SnapshotAck
    sender, worker, scheduler, request, runner, output, receipts, store = prefix_checkpoint(bridge_module, monkeypatch)
    sender.update_connector_output(NS(kv_connector_worker_meta=receipts))
    lookup = NS(**vars(request))
    lookup.request_id = "next-request"
    scheduler.requests[lookup.request_id] = lookup
    assert sender.get_num_new_matched_tokens(lookup, 0) == (8, False)
    snapshot = sender.scheduler_state.before_preempt(request)
    assert sender.scheduler_state.acknowledge(SnapshotAck(*snapshot.identity, 8, 0, "private-swap"))
    sender.scheduler_state.after_output(output)
    assert sender.scheduler_state.match_resume(request, 0) == (8, False)
    selected_restore = object()
    sender._restores.append(selected_restore)
    old_epoch = sender.scheduler_state.epoch
    old_view = sender._prefix_view
    saved = dict(store.objects)
    assert sender.reset_cache() is True
    assert sender._prefix_view != old_view
    assert sender.scheduler_state.epoch == old_epoch
    assert sender.scheduler_state.match_resume(request, 0) == (8, False)
    assert sender._restores == [selected_restore]
    assert store.objects == saved  # Another service still owns its old shared cache view.
    assert sender.get_num_new_matched_tokens(lookup, 0) == (0, False)


def test_late_old_prefix_receipts_cannot_repopulate_reset_view(bridge_module, monkeypatch):
    sender, worker, scheduler, request, runner, old_output, old_receipts, store = prefix_checkpoint(bridge_module, monkeypatch)
    old_view = old_output.kv_connector_metadata.prefix_view
    assert sender.reset_cache() is True
    sender.update_connector_output(NS(kv_connector_worker_meta=old_receipts))
    assert not sender._prefix_publications
    assert sender.get_num_new_matched_tokens(request, 0) == (0, False)
    output = scheduled(4, ())
    output.kv_connector_metadata = sender.build_connector_meta(output)
    assert output.kv_connector_metadata.prefix_view != old_view
    worker.metadata = output.kv_connector_metadata
    worker.wait_for_save()
    worker.observe_and_save(runner, None, output)
    receipt = worker.build_connector_worker_meta()
    assert receipt.prefixes[0].prefix_view == output.kv_connector_metadata.prefix_view
    sender.update_connector_output(NS(kv_connector_worker_meta=receipt))
    assert sender.get_num_new_matched_tokens(request, 0) == (8, False)


def test_prefix_manifest_cannot_cross_local_reset_view(bridge_module, monkeypatch):
    sender, worker, scheduler, request, runner, output, receipts, store = prefix_checkpoint(bridge_module, monkeypatch)
    sender.update_connector_output(NS(kv_connector_worker_meta=receipts))
    old_manifest = sender.metadata_client.get(receipts.prefixes[0].index_key)
    assert sender.reset_cache() is True
    # Simulate a corrupt/misdirected metadata lookup: payload bytes still exist,
    # but the body must bind the requested new index AND its prefix view.
    sender.metadata_client.get = lambda key: dict(old_manifest, index_key=key)
    sender.metadata_client.exists = lambda keys: (True,) * len(keys)
    from oscar_ascend.offload.transport import TransferError
    with pytest.raises(TransferError, match="incompatible identity"):
        sender.get_num_new_matched_tokens(request, 0)
