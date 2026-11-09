"""Archive #37-49/#127/#128/#157/#163/#171: private payload ownership.

Real connector state and registered CPU byte transfers cover TP completion,
late cancellation/reset/branch receipts, and shared-prefix isolation. Host
coverage is not memcache/NPU or service acceptance.
"""
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from test_offload_native_bridge import (
    bridge_module, connector_pair, scheduler_for, scheduled,
)
from oscar_ascend.offload.transport import TransferError


def saved_tp_swap(module, monkeypatch):
    sender, worker, runner, tensors, group, store = connector_pair(module, monkeypatch, "none")
    sender.world_size = worker.world_size = 2
    scheduler, request, pages = scheduler_for(sender, "none", tokens=10)
    sender.scheduler_state.world_size = 2
    output = scheduled()
    output.kv_connector_metadata = sender.build_connector_meta(output)
    runner.requests["r"] = NS(block_ids=pages["r"], prev_num_draft_len=3)
    runner.num_accepted_tokens.gpu[0] = 2
    snapshot = sender.scheduler_state.before_preempt(request)
    metadata = sender.build_connector_meta(NS(num_scheduled_tokens={}, scheduled_spec_decode_tokens={}))
    receipts = []
    for rank in range(2):
        worker.rank = rank
        worker.observe_and_save(runner, None, output)
        worker.handle_preemptions(metadata)
        receipts.append(worker.build_connector_worker_meta())
    sender.scheduler_state.after_output(output)
    assert store.objects
    return sender, worker, runner, scheduler, request, pages, snapshot, receipts, store


def deliver(sender, metadata):
    sender.update_connector_output(NS(kv_connector_worker_meta=metadata))


@pytest.mark.parametrize("reason", ["cancel", "branch", "force_reset", "request_id_reuse"])
def test_retired_snapshot_collects_only_after_every_late_save_ack(bridge_module, monkeypatch, reason):
    sender, worker, runner, scheduler, request, pages, snapshot, receipts, store = saved_tp_swap(bridge_module, monkeypatch)
    original = dict(store.objects)
    store.objects["shared-prefix-object"] = b"retained"
    if reason == "cancel":
        sender.request_finished_all_groups(request, pages["r"])
    elif reason == "branch":
        sender.scheduler_state.branch(request)
    elif reason == "force_reset":
        sender.scheduler_state.reset()
    else:
        sender.scheduler_state.observe_request(NS(**vars(request)))
    deliver(sender, receipts[0])
    assert all(store.objects[key] == data for key, data in original.items())
    assert snapshot.identity in sender._swap_payloads
    deliver(sender, receipts[1])
    assert store.objects == {"shared-prefix-object": b"retained"}
    assert snapshot.identity not in sender._swap_payloads
    # Native metadata may aggregate an already-seen receipt again. It must not
    # resurrect a deleted generation or issue another deletion of newer data.
    deliver(sender, receipts[0].aggregate(receipts[1]))
    assert not sender._swap_payloads


@pytest.mark.parametrize("cancel_during_load", [False, True])
def test_last_tp_reader_releases_private_payload_without_joint_lookup_race(bridge_module, monkeypatch, cancel_during_load):
    sender, worker, runner, scheduler, request, pages, snapshot, receipts, store = saved_tp_swap(bridge_module, monkeypatch)
    deliver(sender, receipts[0].aggregate(receipts[1]))
    assert sender.get_num_new_matched_tokens(request, 0) == (8, False)
    destination = ((0, 12, 13), (10, 11, 12, 13))
    sender.update_state_after_alloc(request, destination, 8)
    metadata = sender.build_connector_meta(scheduled(1, ()))
    plan = metadata.restores[0]
    original = dict(store.objects)
    worker.metadata = metadata
    worker.rank = 0
    worker.start_load_kv(None)
    first = worker.build_connector_worker_meta()
    deliver(sender, first)
    assert store.objects == original  # Rank one still checks ALL rank keys.
    if cancel_during_load:
        sender.scheduler_state.finish(request.request_id)
        assert store.objects == original
    worker.rank = 1
    worker.start_load_kv(None)
    second = worker.build_connector_worker_meta()
    for obj in worker._objects(plan, loading=True):
        assert obj.key in original
    deliver(sender, second)
    assert not store.objects and not sender._swap_payloads
    deliver(sender, first.aggregate(second))
    assert not sender._swap_payloads


@pytest.mark.parametrize("local_tokens", [8, 9])
def test_native_local_prefix_supersedes_and_reclaims_private_snapshot(bridge_module, monkeypatch, local_tokens):
    sender, worker, runner, scheduler, request, pages, snapshot, receipts, store = saved_tp_swap(bridge_module, monkeypatch)
    deliver(sender, receipts[0].aggregate(receipts[1]))
    assert sender.get_num_new_matched_tokens(request, local_tokens) == (0, False)
    assert not store.objects and not sender._swap_payloads
    assert not sender._restores


def test_cancel_queued_restore_reclaims_without_waiting_for_undispatched_reader(bridge_module, monkeypatch):
    sender, worker, runner, scheduler, request, pages, snapshot, receipts, store = saved_tp_swap(bridge_module, monkeypatch)
    deliver(sender, receipts[0].aggregate(receipts[1]))
    sender.update_state_after_alloc(request, ((0, 12, 13), (10, 11, 12, 13)), 8)
    assert sender._restores
    sender.scheduler_state.finish(request.request_id)
    assert not sender._restores and not sender._swap_payloads and not store.objects
    metadata = sender.build_connector_meta(NS(num_scheduled_tokens={}, scheduled_spec_decode_tokens={}))
    assert not metadata.restores


def test_cleanup_sdk_error_is_not_swallowed_or_reported_as_collected(bridge_module, monkeypatch):
    sender, worker, runner, scheduler, request, pages, snapshot, receipts, store = saved_tp_swap(bridge_module, monkeypatch)
    deliver(sender, receipts[0].aggregate(receipts[1]))
    monkeypatch.setattr(sender.metadata_client, "remove_swap", lambda plan: (_ for _ in ()).throw(TransferError("remove failed")))
    with pytest.raises(TransferError, match="remove failed"):
        sender.scheduler_state.finish(request.request_id)
    assert store.objects and snapshot.identity in sender._swap_payloads


def test_invalid_or_conflicting_load_receipt_cannot_delete_payload(bridge_module, monkeypatch):
    sender, worker, runner, scheduler, request, pages, snapshot, receipts, store = saved_tp_swap(bridge_module, monkeypatch)
    deliver(sender, receipts[0].aggregate(receipts[1]))
    ack = receipts[0].acknowledgements[0]
    bad = bridge_module.LoadReceipt(**vars(ack))
    with pytest.raises(RuntimeError, match="load receipt"):
        deliver(sender, bridge_module.OscarOffloadWorkerMetadata(loads=(bad,)))
    assert store.objects
    sender.update_state_after_alloc(request, ((0, 12, 13), (10, 11, 12, 13)), 8)
    sender.build_connector_meta(scheduled(1, ()))
    with pytest.raises(RuntimeError, match="load receipt"):
        deliver(sender, bridge_module.OscarOffloadWorkerMetadata(loads=(replace(bad, payload_key="prefix/data/foreign"),)))
    assert store.objects


def test_partial_private_put_failure_rolls_back_only_attempted_rank_and_publishes_no_ack(bridge_module, monkeypatch):
    sender, worker, runner, tensors, group, store = connector_pair(bridge_module, monkeypatch, "none")
    scheduler, request, pages = scheduler_for(sender, "none", tokens=10)
    output = scheduled()
    output.kv_connector_metadata = sender.build_connector_meta(output)
    runner.requests["r"] = NS(block_ids=pages["r"], prev_num_draft_len=3)
    runner.num_accepted_tokens.gpu[0] = 2
    worker.observe_and_save(runner, None, output)
    sender.scheduler_state.before_preempt(request)
    metadata = sender.build_connector_meta(NS(num_scheduled_tokens={}, scheduled_spec_decode_tokens={}))
    original_put = store.batch_put_from_layers
    def partial_put(keys, addresses, lengths, direction):
        original_put(keys[:1], addresses[:1], lengths[:1], direction)
        return [0, *([-3001] * (len(keys)-1))]
    monkeypatch.setattr(store, "batch_put_from_layers", partial_put)
    store.objects["shared-prefix-object"] = b"retained"
    with pytest.raises(TransferError, match="failed object statuses"):
        worker.handle_preemptions(metadata)
    assert store.objects == {"shared-prefix-object": b"retained"}
    assert not worker.build_connector_worker_meta().acknowledgements
    with pytest.raises(TransferError, match="unusable after an earlier SDK failure"):
        worker.transport.exists(("shared-prefix-object",))


def test_worker_binds_rpc_scope_once_and_rejects_another_scheduler_session(bridge_module, monkeypatch):
    sender, worker, runner, tensors, group, store = connector_pair(bridge_module, monkeypatch, "none")
    scheduler_for(sender, "none", tokens=10)
    metadata = sender.build_connector_meta(NS(num_scheduled_tokens={}, scheduled_spec_decode_tokens={}))
    calls = []
    worker.config = NS(model_config=NS(max_model_len=65536))
    worker.metadata_server = NS(configure_swap_scope=lambda *args, **kwargs: calls.append((args, kwargs)))
    worker.handle_preemptions(metadata)
    worker.handle_preemptions(metadata)
    assert len(calls) == 1
    assert calls[0][1] == {"max_tokens": 65536}
    with pytest.raises(RuntimeError, match="cannot be rebound"):
        worker.handle_preemptions(replace(metadata, session="different-service"))
