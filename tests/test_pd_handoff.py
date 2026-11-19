"""#37-49/#127/#128/#157/#163/#171: real CPU bytes and pinned native PD seams.

No device or model-quality claim. Source and destination have separate tensors,
request lifetimes and physical page mappings; only the simulated store is shared.
"""
from copy import deepcopy
import ctypes
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from oscar_ascend.offload.config import transfer_config, validate_transfer_config
from oscar_ascend.offload.pd import PDState, prompt_digest, index_key, validate_ticket
from oscar_ascend.offload.transport import TransferError
from test_offload_native_bridge import bridge_module, connector_pair, scheduler_for, scheduled, native_postprocess
from test_offload_native_completion import native_completion
from test_offload_async_bridge import empty_step, native_lifecycle


def request_fields(req, count, params):
    req.prompt_token_ids = list(range(count))
    req._all_token_ids = list(range(count))
    req.all_token_ids = req._all_token_ids
    req.num_prompt_tokens = count
    req.num_computed_tokens = 0
    req.kv_transfer_params = params
    req.max_tokens = 1
    req.sampling_params = NS(prompt_logprobs=None)
    return req


def producer_case(module, monkeypatch, mode="align", count=8, world=2):
    p, pw, runner, tensors, group, store = connector_pair(module, monkeypatch, mode)
    scheduler, req, pages = scheduler_for(p, mode, tokens=count, prompt=count)
    p.world_size = pw.world_size = world
    scheduler.parallel_config.tensor_parallel_size = world
    p.pd, pw.pd = PDState(p, "prefill", 60), PDState(pw, "prefill", 60)
    request_fields(req, count, dict(do_remote_decode=True, do_remote_prefill=False))
    p.pd.prepare(req)
    expected_tokens = max(count - 1, 1)
    assert req.num_prompt_tokens == expected_tokens
    runner.num_computed_tokens[0] = 0
    runner.mamba_state_idx["r"] = 3
    runner.requests["r"] = NS(block_ids=pages["r"], prev_num_draft_len=0)
    step = scheduled(expected_tokens, ())
    step.kv_connector_metadata = p.build_connector_meta(step)
    if mode == "align":
        native_postprocess(monkeypatch, runner, group, pages["r"][1], 1,
                           base=0, count=expected_tokens, spec=0)
    pw.observe_and_save(runner, None, step)
    req.num_computed_tokens = expected_tokens
    req.status = NS(name="FINISHED_LENGTH_CAPPED")
    return p, pw, runner, tensors, store, scheduler, req, pages


@pytest.mark.parametrize("role,wire", [("prefill", "kv_producer"), ("decode", "kv_consumer")])
def test_roles_are_explicit_and_cannot_be_mixed(role, wire):
    config = {"kv_offload": {"enabled": True, "pd_role": role}}
    transfer = transfer_config(config)
    assert transfer["kv_role"] == wire
    validate_transfer_config(NS(**transfer))
    with pytest.raises(ValueError):
        validate_transfer_config(NS(**dict(transfer, kv_role="kv_both")))
    for change in ({"load_async": False}, {"enabled": False}, {"pd_role": "invalid"}):
        with pytest.raises(ValueError):
            transfer_config({"kv_offload": dict(config["kv_offload"], **change)})


@pytest.mark.parametrize("mode", ["none", "align"])
@pytest.mark.parametrize("count", [5, 8, 9])
@pytest.mark.parametrize("world", [1, 4])
def test_two_instances_unaligned_handoff_all_tp_and_async_restore(bridge_module, monkeypatch, mode, count, world):
    p, pw, pr, pt, store, ps, request, pp = producer_case(bridge_module, monkeypatch, mode, count, world)
    digest = p.pd.requests["r"][1]["prompt_digest"]
    delayed, params = p.request_finished_all_groups(request, pp["r"])
    assert delayed and params["oscar_pd"]["committed_tokens"] == count - 1
    ticket = params["oscar_pd"]
    dispatch = empty_step(p, finished=("r",))
    handoff = dispatch.kv_connector_metadata.handoffs[0]
    # Destination is another service with separate tensors and request state.
    d, dw, dr, dt, dg, dstore = connector_pair(bridge_module, monkeypatch, mode)
    ds, req, dp = scheduler_for(d, mode, tokens=count, prompt=count)
    d.world_size = dw.world_size = world
    d.pd, dw.pd = PDState(d, "decode", 60), PDState(dw, "decode", 60)
    d._init_load_state(True)
    dw._init_load_state(True)
    d.transport.store = dw.transport.store = store
    d.metadata_client = p.metadata_client
    request_fields(req, count, params)
    d.pd.prepare(req)
    assert prompt_digest(req) == digest
    assert d.get_num_new_matched_tokens(req, 0) == (None, False)
    # Every TP participant must save before the manifest becomes observable.
    for rank in range(world):
        pw.rank = rank
        pw.handle_preemptions(dispatch.kv_connector_metadata)
        pw.metadata = dispatch.kv_connector_metadata
        assert pw.get_finished(set())[0] == {"r"}
        p.update_connector_output(NS(kv_connector_worker_meta=pw.build_connector_worker_meta()))
        if rank < world - 1:
            assert p.metadata_client.get(index_key(ticket)) is None
    assert not p.pd.publishing
    expected = dict(store.objects)
    for tensor in (*pt[0], *pt[1:]):
        tensor.zero_()  # P pages may now be recycled independently of D.
    d.pd.waiting.clear()
    assert d.get_num_new_matched_tokens(req, 0) == (count - 1, True)
    destination = ((12, 13, 14), (9, 10, 11, 12, 13, 14, 15))
    dp["r"] = destination
    d.update_state_after_alloc(req, destination, count - 1)
    restore = empty_step(d)
    plan = restore.kv_connector_metadata.restores[0]
    assert plan.committed_tokens == count - 1 and plan.base_key.endswith(ticket["nonce"])
    for tensor in (*dt[0], *dt[1:]):
        tensor.zero_()
    # The real native mixin returns per-rank receive completion only after IO.
    mixin, aggregator, _ = native_completion(monkeypatch, dw)
    outputs = []
    for rank in range(world):
        dw.rank = rank
        dw._load_queue = None
        mixin.kv_connector_no_forward(restore, None)
        dw._load_queue.shutdown()
        outputs.append(mixin.kv_connector_no_forward(empty_step(d), None))
        for obj in dw._objects(plan, loading=True):
            actual = b"".join(ctypes.string_at(s.address, s.length) for s in obj.segments)
            assert actual == expected[obj.key]  # packed/windows/tags AND canonical GDN
        if rank < world - 1:
            dw._restored_cursors.clear()
    combined = aggregator(world).aggregate(outputs)
    assert combined.kv_connector_output.finished_recving == {"r"}
    assert not combined.kv_connector_output.finished_sending
    req.num_computed_tokens = count - 1
    dr.requests["r"] = NS(block_ids=destination, prev_num_draft_len=3)
    dr.input_batch.num_accepted_tokens_cpu[:] = 4
    resumed = scheduled(1, ())
    resumed.kv_connector_metadata = d.build_connector_meta(resumed)
    dw.after_native_update_states(dr, resumed)
    assert dr.input_batch.num_accepted_tokens_cpu[0] == 1
    assert dr.requests["r"].prev_num_draft_len == 0
    assert "r" not in dr.mamba_state_idx
    assert req.prompt_token_ids == list(range(count))  # D executes original last token.


def test_native_delayed_free_empty_step_keeps_source_until_save(bridge_module, monkeypatch):
    p, pw, runner, tensors, store, scheduler, req, pages = producer_case(bridge_module, monkeypatch, "none", world=1)
    status, freed, _ = native_lifecycle(monkeypatch, bridge_module, scheduler, req, pages)
    req.status = "FINISHED_LENGTH_CAPPED"
    req.is_finished = lambda: True
    params = scheduler._free_request(req)
    assert params["oscar_pd"] and not freed and "r" in pages
    output = empty_step(p, finished=("r",))
    pw.handle_preemptions(output.kv_connector_metadata)
    mixin, aggregator, _ = native_completion(monkeypatch, pw)
    completed = mixin.kv_connector_no_forward(output, None)
    combined = aggregator(1).aggregate([completed]).kv_connector_output
    scheduler._update_from_kv_xfer_finished(combined)
    assert freed == ["r"] and "r" not in pages
    assert p.metadata_client.get(index_key(params["oscar_pd"]))
    assert "r" not in pw.tracker._records


def test_single_token_has_an_explicit_empty_checkpoint(bridge_module, monkeypatch):
    p, pw, runner, tensors, store, scheduler, req, pages = producer_case(bridge_module, monkeypatch, "none", count=1)
    delayed, params = p.request_finished_all_groups(req, pages["r"])
    assert not delayed and params["oscar_pd"]["committed_tokens"] == 0
    assert not p.pd.pending
    d = PDState(p, "decode", 60)
    request_fields(req, 1, params)
    d.prepare(req)
    assert d.lookup(req, 0) == (0, False)


def test_ticket_mismatch_abort_and_bounded_wait(bridge_module, monkeypatch):
    p, pw, runner, tensors, store, scheduler, req, pages = producer_case(bridge_module, monkeypatch)
    ticket = deepcopy(p.pd.requests["r"][1])
    req.status = NS(name="FINISHED_ABORTED")
    assert p.request_finished_all_groups(req, pages["r"]) == (False, None)
    assert not p.pd.pending
    for changes in ({"namespace": "wrong"}, {"committed_tokens": 1}, {"prompt_tokens": True}, {"nonce": "../x"}):
        with pytest.raises(RuntimeError):
            validate_ticket(dict(ticket, **changes), p.namespace)
    d = PDState(p, "decode", 0.1)
    request_fields(req, 8, {"do_remote_prefill": True, "oscar_pd": ticket})
    req.prompt_token_ids[-1] += 1
    with pytest.raises(RuntimeError, match="differs"):
        d.prepare(req)
    req.prompt_token_ids[-1] -= 1
    d.prepare(req)
    assert d.lookup(req, 0) == (None, False)
    d.waiting["r"] = (0, 0)
    with pytest.raises(TransferError, match="timed out"):
        d.lookup(req, 0)


def test_prepare_is_before_native_prefix_lookup_and_idempotent(bridge_module, monkeypatch):
    import oscar_ascend.plugin as plugin
    from oscar_ascend.offload.hooks import request_added
    p, pw, runner, tensors, store, scheduler, req, pages = producer_case(bridge_module, monkeypatch)
    before = list(req.prompt_token_ids)
    request_added(scheduler, req)
    assert req.prompt_token_ids == before
    assert len(req.block_hashes) == req.num_prompt_tokens // p._hash_block_size()
    class Scheduler:
        def add_request(self, request):
            assert request.num_prompt_tokens == 3
            assert request.prompt_token_ids == [0, 1, 2]
            assert len(request.block_hashes) == 1
            return "native"
        def _preempt_request(self): pass
        def update_from_output(self): pass
        def reset_prefix_cache(self): pass
    fresh = NS(**vars(req))
    fresh.request_id = "fresh"
    request_fields(fresh, 4, {"do_remote_decode": True})
    p.scheduler_state.scheduler = scheduler
    instance = Scheduler()
    instance.connector = p
    try:
        plugin._patch_offload_scheduler(NS(Scheduler=Scheduler))
        assert instance.add_request(fresh) == "native"
    finally:
        plugin.unregister()


@pytest.mark.parametrize("damage", ["missing", "ticket", "keys"])
def test_published_handoff_damage_is_error_never_miss(bridge_module, monkeypatch, damage):
    p, pw, runner, tensors, store, scheduler, req, pages = producer_case(bridge_module, monkeypatch, world=1)
    _, params = p.request_finished_all_groups(req, pages["r"])
    dispatch = empty_step(p)
    pw.handle_preemptions(dispatch.kv_connector_metadata)
    p.update_connector_output(NS(kv_connector_worker_meta=pw.build_connector_worker_meta()))
    ticket = params["oscar_pd"]
    manifest = p.metadata_client.get(index_key(ticket))
    if damage == "missing":
        store.objects.pop(manifest["keys"][0])
    elif damage == "keys":
        manifest["keys"].pop()
    else:
        manifest["oscar_pd"] = dict(ticket, prompt_digest="0" * 64)
    request_fields(req, 8, params)
    p.pd = PDState(p, "decode", 60)
    p.pd.prepare(req)
    with pytest.raises(TransferError):
        p.get_num_new_matched_tokens(req, 0)


def test_worker_wrong_committed_boundary_never_publishes_or_releases(bridge_module, monkeypatch):
    p, pw, runner, tensors, store, scheduler, req, pages = producer_case(bridge_module, monkeypatch, world=1)
    p.request_finished_all_groups(req, pages["r"])
    dispatch = empty_step(p)
    record = pw.tracker._records["r"]
    pw.tracker._records["r"] = replace(record, scheduled=record.scheduled - 1)
    with pytest.raises(RuntimeError, match="committed worker"):
        pw.handle_preemptions(dispatch.kv_connector_metadata)
    assert not pw.pd.sent and not pw.pd.receipts
    assert not store.objects
