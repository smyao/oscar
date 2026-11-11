"""Same-step prefix saves retain the original complete first writer.

Archive #127/#128: physical FULL pages are not metadata128 slices.
Archive #157/#163/#171: native committed GDN state and all-TP completion
cannot be inferred from token counts alone. #182/#183: missing/unpublished
indexes remain misses; errors remain visible. These tests execute connector
metadata, worker byte saves, strict manifest publication and remapped restore
with CPU pointer buffers. They are not NPU, SDK or performance acceptance.
"""
from dataclasses import replace
import ctypes
import hashlib
import pickle
from types import SimpleNamespace as NS

import pytest
import torch

from oscar_ascend.offload.metadata import MetadataClient, MetadataServer
from oscar_ascend.offload.plans import expected_keys, prefix_plan
from oscar_ascend.offload.transport import TransferError
from test_offload_native_bridge import bridge_module, connector_pair, scheduler_for
from test_runtime_native import definitions


class LocalMetadataClient(MetadataClient):
    """Use the real RPC dispatcher/protocol without opening an IPC socket."""

    def __init__(self, server):
        super().__init__(server.endpoint)
        self.server = server

    def _rpc(self, request):
        response = self.server.dispatch(request)
        if response["status"] == "error":
            raise TransferError(response["error"])
        return response


def native_hashes(monkeypatch, request):
    native = definitions(
        "references/vllm/vllm/v1/core/kv_cache_utils.py", "single_writer_native_hash",
        {"_gen_mm_extra_hash_keys", "_gen_lora_extra_hash_keys",
         "_gen_prompt_embeds_extra_hash_keys", "generate_block_hash_extra_keys",
         "hash_block_tokens"},
        {"BlockHash": bytes, "NONE_HASH": bytes(32), "hashlib": hashlib}, monkeypatch)
    request.cache_salt = getattr(request, "cache_salt", None)
    request.lora_request = getattr(request, "lora_request", None)
    request.mm_features = getattr(request, "mm_features", [])
    request.prompt_embeds = None
    hashes, parent, mm_index = [], None, 0
    for start in range(0, request.num_prompt_tokens, 2):
        extra, mm_index = native.generate_block_hash_extra_keys(request, start, start + 2, mm_index)
        parent = native.hash_block_tokens(
            lambda value: hashlib.sha256(pickle.dumps(value)).digest(), parent,
            request.all_token_ids[start:start + 2], extra)
        hashes.append(parent)
    return hashes


def make_cluster(module, monkeypatch):
    sender, first, runner, tensors, group, store = connector_pair(module, monkeypatch, "align")
    scheduler, request, pages = scheduler_for(sender, "align", tokens=12, prompt=12)
    scheduler._oscar_prefix_checkpoint_policy = NS(
        alignment_tokens=4, terminal_position=lambda n: (n - 1) // 4 * 4)
    request.num_computed_tokens = 4
    peer = NS(**vars(request))
    peer.request_id = "peer"
    peer.num_computed_tokens = 0  # Full prefill vs the first writer's continuation.
    scheduler.requests["peer"] = peer
    pages["peer"] = ((0, 1, 2), (8, 9, 10, 11, 12, 13, 14, 15))
    for req in (request, peer):
        req.block_hashes = native_hashes(monkeypatch, req)
    other_sender, second, second_runner, second_tensors, _, _ = connector_pair(module, monkeypatch, "align")
    # The fake SDK stores actual pointer bytes from independent worker buffers.
    second.transport.store = store
    workers = (first, second)
    for rank, (worker, active_runner) in enumerate(zip(workers, (runner, second_runner))):
        worker.rank, worker.world_size = rank, 2
        active_runner.input_batch.req_ids = ["r", "peer"]
        active_runner.input_batch.req_id_to_index = {"r": 0, "peer": 1}
        active_runner.num_computed_tokens = torch.tensor([4, 0])
        active_runner.num_accepted_tokens.gpu = torch.ones(2, dtype=torch.int32)
        active_runner.mamba_state_idx = {"r": 3, "peer": 3}
        active_runner.requests = {rid: NS(block_ids=ids, prev_num_draft_len=0) for rid, ids in pages.items()}
        active_runner._get_mamba_bufs = lambda: NS(postprocess_align=NS(
            num_accepted_tokens_out=torch.ones(2, dtype=torch.int32)))
    sender.world_size = sender.scheduler_state.world_size = 2
    store.host_writes = []
    store.data_writes = []
    original_put = store.batch_put_from_layers

    def data_put(keys, addresses, sizes, direction):
        store.data_writes.append((tuple(keys), sum(sum(lengths) for lengths in sizes)))
        return original_put(keys, addresses, sizes, direction)

    def host_put(key, payload):
        if key in store.objects:
            return -3006
        store.objects[key] = payload
        store.host_writes.append(key)
        return 0

    def host_remove(key):
        return 0 if store.objects.pop(key, None) is not None else -3007

    store.batch_put_from_layers = data_put
    store.put, store.get, store.remove = host_put, lambda key: store.objects.get(key, b""), host_remove
    server = MetadataServer(first.transport, "ipc:///unused-single-writer-test")
    sender.metadata_client = LocalMetadataClient(server)
    return NS(sender=sender, scheduler=scheduler, request=request, peer=peer,
              pages=pages, workers=workers, store=store, server=server,
              extra_sender=other_sender, tensors=(tensors, second_tensors))


def metadata(cluster, scheduled=None):
    output = NS(num_scheduled_tokens=scheduled if scheduled is not None else {"r": 4, "peer": 8},
                scheduled_spec_decode_tokens={}, finished_req_ids=set(), preempted_req_ids=set())
    output.kv_connector_metadata = cluster.sender.build_connector_meta(output)
    return output


def save_rank(cluster, output, rank):
    worker = cluster.workers[rank]
    worker.metadata = output.kv_connector_metadata
    worker.wait_for_save()
    worker.observe_and_save(worker.runner, None, output)
    return worker.build_connector_worker_meta()


def deliver(cluster, receipt):
    cluster.sender.update_connector_output(NS(kv_connector_worker_meta=receipt))


def close_cluster(cluster):
    cluster.sender.metadata_client.close()
    for connector in (cluster.sender, cluster.extra_sender, *cluster.workers):
        connector.transport.close()


def test_first_writer_retains_complete_tp_bytes_and_halves_duplicate_save_work(bridge_module, monkeypatch):
    outcomes = []
    for legacy in (True, False):
        cluster = make_cluster(bridge_module, monkeypatch)
        try:
            output = metadata(cluster)
            plans = output.kv_connector_metadata.prefixes
            assert len(plans) == 1 and plans[0].request_id == "r"
            assert plans[0].block_ids == cluster.pages["r"]
            if legacy:
                # Reconstruct the previous valid two-writer metadata, keeping
                # its original serial order and immutable first-writer winner.
                original = tuple(prefix_plan(req, output.kv_connector_metadata.generations[req.request_id],
                    cluster.sender.scheduler_state.epoch, 8, cluster.pages[req.request_id],
                    cluster.sender.geometry, 2, include_page_hashes=False)
                    for req in (cluster.request, cluster.peer))
                output.kv_connector_metadata = replace(output.kv_connector_metadata, prefixes=original)
            first = save_rank(cluster, output, 0)
            assert cluster.sender.get_num_new_matched_tokens(cluster.peer, 0) == (0, False)
            deliver(cluster, first)
            assert not cluster.store.host_writes
            assert cluster.sender.get_num_new_matched_tokens(cluster.peer, 0) == (0, False)
            second = save_rank(cluster, output, 1)
            deliver(cluster, second)
            assert len(cluster.store.host_writes) == 1
            index = first.prefixes[0].index_key
            manifest = cluster.sender.metadata_client.get(index)
            assert manifest["base_key"] == first.prefixes[0].plan.base_key
            selected = {key: cluster.store.objects[key] for key in manifest["keys"]}
            assert tuple(selected) == expected_keys(cluster.sender.namespace, 2, first.prefixes[0].plan,
                                                   cluster.sender.geometry)
            assert cluster.sender.get_num_new_matched_tokens(cluster.peer, 0) == (8, False)
            destination = ((12, 13, 14), (0, 1, 2, 10, 11, 12, 13))
            cluster.sender.update_state_after_alloc(cluster.peer, destination, 8)
            restore = metadata(cluster, {})
            for worker in cluster.workers:
                worker.metadata = restore.kv_connector_metadata
                worker.start_load_kv(None)
                for item in worker._objects(restore.kv_connector_metadata.restores[0], loading=True):
                    actual = b"".join(ctypes.string_at(part.address, part.length) for part in item.segments)
                    assert actual == selected[item.key]
            outcomes.append((len(cluster.store.data_writes), sum(size for _, size in cluster.store.data_writes), selected))
        finally:
            close_cluster(cluster)
    assert outcomes[0][0] == 4 and outcomes[1][0] == 2
    assert outcomes[0][1] == 1088 and outcomes[1][1] == 544
    assert outcomes[0][2] == outcomes[1][2]  # Same first writer, every component, both ranks.


@pytest.mark.parametrize("difference", ["tokens", "salt", "lora", "multimodal"])
def test_native_hash_identity_keeps_distinct_prefixes_separate(bridge_module, monkeypatch, difference):
    cluster = make_cluster(bridge_module, monkeypatch)
    try:
        peer = cluster.peer
        if difference == "tokens":
            peer.all_token_ids = [900, *peer.all_token_ids[1:]]
        elif difference == "salt":
            peer.cache_salt = "isolated-cache-salt"
        elif difference == "lora":
            peer.lora_request = NS(lora_name="different-adapter")
        else:
            peer.mm_features = [NS(identifier="different-image", mm_position=NS(offset=0, length=2))]
        peer.block_hashes = native_hashes(monkeypatch, peer)
        output = metadata(cluster)
        plans = output.kv_connector_metadata.prefixes
        assert [plan.request_id for plan in plans] == ["r", "peer"]
        assert plans[0].base_key != plans[1].base_key
        for rank in range(2):
            deliver(cluster, save_rank(cluster, output, rank))
        assert len(cluster.store.host_writes) == 2
        for request in (cluster.request, peer):
            assert cluster.sender.get_num_new_matched_tokens(request, 0) == (8, False)
    finally:
        close_cluster(cluster)


def test_different_terminal_positions_are_not_coalesced(bridge_module, monkeypatch):
    cluster = make_cluster(bridge_module, monkeypatch)
    try:
        cluster.peer.num_prompt_tokens = 8
        output = metadata(cluster, {"r": 4, "peer": 4})
        assert [(plan.request_id, plan.committed_tokens) for plan in output.kv_connector_metadata.prefixes] == [
            ("r", 8), ("peer", 4)]
    finally:
        close_cluster(cluster)


@pytest.mark.parametrize("change", ["step", "epoch", "view", "namespace"])
def test_selection_never_persists_into_another_frame(bridge_module, monkeypatch, change):
    cluster = make_cluster(bridge_module, monkeypatch)
    try:
        first = metadata(cluster)
        first_plan = first.kv_connector_metadata.prefixes[0]
        first_index = cluster.sender._prefix_index_key(first_plan.base_key)
        if change == "epoch":
            cluster.sender.scheduler_state.reset()
        elif change == "view":
            cluster.sender.reset_cache()
        elif change == "namespace":
            cluster.sender.namespace = "different-PT-or-layout"
        second = metadata(cluster, {"peer": 8, "r": 4})
        plans = second.kv_connector_metadata.prefixes
        assert len(plans) == 1 and plans[0].request_id == "peer"  # Preserve this frame's order.
        assert second.kv_connector_metadata.step_id != first.kv_connector_metadata.step_id
        if change == "epoch":
            assert plans[0].epoch != first_plan.epoch
        if change in ("view", "namespace"):
            assert cluster.sender._prefix_index_key(plans[0].base_key) != first_index
        assert cluster.sender.get_num_new_matched_tokens(cluster.peer, 0) == (0, False)
    finally:
        close_cluster(cluster)


def test_one_rank_failure_cannot_publish_or_select_another_writer(bridge_module, monkeypatch):
    cluster = make_cluster(bridge_module, monkeypatch)
    try:
        output = metadata(cluster)
        deliver(cluster, save_rank(cluster, output, 0))
        original = cluster.store.batch_put_from_layers

        def fail(keys, addresses, sizes, direction):
            original(keys, addresses, sizes, direction)
            return [-1] * len(keys)

        monkeypatch.setattr(cluster.store, "batch_put_from_layers", fail)
        with pytest.raises(TransferError, match="invalid/failed"):
            save_rank(cluster, output, 1)
        assert not cluster.workers[1].build_connector_worker_meta().prefixes
        assert not cluster.store.host_writes
        assert cluster.sender.get_num_new_matched_tokens(cluster.peer, 0) == (0, False)
        assert len(cluster.store.data_writes) == 2  # No automatic second-writer fallback.
    finally:
        close_cluster(cluster)


def test_private_swaps_keep_separate_request_snapshots(bridge_module, monkeypatch):
    cluster = make_cluster(bridge_module, monkeypatch)
    try:
        first = cluster.sender.scheduler_state.before_preempt(cluster.request)
        second = cluster.sender.scheduler_state.before_preempt(cluster.peer)
        output = metadata(cluster, {})
        assert output.kv_connector_metadata.prefixes == ()
        assert output.kv_connector_metadata.preemptions == (first, second)
        assert first.identity != second.identity
    finally:
        close_cluster(cluster)


def test_existing_complete_manifest_skips_all_writers_but_missing_rank_retries(bridge_module, monkeypatch):
    cluster = make_cluster(bridge_module, monkeypatch)
    try:
        output = metadata(cluster)
        receipts = [save_rank(cluster, output, rank) for rank in range(2)]
        for receipt in receipts:
            deliver(cluster, receipt)
        assert metadata(cluster).kv_connector_metadata.prefixes == ()
        # A rank-1 GDN object disappearing invalidates the whole checkpoint;
        # the local same-step set must not retain an earlier positive result.
        missing = expected_keys(cluster.sender.namespace, 2, receipts[0].prefixes[0].plan,
                                cluster.sender.geometry)[-1]
        del cluster.store.objects[missing]
        retry = metadata(cluster).kv_connector_metadata.prefixes
        assert len(retry) == 1 and retry[0].request_id == "r"
    finally:
        close_cluster(cluster)


@pytest.mark.parametrize("retirement", ["request_cancel", "prefix_reset"])
def test_late_completions_preserve_cancel_and_reset_publication_boundaries(
        bridge_module, monkeypatch, retirement):
    cluster = make_cluster(bridge_module, monkeypatch)
    try:
        output = metadata(cluster)
        receipts = [save_rank(cluster, output, rank) for rank in range(2)]
        deliver(cluster, receipts[0])
        assert not cluster.store.host_writes
        if retirement == "request_cancel":
            cluster.sender.request_finished_all_groups(cluster.request, cluster.pages["r"])
        else:
            cluster.sender.reset_cache()
        deliver(cluster, receipts[1])
        if retirement == "request_cancel":
            # Cancellation does not corrupt already-completed immutable
            # checkpoint bytes; the other TP receipt still permits publication.
            assert len(cluster.store.host_writes) == 1
            assert cluster.sender.get_num_new_matched_tokens(cluster.peer, 0) == (8, False)
        else:
            assert not cluster.store.host_writes
            assert not cluster.sender._prefix_publications
            assert cluster.sender.get_num_new_matched_tokens(cluster.peer, 0) == (0, False)
    finally:
        close_cluster(cluster)


def test_prompt_suffix_after_checkpoint_does_not_change_first_writer_selection(bridge_module, monkeypatch):
    cluster = make_cluster(bridge_module, monkeypatch)
    try:
        cluster.peer.all_token_ids = [*cluster.peer.all_token_ids[:8], 91, 92, 93, 94]
        cluster.peer.block_hashes = native_hashes(monkeypatch, cluster.peer)
        assert cluster.peer.block_hashes[3] == cluster.request.block_hashes[3]
        assert cluster.peer.block_hashes[-1] != cluster.request.block_hashes[-1]
        plans = metadata(cluster).kv_connector_metadata.prefixes
        assert len(plans) == 1 and plans[0].request_id == "r"
    finally:
        close_cluster(cluster)
