"""Archive #26/#37-49/#127/#128/#156/#157: continuous CPU lifecycle simulation.

Native allocation, hashing, prefix lookup and release execute unchanged ASTs.
Only the bounded exact-window writer/reader are scalar CPU models. This does
not execute Qwen weights, GDN arithmetic, AscendC pipelines or an ACL graph.
"""
import random
from types import SimpleNamespace

import pytest
import torch

from oscar_ascend.integration.prefix_scheduler import configure_prefix_scheduler
from oscar_ascend.lifecycle import SnapshotLayout, causal_segments
from test_runtime_native import (native_specs, native_groups, _native_coordinator,
                                 _prefix_scheduler)


@pytest.fixture
def qwen_mtp_groups(native_groups):
    # Actual Qwen3.5 TP4/MTP3 shapes, independently derived from native model
    # and Ascend config methods in test_runtime_native. Older generic fixtures
    # deliberately also cover the smaller non-expanded conv state.
    interface, ours, registry, utils, config, _ = native_groups
    full = interface.FullAttentionSpec(block_size=768, num_kv_heads=1, head_size=256,
        dtype=torch.bfloat16, page_size_padded=817152)
    gdn = interface.MambaSpec(block_size=768, shapes=((6, 2560), (12, 128, 128)),
        dtypes=(torch.bfloat16, torch.bfloat16), page_size_padded=817152,
        num_speculative_blocks=3, mamba_cache_mode="align")
    specs = {f"full.{i}": full for i in range(17)} | {f"gdn.{i}": gdn for i in range(48)}
    groups = utils.get_kv_cache_groups(config, ours.transform_native_specs(specs))
    assert groups[0].kv_cache_spec.conv_bytes == 30720
    return interface, ours, registry, utils, config, groups


class SnapshotReplay:
    def __init__(self, manager, snapshot, seed):
        self.manager, self.snapshot = manager, snapshot
        self.rows = {}
        self.random = random.Random(seed)

    def address(self, request, position):
        size = self.snapshot.page.block_size
        blocks = self.manager.req_to_blocks[request.request_id]
        block = blocks[position // size]
        assert not block.is_null
        return block.block_id, self.snapshot.row_for_position(position), position % size

    def publish(self, request, begin, end, identity):
        # Match the production writer's keepRecent ownership rule, then
        # shuffle writes to exercise disjoint ownership across simulated cores.
        size, ring = self.snapshot.page.block_size, self.snapshot.ring_tokens
        writes = []
        for p in range(begin, end):
            page_end = min(end, (p // size + 1) * size)
            if p < self.snapshot.sink_tokens or p + ring >= page_end:
                writes.append((self.address(request, p), (identity, p)))
        self.random.shuffle(writes)
        assert len({(page, row) for (page, row, _), _value in writes}) == len(writes)
        for (page, row, tag), value in writes:
            self.rows[page, row] = tag, value

    def read_window(self, request, context, query, identity):
        sink, _history, recent, _current = causal_segments(
            context, query, self.snapshot.sink_tokens, self.snapshot.recent_tokens)
        for start, end in (sink, recent):
            for p in range(start, end):
                page, row, tag = self.address(request, p)
                assert self.rows.get((page, row)) == (tag, (identity, p)), \
                    f"window mismatch request={request.request_id} position={p} page={page} row={row}"


def test_memory_worksheet_matches_native_pool_and_does_not_hide_workspace(qwen_mtp_groups):
    from tools.local_simulation import memory_ledger
    _, _, _, utils, config, groups = qwen_mtp_groups
    ledger = memory_ledger({"max_num_batched_tokens": 32768})
    layout = groups[0].kv_cache_spec.layout
    assert ledger["tokens_per_full_page"]["oscar"] == layout.block_size == 2304
    assert ledger["native_page_bytes"] == layout.page_size_bytes == 817152
    assert ledger["snapshot_bytes_inside_page_padding"] < ledger["page_padding_bytes"]
    assert ledger["oscar_workspace_bytes_per_rank"] == 1259537664
    for row in ledger["scenarios"]:
        pool = utils.get_kv_cache_config_from_groups(config, groups, row["oscar_live_pool_bytes"])
        assert pool.num_blocks == row["oscar_live_pool_blocks"]
        assert sum(tensor.size for tensor in pool.kv_cache_tensors) == row["oscar_live_pool_bytes"]
        assert row["oscar_live_pool_bytes"] < row["native_live_pool_bytes"]
    # A small request does not earn back the candidate's fixed workspace.
    assert ledger["scenarios"][0]["pool_saved_minus_oscar_workspace_bytes"] < 0
    assert ledger["scenarios"][-1]["pool_saved_minus_oscar_workspace_bytes"] > 0


def allocate_step(managers, request, end):
    for manager in managers:
        manager.new_step_starts()
        manager.remove_skipped_blocks(request.request_id, request.num_computed_tokens)
        manager.allocate_new_blocks(request.request_id, end + 3, end)


@pytest.mark.parametrize("seed", [0, 17, 91])
@pytest.mark.parametrize("length", [4736, 20000, 32768, 50000, 262140])
def test_prefill_prefix_forks_mtp_rejection_release_and_reuse(qwen_mtp_groups, monkeypatch, length, seed):
    coordinator = _native_coordinator(qwen_mtp_groups, monkeypatch, budget=32768)
    scheduler = _prefix_scheduler(qwen_mtp_groups, coordinator, budget=32768, use_eagle=True)
    policy = configure_prefix_scheduler(scheduler)
    managers = coordinator.single_type_managers
    full = managers[0]
    snapshot = SnapshotLayout(full.kv_cache_spec.layout, 64, 256, 3)
    target = SnapshotReplay(full, snapshot, seed)
    draft = SnapshotReplay(full, snapshot, seed + 1)
    free = coordinator.block_pool.get_num_free_blocks()
    cold = SimpleNamespace(request_id="cold", num_tokens=length, num_prompt_tokens=length,
        num_computed_tokens=0, block_hashes=[i.to_bytes(32, "big") for i in range(length // 768)])
    while cold.num_computed_tokens < length:
        begin = cold.num_computed_tokens
        count = policy.split(cold, min(32768, length - begin))
        assert count > 0
        end = begin + count
        allocate_step(managers, cold, end)
        for cache in (target, draft):
            cache.read_window(cold, begin, begin, 1)
            cache.read_window(cold, begin, end - 1, 1)
            cache.publish(cold, begin, end, 1)
        for manager in managers:
            manager.cache_blocks(cold, end)
        cold.num_computed_tokens = end
    # Keep cold active while two forks hit its immutable FULL/GDN prefix.
    hits, hit_tokens = coordinator.find_longest_cache_hit(cold.block_hashes, length - 1)
    assert hit_tokens > 0 and hit_tokens % snapshot.page.block_size == 0
    forks = []
    for name in ("warm", "continuation"):
        request = SimpleNamespace(request_id=name, num_computed_tokens=hit_tokens)
        for manager, blocks in zip(managers, hits):
            manager.allocate_new_computed_blocks(name, blocks, hit_tokens, 0)
        forks.append(request)
    shared = {block.block_id for block in hits[0]}
    original_shared = {key: value for key, value in target.rows.items() if key[0] in shared}
    for cycle in range(24):
        # Reorder request rows every step; physical identity owns snapshots.
        for request in forks[::1 if cycle % 2 else -1]:
            begin = request.num_computed_tokens
            accepted = (cycle + (request.request_id == "warm")) % 4
            allocate_step(managers, request, begin + 6)
            for cache in (target, draft):
                cache.read_window(request, begin, begin, 1)
                cache.read_window(request, begin, begin + 3, 1)
                cache.publish(request, begin, begin + 4, 1)
            for step in (1, 2):
                position = begin + accepted + step
                draft.read_window(request, position, position, 1)
                draft.publish(request, position, position + 1, 1)
            request.num_computed_tokens = begin + accepted + 1
        assert {key: value for key, value in target.rows.items() if key[0] in shared} == original_shared
        tails = [{b.block_id for b in full.req_to_blocks[r.request_id][len(hits[0]):]} for r in forks]
        assert tails[0].isdisjoint(tails[1] | shared) and tails[1].isdisjoint(shared)
    # Release all native GDN speculative states; not a scheduler preemption.
    for request in (cold, *forks):
        for manager in managers:
            manager.free(request.request_id)
            manager.new_step_starts()
    assert coordinator.block_pool.get_num_free_blocks() == free
    # Reallocate the entire free list, not just pages never touched by this run.
    recycled = coordinator.block_pool.get_new_blocks(free)
    assert all(block.ref_cnt == 1 and block.block_hash is None for block in recycled)
    coordinator.block_pool.free_blocks(recycled)
    fresh = SimpleNamespace(request_id="recycled", num_computed_tokens=0)
    allocate_step(managers, fresh, 400)
    target.publish(fresh, 0, 400, 99)
    target.read_window(fresh, 400, 400, 99)
    # Negative control: a stale exact snapshot must fail even if its in-page
    # tag happens to match the recycled request's logical position.
    page, row, tag = target.address(fresh, 200)
    target.rows[page, row] = tag, (1, 200)
    with pytest.raises(AssertionError, match="window mismatch"):
        target.read_window(fresh, 400, 400, 99)
    for manager in managers:
        manager.free(fresh.request_id)
    assert coordinator.block_pool.get_num_free_blocks() == free
