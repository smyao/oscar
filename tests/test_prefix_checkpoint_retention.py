"""Archive #37-49/#127/#128/#157: real native checkpoint publication/turnover.

CPU block ownership and hashes only. These are synthetic distinct identities
and bounded schedules, not model execution, GDN tensor values or NPU evidence.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oscar_ascend import plugin
from oscar_ascend.integration.prefix_checkpoint import (
    PrefixCheckpointLeases, policy_for_scheduler, split_prefix_checkpoint)
from test_runtime_native import definitions, native_specs, native_groups
from test_prefix_caching import _coordinator_with_hashing


def _setup(mp, pool_size, strategy):
    native = native_specs.__wrapped__(mp)
    interface, ours, _ = native
    convert = ours.transform_native_specs
    def real_specs(specs):
        configured = {}
        for name, spec in specs.items():
            args = {"page_size_padded": 817152}
            if isinstance(spec, interface.MambaSpec):
                args.update(shapes=((6, 2560), (12, 128, 128)), num_speculative_blocks=3)
            configured[name] = replace(spec, **args)
        return convert(configured, metadata_bytes=4)
    mp.setattr(ours, "transform_native_specs", real_specs)
    grouped = native_groups.__wrapped__(native, mp, NS(param=True))
    original = _coordinator_with_hashing(grouped, mp)
    cache = type(original)(replace(original.kv_cache_config, num_blocks=pool_size),
        max_model_len=262144, use_eagle=True, enable_caching=True,
        enable_kv_cache_events=False, dcp_world_size=1, pcp_world_size=1,
        hash_block_size=original.hash_block_size, max_num_batched_tokens=16384,
        scheduler_block_size=original.scheduler_block_size)
    methods = ("allocate_new_blocks", "free", "remove_skipped_blocks", "new_step_starts")
    source = definitions("references/vllm/vllm/v1/core/kv_cache_coordinator.py",
        "retention_native_coordinator_methods", set(),
        {"CrossAttentionManager": type("NoCrossAttention", (), {})}, mp,
        methods=[("KVCacheCoordinator", name) for name in methods] +
                [("HybridKVCacheCoordinator", "cache_blocks")])
    for name in (*methods, "cache_blocks"):
        mp.setattr(type(cache), name, getattr(source, name), raising=False)
    # Publish the exact same native helpers through their real import seam.
    utils = ModuleType("vllm.v1.core.kv_cache_utils")
    mp.setitem(sys.modules, utils.__name__, utils)
    helpers = definitions("references/vllm/vllm/v1/core/kv_cache_utils.py",
        "retention_native_hash_helpers", {"make_block_hash_with_group_id", "get_block_hash", "get_group_id"},
        {"BlockHash": bytes, "BlockHashWithGroupId": bytes}, mp)
    mp.setattr(utils, "make_block_hash_with_group_id", helpers.make_block_hash_with_group_id, raising=False)
    mp.setattr(utils, "BlockHashListWithBlockSize", grouped[3].BlockHashListWithBlockSize, raising=False)
    scheduler = NS(block_size=cache.lcm_block_size, kv_cache_manager=NS(coordinator=cache),
        cache_config=NS(block_size=768, enable_prefix_caching=True, mamba_cache_mode="align"),
        use_eagle=True)
    scheduler._oscar_prefix_checkpoint_policy = policy_for_scheduler(scheduler)
    splitter = definitions("references/vllm/vllm/v1/core/sched/scheduler.py",
        "retention_native_splitter", set(), {}, mp,
        methods=[("Scheduler", "_mamba_block_aligned_split")])._mamba_block_aligned_split
    scratch_releases = []
    if strategy == "lease":
        cache._oscar_prefix_checkpoint_leases = PrefixCheckpointLeases(
            cache, scheduler._oscar_prefix_checkpoint_policy)
        plugin._patch_prefix_checkpoint_coordinator(type(cache))
    elif strategy == "extra_scratch_first":
        for manager in cache.single_type_managers:
            if not isinstance(manager.kv_cache_spec, interface.MambaSpec):
                continue
            native_remove = manager.remove_skipped_blocks
            def remove(request_id, computed, manager=manager, original=native_remove):
                index = manager.last_state_block_idx.get(request_id)
                owned = manager.req_to_blocks[request_id]
                old = owned[index] if index is not None and index < len(owned) else None
                native_free = cache.block_pool.free_blocks
                def free(blocks, prepend=False):
                    blocks = list(blocks)
                    if (not prepend and len(blocks) == 1 and blocks[0] is old
                            and old is not None and not old.is_null and old.block_hash is None):
                        prepend = True
                        scratch_releases.append(old.block_id)
                    return native_free(blocks, prepend=prepend)
                cache.block_pool.free_blocks = free
                try:
                    return original(request_id, computed)
                finally:
                    cache.block_pool.free_blocks = native_free
            mp.setattr(manager, "remove_skipped_blocks", remove)
    return cache, scheduler, splitter, interface, scratch_releases


def turnover(pool_size=795, strategy="control", decode_steps=2048):
    with pytest.MonkeyPatch.context() as mp:
        cache, scheduler, splitter, interface, scratch_releases = _setup(mp, pool_size, strategy)
        try:
            requests, trace, evictions = [], [], []
            native_evict = cache.block_pool._maybe_evict_cached_block
            def evict(block):
                if block.block_hash is not None:
                    evictions.append(block.block_id)
                return native_evict(block)
            mp.setattr(cache.block_pool, "_maybe_evict_cached_block", evict)
            groups = [m.kv_cache_group_id for m in cache.single_type_managers
                      if isinstance(m.kv_cache_spec, interface.MambaSpec)]
            assert len(groups) == 3
            def hashes(request):
                request.block_hashes = [(request.index * 1000000 + i).to_bytes(8, "big")
                    for i in range(request.num_tokens // cache.hash_block_size)]
            def advance(request, scheduled, committed):
                before = request.num_computed_tokens
                cache.remove_skipped_blocks(request.request_id, before)
                cache.allocate_new_blocks(request.request_id, before + scheduled + 3,
                                          before + scheduled, 0)
                # Real HybridKVCacheCoordinator aligns cache publication to2304.
                # Calling each manager directly would incorrectly hash768 tails.
                cache.cache_blocks(request, before + committed)
                request.num_computed_tokens += committed
            def snapshot(phase):
                first = requests[0]
                checkpoints = [(i + 1) * cache.hash_block_size
                    for i, value in enumerate(first.block_hashes[:32768 // cache.hash_block_size])
                    if cache.block_pool.get_cached_block(value, groups)]
                leases = getattr(cache, "_oscar_prefix_checkpoint_leases", None)
                held = leases.blocks_by_request if leases else {}
                assert all(len(blocks) <= 3 for blocks in held.values())
                trace.append({"phase": phase,
                    "hit_tokens": cache.find_longest_cache_hit(first.block_hashes, 32767)[1],
                    "joint_checkpoints": [n for n in checkpoints if n % 2304 == 0],
                    "free_blocks": cache.block_pool.get_num_free_blocks(),
                    "hash_evictions": len(evictions), "leased_blocks": sum(map(len, held.values())),
                    "extra_scratch_preferred": len(scratch_releases)})
            for index in range(20):
                request = NS(index=index, request_id=f"request-{index}", num_prompt_tokens=32768,
                             num_tokens=32768, num_computed_tokens=0, has_encoder_inputs=False)
                hashes(request)
                while request.num_computed_tokens < request.num_prompt_tokens:
                    cache.new_step_starts()
                    proposed = min(16384, request.num_prompt_tokens - request.num_computed_tokens)
                    selected = split_prefix_checkpoint(scheduler, request, proposed)
                    count = splitter(scheduler, request, proposed) if selected is None else selected
                    assert count > 0
                    advance(request, count, count)
                requests.append(request)
            snapshot("after20prefill")
            for step in range(decode_steps):
                cache.new_step_starts()
                for request in requests:
                    request.num_tokens += 1
                    hashes(request)
                    advance(request, 4, 1)
                if step in (0, 255, 767, 1020, 1021, 1023, 1535, 2047):
                    snapshot(f"decode{step+1}")
            cache.free(requests[0].request_id)
            snapshot("after_first_completion")
            lookup = definitions("references/vllm/vllm/v1/core/kv_cache_manager.py",
                "retention_native_repeat_lookup", set(), {}, mp,
                methods=[("KVCacheManager", "get_computed_blocks")]).get_computed_blocks
            repeat = NS(request_id="request-21-repeat-first", num_tokens=32768,
                        block_hashes=requests[0].block_hashes[:32768 // cache.hash_block_size],
                        skip_reading_prefix_cache=False)
            kv_manager = NS(enable_caching=True, log_stats=False, coordinator=cache,
                            create_kv_cache_blocks=lambda blocks: blocks)
            _, repeat_hit = lookup(kv_manager, repeat)
            assert repeat_hit == trace[-1]["hit_tokens"]
            for request in requests[1:]:
                cache.free(request.request_id)
            leases = getattr(cache, "_oscar_prefix_checkpoint_leases", None)
            assert not leases or not leases.blocks_by_request
            assert cache.block_pool.get_num_free_blocks() == pool_size - 1
            assert all(b.ref_cnt == 0 for b in cache.block_pool.blocks if not b.is_null)
            return {"pool_blocks": pool_size, "strategy": strategy, "requests": 20,
                    "prompt_tokens": 32768, "scheduled_q": 4, "committed_increment": 1,
                    "decode_steps": decode_steps, "trace": trace,
                    "request21_get_computed_blocks_tokens": repeat_hit,
                    "free_all_no_leak": True, "max_extra_lease_blocks_per_request": 3}
        finally:
            plugin.unregister()


@pytest.mark.parametrize("pool_size", [795, 800])
def test_real_native_turnover_and_checkpoint_lease_retention(pool_size):
    control = turnover(pool_size, "control")
    retained = turnover(pool_size, "lease")
    assert control["trace"][0]["hit_tokens"] == 29952
    assert control["trace"][-1]["hit_tokens"] == 0
    assert control["request21_get_computed_blocks_tokens"] == 0
    assert all(point["hit_tokens"] == 29952 for point in retained["trace"])
    assert retained["trace"][0]["leased_blocks"] == 60
    assert retained["trace"][-1]["leased_blocks"] == 57
    assert retained["request21_get_computed_blocks_tokens"] == 29952


if __name__ == "__main__":
    rows = [turnover(pool, strategy) for pool in (795, 800)
            for strategy in ("control", "extra_scratch_first", "lease")]
    root = Path(__file__).resolve().parents[1]
    sources = ("tests/test_prefix_checkpoint_retention.py", "tests/test_runtime_native.py",
        "tests/test_prefix_caching.py", "oscar_ascend/integration/prefix_checkpoint.py", "oscar_ascend/plugin.py",
        "references/vllm/vllm/v1/core/kv_cache_coordinator.py",
        "references/vllm/vllm/v1/core/kv_cache_manager.py",
        "references/vllm/vllm/v1/core/single_type_kv_cache_manager.py",
        "references/vllm/vllm/v1/core/block_pool.py", "references/vllm/vllm/v1/core/kv_cache_utils.py",
        "references/vllm-ascend/vllm_ascend/patch/platform/patch_mamba_manager.py",
        "references/vllm-ascend/vllm_ascend/patch/platform/patch_kv_cache_coordinator.py")
    report = {"scope": "CPU real native coordinator/manager/pool/hash; synthetic request identities",
              "device_and_tensor_state": "not_run", "cases": rows,
              "source_sha256": {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in sources}}
    destination = root / "reports/prefix_turnover_20261023/retention_results.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    for row in rows:
        print(json.dumps({"pool": row["pool_blocks"], "strategy": row["strategy"],
                          "trace": row["trace"]}))
