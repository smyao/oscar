"""Archive #37-49/#127/#171-173: same-length duplicate hits can cost capacity.

Executes unchanged pinned native pool/FULL-manager contracts on host objects.
No tensor, kernel, production file, serving process or NPU is modified.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


def main():
    import pytest
    from test_prefix_checkpoint_retention import _setup

    with pytest.MonkeyPatch.context() as mp:
        cache, _, _, interface, _ = _setup(mp, 5, "control")
        pool = cache.block_pool
        full = next(m for m in cache.single_type_managers
                    if isinstance(m.kv_cache_spec, interface.FullAttentionSpec))
        a, b = pool.get_new_blocks(2)
        req = NS(block_hashes=[b"a", b"b", b"c"], num_prompt_tokens=2304)
        for block in (a, b):
            pool.cache_full_blocks(req, [block], 0, 1, full.block_size, full.kv_cache_group_id)
        assert a.block_hash == b.block_hash
        # A remains active for X. Producer B has finished and its FULL page
        # is cached but evictable. Its hash still names the same token prefix.
        pool.free_blocks([b])
        held = pool.get_new_blocks(pool.get_num_free_blocks() - 1)
        assert b not in held and b.ref_cnt == 0 and a.ref_cnt == 1
        assert pool.cached_block_hash_to_block.get_one_block(a.block_hash) is a

        def required(block):
            return full.get_num_blocks_to_allocate(
                "new_request", 2308, [block], 2304, 2305)

        free = pool.get_num_free_blocks()
        need_a, need_b = required(a), required(b)
        assert (free, need_a, need_b) == (1, 1, 2)
        pool.touch([a])
        after_a = pool.get_num_free_blocks()
        pool.free_blocks([a])
        pool.touch([b])
        after_b = pool.get_num_free_blocks()
        pool.free_blocks([b])
        assert after_a == free and after_b == free - 1

        result = {
            "scope": "actual pinned native host FULL manager/pool; synthetic duplicate hash, no device inference",
            "same_prefix_tokens": 2304, "canonical_active_full": a.block_id,
            "producer_idle_full": b.block_id, "canonical_refcount": a.ref_cnt,
            "producer_refcount": b.ref_cnt, "free_pages": free,
            "native_needed_canonical": need_a, "native_needed_producer": need_b,
            "canonical_admission_fits": need_a <= free,
            "producer_admission_fits": need_b <= free,
            "free_after_canonical_touch": after_a, "free_after_producer_touch": after_b,
            "no_new_device_allocation_does_not_imply_no_capacity_cost": True,
            "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in (
                "references/vllm/vllm/v1/core/single_type_kv_cache_manager.py",
                "references/vllm/vllm/v1/core/block_pool.py",
                "tests/test_prefix_checkpoint_retention.py",
                "reports/precision_followup_20261028/review_apc_capacity.py")},
        }
    path = Path(__file__).with_name("apc_capacity_review.json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"result": "same-hit-length can change native admission", "evidence": str(path)}))


if __name__ == "__main__":
    main()
