"""Archive #34/#36/#37-49/#127/#128/#157: APC boundary and sparse checkpoint evidence.

Execute the pinned native managers/coordinator through existing AST fixtures.
This is host protocol evidence, never device or model-quality acceptance.
"""
from types import SimpleNamespace

import pytest

from test_prefix_caching import _coordinator_with_hashing
from test_runtime_native import native_groups, native_specs  # noqa: F401


def seed_at_checkpoints(instance, checkpoints):
    total = checkpoints[-1]
    request = SimpleNamespace(
        request_id="analysis-seed", num_prompt_tokens=total,
        block_hashes=[i.to_bytes(4, "big")
                      for i in range(total // instance.hash_block_size)])
    previous = 0
    for count in checkpoints:
        for manager in instance.single_type_managers:
            manager.new_step_starts()
            manager.remove_skipped_blocks(request.request_id, previous)
            manager.allocate_new_blocks(request.request_id, count + 3, count)
            manager.cache_blocks(request, count)
        previous = count
    for manager in instance.single_type_managers:
        manager.free(request.request_id)
    return request


@pytest.mark.parametrize("prompt,expected", [
    (2303, 0), (2304, 0), (2305, 0), (4608, 0),
    (4609, 2304), (6912, 2304), (6913, 4608), (9216, 4608),
])
def test_mtp_prompt_minus_one_boundary(native_groups, monkeypatch, prompt, expected):
    instance = _coordinator_with_hashing(native_groups, monkeypatch)
    request = seed_at_checkpoints(instance, list(range(768, 9217, 768)))
    # KVCacheManager.get_computed_blocks supplies request.num_tokens - 1.
    _, hit = instance.find_longest_cache_hit(request.block_hashes, prompt - 1)
    assert hit == expected


@pytest.mark.parametrize("step,maximum,expected", [
    (768, 6912, 4608), (1536, 6912, 4608),
    (768, 4608, 2304), (1536, 4608, 0),
])
def test_sparse_gdn_checkpoints_reduce_joint_hit(native_groups, monkeypatch, step, maximum, expected):
    instance = _coordinator_with_hashing(native_groups, monkeypatch)
    request = seed_at_checkpoints(instance, list(range(step, 9217, step)))
    _, hit = instance.find_longest_cache_hit(request.block_hashes, maximum)
    assert hit == expected
