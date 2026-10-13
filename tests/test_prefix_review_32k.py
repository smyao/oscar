"""Archive #26/#37-49/#127/#128/#157: independent 32K APC protocol review.

Execute native hash, pool, managers and coordinator bodies. No fake null
blocks or fabricated GDN states; CPU evidence does not establish NPU results.
"""
from dataclasses import replace
import hashlib
import pickle
from types import SimpleNamespace

import pytest

from test_prefix_analysis_matrix import seed_at_checkpoints
from test_prefix_guard import guarded_coordinator, checkpoints
from test_runtime_native import native_specs, native_groups, definitions  # noqa: F401


@pytest.fixture
def native_hash_chain(monkeypatch):
    hashes = definitions("references/vllm/vllm/utils/hashing.py", "review_native_sha256",
        {"sha256"}, {"hashlib": hashlib, "pickle": pickle}, monkeypatch)
    tokens = definitions("references/vllm/vllm/v1/core/kv_cache_utils.py", "review_native_hash_tokens",
        {"hash_block_tokens"}, {"BlockHash": bytes, "NONE_HASH": bytes(32)}, monkeypatch)
    def chain(ids):
        result = []
        for begin in range(0, len(ids) - 767, 768):
            result.append(tokens.hash_block_tokens(hashes.sha256,
                result[-1] if result else None, ids[begin:begin + 768]))
        return result
    return chain


def target_coordinator(native_groups, monkeypatch):
    interface, ours, *_rest, groups = native_groups
    for group in groups:
        spec = group.kv_cache_spec
        if isinstance(spec, ours.OscarFullAttentionSpec):
            group.kv_cache_spec = replace(spec, conv_bytes=30720,
                native_page_size_bytes=817152, page_size_padded=817152)
        elif isinstance(spec, interface.MambaSpec):
            group.kv_cache_spec = replace(spec, shapes=((6, 2560), (12, 128, 128)),
                                         page_size_padded=817152)
    return guarded_coordinator(native_groups, monkeypatch)


MATRIX = [
    (2304, list(range(2304, 32257, 2304)) + [32768],
     [0, 2304, 2304, 4608, 11520, 16128, 29952, 29952]),
    (8192, [6912, 13824, 20736, 27648, 29952, 32768],
     [0, 0, 0, 0, 6912, 13824, 29952, 29952]),
    (16384, [16128, 29952, 32768],
     [0, 0, 0, 0, 0, 16128, 29952, 29952]),
    (66048, [29952, 32768],
     [0, 0, 0, 0, 0, 0, 29952, 29952]),
]
SHARED_TOKENS = (2304, 3072, 4608, 6912, 13824, 16896, 30720, 32768)


@pytest.mark.parametrize("budget,endpoints,expected", MATRIX)
def test_32k_partial_hit_is_exact_checkpoint_intersection(
        native_groups, monkeypatch, native_hash_chain, budget, endpoints, expected):
    instance = target_coordinator(native_groups, monkeypatch)
    ids = [(17 + 37 * i) % 32000 for i in range(32768)]
    hashes = native_hash_chain(ids)
    assert len(hashes) == 42 and all(len(h) == 32 for h in hashes)
    assert checkpoints(32768, budget) == endpoints
    seed_at_checkpoints(instance, endpoints, block_hashes=hashes)
    gids = [m.kv_cache_group_id for m in instance.single_type_managers[1:]]
    saved = [(i + 1) * 768 for i, h in enumerate(hashes)
             if instance.block_pool.get_cached_block(h, gids)]
    assert saved == [end for end in endpoints if end % 768 == 0]
    for shared, want in zip(SHARED_TOKENS, expected):
        fork = ids.copy()
        if shared < len(fork):
            fork[shared] = (fork[shared] + 1) % 32000
        branch_hashes = native_hash_chain(fork)
        assert branch_hashes[:shared // 768] == hashes[:shared // 768]
        assert all(a != b for a, b in zip(branch_hashes[shared // 768:], hashes[shared // 768:]))
        groups, hit = instance.find_longest_cache_hit(branch_hashes, 32767)
        assert hit == want
        # Independent witness: GDN must have a real saved state at the FULL
        # endpoint, and the next hash must still belong to the shared prefix.
        eligible = [end for end in saved if end % 2304 == 0 and end + 768 <= shared]
        assert hit == max(eligible, default=0)
        assert all(len(blocks) * manager.block_size == hit
                   for blocks, manager in zip(groups, instance.single_type_managers))


@pytest.mark.parametrize("width", [4, 16, 32])
@pytest.mark.parametrize("count", [43, 300])
def test_pool_and_lookup_keys_do_not_depend_on_hash_byte_width(native_groups, monkeypatch, width, count):
    instance = target_coordinator(native_groups, monkeypatch)
    full = instance.single_type_managers[0]
    raw = [i.to_bytes(width, "big") for i in range(count)]
    request = SimpleNamespace(request_id="key-width", num_prompt_tokens=count * 768,
                              block_hashes=raw)
    full.allocate_new_blocks(request.request_id, count * 768 + 3, count * 768)
    full.cache_blocks(request, count * 768)
    expected = []
    for start in range(0, count - 3, 3):
        key = b"".join(raw[start:start + 3]) + b"\x00OSCAR-MTP-LOOKAHEAD-v1\x00" + raw[start + 3]
        block = instance.block_pool.get_cached_block(key, [full.kv_cache_group_id])
        assert block is not None
        expected.extend(block)
    native_wrapped = native_groups[3].BlockHashListWithBlockSize(raw, 768, 2304)
    hits = full.find_longest_cache_hit(native_wrapped, count * 768 - 1,
        [full.kv_cache_group_id], instance.block_pool, full.kv_cache_spec, True, 2304)
    assert hits == (expected,)
    assert len(expected) == {43: 14, 300: 99}[count]


def test_gdn_eviction_reduces_to_earlier_real_checkpoint(native_groups, monkeypatch, native_hash_chain):
    instance = target_coordinator(native_groups, monkeypatch)
    ids = list(range(32768)); raw = native_hash_chain(ids)
    seed_at_checkpoints(instance, checkpoints(32768, 2304), block_hashes=raw)
    ids[13824] += 1
    fork = native_hash_chain(ids)
    assert instance.find_longest_cache_hit(fork, 32767)[1] == 11520
    gid = instance.single_type_managers[1].kv_cache_group_id
    last = instance.block_pool.get_cached_block(raw[14], [gid])[0]
    instance.block_pool.evict_blocks({last.block_id})
    assert instance.find_longest_cache_hit(fork, 32767)[1] == 9216
    for end in (2304, 4608, 6912, 9216):
        block = instance.block_pool.get_cached_block(raw[end // 768 - 1], [gid])[0]
        instance.block_pool.evict_blocks({block.block_id})
    # FULL is still cached, but it cannot supply GDN's missing recurrent state.
    full = instance.single_type_managers[0]
    wrapped = native_groups[3].BlockHashListWithBlockSize(fork, 768, 2304)
    assert len(full.find_longest_cache_hit(wrapped, 32767, [full.kv_cache_group_id],
        instance.block_pool, full.kv_cache_spec, True, 2304)[0]) == 5
    assert instance.find_longest_cache_hit(fork, 32767)[1] == 0


@pytest.mark.parametrize("budget,expected", [
    (2304, list(range(2304, 29953, 2304)) + [31488, 32768]),
    (8192, [7680, 15360, 23040, 30720, 31488, 32768]),
    (16384, [16128, 31488, 32768]),
    (66048, [31488, 32768]),
])
def test_native_scheduler_uses_cache_config_768_not_full_page_size(monkeypatch, budget, expected):
    native = definitions("references/vllm/vllm/v1/core/sched/scheduler.py", "review_native_split",
        set(), {}, monkeypatch, methods=[("Scheduler", "_mamba_block_aligned_split")])
    scheduler = SimpleNamespace(cache_config=SimpleNamespace(block_size=768), use_eagle=True)
    request = SimpleNamespace(num_computed_tokens=0, num_tokens=32768, num_prompt_tokens=32768)
    ends = []
    while request.num_computed_tokens < request.num_tokens:
        proposed = min(budget, request.num_tokens - request.num_computed_tokens)
        scheduled = native._mamba_block_aligned_split(scheduler, request, proposed)
        assert 0 < scheduled <= proposed
        request.num_computed_tokens += scheduled
        ends.append(request.num_computed_tokens)
    assert ends == expected
