"""Execute native ownership/hash/GDN code with OSCAR's guarded keys.

Archive #37-49/#127/#128/#142/#157: host protocol evidence only.
"""
from dataclasses import replace
import hashlib
import sys
from types import SimpleNamespace

import pytest
import torch

from oscar_ascend.integration.prefix_cache import (
    GuardedHashes, aligned_chunk_tokens, prefix_hit_bound, chunk_boundary_requests)
from test_prefix_caching import _coordinator_with_hashing
from test_prefix_analysis_matrix import seed_at_checkpoints
from test_runtime_native import native_groups, native_specs, definitions  # noqa: F401


def guarded_coordinator(native_groups, monkeypatch):
    *_, groups = native_groups
    groups[0].kv_cache_spec = replace(groups[0].kv_cache_spec, mtp_prefix_guard=True)
    monkeypatch.setitem(sys.modules, "vllm.v1.core.kv_cache_utils", native_groups[3])
    return _coordinator_with_hashing(native_groups, monkeypatch)


def checkpoints(prompt, budget):
    count, result = 0, []
    while count < prompt:
        proposed = min(budget, prompt - count)
        n = aligned_chunk_tokens(computed=count, proposed=proposed, prompt_tokens=prompt,
            full_tokens=2304, native_tokens=768, hash_tokens=768, guarded=True, mtp=True)
        assert 0 < n <= proposed
        count += n
        result.append(count)
    return result


def test_insufficient_remaining_budget_defers_like_native_without_partial_state():
    assert aligned_chunk_tokens(computed=0, proposed=767, prompt_tokens=9216,
        full_tokens=2304, native_tokens=768, hash_tokens=768, guarded=True, mtp=True) == 0


@pytest.mark.parametrize("prompt,expected", [
    (2303, 0), (2304, 0), (2305, 0), (3071, 0), (3072, 2304),
    (4608, 2304), (4609, 2304), (5376, 4608), (6912, 4608), (9216, 6912),
])
@pytest.mark.parametrize("budget", [768, 1536, 2048, 8192, 16384])
def test_real_sparse_native_checkpoints_hit_guarded_bound(native_groups, monkeypatch, prompt, expected, budget):
    instance = guarded_coordinator(native_groups, monkeypatch)
    req = seed_at_checkpoints(instance, checkpoints(prompt, budget))
    assert prefix_hit_bound(prompt, 2304, 768, guarded=True, mtp=True) == expected
    blocks, hit = instance.find_longest_cache_hit(req.block_hashes, prompt - 1)
    assert hit == expected
    assert all(len(group) * manager.block_size == hit
               for group, manager in zip(blocks, instance.single_type_managers))


def test_next_hash_fork_and_native_refcounts_reset_eviction(native_groups, monkeypatch):
    instance = guarded_coordinator(native_groups, monkeypatch)
    req = seed_at_checkpoints(instance, checkpoints(9216, 1536))
    hashes = list(req.block_hashes)
    # A different token immediately after page 0 must invalidate that page's
    # MTP state even though all 2304 tokens inside the FULL page match.
    changed = hashes[:3] + [b"new-chain"] * (len(hashes) - 3)
    assert instance.find_longest_cache_hit(changed, 9215)[1] == 0
    # Fork after the lookahead leaves earlier pages valid.
    changed = hashes[:6] + [b"new-chain"] * (len(hashes) - 6)
    assert instance.find_longest_cache_hit(changed, 9215)[1] == 2304
    blocks, hit = instance.find_longest_cache_hit(hashes, 9215)
    full = instance.single_type_managers[0]
    for fork in ("left", "right"):
        for manager, group in zip(instance.single_type_managers, blocks):
            manager.allocate_new_computed_blocks(fork, group, hit, 0)
            manager.allocate_new_blocks(fork, hit + 4, hit + 1)
    left, right = (full.req_to_blocks[name] for name in ("left", "right"))
    assert all(b.ref_cnt == 2 for b in left[:hit // 2304])
    assert left[-1].block_id != right[-1].block_id
    assert not instance.block_pool.reset_prefix_cache()
    for fork in ("left", "right"):
        for manager in instance.single_type_managers:
            manager.free(fork)
    victim = blocks[0][0]
    instance.block_pool.evict_blocks({victim.block_id})
    recycled = instance.block_pool.get_new_blocks(instance.block_pool.get_num_free_blocks())
    assert victim in recycled and victim.block_hash is None
    assert instance.find_longest_cache_hit(hashes, 9215)[1] == 0
    instance.block_pool.free_blocks(recycled)
    assert instance.block_pool.reset_prefix_cache()
    assert req.block_hashes == hashes


def test_full_page_not_published_before_committed_lookahead(native_groups, monkeypatch):
    instance = guarded_coordinator(native_groups, monkeypatch)
    full = instance.single_type_managers[0]
    req = SimpleNamespace(request_id="r", num_prompt_tokens=4608,
                          block_hashes=[hashlib.sha256(str(i).encode()).digest() for i in range(6)])
    full.allocate_new_blocks("r", 3075, 3072)
    full.cache_blocks(req, 2304)
    assert full.num_cached_block.get("r", 0) == 0
    full.cache_blocks(req, 3071)
    assert full.num_cached_block.get("r", 0) == 0
    full.cache_blocks(req, 3072)
    assert full.num_cached_block["r"] == 1
    assert len(GuardedHashes(req.block_hashes, 3)) == 3
    full.free("r")


def test_same_step_full_publication_cannot_bypass_older_gdn_checkpoint(native_groups, monkeypatch):
    instance = guarded_coordinator(native_groups, monkeypatch)
    req = SimpleNamespace(request_id="seed", num_prompt_tokens=5376,
                          block_hashes=[hashlib.sha256(str(i).encode()).digest() for i in range(7)])
    for count in (2304, 3072):
        for manager in instance.single_type_managers:
            manager.new_step_starts()
            manager.allocate_new_blocks(req.request_id, count + 3, count)
            manager.cache_blocks(req, count)
    blocks, hit = instance.find_longest_cache_hit(req.block_hashes, 5375)
    assert hit == 2304  # GDN checkpoint is old; FULL hash was published this step.
    needs = [manager.get_num_blocks_to_allocate("warm", hit + 4, group, hit, hit + 1)
             for manager, group in zip(instance.single_type_managers, blocks)]
    assert needs[0] > instance.block_pool.num_gpu_blocks
    assert all(n <= instance.block_pool.num_gpu_blocks for n in needs[1:])
    for manager in instance.single_type_managers:
        manager.new_step_starts()
    assert instance.find_longest_cache_hit(req.block_hashes, 5375)[1] == hit
    assert all(manager.get_num_blocks_to_allocate("warm", hit + 4, group, hit, hit + 1)
               <= instance.block_pool.num_gpu_blocks
               for manager, group in zip(instance.single_type_managers, blocks))
    # Deferral does not touch references or change the shared page identity.
    assert all(b.ref_cnt == 1 for b in blocks[0])


@pytest.mark.parametrize("guard", [False, True])
def test_full_admission_guard_only_affects_new_prefix_consumers(native_groups, monkeypatch, guard):
    instance = (guarded_coordinator if guard else _coordinator_with_hashing)(native_groups, monkeypatch)
    full = instance.single_type_managers[0]
    req = SimpleNamespace(request_id="seed", num_prompt_tokens=9216,
                          block_hashes=[hashlib.sha256(str(i).encode()).digest() for i in range(12)])
    full.allocate_new_blocks(req.request_id, 9219, 9216)
    full.cache_blocks(req, 9216)
    page = full.req_to_blocks[req.request_id][0]
    assert full.get_num_blocks_to_allocate("warm", 2308, [page], 2304, 2305) > instance.block_pool.num_gpu_blocks
    assert full.get_num_blocks_to_allocate("seed", 9220, [], 9216, 9217) == 0
    full.new_step_starts()
    assert full.get_num_blocks_to_allocate("warm", 2308, [page], 2304, 2305) <= instance.block_pool.num_gpu_blocks


def test_padded_mtp_fallback_matches_native_cpu_chunk_boundary(monkeypatch):
    import numpy as np
    native = definitions("references/vllm/vllm/v1/spec_decode/llm_base_proposer.py",
        "native_cpu_mtp_fallback", set(), {"torch": torch}, monkeypatch,
        methods=[("SpecDecodeBaseProposer", "prepare_next_token_ids_cpu")])
    class Request:
        num_prompt_tokens = 10
        num_computed_tokens = 3
        def get_token_id(self, i): return 100 + i
    requests = {"chunk": Request(), "last": Request(), "decode": Request()}
    batch = SimpleNamespace(req_ids=list(requests), num_reqs=3)
    runner = SimpleNamespace(vllm_config=SimpleNamespace(speculative_config=SimpleNamespace(method="qwen3_5_mtp")),
                             optimistic_seq_lens_cpu=torch.tensor([6, 10, 14]))
    corrected = chunk_boundary_requests(runner, requests, batch)
    assert corrected["chunk"].get_token_id(9) == 106
    assert corrected["last"] is requests["last"] and corrected["decode"] is requests["decode"]
    assert requests["chunk"].get_token_id(9) == 109
    native_result = native.prepare_next_token_ids_cpu(SimpleNamespace(input_ids=torch.zeros(1)),
        [[], [999], [998]], requests, batch, {"chunk": 3})
    assert native_result.tolist() == [corrected["chunk"].get_token_id(9), 999, 998]
    padded = definitions("references/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py",
        "native_padded_mtp_fallback", set(), {"torch": torch, "np": np,
        "DeviceOperator": SimpleNamespace(index_fill=lambda t, dim, idx, value: t.index_fill(dim, idx, value))},
        monkeypatch, methods=[("AscendSpecDecodeBaseProposer", "prepare_next_token_ids_padded")])
    backup = SimpleNamespace(np=np.zeros(3, dtype=np.int64))
    def copy_to_gpu(n): backup.gpu = torch.from_numpy(backup.np[:n].copy())
    backup.copy_to_gpu = copy_to_gpu
    proposer = SimpleNamespace(backup_next_token_ids=backup)
    batch.num_tokens_no_spec = np.array([10, 10, 14]); batch.vocab_size = 1000
    next_ids, counts = padded.prepare_next_token_ids_padded(proposer,
        torch.tensor([[321, -1], [999, -1], [997, 998]]), corrected, batch,
        torch.tensor([0]), 1)
    assert next_ids.tolist() == [106, 999, 998]
    assert counts.tolist() == [0, 1, 2]  # native discard/accept accounting unchanged


def test_scheduler_hook_uses_exact_native_proposal_and_restores(monkeypatch):
    from oscar_ascend import plugin
    import oscar_ascend.integration.prefix_cache as prefix
    seen = []
    class Scheduler:
        def _mamba_block_aligned_split(self, request, proposed, local=0, external=0):
            return proposed - 1
    original = Scheduler._mamba_block_aligned_split
    def split(s, request, proposed, local, external):
        seen.append((proposed, local, external))
        return 2304 if request == "oscar" else None
    monkeypatch.setattr(prefix, "scheduler_prefix_split", split)
    monkeypatch.setattr(plugin, "_patches", [])
    plugin._patch_prefix_scheduler(SimpleNamespace(Scheduler=Scheduler))
    assert Scheduler()._mamba_block_aligned_split("oscar", 3072, 10, 20) == 2304
    assert Scheduler()._mamba_block_aligned_split("native", 3072) == 3071
    assert seen == [(3072, 10, 20), (3072, 0, 0)]
    patch = plugin._patches.pop()
    setattr(Scheduler, patch.name, patch.original)
    assert Scheduler._mamba_block_aligned_split is original
