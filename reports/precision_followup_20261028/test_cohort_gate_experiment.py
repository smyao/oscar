"""Native lifecycle evidence for experimental cohort advertisements only."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

from oscar_ascend import plugin
from oscar_ascend.integration.prefix_checkpoint import split_prefix_checkpoint
from reports.precision_followup_20261028.cohort_gate_experiment import attach_experimental_gate
from test_prefix_checkpoint_retention import _setup
from test_runtime_native import definitions


class Scenario:
    def __init__(self, mp, gate_policy):
        self.cache, self.scheduler, self.native_split, self.interface, _ = _setup(mp, 160, 'lease')
        if gate_policy == 'lookup':
            from reports.precision_followup_20261028.cohort_lookup_experiment import attach_experimental_lookup
            self.gate = attach_experimental_lookup(self.cache, mp)
        else:
            self.gate = (attach_experimental_gate(self.cache, mp, gate_policy == 'preserve')
                         if gate_policy else None)
        adopt = definitions('references/vllm/vllm/v1/core/kv_cache_coordinator.py',
            'cohort_gate_native_adopt', set(), {}, mp,
            methods=[('KVCacheCoordinator', 'allocate_new_computed_blocks')]).allocate_new_computed_blocks
        mp.setattr(type(self.cache), 'allocate_new_computed_blocks', adopt, raising=False)
        self.active = set()

    def request(self, name, prompt, shared=0):
        hashes = [(i if i < shared else (100000 if name.startswith('X') else 0) + i).to_bytes(8, 'big')
                  for i in range(prompt // self.cache.hash_block_size)]
        self.active.add(name)
        return NS(request_id=name, num_prompt_tokens=prompt, num_tokens=prompt,
                  num_computed_tokens=0, has_encoder_inputs=False, block_hashes=hashes)

    def physical_snapshot(self):
        pool = self.cache.block_pool
        return (
            tuple(block.ref_cnt for block in pool.blocks), pool.get_num_free_blocks(),
            tuple(tuple((request_id, tuple(b.block_id for b in blocks))
                        for request_id, blocks in sorted(manager.req_to_blocks.items()))
                  for manager in self.cache.single_type_managers),
            tuple(tuple(sorted(manager.num_cached_block.items()))
                  for manager in self.cache.single_type_managers),
            tuple(tuple(sorted(getattr(manager, 'last_state_block_idx', {}).items()))
                  for manager in self.cache.single_type_managers),
        )

    def advance(self, request, count):
        before = request.num_computed_tokens
        end = before + count
        cache = self.cache
        cache.remove_skipped_blocks(request.request_id, before)
        cache.allocate_new_blocks(request.request_id, end + 3, end, 0)
        cache.cache_blocks(request, end)
        request.num_computed_tokens = end
        if self.gate:
            self.gate.assert_coherent()

    def finish(self, request, remaining=16384):
        while request.num_computed_tokens < request.num_prompt_tokens:
            self.cache.new_step_starts()
            proposed = min(remaining, request.num_prompt_tokens - request.num_computed_tokens)
            selected = split_prefix_checkpoint(self.scheduler, request, proposed)
            count = self.native_split(self.scheduler, request, proposed) if selected is None else selected
            assert count > 0
            self.advance(request, count)
        self.cache.new_step_starts()

    def lookup(self, request):
        return self.cache.find_longest_cache_hit(request.block_hashes, request.num_tokens - 1)

    def fork(self, name, source, prompt, shared_tokens=None):
        shared_tokens = source.num_prompt_tokens if shared_tokens is None else shared_tokens
        request = self.request(name, prompt, shared_tokens // self.cache.hash_block_size)
        computed, hit = self.lookup(request)
        self.cache.new_step_starts()
        self.cache.allocate_new_computed_blocks(name, computed, hit, 0)
        request.num_computed_tokens = hit
        return request, hit

    def free(self, request):
        self.cache.free(request.request_id)
        self.active.remove(request.request_id)

    def pressure(self):
        pool = self.cache.block_pool
        blocks = pool.get_new_blocks(pool.get_num_free_blocks())
        pool.free_blocks(blocks)

    def clean(self):
        for name in tuple(self.active):
            self.cache.free(name)
        assert self.cache.block_pool.get_num_free_blocks() == 159
        assert all(b.ref_cnt == 0 for b in self.cache.block_pool.blocks if not b.is_null)
        assert self.cache.block_pool.reset_prefix_cache()
        if self.gate:
            assert not self.gate.records and not self.gate.dependents and not self.gate.seen


@contextmanager
def scenario(policy='promote'):
    with pytest.MonkeyPatch.context() as mp:
        try:
            yield Scenario(mp, policy)
        finally:
            plugin.unregister()


def test_cohort_gate_repairs_mixed_native_hit_without_extra_pages():
    rows = []
    for policy in (None, 'promote'):
        with scenario(policy) as s:
            a = s.request('A', 5000)
            s.finish(a)
            full_a = s.cache.single_type_managers[0].req_to_blocks['A'][0]
            x, hit = s.fork('X', a, 8000)
            assert hit == 2304
            s.free(a)
            s.finish(x)
            s.pressure()
            b = s.request('B', 5000)
            assert s.lookup(b)[1] == 0
            s.finish(b, 768)
            full_b = s.cache.single_type_managers[0].req_to_blocks['B'][0]
            blocks, hit = s.lookup(b)
            assert hit == 2304
            assert blocks[0][0] is (full_b if policy else full_a)
            assert [group[-1] for group in blocks[1:]] == list(
                s.cache._oscar_prefix_checkpoint_leases.blocks_by_request['B'])
            assert max(map(len, s.cache._oscar_prefix_checkpoint_leases.blocks_by_request.values())) <= 3
            rows.append(s.physical_snapshot())
            s.clean()
    assert rows[0] == rows[1]


@pytest.mark.parametrize('policy', ['promote', 'preserve'])
def test_duplicate_recompute_exposes_promote_policy_shorter_checkpoint_loss(policy):
    with scenario(policy) as s:
        a = s.request('A', 5000)
        s.finish(a)
        before, hit = s.lookup(a)
        b = s.request('B', 5000)
        s.finish(b, 768)  # Deliberately duplicate a prompt already computing elsewhere.
        after, after_hit = s.lookup(b)
        assert hit == 2304
        assert after_hit == (0 if policy == 'promote' else 2304)
        if policy == 'preserve':
            assert before == after
        x, hit = s.fork('X', a, 8000)
        assert hit == (0 if policy == 'promote' else 2304)
        s.finish(x)
        assert s.lookup(x)[1] == (0 if policy == 'promote' else 4608)
        assert not s.cache.block_pool.reset_prefix_cache()
        assert s.gate.records
        s.clean()


def test_single_gdn_eviction_invalidates_whole_tuple_and_reuse_cannot_revive_it():
    with scenario() as s:
        a = s.request('A', 5000)
        s.finish(a)
        blocks, _ = s.lookup(a)
        state = [group[-1] for group in blocks[1:]]
        before = s.physical_snapshot()
        s.cache.block_pool.evict_blocks({state[0].block_id})
        assert s.physical_snapshot() == before
        assert all(block.block_hash is None for block in state)
        assert s.lookup(a)[1] == 0
        s.free(a)
        s.pressure()
        b = s.request('B', 5000)
        s.finish(b, 768)
        assert s.lookup(b)[1] == 2304
        s.clean()


def valid_descendant_tradeoff(policy):
    with scenario(policy) as s:
        a = s.request('A', 5000)
        s.finish(a)
        a_state = [group[-1] for group in s.lookup(a)[0][1:]]
        x, hit = s.fork('X', a, 11000, shared_tokens=4608)
        assert hit == 2304  # EAGLE witnesses one extra shared FULL page.
        s.finish(x)
        s.free(a)
        assert s.lookup(x)[1] == 6912
        # Normal active X ownership keeps FULL_A while the old A-only GDN
        # tuple may be evicted. Explicit native hash eviction isolates that
        # one lifecycle edge without destroying X's valid longer checkpoint.
        s.cache.block_pool.evict_blocks({b.block_id for b in a_state})
        assert s.lookup(x)[1] == 6912
        b = s.request('B', 5000)
        assert s.lookup(b)[1] == 0
        s.finish(b, 768)
        b_hit, x_hit = s.lookup(b)[1], s.lookup(x)[1]
        c = s.request('C', 5000)
        s.finish(c, 768)
        c_hit = s.lookup(c)[1]
        report = {'policy': policy, 'long_prefix_before': 6912,
                  'short_B_hit': b_hit, 'long_prefix_after': x_hit,
                  'short_C_hit_after_recompute': c_hit, 'stats': dict(s.gate.stats)}
        s.clean()
        return report


def test_promoting_new_cohort_loses_valid_descendant_but_preserving_it_can_repeat_misses():
    promote = valid_descendant_tradeoff('promote')
    preserve = valid_descendant_tradeoff('preserve')
    assert (promote['short_B_hit'], promote['long_prefix_after']) == (2304, 4608)
    assert (preserve['short_B_hit'], preserve['long_prefix_after']) == (0, 6912)
    assert preserve['short_C_hit_after_recompute'] == 0
    destination = Path(__file__).with_name('cohort_tradeoff_results.json')
    destination.write_text(json.dumps({'backend': 'pinned native host contracts; synthetic identities',
        'production_enabled': False, 'cases': [promote, preserve]}, indent=2) + '\n')
