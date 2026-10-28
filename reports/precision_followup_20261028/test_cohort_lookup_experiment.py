"""Matched-cohort overlay on unchanged native hash/lookup/lifecycle methods."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

from reports.precision_followup_20261028.test_cohort_gate_experiment import scenario
from reports.precision_followup_20261028.cohort_lookup_experiment import NoCoherentCohort


def hash_snapshot(s):
    pool = s.cache.block_pool
    return tuple((block.block_id, block.block_hash) for block in pool.blocks), tuple(
        (key, tuple(blocks) if isinstance(blocks, dict) else (blocks.block_id,))
        for key, blocks in pool.cached_block_hash_to_block._cache.items())


def original_trace(s):
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
    return a, x, b, full_a


def test_mixed_hit_corrected_same_length_and_existing_long_cohort_preserved():
    rows = []
    for policy in (None, 'lookup'):
        with scenario(policy) as s:
            a, x, b, full_a = original_trace(s)
            full_b = s.cache.single_type_managers[0].req_to_blocks['B'][0]
            before = s.physical_snapshot(), hash_snapshot(s)
            corrected, length = s.lookup(b)
            assert length == 2304
            assert corrected[0][0] is (full_b if policy else full_a)
            assert [row[-1] for row in corrected[1:]] == list(
                s.cache._oscar_prefix_checkpoint_leases.blocks_by_request['B'])
            assert (s.physical_snapshot(), hash_snapshot(s)) == before
            if policy:
                native_long = s.gate.native_lookup(x.block_hashes, x.num_tokens - 1)
                assert native_long[1] == 4608
                assert s.gate.correct(native_long) is native_long
            rows.append((s.physical_snapshot(), hash_snapshot(s)))
            s.clean()
    assert rows[0] == rows[1]


def test_three_gdn_first_entries_mixed_choose_one_complete_surviving_tuple():
    with scenario('lookup') as s:
        a = s.request('A', 5000)
        s.finish(a)
        a_state = [row[-1] for row in s.lookup(a)[0][1:]]
        b = s.request('B', 5000)
        s.finish(b, 768)
        b_state = list(s.cache._oscar_prefix_checkpoint_leases.blocks_by_request['B'])
        s.cache.block_pool.evict_blocks({a_state[0].block_id})
        native = s.gate.native_lookup(b.block_hashes, b.num_tokens - 1)
        assert [row[-1] for row in native[0][1:]] == [b_state[0], *a_state[1:]]
        before = s.physical_snapshot(), hash_snapshot(s)
        corrected = s.gate.correct(native)
        assert corrected[1] == native[1] == 2304
        assert [row[-1] for row in corrected[0][1:]] == b_state
        assert corrected[0][0][0] is s.cache.single_type_managers[0].req_to_blocks['B'][0]
        assert (s.physical_snapshot(), hash_snapshot(s)) == before
        s.clean()


def test_missing_producer_full_cannot_be_replaced_by_arbitrary_same_hash_data():
    with scenario('lookup') as s:
        a = s.request('A', 5000)
        s.finish(a)
        a_state = [row[-1] for row in s.lookup(a)[0][1:]]
        b = s.request('B', 5000)
        s.finish(b, 768)
        b_full = s.cache.single_type_managers[0].req_to_blocks['B'][0]
        s.cache.block_pool.evict_blocks({a_state[0].block_id, b_full.block_id})
        native = s.gate.native_lookup(b.block_hashes, b.num_tokens - 1)
        assert native[1] == 2304  # All token hashes exist; coherent payload does not.
        with pytest.raises(NoCoherentCohort):
            s.gate.correct(native)
        s.clean()


def test_native_same_step_hash_admission_and_lookahead_drop_are_preserved():
    with scenario('lookup') as s:
        a, x, b, _ = original_trace(s)
        native = s.gate.native_lookup(b.block_hashes, b.num_tokens - 1)
        corrected = s.gate.correct(native)
        for manager in s.cache.single_type_managers[1:]:
            group = manager.kv_cache_group_id
            value = native[0][group][-1].block_hash
            assert corrected[0][group][-1].block_hash == value
            manager.cached_blocks_this_step.add(value)
            for blocks in (native[0][group], corrected[0][group]):
                assert manager.get_num_blocks_to_allocate(
                    'new', 2308, blocks, 2304, 2308) > s.cache.block_pool.num_gpu_blocks
        # Native proof bounds hit length. Overlay must not undo one-page drop.
        for maximum in (2303, 2304, 4607, 4608, 4999):
            native = s.gate.native_lookup(b.block_hashes, maximum)
            assert s.gate.correct(native)[1] == native[1]
        s.clean()


def test_free_producer_capacity_is_accounted_by_native_admission_not_hidden():
    with scenario('lookup') as s:
        _, x, b, full_a = original_trace(s)
        full_b = s.cache.single_type_managers[0].req_to_blocks['B'][0]
        s.free(b)
        native = s.gate.native_lookup(b.block_hashes, b.num_tokens - 1)
        corrected = s.gate.correct(native)
        assert full_a.ref_cnt > 0 and full_b.ref_cnt == 0
        manager = s.cache.single_type_managers[0]
        native_evictable = manager._get_num_evictable_blocks(native[0][0])
        matched_evictable = manager._get_num_evictable_blocks(corrected[0][0])
        assert matched_evictable == native_evictable + 1
        before = s.cache.block_pool.get_num_free_blocks()
        s.active.add('continued-B')
        s.cache.allocate_new_computed_blocks('continued-B', corrected[0], corrected[1], 0)
        assert full_b.ref_cnt == 1
        after = s.cache.block_pool.get_num_free_blocks()
        assert before - after == 4  # Actual free FULL_B plus three GDN_B pages.
        report = {
            'backend': 'pinned native host contracts; no NPU/model/performance experiment',
            'production_enabled': False, 'native_hit_tokens': native[1],
            'matched_hit_tokens': corrected[1], 'native_FULL_free_pages_needed': native_evictable,
            'matched_FULL_free_pages_needed': matched_evictable,
            'native_references_before': {'FULL_A': full_a.ref_cnt, 'FULL_B': 0},
            'matched_adoption_free_pool_consumption': before - after,
            'extra_allocated_device_pages': 0,
            'capacity_note': 'Selecting the correct existing producer FULL can need an additional free-pool reference; native admission counts it.'}
        Path(__file__).with_name('cohort_lookup_capacity.json').write_text(json.dumps(report, indent=2) + '\n')
        s.clean()


def test_reuse_reset_and_original_coherent_object_identity():
    with scenario('lookup') as s:
        a = s.request('A', 5000)
        s.finish(a)
        native = s.gate.native_lookup(a.block_hashes, a.num_tokens - 1)
        assert s.gate.correct(native) is native
        assert not s.cache.block_pool.reset_prefix_cache()
        assert s.gate.records
        old_state = tuple(s.gate.records)
        s.free(a)
        s.pressure()
        assert not s.gate.records
        b = s.request('B', 5000)
        s.finish(b, 768)
        assert not any(identity in s.gate.records for identity in old_state)
        assert s.lookup(b)[1] == 2304
        s.clean()
