"""Experimental matched-cohort lookup overlay; NOT production enabled.

Archive #37-49/#127/#128/#157/#171/#173. Retain native hash advertisements,
state advance, allocation/free/lease and initial longest-length calculation.
When the native result mixes physical computation trajectories, select a
live recorded FULL-chain/GDN tuple at the same length and same hashes.

This deliberately changes the selected physical lookup result. It therefore
does not satisfy a literal 'native lookup results unchanged' constraint.
No performance/model-quality claim follows from these host contracts.
"""
from collections import defaultdict
from dataclasses import dataclass

from .cohort_gate_experiment import Identity


class NoCoherentCohort(RuntimeError):
    pass


@dataclass(frozen=True)
class LookupCohort:
    end: int
    full: tuple[tuple[Identity, ...], ...]
    state: tuple[Identity, ...]


class MatchedCohortLookup:
    def __init__(self, coordinator):
        from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

        if any(not isinstance(m.kv_cache_spec, (FullAttentionSpec, MambaSpec))
               for m in coordinator.single_type_managers):
            raise ValueError('cohort experiment supports FULL and align-mode Mamba only')
        if any(isinstance(m.kv_cache_spec, MambaSpec)
               and m.kv_cache_spec.mamba_cache_mode != 'align'
               for m in coordinator.single_type_managers):
            raise ValueError('cohort experiment requires native align-mode Mamba')

        self.coordinator = coordinator
        self.pool = coordinator.block_pool
        self.full_managers = tuple(m for m in coordinator.single_type_managers
                                   if not isinstance(m.kv_cache_spec, MambaSpec))
        self.state_managers = tuple(m for m in coordinator.single_type_managers
                                    if isinstance(m.kv_cache_spec, MambaSpec))
        self.generations = [0] * len(self.pool.blocks)
        self.records = {}
        self.by_hash = defaultdict(dict)
        self.dependents = defaultdict(set)
        self.seen = {}
        self.stats = defaultdict(int)

    def _identity(self, block):
        return Identity(block.block_id, self.generations[block.block_id], block.block_hash)

    def _alive(self, identity):
        return (self.generations[identity.block_id] == identity.generation
                and self.pool.blocks[identity.block_id].block_hash == identity.block_hash)

    @staticmethod
    def _dependencies(record):
        return (*record.state, *(identity for group in record.full for identity in group))

    @staticmethod
    def _hash_key(record):
        return record.end, tuple(i.block_hash for i in record.state)

    def allocated(self, blocks):
        for block in blocks:
            assert not self.dependents.get(block.block_id)
            self.generations[block.block_id] += 1

    def evicted(self, block_id):
        for key in tuple(self.dependents.get(block_id, ())):
            record = self.records.pop(key)
            hash_key = self._hash_key(record)
            del self.by_hash[hash_key][key]
            if not self.by_hash[hash_key]:
                del self.by_hash[hash_key]
            for identity in self._dependencies(record):
                owners = self.dependents[identity.block_id]
                owners.remove(key)
                if not owners:
                    del self.dependents[identity.block_id]
            self.stats['invalidated_cohorts'] += 1

    def _expected_hash(self, request, manager, index):
        from vllm.v1.core.kv_cache_utils import (
            BlockHashListWithBlockSize, make_block_hash_with_group_id)

        hashes = request.block_hashes
        if manager.block_size != self.coordinator.hash_block_size:
            hashes = BlockHashListWithBlockSize(hashes, self.coordinator.hash_block_size,
                                               manager.block_size)
        if index >= len(hashes):
            return None
        return make_block_hash_with_group_id(hashes[index], manager.kv_cache_group_id)

    def observe(self, request, num_computed_tokens):
        end = num_computed_tokens // self.coordinator.scheduler_block_size
        if end <= self.seen.get(request.request_id, 0):
            return
        self.seen[request.request_id] = end
        end *= self.coordinator.scheduler_block_size
        self.stats['publication_boundaries'] += 1
        state = []
        for manager in self.state_managers:
            index = end // manager.block_size - 1
            owned = manager.req_to_blocks.get(request.request_id, ())
            expected = self._expected_hash(request, manager, index)
            if (index >= len(owned) or owned[index].is_null or expected is None
                    or owned[index].block_hash != expected):
                return  # Native align may skip states. Never synthesize one.
            state.append(self._identity(owned[index]))
        state = tuple(state)
        if state in self.records:
            return
        full = []
        for manager in self.full_managers:
            count = end // manager.block_size
            owned = manager.req_to_blocks.get(request.request_id, ())
            if len(owned) < count:
                return
            group = []
            for index, block in enumerate(owned[:count]):
                self.stats['full_dependencies_checked'] += 1
                expected = self._expected_hash(request, manager, index)
                if block.is_null or expected is None or block.block_hash != expected:
                    return
                group.append(self._identity(block))
            full.append(tuple(group))
        record = LookupCohort(end, tuple(full), state)
        self.records[state] = record
        self.by_hash[self._hash_key(record)][state] = record
        for identity in self._dependencies(record):
            self.dependents[identity.block_id].add(state)
        self.stats['published_cohorts'] += 1

    def correct(self, native_result):
        groups, end = native_result
        if end == 0:
            return native_result
        state = tuple(self._identity(groups[m.kv_cache_group_id][-1]) for m in self.state_managers)
        candidates = self.by_hash.get((end, tuple(i.block_hash for i in state)), {})
        preferred = candidates.get(state)
        sequence = ([preferred] if preferred is not None else []) + [
            record for record in candidates.values() if record is not preferred]
        for record in sequence:
            if not all(self._alive(identity) for identity in self._dependencies(record)):
                continue
            # Original native hash-chain/length/EAGLE lookahead remains the
            # authority. A witness lookahead page is not returned/consumed and
            # need not have been generated by this cohort.
            full_match = True
            for manager, owned in zip(self.full_managers, record.full):
                row = groups[manager.kv_cache_group_id]
                if len(row) != len(owned) or any(
                        block.block_hash != identity.block_hash for block, identity in zip(row, owned)):
                    full_match = False
                    break
            if not full_match:
                continue
            if state == record.state and all(
                    all(block is self.pool.blocks[identity.block_id]
                        for block, identity in zip(groups[manager.kv_cache_group_id], owned))
                    for manager, owned in zip(self.full_managers, record.full)):
                self.stats['unchanged_coherent_hits'] += 1
                return native_result
            corrected = list(groups)
            for manager, owned in zip(self.full_managers, record.full):
                corrected[manager.kv_cache_group_id] = [self.pool.blocks[i.block_id] for i in owned]
            for manager, identity in zip(self.state_managers, record.state):
                row = groups[manager.kv_cache_group_id]
                corrected[manager.kv_cache_group_id] = [*row[:-1], self.pool.blocks[identity.block_id]]
            self.stats['corrected_hits'] += 1
            return tuple(corrected), end
        self.stats['no_complete_cohort'] += 1
        # Experiment intentionally does not invent an unsafe success or hide a
        # miss. A production design must define safe shorter native lookup here.
        raise NoCoherentCohort(f'no live coherent checkpoint at native hit {end}')

    def freed(self, request_id):
        self.seen.pop(request_id, None)

    def reset(self):
        self.records.clear()
        self.by_hash.clear()
        self.dependents.clear()
        self.seen.clear()

    def assert_coherent(self):
        for record in self.records.values():
            assert all(self._alive(identity) for identity in self._dependencies(record))


def attach_experimental_lookup(cache, monkeypatch):
    """Instance-only instrumentation; no native or production source mutation."""
    overlay = MatchedCohortLookup(cache)
    pool = cache.block_pool
    original_evict, original_allocate = pool._maybe_evict_cached_block, pool.get_new_blocks
    original_cache, original_free = cache.cache_blocks, cache.free
    original_reset, original_lookup = pool.reset_prefix_cache, cache.find_longest_cache_hit

    def evict(block):
        changed = original_evict(block)
        if changed:
            overlay.evicted(block.block_id)
        return changed

    def allocate(count):
        blocks = original_allocate(count)
        overlay.allocated(blocks)
        return blocks

    def cached(request, count):
        result = original_cache(request, count)
        overlay.observe(request, count)
        return result

    def freed(request_id):
        result = original_free(request_id)
        overlay.freed(request_id)
        return result

    def reset():
        result = original_reset()
        if result:
            overlay.reset()
        return result

    def lookup(block_hashes, max_length):
        return overlay.correct(original_lookup(block_hashes, max_length))

    monkeypatch.setattr(pool, '_maybe_evict_cached_block', evict)
    monkeypatch.setattr(pool, 'get_new_blocks', allocate)
    monkeypatch.setattr(cache, 'cache_blocks', cached)
    monkeypatch.setattr(cache, 'free', freed)
    monkeypatch.setattr(pool, 'reset_prefix_cache', reset)
    monkeypatch.setattr(cache, 'find_longest_cache_hit', lookup)
    overlay.native_lookup = original_lookup
    return overlay
