"""Experimental host-only APC cohort publication gate; NOT production enabled.

Archive #37-49/#127/#128/#157/#171/#173: a token hash is not sufficient to
identify a hybrid checkpoint when chunking changes approximate attention.
Use native hash lookup and native eviction; do not change manager state,
references, tensors, or allocations. This *changes cache advertisements*.

The repair keeps one canonical actual FULL-prefix/GDN tuple per checkpoint.
When no complete cohort exists, native first-entry selection is made coherent
by withdrawing conflicting advertisements, never by swapping active tables.
This proves correctness/liveness for a newly recomputed prefix, but can remove
other valid descendants. It is an experiment, not a performance-safe default.
"""
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Identity:
    block_id: int
    generation: int
    block_hash: bytes


@dataclass(frozen=True)
class Cohort:
    full: tuple[Identity, ...]
    state: tuple[Identity, ...]


class CanonicalCohortGate:
    def __init__(self, coordinator, preserve_valid_descendants=False):
        from vllm.v1.kv_cache_interface import MambaSpec

        self.coordinator = coordinator
        self.pool = coordinator.block_pool
        self.full_managers = tuple(m for m in coordinator.single_type_managers
                                   if not isinstance(m.kv_cache_spec, MambaSpec))
        self.state_managers = tuple(m for m in coordinator.single_type_managers
                                    if isinstance(m.kv_cache_spec, MambaSpec))
        self.generations = [0] * len(self.pool.blocks)
        self.records = {}
        self.dependents = defaultdict(set)
        self.seen = {}
        self.pending = deque()
        self.draining = False
        self.stats = defaultdict(int)
        self.preserve_valid_descendants = preserve_valid_descendants

    def _identity(self, block):
        return Identity(block.block_id, self.generations[block.block_id], block.block_hash)

    def _alive(self, identity):
        block = self.pool.blocks[identity.block_id]
        return (self.generations[identity.block_id] == identity.generation
                and block.block_hash == identity.block_hash)

    def _canonical(self, identity):
        return (self._alive(identity) and
                self.pool.cached_block_hash_to_block.get_one_block(identity.block_hash)
                is self.pool.blocks[identity.block_id])

    def allocated(self, blocks):
        # Called after native allocation. Native eviction invalidates all old
        # dependencies before allocation increments the physical generation.
        for block in blocks:
            assert not self.dependents.get(block.block_id)
            self.generations[block.block_id] += 1

    def evicted(self, block_id):
        self.pending.extend(self.dependents.get(block_id, ()))
        if self.draining:
            return
        self.draining = True
        try:
            while self.pending:
                key = self.pending.popleft()
                record = self.records.pop(key, None)
                if record is None:
                    continue
                for identity in (*record.full, *record.state):
                    owners = self.dependents.get(identity.block_id)
                    if owners is not None:
                        owners.discard(key)
                        if not owners:
                            del self.dependents[identity.block_id]
                self.stats['invalidated_cohorts'] += 1
                self._withdraw(record.state)
        finally:
            self.draining = False

    def _withdraw(self, identities, keep=()):
        ids = {i.block_id for i in identities if i not in keep and self._alive(i)}
        self.stats['withdrawn_advertisements'] += len(ids)
        self.pool.evict_blocks(ids)

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
        # Alignment is native cache geometry, not input-length route selection.
        end = num_computed_tokens // self.coordinator.scheduler_block_size
        if end <= self.seen.get(request.request_id, 0):
            return
        self.seen[request.request_id] = end
        end *= self.coordinator.scheduler_block_size
        self.stats['publication_boundaries'] += 1
        state = []
        key = []
        for manager in self.state_managers:
            index = end // manager.block_size - 1
            owned = manager.req_to_blocks.get(request.request_id, ())
            expected = self._expected_hash(request, manager, index)
            if index >= len(owned) or owned[index].is_null:
                # Native align mode can skip intermediate states; do not
                # invent a GDN checkpoint for a skipped row.
                self._withdraw(state)
                return
            block = owned[index]
            if expected is None or block.block_hash != expected:
                self._withdraw(state)
                return
            state.append(self._identity(block))
            key.append(expected)
        state, key = tuple(state), tuple(key)
        if key in self.records:
            # A complete earlier trajectory is already addressable. Preserve
            # it rather than thrashing the native first entries for duplicates.
            self._withdraw(state, keep=self.records[key].state)
            self.stats['existing_cohort_preserved'] += 1
            return
        full = []
        for manager in self.full_managers:
            count = end // manager.block_size
            owned = manager.req_to_blocks.get(request.request_id, ())
            if len(owned) < count:
                self._withdraw(state)
                return
            for index, block in enumerate(owned[:count]):
                self.stats['full_dependencies_checked'] += 1
                expected = self._expected_hash(request, manager, index)
                if block.is_null or expected is None or block.block_hash != expected:
                    self._withdraw(state)
                    self.stats['unadvertised_full_rejection'] += 1
                    return
                full.append(self._identity(block))
        record = Cohort(tuple(full), state)
        if self.preserve_valid_descendants:
            # Preflight the whole promotion before mutating any advertisements.
            # Looking only at the first conflict is insufficient: evicting an
            # unowned first duplicate can expose another owned duplicate.
            for identity in record.full:
                blocks = self.pool.cached_block_hash_to_block._cache[identity.block_hash]
                values = blocks.values() if isinstance(blocks, dict) else (blocks,)
                for block in values:
                    if block.block_id == identity.block_id:
                        break
                    if self.dependents.get(block.block_id):
                        self._withdraw(state)
                        self.stats['valid_descendant_preserved'] += 1
                        return
        # Both FULL and GDN groups independently pick their first surviving
        # native entry. Retire all preceding conflicts as one host operation.
        # Native side effects are hash eviction/events, never tensor/table edits.
        for identity in (*record.full, *record.state):
            while not self._canonical(identity):
                selected = self.pool.cached_block_hash_to_block.get_one_block(identity.block_hash)
                if selected is None or not self._alive(identity):
                    self._withdraw(state)
                    return
                self.stats['promoted_conflicts'] += 1
                self.pool.evict_blocks({selected.block_id})
        self.records[key] = record
        for identity in (*record.full, *record.state):
            self.dependents[identity.block_id].add(key)
        self.stats['published_cohorts'] += 1

    def freed(self, request_id):
        self.seen.pop(request_id, None)

    def reset(self):
        self.records.clear()
        self.dependents.clear()
        self.seen.clear()
        self.pending.clear()

    def assert_coherent(self):
        for record in self.records.values():
            assert all(self._canonical(identity) for identity in (*record.full, *record.state))


def attach_experimental_gate(cache, monkeypatch, preserve_valid_descendants=False):
    """Test-only instance hooks; production plugin and native sources untouched."""
    gate = CanonicalCohortGate(cache, preserve_valid_descendants)
    pool = cache.block_pool
    original_evict = pool._maybe_evict_cached_block
    original_allocate = pool.get_new_blocks
    original_cache = cache.cache_blocks
    original_free = cache.free
    original_reset = pool.reset_prefix_cache

    def evict(block):
        changed = original_evict(block)
        if changed:
            gate.evicted(block.block_id)
        return changed

    def allocate(count):
        blocks = original_allocate(count)
        gate.allocated(blocks)
        return blocks

    def cached(request, count):
        result = original_cache(request, count)
        gate.observe(request, count)
        return result

    def freed(request_id):
        result = original_free(request_id)
        gate.freed(request_id)
        return result

    def reset():
        result = original_reset()
        if result:
            gate.reset()
        return result

    monkeypatch.setattr(pool, '_maybe_evict_cached_block', evict)
    monkeypatch.setattr(pool, 'get_new_blocks', allocate)
    monkeypatch.setattr(cache, 'cache_blocks', cached)
    monkeypatch.setattr(cache, 'free', freed)
    monkeypatch.setattr(pool, 'reset_prefix_cache', reset)
    return gate
