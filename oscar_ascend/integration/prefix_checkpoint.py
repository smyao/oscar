"""Native scheduler checkpoint boundaries for OSCAR hybrid prefix caching.

Archive #26/#37-49: advertise only real GDN states paired with full exact
pages; no synthetic intermediate state, extra snapshot, or commit change.
Archive #127/#128: Mamba state blocks, FULL physical pages and metadata128
are distinct. Never mutate CacheConfig or any native cache spec.
Archive #157/#163: keep native MTP admission/rollback and decode dispatch.
Evidence: native scheduler.py:_mamba_block_aligned_split used CacheConfig's
768 after OSCAR changed FULL to 2304; AscendMambaManager queries the LCM.
Archive #37-49/#171: retain the existing prompt checkpoint with native pool
references until its request is freed. Never copy or manufacture GDN states.
"""

from dataclasses import dataclass
from math import lcm


@dataclass(frozen=True)
class PrefixCheckpointPolicy:
    alignment_tokens: int
    state_block_tokens: int
    full_lookup_blocks: tuple[tuple[int, bool], ...]

    def terminal_position(self, num_tokens: int) -> int:
        # KVCacheManager queries L-1 to recompute logits. Each EAGLE FULL
        # group additionally needs one whole page of lookahead. Using L
        # here would miss again when the prompt ends exactly on a page.
        readable = max(num_tokens - 1, 0)
        limit = min(max(readable // block - int(drop), 0) * block
                    for block, drop in self.full_lookup_blocks)
        return limit // self.alignment_tokens * self.alignment_tokens


class PrefixCheckpointLeases:
    """Keep one real joint prompt checkpoint while FULL pages remain owned.

    Native align-mode GDN removes old states from its request block table.
    Without an additional owner, decode allocation can evict every prompt
    checkpoint even while that request still owns all FULL prefix pages.
    Hold only the final readable prompt checkpoint, not decode history.
    The extra references consume the existing native pool's capacity and
    therefore participate in its ordinary admission/preemption accounting.
    """

    def __init__(self, coordinator, policy: PrefixCheckpointPolicy):
        from vllm.v1.kv_cache_interface import MambaSpec

        self.coordinator = coordinator
        self.policy = policy
        self.managers = tuple(m for m in coordinator.single_type_managers
                              if isinstance(m.kv_cache_spec, MambaSpec))
        self.blocks_by_request: dict[str, tuple] = {}

    def observe(self, request, num_computed_tokens: int) -> None:
        if request.request_id in self.blocks_by_request:
            return
        # Output tokens and speculative lookahead must never move the anchor.
        terminal = self.policy.terminal_position(request.num_prompt_tokens)
        if terminal == 0 or num_computed_tokens < terminal:
            return
        from vllm.v1.core.kv_cache_utils import (
            BlockHashListWithBlockSize, make_block_hash_with_group_id)

        blocks = []
        for manager in self.managers:
            index = terminal // manager.block_size - 1
            owned = manager.req_to_blocks.get(request.request_id, ())
            if index >= len(owned) or owned[index].is_null:
                return  # A partial/external resume may never own this checkpoint.
            block = owned[index]
            hashes = request.block_hashes
            if manager.block_size != self.coordinator.hash_block_size:
                hashes = BlockHashListWithBlockSize(
                    hashes, self.coordinator.hash_block_size, manager.block_size)
            if index >= len(hashes):
                return
            expected = make_block_hash_with_group_id(hashes[index], manager.kv_cache_group_id)
            if block.block_hash != expected:
                return  # Require native published identity, not just a token count.
            blocks.append(block)
        # Acquire the complete same-position group atomically on the scheduler
        # thread. Publication is not device-completion evidence: native same-step
        # admission barriers still apply. No new page, state copy or device op.
        self.coordinator.block_pool.touch(blocks)
        self.blocks_by_request[request.request_id] = tuple(blocks)

    def release(self, request_id: str) -> None:
        blocks = self.blocks_by_request.pop(request_id, ())
        if blocks:
            self.coordinator.block_pool.free_blocks(blocks)

    def release_all(self) -> None:
        for request_id in tuple(self.blocks_by_request):
            self.release(request_id)


def policy_for_scheduler(scheduler) -> PrefixCheckpointPolicy | None:
    """Bind actual native group geometry once, after Scheduler.__init__.

    Called only by the active external plugin. The scheduler process does
    not require the worker runtime or access any device tensors.
    """
    config = scheduler.cache_config
    if not config.enable_prefix_caching or config.mamba_cache_mode != "align":
        return None
    coordinator = scheduler.kv_cache_manager.coordinator
    managers = coordinator.single_type_managers
    from .specs import OscarFullAttentionSpec
    from vllm.v1.kv_cache_interface import MambaSpec

    if not any(isinstance(m.kv_cache_spec, OscarFullAttentionSpec) for m in managers):
        return None
    full = []
    states = []
    for manager in managers:
        spec = manager.kv_cache_spec
        if isinstance(spec, OscarFullAttentionSpec):
            if spec.metadata_block_view or spec.block_size != spec.storage_block_size:
                raise ValueError("prefix checkpoints require physical OSCAR cache specs")
            full.append((spec.block_size, manager.use_eagle))
        elif isinstance(spec, MambaSpec) and spec.mamba_cache_mode == "align":
            states.append(spec.block_size)
        else:
            raise ValueError("unsupported OSCAR prefix checkpoint group geometry")
    if not states or any(type(b) is not int or b <= 0 for b in states):
        raise ValueError("OSCAR prefix checkpoints require positive native GDN blocks")
    if any(type(b) is not int or b <= 0 or type(drop) is not bool for b, drop in full):
        raise ValueError("invalid native FULL prefix checkpoint contract")
    alignment = lcm(*states, *(b for b, _ in full))
    if scheduler.block_size != alignment or coordinator.lcm_block_size != alignment:
        raise ValueError("OSCAR scheduler and prefix lookup alignment disagree")
    return PrefixCheckpointPolicy(alignment, lcm(*states), tuple(sorted(set(full))))


def split_prefix_checkpoint(scheduler, request, num_new_tokens: int,
                            num_new_local_computed_tokens: int = 0,
                            num_external_computed_tokens: int = 0) -> int | None:
    """Select a reachable checkpoint without increasing native token budgets.

    State-only subchunks are permitted when remaining budget cannot reach
    a joint boundary. They retain native GDN progress; the next call clips
    at the next reachable joint boundary rather than stepping over it.
    None delegates to the original native splitter. In particular, reaching
    the joint terminal must NOT bypass native tail alignment: an arbitrary
    partial state could otherwise later be hashed as a full GDN block.
    No computation, cache registration, state copy or rollback is performed
    here: all remain in the native scheduler/manager/runner.
    """
    policy = scheduler._oscar_prefix_checkpoint_policy
    computed = (request.num_computed_tokens + num_new_local_computed_tokens
                + num_external_computed_tokens)
    if num_new_tokens <= 0 or computed >= max(request.num_prompt_tokens, request.num_tokens - 1):
        return None  # Preserve the original normal-decode exclusion.
    # Local hybrid hits and normal preemption start on native state blocks.
    # Streaming/external partial states need their original state protocol:
    # rounding 2300 up to 2304 would create a new warm q4 speculative fold.
    # Encoder scheduling has already constrained indivisible input spans;
    # do not apply a second, different splitter to those spans here.
    if computed % policy.state_block_tokens or getattr(request, "has_encoder_inputs", False):
        return None
    terminal = policy.terminal_position(request.num_tokens)
    if computed >= terminal:
        return None  # Keep native tail/state alignment after the joint checkpoint.
    end = computed + num_new_tokens
    if end >= terminal:
        return terminal - computed
    aligned_end = end // policy.alignment_tokens * policy.alignment_tokens
    if aligned_end > computed:
        return aligned_end - computed
    # Do not introduce a wait for a whole compressed page. In a batch, another
    # request may leave <LCM tokens every step; native state-block progress
    # still works. Absolute endpoints include aligned local/external prefixes.
    state_end = end // policy.state_block_tokens * policy.state_block_tokens
    return max(state_end - computed, 0)
