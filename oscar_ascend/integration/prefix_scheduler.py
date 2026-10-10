"""Archive #127/#128/#156: align GDN checkpoints with physical FULL hits.

External scheduler seam only. Native GDN specs, state copies, cache ownership
and decode scheduling stay unchanged. Mirrors the pinned scheduler prefill
split, using the native coordinator's common alignment and cache-hit ceiling.
"""
from dataclasses import dataclass

from .prefix_geometry import resolve_prefix_geometry
from .runtime_api import OscarReadinessError


@dataclass(frozen=True)
class PrefixCheckpointPolicy:
    alignment: int
    use_eagle: bool

    def split(self, request, num_new_tokens, num_new_local_computed_tokens=0,
              num_external_computed_tokens=0):
        computed = (request.num_computed_tokens + num_new_local_computed_tokens
                    + num_external_computed_tokens)
        # Preserve the native decode bypass, including accepted MTP tokens.
        if computed >= max(request.num_prompt_tokens, request.num_tokens - 1):
            return num_new_tokens
        # KVCacheManager.get_computed_blocks leaves the last token for logits;
        # FullAttentionManager also drops a matched page for EAGLE/MTP. A GDN
        # checkpoint beyond that ceiling cannot be reused by this same prompt.
        last = max(request.num_tokens - 1, 0) // self.alignment * self.alignment
        if self.use_eagle:
            last = max(last - self.alignment, 0)
        after = computed + num_new_tokens
        if after < last:
            # Align the resulting position, also for an external/resumed offset.
            return max(0, after // self.alignment * self.alignment - computed)
        if computed < last < after:
            return last - computed
        return num_new_tokens


def configure_prefix_scheduler(scheduler):
    """Resolve once from actual native cache groups, never from a preset size."""
    cache = scheduler.cache_config
    if not cache.enable_prefix_caching:
        return None
    from .specs import OscarFullAttentionSpec
    from vllm.v1.kv_cache_interface import MambaSpec

    coordinator = scheduler.kv_cache_manager.coordinator
    specs = [group.kv_cache_spec for group in coordinator.kv_cache_config.kv_cache_groups]
    physical_sizes = {spec.block_size for spec in specs if isinstance(spec, OscarFullAttentionSpec)}
    if not physical_sizes:
        return None  # Native-only groups do not acquire OSCAR scheduling rules.
    if len(physical_sizes) != 1:
        raise OscarReadinessError("OSCAR prefix scheduler requires one physical FULL block size")
    layout = resolve_prefix_geometry(cache, physical_sizes.pop())
    alignment = layout["scheduler_lcm_block_size"]
    gdn = [spec for spec in specs if isinstance(spec, MambaSpec)]
    if (not gdn or any(spec.block_size != layout["mamba_block_size"]
                       or spec.mamba_cache_mode != "align" for spec in gdn)
            or not scheduler.need_mamba_block_aligned_split
            or scheduler.block_size != alignment
            or coordinator.lcm_block_size != alignment
            or coordinator.hash_block_size != layout["hash_block_size"]):
        raise OscarReadinessError("OSCAR prefix scheduler/native GDN geometry disagrees")
    budget = scheduler.max_num_scheduled_tokens
    threshold = scheduler.scheduler_config.long_prefill_token_threshold
    if (type(budget) is not int or budget < alignment
            or type(threshold) is not int or threshold < 0
            or 0 < threshold < alignment):
        raise OscarReadinessError(
            f"OSCAR prefix checkpoints require max_num_scheduled_tokens "
            f"(normally max_num_batched_tokens) >= {alignment}; "
            f"long_prefill_token_threshold must be 0 or >= {alignment}, "
            f"got budget={budget}, threshold={threshold}")
    from ..telemetry import emit_once
    emit_once("prefix_checkpoint_policy", key=(alignment, budget, scheduler.use_eagle),
              alignment_tokens=alignment, native_mamba_block_size=cache.mamba_block_size,
              max_num_scheduled_tokens=budget, use_eagle=scheduler.use_eagle)
    return PrefixCheckpointPolicy(alignment, scheduler.use_eagle)
