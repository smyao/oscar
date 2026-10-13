"""OSCAR APC hash dependencies and native-budget checkpoint placement.

Archive #26/#37-49/#127/#128/#142/#155/#157: immutable native-owned FULL
pages keep their exact snapshots; MTP shifted input requires a lookahead
identity, not removal of the native last-block safety condition. Native
llm_base_proposer.set_inputs_first_pass shifts IDs by one while preserving
positions. BlockPool still owns every hash, reference, eviction and reset.
No device readback, KV copy, scheduler budget increase or GDN math change.
"""
from collections.abc import Sequence


class GuardedHashes(Sequence):
    """Native hash-grain sequence whose FULL keys include one lookahead hash.

    The native pool aggregates ``ratio`` consecutive items for a FULL page.
    Attach a versioned dependency to the last item, keeping that aggregation
    and native hash ownership intact. Only groups explicitly using this
    representation see these keys; original Request hashes are never edited.
    """
    DOMAIN = b"\x00OSCAR-MTP-LOOKAHEAD-v1\x00"

    def __init__(self, hashes, ratio):
        if type(ratio) is not int or ratio <= 0:
            raise ValueError("FULL/hash ratio must be a positive integer")
        self.hashes, self.ratio = hashes, ratio

    def __len__(self):
        return max(0, (len(self.hashes) - 1) // self.ratio) * self.ratio

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        value = self.hashes[index]
        if (index + 1) % self.ratio == 0:
            value = value + self.DOMAIN + self.hashes[index + 1]
        return value


class GuardedRequest:
    def __init__(self, request, ratio):
        self.original = request
        self.block_hashes = GuardedHashes(request.block_hashes, ratio)

    def __getattr__(self, name):
        return getattr(self.original, name)


def cacheable_tokens(computed_tokens, hash_tokens):
    """Bound publication by scheduled, non-speculative lookahead hashes.

    As in native allocate_slots, this is a scheduling watermark, not device
    completion. OscarFullAttentionManager separately prevents same-step reuse.
    """
    if type(hash_tokens) is not int or hash_tokens <= 0:
        raise ValueError("native hash block size must be positive")
    return max(0, (computed_tokens // hash_tokens - 1) * hash_tokens)


def prefix_hit_bound(prompt_tokens, full_tokens, hash_tokens, *, guarded, mtp):
    limit = max(0, prompt_tokens - 1)  # native logits recomputation
    if guarded:
        limit = min(limit, cacheable_tokens(prompt_tokens, hash_tokens))
    pages = limit // full_tokens
    if mtp and not guarded:
        pages = max(0, pages - 1)
    return pages * full_tokens


def aligned_chunk_tokens(*, computed, proposed, prompt_tokens, full_tokens,
                         native_tokens, hash_tokens, guarded, mtp):
    """Place useful GDN checkpoints without increasing the proposed budget.

    Keep the largest reachable common boundary and force the final reusable
    boundary. If the budget cannot reach it yet, use a native boundary. Large
    chunks may still skip intermediate checkpoints: no uncomputed GDN state
    is invented and this does not split every physical page into a new step.
    """
    if any(type(n) is not int or n < 0 for n in (computed, proposed, prompt_tokens)):
        raise ValueError("token counts must be nonnegative integers")
    if any(type(n) is not int or n <= 0 for n in (full_tokens, native_tokens, hash_tokens)):
        raise ValueError("prefix checkpoint geometry must be positive")
    if full_tokens % native_tokens or full_tokens % hash_tokens:
        raise ValueError("FULL pages must align with native state and hash blocks")
    if proposed == 0:
        return 0
    last = prefix_hit_bound(prompt_tokens, full_tokens, hash_tokens,
                            guarded=guarded, mtp=mtp)
    end = computed + proposed
    if computed < last <= end:
        return last - computed
    if end < last:
        joint = end // full_tokens * full_tokens
        if joint > computed:
            return joint - computed
        native = end // native_tokens * native_tokens
        if native > computed:
            return native - computed
        return 0  # native scheduler defers when this step cannot reach a state boundary
    return proposed


def scheduler_prefix_split(scheduler, request, proposed, local=0, external=0):
    """Return None for native/non-APC groups; OSCAR owns only its seam."""
    if not scheduler.cache_config.enable_prefix_caching:
        return None
    from .specs import OscarFullAttentionSpec
    specs = [g.kv_cache_spec for g in scheduler.kv_cache_config.kv_cache_groups
             if isinstance(g.kv_cache_spec, OscarFullAttentionSpec)]
    if not specs:
        return None
    geometries = {(s.storage_block_size, s.native_mamba_block_size, s.mtp_prefix_guard)
                  for s in specs}
    if len(geometries) != 1:
        raise ValueError("OSCAR APC requires one FULL/state/guard geometry")
    full, native, guarded = next(iter(geometries))
    computed = request.num_computed_tokens + local + external
    if computed >= max(request.num_prompt_tokens, request.num_tokens - 1):
        return None  # leave the native decode/speculative schedule untouched
    hash_tokens = scheduler.kv_cache_manager.coordinator.hash_block_size
    result = aligned_chunk_tokens(computed=computed, proposed=proposed,
        prompt_tokens=request.num_tokens, full_tokens=full, native_tokens=native,
        hash_tokens=hash_tokens, guarded=guarded, mtp=scheduler.use_eagle)
    from ..telemetry import emit_once, emit_throttled
    emit_once("prefix_cache_contract", key=(full, native, hash_tokens, guarded),
              full_page_tokens=full, state_block_tokens=native, hash_block_tokens=hash_tokens,
              mtp_guard_hash_tokens=hash_tokens if guarded else 0,
              checkpoint_policy="joint_within_native_budget", native_ownership=True)
    emit_throttled("prefix_chunk", key="scheduler", computed_tokens=computed,
                   proposed_tokens=proposed, scheduled_tokens=result,
                   checkpoint_end=computed + result,
                   joint_checkpoint=(computed + result) % full == 0)
    return result


class _ChunkBoundaryRequest:
    def __init__(self, original, end):
        self.original, self.end = original, end

    def get_token_id(self, index):
        # This proxy exists only during padded next-token fallback preparation.
        return self.original.get_token_id(self.end if index == self.num_prompt_tokens - 1 else index)

    def __getattr__(self, name):
        return getattr(self.original, name)


def chunk_boundary_requests(runner, requests, batch):
    """Match native CPU MTP fallback using already-host scheduled endpoints.

    model_runner_v1._prepare_inputs fills optimistic_seq_lens_cpu from CPU
    computed + scheduled counts. An unfinished prefill has no speculative
    rejection, so this is its exact next prompt position. Decode and the final
    prompt chunk retain native sampled IDs/fallbacks. No NPU readback or new
    device copy is introduced; native prepare_next_token_ids_padded owns both.
    """
    spec = runner.vllm_config.speculative_config
    if spec is None or spec.method not in {"mtp", "qwen3_5_mtp"}:
        return requests
    ends = runner.optimistic_seq_lens_cpu
    if ends.device.type != "cpu" or ends.ndim != 1 or ends.numel() < batch.num_reqs:
        raise ValueError("MTP chunk boundary requires native CPU scheduled endpoints")
    result = requests
    for i, req_id in enumerate(batch.req_ids[:batch.num_reqs]):
        request = requests[req_id]
        end = int(ends[i])
        if 0 < end < request.num_prompt_tokens:
            if result is requests:
                result = dict(requests)
            result[req_id] = _ChunkBoundaryRequest(request, end)
    return result
