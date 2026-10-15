"""Archive #163: preserve the native prefill state protocol in uncached MTP.

Native GDN run_mode=0 writes conv history at 0..width-2 and final SSM at
state column 0. Folding a four-token prompt continuation into speculative
decode instead reads history at offset 3 and SSM column 3, and publishes its
final state in speculative slots. Those are different protocols in mode=none.

Keep actual prompt work on the native prefill route and prevent a shape-only
decode graph from overriding that classification. No native source, arithmetic,
MTP depth, cache layout or device guard is changed. APC-on keeps its existing
route. D.4: only existing O(batch) CPU metadata at the dispatch boundary, no
device readback or historical KV scan/materialization.
"""

from .dummy_context import is_native_dummy_run


def uncached_mtp(owner):
    config = getattr(owner, "vllm_config", None)
    cache = getattr(owner, "cache_config", None)
    if cache is None:
        cache = getattr(config, "cache_config", None)
    spec = getattr(owner, "speculative_config", None)
    if spec is None:
        spec = getattr(config, "speculative_config", None)
    return (getattr(cache, "enable_prefix_caching", None) is False
            and getattr(cache, "mamba_cache_mode", None) == "none"
            and getattr(spec, "method", None) == "mtp")


def keep_prefill_state(builder, original, common, mask, counts):
    """Do not turn real prefills into speculative state reads/writes.

    The paired dispatch hook prevents these rows from replaying a decode graph.
    Existing genuine spec rows and their native accepted counts are unchanged.
    """
    if uncached_mtp(builder):
        return mask, counts
    return original(builder, common, mask, counts)


def force_prefill_eager(runner, num_reqs, num_scheduled_tokens_np,
                       max_num_scheduled_tokens, force_uniform_decode=None):
    """Use the native eager choice only for a real warm spec-sized prefill.

    The pinned dispatcher labels every computed>0 row as decode. A qLen=4
    prompt tail then matches its uniform MTP graph, although prompt processing
    has not finished. All arrays here are the runner's existing CPU mirrors.
    Capture/dummy calls retain their explicit native graph selection.
    """
    if (not uncached_mtp(runner) or is_native_dummy_run()
            or force_uniform_decode is not None or num_reqs <= 0):
        return False
    spec = runner.speculative_config
    width = spec.num_speculative_tokens + 1
    if max_num_scheduled_tokens != width:
        return False
    batch = runner.input_batch
    scheduled = num_scheduled_tokens_np[:num_reqs]
    computed = batch.num_computed_tokens_cpu[:num_reqs]
    prompt = batch.num_prompt_tokens[:num_reqs]
    return bool(((scheduled == width) & (computed > 0) & (computed < prompt)).any())
