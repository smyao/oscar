"""Archive #34/#36/#140/#142/#148/#155: unenabled later-MTP row experiment.

Native proposer 1249-1332 updates/consumes only B leading rows in later steps.
The public compile decorator honors forward_context.skip_compiled (509-513).
Keep original N for CV splitting, preserve invalid tail-slot errors on stream,
and restore the native context on every exit. No GDN or native-source edits.
D.4: remove padded dense/communication work, never restore INT2 history.
Padding-store savings overlap active-slot store and must not be counted twice.
"""
from __future__ import annotations

from dataclasses import replace

from .dummy_context import is_native_dummy_run
from .first_draft_current_only import supports_first_draft_current_only
from .metadata import OscarMetadata
from .runtime_api import OscarReadinessError

_NATIVE_FALSE_FLAGS = (
    "enable_sp", "enable_sp_by_pass", "flashcomm2_enable", "shared_expert_dp_enabled",
    "lmhead_tp_enable", "embedding_tp_enable", "mlp_tp_enable", "oproj_tp_enable",
    "olora_tp_enable", "matmul_allreduce_enable", "o_shard_enable")


def should_compact_rows(tokens, requests):
    """Unmeasured conservative policy: avoid changing the short decode path."""
    return (type(tokens) is int and type(requests) is int and 0 < requests <= tokens
            and tokens >= 1024 and tokens >= 8 * requests)


def _fail(message):
    raise OscarReadinessError("compact later MTP: " + message)


def _native_facts():
    from vllm.distributed import get_dp_group, get_pp_group, get_tp_group
    from vllm_ascend import utils
    flags = {name: getattr(utils, name)() for name in _NATIVE_FALSE_FLAGS}
    return flags, (get_dp_group().world_size, get_pp_group().world_size, get_tp_group().world_size)


def _validate_model(model, context, flags, groups):
    config = model.vllm_config
    if not supports_first_draft_current_only(config):
        _fail("requires ordinary eager padded nonparallel Qwen3.5 MTP with PCP/DCP=1")
    parallel = config.parallel_config
    for name in ("data_parallel_size", "pipeline_parallel_size"):
        if type(getattr(parallel, name, None)) is not int or getattr(parallel, name) != 1:
            _fail("requires DP=PP=1")
    tp = getattr(parallel, "tensor_parallel_size", None)
    if type(tp) is not int or tp <= 0 or groups != (1, 1, tp):
        _fail("native DP/PP/TP groups differ from the model configuration")
    if set(flags) != set(_NATIVE_FALSE_FLAGS) or any(value is not False for value in flags.values()):
        _fail("SP, flashcomm, shared-expert DP and special TP paths are outside this experiment")
    if (getattr(context, "flash_comm_v1_enabled", None) is not False or
            getattr(context, "flashcomm_v2_enabled", None) is not False or
            getattr(context, "pad_size", None) != 0):
        _fail("native forward context has communication padding or flashcomm")
    if getattr(config, "lora_config", None) is not None:
        _fail("LoRA mappings have not been audited for compact inputs")
    predictor = getattr(model, "model", None)
    if (type(model).__name__ != "Qwen3_5MTP" or
            type(predictor).__name__ != "Qwen3_5MultiTokenPredictor" or
            getattr(predictor.config, "model_type", None) != "qwen3_5_text" or
            len(predictor.layers) != 1 or predictor.layers[0].layer_type != "full_attention"):
        _fail("requires the actual dense one-FULL-layer Qwen3.5 predictor")
    # Both decorated instances must expose the pinned public bypass contract.
    for instance in (model, predictor):
        if type(getattr(instance, "do_not_compile", None)) is not bool:
            _fail("model lacks the supported compile-decorator instance contract")
    if type(getattr(context, "skip_compiled", None)) is not bool:
        _fail("forward context lacks the supported skip_compiled flag")
    layer = predictor.layers[0]
    attention = layer.self_attn.attn
    if (type(attention.impl).__name__ != "OscarAttentionImpl" or
            getattr(attention, "calculate_kv_scales", None) is not False or
            getattr(attention, "query_quant", None) is not None):
        _fail("requires OSCAR BF16 FULL attention without token-batch scale calibration")
    for linear in (predictor.fc, layer.self_attn.qkv_proj, layer.self_attn.o_proj,
                   layer.mlp.gate_up_proj, layer.mlp.down_proj):
        if getattr(linear, "custom_op", None) is not None:
            _fail("custom linear communication is outside the audited ordinary TP path")
        method = linear.quant_method
        scheme = getattr(method, "quant_method", method)
        if type(scheme).__name__ not in {
                "AscendW8A8LinearMethod", "AscendW8A8DynamicLinearMethod", "AscendUnquantizedLinearMethod"}:
            _fail("linear quantization scheme lacks a verified per-row/fixed-scale contract")


def _tail_slot_guard(provider, slots, begin, end):
    import torch
    if slots.device.type != "npu" or slots.dtype not in (torch.int32, torch.int64):
        _fail("tail-slot guard requires the real NPU slot buffer")
    bad = _tail_slot_errors(slots, begin, end)
    empty = bad[:0]
    provider.ops.status_guard(bad, empty, empty, empty)


def _tail_slot_errors(slots, begin, end):
    """Native 1661 fills tail with -1; preserve violations before slicing."""
    import torch
    return (slots[begin:end] >= 0).to(torch.int32)


def _compact_call(original, model, args, kwargs, provider, context, flags, groups):
    metadata = context.attn_metadata
    if (is_native_dummy_run() or getattr(context, "in_profile_run", False) or
            getattr(context, "capturing", False) or not isinstance(metadata, dict) or not metadata):
        return original(model, *args, **kwargs)
    rows = tuple(metadata.values())
    if any(not isinstance(row, OscarMetadata) for row in rows):
        _fail("expected only OSCAR FULL metadata in the draft model")
    if any(row.dummy_origin or row.capture_origin or not row.is_draft or row.draft_index == 0 for row in rows):
        return original(model, *args, **kwargs)
    ids = kwargs.get("input_ids")
    if ids is not None and not should_compact_rows(ids.shape[0], rows[0].num_actual_tokens):
        return original(model, *args, **kwargs)
    if (getattr(context, "is_draft_model", None) is not True or
            getattr(getattr(context, "cudagraph_runtime_mode", None), "name", None) != "NONE"):
        _fail("later compact execution requires the explicit eager draft context")
    _validate_model(model, context, flags, groups)
    import torch
    if torch.compiler.is_compiling():
        _fail("cannot compact inputs while tracing a compiled model")
    if args or set(kwargs) - {"input_ids", "positions", "hidden_states", "inputs_embeds"}:
        _fail("only the pinned native keyword-only model invocation is supported")
    if not all(name in kwargs for name in ("input_ids", "positions", "hidden_states")):
        _fail("native draft model inputs are incomplete")
    ids, positions, hidden = (kwargs[name] for name in ("input_ids", "positions", "hidden_states"))
    n, b = ids.shape[0], rows[0].num_actual_tokens
    if (ids.ndim != 1 or type(b) is not int or not 0 < b <= n or context.num_accept_tokens != b or context.num_tokens != n
            or hidden.ndim != 2 or hidden.shape[0] != n or positions.ndim not in (1, 2)
            or positions.shape[-1] != n or (positions.ndim == 2 and positions.shape[0] != 3)):
        _fail("N/B, hidden or 1D/mRoPE positions do not match the native later-step contract")
    if b == n:
        return original(model, *args, **kwargs)
    small = {}
    for name, row in metadata.items():
        if (row.num_actual_tokens != b or row.num_reqs != b or row.max_query_len != 1
                or row.num_input_tokens != n or row.draft_index != rows[0].draft_index
                or row.query_start_loc.shape[0] != b + 1 or row.slot_mapping.shape[0] < n
                or row.cv_shape_tokens is not None):
            _fail("later metadata is not B one-query requests with N input rows")
        _tail_slot_guard(provider, row.slot_mapping, b, n)
        small[name] = replace(row, num_input_tokens=b, slot_mapping=row.slot_mapping[:b],
            seq_lens=row.seq_lens[:b], block_tables=row.block_tables[:b], cv_shape_tokens=n,
            current_only_plan=None, first_draft_current_only=False)
    call = {**kwargs, "input_ids": ids[:b], "hidden_states": hidden[:b], "positions": positions[..., :b]}
    embeds = kwargs.get("inputs_embeds")
    if embeds is not None:
        if embeds.ndim != 2 or embeds.shape[0] != n:
            _fail("embedding rows do not match the native later-step buffer")
        call["inputs_embeds"] = embeds[:b]
    saved = {name: getattr(context, name) for name in ("attn_metadata", "num_tokens", "skip_compiled")}
    try:
        context.attn_metadata, context.num_tokens, context.skip_compiled = small, b, True
        result = original(model, **call)
        if result.ndim != 2 or result.shape != (b, hidden.shape[1]):
            _fail("dense Qwen3.5 MTP returned an unexpected hidden-state shape")
        # Native later consumers are arange(B) and hidden[:B] (1299/1332).
        # No re-expansion to N is required without SP/DP/graph gather paths.
        return result
    finally:
        for name, value in saved.items():
            setattr(context, name, value)


def compact_later_mtp_call(original, model, args, kwargs, provider):
    from vllm.forward_context import get_forward_context, is_forward_context_available
    if not is_forward_context_available():
        return original(model, *args, **kwargs)
    context = get_forward_context()
    # First/dummy calls do not need native communication-group initialization.
    metadata = getattr(context, "attn_metadata", None)
    if (is_native_dummy_run() or getattr(context, "in_profile_run", False) or
            getattr(context, "capturing", False) or not isinstance(metadata, dict) or not metadata or not all(
            isinstance(row, OscarMetadata) and row.is_draft and row.draft_index > 0
            and not row.dummy_origin and not row.capture_origin for row in metadata.values())):
        return original(model, *args, **kwargs)
    ids = kwargs.get("input_ids")
    if ids is not None and not should_compact_rows(ids.shape[0], next(iter(metadata.values())).num_actual_tokens):
        return original(model, *args, **kwargs)
    flags, groups = _native_facts()
    return _compact_call(original, model, args, kwargs, provider, context, flags, groups)
