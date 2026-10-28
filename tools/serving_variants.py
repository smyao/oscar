"""Archive #148-151: one candidate preset for probes and direct serving.

Every delivered optimization flag belongs here so installation and observation
cannot silently select different numerical paths. Physical placement is separate.
#34/#36: aisbench-stable defaults to explicit 16384 (user 2026-10-13).
Historical generated whole32K/64K files are normalized without retaining an
OSCAR scheduling override. No native default is guessed here.
"""

FEATURE_FLAGS = {
    "official_math": "experimental_official_math",
    "c16_tuned": "experimental_c16_tuned",
    "aligned_math": "experimental_aligned_math",
    "bf16_attention": "experimental_bf16_attention",
    "history_cluster4": "experimental_history_reuse",
    "later_mtp_q1": "experimental_history_reuse",
    "fast_unpack": "experimental_fast_unpack",
    "mixed_cv": "experimental_mixed_cv",
    "striped_cache": "experimental_striped_cache",
    "quantization_bundle": "experimental_quantization_bundle",
    # 2026-10-07 user-approved hardware-validation candidate. Same-FP32
    # decoder passed complete K32 CAModel output/LSE/status and frozen oracle;
    # real NPU/graph gates remain mandatory in the observe entrypoint.
    "current_only": "experimental_current_only",
    "first_mtp_current_fia": "experimental_first_mtp_current_fia",
    "compact_later_mtp": "experimental_compact_later_mtp",
    "mixed_decode_split": "experimental_mixed_decode_split",
    # Same-step native metadata only; no new HBM, scheduler or math change.
    # Host lifetime/byte proofs are complete. Observe requires the real NPU
    # on/off metadata-chain non-regression gate before starting the service.
    "prepare_reuse": "experimental_prepare_reuse",
    "decode_bundle": "experimental_decode_bundle",
    "mdl_decode": "experimental_mdl_decode",
    "hybrid_l1_decode": "experimental_hybrid_l1_decode",
    # 2026-10-13: current H16 -> complete H32/member pipeline, Q336/C1341,
    # 3540306 -> 2671072 CAModel ticks (1.325425x), all1008 tasks/60 actors
    # and frozen full oracle. Target NPU/graph gates remain required.
    "c4_member_pipeline": "experimental_c4_member_pipeline",
    # User 2026-10-15 accepts the complete stable -> B1 improvement as the
    # next main-branch base, including its 4.1% increment beyond pipeline.
    # C65 single-KV: 3888161 -> 3172246 ticks; not a native/model speed claim.
    "c16_l1_reuse": "experimental_c16_l1_reuse",
    "whole_prefill": "experimental_whole_prefill",
}

# Unfinished weighted-owner trials are not in the deploy source list or
# selector. Future research flags must not leak into native/baseline A/B.
RESEARCH_FEATURE_FLAGS = {
    # Retain the intermediate complete reader for independent comparisons.
    "c16_member_pipeline": "experimental_c16_member_pipeline",
    # Explicit research only until local numerical/CANN qualification and
    # the new NPU gate are integrated. Both one-key entrypoints share this.
    "request_windows": "experimental_request_windows",
    "bounded_whole_prefill": "experimental_bounded_whole_prefill",
}

WHOLE_PREFILL_SCHEDULER = "oscar_ascend.integration.whole_prefill_scheduler.WholePromptAsyncScheduler"
BOUNDED_PREFILL_SCHEDULER = "oscar_ascend.integration.bounded_whole_prefill.BoundedWholePromptAsyncScheduler"
_WHOLE_SOURCE = "_oscar_whole_prefill_base"


def _restore_owned_scheduler(selected):
    """A reused generated candidate config must not contaminate native A/B."""
    saved=selected.pop(_WHOLE_SOURCE,None)
    if saved is None:return
    bounded=isinstance(saved,dict) and "history_token_capacity" in saved
    expected={"max_num_batched_tokens","scheduler_cls"} | ({"history_token_capacity"} if bounded else set())
    if (not isinstance(saved,dict) or set(saved)!=expected
            or type(saved["max_num_batched_tokens"]) is not int or saved["max_num_batched_tokens"]<=0
            or selected.get("scheduler_cls")!=(BOUNDED_PREFILL_SCHEDULER if bounded else WHOLE_PREFILL_SCHEDULER)):
        raise ValueError("invalid generated whole-prefill configuration provenance")
    selected["max_num_batched_tokens"]=saved["max_num_batched_tokens"]
    if saved["scheduler_cls"] is None:selected.pop("scheduler_cls",None)
    else:selected["scheduler_cls"]=saved["scheduler_cls"]
    if bounded:
        if saved["history_token_capacity"] is None:selected.pop("history_token_capacity",None)
        else:selected["history_token_capacity"]=saved["history_token_capacity"]


def variant_config(config: dict, variant: str | None) -> dict:
    if not isinstance(config, dict):
        raise ValueError("target config must be an object")
    if variant not in {None, "baseline", "candidate", "native"}:
        raise ValueError(f"unknown serving variant: {variant}")
    selected = dict(config)
    if selected.get("rotation_variant") not in (None, "shared_v1", "perhead_center"):
        raise ValueError("rotation_variant must be omitted, shared_v1 or perhead_center")
    if selected.get("rotation_k_clip") not in (None, "per_head_calibrated"):
        raise ValueError("rotation_k_clip must be omitted or per_head_calibrated")
    if selected.get('oscar_dequant_policy', 'native_phase') not in ('native_phase', 'fp32_metadata'):
        raise ValueError('oscar_dequant_policy must be native_phase or fp32_metadata')
    if selected.get('oscar_current_only_policy', 'rotated_v4') not in ('rotated_v4', 'native_exact'):
        raise ValueError('oscar_current_only_policy must be rotated_v4 or native_exact')
    if variant == "candidate":
        selected['oscar_dequant_policy'] = 'fp32_metadata'
        # Archive #173: the already-proven zero-history source needs no
        # lossy rotation round trip. Cached writes and all mixed readers stay
        # in their existing rotated frame; no length/dataset rule is added.
        selected['oscar_current_only_policy'] = 'native_exact'
        # Recipe4's proxy-fitted clip table regressed user model quality.
        # The fixed candidate entrypoint restores recipe3 even when its
        # input config retains the experimental clip policy. Explicit
        # research configs may still select it with variant=None.
        selected.pop("rotation_k_clip", None)
        if selected.get("rotation_method", "qqt_sst") == "qqt_sst":
            selected["rotation_variant"] = "perhead_center"
        else:
            selected.pop("rotation_variant", None)
    elif variant in {"baseline", "native"}:
        selected.pop('oscar_dequant_policy', None)
        selected.pop('oscar_current_only_policy', None)
        selected.pop("rotation_variant", None)
        selected.pop("rotation_k_clip", None)
    if selected.get("rotation_variant") == "perhead_center" and selected.get("rotation_method", "qqt_sst") != "qqt_sst":
        raise ValueError("perhead_center requires qqt_sst calibration")
    if selected.get("rotation_k_clip") == "per_head_calibrated" and selected.get("rotation_variant") != "perhead_center":
        raise ValueError("per_head_calibrated clipping requires perhead_center")
    # User 2026-10-13: this stable branch defaults to APC disabled.
    selected.setdefault("enable_prefix_caching", False)
    selected.setdefault("experimental_c4_member_pipeline", False)
    if "enable_prefix_caching" in selected and type(selected["enable_prefix_caching"]) is not bool:
        raise ValueError("enable_prefix_caching must be an explicit boolean")
    _restore_owned_scheduler(selected)
    native_budget = selected.get("use_native_token_budget", True)
    if type(native_budget) is not bool:
        raise ValueError("use_native_token_budget must be an explicit boolean")
    for flag in dict.fromkeys(FEATURE_FLAGS.values()):
        if type(selected.get(flag, False)) is not bool:
            raise ValueError(f"{flag} must be an explicit boolean")
        if variant is not None:
            # The same complete candidate bundle is used by direct serving
            # and observation. Baseline/native clear every experimental flag.
            selected[flag] = variant == "candidate"
    for flag in RESEARCH_FEATURE_FLAGS.values():
        if type(selected.get(flag, False)) is not bool:
            raise ValueError(f"{flag} must be an explicit boolean")
        if variant in {"baseline", "native"} and flag in selected:
            selected[flag] = False
    # Retire owned scheduler variants; stable keeps an explicit token budget
    # while an explicit native-budget request still omits that argument.
    # The worker only observes the resulting native capacity for allocation.
    if selected.get("scheduler_cls") not in (None, WHOLE_PREFILL_SCHEDULER, BOUNDED_PREFILL_SCHEDULER):
        raise ValueError("native token budget conflicts with an explicit other scheduler")
    if "max_num_batched_tokens" in selected and (
            type(selected["max_num_batched_tokens"]) is not int or
            selected["max_num_batched_tokens"] <= 0):
        raise ValueError("configured max_num_batched_tokens must be a positive integer")
    # User 2026-10-13: aisbench-stable explicitly retains its 16384 budget.
    # Native-budget launches still omit the flag; scheduler remains native.
    selected["use_native_token_budget"] = native_budget
    if native_budget:
        selected.pop("max_num_batched_tokens", None)
    elif "max_num_batched_tokens" not in selected:
        raise ValueError("explicit token budget requires max_num_batched_tokens")
    selected.pop("scheduler_cls", None)
    selected.pop("history_token_capacity", None)
    selected["experimental_whole_prefill"] = False
    if "experimental_bounded_whole_prefill" in selected:
        selected["experimental_bounded_whole_prefill"] = False
    if selected.get("experimental_fast_unpack", False) and not selected.get("experimental_history_reuse", False):
        raise ValueError("fast unpack requires explicit candidate history configuration")
    if selected.get("experimental_mixed_cv", False) and not (
            selected.get("experimental_history_reuse", False) and
            selected.get("experimental_fast_unpack", False)):
        raise ValueError("mixed CV requires explicit fast/history candidate configuration")
    if selected.get("experimental_striped_cache", False) and not (
            selected.get("experimental_history_reuse", False) and
            selected.get("experimental_fast_unpack", False) and
            selected.get("experimental_mixed_cv", False)):
        raise ValueError("striped cache requires explicit mixed/fast/history candidate configuration")
    if selected.get("experimental_decode_bundle", False) and not selected.get("experimental_striped_cache", False):
        raise ValueError("decode bundle requires the explicit striped cache bundle")
    if selected.get("experimental_aligned_math", False) and not selected.get("experimental_bf16_attention", False):
        raise ValueError("aligned math requires BF16 attention")
    if selected.get("experimental_official_math", False) and not selected.get("experimental_aligned_math", False):
        raise ValueError("official math requires the aligned BF16 codec")
    if selected.get("experimental_c16_tuned", False) and not selected.get("experimental_aligned_math", False):
        raise ValueError("timer-directed C16 requires aligned BF16 attention")
    if selected.get("experimental_bf16_attention", False) and not (
            selected.get("experimental_striped_cache", False) and
            selected.get("experimental_decode_bundle", False) and
            selected.get("experimental_c4_member_pipeline", False) and
            selected.get("experimental_c16_l1_reuse", False)):
        raise ValueError("BF16 attention requires the complete physical-page candidate bundle")
    if selected.get("experimental_bf16_attention", False) and selected.get("experimental_request_windows", False):
        raise ValueError("BF16 attention requires physical-page storage")
    if selected.get("experimental_mdl_decode", False) and not selected.get("experimental_decode_bundle", False):
        raise ValueError("MDL decode requires the explicit decode bundle")
    if selected.get("experimental_hybrid_l1_decode", False) and not selected.get("experimental_mdl_decode", False):
        raise ValueError("hybrid L1 decode requires the explicit MDL decode bundle")
    if selected.get("experimental_c16_member_pipeline", False) and not (
            selected.get("experimental_decode_bundle", False) and
            selected.get("experimental_mixed_cv", False)):
        raise ValueError("C16 member pipeline requires the mixed striped decode bundle")
    if selected.get("experimental_c16_l1_reuse", False) and not (
            selected.get("experimental_decode_bundle", False) and
            selected.get("experimental_mixed_cv", False)):
        raise ValueError("C16 L1 reuse requires the mixed striped decode bundle")
    # Only this unfinished research layout lacks a qualified MDL reader;
    # default prefix caching uses the established physical-page snapshots.
    if selected.get("experimental_mdl_decode", False) and selected.get("experimental_request_windows", False):
        raise ValueError("MDL decode requires physical-page snapshots; request-window MDL reader is not qualified")
    if selected.get("experimental_c16_member_pipeline", False) and selected.get("experimental_request_windows", False):
        raise ValueError("C16 member pipeline requires physical-page snapshots")
    if selected.get("experimental_c16_l1_reuse", False) and selected.get("experimental_request_windows", False):
        raise ValueError("C16 L1 reuse requires physical-page snapshots")
    if selected.get("experimental_c4_member_pipeline", False) and not selected.get("experimental_decode_bundle", False):
        raise ValueError("C4 member pipeline requires the explicit decode bundle")
    if selected.get("experimental_mixed_decode_split", False) and not selected.get("experimental_decode_bundle", False):
        raise ValueError("mixed decode partition requires the optimized decoder bundle")
    if selected.get("experimental_mixed_decode_split", False) and not selected.get("experimental_first_mtp_current_fia", False):
        raise ValueError("mixed decode partition requires the qualified first-MTP current path")
    if selected.get("experimental_request_windows",False) and not (
            selected.get("experimental_striped_cache",False) and
            selected.get("experimental_decode_bundle",False)):
        raise ValueError("request windows require the striped decode bundle")
    if selected.get('oscar_dequant_policy') == 'fp32_metadata' and not all(selected.get(flag, False)
            for flag in ('experimental_official_math', 'experimental_aligned_math', 'experimental_bf16_attention')):
        raise ValueError('FP32 metadata reconstruction requires the official BF16 reader bundle')
    if selected.get('oscar_current_only_policy') == 'native_exact' and not (
            selected.get('experimental_official_math', False) and
            selected.get('experimental_current_only', False)):
        raise ValueError('native-exact current requires the official current-only route')
    return selected


def variant_features(config: dict) -> dict:
    return {feature: config.get(flag, False)
            for feature, flag in (FEATURE_FLAGS | RESEARCH_FEATURE_FLAGS).items()
            if feature in FEATURE_FLAGS or flag in config}


def runtime_feature_summary(config: dict, variant: str | None) -> dict:
    """One compact startup line shows the actual effective feature selection."""
    return {"variant":variant,"optimizations":variant_features(config),
            **({"aligned_c16_operator": ("attention_cv_official_cluster16_out" if config.get("experimental_official_math",False) else "attention_cv_aligned_cluster16_out"
                                        if config.get("experimental_c16_tuned", False)
                                        else "attention_cv_aligned_cluster16_control_out")}
               if config.get("experimental_aligned_math", False) else {}),
            "attention_compute":{"operands":"bfloat16" if config.get("experimental_bf16_attention", False) else "float32",
                                 "accumulator":"float32", "softmax_lse_merge":"float32"},
            "oscar_math_profile":"oscar_bf16_attention_v4" if config.get("experimental_official_math",False) else "oscar_bf16_attention_v3" if config.get("experimental_aligned_math",False) else "legacy",
            "attention_coordinate_frame":("rotated_cached_and_mixed" if config.get('oscar_current_only_policy') == 'native_exact'
                                          else "all_sources_rotated" if config.get("experimental_official_math",False) else "legacy_mixed"),
            "current_only_policy":config.get('oscar_current_only_policy', 'rotated_v4'),
            "current_only_coordinate_frame":('original_native' if config.get('oscar_current_only_policy') == 'native_exact'
                                             else 'rotated_v4' if config.get('experimental_official_math',False) else 'original_native'),
            "current_only_inverse_rotation":(config.get("experimental_official_math",False)
                                              and config.get('oscar_current_only_policy', 'rotated_v4') != 'native_exact'),
            "inverse_rotation":"once_after_fp32_merge" if config.get("experimental_official_math",False) else "per_history_partial",
            "cached_precision_policy":"native_fixed_per_forward" if config.get("experimental_official_math",False) else "legacy_query_relative",
            "rotation_method":config.get("rotation_method"),
            "rotation_variant":"hadamard_data_free" if config.get("rotation_method")=="hadamard" else config.get("rotation_variant", "shared_v1"),
            "rotation_grouping":"layer_kv_head" if config.get("rotation_variant")=="perhead_center" else "layer",
            "key_centering":config.get("rotation_variant")=="perhead_center",
            "clip_policy":config.get("rotation_k_clip", "scalar"),
            "head_clip_experiment_enabled":config.get("rotation_k_clip")=="per_head_calibrated",
            "calibration_recipe_version":None if config.get("rotation_method")=="hadamard" else
                (4 if config.get("rotation_k_clip")=="per_head_calibrated" else 3)
                if config.get("rotation_variant")=="perhead_center" else 2,
            "metadata_dtype":"float32" if config.get("experimental_aligned_math",False) else "float16",
            "dequant_policy":config.get('oscar_dequant_policy', 'native_phase'),
            "runtime_precision_contract":('oscar_bf16_runtime_fp32_dequant_v1'
                if config.get('oscar_dequant_policy') == 'fp32_metadata' else 'native_phase'),
            "dequant_arithmetic":('float32_then_bfloat16' if config.get('oscar_dequant_policy') == 'fp32_metadata'
                                  else 'prefill_float32_decode_bfloat16' if config.get('experimental_aligned_math', False)
                                  else 'legacy_reader_defined'),
            "prefix_caching":config.get("enable_prefix_caching", False),
            "token_budget_source":"native" if config.get("use_native_token_budget", False) else "explicit",
            "max_num_batched_tokens":config.get("max_num_batched_tokens"),
            **({"history_token_capacity":config["history_token_capacity"]}
               if config.get("history_token_capacity") is not None else {}),
            "scheduler_cls":config.get("scheduler_cls"),
            "devices":config.get("devices"),"port":config.get("port")}
