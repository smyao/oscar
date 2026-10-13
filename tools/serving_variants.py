"""Archive #148-151: one candidate preset for probes and direct serving.

Every delivered optimization flag belongs here so installation and observation
cannot silently select different numerical paths. Physical placement is separate.
#34/#36: all serving now uses native token budgets (user 2026-10-09).
Historical generated whole32K/64K files are normalized without retaining an
OSCAR scheduling override. No native default is guessed here.
"""

FEATURE_FLAGS = {
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
    "decode_bundle": "experimental_decode_bundle",
    "mdl_decode": "experimental_mdl_decode",
    # 2026-10-13: current H16 -> complete H32/member pipeline, Q336/C1341,
    # 3540306 -> 2671072 CAModel ticks (1.325425x), all1008 tasks/60 actors
    # and frozen full oracle. Target NPU/graph gates remain required.
    "c4_member_pipeline": "experimental_c4_member_pipeline",
    "whole_prefill": "experimental_whole_prefill",
}

# Unfinished weighted-owner trials are not in the deploy source list or
# selector. Future research flags must not leak into native/baseline A/B.
RESEARCH_FEATURE_FLAGS = {
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
    # User 2026-10-12: native/baseline/candidate share the explicit APC default.
    selected.setdefault("enable_prefix_caching", True)
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
    # User 2026-10-09: omission is the serving policy, not an extra launch
    # option. Retire both owned scheduler variants, including reused configs.
    # The worker only observes the resulting native capacity for allocation.
    if selected.get("scheduler_cls") not in (None, WHOLE_PREFILL_SCHEDULER, BOUNDED_PREFILL_SCHEDULER):
        raise ValueError("native token budget conflicts with an explicit other scheduler")
    if "max_num_batched_tokens" in selected and (
            type(selected["max_num_batched_tokens"]) is not int or
            selected["max_num_batched_tokens"] <= 0):
        raise ValueError("configured max_num_batched_tokens must be a positive integer")
    selected["use_native_token_budget"] = True
    selected.pop("max_num_batched_tokens", None)
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
    if selected.get("experimental_mdl_decode", False) and not selected.get("experimental_decode_bundle", False):
        raise ValueError("MDL decode requires the explicit decode bundle")
    # Only this unfinished research layout lacks a qualified MDL reader;
    # default prefix caching uses the established physical-page snapshots.
    if selected.get("experimental_mdl_decode", False) and selected.get("experimental_request_windows", False):
        raise ValueError("MDL decode requires physical-page snapshots; request-window MDL reader is not qualified")
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
    return selected


def variant_features(config: dict) -> dict:
    return {feature: config.get(flag, False)
            for feature, flag in (FEATURE_FLAGS | RESEARCH_FEATURE_FLAGS).items()
            if feature in FEATURE_FLAGS or flag in config}


def runtime_feature_summary(config: dict, variant: str | None) -> dict:
    """One compact startup line shows the actual effective feature selection."""
    return {"variant":variant,"optimizations":variant_features(config),
            "prefix_caching":config.get("enable_prefix_caching", True),
            "token_budget_source":"native" if config.get("use_native_token_budget", False) else "explicit",
            "max_num_batched_tokens":config.get("max_num_batched_tokens"),
            **({"history_token_capacity":config["history_token_capacity"]}
               if config.get("history_token_capacity") is not None else {}),
            "scheduler_cls":config.get("scheduler_cls"),
            "devices":config.get("devices"),"port":config.get("port")}
