"""Archive #148-151: one candidate preset for probes and direct serving.

Every delivered optimization flag belongs here so installation and observation
cannot silently select different numerical paths. Physical placement is separate.
"""

FEATURE_FLAGS = {
    "history_cluster4": "experimental_history_reuse",
    "later_mtp_q1": "experimental_history_reuse",
    "fast_unpack": "experimental_fast_unpack",
    "mixed_cv": "experimental_mixed_cv",
    "striped_cache": "experimental_striped_cache",
    # 2026-10-07 user-approved hardware-validation candidate. Same-FP32
    # decoder passed complete K32 CAModel output/LSE/status and frozen oracle;
    # real NPU/graph gates remain mandatory in the observe entrypoint.
    "current_only": "experimental_current_only",
    "first_mtp_current_fia": "experimental_first_mtp_current_fia",
    "compact_later_mtp": "experimental_compact_later_mtp",
    "mixed_decode_split": "experimental_mixed_decode_split",
    "decode_bundle": "experimental_decode_bundle",
    "whole_prefill": "experimental_whole_prefill",
}

# Unfinished weighted-owner trials are not in the deploy source list or
# selector. Future research flags must not leak into native/baseline A/B.
RESEARCH_FEATURE_FLAGS = {}

WHOLE_PREFILL_SCHEDULER = "oscar_ascend.integration.whole_prefill_scheduler.WholePromptAsyncScheduler"
_WHOLE_SOURCE = "_oscar_whole_prefill_base"


def _restore_owned_scheduler(selected):
    """A reused generated candidate config must not contaminate native A/B."""
    saved=selected.pop(_WHOLE_SOURCE,None)
    if saved is None:return
    if (not isinstance(saved,dict) or set(saved)!={"max_num_batched_tokens","scheduler_cls"}
            or type(saved["max_num_batched_tokens"]) is not int or saved["max_num_batched_tokens"]<=0
            or selected.get("scheduler_cls")!=WHOLE_PREFILL_SCHEDULER):
        raise ValueError("invalid generated whole-prefill configuration provenance")
    selected["max_num_batched_tokens"]=saved["max_num_batched_tokens"]
    if saved["scheduler_cls"] is None:selected.pop("scheduler_cls",None)
    else:selected["scheduler_cls"]=saved["scheduler_cls"]


def variant_config(config: dict, variant: str | None) -> dict:
    if not isinstance(config, dict):
        raise ValueError("target config must be an object")
    if variant not in {None, "baseline", "candidate", "native"}:
        raise ValueError(f"unknown serving variant: {variant}")
    selected = dict(config)
    _restore_owned_scheduler(selected)
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
    if selected.get("experimental_mixed_decode_split", False) and not selected.get("experimental_decode_bundle", False):
        raise ValueError("mixed decode partition requires the optimized decoder bundle")
    if selected.get("experimental_mixed_decode_split", False) and not selected.get("experimental_first_mtp_current_fia", False):
        raise ValueError("mixed decode partition requires the qualified first-MTP current path")
    if selected.get("experimental_whole_prefill", False):
        if not (selected.get("experimental_current_only",False) and
                selected.get("experimental_first_mtp_current_fia",False)):
            raise ValueError("whole-prefill requires current-only and first-MTP current FIA")
        budget=selected.get("max_num_batched_tokens")
        if type(budget) is not int or budget<=0:
            raise ValueError("whole-prefill requires an explicit positive token budget")
        scheduler=selected.get("scheduler_cls")
        if scheduler is not None and scheduler!=WHOLE_PREFILL_SCHEDULER:
            raise ValueError("whole-prefill conflicts with an explicit other scheduler")
        selected[_WHOLE_SOURCE]={"max_num_batched_tokens":budget,"scheduler_cls":None}
        selected["max_num_batched_tokens"]=max(32768,budget)
        selected["scheduler_cls"]=WHOLE_PREFILL_SCHEDULER
    return selected


def variant_features(config: dict) -> dict:
    return {feature: config.get(flag, False)
            for feature, flag in (FEATURE_FLAGS | RESEARCH_FEATURE_FLAGS).items()}


def runtime_feature_summary(config: dict, variant: str | None) -> dict:
    """One compact startup line shows the actual effective feature selection."""
    return {"variant":variant,"optimizations":variant_features(config),
            "max_num_batched_tokens":config.get("max_num_batched_tokens"),
            "enable_prefix_caching":config.get("enable_prefix_caching", False),
            "scheduler_cls":config.get("scheduler_cls"),
            "devices":config.get("devices"),"port":config.get("port")}
