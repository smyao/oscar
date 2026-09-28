"""Archive #148-151: one candidate preset for probes and direct serving.

Every delivered optimization flag belongs here so installation and observation
cannot silently select different numerical paths. Physical placement is separate.
"""

FEATURE_FLAGS = {
    "history_cluster4": "experimental_history_reuse",
    "later_mtp_q1": "experimental_history_reuse",
    "fast_unpack": "experimental_fast_unpack",
}


def variant_config(config: dict, variant: str | None) -> dict:
    if not isinstance(config, dict):
        raise ValueError("target config must be an object")
    if variant not in {None, "baseline", "candidate", "native"}:
        raise ValueError(f"unknown serving variant: {variant}")
    selected = dict(config)
    for flag in dict.fromkeys(FEATURE_FLAGS.values()):
        if type(selected.get(flag, False)) is not bool:
            raise ValueError(f"{flag} must be an explicit boolean")
        if variant is not None:
            selected[flag] = variant == "candidate"
    if selected.get("experimental_fast_unpack", False) and not selected.get("experimental_history_reuse", False):
        raise ValueError("fast unpack requires explicit candidate history configuration")
    return selected


def variant_features(config: dict) -> dict:
    return {feature: config.get(flag, False) for feature, flag in FEATURE_FLAGS.items()}
