# Archive #140-151: production has one signed numerical path; legacy shape
# switches are rejected instead of silently selecting a different operator.
"""Canonical production configuration with no numerical-path switches."""

REMOVED_FLAGS = frozenset({
    "experimental_history_reuse", "experimental_fast_unpack",
    "experimental_weighted_q4", "experimental_weighted_q4_split2",
})


def variant_config(config: dict, variant: str | None) -> dict:
    if not isinstance(config, dict):
        raise ValueError("target config must be an object")
    if variant not in {None, "candidate"}:
        raise ValueError(f"unknown serving variant: {variant}")
    selected = dict(config)
    present = sorted(REMOVED_FLAGS.intersection(selected))
    if present:
        raise ValueError("removed production switches are not accepted: " + ", ".join(present))
    return selected


def variant_features(config: dict) -> dict:
    if not isinstance(config, dict):
        raise ValueError("target config must be an object")
    return {"unified_int2_cv": True}
