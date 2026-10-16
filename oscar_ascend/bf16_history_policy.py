"""Archive #148/#153: separate research shape validity from serving acceptance.

The BF16 history experiment has not passed the frozen CPU numerical gate.
Research operator availability must never be mistaken for production acceptance.
This module is deliberately independent of torch, device loading and probes.
"""

BF16_HISTORY_FLAG = "experimental_bf16_history"
BF16_HISTORY_DEPENDENCIES = (
    "experimental_history_reuse", "experimental_fast_unpack",
    "experimental_mixed_cv", "experimental_striped_cache",
    "experimental_decode_bundle", "experimental_mdl_decode",
    "experimental_c4_member_pipeline",
)


def normalize_bf16_history(config, *, variant=None) -> bool:
    """Read a strict boolean, clearing it only for explicit baseline/native.

    Explicit baseline/native selection clears a valid boolean request, just as
    the existing experiment presets do. Type validation precedes that clearing.
    Missing/False leaves all established serving configuration untouched.
    """
    enabled = config.get(BF16_HISTORY_FLAG, False)
    if type(enabled) is not bool:
        raise ValueError(f"{BF16_HISTORY_FLAG} must be an explicit boolean")
    if variant in {"baseline", "native"}:
        return False
    return enabled


def validate_bf16_history(config, *, head_dim=None, cache_format=None) -> bool:
    """Validate research configuration/geometry without granting acceptance."""
    enabled = normalize_bf16_history(config)
    if not enabled:
        return False
    for flag in BF16_HISTORY_DEPENDENCIES:
        if type(config.get(flag, False)) is not bool:
            raise ValueError(f"{flag} must be an explicit boolean")
        if not config.get(flag, False):
            raise ValueError(f"{BF16_HISTORY_FLAG} requires {flag}=true")
    request_windows = config.get("experimental_request_windows", False)
    if type(request_windows) is not bool:
        raise ValueError("experimental_request_windows must be an explicit boolean")
    if request_windows:
        raise ValueError(f"{BF16_HISTORY_FLAG} requires physical-page snapshots; request windows are unsupported")
    if head_dim is not None and (type(head_dim) is not int or head_dim != 256):
        raise ValueError(f"{BF16_HISTORY_FLAG} requires head_dim=256")
    if cache_format is not None and cache_format != "striped_v1":
        raise ValueError(f"{BF16_HISTORY_FLAG} requires striped_v1 physical-page snapshots")
    return True


def require_bf16_history_production(config) -> None:
    """Reject known-unqualified math at every service entry; no bypass token."""
    if validate_bf16_history(config):
        raise ValueError(
            f"{BF16_HISTORY_FLAG} is unavailable: the BF16 history candidate "
            "has not passed the frozen CPU numerical gate; its research operators "
            "are not qualified for production; keep this flag false")
