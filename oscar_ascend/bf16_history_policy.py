"""Archive #148/#153/#154/#157: explicit precision, layout and serving contracts.

User 2026-10-16 selects BF16 QK/PV operands for every OSCAR attention source.
The historical ABI/boolean name is retained for compatibility; FP32 rotation,
softmax, accumulation and merge remain separate from matrix operand precision.
Configuration validity is not numerical, device, graph or model acceptance.
"""

BF16_HISTORY_FLAG = "experimental_bf16_history"
ATTENTION_COMPUTE_DTYPE = "attention_compute_dtype"
BF16_HISTORY_DEPENDENCIES = (
    "experimental_history_reuse", "experimental_fast_unpack",
    "experimental_mixed_cv", "experimental_striped_cache",
    "experimental_decode_bundle", "experimental_mdl_decode",
    "experimental_c4_member_pipeline",
)


def normalize_bf16_history(config, *, variant=None) -> bool:
    """Strict compatibility alias; pure diagnostic presets may clear it."""
    enabled = config.get(BF16_HISTORY_FLAG, False)
    if type(enabled) is not bool:
        raise ValueError(f"{BF16_HISTORY_FLAG} must be an explicit boolean")
    dtype = config.get(ATTENTION_COMPUTE_DTYPE, "bfloat16")
    if type(dtype) is not str or dtype != "bfloat16":
        raise ValueError(f"{ATTENTION_COMPUTE_DTYPE} must be 'bfloat16'; FP32 production attention is retired")
    if variant in {"baseline", "native"}:
        return False
    return enabled


def production_attention_config(config) -> dict:
    """Supply only missing defaults; explicit incompatible requests fail."""
    selected = dict(config)
    normalize_bf16_history(selected)
    if selected.get(BF16_HISTORY_FLAG, True) is not True:
        raise ValueError(f"{BF16_HISTORY_FLAG}=false cannot select retired FP32 production attention; use native for the no-OSCAR baseline")
    selected[BF16_HISTORY_FLAG] = True
    selected[ATTENTION_COMPUTE_DTYPE] = "bfloat16"
    for flag in BF16_HISTORY_DEPENDENCIES:
        selected.setdefault(flag, True)
    validate_bf16_history(selected)
    return selected


def validate_bf16_history(config, *, head_dim=None, cache_format=None,
                          production=False) -> bool:
    """Validate all-source BF16 layout; False is diagnostic-only."""
    enabled = normalize_bf16_history(config)
    if not enabled:
        if production:
            raise ValueError(f"{BF16_HISTORY_FLAG}=false cannot select retired FP32 production attention")
        return False
    for flag in BF16_HISTORY_DEPENDENCIES:
        if type(config.get(flag, False)) is not bool:
            raise ValueError(f"{flag} must be an explicit boolean")
        if not config.get(flag, False):
            raise ValueError(f"BF16 attention requires {flag}=true")
    request_windows = config.get("experimental_request_windows", False)
    if type(request_windows) is not bool:
        raise ValueError("experimental_request_windows must be an explicit boolean")
    if request_windows:
        raise ValueError("BF16 attention requires physical-page snapshots; request windows are unsupported")
    if head_dim is not None and (type(head_dim) is not int or head_dim != 256):
        raise ValueError("BF16 attention requires head_dim=256")
    if cache_format is not None and cache_format != "striped_v1":
        raise ValueError("BF16 attention requires striped_v1 physical-page snapshots")
    return True


def require_bf16_history_production(config) -> None:
    """Require the sole production precision/layout; never grant acceptance."""
    production_attention_config(config)
