"""Archive #28/#77/#86/#97/#127: bind rotations to the actual native engine.

The worker's model_config is available before load_model constructs Attention.
Only host config metadata is read here; no native source scan, weights, tensors,
device synchronization or calibration runs are added to production.
"""
from pathlib import Path

from .runtime_api import OscarReadinessError


def _dtype(value):
    return str(value).removeprefix("torch.")


def validate_runner_model_identity(config, runner):
    """Reject manual CLI/config drift before any model/rotation allocation."""
    from tools.model_config import apply_model_options, resolve_mamba_cache_dtypes

    try:
        selected = apply_model_options(config, require_model=True)
    except (OSError, ValueError, TypeError) as error:
        raise OscarReadinessError(f"OSCAR target model configuration is invalid: {error}") from error
    native = getattr(runner, "vllm_config", None)
    model = getattr(runner, "model_config", None)
    if model is None:
        model = getattr(native, "model_config", None)
    cache = getattr(native, "cache_config", None)
    if model is None or cache is None:
        raise OscarReadinessError("OSCAR requires the native model/cache configuration before loading weights")
    actual_model = getattr(model, "model", None)
    if not isinstance(actual_model, str) or not actual_model:
        raise OscarReadinessError("native model_config.model is unavailable")
    expected_path = str(Path(selected["model"]).expanduser().resolve())
    actual_path = str(Path(actual_model).expanduser().resolve())
    differences = {}
    for name, expected, actual in (
        ("model", expected_path, actual_path),
        ("quantization", selected["quantization"], getattr(model, "quantization", None)),
        ("dtype", selected["dtype"], _dtype(getattr(model, "dtype", None))),
        ("revision", selected.get("revision"), getattr(model, "revision", None)),
        ("hf_overrides", selected.get("hf_overrides") or {}, getattr(model, "hf_overrides", None) or {}),
    ):
        if expected != actual:
            differences[name] = {"oscar": expected, "native": actual}
    expected_state = resolve_mamba_cache_dtypes(selected)
    conv = _dtype(getattr(cache, "mamba_cache_dtype", None))
    conv = _dtype(getattr(model, "dtype", None)) if conv == "auto" else conv
    state = _dtype(getattr(cache, "mamba_ssm_cache_dtype", None))
    state = conv if state == "auto" else state
    for name, actual in (("mamba_cache_dtype", conv), ("mamba_ssm_cache_dtype", state)):
        if expected_state[name] != actual:
            differences[name] = {"oscar": expected_state[name], "native": actual}
    if differences:
        raise OscarReadinessError(
            f"OSCAR/native model identity mismatch before load_model: {differences}. "
            "Use the same --model/--quantization through scripts/install_serve.sh "
            "or align OSCAR_TARGET_CONFIG and prepare rotations for that configuration."
        )
    return {"model": actual_path, "quantization": selected["quantization"],
            "dtype": selected["dtype"], **expected_state}
