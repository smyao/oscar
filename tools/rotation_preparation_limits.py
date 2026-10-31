"""Archive #86/#94/#95/#97/#166: bounded offline calibration, visible failure.

OptR fits every FULL-layer KV head on the host after native capture. Its total
work budget is separate from the unchanged no-progress deadline enforced by
the preparer. These are preparation limits, never serving/timer settings or
runtime estimates. No model weights, torch or device APIs are loaded here.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from oscar_ascend.optr_policy import optr_policy


MAX_PREPARATION_SECONDS = 24 * 60 * 60
MAX_PROGRESS_SECONDS = 30 * 60
_FIXED_PREPARATION_SECONDS = 60 * 60


def preparation_progress_timeout(config: dict) -> float:
    """A real completed-work event must arrive within the old phase deadline."""
    value = config.get("phase_timeout_seconds", MAX_PROGRESS_SECONDS)
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("phase_timeout_seconds must be finite and positive")
    return min(value, MAX_PROGRESS_SECONDS)


def _fit_geometry(config: dict) -> tuple[int, int] | None:
    """Read only small metadata; unavailable plan geometry uses the hard cap.

    This does not validate a model or permit a missing model to run. The
    existing preparer still authenticates its complete checkpoint identity.
    """
    model = config.get("model")
    if not model:
        return None
    path = Path(model) / "config.json"
    if not path.is_file():
        return None
    document = json.loads(path.read_text())
    text = document.get("text_config", document)
    layers, heads = text.get("num_hidden_layers"), text.get("num_key_value_heads")
    if layers is None or heads is None:
        return None
    if type(layers) is not int or layers <= 0 or type(heads) is not int or heads <= 0:
        raise ValueError("invalid model geometry for rotation preparation budget")
    kinds = text.get("layer_types")
    if kinds is not None:
        if (not isinstance(kinds, list) or len(kinds) != layers or
                any(kind not in ("full_attention", "linear_attention") for kind in kinds)):
            raise ValueError("invalid layer_types for rotation preparation budget")
        full = kinds.count("full_attention")
    else:
        interval = text.get("full_attention_interval")
        if interval is None:
            return None
        if type(interval) is not int or interval <= 0:
            raise ValueError("invalid full_attention_interval for rotation preparation budget")
        full = layers // interval
    if full <= 0:
        raise ValueError("model has no FULL layers for rotation preparation")
    return full, heads


def preparation_phase_timeout(config: dict) -> int | float:
    """Bound all OptR work; unchanged recipes keep their existing phase limit.

    At 16 FULL layers, four KV heads and 16 steps this permits at most 24h.
    Half of the variable allowance is for fitting and half for validation,
    which still runs with short research fitting schedules. Resume uses the
    same bound and skips only authenticated completed work in the preparer.
    A shorter real-progress watchdog is mandatory throughout CPU preparation.
    """
    if config.get("rotation_refinement") != "optr":
        return config["phase_timeout_seconds"]
    preparation_progress_timeout(config)
    policy = optr_policy(config)
    geometry = _fit_geometry(config)
    if geometry is None:
        return MAX_PREPARATION_SECONDS
    layers, heads = geometry
    relative_work = layers * heads * (policy["steps"] + 16) / (16 * 4 * 32)
    budget = _FIXED_PREPARATION_SECONDS + math.ceil(
        (MAX_PREPARATION_SECONDS - _FIXED_PREPARATION_SECONDS) * relative_work)
    return min(MAX_PREPARATION_SECONDS, budget)


def phase_timeout(config: dict, name: str) -> int | float:
    """Apply calibration time only to preparation and its BF16 parent phase."""
    ordinary = config["phase_timeout_seconds"]
    if config.get("rotation_refinement") == "optr":
        if name == "prepare-rotations":
            return preparation_phase_timeout(config)
        if name == "bf16-correctness-gates":
            # observe_serve nests prepare-rotations; retain its old additional
            # budget for build/correctness work while allowing the child work.
            return preparation_phase_timeout(config) + ordinary
    return ordinary
