# Archive #26/#28-30/#84/#86/#96/#97: preserve native MTP/GDN and lightweight
# startup ordering; inspect only model JSON, never import torch or native code.
"""Shared model selection for direct, observed and native service entrypoints.

Qwen3.8 uses the supported Qwen3_5 architecture according to the official
release. A model's directory name is never evidence of its architecture or
quantization. Actual startup requires local metadata; plans may defer it.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

QUANTIZATION_MODES = ("auto", "none", "ascend")
SUPPORTED_ARCHITECTURES = {"Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM"}


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", help="local Qwen3.8/3.5 model directory; metadata determines quantization")
    parser.add_argument("--served-model-name", "--served-name", dest="served_model_name",
                        help="public API model name; independent of the checkpoint directory")
    parser.add_argument("--quantization", choices=QUANTIZATION_MODES,
                        help="auto: inspect metadata; none: require float weights; ascend: explicit ModelSlim")


def _read_object(path: Path) -> tuple[dict, str]:
    data = path.read_bytes()
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"model metadata must be a JSON object: {path}")
    return value, hashlib.sha256(data).hexdigest()


def _nonempty_string(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def apply_model_options(config: dict, args=None, *, require_model: bool = False) -> dict:
    """Return effective model options without modifying the input or model.

    A new --model starts in auto mode unless --quantization is also supplied;
    this prevents a prior checkpoint's explicit ascend/none choice leaking.
    ``none`` cannot silently disable native metadata auto-detection: quantized
    metadata with that choice is rejected instead of loading quantized weights.
    """
    result = deepcopy(config)
    model_override = getattr(args, "model", None)
    name_override = getattr(args, "served_model_name", None)
    quant_override = getattr(args, "quantization", None)
    if model_override is not None:
        result["model"] = model_override
    if name_override is not None:
        result["served_model_name"] = name_override
    model = _nonempty_string(result.get("model"), "model")
    _nonempty_string(result.get("served_model_name"), "served_model_name")
    mode = (quant_override if quant_override is not None else
            "auto" if model_override is not None else
            result.get("quantization_mode", result.get("quantization", "auto")))
    mode = "none" if mode is None else mode
    if mode not in QUANTIZATION_MODES:
        raise ValueError(f"quantization must be one of {QUANTIZATION_MODES}, got {mode!r}")
    if result.get("dtype", "bfloat16") != "bfloat16":
        raise ValueError("OSCAR Qwen3.8/3.5 requires dtype=bfloat16 activations")
    result.update(dtype="bfloat16", quantization_mode=mode)
    model_dir = Path(model).expanduser()
    model_file = model_dir / "config.json"
    resolution = {"model": model, "requested_quantization": mode,
                  "dtype": "bfloat16", "metadata": {}}
    if not model_file.is_file():
        if require_model:
            raise ValueError(f"model config.json is unavailable: {model_file}; select a local checkpoint with --model")
        result["quantization"] = {"auto": "auto", "none": None, "ascend": "ascend"}[mode]
        resolution.update(status="deferred_model_unavailable", quantization=result["quantization"])
        result["model_resolution"] = resolution
        return result

    metadata, digest = _read_object(model_file)
    resolution["metadata"]["config.json"] = digest
    architectures = metadata.get("architectures")
    if (not isinstance(architectures, list) or not architectures or
            any(not isinstance(a, str) or a not in SUPPORTED_ARCHITECTURES for a in architectures)):
        raise ValueError(f"unsupported model architectures {architectures!r}; this plugin requires an existing dense Qwen3_5 implementation")
    text_config = metadata.get("text_config", metadata)
    if not isinstance(text_config, dict):
        raise ValueError("model text_config must be an object")
    if text_config.get("model_type") not in (None, "qwen3_5_text", "qwen3_5"):
        raise ValueError(f"unsupported model text model_type {text_config.get('model_type')!r}")
    methods = set()
    for section in (metadata, text_config):
        quant = section.get("quantization_config")
        if quant is None:
            continue
        if not isinstance(quant, dict):
            raise ValueError("model quantization_config must be an object or null")
        method = quant.get("quant_method")
        if method not in (None, ""):
            methods.add(_nonempty_string(method, "quant_method"))
        else:
            methods.add("ascend")  # native ModelSlim override for method-less metadata
    sidecar = model_dir / "quant_model_description.json"
    if sidecar.exists():
        description, digest = _read_object(sidecar)
        if not description:
            raise ValueError(f"empty ModelSlim metadata: {sidecar}")
        resolution["metadata"][sidecar.name] = digest
        methods.add("ascend")
    if len(methods) > 1:
        raise ValueError(f"conflicting checkpoint quantization metadata: {sorted(methods)}")
    detected = next(iter(methods), None)
    if detected not in (None, "ascend"):
        raise ValueError(f"unsupported checkpoint quantization {detected!r}; this entrypoint supports original or Ascend ModelSlim weights")
    if mode == "none" and detected is not None:
        raise ValueError("--quantization none conflicts with quantized checkpoint metadata; use auto or select original weights")
    resolved = detected if mode == "auto" else None if mode == "none" else "ascend"
    result["quantization"] = resolved
    resolution.update(status="resolved", architectures=architectures,
                      detected_quantization=detected, quantization=resolved)
    result["model_resolution"] = resolution
    return result
