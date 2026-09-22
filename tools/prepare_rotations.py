# 档案 G19/G20/G23/#86/#96/#97：自动生成可追溯 data-free artifact；不冒充样本校准。
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def model_geometry(model: Path) -> tuple[list[str], int, str]:
    raw = (model / "config.json").read_bytes()
    config = json.loads(raw)
    text = config.get("text_config", config)
    kinds = text.get("layer_types")
    count = text["num_hidden_layers"]
    if kinds is None:
        interval = text.get("full_attention_interval")
        if not isinstance(interval, int) or interval <= 0:
            raise ValueError("model must declare layer_types or full_attention_interval")
        kinds = ["full_attention" if (i+1) % interval == 0 else "linear_attention" for i in range(count)]
    if len(kinds) != count or any(x not in {"full_attention", "linear_attention"} for x in kinds):
        raise ValueError("unexpected hybrid layer_types")
    layers = [f"model.layers.{i}.self_attn.attn" for i, kind in enumerate(kinds) if kind == "full_attention"]
    dim = text.get("head_dim")
    if dim is None:
        hidden, heads = text["hidden_size"], text["num_attention_heads"]
        if hidden % heads:
            raise ValueError("hidden_size is not divisible by query heads")
        dim = hidden // heads
    if not layers or not isinstance(dim, int) or dim <= 0:
        raise ValueError("model has no FULL layers or invalid head_dim")
    # Source identity binds to actual config and weight index, not a path-only identifier.
    hasher = hashlib.sha256(raw)
    index = model / "model.safetensors.index.json"
    if index.exists():
        hasher.update(index.read_bytes())
    return layers, dim, hasher.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/target.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    from .target_cli import target_env
    os.environ.update(target_env(config))
    layers, dim, fingerprint = model_geometry(Path(config["model"]))
    import torch
    import torch_npu
    torch.npu.set_device(0)  # logical 0 after physical selection
    from oscar_ascend.rotations import build_hadamard_artifact, load_artifact, save_artifact
    output = args.output or ROOT / "artifacts/rotations" / fingerprint / "hadamard.pt"
    if config["rotation_method"] != "hadamard":
        raise ValueError("automatic preparation currently implements the paper's explicit data-free Hadamard method only; calibrated artifact generation is not integrated")
    if output.exists():
        # Strict validation is repeated below through the public validator.
        artifact = load_artifact(output, layer_names=layers, head_dim=dim,
                                 model_fingerprint=fingerprint, device="npu:0")
    else:
        artifact = build_hadamard_artifact(layer_names=layers, head_dim=dim,
                                           model_fingerprint=fingerprint, device="npu:0")
        save_artifact(artifact, output)
    print(json.dumps({"artifact": str(output), "source": "hadamard_data_free", "model_fingerprint": fingerprint,
                      "full_layers": layers, "quality_acceptance": "not_run", "draft_layer": "not_yet_resolved"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
