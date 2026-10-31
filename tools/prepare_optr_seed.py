"""Prepare the user-selected OSCAR shared seed in an isolated native process.

Archive #86/#97/#166/#170/#176: complete native capture, source-bound constants,
atomic publication, and no CPU artifact in serving. Recipe2/3 are unchanged.
Only the offline OptR preparer/worker consumes this intermediate artifact.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import tempfile

from oscar_ascend.optr_artifacts import file_sha256


def _identity(config):
    from oscar_ascend.optr_oscar_seed import SEED_RECIPE
    from oscar_ascend.rotations import calibration_cache_identity, checkpoint_identity
    if config.get("rotation_seed") != SEED_RECIPE or config.get("rotation_refinement") is not None:
        raise ValueError("source OSCAR seed requires its isolated preparation configuration")
    names, dim, fingerprint = checkpoint_identity(config["model"])
    profile = calibration_cache_identity(config, fingerprint)
    model = json.loads((Path(config["model"]) / "config.json").read_text())
    model = model.get("text_config", model)
    geometry = dict(layer_names=names, head_dim=dim,
        global_q_heads=model["num_attention_heads"], global_kv_heads=model["num_key_value_heads"],
        tp_size=config["tensor_parallel_size"])
    return profile, geometry


def publish_seed(artifact, destination):
    """Publish the single intermediate generation with a hashed commit marker."""
    import torch
    from tools.phase import atomic_json
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (destination.parent / ".seed.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        fd, temporary = tempfile.mkstemp(prefix=".seed-", suffix=".pt", dir=destination.parent)
        os.close(fd)
        pending = destination.parent / ".seed.pending.json"
        try:
            torch.save(artifact, temporary)
            with open(temporary, "rb") as stream:
                os.fsync(stream.fileno())
            manifest = {"schema_version": 1, "file": destination.name,
                        "sha256": file_sha256(temporary), "profile": artifact["calibration_profile"]}
            atomic_json(pending, manifest)
            os.replace(temporary, destination)
            atomic_json(destination.parent / ".seed.complete.json", manifest)
            pending.unlink()
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def load_seed(config, *, testing=False):
    import torch
    from oscar_ascend.optr_oscar_seed import validate_seed_artifact
    profile, geometry = _identity(config)
    destination = Path(config["rotation_artifact"])
    with (destination.parent / ".seed.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        if (destination.parent / ".seed.pending.json").exists():
            raise ValueError("incomplete source OSCAR seed publication")
        manifest = json.loads((destination.parent / ".seed.complete.json").read_text())
        if (manifest.get("schema_version") != 1 or manifest.get("file") != destination.name or
                manifest.get("profile") != profile or manifest.get("sha256") != file_sha256(destination)):
            raise ValueError("source OSCAR seed manifest/hash/profile differs")
        artifact = torch.load(destination, map_location="cpu", weights_only=True)
    return validate_seed_artifact(artifact, expected_profile=profile, testing=testing, **geometry)


def prepare(config, *, config_path, testing=False, capture_runner=None):
    import torch
    from oscar_ascend.optr_oscar_seed import fit_seed_artifact
    from tools.prepare_rotations import _capture
    if capture_runner is not None and not testing:
        raise ValueError("injected source seed capture is test-only")
    profile, geometry = _identity(config)
    destination = Path(config["rotation_artifact"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and not (destination.parent / ".seed.pending.json").exists():
        load_seed(config, testing=testing)
        return {"artifact": str(destination), "source": "oscar_base_shared_seed", "reused": True}
    directory = Path(tempfile.mkdtemp(prefix="capture-", dir=destination.parent))
    capture = (capture_runner or _capture)(config_path, directory)
    if (capture.get("status") != "native_post_rope_capture_complete" or
            capture.get("calibration_profile") != profile or
            capture.get("model_fingerprint") != profile["model_fingerprint"] or
            capture.get("layer_ids") != [int(n.split(".")[2]) for n in geometry["layer_names"]]):
        raise ValueError("source OSCAR seed native capture identity differs")
    paths, hashes = capture.get("rank_paths"), capture.get("rank_sha256")
    if (not isinstance(paths, list) or len(paths) != geometry["tp_size"] or
            len(set(paths)) != len(paths) or not isinstance(hashes, dict) or
            set(hashes) != set(paths) or any(hashes[p] != file_sha256(p) for p in paths)):
        raise ValueError("source OSCAR seed native rank payload hashes differ")
    ranks = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    if any(row.get("count") != capture.get("token_count")
           for rank in ranks for row in rank.get("layers", {}).values()):
        raise ValueError("source OSCAR seed capture token coverage differs")
    artifact = fit_seed_artifact(ranks, profile=profile, prompt_sha256=capture["prompt_sha256"],
                                testing=testing, **geometry)
    artifact["capture_manifest"] = capture
    publish_seed(artifact, destination)
    load_seed(config, testing=testing)
    return {"artifact": str(destination), "source": "oscar_base_shared_seed", "reused": False,
            "model_quality": "not_measured", "performance": "not_measured"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    from tools.model_config import apply_model_options
    from tools.target_cli import target_env
    config = apply_model_options(json.loads(args.config.read_text()), require_model=True)
    os.environ.update(target_env(config))
    print(json.dumps(prepare(config, config_path=args.config)), flush=True)


if __name__ == "__main__":
    main()
