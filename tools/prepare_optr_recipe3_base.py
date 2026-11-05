"""Archive #86/#97/#165/#166/#170/#172/#174/#177: refine the actual recipe3 base.

This offline adapter preserves the original per-head FP64 fitting, key means,
pair manifest and cache identity. It never substitutes the shared zero-mean
seed, modifies serving operations, or initializes an NPU in the parent.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
_REFINEMENT_FIELDS = ("rotation_refinement", "optr_baseline", "optr_calibration", "optr_seed_capture",
                      "rotation_seed", "optr_recipe3_repair", "optr_recipe3_optimizer", "rotation_artifact")


def base_config(config, fingerprint):
    """Bind the same canonical recipe3 cache that pre-OptR candidate used.

    An explicit final OptR path and the shell's OSCAR_ROTATION_ARTIFACT must
    not redirect either reading or publishing the independent baseline pair.
    """
    from oscar_ascend.rotations import calibration_cache_identity, rotation_artifact_path
    selected = copy.deepcopy(config)
    for key in _REFINEMENT_FIELDS:
        selected.pop(key, None)
    if (selected.get("rotation_method", "qqt_sst") != "qqt_sst" or
            selected.get("rotation_variant") != "perhead_center" or
            selected.get("rotation_k_clip") is not None):
        raise ValueError("OptR baseline requires the unchanged recipe3 per-head scalar-clip configuration")
    profile = calibration_cache_identity(selected, fingerprint)
    if profile.get("recipe_version") != 3:
        raise ValueError("OptR baseline must identify recipe3")
    digest = hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]
    # Use the original path contract with an explicit canonical path so the
    # caller's process environment remains untouched and cannot redirect it.
    canonical = ROOT / "artifacts/rotations" / fingerprint / ("qqt_sst_" + digest) / "calibrated.pt"
    selected["rotation_artifact"] = str(rotation_artifact_path(
        {**selected, "rotation_artifact": str(canonical)}, fingerprint))
    return selected


def _identity(config):
    from oscar_ascend.rotations import calibration_cache_identity, checkpoint_identity
    if any(config.get(key) is not None for key in _REFINEMENT_FIELDS if key != "rotation_artifact"):
        raise ValueError("recipe3 base requires an isolated configuration without OptR or shared-seed fields")
    names, dim, fingerprint = checkpoint_identity(config["model"])
    expected = base_config(config, fingerprint)
    destination = config.get("rotation_artifact")
    if not destination or Path(destination).expanduser().resolve() != Path(expected["rotation_artifact"]):
        raise ValueError("recipe3 baseline destination differs from its authenticated canonical cache")
    return names, dim, fingerprint, calibration_cache_identity(expected, fingerprint)


def capture_source_metadata(artifact):
    """Describe only authenticated baseline sampling evidence, without guessing IDs.

    The caller must obtain the artifact through load_base. Older imported
    pairs may omit raw capture manifests: their known profile/token/hash
    evidence remains available, and missing prompt lengths stay explicit.
    """
    calibration = artifact["calibration"]
    profile = artifact["calibration_profile"]
    capture = artifact.get("capture_manifest")
    result = {"schema_version": 1, "model_fingerprint": artifact["model_fingerprint"],
              "calibration_profile": copy.deepcopy(profile),
              "prompt_source": copy.deepcopy(profile.get("prompt_source")),
              "prompt_sha256": calibration.get("prompt_sha256"),
              "token_count": calibration.get("tokens"),
              "capture_manifest_available": capture is not None, "capture_manifest": None}
    if capture is None:
        return result
    if (not isinstance(capture, dict) or
            capture.get("status") != "native_post_rope_capture_complete" or
            capture.get("model_fingerprint") != artifact["model_fingerprint"] or
            capture.get("prompt_sha256") != calibration.get("prompt_sha256") or
            capture.get("token_count") != calibration.get("tokens")):
        raise ValueError("recipe3 baseline capture identity differs from its authenticated pair")
    raw_profile = capture.get("calibration_profile")
    if raw_profile != profile and not (
            artifact.get("capture_fit_profile") == profile and
            artifact.get("refit_provenance", {}).get("source_calibration_profile") == raw_profile):
        raise ValueError("recipe3 baseline retained-capture provenance differs")
    lengths = capture.get("prompt_lengths")
    if lengths is not None:
        if (not isinstance(lengths, list) or not lengths or
                any(type(n) is not int or n <= 0 for n in lengths) or
                sum(lengths) != calibration.get("tokens")):
            raise ValueError("recipe3 baseline prompt lengths differ from captured token count")
        result["prompt_lengths"] = list(lengths)
    result["capture_calibration_profile"] = copy.deepcopy(raw_profile)
    if "prompt_source_sha256" in capture:
        result["prompt_source_sha256"] = capture["prompt_source_sha256"]
    result["capture_manifest"] = {key: copy.deepcopy(capture[key]) for key in (
        "status", "model_fingerprint", "prompt_sha256", "token_count", "prompt_lengths",
        "prompt_source_sha256", "calibration_profile") if key in capture}
    return result


def load_base(config, *, testing=False):
    """Verify the original pair and combined constants on CPU before fitting."""
    from oscar_ascend.rotations import load_active_artifact
    names, dim, fingerprint, _ = _identity(config)
    # testing=True enables CPU mathematical validation only. Explicit checks
    # below still reject every test-only artifact/pair in production use.
    artifact = load_active_artifact(config, path=config["rotation_artifact"],
        layer_names=names, head_dim=dim, model_fingerprint=fingerprint,
        device="cpu", testing=True)
    if not testing and (artifact.get("test_only") or artifact.get("calibration", {}).get("test_only") or
                        artifact.get("calibration", {}).get("model_forward_device") != "npu"):
        raise ValueError("test-only recipe3 baseline cannot enter production OptR fitting")
    capture_source_metadata(artifact)
    return artifact


def prepare_base(config, directory, *, testing=False, import_pair=None, base_runner=None):
    """Reuse the original pair or run its unchanged preparer in a child process."""
    if base_runner is not None and not testing:
        raise ValueError("injected recipe3 baseline preparation is test-only")
    _identity(config)
    destination = Path(config["rotation_artifact"])
    if destination.is_file() and not (destination.parent / ".oscar.pending.json").exists():
        return load_base(config, testing=testing)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Preserve the old preparer's automatic-cache behavior (retained recipe2
    # samples and interrupted publication included). Its child environment is
    # cleared of explicit PT overrides; the parent/worker retain the bound path.
    child_config = copy.deepcopy(config)
    child_config.pop("rotation_artifact")
    config_path = directory / "recipe3-base-config.json"
    config_path.write_text(json.dumps(child_config, indent=2) + "\n")
    if base_runner is not None:
        base_runner(child_config, config_path)
    elif testing:
        from tools.prepare_rotations import prepare
        # Explicit output prevents a test runner's inherited environment from
        # affecting the destination. Production uses the unchanged child path.
        prepare(config, config_path=config_path, output=destination,
                import_pair=import_pair, testing=True)
    else:
        command = [sys.executable, "-m", "tools.prepare_rotations", "--config", str(config_path)]
        if import_pair is not None:
            if not isinstance(import_pair, (tuple, list)) or len(import_pair) != 2:
                raise ValueError("recipe3 import requires the K/V pair")
            command.extend(["--import-k", str(Path(import_pair[0]).expanduser().resolve()),
                            "--import-v", str(Path(import_pair[1]).expanduser().resolve())])
        environment = dict(os.environ)
        environment.pop("OSCAR_ROTATION_ARTIFACT", None)
        from tools.rotation_preparation_limits import preparation_progress_timeout
        subprocess.run(command, check=True, env=environment,
                       timeout=preparation_progress_timeout(config))
    return load_base(config, testing=testing)
