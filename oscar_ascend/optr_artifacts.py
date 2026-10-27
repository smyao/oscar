"""Archive #86/#97/#148: authenticated offline OptR constants and certificates.

Used during preparation/loading only. No optimizer, traces or extra matrix
operations are installed in the serving forward path. The old qqt/sst pair
contract is deliberately kept separate.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import uuid

from .optr_policy import OBJECTIVE, RECIPE_VERSION, UPSTREAM_COMMIT


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix_digest(layers):
    """Bind final matrices and means, with names, shapes, dtype and ordering."""
    import torch
    digest = hashlib.sha256()
    for name in sorted(layers):
        for key in ("Rk", "Rv", "k_mean"):
            tensor = layers[name].get(key)
            if not torch.is_tensor(tensor) or tensor.dtype != torch.float32:
                raise ValueError("OptR constants must be FP32 tensors")
            digest.update(json.dumps([name, key, list(tensor.shape)], separators=(",", ":")).encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _metric(metric):
    if not isinstance(metric, dict):
        raise ValueError("OptR metric must be an object")
    for key in ("normalized_error", "squared_error", "teacher_energy"):
        value = metric.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("OptR metrics must be finite and nonnegative")
    rows = metric.get("requests")
    if not isinstance(rows, list) or not rows:
        raise ValueError("OptR certificate requires per-request metrics")
    ids = [row.get("request_id") for row in rows]
    if any(not isinstance(x, str) or not x for x in ids) or len(ids) != len(set(ids)):
        raise ValueError("OptR metric request identities must be unique")
    for row in rows:
        for key in ("normalized_error", "squared_error", "teacher_energy"):
            value = row.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("OptR per-request metric must be finite and nonnegative")
        if (not isinstance(row.get("sample_sha256"), str) or len(row["sample_sha256"]) != 64 or
                any(c not in "0123456789abcdef" for c in row["sample_sha256"])):
            raise ValueError("OptR metric must identify the exact sample")
        expected = row["squared_error"] / max(row["teacher_energy"], 1e-20)
        if not math.isclose(row["normalized_error"], expected, rel_tol=1e-12, abs_tol=0):
            raise ValueError("OptR per-request normalization differs")
    for key in ("squared_error", "teacher_energy"):
        if not math.isclose(metric[key], sum(row[key] for row in rows), rel_tol=1e-12, abs_tol=0):
            raise ValueError("OptR metric totals differ from requests")
    expected = metric["squared_error"] / max(metric["teacher_energy"], 1e-20)
    if not math.isclose(metric["normalized_error"], expected, rel_tol=1e-12, abs_tol=0):
        raise ValueError("OptR metric normalization differs")
    return {row["request_id"]: row for row in rows}


def nonworse(before, after):
    """No relaxed margin: same samples, same denominator, every request passes."""
    left, right = _metric(before), _metric(after)
    if set(left) != set(right) or before["teacher_energy"] != after["teacher_energy"]:
        raise ValueError("OptR comparison used different requests or teacher")
    for key in left:
        if any(left[key][field] != right[key][field] for field in ("sample_sha256", "teacher_energy")):
            raise ValueError("OptR comparison sample/teacher identity differs")
    return (after["squared_error"] <= before["squared_error"] and
            after["normalized_error"] <= before["normalized_error"] and
            all(right[key]["squared_error"] <= left[key]["squared_error"] and
                right[key]["normalized_error"] <= left[key]["normalized_error"] for key in left))


def validate_refinement_metadata(artifact, *, expected_profile=None):
    """Check the complete certificate before constants can reach device memory."""
    profile = artifact.get("calibration_profile", {})
    if (artifact.get("objective") != OBJECTIVE or artifact.get("calibrated") is not True or
            artifact.get("accumulation_dtype") != "float32" or
            profile.get("recipe_version") != RECIPE_VERSION or
            profile.get("objective") != OBJECTIVE or profile.get("rotation_refinement") != "optr" or
            profile.get("refinement_upstream_commit") != UPSTREAM_COMMIT):
        raise ValueError("OptR recipe/objective provenance differs")
    if expected_profile is not None and profile != expected_profile:
        raise ValueError("OptR calibration profile differs from selected configuration")
    base = profile.get("base_calibration_profile", {})
    if (base.get("recipe_version") != 3 or base.get("rotation_variant") != "perhead_center" or
            base.get("model_fingerprint") != artifact.get("model_fingerprint")):
        raise ValueError("OptR must refine the authenticated recipe3 model")
    certificate = artifact.get("refinement_certificate", {})
    if (certificate.get("schema_version") != 1 or
            certificate.get("policy") != profile.get("refinement_policy") or
            certificate.get("metric") != "all_tp_heads_bf16_gate_fp32_wo_output" or
            certificate.get("arithmetic_evidence") != "cpu_deployment_reference_not_npu_bitwise" or
            certificate.get("model_quality") != "not_measured" or
            certificate.get("performance") != "not_measured"):
        raise ValueError("OptR certificate policy/evidence boundary differs")
    for key in ("base_artifact_sha256", "trace_manifest_sha256", "matrices_sha256"):
        value = certificate.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("OptR certificate lacks content hashes")
    layers = artifact.get("layers", {})
    if set(certificate.get("layers", {})) != set(layers) or not layers:
        raise ValueError("OptR certificate FULL layer coverage differs")
    for name, record in certificate["layers"].items():
        selected_digest = matrix_digest({name: layers[name]})
        if record.get("selected_constants_sha256") != selected_digest:
            raise ValueError("OptR layer certificate does not bind selected constants")
        base_digest = record.get("base_constants_sha256", "")
        if not isinstance(base_digest, str) or len(base_digest) != 64 or any(c not in "0123456789abcdef" for c in base_digest):
            raise ValueError("OptR layer certificate lacks baseline constants hash")
        for split in ("validation", "test"):
            result = record.get(split, {})
            if not nonworse(result.get("baseline"), result.get("selected")):
                raise ValueError(f"OptR {name} worsens {split} output error")
        if type(record.get("accepted")) is not bool:
            raise ValueError("OptR layer decision is missing")
        if record["accepted"] and not (record["validation"]["selected"]["squared_error"] <
                                       record["validation"]["baseline"]["squared_error"]):
            raise ValueError("OptR changed layer must strictly improve validation output")
        if not record["accepted"] and (base_digest != selected_digest or any(
                record[split]["selected"] != record[split]["baseline"] for split in ("validation", "test"))):
            raise ValueError("OptR rejected layer must retain baseline constants and metrics")
        val_ids = {r["request_id"] for r in record["validation"]["baseline"]["requests"]}
        test_ids = {r["request_id"] for r in record["test"]["baseline"]["requests"]}
        val_hashes = {r["sample_sha256"] for r in record["validation"]["baseline"]["requests"]}
        test_hashes = {r["sample_sha256"] for r in record["test"]["baseline"]["requests"]}
        if val_ids & test_ids or val_hashes & test_hashes:
            raise ValueError("OptR independent test overlaps matrix selection requests")
    accepted = sorted(name for name, record in certificate["layers"].items() if record["accepted"])
    if certificate.get("accepted_layers") != accepted:
        raise ValueError("OptR accepted layer list differs")
    if matrix_digest(layers) != certificate["matrices_sha256"]:
        raise ValueError("OptR certificate does not bind the deployed matrices")
    return certificate


def publish_artifact(artifact, destination):
    """Atomic single generation, with an explicit interrupted-publication mark."""
    import torch
    from .rotation_calibration import _atomic_json
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    validate_refinement_metadata(artifact)
    pending = destination.parent / ".optr.pending.json"
    complete = destination.parent / ".optr.complete.json"
    with (destination.parent / ".optr.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        fd, temporary = tempfile.mkstemp(prefix=".optr-", suffix=".pt", dir=destination.parent)
        os.close(fd)
        try:
            torch.save(artifact, temporary)
            with open(temporary, "rb") as stream:
                os.fsync(stream.fileno())
            manifest = {"schema_version": 1, "generation_id": uuid.uuid4().hex,
                        "model_fingerprint": artifact["model_fingerprint"],
                        "artifact_name": destination.name, "sha256": file_sha256(temporary)}
            _atomic_json(pending, manifest)
            os.replace(temporary, destination)
            _atomic_json(complete, manifest)
            pending.unlink()
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
            fcntl.flock(lock, fcntl.LOCK_UN)


def load_refined_artifact(config, destination, *, device, testing, layer_names, head_dim, model_fingerprint):
    import torch
    from .rotation_calibration import OFFICIAL_COMMIT
    from .rotations import calibration_cache_identity, validate_artifact
    destination = Path(destination)
    with (destination.parent / ".optr.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        if (destination.parent / ".optr.pending.json").exists():
            raise ValueError("incomplete OptR artifact publication")
        complete = destination.parent / ".optr.complete.json"
        if not complete.is_file():
            raise ValueError("authenticated OptR manifest required; old recipe3 PT cannot be relabelled")
        manifest = json.loads(complete.read_text())
        if (manifest.get("schema_version") != 1 or manifest.get("artifact_name") != destination.name or
                manifest.get("model_fingerprint") != model_fingerprint or
                manifest.get("sha256") != file_sha256(destination)):
            raise ValueError("OptR artifact hash or identity mismatch")
        artifact = torch.load(destination, map_location="cpu", weights_only=True)
    if artifact.get("test_only") and not testing:
        raise ValueError("test-only OptR artifacts cannot enter production")
    validate_refinement_metadata(artifact, expected_profile=calibration_cache_identity(config, model_fingerprint))
    model = json.loads((Path(config["model"]) / "config.json").read_text())
    model = model.get("text_config", model)
    if artifact.get("global_kv_heads") != model.get("num_key_value_heads"):
        raise ValueError("OptR KV-head count differs from selected model")
    return validate_artifact(artifact, layer_names=layer_names, head_dim=head_dim,
        model_fingerprint=model_fingerprint, pr_fingerprint=OFFICIAL_COMMIT,
        require_calibrated=True, device=device, testing=testing)
