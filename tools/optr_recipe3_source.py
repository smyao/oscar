"""Archive #86/#97/#165/#170/#174/#176: reuse authenticated rejected v10 work.

Saved proposals and validation measurements are inputs to a new offline search,
never publishable v11 results. This reader neither loads trace tensors nor fits,
replays, initializes NPU state, changes gates or writes the historical cache.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re


LEGACY_FITTING_SHA256 = "a30488344d57137d7bfc0623e074716744369eef999ac1827014fd907e02a917"
LEGACY_POLICY = "optr_recipe3_centered_candidates_ascend_v10"
_ACCEPTANCE = dict(require_frames=True, require_terminal=True,
                   require_queries=True, require_projection=True)


def _json(path):
    return json.loads(Path(path).read_text())


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _layer_ids(names):
    if not isinstance(names, list) or not names or len(names) != len(set(names)):
        raise ValueError("recipe3 source requires unique FULL layer names")
    ids = []
    for name in names:
        match = re.fullmatch(r"model\.layers\.(\d+)\.self_attn\.attn", name) if isinstance(name, str) else None
        if match is None:
            raise ValueError("recipe3 source FULL layer name is invalid")
        ids.append(int(match[1]))
    if len(ids) != len(set(ids)):
        raise ValueError("recipe3 source FULL layer owners overlap")
    return ids


def _generation_name(identity):
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def source_snapshot_digest(source_spec):
    """Bind a new fitting generation to the entire authenticated old snapshot."""
    if not isinstance(source_spec, dict) or source_spec.get("schema_version") != 1:
        raise ValueError("recipe3 source snapshot schema differs")
    return _generation_name(source_spec)


def _generation(root, identity):
    """Absent historical work is optional; damaged matching work is an error."""
    expected = _generation_name(identity)
    if not root.exists():
        return None
    if not root.is_dir() or root.is_symlink():
        raise ValueError("recipe3 source checkpoint directory is invalid")
    found = None
    for child in sorted(root.iterdir()):
        if child.name == expected and (not child.is_dir() or child.is_symlink()):
            raise ValueError("recipe3 source generation is not a real directory")
        if not child.is_dir():
            continue
        matching = child.name == expected
        # Detect duplicate/noncanonical generations rather than choosing one.
        for path in child.glob("layer-*.json"):
            try:
                matching = matching or _json(path).get("identity") == identity
            except (ValueError, OSError, AttributeError):
                if child.name == expected:
                    raise ValueError("recipe3 source checkpoint manifest is unreadable") from None
        if not matching:
            continue
        if child.is_symlink() or child.name != expected or found is not None:
            raise ValueError("recipe3 source generation is ambiguous or noncanonical")
        expected_keys = {f"layer-{index}" for index in _layer_ids(identity["full_layers"])}
        actual = {path.stem for path in child.glob("layer-*.json")
                  if re.fullmatch(r"layer-\d+", path.stem)}
        if actual != expected_keys:
            raise ValueError("recipe3 source requires completed coverage of every model FULL layer")
        found = child
    return found


def _read_checkpoint(path, key, identity, expected_hash=None):
    from oscar_ascend.optr_artifacts import file_sha256
    from oscar_ascend.optr_checkpoint import FitCheckpoints
    if path.is_symlink() or not path.is_file():
        raise ValueError("recipe3 source checkpoint manifest is missing or linked")
    digest = file_sha256(path)
    if expected_hash is not None and digest != expected_hash:
        raise ValueError("recipe3 source manifest changed after discovery")
    manifest = _json(path)
    filename = manifest.get("file")
    if not isinstance(filename, str) or not re.fullmatch(re.escape(key) + r"-[0-9a-f]{32}\.pt", filename):
        raise ValueError("recipe3 source checkpoint filename is invalid")
    if (path.parent / filename).is_symlink():
        raise ValueError("recipe3 source checkpoint tensor file is linked")
    payload = FitCheckpoints._load_from(path, key, identity)
    if file_sha256(path) != digest:
        raise ValueError("recipe3 source manifest changed while reading")
    return payload, digest


def _trace(path, profile, identity):
    from oscar_ascend.optr_artifacts import file_sha256
    if path.is_symlink() or not path.is_file() or file_sha256(path) != identity["trace_manifest_sha256"]:
        raise ValueError("recipe3 source trace manifest hash differs")
    trace = _json(path)
    test_only = trace.get("test_only", False)  # Production capture omits this optional field.
    if (type(test_only) is not bool or (test_only and not identity["test_only"]) or
            trace.get("schema_version") != 1 or trace.get("status") != "native_optr_capture_complete" or
            trace.get("model_fingerprint") != profile["model_fingerprint"] or
            trace.get("calibration_profile") != profile or trace.get("optr_policy") != profile["refinement_policy"] or
            trace.get("global_kv_heads") != identity["global_kv_heads"] or
            trace.get("head_dim") != identity["head_dim"] or
            trace.get("layer_ids") != _layer_ids(identity["full_layers"]) or
            trace.get("recipe3_sampling", {}).get("seed_capture", {}).get("baseline_artifact_sha256") !=
            identity["base_artifact_sha256"]):
        raise ValueError("recipe3 source trace profile/base/geometry or test-only permission differs")
    return trace


def _validate_layer(payload, proposal, identity, name, seed, policy):
    import torch
    from oscar_ascend.optr_artifacts import matrix_digest, nonworse, validate_refinement_metadata
    from tools.prepare_optr_recipe3 import _validate_result
    profile = identity["profile"]
    artifact = payload.get("artifact")
    if (not isinstance(artifact, dict) or artifact.get("test_only") is not identity["test_only"] or
            artifact.get("format_version") != 2 or artifact.get("source_grouping") != "layer_kv_head" or
            artifact.get("rotation_variant") != "perhead_center" or
            artifact.get("model_fingerprint") != profile["model_fingerprint"] or
            artifact.get("head_dim") != identity["head_dim"] or
            artifact.get("global_kv_heads") != identity["global_kv_heads"] or
            set(artifact.get("layers", {})) != {name}):
        raise ValueError("recipe3 source completed artifact identity differs")
    certificate = validate_refinement_metadata(artifact, expected_profile=profile)
    if (certificate.get("schema_version") != 3 or certificate.get("full_layer_names") != identity["full_layers"] or
            certificate.get("accepted_layers") != [] or certificate.get("independent_test", {}).get("status") != "passed"):
        raise ValueError("recipe3 source requires rejected schema3 layers and a passing independent seed test")
    for field in ("base_artifact_sha256", "trace_manifest_sha256", "fitting_implementation_sha256"):
        if certificate.get(field) != identity[field]:
            raise ValueError("recipe3 source completed artifact source/implementation differs")
    record, constants = certificate["layers"][name], artifact["layers"][name]
    base_digest = matrix_digest({name: seed})
    if (seed.get("layer_id") != _layer_ids([name])[0] or constants.get("layer_id") != seed["layer_id"] or
            tuple(seed["Rk"].shape) != (identity["global_kv_heads"], identity["head_dim"], identity["head_dim"]) or
            record["accepted"] is not False or record["base_constants_sha256"] != base_digest or
            record.get("recipe3_baseline") != {"recipe_version": 3, "rotation_variant": "perhead_center",
                                               "constants_sha256": base_digest} or
            any(not torch.equal(constants[field], seed[field]) for field in ("Rk", "Rv", "k_mean"))):
        raise ValueError("recipe3 source rejected layer does not retain the actual recipe3 seed")
    binding = {"strategy": policy["deployment_candidate_selection"], "policy": policy,
               "base_constants_sha256": base_digest, "fitting_implementation_sha256": LEGACY_FITTING_SHA256}
    if proposal.get("recipe3_binding") != binding:
        raise ValueError("recipe3 source proposal binding differs")
    result = _validate_result(proposal.get("result"), seed, policy)
    if record.get("shared_decision") != result["decision"]:
        raise ValueError("recipe3 source saved decision differs from proposal")
    selection = record.get("shared_layer_selection", {})
    measured = selection.get("evaluated_candidates")
    if (selection.get("method") != "recipe3_perhead_existing_kv_checkpoints_v1" or
            selection.get("candidate_order") != "all_existing_checkpoints_best_validation_squared_error" or
            selection.get("accepted") is not False or selection.get("selected_candidate") != "seed" or
            selection.get("test_used_for_selection") is not False or
            selection.get("max_candidates") != policy["max_deployment_candidates"] or
            not isinstance(measured, list) or len(measured) != len(result["candidates"])):
        raise ValueError("recipe3 source candidate coverage/selection differs")
    before = record["validation"]["baseline"]
    for candidate, evaluation in zip(result["candidates"], measured):
        digest = matrix_digest({name: {**seed, "Rk": candidate["Rk"], "Rv": candidate["Rv"]}})
        if (not isinstance(evaluation, dict) or evaluation.get("candidate") != candidate["id"] or
                evaluation.get("phase") != candidate["phase"] or evaluation.get("step") != candidate["step"] or
                evaluation.get("constants_sha256") != digest or evaluation.get("accepted") is not False):
            raise ValueError("recipe3 source candidate metrics do not bind rejected matrices")
        after = evaluation.get("validation")
        if not nonworse(after, after, **_ACCEPTANCE):
            raise ValueError("recipe3 source candidate metric is invalid")
        passes = nonworse(before, after, **_ACCEPTANCE) and after["squared_error"] < before["squared_error"]
        if passes or (digest == base_digest and after != before):
            raise ValueError("recipe3 source candidate metrics contradict the unchanged strict gate")
    if selection.get("proposal") != measured[0]["validation"]:
        raise ValueError("recipe3 source primary proposal metrics differ")
    return result, record


def discover_source(config, trace_manifest, baseline_hashes, names, global_heads, D, testing=False):
    """Return a JSON snapshot of complete historical work, or None if absent.

    The caller authenticates raw capture tensors separately. This reader binds
    their exact manifest bytes and saved validation measurements, and only the
    explicitly reviewed historical implementation is eligible for reuse.
    """
    from oscar_ascend.optr_artifacts import file_sha256
    from oscar_ascend.rotations import calibration_cache_identity, checkpoint_identity
    if (type(testing) is not bool or set(baseline_hashes) != {"base_artifact_sha256"} or
            not _sha(baseline_hashes["base_artifact_sha256"]) or type(global_heads) is not int or global_heads < 1):
        raise ValueError("recipe3 source baseline hashes/geometry/testing permission is invalid")
    actual_names, actual_dim, fingerprint = checkpoint_identity(config["model"])
    if names != actual_names or D != actual_dim or type(D) is not int:
        raise ValueError("recipe3 source requires actual model FULL layer coverage and dimension")
    legacy_config = {key: value for key, value in config.items() if key != "optr_recipe3_repair"}
    profile = calibration_cache_identity(legacy_config, fingerprint)
    policy = profile["refinement_policy"]
    if config.get("optr_baseline") != "recipe3" or policy.get("policy_version") != LEGACY_POLICY:
        raise ValueError("recipe3 source requires the exact historical recipe3 policy")
    trace_path = Path(trace_manifest).absolute()
    trace = _json(trace_path)
    if trace.get("calibration_profile") != profile:
        if (config.get("optr_recipe3_repair") == "single_head_trust_region_v1" and
                trace.get("calibration_profile") == calibration_cache_identity(config, fingerprint)):
            return None  # Fresh v11 captures have no historical v10 work.
        raise ValueError("recipe3 source trace does not have the historical profile")
    identity = {"profile": profile, **baseline_hashes, "trace_manifest_sha256": file_sha256(trace_path),
                "full_layers": names, "global_kv_heads": global_heads, "head_dim": D,
                "fitting_implementation_sha256": LEGACY_FITTING_SHA256, "test_only": testing}
    _trace(trace_path, profile, identity)
    generation = _generation(trace_path.parent.parent / "fit-checkpoints", identity)
    if generation is None:
        return None
    snapshots = {}
    for name, index in zip(names, _layer_ids(names)):
        key = f"layer-{index}"
        payload, completed_hash = _read_checkpoint(generation / (key + ".json"), key, identity)
        proposal, proposal_hash = _read_checkpoint(generation / (key + "-shared.json"), key + "-shared", identity)
        seed = payload.get("artifact", {}).get("layers", {}).get(name)
        if not isinstance(seed, dict):
            raise ValueError("recipe3 source completed layer has no seed constants")
        _validate_layer(payload, proposal, identity, name, seed, policy)
        snapshots[name] = {"completed_manifest_sha256": completed_hash, "proposal_manifest_sha256": proposal_hash}
    return copy.deepcopy({"schema_version": 1, "source_profile": profile, "policy": policy,
        "identity": identity, "generation": str(generation), "trace_manifest": str(trace_path), "layers": snapshots})


def load_source_layer(source_spec, name, seed, heads, source_policy):
    """Reauthenticate one saved layer and bind it to the caller's real seed."""
    identity = source_spec["identity"]
    profile = source_spec["source_profile"]
    if (source_spec.get("schema_version") != 1 or identity.get("profile") != profile or
            source_policy != source_spec.get("policy") or source_policy != profile.get("refinement_policy") or
            source_policy.get("policy_version") != LEGACY_POLICY or
            identity.get("fitting_implementation_sha256") != LEGACY_FITTING_SHA256 or
            type(identity.get("test_only")) is not bool or type(heads) is not int or heads != identity["global_kv_heads"] or
            set(source_spec.get("layers", {})) != set(identity["full_layers"]) or name not in source_spec["layers"]):
        raise ValueError("recipe3 source specification policy/identity/coverage differs")
    trace_path = Path(source_spec["trace_manifest"])
    _trace(trace_path, profile, identity)
    generation = _generation(trace_path.parent.parent / "fit-checkpoints", identity)
    if generation is None or str(generation) != source_spec["generation"]:
        raise ValueError("recipe3 source generation changed after discovery")
    key = f"layer-{_layer_ids([name])[0]}"
    snapshot = source_spec["layers"][name]
    payload, completed_hash = _read_checkpoint(generation / (key + ".json"), key, identity,
                                              snapshot["completed_manifest_sha256"])
    proposal, proposal_hash = _read_checkpoint(generation / (key + "-shared.json"), key + "-shared", identity,
                                             snapshot["proposal_manifest_sha256"])
    result, record = _validate_layer(payload, proposal, identity, name, seed, source_policy)
    provenance = {"schema_version": 1, "fitting_implementation_sha256": LEGACY_FITTING_SHA256,
        "generation": str(generation), "base_artifact_sha256": identity["base_artifact_sha256"],
        "trace_manifest_sha256": identity["trace_manifest_sha256"], "source_profile": profile,
        "layer": name, "completed_manifest_sha256": completed_hash, "proposal_manifest_sha256": proposal_hash,
        "candidate_count": len(result["candidates"]), "test_used_for_selection": False,
        "independent_test_evidence": "unchanged_recipe3_seed_only_not_rejected_proposals"}
    return result, record, copy.deepcopy(provenance)
