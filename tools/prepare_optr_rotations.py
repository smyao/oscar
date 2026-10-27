"""Archive #86/#97/#148: offline-only, authenticated output-aware refinement.

The normal candidate preparation calls this module; serving loads only final
constants. Native sampling runs in an isolated child and releases its model
before CPU optimization. No runtime/kernel/scheduler implementation is changed.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from oscar_ascend.optr_artifacts import (file_sha256, matrix_digest, nonworse,
    publish_artifact)
from oscar_ascend.optr_policy import OBJECTIVE, optr_policy, validate_refinement_config


def _progress(**fields):
    print(json.dumps({"rotation_prepare": "optr", **fields}), flush=True)


def _capture(config_path, directory):
    from tools.calibrate_model_rotations import calibration_process_env
    subprocess.run([sys.executable, "-m", "tools.calibrate_optr_traces",
                    "--config", str(config_path), "--output-dir", str(directory)],
                   env=calibration_process_env(), check=True)
    return json.loads((directory / "capture-manifest.json").read_text())


def _base_config(config, fingerprint):
    """Keep recipe3's exact numerical route; this is NOT variant='baseline'."""
    from oscar_ascend.rotations import calibration_cache_identity
    base = copy.deepcopy(config)
    for key in ("rotation_refinement", "optr_calibration", "rotation_artifact"):
        base.pop(key, None)
    profile = calibration_cache_identity(base, fingerprint)
    digest = hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]
    root = Path(__file__).resolve().parents[1]
    # Explicit final-artifact overrides must not overwrite/import the seed PT.
    base["rotation_artifact"] = str(root / "artifacts/rotations" / fingerprint /
                                     ("qqt_sst_" + digest) / "calibrated.pt")
    return base


def _prepare_base(config, directory, *, testing, import_pair, base_runner):
    from oscar_ascend.rotations import load_active_artifact
    config_path = directory / "base-config.json"
    child_config = copy.deepcopy(config)
    if not testing:
        # Retain the seed preparer's authenticated recipe2 raw-sample reuse.
        # Its automatic cache decision intentionally excludes explicit paths.
        child_config.pop("rotation_artifact", None)
    config_path.write_text(json.dumps(child_config, indent=2) + "\n")
    if base_runner is not None:
        if not testing:
            raise ValueError("injected base preparation is test-only")
        base_runner(config, config_path)
    elif testing:
        from tools.prepare_rotations import prepare
        prepare(config, config_path=config_path, import_pair=import_pair, testing=True)
    else:
        # Even base validation may initialize torch_npu. Keep that context out
        # of this parent so the subsequent native trace model has all memory.
        command = [sys.executable, "-m", "tools.prepare_rotations", "--config", str(config_path)]
        if import_pair:
            command += ["--import-k", str(import_pair[0]), "--import-v", str(import_pair[1])]
        environment = dict(os.environ)
        environment.pop("OSCAR_ROTATION_ARTIFACT", None)
        subprocess.run(command, check=True, env=environment)
    artifact = load_active_artifact(config, device="cpu", testing=True)
    if artifact.get("test_only") and not testing:
        raise ValueError("test-only base artifact cannot enter production")
    return artifact


def _metrics(constants, samples, policy):
    """Check both the fitting partition and native-captured frame partition."""
    from oscar_ascend.optr_traces import aggregate_layer_error
    rows = []
    for partition in policy["validation_partitions"]:
        metric = aggregate_layer_error(constants, samples, {**policy, "replay_partition": partition})
        for row in metric["requests"]:
            rows.append({**row, "request_id": row["request_id"] + ":" + partition,
                         "partition": partition})
    error = sum(row["squared_error"] for row in rows)
    energy = sum(row["teacher_energy"] for row in rows)
    return {"squared_error": error, "teacher_energy": energy,
            "normalized_error": error / max(energy, 1e-20), "requests": rows}


def refine_layer(name, baseline, samples, policy, *, fitter=None):
    """Per-head fitting, then all-TP output selection, then untouched test."""
    from oscar_ascend.optr_calibration import fit_head
    fitter = fit_head if fitter is None else fitter
    trial = {key: value.clone() if hasattr(value, "clone") else value
             for key, value in baseline.items()}
    decisions = []
    for head in range(baseline["Rk"].shape[0]):
        _progress(layer=name, head=head, phase="fitting")
        rk, rv, decision = fitter(baseline["Rk"][head], baseline["Rv"][head],
            baseline["k_mean"][head], samples["train"][head], samples["validation"][head], policy=policy)
        trial["Rk"][head], trial["Rv"][head] = rk, rv
        decisions.append(decision)
    before = _metrics(baseline, samples["validation"], policy)
    after = _metrics(trial, samples["validation"], policy)
    accepted = nonworse(before, after) and after["squared_error"] < before["squared_error"]
    selected = trial if accepted else baseline
    if not accepted:
        after = before
    # This dataset never chooses a matrix/step/hyperparameter. A failure is
    # reported and blocks publication; it cannot silently reselect on test.
    test_before = _metrics(baseline, samples["test"], policy)
    test_after = _metrics(selected, samples["test"], policy) if accepted else test_before
    if not nonworse(test_before, test_after):
        raise ValueError(f"OptR {name}: independent test regressed; artifact not published")
    record = {"accepted": accepted, "head_decisions": decisions,
              "base_constants_sha256": matrix_digest({name: baseline}),
              "selected_constants_sha256": matrix_digest({name: selected}),
              "validation": {"baseline": before, "selected": after},
              "test": {"baseline": test_before, "selected": test_after}}
    _progress(layer=name, phase="checked", accepted=accepted,
              validation_before=before["normalized_error"], validation_after=after["normalized_error"],
              independent_test_before=test_before["normalized_error"],
              independent_test_after=test_after["normalized_error"])
    return selected, record


def prepare(config, *, config_path=None, output=None, import_pair=None,
            capture_runner=None, testing=False, base_runner=None, fitter=None):
    import torch
    from oscar_ascend.rotations import (checkpoint_identity, rotation_artifact_path,
        calibration_cache_identity, load_active_artifact, validate_artifact)
    from oscar_ascend.rotation_calibration import OFFICIAL_COMMIT
    from oscar_ascend.optr_traces import load_trace_manifest, load_layer_samples
    validate_refinement_config(config)
    policy = optr_policy(config)
    names, dim, fingerprint = checkpoint_identity(config["model"])
    destination = Path(output).expanduser().resolve() if output else rotation_artifact_path(config, fingerprint)
    profile = calibration_cache_identity(config, fingerprint)
    if destination.is_file() and not (destination.parent / ".optr.pending.json").exists():
        artifact = load_active_artifact(config, path=destination, device="cpu", testing=True)
        if artifact.get("test_only") and not testing:
            raise ValueError("test-only refined artifact cannot enter production")
        return {"artifact": str(destination), "source": "optr_authenticated_cache", "reused": True,
                "recipe_version": 5, "accepted_layers": artifact["refinement_certificate"]["accepted_layers"],
                "quality_acceptance": "not_run", "performance": "not_measured"}
    if not testing and (base_runner is not None or fitter is not None or capture_runner is not None):
        raise ValueError("injected OptR preparation hooks are test-only")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _progress(phase="preparing_seed", artifact=str(destination), recipe_version=5,
              online_operations="unchanged", first_preparation="native_capture_and_cpu_optimization")
    base_config = _base_config(config, fingerprint)
    if Path(base_config["rotation_artifact"]).resolve() == destination:
        raise ValueError("refined destination must not overwrite the recipe3 seed")
    baseline = _prepare_base(base_config, destination.parent, testing=testing,
                             import_pair=import_pair, base_runner=base_runner)
    base_hash = file_sha256(base_config["rotation_artifact"])
    trace_dir = destination.parent / "native-traces"
    trace_dir.mkdir(exist_ok=True)
    trace_manifest = trace_dir / "capture-manifest.json"
    # The selected config is serialized, never an earlier raw --config file.
    capture_config = destination.parent / "refinement-config.json"
    capture_config.write_text(json.dumps(config, indent=2) + "\n")
    if not trace_manifest.is_file():
        _progress(phase="native_capture", lengths=policy["capture_lengths"])
        # An interrupted generation remains available for diagnostics and
        # can never prevent a later invocation from creating a fresh capture.
        generation = Path(tempfile.mkdtemp(prefix="generation-", dir=trace_dir))
        captured = (capture_runner or _capture)(capture_config, generation)
        from tools.phase import atomic_json
        atomic_json(trace_manifest, captured)
    manifest = load_trace_manifest(trace_manifest, config, names, dim, fingerprint, testing=testing)
    artifact = {"format_version": 2, "source_grouping": "layer_kv_head",
        "rotation_variant": "perhead_center", "global_kv_heads": baseline["global_kv_heads"],
        "head_dim": dim, "model_fingerprint": fingerprint, "pr_fingerprint": OFFICIAL_COMMIT,
        "objective": OBJECTIVE, "calibrated": True, "accumulation_dtype": "float32",
        "test_only": testing, "calibration_profile": profile, "layers": {}}
    certificate = {"schema_version": 1, "policy": policy,
        "metric": "all_tp_heads_bf16_gate_fp32_wo_output",
        "arithmetic_evidence": "cpu_deployment_reference_not_npu_bitwise",
        "model_quality": "not_measured", "performance": "not_measured",
        "base_artifact_sha256": base_hash, "trace_manifest_sha256": file_sha256(trace_manifest),
        "layers": {}}
    for name in names:
        samples = load_layer_samples(manifest, baseline["layers"][name]["layer_id"])
        selected, record = refine_layer(name, baseline["layers"][name], samples, policy, fitter=fitter)
        artifact["layers"][name] = selected
        certificate["layers"][name] = record
        del samples
    certificate["accepted_layers"] = sorted(n for n, row in certificate["layers"].items() if row["accepted"])
    certificate["matrices_sha256"] = matrix_digest(artifact["layers"])
    artifact["refinement_certificate"] = certificate
    # CPU math validation here does not permit a CPU test artifact in serving.
    # The existing runtime performs its normal NPU validation on final load.
    validate_artifact(artifact, layer_names=names, head_dim=dim, model_fingerprint=fingerprint,
                      pr_fingerprint=OFFICIAL_COMMIT, require_calibrated=True, device="cpu", testing=True)
    publish_artifact(artifact, destination)
    load_active_artifact(config, path=destination, device="cpu", testing=True)
    return {"artifact": str(destination), "source": "optr_native_output_refinement", "reused": False,
            "recipe_version": 5, "accepted_layers": certificate["accepted_layers"],
            "quality_acceptance": "not_run", "performance": "not_measured",
            "model_forward_device": "cpu_test" if testing else "npu",
            "optimization_device": "cpu_offline", "online_operations": "unchanged"}
