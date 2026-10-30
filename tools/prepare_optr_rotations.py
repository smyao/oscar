"""Archive #86/#97/#148/#170/#172/#173: offline output-aware refinement.

The fixed candidate calls this module. User 2026-10-28 authorizes experimental
constants after validation, with independent test recorded, never reselected.
Native sampling runs in an isolated
child and releases its model before NPU proxy fitting and CPU verification. No runtime/kernel/
scheduler implementation is changed; candidate defaults remain separate.
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
    publish_artifact, experimental_certificate, artifact_test_status)
from oscar_ascend.optr_policy import (OBJECTIVE, optr_policy, validate_refinement_config,
    EXPERIMENTAL_PUBLICATION_POLICY)


def _progress(**fields):
    print("[oscar] OPTR_PROGRESS " + json.dumps({"rotation_prepare": "optr", **fields}), flush=True)


def _capture(config_path, directory):
    from tools.calibrate_model_rotations import calibration_process_env
    from tools.rotation_preparation_limits import preparation_progress_timeout
    timeout = preparation_progress_timeout(json.loads(Path(config_path).read_text()))
    subprocess.run([sys.executable, "-m", "tools.calibrate_optr_traces",
                    "--config", str(config_path), "--output-dir", str(directory)],
                   env=calibration_process_env(), check=True, timeout=timeout)
    return json.loads((directory / "capture-manifest.json").read_text())


def _base_config(config, fingerprint):
    """Source-bound shared OSCAR seed; preserve the existing serving layout."""
    from oscar_ascend.rotations import calibration_cache_identity
    from oscar_ascend.optr_oscar_seed import SEED_RECIPE
    base = copy.deepcopy(config)
    for key in ("rotation_refinement", "optr_calibration", "rotation_artifact"):
        base.pop(key, None)
    base["rotation_seed"] = SEED_RECIPE
    profile = calibration_cache_identity(base, fingerprint)
    digest = hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]
    root = Path(__file__).resolve().parents[1]
    # Explicit final-artifact overrides must not overwrite/import the seed PT.
    base["rotation_artifact"] = str(root / "artifacts/rotations" / fingerprint /
                                     ("oscar_base_" + digest) / "seed.pt")
    return base


def _prepare_base(config, directory, *, testing, import_pair, base_runner):
    from tools.prepare_optr_seed import load_seed
    if import_pair is not None or config.get("rotation_k_path") or config.get("rotation_v_path"):
        raise ValueError("OptR source OSCAR seed must use authenticated native source statistics; old recipe3 imports are incompatible")
    config_path = directory / "base-config.json"
    child_config = copy.deepcopy(config)
    config_path.write_text(json.dumps(child_config, indent=2) + "\n")
    if base_runner is not None:
        if not testing:
            raise ValueError("injected base preparation is test-only")
        base_runner(config, config_path)
    elif testing:
        from tools.prepare_optr_seed import prepare
        prepare(config, config_path=config_path, testing=True)
    else:
        # Even base validation may initialize torch_npu. Keep that context out
        # of this parent so the subsequent native trace model has all memory.
        command = [sys.executable, "-m", "tools.prepare_optr_seed", "--config", str(config_path)]
        environment = dict(os.environ)
        environment.pop("OSCAR_ROTATION_ARTIFACT", None)
        from tools.rotation_preparation_limits import preparation_progress_timeout
        subprocess.run(command, check=True, env=environment, timeout=preparation_progress_timeout(config))
    artifact = load_seed(config, testing=testing)
    if artifact.get("test_only") and not testing:
        raise ValueError("test-only base artifact cannot enter production")
    return artifact


def _single_partition_equivalent(samples, policy):
    """Only captured, valid one-split geometry can reuse fixed-one replay.

    The first real replay still validates every tensor and policy. Missing
    captured-only metadata must reach the captured replay's existing error,
    never be hidden by this arithmetic equivalence shortcut.
    """
    import torch
    if type(policy.get("history_splits", 1)) is not int or policy.get("history_splits", 1) != 1:
        return False
    for head in samples:
        for sample in head:
            positions = sample.get("positions")
            ends = sample.get("current_ends")
            splits = sample.get("history_splits")
            if (not isinstance(positions, torch.Tensor) or positions.device.type != "cpu"
                    or positions.dtype != torch.int64 or positions.ndim != 1 or not positions.numel()):
                return False
            for vector in (ends, splits):
                if (not isinstance(vector, torch.Tensor) or vector.device.type != "cpu"
                        or vector.dtype != torch.int64 or vector.shape != positions.shape):
                    return False
            if not bool((splits == 1).all()):
                return False
    return True


def _metrics(constants, samples, policy, *, progress=None):
    """Check both the fitting partition and native-captured frame partition."""
    from oscar_ascend.optr_traces import aggregate_layer_error
    rows = {view: [] for view in policy.get("validation_projection_views", ["fp32"])}
    single = None
    for partition in policy["validation_partitions"]:
        if partition == "captured_runtime_frames" and single is not None and _single_partition_equivalent(samples, policy):
            # _replay differs only in split_counts, which are identical here.
            # Preserve separate mutable evidence and both partition identities.
            metric = copy.deepcopy(single)
        else:
            metric = aggregate_layer_error(constants, samples, {**policy, "replay_partition": partition})
        if partition == "single_split":
            single = metric
        if progress:
            progress(event="partition_checked", partition=partition)
        views = {"fp32": metric, **metric.get("projection_checks", {})}
        if set(views) != set(rows):
            raise ValueError("OptR replay omitted required projection checks")
        for view, values in views.items():
            rows[view].extend({**row, "request_id": row["request_id"] + ":" + partition,
                              "partition": partition} for row in values["requests"])
    from oscar_ascend.optr_traces import _aggregate_requests
    result = _aggregate_requests(rows.pop("fp32"))
    if rows:
        result["projection_checks"] = {view: _aggregate_requests(values) for view, values in rows.items()}
    return result


def _validate_shared_result(result, baseline, *, policy=None, require_candidates=False):
    import torch
    from oscar_ascend.optr_shared import validate_shared_pair
    if not isinstance(result, dict) or not isinstance(result.get("decision"), dict):
        raise ValueError("invalid completed OptR shared checkpoint")
    validate_shared_pair(result.get("Rk"), result.get("Rv"), baseline["k_mean"])
    if result["decision"].get("accepted") is False and any(not torch.equal(
            result[key], baseline[key]) for key in ("Rk", "Rv")):
        raise ValueError("rejected OptR shared checkpoint changed baseline matrices")
    if require_candidates or result.get("value_candidates") is not None:
        from oscar_ascend.optr_candidates import validate_candidates
        validate_candidates(result.get("value_candidates"), result, baseline, policy)
    return result["Rk"], result["Rv"], result["decision"]


class IndependentTestRegression(ValueError):
    """Test regression evidence whose mandatory reporting could not complete."""

    def __init__(self, message, report):
        super().__init__(message)
        self.report = report
        self.reporting_errors = ()


def _report_independent_test_failure(name, layer_id, record, policy, checkpoints, emit):
    """Archive #148/#170/#173: retain the failed test; never use it to reselect.

    This filename is deliberately not a completed layer checkpoint. The
    original scalar values are serialized without rounding or conversion to
    task accuracy. Persist before terminal/callback output. Reporting I/O must
    complete before an experimental artifact can publish.
    """
    baseline, selected = record["test"]["baseline"], record["test"]["selected"]
    compared = ("squared_error", "normalized_error")
    scalar_keys = ("squared_error", "teacher_energy", "normalized_error")
    total = {"baseline": {key: baseline[key] for key in scalar_keys},
             "selected": {key: selected[key] for key in scalar_keys},
             "regressed_metrics": [key for key in compared if selected[key] > baseline[key]]}
    before = {row["request_id"]: row for row in baseline["requests"]}
    regressions = []
    def query_regressions(previous, selected):
        old_queries = {query["position"]: query for query in previous.get("queries", [])}
        return [{"position": query["position"], "baseline": old_queries[query["position"]],
                 "selected": query, "regressed_metrics": [key for key in compared
                     if query[key] > old_queries[query["position"]][key]]}
                for query in selected.get("queries", [])
                if any(query[key] > old_queries[query["position"]][key] for key in compared)]
    for row in selected["requests"]:
        previous = before[row["request_id"]]
        regressed = [key for key in compared if row[key] > previous[key]]
        previous_frames = {frame["frame_id"]: frame for frame in previous.get("frames", [])}
        frame_regressions = []
        for frame in row.get("frames", []):
            old = previous_frames[frame["frame_id"]]
            fields = [key for key in compared if frame[key] > old[key]]
            terminal_fields = [key for key in compared if frame.get("terminal_query", {}).get(key, 0) >
                               old.get("terminal_query", {}).get(key, 0)]
            queries = query_regressions(old, frame)
            if fields or terminal_fields or queries:
                frame_regressions.append({"frame_id": frame["frame_id"], "baseline": old,
                                          "selected": frame, "regressed_metrics": fields,
                                          "regressed_terminal_metrics": terminal_fields,
                                          "regressed_queries": queries})
        if regressed or frame_regressions:
            regressions.append({"request_id": row["request_id"], "partition": row.get("partition"),
                                "baseline": previous, "selected": row,
                                "regressed_metrics": regressed, "regressed_frames": frame_regressions})
    projection_regressions = []
    for view, view_selected in selected.get("projection_checks", {}).items():
        previous = baseline["projection_checks"][view]
        if nonworse(previous, view_selected, require_queries=True):
            continue
        previous_requests = {row["request_id"]: row for row in previous["requests"]}
        queries = []
        for row in view_selected["requests"]:
            for old, frame in zip(previous_requests[row["request_id"]]["frames"], row["frames"]):
                queries.extend({"request_id": row["request_id"], "frame_id": frame["frame_id"], **query}
                               for query in query_regressions(old, frame))
        projection_regressions.append({"projection_view": view,
            "baseline": {key: previous[key] for key in scalar_keys},
            "selected": {key: view_selected[key] for key in scalar_keys}, "regressed_queries": queries})
    report = {"schema_version": 1, "status": "independent_test_failed", "layer": name,
        "layer_id": layer_id, "identity": copy.deepcopy(checkpoints.identity) if checkpoints else None,
        "policy": copy.deepcopy(policy), "metric": "all_tp_heads_bf16_gate_fp32_wo_output",
        "arithmetic_evidence": "cpu_deployment_reference_not_npu_bitwise",
        "model_quality": "not_measured", "performance": "not_measured",
        "artifact_publish_allowed": True, "test_used_for_selection": False,
        "experimental": True, "publication_policy": EXPERIMENTAL_PUBLICATION_POLICY,
        "test_policy_unchanged": False,
        "record": record, "total": total, "regressed_requests": regressions,
        "regressed_projection_views": projection_regressions}
    message = f"OptR {name}: independent test report failed; artifact not published"
    reporting_errors = []
    persistence_event = None
    if checkpoints:
        from tools.phase import atomic_json
        path = checkpoints.root / f"layer-{layer_id}-independent-test-failed.json"
        report["report_path"] = str(path)
        try:
            atomic_json(path, report)
        except Exception as error:
            reporting_errors.append(error)
            persistence_event = {"layer": name, "phase": "independent_test_failure_report_error",
                                 "path": str(path), "error": str(error)}
        else:
            persistence_event = {"layer": name, "phase": "independent_test_failure_saved", "path": str(path)}
    events = [{"layer": name, "phase": "independent_test_failed", "metric": report["metric"],
               "baseline": total["baseline"], "selected": total["selected"],
               "regressed_total_metrics": total["regressed_metrics"], "regressed_requests": len(regressions),
               "regressed_projection_views": [row["projection_view"] for row in projection_regressions],
               "artifact_publish_allowed": True, "experimental": True,
               "publication_policy": EXPERIMENTAL_PUBLICATION_POLICY, "test_used_for_selection": False}]
    events.extend({"layer": name, "phase": "independent_test_request_regressed", **row} for row in regressions)
    events.extend({"layer": name, "phase": "independent_test_projection_regressed", **row}
                  for row in projection_regressions)
    if persistence_event is not None:
        events.append(persistence_event)
    for event in events:
        try:
            emit(**event)
        except Exception as error:
            reporting_errors.append(error)
            break  # Required diagnostic output failed; do not publish silently.
    if reporting_errors:
        failure = IndependentTestRegression(message, report)
        failure.reporting_errors = tuple(reporting_errors)
        # Preserve the first original I/O exception as the cause; retain any
        # secondary output failure too, including on Python 3.10 (no add_note).
        if hasattr(failure, "add_note"):
            for error in reporting_errors[1:]:
                failure.add_note(f"Additional diagnostic failure: {type(error).__name__}: {error}")
        raise failure from reporting_errors[0]
    return report


def refine_layer(name, baseline, samples, policy, *, fitter=None, checkpoints=None, progress=None,
                 fit_device="cpu"):
    """Search one shared pair, accept it as a whole layer, then record test."""
    from oscar_ascend.optr_shared import fit_shared, validate_shared_pair
    validate_shared_pair(baseline["Rk"], baseline["Rv"], baseline["k_mean"])
    real_fitter = fitter is None
    fitter = fit_shared if real_fitter else fitter
    candidate_search = "deployment_candidate_selection" in policy
    collector = None
    if real_fitter and candidate_search:
        from oscar_ascend.optr_candidates import ValueCandidates
        collector = ValueCandidates(policy)
    emit = _progress if progress is None else progress
    owner = f"layer-{baseline['layer_id']}-shared"
    emit(layer=name, phase="fitting", fitted_grouping="layer")
    saved = checkpoints.load(owner) if checkpoints else None
    if saved is None:
        options = {"policy": policy}
        if real_fitter:
            options["progress"] = lambda row: emit(layer=name, **row)
            options["device"] = fit_device
            if collector is not None:
                options["value_candidate_callback"] = collector.observe
        rk, rv, decision = fitter(baseline["Rk"], baseline["Rv"], baseline["k_mean"],
            samples["train"], samples["validation"], **options)
        saved = {"Rk": rk.detach().cpu().contiguous(), "Rv": rv.detach().cpu().contiguous(),
                 "decision": decision,
                 "value_candidates": collector.payload(rk) if collector is not None else None}
        _validate_shared_result(saved, baseline, policy=policy,
                                require_candidates=real_fitter and candidate_search)
        if checkpoints:
            checkpoints.save(owner, saved)
        emit(layer=name, phase="shared_fit_saved")
    else:
        rk, rv, decision = _validate_shared_result(saved, baseline, policy=policy,
            require_candidates=real_fitter and candidate_search)
        emit(layer=name, phase="shared_fit_resumed")
    trial = {**baseline, "Rk": rk, "Rv": rv}
    def measure(constants, split, candidate):
        emit(layer=name, phase="joint_output_check", split=split, candidate=candidate)
        options = {} if progress is None else {"progress": lambda **row: emit(
            layer=name, phase="joint_output_check", split=split, candidate=candidate, **row)}
        return _metrics(constants, samples[split], policy, **options)
    before = measure(baseline, "validation", "baseline")
    proposal = measure(trial, "validation", "refined")
    strict = policy.get("fitted_grouping") == "layer"
    projection = "validation_projection_views" in policy
    acceptance = {"require_frames": strict, "require_terminal": strict,
                  "require_queries": projection, "require_projection": projection}
    accepted = nonworse(before, proposal, **acceptance) and (
        proposal["squared_error"] < before["squared_error"])
    selected, after = (trial, proposal) if accepted else (baseline, before)
    selection = {"method": "shared_layer_projection_validation_v2" if projection else "shared_layer_validation_v1", "accepted": accepted,
                 "test_used_for_selection": False, "proposal": proposal}
    if candidate_search:
        from oscar_ascend.optr_candidates import alternatives
        selection.update(method="shared_existing_v_checkpoints_v1",
            candidate_order="primary_then_proxy_loss_then_step_first_nonworse",
            max_candidates=policy["max_deployment_candidates"], selected_candidate="primary" if accepted else "seed",
            evaluated_candidates=[{"candidate": "primary", "accepted": accepted,
                "constants_sha256": matrix_digest({name: trial}), "validation": proposal}])
        # A primary passing the unchanged strict gate needs no alternatives.
        # Only a rejected primary opens this bounded checkpoint search.
        # K stays fixed throughout; no test data is touched in this loop.
        if not accepted and saved.get("value_candidates") is not None:
            import torch
            for candidate in alternatives(saved["value_candidates"]["values"], rv):
                candidate_id = f"v_step_{candidate['step']}"
                value = candidate["Rv"].expand_as(rv).clone().contiguous()
                if torch.equal(rk, baseline["Rk"]) and torch.equal(value, baseline["Rv"]):
                    continue  # The unchanged seed cannot strictly improve itself.
                constants = {**baseline, "Rk": rk, "Rv": value}
                measured = measure(constants, "validation", candidate_id)
                accepted = nonworse(before, measured, **acceptance) and (
                    measured["squared_error"] < before["squared_error"])
                selection["evaluated_candidates"].append({"candidate": candidate_id,
                    "step": candidate["step"], "proxy_loss": candidate["proxy_loss"],
                    "constants_sha256": matrix_digest({name: constants}),
                    "accepted": accepted, "validation": measured})
                emit(layer=name, phase="joint_output_check", split="validation",
                     event="existing_v_candidate_checked", candidate=candidate_id, accepted=accepted)
                if accepted:
                    selected, after = constants, measured
                    selection.update(accepted=True, selected_candidate=candidate_id)
                    break
    emit(layer=name, phase="joint_output_check", split="validation", event="shared_layer_selected",
         accepted=accepted, fitted_grouping="layer")
    # This dataset never chooses a matrix/step/hyperparameter. User 2026-10-28
    # permits experimental publication with failed test explicitly recorded.
    test_before = measure(baseline, "test", "baseline")
    test_after = measure(selected, "test", "selected") if accepted else test_before
    record = {"accepted": accepted, "shared_decision": decision,
              "base_constants_sha256": matrix_digest({name: baseline}),
              "selected_constants_sha256": matrix_digest({name: selected}),
              "validation": {"baseline": before, "selected": after},
              "test": {"baseline": test_before, "selected": test_after}}
    record["shared_layer_selection"] = selection
    test_passed = nonworse(test_before, test_after, **acceptance)
    if not test_passed:
        _report_independent_test_failure(name, baseline["layer_id"], record, policy, checkpoints, emit)
    emit(layer=name, phase="checked", accepted=accepted,
              experimental=True, independent_test="passed" if test_passed else "failed",
              validation_before=before["normalized_error"], validation_after=after["normalized_error"],
              independent_test_before=test_before["normalized_error"],
              independent_test_after=test_after["normalized_error"])
    return selected, record


def _existing_trace_manifest(destination, profile, config):
    """Reuse only captures with identical train/held-out coverage and source."""
    from oscar_ascend.optr_policy import compatible_capture_profiles
    current = destination.parent / "native-traces" / "capture-manifest.json"
    if current.is_file():
        return current
    for legacy in compatible_capture_profiles(profile, config):
        digest = hashlib.sha256(json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]
        old = destination.parent.parent / ("optr_" + digest) / "native-traces" / "capture-manifest.json"
        if old.is_file():
            return old
    return current


def _layer_artifact(template, certificate, name, selected, record):
    return {**template, "layers": {name: selected}, "refinement_certificate": experimental_certificate({
        **certificate, "layers": {name: record}, "accepted_layers": [name] if record["accepted"] else [],
        "matrices_sha256": matrix_digest({name: selected})})}


def _validate_layer_checkpoint(saved, template, certificate, name, baseline):
    import torch
    from oscar_ascend.rotations import validate_artifact
    from oscar_ascend.optr_artifacts import validate_refinement_metadata
    artifact = saved.get("artifact")
    if not isinstance(artifact, dict) or artifact.get("test_only") != template["test_only"]:
        raise ValueError("OptR completed layer checkpoint identity differs")
    validate_refinement_metadata(artifact, expected_profile=template["calibration_profile"])
    got = validate_artifact(artifact, layer_names=[name], head_dim=template["head_dim"],
        model_fingerprint=template["model_fingerprint"], pr_fingerprint=template["pr_fingerprint"],
        device="cpu", testing=True, require_calibrated=True)
    record = got["refinement_certificate"]["layers"][name]
    for key in ("base_artifact_sha256", "trace_manifest_sha256"):
        if got["refinement_certificate"][key] != certificate[key]:
            raise ValueError("OptR completed layer checkpoint source differs")
    if (record["base_constants_sha256"] != matrix_digest({name: baseline}) or
            not torch.equal(got["layers"][name]["k_mean"], baseline["k_mean"])):
        raise ValueError("OptR completed layer checkpoint seed/mean differs")
    return got["layers"][name], record


def _parallel_layers(config, names, baseline, artifact, certificate, base_config,
                     trace_manifest, checkpoints, watchdog, *, testing=False):
    from tools.optr_parallel import run_layer_workers
    from tools.rotation_preparation_limits import preparation_progress_timeout
    pending = []

    def completed(name):
        seed = baseline["layers"][name]
        saved = checkpoints.load(f"layer-{seed['layer_id']}")
        if saved is None:
            raise ValueError(f"OptR worker did not publish a completed layer: {name}")
        selected, record = _validate_layer_checkpoint(saved, artifact, certificate, name, seed)
        artifact["layers"][name] = selected
        certificate["layers"][name] = record
        watchdog.progress(layer=name, phase="layer_checked_collected")

    for name in names:
        seed = baseline["layers"][name]
        if checkpoints.load(f"layer-{seed['layer_id']}") is None:
            pending.append(name)
        else:
            completed(name)
            watchdog.progress(layer=name, phase="layer_checked_resumed")
    # Keep tensors out of spawn arguments and IPC. Workers load the verified
    # seed once, then only their assigned layer's histories at any one time.
    spec = {"config": config, "base_config": base_config, "testing": testing,
            "names": names, "trace_manifest": str(trace_manifest),
            "template": {**artifact, "layers": {}}, "certificate": {**certificate, "layers": {}},
            "checkpoint_directory": str(checkpoints.root.parent), "checkpoint_identity": checkpoints.identity}
    run_layer_workers(spec, pending, progress=watchdog.progress, completed=completed,
                      timeout=preparation_progress_timeout(config))
    # Completion order cannot change the serialized layer order or reductions.
    artifact["layers"] = {name: artifact["layers"][name] for name in names}
    certificate["layers"] = {name: certificate["layers"][name] for name in names}


def prepare(config, *, config_path=None, output=None, import_pair=None,
            capture_runner=None, testing=False, base_runner=None, fitter=None, parallel_testing=False):
    import torch
    from oscar_ascend.rotations import (checkpoint_identity, rotation_artifact_path,
        calibration_cache_identity, load_active_artifact, validate_artifact)
    from oscar_ascend.rotation_calibration import OFFICIAL_COMMIT
    from oscar_ascend.optr_traces import load_trace_manifest, load_layer_samples
    validate_refinement_config(config)
    if parallel_testing and (not testing or fitter is not None):
        raise ValueError("parallel CPU preparation is test-only and requires the real fitter")
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
                **artifact_test_status(artifact),
                "quality_acceptance": "not_run", "performance": "not_measured"}
    if not testing and (base_runner is not None or fitter is not None or capture_runner is not None):
        raise ValueError("injected OptR preparation hooks are test-only")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _progress(phase="preparing_seed", artifact=str(destination), recipe_version=5,
              online_operations="unchanged", first_preparation="native_capture_and_npu_sequential_optimization")
    base_config = _base_config(config, fingerprint)
    if Path(base_config["rotation_artifact"]).resolve() == destination:
        raise ValueError("refined destination must not overwrite the source OSCAR seed")
    baseline = _prepare_base(base_config, destination.parent, testing=testing,
                             import_pair=import_pair, base_runner=base_runner)
    base_hash = file_sha256(base_config["rotation_artifact"])
    trace_dir = destination.parent / "native-traces"
    trace_dir.mkdir(exist_ok=True)
    trace_manifest = _existing_trace_manifest(destination, profile, config)
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
    if trace_manifest.parent != trace_dir:
        _progress(phase="native_trace_resumed", manifest=str(trace_manifest),
                  fitting="new_sequential_objective_old_fitted_constants_not_reused")
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
    from oscar_ascend.optr_checkpoint import FitCheckpoints, fitting_implementation_digest
    from tools.optr_progress import ProgressWatchdog
    from tools.rotation_preparation_limits import preparation_progress_timeout
    checkpoints = FitCheckpoints(destination.parent / "fit-checkpoints", {
        "profile": profile, "base_artifact_sha256": base_hash,
        "trace_manifest_sha256": certificate["trace_manifest_sha256"],
        "full_layers": names, "global_kv_heads": baseline["global_kv_heads"], "head_dim": dim,
        "fitting_implementation_sha256": fitting_implementation_digest(), "test_only": testing})
    _progress(phase="fit_plan", layers=len(names), heads_per_layer=baseline["global_kv_heads"],
              steps_per_phase=policy["steps"], progress_timeout_seconds=preparation_progress_timeout(config),
              checkpoints=str(checkpoints.root), fitting_objective=policy["fitting_objective"],
              fitting_device="cpu_test" if testing else "selected_npus_per_layer", matrix_exp_device="cpu",
              deployment_validation_device="cpu", fitted_grouping=policy["fitted_grouping"])
    fit_device = None
    with ProgressWatchdog(preparation_progress_timeout(config), output=_progress) as watchdog:
        if not testing or parallel_testing:
            _parallel_layers(config, names, baseline, artifact, certificate, base_config,
                             trace_manifest, checkpoints, watchdog, testing=testing)
        for layer_index, name in enumerate(names if testing and not parallel_testing else []):
            seed = baseline["layers"][name]
            owner = f"layer-{seed['layer_id']}"
            watchdog.progress(layer=name, layer_index=layer_index + 1, layers=len(names), phase="layer_start")
            saved = checkpoints.load(owner)
            if saved is None:
                samples = load_layer_samples(manifest, seed["layer_id"])
                watchdog.progress(layer=name, phase="layer_traces_verified")
                if fit_device is None:
                    from oscar_ascend.optr_sequential import fitting_device
                    fit_device = fitting_device(config, testing=testing)
                selected, record = refine_layer(name, seed, samples, policy, fitter=fitter,
                    checkpoints=checkpoints, progress=watchdog.progress, fit_device=fit_device)
                saved = {"artifact": _layer_artifact(artifact, certificate, name, selected, record)}
                _validate_layer_checkpoint(saved, artifact, certificate, name, seed)
                checkpoints.save(owner, saved)
                del samples
                watchdog.progress(layer=name, phase="layer_checked_saved")
            else:
                selected, record = _validate_layer_checkpoint(saved, artifact, certificate, name, seed)
                watchdog.progress(layer=name, phase="layer_checked_resumed")
            artifact["layers"][name] = selected
            certificate["layers"][name] = record
    certificate["accepted_layers"] = sorted(n for n, row in certificate["layers"].items() if row["accepted"])
    certificate["matrices_sha256"] = matrix_digest(artifact["layers"])
    artifact["refinement_certificate"] = experimental_certificate(certificate)
    # CPU math validation here does not permit a CPU test artifact in serving.
    # The existing runtime performs its normal NPU validation on final load.
    validate_artifact(artifact, layer_names=names, head_dim=dim, model_fingerprint=fingerprint,
                      pr_fingerprint=OFFICIAL_COMMIT, require_calibrated=True, device="cpu", testing=True)
    publish_artifact(artifact, destination)
    load_active_artifact(config, path=destination, device="cpu", testing=True)
    return {"artifact": str(destination), "source": "optr_native_output_refinement", "reused": False,
            "recipe_version": 5, "accepted_layers": certificate["accepted_layers"],
            **artifact_test_status(artifact),
            "quality_acceptance": "not_run", "performance": "not_measured",
            "model_forward_device": "cpu_test" if testing else "npu",
            "optimization_device": "cpu_test" if testing else "parallel_npu_fp32_cpu_matrix_exp",
            "deployment_validation_device": "cpu", "online_operations": "unchanged"}
