"""Archive #86/#97/#170/#172/#174/#175/#176/#177: explain saved OptR rejection evidence.

With --config, read authenticated completed checkpoints. With --saved-report,
summarize existing diagnostic JSON without reauthenticating checkpoints.
Neither mode runs an optimizer/replay, initializes NPU, or changes acceptance,
selection or publication.
This module deliberately stays outside the fitting implementation digest so
adding diagnostics cannot invalidate the costly work that it explains.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


_ACCEPTANCE = dict(require_frames=True, require_terminal=True,
                   require_queries=True, require_projection=True)
_FIELDS = ("squared_error", "normalized_error")
_TOP_FAILURES = 6


def _change(before, after):
    return {"baseline": before, "candidate": after, "delta": after - before,
            "relative_delta": (after - before) / before if before else None,
            "positive_from_zero": before == 0 and after > 0}


def metric_rejections(before, after):
    """Explain the existing exact gate without inventing a tolerance.

    Counts include each projection/partition. Terminal checks duplicate their
    final query by design and are reported separately, never summed as queries.
    """
    from oscar_ascend.optr_artifacts import nonworse
    from oscar_ascend.optr_policy import PROJECTION_VIEWS
    passed = nonworse(before, after, **_ACCEPTANCE)
    counts = {scope: 0 for scope in ("total", "request", "frame", "query", "terminal")}
    stages = {stage: {scope: 0 for scope in ("frame", "query", "terminal")}
              for stage in ("first_prefill", "cached_context")}
    failures, views = [], {}

    def check(previous, selected, view, scope, **location):
        fields = [key for key in _FIELDS if selected[key] > previous[key]]
        if not fields:
            return
        counts[scope] += 1
        if scope in stages["first_prefill"]:
            stages["first_prefill" if location["context"] == 0 else "cached_context"][scope] += 1
        failures.append({"projection_view": view, "scope": scope, **location,
                         "regressed_metrics": fields,
                         **{key: _change(previous[key], selected[key]) for key in _FIELDS}})

    for view in PROJECTION_VIEWS:
        left = before if view == "fp32" else before["projection_checks"][view]
        right = after if view == "fp32" else after["projection_checks"][view]
        # The FP32 root also carries the two BF16 views. Compare each view
        # alone here; the original top-level call still enforces every view.
        left = {key: value for key, value in left.items() if key != "projection_checks"}
        right = {key: value for key, value in right.items() if key != "projection_checks"}
        view_passed = nonworse(left, right, require_frames=True,
                             require_terminal=True, require_queries=True)
        views[view] = {"nonworse": view_passed,
                       **{key: _change(left[key], right[key]) for key in _FIELDS}}
        check(left, right, view, "total")
        by_id = {row["request_id"]: row for row in right["requests"]}
        for previous in left["requests"]:
            selected = by_id[previous["request_id"]]
            location = {"request_id": previous["request_id"], "partition": previous.get("partition")}
            if selected.get("partition") != location["partition"]:
                raise ValueError("OptR diagnostic partition identity differs")
            check(previous, selected, view, "request", **location)
            for old_frame, new_frame in zip(previous["frames"], selected["frames"]):
                frame_location = {**location, **{key: old_frame[key] for key in
                    ("frame_id", "context", "current_end")}}
                check(old_frame, new_frame, view, "frame", **frame_location)
                for old_query, new_query in zip(old_frame["queries"], new_frame["queries"]):
                    check(old_query, new_query, view, "query", **frame_location,
                          position=old_query["position"])
                check(old_frame["terminal_query"], new_frame["terminal_query"], view,
                      "terminal", **frame_location, position=old_frame["terminal_query"]["position"])
    strict = after["squared_error"] < before["squared_error"]
    if after["squared_error"] > before["squared_error"]:
        reason = "aggregate_error_regression"
    elif not strict:
        reason = "no_strict_total_improvement"
    elif counts["total"]:
        reason = "projection_total_regression"
    elif not passed:
        reason = "fine_grained_regression"
    else:
        reason = "passed"
    scope_order = {"query": 0, "terminal": 1, "frame": 2, "request": 3, "total": 4}
    failures.sort(key=lambda row: (-row["normalized_error"]["delta"],
                                  -row["squared_error"]["delta"], scope_order[row["scope"]], row["projection_view"],
                                  row.get("request_id", ""), row.get("position", -1)))
    return {"nonworse": passed, "strict_total_improvement": strict,
            "accepted": passed and strict, "reason": reason,
            "bf16_only_regression": views["fp32"]["nonworse"] and not passed,
            "views": views, "regressed_checks": counts, "regressed_frame_stages": stages,
            "top_failures": failures[:_TOP_FAILURES]}


def _json(path):
    return json.loads(Path(path).read_text())


def _generation(directory, identity, layer_ids):
    """Find one complete canonical identity; do not create checkpoint folders."""
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    expected_name = hashlib.sha256(encoded.encode()).hexdigest()
    matches = []
    if not directory.is_dir():
        raise ValueError("OptR diagnostic has no saved fit-checkpoint directory")
    for child in sorted(directory.iterdir()):
        if not child.is_dir():
            continue
        present = [child / f"layer-{layer}.json" for layer in layer_ids
                   if (child / f"layer-{layer}.json").is_file()]
        if not present:
            if child.name == expected_name:
                raise ValueError("OptR diagnostic current generation lacks completed layers")
            continue
        try:
            manifest = _json(present[0])
        except (ValueError, OSError):
            if child.name == expected_name:
                raise
            continue
        if manifest.get("identity") != identity:
            if child.name == expected_name:
                raise ValueError("OptR diagnostic current checkpoint identity was altered")
            continue
        if child.is_symlink() or child.name != expected_name:
            raise ValueError("OptR diagnostic ambiguous or noncanonical checkpoint generation")
        expected = {f"layer-{layer}.json" for layer in layer_ids}
        actual = {path.name for path in child.glob("layer-*.json")
                  if re.fullmatch(r"layer-\d+\.json", path.name)}
        if actual != expected:
            raise ValueError("OptR diagnostic requires completed coverage of every model FULL layer")
        matches.append(child)
    if len(matches) != 1:
        raise ValueError("OptR diagnostic requires one complete current-implementation checkpoint generation")
    return matches[0]


def _load_layer(directory, identity, name, seed, profile, source_checkpoint=None):
    import torch
    from oscar_ascend.optr_artifacts import matrix_digest, validate_refinement_metadata
    from oscar_ascend.optr_checkpoint import FitCheckpoints
    from tools.prepare_optr_recipe3 import _validate_result
    layer_id = seed["layer_id"]
    key = f"layer-{layer_id}"
    manifest_path = directory / (key + ".json")
    payload = FitCheckpoints._load_from(manifest_path, key, identity)
    artifact = payload.get("artifact")
    if (not isinstance(artifact, dict) or artifact.get("test_only") is not identity["test_only"] or
            artifact.get("model_fingerprint") != profile["model_fingerprint"] or
            artifact.get("head_dim") != identity["head_dim"] or
            artifact.get("global_kv_heads") != identity["global_kv_heads"] or
            set(artifact.get("layers", {})) != {name}):
        raise ValueError("OptR diagnostic completed-layer artifact identity differs")
    certificate = validate_refinement_metadata(artifact, expected_profile=profile)
    for field in ("base_artifact_sha256", "trace_manifest_sha256", "fitting_implementation_sha256",
                  "source_fit_snapshot_sha256"):
        if certificate.get(field) != identity.get(field):
            raise ValueError("OptR diagnostic completed-layer source/implementation identity differs")
    if certificate.get("full_layer_names") != identity["full_layers"]:
        raise ValueError("OptR diagnostic completed-layer model coverage differs")
    record = certificate["layers"][name]
    constants = artifact["layers"][name]
    base_digest = matrix_digest({name: seed})
    if (record["accepted"] or certificate["accepted_layers"] or
            record["base_constants_sha256"] != base_digest or
            any(not torch.equal(constants[field], seed[field]) for field in ("Rk", "Rv", "k_mean"))):
        raise ValueError("OptR diagnostic expects rejected layers retaining the actual recipe3 baseline")
    proposal_key = key + "-shared"
    proposal_path = directory / (proposal_key + ".json")
    proposal = FitCheckpoints._load_from(proposal_path, proposal_key, identity)
    policy = profile["refinement_policy"]
    if policy.get("policy_version") == "optr_recipe3_native_residual_ascend_v12":
        if source_checkpoint is not None:
            raise ValueError("native-residual diagnostic cannot consume old fitted proposals")
        return _native_layer(name, seed, record, constants, proposal, policy, identity,
                             manifest_path, proposal_path)
    source_policy = policy
    source_record = None
    provenance = None
    if source_checkpoint is not None:
        from tools.optr_recipe3_source import load_source_layer
        source_policy = source_checkpoint["policy"]
        _, source_record, provenance = load_source_layer(
            source_checkpoint, name, seed, identity["global_kv_heads"], source_policy)
    binding = {"strategy": policy["deployment_candidate_selection"], "policy": policy,
               "base_constants_sha256": base_digest,
               "fitting_implementation_sha256": identity["fitting_implementation_sha256"]}
    if provenance is not None:
        binding["source_fit_provenance"] = provenance
    if record.get("source_fit_provenance") != provenance:
        raise ValueError("OptR diagnostic imported fitting provenance differs")
    if proposal.get("recipe3_binding") != binding:
        raise ValueError("OptR diagnostic proposal binding differs")
    result = _validate_result(proposal.get("result"), seed, source_policy)
    if record.get("shared_decision") != result["decision"]:
        raise ValueError("OptR diagnostic completed-layer proposal decision differs")
    selection = record.get("shared_layer_selection", {})
    if source_record is not None and (selection != source_record["shared_layer_selection"] or
            record["validation"]["baseline"] != source_record["validation"]["baseline"]):
        raise ValueError("OptR diagnostic imported candidate measurements differ")
    measured = selection.get("evaluated_candidates")
    if (selection.get("method") != "recipe3_perhead_existing_kv_checkpoints_v1" or
            selection.get("accepted") is not False or selection.get("selected_candidate") != "seed" or
            selection.get("test_used_for_selection") is not False or
            selection.get("max_candidates") != policy["max_deployment_candidates"] or
            not isinstance(measured, list) or len(measured) != len(result["candidates"])):
        raise ValueError("OptR diagnostic saved candidate coverage/selection differs")
    candidates = []
    before = record["validation"]["baseline"]
    for matrix, evaluation in zip(result["candidates"], measured):
        digest = matrix_digest({name: {**seed, "Rk": matrix["Rk"], "Rv": matrix["Rv"]}})
        if (not isinstance(evaluation, dict) or evaluation.get("candidate") != matrix["id"] or
                evaluation.get("phase") != matrix["phase"] or evaluation.get("step") != matrix["step"] or
                evaluation.get("constants_sha256") != digest):
            raise ValueError("OptR diagnostic metrics do not bind the saved candidate matrices")
        detail = metric_rejections(before, evaluation.get("validation"))
        if evaluation.get("accepted") is not detail["accepted"] or detail["accepted"]:
            raise ValueError("OptR diagnostic candidate decisions conflict with the unchanged gate")
        candidates.append({"candidate": matrix["id"], "phase": matrix["phase"], "step": matrix["step"],
                           "constants_sha256": digest, "unchanged_baseline": digest == base_digest,
                           "proxy_losses": matrix.get("proxy_losses"), **detail})
    repair_candidates = []
    repair = record.get("recipe3_repair_selection")
    if policy.get("policy_version") == "optr_recipe3_trust_region_ascend_v11":
        from oscar_ascend.optr_trust_region import REPAIR_POLICY, _source_digest, _source_directions
        # validate_refinement_metadata above has checked the frozen policy,
        # exact strict decisions, selected minimum, budget and independent test.
        # Describe saved scalar evidence only: no reconstruction or replay.
        if not isinstance(repair, dict) or repair.get("accepted") is not False:
            raise ValueError("OptR diagnostic requires completed rejected repair evidence")
        directions = [{key: value for key, value in row.items() if key != "unit_tangent"}
                      for row in _source_directions(seed, result)]
        if (repair.get("source_candidates_sha256") != _source_digest(name, seed, result) or
                repair.get("directions") != directions or
                len(repair["evaluated_candidates"]) != len(directions) * len(REPAIR_POLICY["radii"])):
            raise ValueError("OptR diagnostic repair source/direction coverage differs")
        expected = [(direction, radius) for direction in directions for radius in REPAIR_POLICY["radii"]]
        for evaluation, (direction, radius) in zip(repair["evaluated_candidates"], expected):
            if (evaluation.get("radius") != radius or any(evaluation.get(key) != direction[key]
                    for key in ("source_id", "source_phase", "source_step", "head", "side"))):
                raise ValueError("OptR diagnostic repair fixed candidate order differs")
            detail = metric_rejections(before, evaluation["validation"])
            if evaluation.get("accepted") is not detail["accepted"] or detail["accepted"]:
                raise ValueError("OptR diagnostic repair decisions conflict with the unchanged gate")
            repair_candidates.append({**{key: evaluation[key] for key in
                ("candidate", "source_id", "source_phase", "source_step", "head", "side", "radius",
                 "constants_sha256", "same_bf16_operand_as_seed")},
                "phase": "repair", "unchanged_baseline": evaluation["constants_sha256"] == base_digest,
                **detail})
        candidates.extend(repair_candidates)
    changed = [row for row in candidates if not row["unchanged_baseline"]]
    best = min(changed or candidates, key=lambda row: row["views"]["fp32"]["squared_error"]["candidate"])
    detail = {"layer": name, "layer_id": layer_id, "base_constants_sha256": base_digest,
            "candidate_count": len(candidates), "best_by_total_error": best["candidate"],
            "best_ranking_scope": "changed_candidates" if changed else "unchanged_candidates_only",
            "independent_test": "baseline_compared_with_itself_no_proposal_tested",
            "candidate_checkpoint": str(proposal_path), "completed_checkpoint": str(manifest_path),
            "checkpoint_manifest_sha256": {
                "completed": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "proposal": hashlib.sha256(proposal_path.read_bytes()).hexdigest()},
            "candidates": candidates,
            "original_candidate_count": len(result["candidates"]),
            "repair_candidate_count": len(repair_candidates),
            "source_fit_provenance": provenance}
    from tools.optr_rejection_summary import summarize_layer
    return detail, summarize_layer(detail)


def _native_layer(name, seed, record, constants, saved, policy, identity, completed_path, lock_path):
    """Authenticate the locked v12 evidence, keeping train and validation apart.

    Small matrix reconstruction authenticates saved tangents and their hashes;
    no captured tensor, gradient, candidate search or attention replay is used.
    Passing train candidates are expected here when validation rejected them.
    """
    import torch
    from oscar_ascend.optr_artifacts import matrix_digest
    from oscar_ascend.optr_native_residual import validate_selection
    from tools.optr_rejection_summary import summarize_layer
    baseline_hash = matrix_digest({name: seed})
    binding = {"strategy": policy["deployment_candidate_selection"], "policy": policy,
               "base_constants_sha256": baseline_hash,
               "fitting_implementation_sha256": identity["fitting_implementation_sha256"]}
    if saved.get("native_residual_binding") != binding:
        raise ValueError("native-residual diagnostic lock binding differs")
    selected, before, after, decision, proposal = (saved.get(key) for key in
        ("selected", "before", "after", "decision", "proposal"))
    validate_selection(name, seed, selected, before, after, decision, proposal, policy)
    if (record.get("native_residual_selection") != decision or
            record["validation"] != {"baseline": before, "selected": after} or
            selected.get("layer_id") != constants.get("layer_id") or
            any(not torch.equal(selected[key], constants[key]) for key in ("Rk", "Rv", "k_mean"))):
        raise ValueError("native-residual diagnostic completed layer differs from its selection lock")

    def describe(row, phase, previous, measured):
        detail = metric_rejections(previous, measured)
        if row.get("accepted") is not detail["accepted"]:
            raise ValueError("native-residual diagnostic candidate conflicts with its strict gate")
        return {**{key: row[key] for key in ("candidate", "head", "side", "radius", "radius_index",
                    "constants_sha256", "same_bf16_operand_as_seed")},
                "phase": phase, "unchanged_baseline": row["constants_sha256"] == baseline_hash, **detail}

    train = [describe(row, "native_train", decision["train_baseline"], row["train"])
             for row in decision["evaluated_train_candidates"]]
    validation = [describe(row, "native_validation", before, row["validation"])
                  for row in decision["evaluated_validation_candidates"]]
    if decision["accepted"] or any(row["accepted"] for row in validation):
        raise ValueError("native-residual diagnostic expects no validation winner")
    detail = {"layer": name, "layer_id": seed["layer_id"], "strategy": binding["strategy"],
        "base_constants_sha256": baseline_hash,
        "candidate_count": len(train), "train_candidate_count": len(train),
        "validation_candidate_count": len(validation),
        "candidate_count_scope": "unique_train_candidates_validation_rechecks_counted_separately",
        "independent_test": "baseline_compared_with_itself_no_proposal_tested",
        "candidate_checkpoint": str(lock_path), "completed_checkpoint": str(completed_path),
        "checkpoint_manifest_sha256": {
            "completed": hashlib.sha256(completed_path.read_bytes()).hexdigest(),
            "proposal": hashlib.sha256(lock_path.read_bytes()).hexdigest()},
        "candidates": train, "validation_candidates": validation,
        "gradient_evaluations": decision["gradient_evaluations"]}
    return detail, summarize_layer(detail)


def report_no_improvement(config, *, output=None, testing=False):
    """Save a report from current completed work; return compact layer summaries.

    ``output`` is the same refined-PT destination accepted by prepare. It never
    redirects or overwrites any PT/checkpoint. Test-only evidence requires the
    explicit testing flag; the default path accepts production evidence only.
    """
    from oscar_ascend.optr_artifacts import file_sha256
    from oscar_ascend.optr_checkpoint import fitting_implementation_digest
    from oscar_ascend.optr_policy import (RECIPE3_POLICY_VERSION, RECIPE3_REPAIR_POLICY_VERSION,
                                        RECIPE3_NATIVE_POLICY_VERSION, compatible_capture_profiles)
    from oscar_ascend.rotations import calibration_cache_identity, checkpoint_identity, rotation_artifact_path
    from tools.prepare_optr_recipe3_base import base_config, load_base
    from tools.phase import atomic_json
    if type(testing) is not bool:
        raise ValueError("OptR diagnostic testing permission must be an explicit boolean")
    names, dim, fingerprint = checkpoint_identity(config["model"])
    profile = calibration_cache_identity(config, fingerprint)
    if (config.get("optr_baseline") != "recipe3" or
            profile.get("refinement_policy", {}).get("policy_version") not in
            (RECIPE3_POLICY_VERSION, RECIPE3_REPAIR_POLICY_VERSION, RECIPE3_NATIVE_POLICY_VERSION)):
        raise ValueError("OptR diagnostic requires the current recipe3 refinement policy")
    destination = Path(output).expanduser().resolve() if output else rotation_artifact_path(config, fingerprint)
    selected = base_config(config, fingerprint)
    baseline = load_base(selected, testing=testing)
    base_hash = file_sha256(selected["rotation_artifact"])
    from tools.prepare_optr_rotations import _existing_trace_manifest
    trace_path = _existing_trace_manifest(destination, profile, config)
    trace_hash = file_sha256(trace_path)
    trace = _json(trace_path)
    test_only = trace.get("test_only", False)
    if (type(test_only) is not bool or (test_only and not testing) or
            trace.get("status") != "native_optr_capture_complete" or
            trace.get("model_fingerprint") != fingerprint or
            trace.get("calibration_profile") not in [profile, *compatible_capture_profiles(profile, config)] or
            trace.get("optr_policy") != trace.get("calibration_profile", {}).get("refinement_policy") or
            trace.get("recipe3_sampling", {}).get("seed_capture", {}).get("baseline_artifact_sha256") != base_hash):
        raise ValueError("OptR diagnostic capture identity or test-only permission differs")
    identity = {"profile": profile, "base_artifact_sha256": base_hash,
                "trace_manifest_sha256": trace_hash, "full_layers": names,
                "global_kv_heads": baseline["global_kv_heads"], "head_dim": dim,
                "fitting_implementation_sha256": fitting_implementation_digest(), "test_only": testing}
    ids = [baseline["layers"][name]["layer_id"] for name in names]
    source_checkpoint = None
    if profile["refinement_policy"]["policy_version"] == RECIPE3_REPAIR_POLICY_VERSION:
        from tools.optr_recipe3_source import discover_source, source_snapshot_digest
        source_checkpoint = discover_source(config, trace_path, {"base_artifact_sha256": base_hash},
            names, baseline["global_kv_heads"], dim, testing)
        if source_checkpoint is not None:
            identity["source_fit_snapshot_sha256"] = source_snapshot_digest(source_checkpoint)
    generation = _generation(destination.parent / "fit-checkpoints", identity, ids)
    layers, compact = [], []
    for name in names:
        detail, summary = _load_layer(generation, identity, name, baseline["layers"][name], profile, source_checkpoint)
        layers.append(detail)
        compact.append(summary)
    report_path = destination.parent / "no-improvement-report.json"
    result = {"schema_version": 1, "status": "no_strict_validation_improvement",
              "checkpoint_generation": str(generation), "identity": identity, "layers": layers,
              "model_quality": "not_measured", "performance": "not_measured",
              "arithmetic_evidence": "saved_cpu_deployment_reference_not_npu_bitwise",
              "evidence_limits": [
                  "Saved candidate validation metrics explain rejection; no new replay or fitting was performed.",
                  "Candidate ordering in this report is descriptive only; no selection or publication changed.",
                  "Rejected proposals were not tested independently; test passed compares recipe3 with itself.",
                  "Repair rows bind saved checkpoint evidence; this report does not reconstruct microstep matrices.",
                  "Native-residual rows separate train checks from validation rechecks; a train pass is not a validation pass.",
                  "Native-residual locks authenticate saved small-matrix directions without attention replay or training.",
                  "Frame context zero identifies first prefill; positive context identifies cached-context frames.",
                  "Query counts include projections and partitions; terminal checks duplicate their final query.",
                  "This report does not establish LongBench accuracy, NPU equivalence, or serving performance."]}
    atomic_json(report_path, result)
    native_counts = ({"validation_candidate_count": sum(row.get("validation_candidate_count", 0) for row in layers)}
                     if any("validation_candidates" in row for row in layers) else {})
    return {"status": result["status"], "report": str(report_path), **native_counts,
            "checkpoint_generation": str(generation), "layer_count": len(layers),
            "candidate_count": sum(row["candidate_count"] for row in layers),
            "layers": compact, "recomputed_fit_or_replay": False,
            "artifact_published": False, "model_quality": "not_measured", "performance": "not_measured"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="Authenticate completed checkpoints using this config")
    source.add_argument("--saved-report", type=Path,
                        help="Summarize existing diagnostic JSON only; no Torch/model/PT/checkpoint access")
    parser.add_argument("--output", type=Path, help="Original refined-PT destination, if explicitly overridden")
    args = parser.parse_args()
    if args.saved_report:
        if args.output is not None:
            parser.error("--output is only valid with --config; saved-report mode never writes files")
        from tools.optr_rejection_summary import summarize_saved_report
        result = summarize_saved_report(args.saved_report)
    else:
        result = report_no_improvement(_json(args.config), output=args.output)
    for row in result.pop("layers"):
        print("[oscar] OPTR_PROGRESS " + json.dumps({"rotation_prepare": "optr",
            "phase": "no_improvement_layer", **row}, allow_nan=False), flush=True)
    print("[oscar] OPTR_PROGRESS " + json.dumps({"rotation_prepare": "optr",
        "phase": "no_improvement_report", **result}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
