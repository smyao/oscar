"""Archive #86/#97/#148/#170/#173: authenticated OptR constants/certificates.

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

from .optr_policy import (OBJECTIVE, RECIPE_VERSION, UPSTREAM_COMMIT,
                          EXPERIMENTAL_PUBLICATION_POLICY, PROJECTION_VIEWS)


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


def _frames(row, *, required=False, require_terminal=False, require_queries=False):
    """Validate captured-forward evidence without weakening old certificates."""
    if "frames" not in row:
        if required or require_terminal or require_queries:
            raise ValueError("OptR frame-aware policy requires per-frame metrics")
        return None
    frames = row["frames"]
    if not isinstance(frames, list) or not frames:
        raise ValueError("OptR frame metrics must be a nonempty list")
    previous_end = 0
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError("OptR frame metric must be an object")
        frame_id, context, end = (frame.get(key) for key in ("frame_id", "context", "current_end"))
        positions = frame.get("query_positions")
        if (any(type(value) is not int for value in (frame_id, context, end))
                or frame_id != index or context != previous_end or end <= context
                or not isinstance(positions, list) or not positions
                or any(type(position) is not int or not context <= position < end for position in positions)
                or any(left >= right for left, right in zip(positions, positions[1:]))):
            raise ValueError("OptR captured frame geometry/positions differ")
        previous_end = end
        for key in ("normalized_error", "squared_error", "teacher_energy"):
            value = frame.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("OptR per-frame metric must be finite and nonnegative")
        expected = frame["squared_error"] / max(frame["teacher_energy"], 1e-20)
        if not math.isclose(frame["normalized_error"], expected, rel_tol=1e-12, abs_tol=0):
            raise ValueError("OptR per-frame normalization differs")
        terminal = frame.get("terminal_query")
        if terminal is None:
            if require_terminal or require_queries or "queries" in frame or "terminal_query" in frame:
                raise ValueError("OptR terminal-aware policy requires terminal query metrics")
            continue
        if (not isinstance(terminal, dict) or type(terminal.get("position")) is not int
                or terminal["position"] != positions[-1] or positions[-1] != end - 1):
            raise ValueError("OptR terminal query must identify the captured frame's final token")
        for key in ("normalized_error", "squared_error", "teacher_energy"):
            value = terminal.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("OptR terminal query metric must be finite and nonnegative")
        expected = terminal["squared_error"] / max(terminal["teacher_energy"], 1e-20)
        if not math.isclose(terminal["normalized_error"], expected, rel_tol=1e-12, abs_tol=0):
            raise ValueError("OptR terminal query normalization differs")
        if any(terminal[key] > frame[key] for key in ("squared_error", "teacher_energy")):
            raise ValueError("OptR terminal query metrics exceed captured frame totals")
        if len(positions) == 1 and any(terminal[key] != frame[key] for key in
                                     ("squared_error", "teacher_energy", "normalized_error")):
            raise ValueError("OptR single-query frame and terminal query metrics differ")
        queries = frame.get("queries")
        if queries is None and not require_queries and "queries" not in frame:
            continue
        if not isinstance(queries, list) or len(queries) != len(positions):
            raise ValueError("OptR query-aware policy requires every captured query metric")
        for position, query in zip(positions, queries):
            if (not isinstance(query, dict) or type(query.get("position")) is not int or
                    query["position"] != position):
                raise ValueError("OptR per-query position/coverage differs")
            for key in ("normalized_error", "squared_error", "teacher_energy"):
                value = query.get(key)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError("OptR per-query metric must be finite and nonnegative")
            expected = query["squared_error"] / max(query["teacher_energy"], 1e-20)
            if not math.isclose(query["normalized_error"], expected, rel_tol=1e-12, abs_tol=0):
                raise ValueError("OptR per-query normalization differs")
        if terminal != queries[-1]:
            raise ValueError("OptR terminal query differs from final per-query metric")
        for key in ("squared_error", "teacher_energy"):
            if not math.isclose(frame[key], sum(query[key] for query in queries), rel_tol=1e-12, abs_tol=0):
                raise ValueError("OptR captured frame totals differ from per-query metrics")
    for key in ("squared_error", "teacher_energy"):
        if not math.isclose(row[key], sum(frame[key] for frame in frames), rel_tol=1e-12, abs_tol=0):
            raise ValueError("OptR per-request totals differ from captured frames")
    return frames


def _projection_geometry(rows):
    """Rounding views share samples/queries, but have different teacher energy."""
    return {key: (row["sample_sha256"], None if "frames" not in row else [
        (frame["frame_id"], frame["context"], frame["current_end"], frame["query_positions"],
         [query["position"] for query in frame["queries"]] if "queries" in frame else None,
         frame.get("terminal_query", {}).get("position")) for frame in row["frames"]])
        for key, row in rows.items()}


def _metric(metric, *, require_frames=False, require_terminal=False, require_queries=False,
            require_projection=False, allow_projection=True):
    if not isinstance(metric, dict):
        raise ValueError("OptR metric must be an object")
    for key in ("normalized_error", "squared_error", "teacher_energy"):
        value = metric.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("OptR metrics must be finite and nonnegative")
    rows = metric.get("requests")
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
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
        _frames(row, required=require_frames, require_terminal=require_terminal, require_queries=require_queries)
    for key in ("squared_error", "teacher_energy"):
        if not math.isclose(metric[key], sum(row[key] for row in rows), rel_tol=1e-12, abs_tol=0):
            raise ValueError("OptR metric totals differ from requests")
    expected = metric["squared_error"] / max(metric["teacher_energy"], 1e-20)
    if not math.isclose(metric["normalized_error"], expected, rel_tol=1e-12, abs_tol=0):
        raise ValueError("OptR metric normalization differs")
    result = {row["request_id"]: row for row in rows}
    if "projection_checks" in metric or require_projection:
        if not allow_projection:
            raise ValueError("OptR projection metrics cannot contain nested projection checks")
        views = metric.get("projection_checks")
        if not isinstance(views, dict) or set(views) != set(PROJECTION_VIEWS[1:]):
            raise ValueError("OptR projection-aware policy requires both BF16 projection metrics")
        for view in views.values():
            view_rows = _metric(view, require_frames=require_frames, require_terminal=require_terminal,
                                require_queries=require_queries, allow_projection=False)
            if _projection_geometry(view_rows) != _projection_geometry(result):
                raise ValueError("OptR projection sample/frame/query coverage differs")
    return result


def nonworse(before, after, *, require_frames=False, require_terminal=False, require_queries=False,
             require_projection=False):
    """Exact nonregression for every request and its captured-forward metrics.

    Historical request/frame-only evidence remains readable when both sides
    omit finer evidence. New policies explicitly require frame/terminal,
    query and projection metrics; discarding one side or changing geometry
    is never permitted.
    """
    options = dict(require_frames=require_frames, require_terminal=require_terminal,
                   require_queries=require_queries, require_projection=require_projection)
    left = _metric(before, **options)
    right = _metric(after, **options)
    if set(left) != set(right) or before["teacher_energy"] != after["teacher_energy"]:
        raise ValueError("OptR comparison used different requests or teacher")
    frame_pairs = []
    for key in left:
        if any(left[key][field] != right[key][field] for field in ("sample_sha256", "teacher_energy")):
            raise ValueError("OptR comparison sample/teacher identity differs")
        before_frames, after_frames = left[key].get("frames"), right[key].get("frames")
        if (before_frames is None) != (after_frames is None):
            raise ValueError("OptR comparison captured frame coverage differs")
        if before_frames is not None:
            if len(before_frames) != len(after_frames):
                raise ValueError("OptR comparison captured frame coverage differs")
            for previous, selected in zip(before_frames, after_frames):
                if any(previous[field] != selected[field] for field in
                       ("frame_id", "context", "current_end", "query_positions", "teacher_energy")):
                    raise ValueError("OptR comparison frame geometry/teacher identity differs")
                frame_pairs.append((previous, selected))
                before_queries, after_queries = previous.get("queries"), selected.get("queries")
                if (before_queries is None) != (after_queries is None):
                    raise ValueError("OptR comparison per-query coverage differs")
                if before_queries is not None:
                    for before_query, after_query in zip(before_queries, after_queries):
                        if any(before_query[field] != after_query[field] for field in
                               ("position", "teacher_energy")):
                            raise ValueError("OptR comparison per-query position/teacher identity differs")
                        frame_pairs.append((before_query, after_query))
                before_terminal, after_terminal = previous.get("terminal_query"), selected.get("terminal_query")
                if (before_terminal is None) != (after_terminal is None):
                    raise ValueError("OptR comparison terminal query coverage differs")
                if before_terminal is not None:
                    if any(before_terminal[field] != after_terminal[field] for field in
                           ("position", "teacher_energy")):
                        raise ValueError("OptR comparison terminal query position/teacher identity differs")
                    frame_pairs.append((before_terminal, after_terminal))
    before_views, after_views = before.get("projection_checks"), after.get("projection_checks")
    if (before_views is None) != (after_views is None):
        raise ValueError("OptR comparison projection coverage differs")
    projection_passed = []
    if before_views is not None:
        for view in PROJECTION_VIEWS[1:]:
            projection_passed.append(nonworse(before_views[view], after_views[view],
                require_frames=require_frames, require_terminal=require_terminal, require_queries=require_queries))
    return (all(projection_passed) and after["squared_error"] <= before["squared_error"] and
            after["normalized_error"] <= before["normalized_error"] and
            all(right[key]["squared_error"] <= left[key]["squared_error"] and
                right[key]["normalized_error"] <= left[key]["normalized_error"] for key in left) and
            all(selected["squared_error"] <= previous["squared_error"] and
                selected["normalized_error"] <= previous["normalized_error"]
                for previous, selected in frame_pairs))


def independent_test_summary(records):
    regressed = sorted(name for name, record in records.items()
                       if not nonworse(record["test"]["baseline"], record["test"]["selected"]))
    return {"status": "failed" if regressed else "passed", "regressed_layers": regressed}


def experimental_certificate(certificate):
    """Version publication separately from the unchanged fitting/cache policy."""
    return {**certificate, "schema_version": 2,
            "publication_policy": EXPERIMENTAL_PUBLICATION_POLICY, "experimental": True,
            "independent_test": independent_test_summary(certificate["layers"])}


def artifact_test_status(artifact):
    """Startup-only receipt; callers validate the artifact before reporting."""
    certificate = artifact["refinement_certificate"]
    summary = certificate.get("independent_test", {"status": "passed", "regressed_layers": []})
    return {"experimental": certificate.get("experimental", False),
            "refinement_policy_version": certificate.get("policy", {}).get("policy_version"),
            "fitting_objective": certificate.get("policy", {}).get("fitting_objective"),
            "validation_projection_views": certificate.get("policy", {}).get("validation_projection_views", ["fp32"]),
            "publication_policy": certificate.get("publication_policy", "strict_independent_test_v1"),
            "independent_test": summary["status"],
            "independent_test_regressed_layers": summary["regressed_layers"],
            "model_quality": "not_measured", "performance": "not_measured"}


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
    policy = profile.get("refinement_policy", {})
    projection_policy = policy.get("policy_version") == "optr_oscar_shared_projection_ascend_v8"
    train_seed = projection_policy or policy.get("policy_version") == "optr_oscar_shared_train_seed_ascend_v7"
    shared_layer = train_seed or policy.get("policy_version") == "optr_oscar_shared_upstream_ascend_v6"
    source_seed = shared_layer or policy.get("policy_version") == "optr_oscar_base_upstream_ascend_v5"
    selection = ("shared_layer_each_query_and_projection_nonworse" if projection_policy
                 else "shared_layer_each_request_frame_and_terminal_nonworse")
    if shared_layer and (policy.get("fitted_grouping") != "layer" or policy.get("selection") != selection):
        raise ValueError("shared-layer OptR requires shared fitting and whole-layer selection")
    if projection_policy and policy.get("validation_projection_views") != list(PROJECTION_VIEWS):
        raise ValueError("OptR projection policy requires all frozen validation views")
    if source_seed:
        from .optr_oscar_seed import validate_seed_profile
        validate_seed_profile(base, artifact.get("model_fingerprint"), allow_legacy=not train_seed)
    elif (base.get("recipe_version") != 3 or base.get("rotation_variant") != "perhead_center" or
            base.get("model_fingerprint") != artifact.get("model_fingerprint")):
        raise ValueError("OptR must refine the authenticated recipe3 model")
    certificate = artifact.get("refinement_certificate", {})
    schema = certificate.get("schema_version")
    if (type(schema) is not int or schema not in (1, 2) or
            certificate.get("policy") != profile.get("refinement_policy") or
            certificate.get("metric") != "all_tp_heads_bf16_gate_fp32_wo_output" or
            certificate.get("arithmetic_evidence") != "cpu_deployment_reference_not_npu_bitwise" or
            certificate.get("model_quality") != "not_measured" or
            certificate.get("performance") != "not_measured"):
        raise ValueError("OptR certificate policy/evidence boundary differs")
    if schema == 2:
        if (certificate.get("publication_policy") != EXPERIMENTAL_PUBLICATION_POLICY or
                certificate.get("experimental") is not True):
            raise ValueError("OptR report-only publication requires an explicit experimental certificate")
    elif any(key in certificate for key in ("publication_policy", "experimental", "independent_test")):
        raise ValueError("legacy OptR certificate cannot change its publication policy")
    for key in ("base_artifact_sha256", "trace_manifest_sha256", "matrices_sha256"):
        value = certificate.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("OptR certificate lacks content hashes")
    layers = artifact.get("layers", {})
    if set(certificate.get("layers", {})) != set(layers) or not layers:
        raise ValueError("OptR certificate FULL layer coverage differs")
    require_terminal = (shared_layer or certificate["policy"].get("policy_version") == "optr_joint_gqa_bf16_ascend_v4"
                        or certificate["policy"].get("selection") == "joint_all_heads_each_request_frame_and_terminal_nonworse")
    require_frames = (require_terminal or certificate["policy"].get("policy_version") == "optr_frame_bf16_ascend_v3"
                      or certificate["policy"].get("selection") == "joint_all_heads_each_request_and_frame_nonworse")
    for name, record in certificate["layers"].items():
        if source_seed:
            import torch
            mean = layers[name].get("k_mean")
            if not torch.is_tensor(mean) or bool(torch.count_nonzero(mean)):
                raise ValueError("source OSCAR OptR must retain zero key mean")
        if shared_layer:
            heads, dim = artifact.get("global_kv_heads"), artifact.get("head_dim")
            if type(heads) is not int or heads < 1 or type(dim) is not int or dim < 1:
                raise ValueError("shared-layer OptR head geometry is invalid")
            if (artifact.get("format_version") != 2 or artifact.get("source_grouping") != "layer_kv_head" or
                    artifact.get("rotation_variant") != "perhead_center" or
                    mean.dtype != torch.float32 or mean.shape != (heads, dim)):
                raise ValueError("shared-layer OptR requires the existing perhead_center storage ABI")
            for key in ("Rk", "Rv"):
                matrix = layers[name].get(key)
                if (not torch.is_tensor(matrix) or matrix.dtype != torch.float32 or
                        matrix.shape != (heads, dim, dim) or not bool(torch.isfinite(matrix).all()) or
                        not torch.equal(matrix, matrix[:1].expand_as(matrix))):
                    raise ValueError("shared-layer OptR requires identical shared Rk/Rv for every KV head")
            if "head_subset_selection" in record:
                raise ValueError("shared-layer OptR cannot use head subset selection")
        selected_digest = matrix_digest({name: layers[name]})
        if record.get("selected_constants_sha256") != selected_digest:
            raise ValueError("OptR layer certificate does not bind selected constants")
        base_digest = record.get("base_constants_sha256", "")
        if not isinstance(base_digest, str) or len(base_digest) != 64 or any(c not in "0123456789abcdef" for c in base_digest):
            raise ValueError("OptR layer certificate lacks baseline constants hash")
        for split in ("validation", "test"):
            result = record.get(split, {})
            passed = nonworse(result.get("baseline"), result.get("selected"), require_frames=require_frames,
                             require_terminal=require_terminal, require_queries=projection_policy,
                             require_projection=projection_policy)
            if not passed and (split == "validation" or schema == 1):
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
    if schema == 2 and certificate.get("independent_test") != independent_test_summary(certificate["layers"]):
        raise ValueError("OptR independent test status does not match recorded errors")
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
    validated = validate_artifact(artifact, layer_names=layer_names, head_dim=head_dim,
        model_fingerprint=model_fingerprint, pr_fingerprint=OFFICIAL_COMMIT,
        require_calibrated=True, device=device, testing=testing)
    print("[oscar] OPTR_ARTIFACT " + json.dumps({"artifact": str(destination),
          **artifact_test_status(validated)}), flush=True)
    return validated
