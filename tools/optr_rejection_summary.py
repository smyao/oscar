"""Archive #176/#177: separate proxy, repair and native train/validation evidence.

Standard-library-only reader of saved diagnostic JSON. This is descriptive
evidence, not checkpoint authentication, candidate selection or certification.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path


NATIVE_STRATEGY = "recipe3_native_residual_train_then_validation_v1"


def _best(rows):
    changed = [row for row in rows if not row["unchanged_baseline"]]
    return min(changed or rows, key=lambda row: row["views"]["fp32"]["squared_error"]["candidate"])


def _candidate_summary(row):
    # The report keeps all original values; only the terminal rendering is
    # shortened. Counts span projections/partitions, not unique query tokens.
    worst = row["top_failures"][:1]
    return {"candidate": row["candidate"],
            **{key: row[key] for key in ("head", "side", "radius") if key in row},
            "total_relative_delta": row["views"]["fp32"]["squared_error"]["relative_delta"],
            "reason": row["reason"], "regressed_checks": row["regressed_checks"],
            "worst_gate": [{**{key: gate[key] for key in
                ("projection_view", "scope", "request_id", "partition", "frame_id", "context",
                 "current_end", "position") if key in gate},
                "normalized_error_relative_delta": gate["normalized_error"]["relative_delta"],
                "positive_from_zero": gate["normalized_error"]["positive_from_zero"]} for gate in worst]}


def summarize_search(rows):
    improving = [row for row in rows if row["strict_total_improvement"]]
    return {"candidate_count": len(rows),
            "unchanged_constants_count": sum(row["unchanged_baseline"] for row in rows),
            "same_bf16_operand_count": sum(row.get("same_bf16_operand_as_seed", False) for row in rows),
            "strict_total_improvement_count": len(improving),
            "bf16_only_regression_count": sum(row["bf16_only_regression"] for row in rows),
            "reason_counts": dict(sorted(Counter(row["reason"] for row in rows).items())),
            "fewest_regressed_query_checks_among_improving":
                min((row["regressed_checks"]["query"] for row in improving), default=None),
            "best_total": _candidate_summary(_best(rows)) if rows else None}


def summarize_layer(layer):
    """Preserve legacy compact fields; expose repair evidence independently."""
    rows = layer["candidates"]
    if layer.get("strategy") == NATIVE_STRATEGY:
        validation = layer["validation_candidates"]
        return {"layer": layer["layer"], "strategy": NATIVE_STRATEGY,
                "candidates": len(rows), "train_candidates": len(rows),
                "validation_candidates": len(validation),
                "candidate_count_scope": "unique_train_candidates_validation_rechecks_counted_separately",
                "reason": "all_validation_finalists_rejected" if validation else "no_train_candidate_passed",
                "train_search": {**summarize_search(rows), "accepted_count": sum(row["accepted"] for row in rows)},
                "validation_search": {**summarize_search(validation), "accepted_count": 0}}
    best = _best(rows)
    original = [row for row in rows if row["phase"] != "repair"]
    repair = [row for row in rows if row["phase"] == "repair"]
    return {"layer": layer["layer"], "candidates": len(rows),
            "best_total_candidate": best["candidate"],
            "ranking_scope": "changed" if any(not row["unchanged_baseline"] for row in rows) else "unchanged_only",
            "total_relative_delta": best["views"]["fp32"]["squared_error"]["relative_delta"],
            "reason": best["reason"], "regressed_queries": best["regressed_checks"]["query"],
            "worst_gate": best["top_failures"][:1],
            "original_candidates": len(original), "repair_candidates": len(repair),
            "repair_search": summarize_search(repair)}


def _validate_rows(rows, *, phases, allow_accepted=False):
    """Validate one search's metrics; train and validation have separate baselines."""
    ids = set()
    baselines = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("candidate"), str) or row["candidate"] in ids:
            raise ValueError("saved OptR report has an invalid/duplicate candidate")
        ids.add(row["candidate"])
        if row.get("phase") not in phases:
            raise ValueError("saved OptR report has an invalid candidate phase")
        for key in ("unchanged_baseline", "strict_total_improvement", "bf16_only_regression", "nonworse", "accepted"):
            if type(row.get(key)) is not bool:
                raise ValueError("saved OptR report has a missing/invalid decision")
        if not allow_accepted and (row["accepted"] or (row["nonworse"] and row["strict_total_improvement"])):
            raise ValueError("saved no-improvement report contains an accepted candidate")
        counts = row.get("regressed_checks", {})
        if set(counts) != {"total", "request", "frame", "query", "terminal"} or any(
                type(count) is not int or count < 0 for count in counts.values()):
            raise ValueError("saved OptR report has invalid regression counts")
        if not isinstance(row.get("top_failures"), list):
            raise ValueError("saved OptR report lacks rejection details")
        views = row.get("views", {})
        if set(views) != {"fp32", "bf16_output", "bf16_rank_output"}:
            raise ValueError("saved OptR report lacks projection views")
        for name, view in views.items():
            if type(view.get("nonworse")) is not bool:
                raise ValueError("saved OptR report has an invalid projection decision")
            for key in ("squared_error", "normalized_error"):
                change = view[key]
                before, after = change["baseline"], change["candidate"]
                if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                       for value in (before, after)):
                    raise ValueError("saved OptR report has invalid errors")
                baseline_key = (name, key)
                if baseline_key in baselines and baselines[baseline_key] != before:
                    raise ValueError("saved OptR report candidates have different baselines")
                baselines[baseline_key] = before
                if (change["delta"] != after - before or
                        change["relative_delta"] != ((after - before) / before if before else None) or
                        change["positive_from_zero"] is not (before == 0 and after > 0)):
                    raise ValueError("saved OptR report has inconsistent error deltas")
        total = views["fp32"]["squared_error"]
        strict = total["candidate"] < total["baseline"]
        nonworse = all(view["nonworse"] for view in views.values())
        reason = ("aggregate_error_regression" if total["candidate"] > total["baseline"] else
                  "no_strict_total_improvement" if not strict else
                  "projection_total_regression" if counts["total"] else
                  "fine_grained_regression" if not nonworse else "passed")
        if (row["strict_total_improvement"] is not strict or row["nonworse"] is not nonworse or
                row["accepted"] is not (strict and nonworse) or
                row["reason"] != reason or
                row["bf16_only_regression"] is not (views["fp32"]["nonworse"] and not nonworse)):
            raise ValueError("saved OptR report has inconsistent rejection decisions")
        if row["phase"] in ("repair", "native_train", "native_validation"):
            if (type(row.get("same_bf16_operand_as_seed")) is not bool or
                    type(row.get("head")) is not int or row["head"] < 0 or
                    row.get("side") not in ("Rk", "Rv") or
                    type(row.get("radius")) not in (int, float) or
                    not math.isfinite(row["radius"]) or row["radius"] <= 0):
                raise ValueError("saved OptR report has invalid repair coordinates")


def _validate_native_layer(layer, rows):
    validation = layer.get("validation_candidates")
    if (not isinstance(validation, list) or layer.get("train_candidate_count") != len(rows) or
            layer.get("validation_candidate_count") != len(validation) or
            layer.get("candidate_count_scope") != "unique_train_candidates_validation_rechecks_counted_separately"):
        raise ValueError("saved native-residual report has inconsistent train/validation counts")
    _validate_rows(rows, phases=("native_train",), allow_accepted=True)
    _validate_rows(validation, phases=("native_validation",))
    by_id = {row["candidate"]: row for row in rows}
    # A trained finalist can pass train and fail validation. It must still be
    # the same matrix, and at most one radius per head/side reaches validation.
    directions = set()
    for row in validation:
        training = by_id.get(row["candidate"])
        if (training is None or not training["accepted"] or any(row.get(key) != training.get(key)
                for key in ("head", "side", "radius", "radius_index", "constants_sha256",
                            "unchanged_baseline", "same_bf16_operand_as_seed"))):
            raise ValueError("saved native-residual validation candidate differs from its training source")
        direction = (row["head"], row["side"])
        if direction in directions:
            raise ValueError("saved native-residual report repeats a validation direction")
        directions.add(direction)
    passing_directions = {(row["head"], row["side"]) for row in rows if row["accepted"]}
    if directions != passing_directions:
        raise ValueError("saved native-residual report lacks a training finalist's validation evidence")


def _validate_layer(layer):
    """Check summary structure/consistency, without claiming authentication."""
    if not isinstance(layer, dict) or not isinstance(layer.get("layer"), str):
        raise ValueError("saved OptR report has an invalid layer")
    rows = layer.get("candidates")
    if not isinstance(rows, list) or not rows or layer.get("candidate_count") != len(rows):
        raise ValueError("saved OptR report has incomplete candidate coverage")
    if layer.get("strategy") == NATIVE_STRATEGY:
        _validate_native_layer(layer, rows)
        return
    _validate_rows(rows, phases=("primary", "k", "v", "repair"))
    original = sum(row["phase"] != "repair" for row in rows)
    # v10 diagnostics predating v11 did not include these additive fields.
    if (layer.get("original_candidate_count", original) != original or
            layer.get("repair_candidate_count", len(rows) - original) != len(rows) - original):
        raise ValueError("saved OptR report has inconsistent search counts")


def summarize_saved_report(path):
    """Read one JSON file only; never import Torch or touch PT/checkpoints."""
    path = Path(path).expanduser().resolve()
    data = path.read_bytes()
    try:
        report = json.loads(data)
        if (report.get("schema_version") != 1 or report.get("status") != "no_strict_validation_improvement" or
                not isinstance(report.get("layers"), list) or not report["layers"]):
            raise ValueError("expected a complete schema-1 no-improvement report, not terminal progress")
        for layer in report["layers"]:
            _validate_layer(layer)
        names = [layer["layer"] for layer in report["layers"]]
        if len(set(names)) != len(names) or names != report["identity"]["full_layers"]:
            raise ValueError("saved OptR report has inconsistent layer coverage")
        if type(report["identity"].get("test_only")) is not bool:
            raise ValueError("saved OptR report lacks evidence scope")
        compact = [summarize_layer(layer) for layer in report["layers"]]
    except (KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError("saved OptR report has incomplete/invalid diagnostic fields") from error
    native_counts = ({"validation_candidate_count": sum(layer.get("validation_candidate_count", 0)
                                                       for layer in report["layers"])}
                     if any(layer.get("strategy") == NATIVE_STRATEGY for layer in report["layers"]) else {})
    return {"status": report["status"], "report": str(path), **native_counts,
            "report_sha256": hashlib.sha256(data).hexdigest(),
            "evidence_scope": "saved_json_only_checkpoints_not_reauthenticated",
            "test_only": report["identity"]["test_only"],
            "query_count_scope": "projection_and_partition_checks_not_unique_tokens",
            "layer_count": len(compact), "candidate_count": sum(row["candidates"] for row in compact),
            "layers": compact, "recomputed_fit_or_replay": False, "artifact_published": False,
            "model_quality": "not_measured", "performance": "not_measured"}
