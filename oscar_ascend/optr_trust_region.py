"""Archive #170/#174/#175/#176/#177, G26: bounded offline recipe3 repair.

Reuse authenticated existing OptR checkpoints, without another optimizer or
native capture. Directions are selected by their fixed step order, never by
validation/test scores. Only the changed head is replayed; cached FP32 head
projections reconstruct the unchanged whole-layer, per-query BF16 gates.
Neither this module nor its caches belongs to serving. CPU replay cost is
additional offline work, not a measured performance improvement.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time

import torch


# Imported by optr_policy: keep imports of policy/artifact/replay modules local
# to functions to avoid an import cycle during policy initialization.
REPAIR_POLICY = {
    "version": "recipe3_first_checkpoint_tangent_radii_v1",
    "direction": "earliest_positive_phase_step_then_primary",
    "radii": [.02, .01, .005, .0025, .00125, .000625, .0003125, .00015625],
    "parameter_scope": "single_kv_head_single_rotation_from_recipe3",
    "tangent_normalization": "frobenius_over_sqrt_head_dim",
    "minimum_tangent_rms": 1e-12,
    "max_kv_heads": 4,
    "selection": "strict_query_projection_nonworse_best_validation_sse",
    "test_used_for_selection": False,
}


def _validate_policy(policy):
    if policy.get("recipe3_repair") != REPAIR_POLICY:
        raise ValueError("recipe3 repair requires its exact versioned policy")
    if (policy.get("validation_partitions") != ["single_split", "captured_runtime_frames"] or
            policy.get("validation_projection_views") != ["fp32", "bf16_output", "bf16_rank_output"]):
        raise ValueError("recipe3 repair requires both partitions and all projection views")


def _source_directions(baseline, source_result):
    """Choose at most one existing direction per head and K/V side."""
    from .optr_recipe3_fit import validate_perhead_pair

    heads, dim = validate_perhead_pair(baseline["Rk"], baseline["Rv"], baseline["k_mean"])
    if heads > REPAIR_POLICY["max_kv_heads"]:
        raise ValueError("recipe3 repair exceeds the supported KV-head count")
    if not isinstance(source_result, dict) or not isinstance(source_result.get("decision"), dict):
        raise ValueError("recipe3 repair requires a completed source fitting result")
    validate_perhead_pair(source_result.get("Rk"), source_result.get("Rv"), baseline["k_mean"])
    candidates = source_result.get("candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 12:
        raise ValueError("recipe3 repair requires the bounded existing candidate pool")
    ids = set()
    primary = None
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("invalid recipe3 repair source candidate")
        identity, phase, step = (candidate.get(key) for key in ("id", "phase", "step"))
        if not isinstance(identity, str) or not identity or identity in ids:
            raise ValueError("recipe3 repair source candidate identity differs")
        ids.add(identity)
        validate_perhead_pair(candidate.get("Rk"), candidate.get("Rv"), baseline["k_mean"])
        if phase == "primary":
            if identity != "primary" or step is not None or primary is not None:
                raise ValueError("recipe3 repair primary source is invalid")
            if any(not torch.equal(candidate[key], source_result.get(key)) for key in ("Rk", "Rv")):
                raise ValueError("recipe3 repair primary differs from completed source")
            primary = candidate
        elif (phase not in ("k", "v") or type(step) is not int or step < 0 or
              identity != f"{phase}_step_{step}"):
            raise ValueError("recipe3 repair source phase/step is invalid")
    if primary is None:
        raise ValueError("recipe3 repair source primary is missing")
    directions = []
    for head in range(heads):
        for side, phase in (("Rk", "k"), ("Rv", "v")):
            ordered = sorted((row for row in candidates if row["phase"] == phase and row["step"] > 0),
                             key=lambda row: (row["step"], row["id"]))
            ordered.append(primary)
            origin = baseline[side][head]
            for candidate in ordered:
                source = candidate[side][head]
                if torch.equal(source, origin):
                    continue
                relative = origin.T @ source
                tangent = (relative - relative.T) * .5
                rms = float(torch.linalg.vector_norm(tangent)) / math.sqrt(dim)
                if not math.isfinite(rms):
                    raise ValueError("nonfinite recipe3 repair tangent")
                if rms <= REPAIR_POLICY["minimum_tangent_rms"]:
                    continue
                directions.append({"head": head, "side": side, "source_id": candidate["id"],
                    "source_phase": candidate["phase"], "source_step": candidate["step"],
                    "tangent_rms": rms, "unit_tangent": tangent / rms})
                break
    return directions


def _proposal(baseline, direction, radius):
    head, side = direction["head"], direction["side"]
    constants = {**baseline, side: baseline[side].clone()}
    constants[side][head] = baseline[side][head] @ torch.matrix_exp(direction["unit_tangent"] * radius)
    return constants


def _source_digest(name, baseline, source_result):
    from .optr_artifacts import matrix_digest
    rows = [{key: row[key] for key in ("id", "phase", "step")} | {
        "constants_sha256": matrix_digest({name: {**baseline, "Rk": row["Rk"], "Rv": row["Rv"]}})}
        for row in source_result["candidates"]]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _validate_partitions(metric):
    """Require both recorded partitions, in the exact full-metric order."""
    for view in [metric, *metric["projection_checks"].values()]:
        rows = view["requests"]
        if len(rows) % 2:
            raise ValueError("recipe3 repair metric partition coverage differs")
        count = len(rows) // 2
        for index, partition in enumerate(("single_split", "captured_runtime_frames")):
            group = rows[index * count:(index + 1) * count]
            if any(row.get("partition") != partition or
                   not row["request_id"].endswith(":" + partition) for row in group):
                raise ValueError("recipe3 repair metric partition identity differs")
        for left, right in zip(rows[:count], rows[count:], strict=True):
            if (left["request_id"].removesuffix(":single_split") !=
                    right["request_id"].removesuffix(":captured_runtime_frames") or
                    any(left[key] != right[key] for key in ("sample_sha256", "teacher_energy"))):
                raise ValueError("recipe3 repair partitions identify different samples")


def validate_selection(name, baseline, source_result, selected, before, after, decision, policy):
    """Authenticate a locked selection without replay, fit or test access.

    Reconstruct the fixed matrix bank only to verify hashes and the selected
    tensors. Recorded validation evidence determines the same unique winner;
    it cannot add directions, change radii, or select a second winner after a
    failed independent test. Checkpoint identity remains the caller's duty.
    """
    from .optr_artifacts import matrix_digest, nonworse
    from .optr_recipe3_fit import validate_perhead_pair

    _validate_policy(policy)
    directions = _source_directions(baseline, source_result)
    expected_directions = [{key: value for key, value in row.items() if key != "unit_tangent"}
                           for row in directions]
    if (not isinstance(decision, dict) or decision.get("method") != REPAIR_POLICY["version"] or
            decision.get("policy") != REPAIR_POLICY or decision.get("test_used_for_selection") is not False or
            decision.get("layer") != name or decision.get("baseline_layer_id") != baseline.get("layer_id") or
            decision.get("directions") != expected_directions or
            decision.get("base_constants_sha256") != matrix_digest({name: baseline}) or
            decision.get("source_candidates_sha256") != _source_digest(name, baseline, source_result)):
        raise ValueError("recipe3 repair selection policy/source/baseline identity differs")
    rows = decision.get("evaluated_candidates")
    expected_count = len(directions) * len(REPAIR_POLICY["radii"])
    if (not isinstance(rows, list) or len(rows) != expected_count or
            type(decision.get("candidate_count")) is not int or decision["candidate_count"] != expected_count or
            type(decision.get("max_candidates")) is not int or
            decision["max_candidates"] != 16 * baseline["Rk"].shape[0]):
        raise ValueError("recipe3 repair fixed candidate coverage differs")
    strict = dict(require_frames=True, require_terminal=True, require_queries=True, require_projection=True)
    nonworse(before, before, **strict)
    _validate_partitions(before)
    winner, winner_metric, winner_id, accepted = baseline, before, "seed", False
    offset = 0
    with torch.no_grad():
        for direction in directions:
            head, side = direction["head"], direction["side"]
            for index, radius in enumerate(REPAIR_POLICY["radii"]):
                row = rows[offset]
                offset += 1
                expected = {"candidate": f"repair_h{head}_{side}_r{index}", "head": head, "side": side,
                    "source_id": direction["source_id"], "source_phase": direction["source_phase"],
                    "source_step": direction["source_step"], "radius": radius}
                if not isinstance(row, dict) or any(row.get(key) != value for key, value in expected.items()):
                    raise ValueError("recipe3 repair candidate is outside its fixed direction/radius grid")
                constants = _proposal(baseline, direction, radius)
                if row.get("constants_sha256") != matrix_digest({name: constants}):
                    raise ValueError("recipe3 repair candidate constants differ from their fixed source")
                same_operand = torch.equal(constants[side][head].bfloat16(), baseline[side][head].bfloat16())
                measured = row.get("validation")
                if row.get("same_bf16_operand_as_seed") is not same_operand or (same_operand and measured != before):
                    raise ValueError("recipe3 repair unchanged BF16 operand evidence differs")
                passes = nonworse(before, measured, **strict) and measured["squared_error"] < before["squared_error"]
                _validate_partitions(measured)
                if row.get("accepted") is not passes:
                    raise ValueError("recipe3 repair candidate disagrees with the strict validation gate")
                if passes and measured["squared_error"] < winner_metric["squared_error"]:
                    winner, winner_metric, winner_id, accepted = constants, measured, row["candidate"], True
    if not isinstance(selected, dict):
        raise ValueError("recipe3 repair locked constants must be an object")
    validate_perhead_pair(selected.get("Rk"), selected.get("Rv"), selected.get("k_mean"))
    if (decision.get("accepted") is not accepted or decision.get("selected_candidate") != winner_id or
            after != winner_metric or selected.get("layer_id") != baseline.get("layer_id") or
            any(not torch.equal(selected[key], winner[key]) for key in ("Rk", "Rv", "k_mean")) or
            decision.get("selected_constants_sha256") != matrix_digest({name: selected})):
        raise ValueError("recipe3 repair locked selection differs from the best strict validation candidate")
    return True


def _project_head(constants, head, samples, policy):
    from .optr_calibration import project_output, replay_sample
    return [project_output(replay_sample(constants["Rk"][head], constants["Rv"][head],
                                        constants["k_mean"][head], sample, policy), sample)
            for sample in samples[head]]


def _project_head_partitions(constants, head, samples, policy, partitions):
    from .optr_calibration import project_output, replay_sample_partitions

    if len(partitions) == 1:
        partition = partitions[0]
        return {partition: _project_head(constants, head, samples, {**policy, "replay_partition": partition})}
    outputs = {partition: [] for partition in partitions}
    rk, rv, mean = constants["Rk"][head], constants["Rv"][head], constants["k_mean"][head]
    for sample in samples[head]:
        # Only this request's expanded history exists during the two replays.
        # Persist projected outputs, never prepared KV or attention graphs.
        attention = replay_sample_partitions(rk, rv, mean, sample, policy, partitions)
        for partition in partitions:
            outputs[partition].append(project_output(attention[partition], sample))
    return outputs


def _combine_partition_metrics(values, partitions):
    """Match prepare_optr_rotations._metrics ordering and scalar reductions."""
    from .optr_traces import _aggregate_requests
    rows = {view: [] for view in ("fp32", "bf16_output", "bf16_rank_output")}
    for metric, partition in zip(values, partitions, strict=True):
        for view, result in {"fp32": metric, **metric["projection_checks"]}.items():
            rows[view].extend({**row, "request_id": row["request_id"] + ":" + partition,
                              "partition": partition} for row in result["requests"])
    result = _aggregate_requests(rows.pop("fp32"))
    result["projection_checks"] = {view: _aggregate_requests(items) for view, items in rows.items()}
    return result


class ProjectionCache:
    """Baseline projected outputs only; no persistent rotated/expanded KV."""

    def __init__(self, baseline, samples, policy, *, progress=None):
        from .optr_calibration import project_output
        from .optr_traces import aggregate_layer_error
        from tools.prepare_optr_rotations import _single_partition_equivalent

        self.baseline, self.samples, self.policy = baseline, samples, policy
        self.partitions = policy["validation_partitions"]
        self.equivalent = _single_partition_equivalent(samples, policy)
        self.progress = progress
        self.outputs = {}
        self.teachers = [[project_output(sample["teacher"], sample) for sample in head] for head in samples]
        active = self.partitions[:1] if self.equivalent else self.partitions
        self.outputs = {partition: [] for partition in active}
        for head in range(len(samples)):
            projected = _project_head_partitions(baseline, head, samples, policy, active)
            for partition in active:
                self.outputs[partition].append(projected[partition])
                if progress is not None:
                    progress(event="repair_baseline_head_replayed", partition=partition, head=head)
        metrics = []
        for partition in self.partitions:
            if partition == "captured_runtime_frames" and self.equivalent:
                self.outputs[partition] = self.outputs["single_split"]
                metrics.append(copy.deepcopy(metrics[0]))
                continue
            replay_policy = {**policy, "replay_partition": partition}
            metrics.append(aggregate_layer_error(baseline, samples, replay_policy,
                projected_outputs=self.outputs[partition], projected_teachers=self.teachers))
        self.before = _combine_partition_metrics(metrics, self.partitions)

    def measure(self, constants, head):
        from .optr_traces import aggregate_layer_error

        active = self.partitions[:1] if self.equivalent else self.partitions
        projected = _project_head_partitions(constants, head, self.samples, self.policy, active)
        metrics = []
        for partition in self.partitions:
            if partition == "captured_runtime_frames" and self.equivalent:
                metrics.append(copy.deepcopy(metrics[0]))
                continue
            replay_policy = {**self.policy, "replay_partition": partition}
            outputs = list(self.outputs[partition])
            outputs[head] = projected[partition]
            metrics.append(aggregate_layer_error(constants, self.samples, replay_policy,
                projected_outputs=outputs, projected_teachers=self.teachers))
            # aggregate_layer_error owns scalar evidence, not these tensors.
            del outputs
            if self.progress is not None:
                self.progress(event="repair_changed_head_replayed", partition=partition, head=head)
        return _combine_partition_metrics(metrics, self.partitions)

    def measure_many(self, constants, changed_heads):
        """Replay an explicit head subset, then apply the unchanged whole-layer metric.

        Archive #175-177: joint candidates may alter multiple heads, but their
        errors cannot be added independently. Keep the original global head
        reduction order, all query/view evidence and both source partitions.
        """
        from .optr_traces import aggregate_layer_error
        from .optr_calibration import _matrices

        heads = len(self.samples)
        if (not isinstance(changed_heads, list) or not changed_heads or
                any(type(head) is not int or not 0 <= head < heads for head in changed_heads) or
                changed_heads != sorted(set(changed_heads))):
            raise ValueError("changed_heads must be a nonempty ascending unique head list")
        for key in ("Rk", "Rv", "k_mean"):
            value, baseline = constants.get(key), self.baseline[key]
            if (not torch.is_tensor(value) or value.shape != baseline.shape or
                    value.dtype != baseline.dtype or value.device != baseline.device or
                    not bool(torch.isfinite(value).all())):
                raise ValueError("multi-head candidate constants differ from baseline geometry/dtype")
            for head in range(heads):
                # A tiny FP32 correction may leave this head's deployed BF16
                # operands unchanged; such heads legitimately reuse baseline.
                if head not in changed_heads and not torch.equal(value[head].bfloat16(), baseline[head].bfloat16()):
                    raise ValueError("multi-head candidate changed an undeclared head")
        for head in range(heads):
            if head not in changed_heads:
                _matrices(constants["Rk"][head], constants["Rv"][head], constants["k_mean"][head])
        if len(changed_heads) == 1:
            return self.measure(constants, changed_heads[0])

        active = self.partitions[:1] if self.equivalent else self.partitions
        projected = {head: _project_head_partitions(constants, head, self.samples, self.policy, active)
                     for head in changed_heads}
        metrics = []
        for partition in self.partitions:
            if partition == "captured_runtime_frames" and self.equivalent:
                metrics.append(copy.deepcopy(metrics[0]))
                continue
            outputs = list(self.outputs[partition])
            for head in changed_heads:
                outputs[head] = projected[head][partition]
            metrics.append(aggregate_layer_error(constants, self.samples,
                {**self.policy, "replay_partition": partition}, projected_outputs=outputs,
                projected_teachers=self.teachers))
            del outputs
            if self.progress is not None:
                for head in changed_heads:
                    self.progress(event="repair_changed_head_replayed", partition=partition, head=head)
        return _combine_partition_metrics(metrics, self.partitions)


def select_repair(name, baseline, source_result, validation_samples, policy, *, progress=None):
    """Select a strict whole-layer validation improvement from <=64 microsteps.

    Only validation samples are accepted. Independent test is the caller's
    mandatory subsequent gate and can neither reselect nor modify this bank.
    The unchanged recipe3 seed is returned if no proposal passes every gate.
    """
    from .optr_artifacts import matrix_digest, nonworse
    from .optr_recipe3_fit import validate_perhead_pair

    _validate_policy(policy)
    directions = _source_directions(baseline, source_result)
    if (not isinstance(validation_samples, list) or len(validation_samples) != baseline["Rk"].shape[0]
            or any(not isinstance(head, list) or not head for head in validation_samples)):
        raise ValueError("recipe3 repair requires nonempty validation samples for every KV head")
    if any(sample.get("split", "validation") != "validation" for head in validation_samples for sample in head):
        raise ValueError("recipe3 repair cannot select on training or independent-test samples")
    started = time.perf_counter()
    accepted, selected, selected_id = False, baseline, "seed"
    strict = dict(require_frames=True, require_terminal=True, require_queries=True, require_projection=True)
    evaluated = []
    emit = (lambda **fields: progress(layer=name, **fields)) if progress is not None else None
    with torch.no_grad():
        cache = ProjectionCache(baseline, validation_samples, policy, progress=emit)
        before = after = cache.before
        for direction in directions:
            head, side = direction["head"], direction["side"]
            for index, radius in enumerate(REPAIR_POLICY["radii"]):
                candidate_id = f"repair_h{head}_{side}_r{index}"
                constants = _proposal(baseline, direction, radius)
                validate_perhead_pair(constants["Rk"], constants["Rv"], baseline["k_mean"])
                same_operand = torch.equal(constants[side][head].bfloat16(), baseline[side][head].bfloat16())
                measured = before if same_operand else cache.measure(constants, head)
                passes = nonworse(before, measured, **strict) and measured["squared_error"] < before["squared_error"]
                evaluated.append({"candidate": candidate_id, "source_id": direction["source_id"],
                    "source_phase": direction["source_phase"], "source_step": direction["source_step"],
                    "head": head, "side": side, "radius": radius, "accepted": passes,
                    "same_bf16_operand_as_seed": same_operand,
                    "constants_sha256": matrix_digest({name: constants}), "validation": measured})
                if passes and measured["squared_error"] < after["squared_error"]:
                    accepted, selected, after, selected_id = True, constants, measured, candidate_id
                if emit is not None:
                    emit(event="repair_candidate_checked", candidate=candidate_id, head=head, side=side,
                         source_id=direction["source_id"], radius=radius, accepted=passes,
                         validation_squared_error=measured["squared_error"])
    decision = {"method": REPAIR_POLICY["version"], "policy": copy.deepcopy(REPAIR_POLICY),
        "layer": name, "baseline_layer_id": baseline.get("layer_id"),
        "accepted": accepted, "selected_candidate": selected_id, "test_used_for_selection": False,
        "base_constants_sha256": matrix_digest({name: baseline}),
        "source_candidates_sha256": _source_digest(name, baseline, source_result),
        "selected_constants_sha256": matrix_digest({name: selected}),
        "directions": [{key: value for key, value in row.items() if key != "unit_tangent"} for row in directions],
        "candidate_count": len(evaluated), "max_candidates": 16 * baseline["Rk"].shape[0],
        "elapsed_seconds": time.perf_counter() - started,
        "evaluated_candidates": evaluated,
        "cache": {"persistent": "baseline_head_projections_and_teachers", "changed_heads_per_replay": 1,
                  "persistent_rotated_kv": False},
        "model_quality": "not_measured", "performance": "not_measured"}
    validate_selection(name, baseline, source_result, selected, before, after, decision, policy)
    decision["elapsed_seconds"] = time.perf_counter() - started
    return selected, before, after, decision
