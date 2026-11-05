"""Archive #170/#175-177: bounded offline fitting of deployed layer residuals.

The BF16/INT2 straight-through gradient only proposes a direction. Actual
forward train/validation evidence, with unchanged strict gates, decides whether
it is usable. Every proposal starts at the original recipe3; no online code,
clip, layout, sampling policy, or independent-test decision is changed here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time

import torch


NATIVE_RESIDUAL_POLICY = {
    "version": "recipe3_native_layer_residual_joint_candidates_v2",
    "objective": "mean_query_teacher_energy_normalized_joint_postgate_three_views",
    "direction": "negative_streamed_train_ste_tangent_at_recipe3",
    "parameter_scope": "single_rotation_and_all_train_directions_joint_from_recipe3",
    "tangent_normalization": "frobenius_over_sqrt_head_dim",
    "radii": [.01, .0025],
    "max_kv_heads": 4,
    "gradient_evaluations_per_direction": 1,
    "train_candidates_per_direction": 2,
    "validation_candidates_per_direction": 1,
    "joint_train_candidates": 2,
    "joint_validation_candidates": 1,
    "joint_direction": "all_original_train_head_side_unit_tangents_same_radius",
    "max_train_candidates": 18,
    "max_validation_candidates": 9,
    "selection": "strict_train_then_validation_query_projection_nonworse",
    "test_used_for_selection": False,
}
STRICT = dict(require_frames=True, require_terminal=True, require_queries=True, require_projection=True)
VIEWS = ("fp32", "bf16_output", "bf16_rank_output")
PARTITIONS = ("single_split", "captured_runtime_frames")


def _validate_policy(policy):
    if (policy.get("native_residual") != NATIVE_RESIDUAL_POLICY or
            policy.get("validation_projection_views") != list(VIEWS) or
            policy.get("validation_partitions") != list(PARTITIONS)):
        raise ValueError("native residual requires its exact policy, partitions and projection views")


def _hash(value):
    if (not isinstance(value, str) or len(value) != 64 or
            any(char not in "0123456789abcdef" for char in value)):
        raise ValueError("native residual requires a lowercase SHA256")
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _sources(samples, split, heads):
    if (not isinstance(samples, list) or len(samples) != heads or
            any(not isinstance(head, list) or not head for head in samples)):
        raise ValueError("native residual requires all heads and nonempty samples")
    rows = []
    for head in samples:
        rows.append([])
        for sample in head:
            if not isinstance(sample, dict):
                raise ValueError("native residual sample must be an object")
            identity = sample.get("request_id")
            if (not isinstance(identity, str) or not identity or
                    sample.get("split", split) != split):
                raise ValueError("native residual sample split/request identity differs")
            rows[-1].append({"request_id": identity, "sample_sha256": _hash(sample.get("sample_sha256"))})
    ids = [row["request_id"] for row in rows[0]]
    if len(set(ids)) != len(ids) or any([row["request_id"] for row in head] != ids for head in rows):
        raise ValueError("native residual sample head/request order differs")
    return rows


def _check_sources(metric, sources):
    from .optr_artifacts import nonworse
    from .optr_trust_region import _validate_partitions
    nonworse(metric, metric, **STRICT)
    _validate_partitions(metric)
    count = len(sources[0])
    if len(metric["requests"]) != 2 * count:
        raise ValueError("native residual train metric source coverage differs")
    for partition_index, partition in enumerate(PARTITIONS):
        for index in range(count):
            row = metric["requests"][partition_index * count + index]
            if (row["request_id"] != sources[0][index]["request_id"] + ":" + partition or
                    row["sample_sha256"] != _digest([head[index]["sample_sha256"] for head in sources])):
                raise ValueError("native residual train metric source identity differs")


def _views(projected, teachers):
    """Differentiable values with aggregate_layer_error's exact cast/sum order."""
    from .optr_calibration import _bf16
    actual_sum = teacher_sum = rank_actual = rank_teacher = None
    for actual, teacher in zip(projected, teachers, strict=True):
        if actual_sum is None:
            actual_sum, teacher_sum = actual.double(), teacher.double()
            rank_actual, rank_teacher = _bf16(actual).double(), teacher.bfloat16().double()
        else:
            actual_sum = actual_sum + actual.double()
            teacher_sum = teacher_sum + teacher.double()
            rank_actual = rank_actual + _bf16(actual).double()
            rank_teacher = rank_teacher + teacher.bfloat16().double()
    return {
        "fp32": (actual_sum, teacher_sum),
        "bf16_output": (_bf16(actual_sum), teacher_sum.bfloat16().double()),
        "bf16_rank_output": (_bf16(rank_actual), rank_teacher.bfloat16().double()),
    }


def _project_batch(attention, sample, begin, end):
    """Preserve the whole-sample W_O GEMM shape while releasing each graph.

    Projection is row independent, but changing its outer dimension can select
    another BLAS reduction. Zero padding retains the validator's exact shape;
    only the live rows are used by this batch's objective and derivative.
    """
    from .optr_calibration import project_output
    padded = torch.zeros_like(sample["q"], dtype=torch.float32)
    padded[begin:end] = attention
    return project_output(padded, sample)[begin:end]


@torch.enable_grad()
def _gradient_direction(baseline, head, side, samples, policy, cache, *, progress=None):
    from .optr_calibration import _replay, _training_batches

    dim = baseline[side].shape[-1]
    correction = torch.zeros((dim, dim), dtype=torch.float32, requires_grad=True)
    phase = "k" if side == "Rk" else "v"
    total_queries = sum(sample["q"].shape[0] for sample in samples[head])
    denominator = total_queries * len(PARTITIONS) * len(VIEWS)
    partitions = PARTITIONS[:1] if cache.equivalent else PARTITIONS
    multiplicity = 2 if cache.equivalent else 1
    loss_sum = 0.
    for request, sample in enumerate(samples[head]):
        # Only this request's fixed side is resident; every attention graph is
        # released by backward before the next query microbatch is constructed.
        batches = _training_batches([sample], phase, baseline["Rk"][head],
                                   baseline["Rv"][head], baseline["k_mean"][head], policy)
        for partition in partitions:
            replay_policy = {**policy, "replay_partition": partition}
            offset = 0
            for batch, frozen in batches:
                count = batch["q"].shape[0]
                end = offset + count
                matrix = baseline[side][head] @ torch.matrix_exp(correction - correction.T)
                rk = matrix if side == "Rk" else baseline["Rk"][head]
                rv = matrix if side == "Rv" else baseline["Rv"][head]
                attention = _replay(rk, rv, baseline["k_mean"][head], batch, replay_policy, frozen=frozen)
                actual = _project_batch(attention, sample, offset, end)
                if not torch.equal(actual.detach(), cache.outputs[partition][head][request][offset:end]):
                    raise ValueError("native residual gradient forward differs from the recipe3 baseline replay")
                projected = [rows[request][offset:end] for rows in cache.outputs[partition]]
                projected[head] = actual
                teachers = [rows[request][offset:end] for rows in cache.teachers]
                loss = sum(((output - target).square().sum(-1) /
                            target.square().sum(-1).clamp_min(1e-20)).sum()
                           for output, target in _views(projected, teachers).values())
                loss = loss * (multiplicity / denominator)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite native residual training objective")
                loss_sum += float(loss.detach())
                loss.backward()
                offset = end
            if progress is not None:
                progress(event="native_residual_gradient_request", head=head, side=side,
                         request_id=sample["request_id"], partition=partition)
        del batches
    gradient = correction.grad
    if gradient is None or not bool(torch.isfinite(gradient).all()):
        raise ValueError("nonfinite or missing native residual tangent gradient")
    # The teacher-energy floor can create large but finite FP32 gradients.
    # Normalize in FP64 so subtracting opposite finite extremes, or converting
    # a tiny scalar denominator to FP32, cannot fabricate an inf/zero direction.
    tangent = -(gradient.double() - gradient.double().T) * .5
    rms = float(torch.linalg.vector_norm(tangent)) / math.sqrt(dim)
    if not math.isfinite(rms):
        raise ValueError("nonfinite native residual gradient norm")
    unit = (torch.zeros_like(tangent, dtype=torch.float32) if rms == 0 else
            (tangent / rms).float().contiguous())
    return {"head": head, "side": side, "gradient_rms": rms,
            "training_objective": loss_sum, "zero_gradient": rms == 0,
            "unit_tangent_sha256": _tensor_digest(unit), "unit_tangent": unit}


def _proposal(baseline, direction, radius):
    head, side = direction["head"], direction["side"]
    constants = {**baseline, side: baseline[side].clone()}
    constants[side][head] = baseline[side][head] @ torch.matrix_exp(direction["unit_tangent"] * radius)
    return constants


def _joint_proposal(baseline, directions, radius):
    """Use the original train directions together, without another gradient."""
    constants = {**baseline, "Rk": baseline["Rk"].clone(), "Rv": baseline["Rv"].clone()}
    for direction in directions:
        head, side = direction["head"], direction["side"]
        constants[side][head] = baseline[side][head] @ torch.matrix_exp(direction["unit_tangent"] * radius)
    return constants


def _same_operands(baseline, constants):
    return all(torch.equal(constants[side].bfloat16(), baseline[side].bfloat16()) for side in ("Rk", "Rv"))


def _measure_joint(cache, baseline, constants):
    changed = [head for head in range(baseline["Rk"].shape[0]) if any(
        not torch.equal(constants[side][head].bfloat16(), baseline[side][head].bfloat16())
        for side in ("Rk", "Rv"))]
    return cache.before if not changed else cache.measure_many(constants, changed)


def _passes(before, after):
    from .optr_artifacts import nonworse
    return nonworse(before, after, **STRICT) and after["squared_error"] < before["squared_error"]


def _scalar_direction(row):
    return {key: value for key, value in row.items() if key != "unit_tangent"}


def _candidate_identity(direction, index):
    head, side = direction["head"], direction["side"]
    return {"candidate": f"native_h{head}_{side}_r{index}", "scope": "single", "head": head, "side": side,
            "radius_index": index, "radius": NATIVE_RESIDUAL_POLICY["radii"][index]}


def _joint_identity(index):
    return {"candidate": f"native_joint_r{index}", "scope": "joint", "head": None, "side": "Rk_Rv",
            "radius_index": index, "radius": NATIVE_RESIDUAL_POLICY["radii"][index]}


def select_native_residual(name, baseline, train_samples, validation_samples, policy, *, progress=None):
    """One gradient per head/side; <=18 train/9 val trials, no test selection."""
    from .optr_artifacts import matrix_digest
    from .optr_recipe3_fit import validate_perhead_pair
    from .optr_trust_region import ProjectionCache

    _validate_policy(policy)
    heads, _ = validate_perhead_pair(baseline["Rk"], baseline["Rv"], baseline["k_mean"])
    if heads > NATIVE_RESIDUAL_POLICY["max_kv_heads"]:
        raise ValueError("native residual exceeds its supported KV head budget")
    sources = _sources(train_samples, "train", heads)
    validation_sources = _sources(validation_samples, "validation", heads)
    if {row["request_id"] for row in sources[0]} & {row["request_id"] for row in validation_sources[0]}:
        raise ValueError("native residual training and validation overlap")
    emit = (lambda **fields: progress(layer=name, **fields)) if progress is not None else None
    started = time.perf_counter()
    with torch.no_grad():
        train_cache = ProjectionCache(baseline, train_samples, policy, progress=emit)
    directions, train_rows, finalists = [], [], []
    for head in range(heads):
        for side in ("Rk", "Rv"):
            direction = _gradient_direction(baseline, head, side, train_samples, policy, train_cache, progress=emit)
            winner = None
            with torch.no_grad():
                for index, radius in enumerate(NATIVE_RESIDUAL_POLICY["radii"]):
                    constants = _proposal(baseline, direction, radius)
                    validate_perhead_pair(constants["Rk"], constants["Rv"], constants["k_mean"])
                    same = torch.equal(constants[side][head].bfloat16(), baseline[side][head].bfloat16())
                    measured = train_cache.before if same else train_cache.measure(constants, head)
                    passes = _passes(train_cache.before, measured)
                    row = {**_candidate_identity(direction, index), "accepted": passes,
                           "same_bf16_operand_as_seed": same,
                           "constants_sha256": matrix_digest({name: constants}), "train": measured}
                    train_rows.append(row)
                    if passes and (winner is None or measured["squared_error"] < winner["train"]["squared_error"]):
                        winner = row
                    if emit is not None:
                        emit(event="native_residual_train_checked", **{key: row[key] for key in
                             ("candidate", "head", "side", "radius", "accepted")},
                             squared_error=measured["squared_error"])
            direction["train_selected_candidate"] = "seed" if winner is None else winner["candidate"]
            directions.append(direction)
            if winner is not None:
                finalists.append(winner)
    joint_winner = None
    with torch.no_grad():
        for index, radius in enumerate(NATIVE_RESIDUAL_POLICY["radii"]):
            constants = _joint_proposal(baseline, directions, radius)
            validate_perhead_pair(constants["Rk"], constants["Rv"], constants["k_mean"])
            same = _same_operands(baseline, constants)
            measured = _measure_joint(train_cache, baseline, constants)
            passes = _passes(train_cache.before, measured)
            row = {**_joint_identity(index), "accepted": passes,
                   "same_bf16_operand_as_seed": same,
                   "constants_sha256": matrix_digest({name: constants}), "train": measured}
            train_rows.append(row)
            if passes and (joint_winner is None or measured["squared_error"] < joint_winner["train"]["squared_error"]):
                joint_winner = row
            if emit is not None:
                emit(event="native_residual_train_checked", **{key: row[key] for key in
                     ("candidate", "scope", "head", "side", "radius", "accepted")},
                     squared_error=measured["squared_error"])
    if joint_winner is not None:
        finalists.append(joint_winner)
    with torch.no_grad():
        validation_cache = ProjectionCache(baseline, validation_samples, policy, progress=emit)
        selected, before, after, selected_id = baseline, validation_cache.before, validation_cache.before, "seed"
        validation_rows = []
        for finalist in finalists:
            if finalist["scope"] == "joint":
                constants = _joint_proposal(baseline, directions, finalist["radius"])
                measured = _measure_joint(validation_cache, baseline, constants)
            else:
                direction = next(row for row in directions if (row["head"], row["side"]) ==
                                 (finalist["head"], finalist["side"]))
                constants = _proposal(baseline, direction, finalist["radius"])
                measured = validation_cache.measure(constants, direction["head"])
            passes = _passes(before, measured)
            row = {key: value for key, value in finalist.items() if key not in ("train", "accepted")}
            row.update(validation=measured, accepted=passes)
            validation_rows.append(row)
            if passes and measured["squared_error"] < after["squared_error"]:
                selected, after, selected_id = constants, measured, row["candidate"]
            if emit is not None:
                emit(event="native_residual_validation_checked", candidate=row["candidate"],
                     accepted=passes, squared_error=measured["squared_error"])
    proposal = {"method": NATIVE_RESIDUAL_POLICY["version"], "policy": copy.deepcopy(NATIVE_RESIDUAL_POLICY),
                "layer": name, "base_constants_sha256": matrix_digest({name: baseline}),
                "train_sources": sources, "train_samples_sha256": _digest(sources),
                "directions": directions}
    decision = {"method": proposal["method"], "policy": copy.deepcopy(NATIVE_RESIDUAL_POLICY),
                "layer": name, "baseline_layer_id": baseline.get("layer_id"),
                "base_constants_sha256": proposal["base_constants_sha256"],
                "selected_constants_sha256": matrix_digest({name: selected}),
                "selected_candidate": selected_id, "accepted": selected_id != "seed",
                "test_used_for_selection": False, "train_sources": sources,
                "train_samples_sha256": proposal["train_samples_sha256"],
                "directions": [_scalar_direction(row) for row in directions],
                "joint_train_selected_candidate": "seed" if joint_winner is None else joint_winner["candidate"],
                "train_baseline": train_cache.before, "evaluated_train_candidates": train_rows,
                "evaluated_validation_candidates": validation_rows,
                "train_candidate_count": len(train_rows), "validation_candidate_count": len(validation_rows),
                "max_train_candidates": heads * 4 + 2, "max_validation_candidates": heads * 2 + 1,
                "gradient_evaluations": heads * 2, "elapsed_seconds": time.perf_counter() - started,
                "model_quality": "not_measured", "performance": "not_measured"}
    validate_selection(name, baseline, selected, before, after, decision, proposal, policy)
    return selected, before, after, decision, proposal


def validate_decision(name, before, after, decision, heads, policy):
    """Validate JSON-only selection evidence, without tensors, replay or test."""
    from .optr_artifacts import nonworse
    from .optr_trust_region import _validate_partitions
    _validate_policy(policy)
    if (not isinstance(decision, dict) or decision.get("method") != NATIVE_RESIDUAL_POLICY["version"] or
            decision.get("policy") != NATIVE_RESIDUAL_POLICY or decision.get("layer") != name or
            decision.get("test_used_for_selection") is not False):
        raise ValueError("native residual decision policy/layer differs")
    try:
        json.dumps(decision, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("native residual decision must contain finite JSON evidence") from error
    for key in ("base_constants_sha256", "selected_constants_sha256", "train_samples_sha256"):
        _hash(decision.get(key))
    directions = decision.get("directions")
    if not isinstance(directions, list) or len(directions) not in (2, 4, 6, 8):
        raise ValueError("native residual direction budget differs")
    if type(heads) is not int or not 1 <= heads <= 4 or len(directions) != heads * 2:
        raise ValueError("native residual certified head count differs")
    sources = decision.get("train_sources")
    if (not isinstance(sources, list) or len(sources) != heads or any(not isinstance(h, list) or not h for h in sources)):
        raise ValueError("native residual train sources are incomplete")
    ids = [row.get("request_id") for row in sources[0] if isinstance(row, dict)]
    if (len(ids) != len(sources[0]) or len(set(ids)) != len(ids) or
            any(not isinstance(identity, str) or not identity for identity in ids)):
        raise ValueError("native residual train source IDs differ")
    for rows in sources:
        if len(rows) != len(ids) or any(not isinstance(row, dict) or row.get("request_id") != identity
                                      for row, identity in zip(rows, ids)):
            raise ValueError("native residual train source head coverage differs")
        for row in rows:
            _hash(row.get("sample_sha256"))
    if _digest(sources) != decision["train_samples_sha256"]:
        raise ValueError("native residual train source digest differs")
    train_before = decision.get("train_baseline")
    _check_sources(train_before, sources)
    nonworse(before, before, **STRICT)
    _validate_partitions(before)
    if {row["request_id"].removesuffix(":" + row["partition"]) for row in before["requests"]} & set(ids):
        raise ValueError("native residual train/validation evidence overlaps")
    train_rows = decision.get("evaluated_train_candidates")
    val_rows = decision.get("evaluated_validation_candidates")
    if (not isinstance(train_rows, list) or len(train_rows) != heads * 4 + 2 or not isinstance(val_rows, list) or
            len(val_rows) > heads * 2 + 1 or any(decision.get(key) != expected or type(decision.get(key)) is not int
                for key, expected in (("train_candidate_count", len(train_rows)),
                                      ("validation_candidate_count", len(val_rows)),
                                      ("max_train_candidates", heads * 4 + 2),
                                      ("max_validation_candidates", heads * 2 + 1), ("gradient_evaluations", heads * 2)))):
        raise ValueError("native residual candidate/gradient budget differs")
    finalists = []
    for direction_index, direction in enumerate(directions):
        if (not isinstance(direction, dict) or type(direction.get("head")) is not int or
                direction.get("head") != direction_index // 2 or
                direction.get("side") != ("Rk", "Rv")[direction_index % 2]):
            raise ValueError("native residual direction order differs")
        for key in ("gradient_rms", "training_objective"):
            value = direction.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid native residual gradient/objective evidence")
        if direction.get("zero_gradient") is not (direction["gradient_rms"] == 0):
            raise ValueError("native residual zero-gradient identity differs")
        _hash(direction.get("unit_tangent_sha256"))
        winner = None
        for index in range(2):
            row = train_rows[2 * direction_index + index]
            identity = _candidate_identity(direction, index)
            if (not isinstance(row, dict) or type(row.get("head")) is not int or
                    type(row.get("radius_index")) is not int or
                    any(row.get(key) != value for key, value in identity.items())):
                raise ValueError("native residual train candidate grid differs")
            _hash(row.get("constants_sha256"))
            same = row.get("same_bf16_operand_as_seed")
            measured = row.get("train")
            if type(same) is not bool or (same and measured != train_before) or (direction["zero_gradient"] and not same):
                raise ValueError("native residual unchanged operand evidence differs")
            passes = _passes(train_before, measured)
            _check_sources(measured, sources)
            if row.get("accepted") is not passes:
                raise ValueError("native residual train acceptance disagrees with strict gate")
            if passes and (winner is None or measured["squared_error"] < winner["train"]["squared_error"]):
                winner = row
        winner_id = "seed" if winner is None else winner["candidate"]
        if direction.get("train_selected_candidate") != winner_id:
            raise ValueError("native residual per-direction train winner differs")
        if winner is not None:
            finalists.append(winner)
    joint_winner = None
    for index in range(2):
        row = train_rows[heads * 4 + index]
        if (not isinstance(row, dict) or type(row.get("radius_index")) is not int or
                any(row.get(key) != value for key, value in _joint_identity(index).items())):
            raise ValueError("native residual joint candidate grid differs")
        _hash(row.get("constants_sha256"))
        same, measured = row.get("same_bf16_operand_as_seed"), row.get("train")
        if (type(same) is not bool or (same and measured != train_before) or
                (all(direction["zero_gradient"] for direction in directions) and not same)):
            raise ValueError("native residual unchanged joint operand evidence differs")
        passes = _passes(train_before, measured)
        _check_sources(measured, sources)
        if row.get("accepted") is not passes:
            raise ValueError("native residual joint train acceptance disagrees with strict gate")
        if passes and (joint_winner is None or measured["squared_error"] < joint_winner["train"]["squared_error"]):
            joint_winner = row
    joint_id = "seed" if joint_winner is None else joint_winner["candidate"]
    if decision.get("joint_train_selected_candidate") != joint_id:
        raise ValueError("native residual joint train winner differs")
    if joint_winner is not None:
        finalists.append(joint_winner)
    if len(val_rows) != len(finalists):
        raise ValueError("native residual validation finalist coverage differs")
    winner_metric, winner_id, winner_hash = before, "seed", decision["base_constants_sha256"]
    for row, source in zip(val_rows, finalists, strict=True):
        if not isinstance(row, dict) or any(row.get(key) != source[key] for key in
            ("candidate", "scope", "head", "side", "radius_index", "radius", "constants_sha256", "same_bf16_operand_as_seed")):
            raise ValueError("native residual validation candidate differs from train winner")
        measured = row.get("validation")
        passes = _passes(before, measured)
        _validate_partitions(measured)
        if row.get("accepted") is not passes:
            raise ValueError("native residual validation acceptance disagrees with strict gate")
        if passes and measured["squared_error"] < winner_metric["squared_error"]:
            winner_metric, winner_id, winner_hash = measured, row["candidate"], row["constants_sha256"]
    if (decision.get("accepted") is not (winner_id != "seed") or decision.get("selected_candidate") != winner_id or
            decision["selected_constants_sha256"] != winner_hash or after != winner_metric):
        raise ValueError("native residual selected evidence differs from validation winner")
    return True


def validate_selection(name, baseline, selected, before, after, decision, proposal, policy):
    """Authenticate a locked result without replay, gradients, or test access."""
    from .optr_artifacts import matrix_digest
    from .optr_recipe3_fit import validate_perhead_pair
    heads, dim = validate_perhead_pair(baseline["Rk"], baseline["Rv"], baseline["k_mean"])
    validate_decision(name, before, after, decision, heads, policy)
    if (decision.get("base_constants_sha256") != matrix_digest({name: baseline}) or
            decision.get("baseline_layer_id") != baseline.get("layer_id") or len(decision["directions"]) != heads * 2 or
            not isinstance(proposal, dict) or any(proposal.get(key) != decision.get(key) for key in
                ("method", "policy", "layer", "base_constants_sha256", "train_sources", "train_samples_sha256"))):
        raise ValueError("native residual baseline/proposal identity differs")
    directions = proposal.get("directions")
    if not isinstance(directions, list) or len(directions) != heads * 2:
        raise ValueError("native residual proposal direction coverage differs")
    winner = baseline
    with torch.no_grad():
        for index, direction in enumerate(directions):
            if not isinstance(direction, dict) or _scalar_direction(direction) != decision["directions"][index]:
                raise ValueError("native residual tensor/scalar direction evidence differs")
            tangent = direction.get("unit_tangent")
            if (not isinstance(tangent, torch.Tensor) or tangent.device.type != "cpu" or
                    tangent.dtype != torch.float32 or tuple(tangent.shape) != (dim, dim) or tangent.requires_grad or
                    not bool(torch.isfinite(tangent).all()) or not torch.equal(tangent, -tangent.T) or
                    _tensor_digest(tangent) != direction["unit_tangent_sha256"]):
                raise ValueError("native residual tangent tensor identity differs")
            norm = float(torch.linalg.vector_norm(tangent.double())) / math.sqrt(dim)
            expected = 0. if direction["zero_gradient"] else 1.
            if not math.isclose(norm, expected, rel_tol=2e-6, abs_tol=1e-12):
                raise ValueError("native residual tangent normalization differs")
            for radius_index, radius in enumerate(NATIVE_RESIDUAL_POLICY["radii"]):
                constants = _proposal(baseline, direction, radius)
                row = decision["evaluated_train_candidates"][index * 2 + radius_index]
                same = torch.equal(constants[direction["side"]][direction["head"]].bfloat16(),
                                   baseline[direction["side"]][direction["head"]].bfloat16())
                if row["constants_sha256"] != matrix_digest({name: constants}) or row["same_bf16_operand_as_seed"] is not same:
                    raise ValueError("native residual proposal matrix/operand identity differs")
                if row["candidate"] == decision["selected_candidate"]:
                    winner = constants
        for radius_index, radius in enumerate(NATIVE_RESIDUAL_POLICY["radii"]):
            constants = _joint_proposal(baseline, directions, radius)
            row = decision["evaluated_train_candidates"][heads * 4 + radius_index]
            if (row["constants_sha256"] != matrix_digest({name: constants}) or
                    row["same_bf16_operand_as_seed"] is not _same_operands(baseline, constants)):
                raise ValueError("native residual joint proposal matrix/operand identity differs")
            if row["candidate"] == decision["selected_candidate"]:
                winner = constants
    if not isinstance(selected, dict):
        raise ValueError("native residual selected constants must be an object")
    validate_perhead_pair(selected.get("Rk"), selected.get("Rv"), selected.get("k_mean"))
    if (selected.get("layer_id") != baseline.get("layer_id") or
            any(not torch.equal(selected[key], winner[key]) for key in ("Rk", "Rv", "k_mean")) or
            matrix_digest({name: selected}) != decision["selected_constants_sha256"]):
        raise ValueError("native residual selected matrices differ from locked winner")
    return True
