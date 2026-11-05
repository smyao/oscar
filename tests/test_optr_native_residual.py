"""Archive #170/#175-177: real BF16 residual gradients and strict offline gates.

Synthetic CPU evidence only; no model accuracy or device equivalence claim.
"""
import copy
import json

import pytest
import torch

from oscar_ascend import optr_native_residual as residual
from oscar_ascend.optr_artifacts import matrix_digest, nonworse
from oscar_ascend.optr_calibration import project_output, replay_sample
from oscar_ascend.optr_trust_region import ProjectionCache
from test_optr_trust_region import basic_case
from tools.prepare_optr_rotations import _metrics


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def case(*, dim=8, splits=2):
    baseline, validation, policy = basic_case(dim=dim, splits=splits)
    train = copy.deepcopy(validation)
    for head_index, head in enumerate(train):
        for sample in head:
            sample.update(request_id="train", split="train", sample_sha256=f"{1000 + head_index:064x}")
    for head in validation:
        for sample in head:
            sample["split"] = "validation"
    policy = {**policy, "query_batch_size": 2,
              "native_residual": copy.deepcopy(residual.NATIVE_RESIDUAL_POLICY)}
    return baseline, train, validation, policy


@pytest.mark.parametrize("dim", [8, 256])
def test_tensor_views_and_streamed_projection_match_original_validator(dim):
    baseline, train, _, policy = case(dim=dim)
    with torch.no_grad():
        cache = ProjectionCache(baseline, train, policy)
        for partition in residual.PARTITIONS:
            views = residual._views([head[0] for head in cache.outputs[partition]],
                                    [head[0] for head in cache.teachers])
            for view, (actual, target) in views.items():
                metric = cache.before if view == "fp32" else cache.before["projection_checks"][view]
                row = next(row for row in metric["requests"] if row["partition"] == partition)
                expected = [query["squared_error"] for frame in row["frames"] for query in frame["queries"]]
                assert (actual - target).square().sum(-1).tolist() == expected
            for head, samples in enumerate(train):
                sample = samples[0]
                attention = replay_sample(baseline["Rk"][head], baseline["Rv"][head],
                    baseline["k_mean"][head], sample, {**policy, "replay_partition": partition})
                expected = project_output(attention, sample)
                actual = torch.cat([residual._project_batch(attention[begin:begin + 2], sample,
                                   begin, min(begin + 2, attention.shape[0]))
                                    for begin in range(0, attention.shape[0], 2)])
                assert torch.equal(actual, expected)


@pytest.mark.parametrize("side,dim", [("Rk", 8), ("Rv", 8), ("Rk", 256), ("Rv", 256)])
def test_streamed_gradient_matches_whole_sample_joint_residual_gradient(side, dim):
    baseline, train, _, policy = case(dim=dim)
    with torch.no_grad():
        cache = ProjectionCache(baseline, train, policy)
    actual = residual._gradient_direction(baseline, 0, side, train, policy, cache)
    correction = torch.zeros_like(baseline[side][0], requires_grad=True)
    matrix = baseline[side][0] @ torch.matrix_exp(correction - correction.T)
    losses = []
    for partition in residual.PARTITIONS:
        output = replay_sample(matrix if side == "Rk" else baseline["Rk"][0],
            matrix if side == "Rv" else baseline["Rv"][0], baseline["k_mean"][0],
            train[0][0], {**policy, "replay_partition": partition})
        values = [project_output(output, train[0][0]), cache.outputs[partition][1][0]]
        for prediction, teacher in residual._views(values, [head[0] for head in cache.teachers]).values():
            losses.append(((prediction - teacher).square().sum(-1) /
                           teacher.square().sum(-1).clamp_min(1e-20)).mean())
    expected_loss = torch.stack(losses).mean()
    expected_loss.backward()
    gradient = -(correction.grad - correction.grad.T) * .5
    rms = float(torch.linalg.vector_norm(gradient.double())) / gradient.shape[0] ** .5
    torch.testing.assert_close(actual["unit_tangent"], gradient / rms, atol=2e-6, rtol=2e-5)
    assert actual["training_objective"] == pytest.approx(float(expected_loss.detach()), rel=1e-12)
    assert actual["gradient_rms"] == pytest.approx(rms, rel=2e-5)
    assert torch.equal(actual["unit_tangent"], -actual["unit_tangent"].T)


@pytest.fixture(scope="module")
def completed():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    baseline, train, validation, policy = case(splits=1)
    result = residual.select_native_residual("layer", baseline, train, validation, policy)
    torch.set_num_threads(old)
    return baseline, train, validation, policy, result


def test_real_candidates_are_bounded_and_verified_with_original_strict_gate(completed):
    baseline, train, validation, policy, (selected, before, after, decision, proposal) = completed
    assert before == _metrics(baseline, validation, policy)
    assert decision["train_baseline"] == _metrics(baseline, train, policy)
    assert after == _metrics(selected, validation, policy)
    assert decision["train_candidate_count"] == 10
    assert decision["gradient_evaluations"] == 4
    assert decision["validation_candidate_count"] <= 5
    # This small, disclosed synthetic case has a feasible actual-forward step.
    assert decision["accepted"]
    assert nonworse(before, after, **residual.STRICT)
    assert after["squared_error"] < before["squared_error"]
    assert not decision["test_used_for_selection"]
    json.dumps(decision, allow_nan=False)
    assert matrix_digest({"layer": selected}) == decision["selected_constants_sha256"]
    assert sum(not torch.equal(selected[side][head], baseline[side][head])
               for head in range(2) for side in ("Rk", "Rv")) == 1
    assert torch.equal(selected["k_mean"], baseline["k_mean"])
    for row in decision["evaluated_validation_candidates"]:
        parent = next(item for item in decision["evaluated_train_candidates"] if item["candidate"] == row["candidate"])
        assert parent["accepted"]
        assert row["constants_sha256"] == parent["constants_sha256"]
    assert residual.validate_decision("layer", before, after, decision, 2, policy)
    assert proposal["train_samples_sha256"] == decision["train_samples_sha256"]
    # The STE direction does not promise actual descent, and a lower total
    # does not promise every query/view. Both failures occur in this real replay.
    train_error = decision["train_baseline"]["squared_error"]
    assert any(not row["accepted"] and row["train"]["squared_error"] > train_error
               for row in decision["evaluated_train_candidates"])
    assert any(not row["accepted"] and row["train"]["squared_error"] < train_error
               for row in decision["evaluated_train_candidates"])


def test_zero_gradient_is_exact_noop_and_never_false_improvement():
    baseline, train, validation, policy = case(splits=1)
    # Identical captured BF16 output makes every teacher residual exactly zero.
    for head, samples in enumerate(train):
        for sample in samples:
            sample["teacher"] = replay_sample(baseline["Rk"][head], baseline["Rv"][head],
                baseline["k_mean"][head], sample, {**policy, "replay_partition": "single_split"}).bfloat16()
    selected, before, after, decision, _ = residual.select_native_residual("layer", baseline, train, validation, policy)
    assert not decision["accepted"] and before == after
    assert all(row["zero_gradient"] and row["gradient_rms"] == 0 for row in decision["directions"])
    assert all(row["same_bf16_operand_as_seed"] and not row["accepted"] for row in decision["evaluated_train_candidates"])
    assert decision["evaluated_validation_candidates"] == []
    assert matrix_digest({"layer": selected}) == matrix_digest({"layer": baseline})


@pytest.mark.parametrize("heads", [1, 2, 3, 4])
def test_all_supported_head_budgets_and_zero_teacher_energy(heads):
    baseline, train, validation, policy = case(splits=1)
    baseline = {**baseline, **{key: torch.stack([baseline[key][index % 2].clone() for index in range(heads)])
                              for key in ("Rk", "Rv", "k_mean")}}
    train = [copy.deepcopy(train[index % 2]) for index in range(heads)]
    validation = [copy.deepcopy(validation[index % 2]) for index in range(heads)]
    for split in (train, validation):
        for head, samples in enumerate(split):
            for sample in samples:
                sample["sample_sha256"] = f"{2000 + head:064x}"
                sample["wo"].zero_()
    # Entry from a caller's no_grad context must still compute the intended
    # offline gradients (all truly zero here), never report missing autograd.
    with torch.no_grad():
        selected, before, after, decision, proposal = residual.select_native_residual(
            "layer", baseline, train, validation, policy)
    assert before["teacher_energy"] == before["squared_error"] == 0.
    assert before == after and not decision["accepted"]
    assert len(decision["directions"]) == heads * 2
    assert decision["train_candidate_count"] == heads * 4 + 2
    assert decision["validation_candidate_count"] == 0
    assert all(row["zero_gradient"] for row in decision["directions"])
    assert residual.validate_selection("layer", baseline, selected, before, after, decision, proposal, policy)


@pytest.mark.parametrize("side", ["Rk", "Rv"])
def test_zero_teacher_energy_nonzero_error_has_finite_normalized_direction(side):
    baseline, train, _, policy = case()
    for samples in train:
        for sample in samples:
            sample["teacher"].zero_()
    with torch.no_grad():
        cache = ProjectionCache(baseline, train, policy)
    assert cache.before["teacher_energy"] == 0 and cache.before["squared_error"] > 0
    direction = residual._gradient_direction(baseline, 0, side, train, policy, cache)
    assert direction["gradient_rms"] > 0
    assert torch.isfinite(direction["unit_tangent"]).all()
    assert float(torch.linalg.vector_norm(direction["unit_tangent"].double())) / 8 ** .5 == pytest.approx(1.)


def test_finite_forward_with_overflowing_backward_gradient_fails_closed():
    baseline, train, _, policy = case()
    for samples in train:
        for sample in samples:
            sample["teacher"].zero_()
            sample["wo"] *= 1e25
    with torch.no_grad():
        cache = ProjectionCache(baseline, train, policy)
    assert cache.before["squared_error"] > 0
    with pytest.raises(ValueError, match="nonfinite.*gradient"):
        residual._gradient_direction(baseline, 0, "Rk", train, policy, cache)


def test_resume_reconstructs_identity_without_fit_replay_or_independent_test(monkeypatch, completed):
    baseline, _, _, policy, (selected, before, after, decision, proposal) = completed
    import oscar_ascend.optr_calibration as calibration

    def forbidden(*args, **kwargs):
        pytest.fail("locked native residual validation must not fit or replay")
    monkeypatch.setattr(residual, "_gradient_direction", forbidden)
    monkeypatch.setattr(calibration, "replay_sample", forbidden)
    monkeypatch.setattr(calibration, "_replay", forbidden)
    assert residual.validate_selection("layer", baseline, selected, before, after, decision, proposal, policy)


@pytest.mark.parametrize("fault", ["accept", "radius", "coverage", "winner", "source", "gradient_hash",
                                   "tensor", "noop", "head_count", "test_selection", "metric", "val_coverage",
                                   "nonfinite_tensor", "gradient_zero", "nonskew", "normalization", "direction_order",
                                   "selected_matrix", "train_winner", "nonjson", "candidate_count"])
def test_tampered_locked_selection_is_rejected(completed, fault):
    baseline, _, _, policy, result = completed
    selected, before, after, decision, proposal = copy.deepcopy(result)
    if fault == "accept":
        decision["evaluated_train_candidates"][0]["accepted"] = not decision["evaluated_train_candidates"][0]["accepted"]
    elif fault == "radius":
        decision["evaluated_train_candidates"][0]["radius"] *= 2
    elif fault == "coverage":
        decision["evaluated_train_candidates"].pop()
    elif fault == "winner":
        decision["selected_candidate"] = "seed"
    elif fault == "source":
        decision["train_sources"][0][0]["sample_sha256"] = "f" * 64
    elif fault == "gradient_hash":
        decision["directions"][0]["unit_tangent_sha256"] = "f" * 64
    elif fault == "tensor":
        proposal["directions"][0]["unit_tangent"][0, 1] += .1
    elif fault == "noop":
        decision["evaluated_train_candidates"][0]["same_bf16_operand_as_seed"] = True
    elif fault == "head_count":
        decision["directions"].pop()
    elif fault == "test_selection":
        decision["test_used_for_selection"] = True
    elif fault == "metric":
        decision["evaluated_train_candidates"][0]["train"]["requests"][0]["sample_sha256"] = "f" * 64
    elif fault == "val_coverage":
        decision["evaluated_validation_candidates"].clear()
    elif fault == "nonfinite_tensor":
        proposal["directions"][0]["unit_tangent"][0, 1] = float("nan")
    elif fault == "gradient_zero":
        decision["directions"][0]["zero_gradient"] = True
    elif fault == "nonskew":
        tangent = proposal["directions"][0]["unit_tangent"]
        tangent[0, 0] = .1
        new_hash = residual._tensor_digest(tangent)
        proposal["directions"][0]["unit_tangent_sha256"] = new_hash
        decision["directions"][0]["unit_tangent_sha256"] = new_hash
    elif fault == "normalization":
        tangent = proposal["directions"][0]["unit_tangent"]
        tangent *= 2
        new_hash = residual._tensor_digest(tangent)
        proposal["directions"][0]["unit_tangent_sha256"] = new_hash
        decision["directions"][0]["unit_tangent_sha256"] = new_hash
    elif fault == "direction_order":
        decision["directions"] = decision["directions"][::-1]
    elif fault == "selected_matrix":
        selected["Rv"][0, :, :2] *= -1
    elif fault == "train_winner":
        winner = next(row for row in decision["directions"] if row["train_selected_candidate"] != "seed")
        winner["train_selected_candidate"] = "seed"
    elif fault == "nonjson":
        decision["extra"] = torch.tensor(1.)
    elif fault == "candidate_count":
        decision["train_candidate_count"] = True
    with pytest.raises(ValueError):
        residual.validate_selection("layer", baseline, selected, before, after, decision, proposal, policy)


def test_train_and_validation_overlap_or_independent_test_input_is_rejected():
    baseline, train, validation, policy = case()
    with pytest.raises(ValueError, match="split"):
        residual.select_native_residual("layer", baseline, validation, train, policy)
    for head in validation:
        head[0]["request_id"] = "train"
    with pytest.raises(ValueError, match="overlap"):
        residual.select_native_residual("layer", baseline, train, validation, policy)


@pytest.mark.parametrize("fault", ["missing_head", "not_object", "empty_requests", "duplicate_requests", "too_many_heads"])
def test_malformed_training_geometry_fails_before_any_gradient(monkeypatch, fault):
    baseline, train, validation, policy = case()
    if fault == "missing_head":
        train.pop()
    elif fault == "not_object":
        train[0][0] = None
    elif fault == "empty_requests":
        train[0].clear()
    elif fault == "duplicate_requests":
        for head in train:
            head.append(copy.deepcopy(head[0]))
    elif fault == "too_many_heads":
        baseline = {**baseline, **{key: torch.stack([baseline[key][index % 2] for index in range(5)])
                                  for key in ("Rk", "Rv", "k_mean")}}
    def forbidden(*args, **kwargs):
        pytest.fail("invalid sample geometry must be rejected before gradients")
    monkeypatch.setattr(residual, "_gradient_direction", forbidden)
    with pytest.raises(ValueError):
        residual.select_native_residual("layer", baseline, train, validation, policy)


def test_independent_head_proxy_and_positive_radius_can_improve_total_while_regressing_query():
    target = torch.tensor([1., 1.])
    baseline = torch.tensor([0., 3.])
    local = torch.tensor([1., 3.])
    assert (local[0] - target[0]).square() < (baseline[0] - target[0]).square()
    assert (local.sum() - target.sum()).square() > (baseline.sum() - target.sum()).square()
    for radius in residual.NATIVE_RESIDUAL_POLICY["radii"]:
        errors = ((1 + radius) ** 2, (1 - 2 * radius) ** 2)
        assert sum(errors) < 2 and errors[0] > 1


@pytest.mark.parametrize("fault", ["joint_scope", "joint_radius", "joint_hash", "joint_noop", "joint_winner", "joint_missing"])
def test_joint_candidate_lock_authenticates_fixed_grid_and_full_matrix(completed, fault):
    baseline, _, _, policy, result = completed
    selected, before, after, decision, proposal = copy.deepcopy(result)
    row = decision["evaluated_train_candidates"][-1]
    if fault == "joint_scope":
        row["scope"] = "single"
    elif fault == "joint_radius":
        row["radius"] *= 2
    elif fault == "joint_hash":
        row["constants_sha256"] = "f" * 64
    elif fault == "joint_noop":
        row["same_bf16_operand_as_seed"] = not row["same_bf16_operand_as_seed"]
    elif fault == "joint_winner":
        decision["joint_train_selected_candidate"] = "invalid"
    else:
        decision["evaluated_train_candidates"].pop()
    with pytest.raises(ValueError):
        residual.validate_selection("layer", baseline, selected, before, after, decision, proposal, policy)
