"""Archive #170/#174/#175/#176, G26: bounded repair without weaker gates.

Synthetic CPU evidence only. Neither a passing local example nor cache
equivalence establishes NPU/model quality or serving performance.
"""
import copy
import inspect
import weakref

import pytest
import torch

from oscar_ascend import optr_calibration as calibration
from oscar_ascend import optr_recipe3_fit as recipe3
from oscar_ascend import optr_trust_region as repair
from oscar_ascend.optr_artifacts import matrix_digest, nonworse
from oscar_ascend.optr_policy import optr_policy
from test_optr_calibration import trace
from test_optr_recipe3_fit import recall_case
from tools.prepare_optr_rotations import _metrics


STRICT = dict(require_frames=True, require_terminal=True, require_queries=True, require_projection=True)


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def rotation(dim, amount):
    tangent = torch.zeros(dim, dim)
    tangent[0, 1], tangent[1, 0] = amount, -amount
    return torch.matrix_exp(tangent)


def basic_case(*, dim=8, heads=2, splits=1):
    rng = torch.Generator().manual_seed(735)
    rk = torch.linalg.qr(torch.randn(heads, dim, dim, generator=rng)).Q.contiguous()
    rv = torch.linalg.qr(torch.randn(heads, dim, dim, generator=rng)).Q.contiguous()
    base = {"layer_id": 3, "Rk": rk, "Rv": rv,
            "k_mean": torch.randn(heads, dim, generator=rng) * .2}
    samples = []
    for head in range(heads):
        sample = trace(90 + head, "validation", length=541, dim=dim, queries=3, group=6)
        sample.update(positions=torch.tensor([514, 515, 540]), contexts=torch.tensor([0, 515, 515]),
            current_ends=torch.tensor([515, 541, 541]), frame_ids=torch.tensor([0, 1, 1]),
            token_ids=torch.arange(541), history_splits=torch.tensor([1, splits, splits]),
            sample_sha256=f"{90 + head:064x}")
        scores = torch.einsum("mgd,ld->mgl", sample["q"].float(), sample["k"].float()) * dim ** -.5
        scores.masked_fill_(torch.arange(541)[None, None] > sample["positions"][:, None, None], -torch.inf)
        sample["teacher"] = torch.einsum("mgl,ld->mgd", scores.softmax(-1), sample["v"].float()).bfloat16()
        samples.append([sample])
    policy = {**optr_policy({}), "sink_tokens": 2, "recent_tokens": 4,
              "recipe3_repair": copy.deepcopy(repair.REPAIR_POLICY)}
    return base, samples, policy


def source_result(base):
    dim = base["Rk"].shape[-1]
    rk, rv = base["Rk"] @ rotation(dim, .2), base["Rv"] @ rotation(dim, -.2)
    return {"Rk": rk, "Rv": rv, "decision": {}, "candidates": [
        {"id": "primary", "phase": "primary", "step": None, "Rk": rk, "Rv": rv},
        {"id": "k_step_1", "phase": "k", "step": 1, "Rk": rk, "Rv": base["Rv"]},
        {"id": "v_step_1", "phase": "v", "step": 1, "Rk": rk, "Rv": rv},
    ]}


@pytest.mark.parametrize("dim,splits", [(8, 1), (8, 2), (256, 2)])
def test_projection_cache_exactly_matches_all_partition_query_views_and_releases_work(monkeypatch, dim, splits):
    base, samples, policy = basic_case(dim=dim, splits=splits)
    candidate = {**base, "Rv": base["Rv"].clone()}
    candidate["Rv"][1] = candidate["Rv"][1] @ rotation(dim, .003)
    expected_before, expected_after = _metrics(base, samples, policy), _metrics(candidate, samples, policy)
    calls, projected = [], []
    replay, project = calibration.replay_sample, calibration.project_output

    def count(*args, **kwargs):
        calls.append(1)
        return replay(*args, **kwargs)

    def track(*args, **kwargs):
        value = project(*args, **kwargs)
        projected.append(weakref.ref(value))
        return value

    monkeypatch.setattr(calibration, "replay_sample", count)
    monkeypatch.setattr(calibration, "project_output", track)
    cache = repair.ProjectionCache(base, samples, policy)
    partitions = 1 if splits == 1 else 2
    assert len(calls) == 2 * partitions
    assert cache.before == expected_before
    originals = {key: [[value.clone() for value in head] for head in outputs]
                 for key, outputs in cache.outputs.items()}
    persistent_count = len(projected)
    assert cache.measure(candidate, 1) == expected_after
    assert len(calls) == 3 * partitions  # Exactly one changed head per partition.
    assert all(reference() is not None for reference in projected[:persistent_count])
    assert all(reference() is None for reference in projected[persistent_count:])
    for key, outputs in cache.outputs.items():
        assert all(torch.equal(value, original) for head, old in zip(outputs, originals[key])
                   for value, original in zip(head, old))
    assert (cache.outputs["single_split"] is cache.outputs["captured_runtime_frames"]) == (splits == 1)


def test_source_order_uses_earliest_nonzero_phase_step_then_primary_without_scores():
    base, _, _ = basic_case()
    source = source_result(base)
    early = source["candidates"][1]
    early["Rk"] = early["Rk"].clone()
    early["Rk"][0] = base["Rk"][0]  # This head alone has no step-1 K direction.
    late = {**early, "id": "k_step_21", "step": 21, "Rk": base["Rk"] @ rotation(8, .4)}
    source["candidates"][2]["Rv"] = base["Rv"]  # V falls back to primary.
    source["candidates"].append(late)
    source["candidates"].reverse()
    directions = repair._source_directions(base, source)
    assert [(row["head"], row["side"], row["source_id"]) for row in directions] == [
        (0, "Rk", "k_step_21"), (0, "Rv", "primary"),
        (1, "Rk", "k_step_1"), (1, "Rv", "primary")]
    for row in directions:
        tangent = row["unit_tangent"]
        assert torch.equal(tangent, -tangent.T)
        assert float(torch.linalg.vector_norm(tangent)) / 8 ** .5 == pytest.approx(1.)


@pytest.fixture(scope="module")
def nonzero_prefill_case():
    base, samples, policy = recall_case()
    for split in samples.values():
        for head in split:
            head[0]["gate"][0].fill_(.5)
    policy = {**policy, "recipe3_repair": copy.deepcopy(repair.REPAIR_POLICY)}
    source = recipe3.fit_perhead(base["Rk"], base["Rv"], base["k_mean"], samples["train"],
                                samples["validation"], policy=policy)
    return base, samples, policy, source


def test_existing_failed_checkpoints_produce_a_strict_repair_with_nonzero_prefill(monkeypatch, nonzero_prefill_case):
    base, samples, policy, source = nonzero_prefill_case
    base_hash, source_hash = matrix_digest({"layer": base}), matrix_digest({"layer": {**base, **source}})
    before = _metrics(base, samples["validation"], policy)
    for row in source["candidates"]:
        candidate = {**base, "Rk": row["Rk"], "Rv": row["Rv"]}
        measured = _metrics(candidate, samples["validation"], policy)
        assert not (nonworse(before, measured, **STRICT) and measured["squared_error"] < before["squared_error"])

    def forbidden(*args, **kwargs):
        pytest.fail("Repair must not launch another fit or consume training/test samples")

    monkeypatch.setattr(recipe3, "fit_head", forbidden)
    monkeypatch.setattr(recipe3, "fit_perhead", forbidden)
    assert "test" not in inspect.signature(repair.select_repair).parameters
    events = []
    selected, cached_before, after, decision = repair.select_repair("layer", base, source,
        samples["validation"], policy, progress=lambda **event: events.append(event))
    assert cached_before == before
    assert after == _metrics(selected, samples["validation"], policy)
    assert decision["accepted"] and not decision["test_used_for_selection"]
    assert nonworse(before, after, **STRICT)
    assert before["squared_error"] == pytest.approx(.3173715594330846, rel=1e-5)
    assert after["squared_error"] == pytest.approx(.1843001311584509, rel=1e-5)
    assert decision["candidate_count"] == decision["max_candidates"] == 32
    passing = [row for row in decision["evaluated_candidates"] if row["accepted"]]
    assert any(row["source_id"] == "k_step_1" and row["head"] == 1 and row["side"] == "Rk"
               and row["radius"] == .00125 for row in passing)
    assert after["squared_error"] == min(row["validation"]["squared_error"] for row in passing)
    # Test is evaluated only AFTER the validation selection is complete. It
    # cannot change the selected radius/head/side and is never passed to repair.
    test_before, test_after = (_metrics(constants, samples["test"], policy) for constants in (base, selected))
    assert nonworse(test_before, test_after, **STRICT)
    assert test_before["squared_error"] == pytest.approx(1.9121566627677629, rel=1e-5)
    assert test_after["squared_error"] == pytest.approx(1.8243649711858363, rel=1e-5)
    assert decision["base_constants_sha256"] == base_hash == matrix_digest({"layer": base})
    assert source_hash == matrix_digest({"layer": {**base, **source}})
    assert decision["selected_constants_sha256"] == matrix_digest({"layer": selected})
    assert torch.equal(base["k_mean"], selected["k_mean"])
    assert torch.equal(base["Rv"], selected["Rv"])
    assert torch.equal(base["Rk"][0], selected["Rk"][0])
    assert len([event for event in events if event["event"] == "repair_candidate_checked"]) == 32


def test_maximum_budget_is_64_and_every_proposal_changes_one_side_of_one_head(monkeypatch):
    base, samples, policy = basic_case(heads=4)
    source = source_result(base)
    original_measure = repair.ProjectionCache.measure
    seen = []

    def inspect_candidate(cache, constants, head):
        changed = [(side, h) for side in ("Rk", "Rv") for h in range(4)
                   if not torch.equal(constants[side][h], base[side][h])]
        assert len(changed) == 1 and changed[0][1] == head
        assert constants["k_mean"] is base["k_mean"]
        recipe3.validate_perhead_pair(constants["Rk"], constants["Rv"], constants["k_mean"])
        seen.append(changed[0])
        return original_measure(cache, constants, head)

    monkeypatch.setattr(repair.ProjectionCache, "measure", inspect_candidate)
    _, _, _, decision = repair.select_repair("layer", base, source, samples, policy)
    assert decision["candidate_count"] == decision["max_candidates"] == 64
    assert len(seen) + sum(row["same_bf16_operand_as_seed"] for row in decision["evaluated_candidates"]) == 64
    for head in range(4):
        for side in ("Rk", "Rv"):
            rows = [row for row in decision["evaluated_candidates"] if row["head"] == head and row["side"] == side]
            assert [row["radius"] for row in rows] == repair.REPAIR_POLICY["radii"]
            assert all(len(row["constants_sha256"]) == 64 and "projection_checks" in row["validation"] for row in rows)


def test_no_nonzero_direction_keeps_recipe3_without_pseudo_improvement():
    base, samples, policy = basic_case()
    source = {"Rk": base["Rk"], "Rv": base["Rv"], "decision": {}, "candidates": [
        {"id": "primary", "phase": "primary", "step": None, "Rk": base["Rk"], "Rv": base["Rv"]}]}
    selected, before, after, decision = repair.select_repair("layer", base, source, samples, policy)
    assert selected is base and before == after
    assert not decision["accepted"] and decision["selected_candidate"] == "seed"
    assert decision["candidate_count"] == 0


@pytest.fixture(scope="module")
def locked_selection(nonzero_prefill_case):
    base, samples, policy, source = nonzero_prefill_case
    selected, before, after, decision = repair.select_repair("layer", base, source,
        samples["validation"], policy)
    return base, source, selected, before, after, decision, policy


def test_locked_selection_authentication_does_not_replay_fit_or_access_test(monkeypatch, locked_selection):
    def forbidden(*args, **kwargs):
        pytest.fail("Locked-selection authentication must not repeat search/replay/fit")

    monkeypatch.setattr(calibration, "replay_sample", forbidden)
    monkeypatch.setattr(recipe3, "fit_perhead", forbidden)
    monkeypatch.setattr(repair, "select_repair", forbidden)
    assert repair.validate_selection("layer", *locked_selection)


@pytest.mark.parametrize("fault", ["radius", "missing_candidate", "candidate_id", "accepted",
    "candidate_digest", "selected_digest", "source", "mean", "other_head", "after",
    "test_selection", "partition", "projection", "baseline"])
def test_locked_selection_rejects_tampering(fault, locked_selection):
    base, source, selected, before, after, decision, policy = copy.deepcopy(locked_selection)
    rows = decision["evaluated_candidates"]
    if fault == "radius":
        rows[0]["radius"] *= .5
    elif fault == "missing_candidate":
        rows.pop()
        decision["candidate_count"] -= 1
    elif fault == "candidate_id":
        rows[0]["candidate"] = rows[1]["candidate"]
    elif fault == "accepted":
        rows[0]["accepted"] = not rows[0]["accepted"]
    elif fault == "candidate_digest":
        rows[0]["constants_sha256"] = "0" * 64
    elif fault == "selected_digest":
        decision["selected_constants_sha256"] = "0" * 64
    elif fault == "source":
        source["candidates"][-1]["Rv"] = source["candidates"][-1]["Rv"] @ rotation(8, .001)
    elif fault == "mean":
        selected["k_mean"] = selected["k_mean"].clone() + .001
    elif fault == "other_head":
        selected["Rv"] = selected["Rv"].clone()
        selected["Rv"][0] = selected["Rv"][0] @ rotation(8, .001)
    elif fault == "after":
        after = before
    elif fault == "test_selection":
        decision["test_used_for_selection"] = True
    elif fault == "partition":
        rows[0]["validation"]["requests"][0]["partition"] = "captured_runtime_frames"
    elif fault == "projection":
        rows[0]["validation"].pop("projection_checks")
    elif fault == "baseline":
        base["k_mean"] = base["k_mean"].clone() + .001
    with pytest.raises(ValueError):
        repair.validate_selection("layer", base, source, selected, before, after, decision, policy)


@pytest.mark.parametrize("fault", ["policy", "projection", "source_tensor", "source_phase", "empty_samples", "test_samples"])
def test_invalid_inputs_fail_before_projection(monkeypatch, fault):
    base, samples, policy = basic_case()
    source = source_result(base)
    if fault == "policy":
        policy["recipe3_repair"]["radii"].append(.1)
    elif fault == "projection":
        policy["validation_projection_views"] = ["fp32"]
    elif fault == "source_tensor":
        source.pop("Rk")
    elif fault == "source_phase":
        source["candidates"][1]["step"] = -1
    elif fault == "empty_samples":
        samples[0] = []
    elif fault == "test_samples":
        samples[0][0]["split"] = "test"

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid input reached replay")

    monkeypatch.setattr(repair, "ProjectionCache", forbidden)
    with pytest.raises(ValueError):
        repair.select_repair("layer", base, source, samples, policy)
