"""Archive #172/#173/#176: bounded candidates preserve fitting and resume identity."""
import copy
import math

import pytest
import torch

from oscar_ascend.optr_candidates import ValueCandidates, alternatives, validate_candidates
from oscar_ascend.optr_shared import fit_shared
from test_optr_upstream_parity import inputs, options


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_retaining_existing_checkpoints_does_not_change_original_fit_or_proxy_calls(monkeypatch):
    from oscar_ascend import optr_shared as shared
    heads, dim = 2, 8
    train, _, _ = inputs(191, "train", heads=heads)
    held, _, _ = inputs(197, "held", heads=heads)
    for head in range(heads):
        held[head]["wo"] = train[head]["wo"]
    base = torch.eye(dim).repeat(heads, 1, 1)
    mean = torch.zeros(heads, dim)
    policy = options(steps=3, eval_every=1)
    calls = []
    original = shared._score
    def score(*args, **kwargs):
        calls.append(args[2])
        return original(*args, **kwargs)
    monkeypatch.setattr(shared, "_score", score)
    args = (base, base, mean, [[row] for row in train], [[row] for row in held])
    expected = fit_shared(*args, policy=policy)
    expected_calls = calls[:]
    calls.clear()
    collector = ValueCandidates(policy)
    actual = fit_shared(*args, policy=policy, value_candidate_callback=collector.observe)
    assert calls == expected_calls
    for tensor, reference in zip(actual[:2], expected[:2]):
        assert torch.equal(tensor, reference)
    assert actual[2] == expected[2]
    result = dict(Rk=actual[0], Rv=actual[1], decision=actual[2])
    values = validate_candidates(collector.payload(actual[0]), result,
        dict(Rk=base, Rv=base, k_mean=mean), policy)
    assert [row["step"] for row in values[:1]] == [0]
    assert sorted(row["step"] for row in values[1:]) == [1, 2, 3]


def dense_case():
    policy = options(steps=160, eval_every=1)
    base = torch.eye(2).repeat(2, 1, 1)
    baseline = dict(Rk=base, Rv=base, k_mean=torch.zeros(2, 2))
    collector = ValueCandidates(policy)
    collector.observe(step=0, rotation=base[0], proxy_loss=1.)
    scored = []
    for step in range(1, 161):
        angle = step / 300.
        rotation = torch.tensor([[math.cos(angle), -math.sin(angle)],
                                 [math.sin(angle), math.cos(angle)]], requires_grad=True)
        loss = 2 - step / 1000.
        collector.observe(step=step, rotation=rotation, proxy_loss=loss)
        scored.append(dict(step=step, proxy_loss=loss, accepted=False))
        # Retention must own a detached snapshot, not this caller's storage.
        with torch.no_grad():
            rotation.zero_()
    decision = dict(policy=policy, phases=[dict(phase="k"), dict(phase="v",
        baseline_proxy_loss=1., selected_proxy_loss=1., accepted=False, candidates=scored)])
    result = dict(Rk=base, Rv=base, decision=decision)
    return collector.payload(base), result, baseline, policy


def test_dense_research_schedule_is_bounded_and_keeps_initial_value_even_when_all_proxies_worsen():
    payload, result, baseline, policy = dense_case()
    values = validate_candidates(payload, result, baseline, policy)
    assert [row["step"] for row in values] == [0, 160, 159, 158, 157, 156]
    assert sum(row["Rv"].numel() for row in values) == 6 * 2 * 2
    assert all(not row["Rv"].requires_grad for row in values)
    assert [row["step"] for row in alternatives(values, result["Rv"])] == [160, 159, 158, 157, 156]


@pytest.mark.parametrize("fault", ["missing", "key", "seed", "step", "score", "nan", "coverage",
                                   "schedule", "primary", "nonshared_shape", "policy"])
def test_resumable_candidate_pool_cannot_change_its_fit_identity(fault):
    payload, result, baseline, policy = dense_case()
    if fault == "missing": payload = None
    elif fault == "key": payload["fixed_key_sha256"] = "f" * 64
    elif fault == "seed": payload["values"][0]["Rv"].neg_()
    elif fault == "step": payload["values"][1]["step"] = 159
    elif fault == "score": payload["values"][1]["proxy_loss"] += .1
    elif fault == "nan": payload["values"][1]["Rv"][0, 0] = float("nan")
    elif fault == "coverage": payload["values"].pop()
    elif fault == "schedule": result["decision"]["phases"][1]["candidates"].pop()
    elif fault == "primary": result["Rv"] = -result["Rv"]
    elif fault == "nonshared_shape": payload["values"][1]["Rv"] = payload["values"][1]["Rv"][None]
    elif fault == "policy": result["decision"]["policy"] = {**policy, "steps": 80}
    with pytest.raises(ValueError):
        validate_candidates(payload, result, baseline, policy)


def test_duplicate_values_are_never_replayed_or_allowed_to_mutate_saved_candidates():
    payload, result, baseline, policy = dense_case()
    values = validate_candidates(payload, result, baseline, policy)
    values.append(copy.deepcopy(values[1]))
    values.append(copy.deepcopy(values[0]))
    distinct = list(alternatives(values, result["Rv"]))
    assert len(distinct) == 5
    assert all(row["step"] != 0 for row in distinct)


@pytest.mark.parametrize("primary_passes", [True, False])
def test_resume_search_stops_at_first_pass_and_independent_test_never_reselects(tmp_path, monkeypatch, primary_passes):
    """Synthetic metrics isolate control flow; the other test runs real fitting."""
    from oscar_ascend.optr_candidates import rotation_digest
    from oscar_ascend.optr_checkpoint import FitCheckpoints
    from tools import prepare_optr_rotations as prep
    from test_optr_integration import metric
    payload, result, baseline, policy = dense_case()
    baseline["layer_id"] = 3
    result["Rk"] = -result["Rk"]
    payload["fixed_key_sha256"] = rotation_digest(result["Rk"][0])
    result["decision"]["accepted"] = True
    result["value_candidates"] = payload
    store = FitCheckpoints(tmp_path, {"test_only": True, "source": "control_flow_fixture"})
    store.save("layer-3-shared", result)
    calls = []
    first, passing = (row["Rv"] for row in payload["values"][1:3])

    def measured(constants, samples, policy, **kwargs):
        split = samples[0][0]["request_id"]
        if torch.equal(constants["Rk"], baseline["Rk"]):
            label, error = "seed", 1.
        elif torch.equal(constants["Rv"], result["Rv"]):
            label, error = "primary", .5 if primary_passes else 2.
        elif torch.equal(constants["Rv"][0], first):
            label, error = "first", 1.1
        elif torch.equal(constants["Rv"][0], passing):
            label, error = "passing", .5
        else:
            raise AssertionError("search continued after a validation pass")
        calls.append((split, label))
        # Deliberate independent-test regression must be reported, not used
        # to try the next candidate or to select the seed after seeing test.
        if split == "test" and label != "seed":
            error = 2.
        return metric(split, error)

    monkeypatch.setattr(prep, "_metrics", measured)
    samples = {split: [[{"request_id": split}]] for split in ("train", "validation", "test")}
    selected, record = prep.refine_layer("layer", baseline, samples, policy,
        checkpoints=store, progress=lambda **event: None)
    expected = [("validation", "seed"), ("validation", "primary")]
    if not primary_passes:
        expected += [("validation", "first"), ("validation", "passing")]
    expected += [("test", "seed"), ("test", "primary" if primary_passes else "passing")]
    assert calls == expected
    assert record["accepted"]
    assert record["test"]["selected"]["squared_error"] > record["test"]["baseline"]["squared_error"]
    assert torch.equal(selected["Rv"][0], result["Rv"][0] if primary_passes else passing)
    assert record["shared_layer_selection"]["test_used_for_selection"] is False
