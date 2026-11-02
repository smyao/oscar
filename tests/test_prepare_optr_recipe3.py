"""Whole-layer recipe3 selection and mandatory independent-test acceptance."""
import copy
import json

import pytest
import torch

from oscar_ascend.optr_checkpoint import FitCheckpoints
from oscar_ascend.optr_policy import optr_policy
from oscar_ascend.optr_recipe3_fit import CANDIDATE_STRATEGY
from tools import prepare_optr_recipe3 as recipe3
from tools import prepare_optr_rotations as orchestrator
from test_optr_integration import metric


NAME = "model.layers.3.self_attn.attn"


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def case():
    base = {"layer_id": 3, "Rk": torch.eye(4).repeat(2, 1, 1),
            "Rv": -torch.eye(4).repeat(2, 1, 1), "k_mean": torch.full((2, 4), .15)}
    policy = {**optr_policy({"optr_baseline": "recipe3"}), "steps": 3, "eval_every": 2}
    primary_k = base["Rk"].roll(1, dims=-1)
    candidates = [
        {"id": "primary", "phase": "primary", "step": None,
         "Rk": primary_k, "Rv": base["Rv"], "proxy_losses": None},
        {"id": "k_step_1", "phase": "k", "step": 1,
         "Rk": base["Rk"].roll(2, dims=-1), "Rv": base["Rv"], "proxy_losses": [.3, .4]},
        {"id": "v_step_3", "phase": "v", "step": 3,
         "Rk": primary_k, "Rv": base["Rv"].roll(1, dims=-1), "proxy_losses": [.2, .3]},
    ]
    result = {"Rk": primary_k, "Rv": base["Rv"], "candidates": candidates,
        "decision": {"accepted": True, "policy": policy, "fitted_grouping": "layer_kv_head",
                     "deployment_candidate_selection": CANDIDATE_STRATEGY,
                     "candidate_steps": [0, 1, 3], "candidate_count": 3}}
    samples = {split: [[{"request_id": split}] for _ in range(2)]
               for split in ("train", "validation", "test")}
    return base, samples, policy, result


def install_metrics(monkeypatch, base, result, *, validation=(.7, .2, .5), test=.1):
    calls = []

    def measured(constants, rows, policy, **kwargs):
        split = rows[0][0]["request_id"]
        if all(torch.equal(constants[key], base[key]) for key in ("Rk", "Rv")):
            identity, error = "seed", 1.
        else:
            index = next(i for i, row in enumerate(result["candidates"])
                         if all(torch.equal(constants[key], row[key]) for key in ("Rk", "Rv")))
            identity, error = result["candidates"][index]["id"], validation[index] if split == "validation" else test
        calls.append((split, identity))
        return metric(split, error)

    monkeypatch.setattr(orchestrator, "_metrics", measured)
    return calls


def test_all_candidates_face_strict_gate_and_best_validation_sse_wins(monkeypatch):
    base, samples, policy, result = case()
    calls = install_metrics(monkeypatch, base, result)
    selected, record = recipe3.refine_layer(NAME, base, samples, policy,
        fitter=lambda *_args, **_kwargs: result, progress=lambda **row: None)
    assert calls == [("validation", "seed"), ("validation", "primary"),
                     ("validation", "k_step_1"), ("validation", "v_step_3"),
                     ("test", "seed"), ("test", "k_step_1")]
    assert torch.equal(selected["Rk"], result["candidates"][1]["Rk"])
    assert torch.equal(selected["k_mean"], base["k_mean"])
    assert record["shared_layer_selection"]["selected_candidate"] == "k_step_1"
    assert len(record["shared_layer_selection"]["evaluated_candidates"]) == 3
    assert not record["shared_layer_selection"]["test_used_for_selection"]
    assert record["recipe3_baseline"]["recipe_version"] == 3


def test_independent_test_failure_is_persisted_and_never_reselects(tmp_path, monkeypatch):
    base, samples, policy, result = case()
    calls = install_metrics(monkeypatch, base, result, test=1.01)
    store = FitCheckpoints(tmp_path, {"test_only": True, "source": "recipe3_gate_fixture"})
    events = []
    with pytest.raises(orchestrator.IndependentTestRegression, match="test regressed") as failure:
        recipe3.refine_layer(NAME, base, samples, policy, fitter=lambda *_args, **_kwargs: result,
                            checkpoints=store, progress=lambda **row: events.append(row))
    report = json.loads((store.root / "layer-3-independent-test-failed.json").read_text())
    assert report == failure.value.report
    assert not report["artifact_publish_allowed"] and not report["test_used_for_selection"]
    assert report["record"]["shared_layer_selection"]["selected_candidate"] == "k_step_1"
    assert calls[-2:] == [("test", "seed"), ("test", "k_step_1")]
    assert len(calls) == 6  # No V fallback or baseline selection after observing test.
    assert store.load("layer-3") is None
    assert store.load("layer-3-shared") is not None
    assert all(not event["artifact_publish_allowed"] for event in events if "artifact_publish_allowed" in event)


def test_no_validation_candidate_keeps_original_recipe3_and_tests_that_baseline(monkeypatch):
    base, samples, policy, result = case()
    calls = install_metrics(monkeypatch, base, result, validation=(1., 1.1, 1.2))
    selected, record = recipe3.refine_layer(NAME, base, samples, policy,
        fitter=lambda *_args, **_kwargs: result, progress=lambda **row: None)
    assert selected is base and not record["accepted"]
    assert record["validation"]["selected"] == record["validation"]["baseline"]
    assert record["test"]["selected"] == record["test"]["baseline"]
    assert calls[-1] == ("test", "seed") and len(calls) == 5


@pytest.mark.parametrize("changed", [None, "baseline", "policy", "implementation"])
def test_resume_binds_baseline_policy_and_implementation_then_revalidates(tmp_path, monkeypatch, changed):
    base, samples, policy, result = case()
    calls = install_metrics(monkeypatch, base, result)
    store = FitCheckpoints(tmp_path, {"test_only": True, "trace_manifest_sha256": "a" * 64})
    fits = []

    def fit(*args, **kwargs):
        fits.append(True)
        return result

    recipe3.refine_layer(NAME, base, samples, policy, fitter=fit, checkpoints=store, progress=lambda **row: None)
    if changed == "baseline":
        base = {**base, "k_mean": base["k_mean"] + .001}
    elif changed == "policy":
        policy = {**policy, "lr": policy["lr"] / 2}
    elif changed == "implementation":
        monkeypatch.setattr(recipe3, "fitting_implementation_digest", lambda: "b" * 64)
    if changed:
        with pytest.raises(ValueError, match="resumed checkpoint"):
            recipe3.refine_layer(NAME, base, samples, policy, fitter=fit, checkpoints=store, progress=lambda **row: None)
        assert len(calls) == 6
    else:
        recipe3.refine_layer(NAME, base, samples, policy, fitter=fit, checkpoints=store, progress=lambda **row: None)
        assert calls[:6] == calls[6:]
    assert len(fits) == 1


def test_projection_regression_cannot_hide_behind_better_total_sse(monkeypatch):
    base, samples, policy, result = case()
    install_metrics(monkeypatch, base, result, validation=(.7, .2, .5))
    original = orchestrator._metrics

    def measured(constants, *args, **kwargs):
        value = original(constants, *args, **kwargs)
        if torch.equal(constants["Rk"], result["candidates"][1]["Rk"]):
            value["projection_checks"]["bf16_rank_output"] = metric("validation", 1.01)
            value["projection_checks"]["bf16_rank_output"].pop("projection_checks")
        return value

    monkeypatch.setattr(orchestrator, "_metrics", measured)
    _, record = recipe3.refine_layer(NAME, base, samples, policy,
        fitter=lambda *_args, **_kwargs: result, progress=lambda **row: None)
    assert record["shared_layer_selection"]["selected_candidate"] == "v_step_3"
    assert not record["shared_layer_selection"]["evaluated_candidates"][1]["accepted"]


def test_real_80_step_candidates_use_actual_replay_and_independent_test(tmp_path):
    from test_optr_recipe3_fit import recall_case
    base, samples, policy = recall_case()
    policy = {**optr_policy({"optr_baseline": "recipe3"}), "sink_tokens": 2, "recent_tokens": 4}
    store = FitCheckpoints(tmp_path, {"test_only": True, "source": "synthetic_bf16_qkv"})
    _, record = recipe3.refine_layer(NAME, base, samples, policy, checkpoints=store, progress=lambda **row: None)
    assert record["shared_layer_selection"]["selected_candidate"] == "k_step_1"
    assert record["validation"]["selected"]["squared_error"] == pytest.approx(.16930189719437183, rel=1e-4)
    assert record["validation"]["baseline"]["squared_error"] == pytest.approx(.3172262930602623, rel=1e-4)
    assert record["test"]["baseline"]["squared_error"] == pytest.approx(1.9119057461743831, rel=1e-4)
    assert record["test"]["selected"]["squared_error"] == pytest.approx(.37343135606924893, rel=1e-4)
    assert record["test"]["baseline"]["requests"][0]["request_id"].startswith("test:")
    assert all(row["request_id"].startswith("validation:")
               for row in record["validation"]["selected"]["requests"])


@pytest.mark.parametrize("fault", ["fixed_side", "schedule", "policy", "duplicate", "test_overlap"])
def test_invalid_fit_or_split_cannot_reach_metrics(monkeypatch, fault):
    base, samples, policy, result = case()
    if fault == "fixed_side":
        result["candidates"][1]["Rv"] = -result["candidates"][1]["Rv"]
    elif fault == "schedule":
        result["decision"]["candidate_steps"] = [0, 1]
    elif fault == "policy":
        result["decision"]["policy"] = {**policy, "steps": 2}
    elif fault == "duplicate":
        result["candidates"].append(copy.deepcopy(result["candidates"][1]))
        result["decision"]["candidate_count"] = 4
    elif fault == "test_overlap":
        samples["test"] = copy.deepcopy(samples["validation"])

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid fitting result/split reached deployment replay")

    monkeypatch.setattr(orchestrator, "_metrics", forbidden)
    with pytest.raises(ValueError):
        recipe3.refine_layer(NAME, base, samples, policy,
            fitter=lambda *_args, **_kwargs: result, progress=lambda **row: None)
