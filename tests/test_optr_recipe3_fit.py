"""Recipe3 parameter freedom and existing-checkpoint recall, CPU evidence only."""
import copy
import weakref

import pytest
import torch

from oscar_ascend import optr_recipe3_fit as recipe3
from oscar_ascend import optr_sequential as sequential
from oscar_ascend.optr_policy import optr_policy
from test_optr_calibration import trace


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def inputs(seed=83, heads=2, **overrides):
    rng = torch.Generator().manual_seed(seed)
    rk = torch.linalg.qr(torch.randn(heads, 8, 8, generator=rng)).Q.contiguous()
    rv = torch.linalg.qr(torch.randn(heads, 8, 8, generator=rng)).Q.contiguous()
    mean = torch.randn(heads, 8, generator=rng) * .2
    train = [[trace(seed + h, "train", length=81)] for h in range(heads)]
    held = [[trace(seed + 20 + h, "validation", length=81)] for h in range(heads)]
    for h in range(heads):
        held[h][0]["wo"] = train[h][0]["wo"]
    policy = {**optr_policy({}), "sink_tokens": 2, "recent_tokens": 4,
              "steps": 3, "eval_every": 2, **overrides}
    return rk, rv, mean, train, held, policy


def test_optional_observer_preserves_original_trajectory_and_owns_cpu_snapshots():
    rk, rv, mean, train, held, policy = inputs()
    kwargs = dict(policy=policy, global_kv_heads=2)
    expected = sequential.fit_head(rk[0], rv[0], mean[0], train[0], held[0], **kwargs)
    events = []

    def observe(**event):
        assert event["rotation"].device.type == "cpu"
        assert not event["rotation"].requires_grad
        events.append({**event, "rotation": event["rotation"].clone()})
        event["rotation"].zero_()  # Callback cannot mutate the optimizer/seed.

    actual = sequential.fit_head(rk[0], rv[0], mean[0], train[0], held[0],
                                 candidate_callback=observe, **kwargs)
    assert all(torch.equal(x, y) for x, y in zip(actual[:2], expected[:2]))
    assert actual[2] == expected[2]
    assert [(row["phase"], row["step"]) for row in events] == [
        (phase, step) for phase in ("k", "v") for step in (0, 1, 3)]
    assert torch.equal(events[0]["rotation"], rk[0])
    assert torch.equal(events[3]["rotation"], rv[0])


def test_perhead_nonzero_mean_and_distinct_rotations_match_original_unconstrained_fit(monkeypatch):
    rk, rv, mean, train, held, policy = inputs()
    originals = [value.clone() for value in (rk, rv, mean)]
    reference = [sequential.fit_head(rk[h], rv[h], mean[h], train[h], held[h],
                 policy=policy, global_kv_heads=2) for h in range(2)]
    previous, calls = [], []
    original_fit = recipe3.fit_head
    original_prepare = sequential.prepare_samples

    def stage(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        previous.extend(weakref.ref(row[key]) for row in result for key in ("q", "k", "v"))
        return result

    def fit(*args, **kwargs):
        assert all(reference() is None for reference in previous)
        calls.append(kwargs["global_kv_heads"])
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(sequential, "prepare_samples", stage)
    monkeypatch.setattr(recipe3, "fit_head", fit)
    actual = recipe3.fit_perhead(rk, rv, mean, train, held, policy=policy)
    assert calls == [2, 2]
    assert all(reference() is None for reference in previous)
    assert not torch.equal(rk[0], rk[1]) and torch.count_nonzero(mean) == mean.numel()
    for h in range(2):
        assert torch.equal(actual["Rk"][h], reference[h][0])
        assert torch.equal(actual["Rv"][h], reference[h][1])
    assert all(torch.equal(a, b) for a, b in zip((rk, rv, mean), originals))
    assert actual["decision"]["staging_resources"]["max_resident_heads"] == 1
    assert actual["decision"]["optimizer_loss_divisor"] == 2
    assert actual["candidates"][0]["id"] == "primary"
    for row in actual["candidates"]:
        assert row["Rk"].device.type == row["Rv"].device.type == "cpu"
        recipe3.validate_perhead_pair(row["Rk"], row["Rv"], mean)
        if row["phase"] == "k":
            assert torch.equal(row["Rv"], rv)
        elif row["phase"] == "v":
            assert torch.equal(row["Rk"], actual["Rk"])


def test_default_and_dense_schedules_keep_existing_steps_with_fixed_budget():
    assert recipe3.checkpoint_steps(optr_policy({})) == [1, 21, 41, 61, 80]
    assert recipe3.checkpoint_steps({"steps": 160, "eval_every": 1}) == [1, 40, 80, 120, 160]
    rk, rv, mean, train, held, policy = inputs(steps=10, eval_every=1)
    actual = recipe3.fit_perhead(rk, rv, mean, train, held, policy=policy)
    assert len(actual["candidates"]) <= 12
    assert actual["decision"]["candidate_steps"] == [0, 1, 3, 5, 7, 10]
    assert len({(row["Rk"].numpy().tobytes(), row["Rv"].numpy().tobytes())
                for row in actual["candidates"]}) == len(actual["candidates"])
    assert all(row["step"] in (None, 0, 1, 3, 5, 7, 10) for row in actual["candidates"])


def recall_case():
    # Fixed synthetic BF16 QKV, independent train/held requests, two different
    # head rotations and nonzero means. No mocked gradients/replay/metrics.
    # The setup frame's saturated zero gate isolates the following decode;
    # the current each-query/frame/terminal and all three projection gates
    # still apply. This proves candidate recall, not model-quality recovery.
    seed, heads, dim, group, length = 12, 2, 8, 2, 81
    rng = torch.Generator().manual_seed(seed)
    rk = torch.linalg.qr(torch.randn(heads, dim, dim, generator=rng)).Q.contiguous()
    rv = torch.linalg.qr(torch.randn(heads, dim, dim, generator=rng)).Q.contiguous()
    mean = torch.randn(heads, dim, generator=rng) * .2
    weights = [torch.randn(group, 12, dim, generator=rng).bfloat16().float() for _ in range(heads)]

    def sample(number, label, weight):
        generator = torch.Generator().manual_seed(number)
        q = torch.randn(2, group, dim, generator=generator).bfloat16()
        k = torch.randn(length, dim, generator=generator).bfloat16()
        v = torch.randn(length, dim, generator=generator).bfloat16()
        gate = torch.sigmoid(torch.randn(2, group, dim, generator=generator) * 2).bfloat16()
        gate[0] = 0
        logits = torch.einsum("mgd,ld->mgl", q.float(), k.float()) * dim ** -.5
        logits[0, :, -1] = -torch.inf
        teacher = torch.einsum("mgl,ld->mgd", logits.softmax(-1), v.float()).bfloat16()
        return {"request_id": label, "q": q, "k": k, "v": v, "gate": gate,
                "teacher": teacher, "wo": weight, "positions": torch.tensor([79, 80]),
                "contexts": torch.tensor([0, 80]), "current_ends": torch.tensor([80, 81]),
                "history_splits": torch.ones(2, dtype=torch.int64), "frame_ids": torch.tensor([0, 1]),
                "token_ids": torch.arange(length), "sample_sha256": f"{number:064x}"}

    splits = {label: [[sample(seed * 30 + i * heads + h, label, weights[h])]
                      for h in range(heads)] for i, label in enumerate(("train", "validation", "test"))}
    policy = {**optr_policy({}), "sink_tokens": 2, "recent_tokens": 4}
    baseline = {"layer_id": 3, "Rk": rk, "Rv": rv, "k_mean": mean}
    return baseline, splits, policy


def test_real_80_step_search_recalls_k_candidate_rejected_by_primary_and_all_v_pairs():
    from oscar_ascend.optr_artifacts import nonworse
    from tools.prepare_optr_rotations import _metrics

    baseline, splits, policy = recall_case()
    rk, rv, mean = (baseline[key] for key in ("Rk", "Rv", "k_mean"))
    fitted = recipe3.fit_perhead(rk, rv, mean, splits["train"], splits["validation"], policy=policy)
    before = _metrics(baseline, splits["validation"], policy)
    results = {}
    for row in fitted["candidates"]:
        after = _metrics({**baseline, "Rk": row["Rk"], "Rv": row["Rv"]}, splits["validation"], policy)
        passed = nonworse(before, after, require_frames=True, require_terminal=True,
                          require_queries=True, require_projection=True)
        passed = passed and after["squared_error"] < before["squared_error"]
        results[row["id"]] = (row["phase"], passed, after["squared_error"])
    assert not any(passed for phase, passed, _ in results.values() if phase != "k")
    assert results["k_step_1"][1]
    assert results["k_step_1"][2] < before["squared_error"] * .6
    assert before["squared_error"] == pytest.approx(.3172262930602623, rel=1e-4)
    assert results["k_step_1"][2] == pytest.approx(.16930189719437183, rel=1e-4)


@pytest.mark.parametrize("fault", ["missing_head", "different_requests", "overlap", "nonfinite"])
def test_invalid_geometry_and_split_identity_fail_before_fitting(monkeypatch, fault):
    rk, rv, mean, train, held, policy = inputs()
    if fault == "missing_head":
        train.pop()
    elif fault == "different_requests":
        train[1][0]["request_id"] = "other"
    elif fault == "overlap":
        held = copy.deepcopy(train)
    elif fault == "nonfinite":
        mean[1, 0] = float("nan")

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid input reached the optimizer")

    monkeypatch.setattr(recipe3, "fit_head", forbidden)
    with pytest.raises(ValueError):
        recipe3.fit_perhead(rk, rv, mean, train, held, policy=policy)
