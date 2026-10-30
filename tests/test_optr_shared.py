"""Archive #170/#173/#175/#176: shared OSCAR constraint, independent upstream losses."""
import copy

import pytest
import torch

from oscar_ascend import optr_shared as shared
from oscar_ascend import optr_sequential as sequential
from oscar_ascend import optr_shared_cache as cache
from test_optr_upstream_parity import inputs, options, source_loss


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("phase", ["k", "v"])
@pytest.mark.parametrize("heads,dim,group,length", [(4, 8, 3, 81), (2, 256, 6, 337)])
def test_streamed_shared_gradient_matches_official_loss_with_tied_rotations(phase, heads, dim, group, length):
    samples, sequence, wo = inputs(153, "train", heads=heads, dim=dim, group=group, length=length)
    p = options()
    mean = torch.zeros(heads, dim)
    base = torch.linalg.qr(torch.randn(dim, dim, generator=torch.Generator().manual_seed(132))).Q
    expected_a = torch.zeros(dim, dim, requires_grad=True)
    reference = base @ torch.matrix_exp(expected_a - expected_a.T)
    expanded = reference.expand(heads, dim, dim)
    expected = source_loss([sequence], expanded, expanded, mean, wo, phase, p).mean()
    # For V, K is fixed; don't let the independent reference differentiate K.
    if phase == "v":
        expected = source_loss([sequence], expanded.detach(), expanded, mean, wo, phase, p).mean()
    expected.backward()
    rows = []
    for h, sample in enumerate(samples):
        row = sequential.prepare_samples([sample], mean[h], p, "cpu")[0]
        if phase == "v":
            row["selected_k_quant_p"] = sequential._key_probabilities(base, row, p).masked_fill(row["exact"], 0).detach()
        rows.append((row, 1. / heads))
    actual_a = torch.zeros(dim, dim, requires_grad=True)
    rotation = base @ torch.matrix_exp(actual_a - actual_a.T)
    shared.backward_shared(rotation, rows, phase, p)
    torch.testing.assert_close(actual_a.grad, expected_a.grad, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("streamed", [False, True])
def test_shared_adam_trajectory_and_final_k_then_v_match_tied_upstream(monkeypatch, streamed):
    heads, dim = 4, 8
    train, source_train, wo = inputs(191, "train", heads=heads, wo_scale=1e-3)
    held, source_held, _ = inputs(197, "held", heads=heads, wo_scale=1e-3)
    for h in range(heads):
        held[h]["wo"] = train[h]["wo"]
    mean = torch.zeros(heads, dim)
    base = torch.eye(dim)
    p = options(steps=3, eval_every=20)
    if streamed:
        monkeypatch.setattr(cache, "RESIDENT_CACHE_BYTES", 0)
        monkeypatch.setattr(cache, "STREAMED_ROW_CACHE_BYTES", max(
            cache.sample_cache_bytes(sample, queries=1) for sample in (*train, *held)))
    snapshots = []
    original = torch.optim.Adam

    class RecordingAdam(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.saved = []
            snapshots.append(self.saved)

        def step(self, *args, **kwargs):
            result = super().step(*args, **kwargs)
            self.saved.append(self.param_groups[0]["params"][0].detach().clone())
            return result

    monkeypatch.setattr(torch.optim, "Adam", RecordingAdam)
    selected = {}
    for phase in ("k", "v"):
        def loss(rotation, sequence):
            expanded = rotation.expand(heads, dim, dim)
            rk = expanded if phase == "k" else selected["k"].expand_as(expanded)
            return source_loss([sequence], rk, expanded, mean, wo, phase, p).mean()
        best = float(loss(base, source_held))
        best_r = base.clone()
        correction = torch.zeros(dim, dim, requires_grad=True)
        optimizer = torch.optim.Adam([correction], lr=p["lr"])
        for step in range(p["steps"]):
            optimizer.zero_grad(set_to_none=True)
            rotation = base @ torch.matrix_exp(correction - correction.T)
            loss(rotation, source_train).backward()
            optimizer.step()
            if step % p["eval_every"] == 0 or step == p["steps"] - 1:
                with torch.no_grad():
                    candidate = base @ torch.matrix_exp(correction - correction.T)
                    score = float(loss(candidate, source_held))
                    if score < best:
                        best, best_r = score, candidate.clone()
        selected[phase] = best_r
    events = []
    rk, rv, decision = shared.fit_shared(base.repeat(heads, 1, 1), base.repeat(heads, 1, 1), mean,
        [[row] for row in train], [[row] for row in held], policy=p, progress=events.append)
    assert len(snapshots) == 4  # Only one optimizer per phase, not per KV head.
    for phase in range(2):
        for expected, actual in zip(snapshots[phase], snapshots[phase + 2], strict=True):
            torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-6)
    for phase, actual in (("k", rk), ("v", rv)):
        assert torch.equal(actual, actual[:1].expand_as(actual))
        torch.testing.assert_close(actual[0], selected[phase], rtol=5e-5, atol=5e-6)
    assert decision["fitted_grouping"] == "layer"
    assert decision["staging_resources"]["mode"] == ("streamed" if streamed else "resident")
    assert len([e for e in events if e["event"] == "step_done"]) == 2 * p["steps"]


def test_complete_chunked_shared_search_preserves_selection_and_releases_rows(monkeypatch):
    import weakref
    from test_optr_shared_cache import capture, force_stream

    train, held, mean, p = capture()
    p = {**p, "steps": 3, "eval_every": 2}
    base = torch.eye(mean.shape[-1]).repeat(len(train), 1, 1)
    expected = shared.fit_shared(base, base, mean, train, held, policy=p)
    force_stream(monkeypatch, train, held)
    previous, original = [], cache.prepare_samples

    def stage(*args, **kwargs):
        assert all(reference() is None for reference in previous)
        rows = original(*args, **kwargs)
        previous[:] = [weakref.ref(rows[0][key]) for key in ("q", "k", "v", "teacher_p")]
        return rows

    monkeypatch.setattr(cache, "prepare_samples", stage)
    actual = shared.fit_shared(base, base, mean, train, held, policy=p)
    assert all(reference() is None for reference in previous)
    for result, reference in zip(actual[:2], expected[:2], strict=True):
        torch.testing.assert_close(result, reference, rtol=5e-5, atol=5e-6)
    for result, reference in zip(actual[2]["phases"], expected[2]["phases"], strict=True):
        assert result["accepted"] == reference["accepted"]
        for name in ("baseline_proxy_loss", "selected_proxy_loss"):
            assert result[name] == pytest.approx(reference[name], rel=3e-5, abs=2e-8)
        assert [row["accepted"] for row in result["candidates"]] == [
            row["accepted"] for row in reference["candidates"]]


def test_v_steps_reuse_selected_shared_k_for_all_heads(monkeypatch):
    heads, dim = 2, 8
    train, _, _ = inputs(83, "train", heads=heads)
    held, _, _ = inputs(87, "held", heads=heads)
    events, probabilities = [], []
    original = cache._key_probabilities

    def fixed(rotation, row, policy):
        assert events[-1]["event"] == "phase_start" and events[-1]["phase"] == "v"
        probabilities.append(rotation.clone())
        return original(rotation, row, policy)

    monkeypatch.setattr(cache, "_key_probabilities", fixed)
    base = torch.eye(dim).repeat(heads, 1, 1)
    rk, _, _ = shared.fit_shared(base, base, torch.zeros(heads, dim), [[r] for r in train],
        [[r] for r in held], policy=options(steps=2), progress=events.append)
    assert len(probabilities) == 2 * heads
    assert all(torch.equal(rotation, rk[0]) for rotation in probabilities)


@pytest.mark.parametrize("fault", ["nonshared", "mean", "missing_head", "different_requests", "overlap", "nonfinite"])
def test_shared_fit_rejects_invalid_basis_or_capture(fault):
    train, _, _ = inputs(83, "train")
    held, _, _ = inputs(87, "held")
    base = torch.eye(8).repeat(2, 1, 1)
    mean = torch.zeros(2, 8)
    training, validation = [[r] for r in train], [[r] for r in held]
    if fault == "nonshared": base[1].neg_()
    if fault == "mean": mean[1, 0] = 1
    if fault == "missing_head": training.pop()
    if fault == "different_requests": training[1][0]["request_id"] = "other"
    if fault == "overlap": validation = copy.deepcopy(training)
    if fault == "nonfinite": base[1, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        shared.fit_shared(base, base, mean, training, validation, policy=options(steps=1))
