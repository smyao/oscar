"""Archive #13-20/#148: history bias matters beside exact-window keys.

These independent CPU checks constrain a local calibration selector; they do
not measure model quality or redefine the frozen attention oracle.
"""
import math

import pytest
import torch

from oscar_ascend import clip_calibration


def test_centered_history_error_is_not_a_global_softmax_invariance():
    # Q=1, original history/exact keys both zero, and V=[+1,-1].  Centering
    # the history error alone erases its mean, although the exact key has no
    # quantization error and therefore does not receive the same logit shift.
    history_error = torch.tensor([[1.0]], dtype=torch.float64)
    centered = history_error - history_error.mean(0)
    assert centered.square().mean().item() == 0.0
    assert history_error.square().mean().item() == 1.0
    values = torch.tensor([1.0, -1.0], dtype=torch.float64)
    original = torch.softmax(torch.zeros(2, dtype=torch.float64), 0) @ values
    changed = torch.softmax(torch.tensor([1.0, 0.0], dtype=torch.float64), 0) @ values
    assert original.item() == 0.0
    assert changed.item() == pytest.approx(math.tanh(0.5), abs=1e-15)
    assert changed.item() > 0.46


@pytest.mark.parametrize("regressed_metric", ["prefill_raw", "decode_raw"])
def test_selector_rejects_history_bias_despite_better_centered_scores(monkeypatch, regressed_metric):
    policy = {
        "policy": clip_calibration.POLICY,
        "criterion": clip_calibration.CRITERION,
        "baseline_ratio": .96,
        "candidate_ratios": list(clip_calibration.CANDIDATE_RATIOS),
        "max_samples": clip_calibration.MAX_SAMPLES,
        "dequant_modes": [0, 1],
    }
    baseline_scores = {name: 1.0 for name in clip_calibration.METRICS}
    biased_scores = {name: .5 for name in clip_calibration.METRICS}
    biased_scores[regressed_metric] = 1.25

    def scores(_keys, _covariance, ratio):
        if ratio == .96:
            return dict(baseline_scores)
        if ratio == .90:
            return dict(biased_scores)
        return {name: 2.0 for name in clip_calibration.METRICS}

    monkeypatch.setattr(clip_calibration, "candidate_scores", scores)
    selected, audit = clip_calibration.select_head_clip(
        torch.zeros(8, 64, dtype=torch.bfloat16),
        torch.eye(64, dtype=torch.float64),
        torch.eye(64, dtype=torch.float32),
        torch.zeros(64, dtype=torch.float32),
        policy,
    )
    # Either centered-only mode would have preferred .90.  The extra raw
    # gate must reject its source-relative bias in either dequant mode.
    assert biased_scores["prefill_centered"] < baseline_scores["prefill_centered"]
    assert biased_scores["decode_centered"] < baseline_scores["decode_centered"]
    assert selected == .96
    assert audit["selected_scores"] == baseline_scores
    rejected = next(row for row in audit["candidates"] if row["ratio"] == .90)
    assert rejected["eligible"] is False
    assert rejected["scores"][regressed_metric] > baseline_scores[regressed_metric]
