"""Regression checks for sampled log accounting; archive #137/#152/#155."""

import json
from pathlib import Path

import pytest

from tools.analyze_service_log import analyze
from tools.whole_run_model import (
    counterfactual, model, previous_projection_backtest, required_fraction,
)


REPORTS = Path(__file__).resolve().parents[1] / "reports"


def test_real_log_mtp_accounting_and_unequal_windows():
    candidate = analyze(REPORTS / "target_striped_probe_and_longbench_20260930.txt")
    native = analyze(REPORTS / "target_native_longbench_20260930.txt")
    assert (candidate["groups"]["all"]["samples"], native["groups"]["all"]["samples"]) == (90, 69)
    assert (candidate["spans"]["sample_timestamp_span_seconds"], native["spans"]["sample_timestamp_span_seconds"]) == (891, 680)
    assert candidate["groups"]["all"]["drafted_tokens_spec_sum"] == 99015
    assert native["groups"]["all"]["drafted_tokens_spec_sum"] == 74100
    assert "first EngineCoreOutput" in candidate["prompt_accounting"]
    assert abs(candidate["groups"]["all"]["generation_rate_minus_mtp_count"]) <= 25
    assert abs(native["groups"]["all"]["generation_rate_minus_mtp_count"]) <= 45
    assert candidate["groups"]["after_last_prompt_until_last_active"]["samples"] == 5
    assert native["groups"]["after_last_prompt_until_last_active"]["samples"] == 12


def test_two_anchors_do_not_identify_speedup_share():
    candidate = json.loads((REPORTS / "service_trace_candidate_20260930.json").read_text())
    native = json.loads((REPORTS / "service_trace_native_20260930.json").read_text())
    result = model(candidate, native)
    assert result["prediction_status"] == "not_supported"
    assert "prefill" in result["prompt_timing_caveat"]
    assert "if_prefill_only_fraction_of_prompt_visible_time_required" not in result
    assert result["conditional_additive_simplex"]["candidate_specific_residual_seconds_range"][0] < 360
    assert required_fraction(840, 480, 2) == pytest.approx(6 / 7)
    assert required_fraction(840, 480, 1.5) > 1
    assert counterfactual(840, 0, 2) == 840
    assert counterfactual(840, 1, 2) == 420
    with pytest.raises(ValueError):
        counterfactual(840, 1.1, 2)


def test_previous_simulator_projection_is_falsified_not_reused_as_a_profile():
    result = previous_projection_backtest(960, 840, 2.0954974660993666)
    assert result["observed_whole_run_speedup"] == pytest.approx(8 / 7)
    assert result["rejected_naive_whole_run_prediction_seconds"] < 480
    assert result["naive_prediction_error_seconds"] > 380
    assert result["conditional_effective_affected_share"] == pytest.approx(0.2391034139)
    assert result["conditional_share_is_measured"] is False
    assert result["transfer_share_to_next_candidate"] is False
