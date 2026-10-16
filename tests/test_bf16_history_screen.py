"""Archive #126/#145/#148/#154: reject numerical regressions without changing gates.

These CPU tests validate the experiment and its failure reporting. They do not
qualify a CANN implementation, physical cache protocol, graph or service.
"""
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from oscar_ascend.ops.reference import attention
from tools.experiments import bf16_history_screen as screen

ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE = json.loads((ROOT / "configs/acceptance.json").read_text())


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_exact_cancellation_rejects_full_candidate_not_cpu_reduction_noise():
    # At coordinate 0, each opposite V pair's FP32 decoded levels sum to
    # -1/16 while BF16 levels sum to zero. All scores are zero, P=1 exactly.
    # There are 704 history rows of 1024 visible rows: -1/32*704/1024.
    # Other coordinates have separate codec rounding and larger errors.
    row = screen.evaluate(screen.CASES[0], ACCEPTANCE["fused_attention"])
    assert row["baseline_passed"]
    assert row["codec_oracle_identity"] and row["input_bytes_unchanged"]
    assert row["status"] == "cpu_candidate_failed"
    stats = row["stages"]
    for stage in ("fp32", "q_only", "p_only"):
        assert stats[stage]["output"]["passed"]
    for stage in ("kv_only", "full"):
        error = stats[stage]["output"]
        assert error["failed_elements"] == 6 * 256
        assert error["first_failure"]["actual"] == 0.0
        assert error["first_failure"]["expected"] == pytest.approx(-0.021484375, abs=1e-7)
        assert error["max_abs"] > error["first_failure"]["allowed"]
        assert stats[stage]["partial_output"]["first_failure"]["expected"] == -0.03125
        assert stats[stage]["lse"]["passed"]  # Correct LSE must not hide bad output.
        assert stats[stage]["unchanged_source1_source2_bits"]


def test_bf16_cast_counterexample_is_not_misreported_as_attention_failure():
    row = screen.codec_cast_counterexample(ACCEPTANCE["store_dequant"])
    assert row["scale"] == 0.66650390625
    assert row["fp32_levels"][1] == 16.66650390625
    assert row["bf16_levels"][1] == 16.625
    assert row["cast_vs_store_tolerance"]["failed_elements"] == 128


def test_gate_handles_empty_lse_zero_reference_and_invalid_nonfinite():
    tol = ACCEPTANCE["fused_attention"]
    empty = torch.tensor([-torch.inf])
    assert screen.compare(empty, empty, tol, allow_negative_infinity=True)["passed"]
    assert not screen.compare(empty, empty, tol)["passed"]
    for bad in (torch.nan, torch.inf, -torch.inf):
        assert not screen.compare(torch.tensor([bad]), torch.zeros(1), tol,
                                  allow_negative_infinity=True)["passed"]
    row = screen.compare(torch.tensor([0.006]), torch.zeros(1), tol)
    assert row["failed_elements"] == 1
    assert row["zero_reference_nonzero_error"] == 1
    assert row["max_rel_nonzero_reference"] is None
    json.dumps(row, allow_nan=False)


def test_online_empty_first_tile_and_empty_row_preserve_fp32_softmax_state():
    generator = torch.Generator().manual_seed(1641)
    q = torch.randn((2, 6, 256), generator=generator).bfloat16().float()
    k = torch.randn((513, 1, 256), generator=generator).bfloat16().float()
    v = torch.randn(k.shape, generator=generator).bfloat16().float()
    mask = torch.zeros((2, 513), dtype=torch.bool)
    mask[0, 256:] = True
    result = screen.online(q, k, v, mask, "fp32")
    expected = attention(q[:1], k[256:], v[256:], causal=False)
    torch.testing.assert_close(result.output[:1], expected.output, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(result.lse[:1], expected.lse, atol=1e-6, rtol=1e-6)
    assert torch.equal(result.output[1], torch.zeros_like(result.output[1]))
    assert bool(torch.isneginf(result.lse[1]).all())


@pytest.mark.parametrize("context,qlen,recent", [(0, 4, 256), (320, 4, 256), (641, 168, 32)])
def test_task_sources_partition_every_query_without_gaps_or_overlap(context, qlen, recent):
    case = screen.Case("partition", qlen, context, splits=3, recent=recent)
    for begin in range(0, qlen, 21):
        end = min(qlen, begin + 21)
        qpos = torch.arange(context + begin, context + end)
        coverage = torch.zeros((end - begin, context + qlen), dtype=torch.int32)
        for source, split, positions in screen.task_ranges(case, begin, end - begin):
            mask = screen.visible(case, qpos, positions, source)
            coverage[:, positions] += mask.int()
        expected = torch.arange(context + qlen)[None, :] <= qpos[:, None]
        assert torch.equal(coverage, expected.int())


def test_no_history_control_and_report_fail_closed(tmp_path, monkeypatch):
    control = screen.evaluate(screen.CASES[1], ACCEPTANCE["fused_attention"])
    assert control["baseline_passed"] and control["candidate_passed"]
    assert all(row["unchanged_source1_source2_bits"] for row in control["stages"].values())
    run = screen.run
    monkeypatch.setattr(screen, "run", lambda **kwargs: run(cases=(screen.CASES[0],), **kwargs))
    output = tmp_path / "failed.json"
    assert screen.main(["--output", str(output)]) == 2
    report = json.loads(output.read_text())
    assert report["status"] == "cpu_screen_failed"
    assert report["returncode"] == 2 and report["production_qualified"] is False
    assert report["next_stage"] == "stop_production_integration"
    assert report["frozen_attention_gate"] == ACCEPTANCE["fused_attention"]
    assert "NPU_completion" in report["not_run"] and "graph_replay" in report["not_run"]
    json.dumps(report, allow_nan=False)


def test_passing_subset_cannot_unlock_next_stage_even_with_a_required_case_name():
    empty = screen.CASES[1]
    for cases in ((empty,), (replace(empty, name=screen.CASES[0].name),)):
        report = screen.run(cases)
        assert report["cases"][0]["candidate_passed"]
        assert report["status"] == "cpu_screen_incomplete"
        assert report["required_cases_complete"] is False
        assert report["returncode"] == 2
        assert report["next_stage"] == "incomplete_screen"


def test_relative_l2_uses_strict_frozen_threshold_and_checks_zero_norm():
    limit = ACCEPTANCE["prefill_window"]["relative_l2_lt"]
    assert screen.relative_l2(torch.zeros(1), torch.zeros(1), limit)["passed"]
    assert not screen.relative_l2(torch.ones(1), torch.zeros(1), limit)["passed"]
    # Exactly equal to a supplied limit must fail the strict inequality.
    assert not screen.relative_l2(torch.tensor([2.]), torch.tensor([1.]), 1.)["passed"]
