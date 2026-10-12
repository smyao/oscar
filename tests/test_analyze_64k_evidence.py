"""档案 #137/#140/#152/#155: preserve distinct denominators and missing evidence.

Golden values below come from the supplied 2026-10-09 attachment, not from
production code or a simulated performance result.
"""

from pathlib import Path

import pytest

from tools.analyze_64k_evidence import parse_evidence


@pytest.fixture(scope="module")
def evidence():
    source = Path(__file__).resolve().parents[1] / "reports/target_64k_120_20261009.txt"
    return parse_evidence(source.read_bytes(), source.as_posix())


def test_all_supplied_engine_rows_are_parsed_without_pid_or_clock_mixing(evidence):
    a, b = (evidence["runs"][v] for v in ("oscar", "native"))
    assert (a["pid"], b["pid"]) == (93238, 8961)
    assert (len(a["engine"]), len(b["engine"])) == (249, 137)
    assert a["engine"][0][1] == "02:06:20"
    assert a["engine"][-1][1] == "02:47:40"
    assert b["engine"][0][1] == "03:04:52"
    assert b["engine"][-1][1] == "03:27:32"
    assert a["server_summary"]["server_clock"]["interval_seconds_histogram"] == {10: 248}
    assert b["server_summary"]["server_clock"]["interval_seconds_histogram"] == {10: 136}


def test_truncated_records_preserve_known_fields_and_do_not_invent_counts(evidence):
    a, b = (evidence["runs"][v] for v in ("oscar", "native"))
    partial = next(r for r in a["spec"] if r[1] == "02:44:10")
    assert partial == [565, "02:44:10", 2.88, 25.0, 39.9, None, None, None, None]
    assert b["fragments"][0]["line"] == 917
    assert b["fragments"][0]["preceding_engine_time"] == "03:15:32"
    assert not any(r[1] == "03:15:32" for r in b["spec"])
    assert (len(a["spec"]), len(b["spec"])) == (228, 135)


def test_client_tables_and_warmup_have_separate_time_denominators(evidence):
    a, b = (evidence["runs"][v]["client"] for v in ("oscar", "native"))
    for run in (a, b):
        assert len(run["performance"]) == 7
        assert len(run["common"]) == 11
        assert run["common"]["Total Input Tokens"]["value"] == 7812000
        assert run["common"]["Total Generated Tokens"]["value"] == 61440
        assert run["common"]["Success Requests"]["value"] == 120
        assert run["common"]["Failed Requests"]["value"] == 0
        assert run["concurrency"] == 32
        assert run["warmup_counts"] == {"total": 1, "success": 1, "failed": 0}
    assert a["benchmark_duration_s"] == pytest.approx(2467.1275737)
    assert b["benchmark_duration_s"] == pytest.approx(1342.4437259)
    assert (a["warmup_elapsed_s"], b["warmup_elapsed_s"]) == (28.077, 15.01)
    assert (a["api_task_elapsed_s"], b["api_task_elapsed_s"]) == (2503.8, 1374.95)
    assert a["benchmark_duration_s"] != a["dispatch_to_api_end_s"]


def test_client_latency_decomposition_respects_fixed_512_output_tokens(evidence):
    a, b = (evidence["runs"][v]["client"] for v in ("oscar", "native"))
    assert a["mean_after_first_token_s"] == pytest.approx(440.3929)
    assert b["mean_after_first_token_s"] == pytest.approx(105.5514)
    assert a["mean_tpot_from_mean_latencies_ms"] == pytest.approx(861.8, abs=.05)
    assert b["mean_tpot_from_mean_latencies_ms"] == pytest.approx(206.6, abs=.05)
    assert a["performance"]["ITL"]["p99"] == 13445.1
    assert b["performance"]["ITL"]["p99"] == 2873.1
    assert evidence["comparison"]["same_workload"]


def test_spec_totals_use_counts_rather_than_unweighted_percentage_means(evidence):
    a, b = (evidence["runs"][v]["server_summary"]["speculative_complete_logged_windows"]
            for v in ("oscar", "native"))
    assert (a["accepted_tokens"], a["drafted_tokens"], a["drafts_if_three_tokens_each"]) == (40428, 63381, 21127)
    assert (b["accepted_tokens"], b["drafted_tokens"], b["drafts_if_three_tokens_each"]) == (45505, 48948, 16316)
    assert a["accepted_over_drafted"] == pytest.approx(40428 / 63381)
    assert b["accepted_over_drafted"] == pytest.approx(45505 / 48948)
    assert a["fixed_three_draft_inference_matches_every_rounded_mean"]
    assert b["fixed_three_draft_inference_matches_every_rounded_mean"]


def test_observed_capacity_and_zero_counters_do_not_invent_memory_or_phases(evidence):
    a, b = (evidence["runs"][v]["server_summary"] for v in ("oscar", "native"))
    assert (a["peak_running"], b["peak_running"]) == (27, 12)
    assert (a["peak_kv_usage_pct"], b["peak_kv_usage_pct"]) == (99.7, 98.2)
    assert a["both_reported_throughputs_zero_while_running"]["sample_count"] == 19
    assert b["both_reported_throughputs_zero_while_running"]["sample_count"] == 0
    for run in (a, b):
        assert run["absolute_kv_bytes"] is None
        assert run["preemptions"]["count"] is None
        assert "pure_decode" not in run["sample_groups_not_device_phases"]


def test_server_counter_integration_does_not_use_client_duration(evidence):
    a, b = (evidence["runs"][v]["server_summary"] for v in ("oscar", "native"))
    assert a["approx_counter_integrals_after_first_engine_sample"]["generation_tokens"] == pytest.approx(61503)
    assert b["approx_counter_integrals_after_first_engine_sample"]["generation_tokens"] == pytest.approx(61740)
    assert a["queue_drain"]["sampled_waiting_zero_to_running_zero_s"] == 60
    assert b["queue_drain"]["sampled_waiting_zero_to_running_zero_s"] == 20


def test_conditional_warmup_pool_bounds_include_the_reference_null_block(evidence):
    inference = evidence["conditional_single_request_warmup_pool_inference"]
    assert inference["status"] == "conditional_inference_not_measured_pool_size"
    a, b = inference["oscar"], inference["native"]
    assert (a["engine_line"], b["engine_line"]) == (12, 709)
    assert a["cases"][0]["usable_pool_blocks_range"] == [960, 986]
    assert a["union_configured_num_gpu_blocks_range"] == [961, 987]
    assert b["cases"][0]["configured_num_gpu_blocks_range"] == [1149, 1162]
    assert b["cases"][1]["configured_num_gpu_blocks_range"] == [1161, 1174]
    assert b["union_configured_num_gpu_blocks_range"] == [1149, 1174]
    for run in (a, b):
        for case in run["cases"]:
            lo, hi = case["configured_num_gpu_blocks_range"]
            used, pct = case["assumed_request_used_blocks_excluding_null"], case["displayed_usage_pct"]
            for n in (lo, hi):
                assert "%.1f" % (100 * (1 - (n - 1 - used) / (n - 1))) == format(pct, ".1f")
            for n in (lo - 1, hi + 1):
                assert "%.1f" % (100 * (1 - (n - 1 - used) / (n - 1))) != format(pct, ".1f")
    # Reference subtraction is material at the 3.75% mathematical tie.
    assert "%.1f" % (100 * (1 - 924 / 960)) == "3.7"
    assert "%.1f" % (100 * 36 / 960) == "3.8"
