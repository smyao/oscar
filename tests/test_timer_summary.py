"""Archive #70-73: distinguish launch CPU, asynchronous device and client wall.

Synthetic profiler formats test accounting only, not NPU acceptance/performance.
"""
import csv
import json
from pathlib import Path

import pytest

from tools.summarize_timer import (
    _bound_evidence,
    _exclusive_scopes,
    classify_kernel,
    parse_csv,
    parse_trace,
    summarize,
    summary_lines,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_phase_classification_current_and_unknown():
    expected = {
        "oscar_store_aligned_int2_kernel": "oscar.store_quant",
        "oscar_copy_validate_bf16_query_kernel": "oscar.query_validate_copy",
        "oscar_attention_cv_aligned_cluster16_kernel": "oscar.history_window_attention",
        "oscar_merge_lse_kernel": "oscar.lse_merge",
        "chunk_gated_delta_rule_fwd": "native.GDN",
        "CausalConv1d": "native.GDN",
        "FusedInferAttentionScore": "native.FULL_attention",
        "QuantBatchMatmulV3": "native.GEMM_unattributed",
        "HcclAllReduce": "communication.HCCL",
        "NpuMemcpy": "copy_layout",
        "UnknownVendorKernel": "unknown",
    }
    assert {name: classify_kernel(name) for name in expected} == expected


def test_csv_never_confuses_theoretical_core_time_with_actual(tmp_path):
    path = tmp_path / "op_summary_1.csv"
    write_csv(path, [{"Op Name": "matmul", "aicore_time(us)": "9000", "aic_mac_ratio": "0.8"}])
    missing = []
    assert parse_csv(path, missing) == []
    assert any("actual_duration_or_unit_missing" in m for m in missing)


@pytest.mark.parametrize("duration_key,value,expected", [("Task Duration(us)", "1,200", 1.2),
                                                           ("Duration(ns)", "1200000", 1.2),
                                                           ("Duration(ms)", "1.2", 1.2)])
def test_csv_duration_units_counter_names_survive(tmp_path, duration_key, value, expected):
    path = tmp_path / "op_summary.csv"
    write_csv(path, [{"Op Name": "oscar_store_aligned_int2", duration_key: value,
                     "aic_main_mem_read_bw(GB/s)": "80.5", "aic_mac_ratio": "N/A",
                     "aiv_vec_ratio": "0.5", "aiv_mte2_ratio": "0.9"}])
    rows = parse_csv(path)
    assert rows[0]["duration_ms"] == pytest.approx(expected)
    assert rows[0]["counters"]["aic_main_mem_read_bw(GB/s)"] == {"value": 80.5, "unit": "GB/s", "raw": "80.5"}
    assert "aic_mac_ratio" not in rows[0]["counters"]
    bound = _bound_evidence(rows)
    assert bound["classification"] == "inconclusive"
    assert bound["mte2_over_max_cube_vector"] is None  # unavailable AIV cube is not invented as zero.
    assert bound["per_core_pipe_ratios"][0]["formula"] == "mte2/vector"
    assert bound["per_core_pipe_ratios"][0]["duration_weighted_mean"] == pytest.approx(1.8)


def test_memory_bandwidth_does_not_imply_hbm_bound(tmp_path):
    path = tmp_path / "kernel_details.csv"
    write_csv(path, [{"Name": "oscar_attention_cv_aligned", "Duration(us)": "300", "HBM Read Bandwidth(GB/s)": "2000"}])
    bound = _bound_evidence(parse_csv(path))
    assert bound["classification"] == "inconclusive"
    assert bound["mte2_over_max_cube_vector"] is None
    assert bound["memory_counters"]
    assert bound["evidence_missing"]


def test_mixed_core_units_not_combined(tmp_path):
    path = tmp_path / "kernel_details.csv"
    write_csv(path, [{"Name": "matmul", "Duration(us)": "300", "aic_mac_ratio": "0.5", "aiv_mte2_ratio": "0.9"}])
    assert _bound_evidence(parse_csv(path))["mte2_over_max_cube_vector"] is None


def test_nested_host_scopes_partition_instead_of_sum():
    scopes = [{"phase": "runner", "start_ns": 0, "end_ns": 10000000},
              {"phase": "full", "start_ns": 1000000, "end_ns": 9000000},
              {"phase": "rotate", "start_ns": 2000000, "end_ns": 4000000},
              {"phase": "store", "start_ns": 5000000, "end_ns": 8000000}]
    exclusive = _exclusive_scopes(scopes)
    assert exclusive == {"runner": 2.0, "full": 3.0, "rotate": 2.0, "store": 3.0}
    assert sum(exclusive.values()) == 10


def test_trace_does_not_count_cpu_and_uses_explicit_device_correlation(tmp_path):
    path = tmp_path / "trace_view.json"
    write_json(path, {"displayTimeUnit": "ns", "traceEvents": [
        {"ph": "M", "name": "process_name", "pid": 2, "args": {"name": "Ascend Hardware"}},
        {"ph": "X", "name": "oscar_timer::rotation_q", "pid": 1, "tid": 1, "ts": 0, "dur": 20},
        {"ph": "X", "name": "aten::matmul", "pid": 1, "tid": 1, "ts": 2, "dur": 3,
         "args": {"External id": 99}},
        {"ph": "X", "name": "matmul", "cat": "kernel", "pid": 2, "tid": 1, "ts": 10, "dur": 1000,
         "args": {"External id": 99}},
        {"ph": "X", "name": "FusedInferAttentionScore", "cat": "cpu_op", "pid": 1, "tid": 1,
         "ts": 30, "dur": 5000},
    ]})
    rows = parse_trace(path)
    assert len(rows) == 1
    assert rows[0]["duration_ms"] == 1  # Chrome displayTimeUnit does not change ts/dur.
    assert rows[0]["phase"] == "rotation_q"
    assert rows[0]["attribution"] == "trace_correlation"
    assert parse_trace(path, time_unit="ns")[0]["duration_ms"] == .001


def batch(run_id, variant, concurrency, mode, wall=10):
    return {"run_id": run_id, "variant": variant, "concurrency": concurrency, "mode": mode,
            "status": "completed", "pair_hash": "pair", "prompt_manifest_sha256": f"prompts{concurrency}",
            "sample": {"batch_elapsed_seconds": wall, "completed_requests": concurrency,
                       "failed_requests": 0, "timeouts": 0,
                       "requests": [{"usage": {"completion_tokens": 256}} for _ in range(concurrency)]},
            "mtp_counter_delta": {"accepted_tokens": 3, "draft_tokens": 6}}


def add_window(root, run_id, rank, metric="PipeUtilization", *, op_duration=1000, counter_start=1, bucket="decode"):
    window_id = f"{run_id}-r{rank}-{metric}-{bucket}"
    profile_dir = root / "profile" / window_id
    manifest = {"run_id": run_id, "rank": rank, "window_id": window_id, "bucket": bucket,
                "metric": metric, "status": "complete", "steps": [7], "profile_dir": str(profile_dir)}
    write_json(root / "windows" / run_id / f"rank{rank}" / f"{window_id}.json", manifest)
    write_csv(profile_dir / "ASCEND_PROFILER_OUTPUT" / "kernel_details.csv", [
        {"Name": "oscar_attention_cv_aligned", "Duration(us)": op_duration, "Start Time(us)": 1},
        {"Name": "FusedInferAttentionScore", "Duration(us)": 500, "Start Time(us)": 1},
    ])
    counter = {"aic_mac_ratio": .4, "aic_vec_ratio": .2, "aic_mte2_ratio": .8} if metric == "PipeUtilization" else {"aic_main_mem_read_bw(GB/s)": 80.5}
    write_csv(profile_dir / "PROF_0" / "op_summary_1.csv", [
        {"Op Name": "oscar_attention_cv_aligned", "Task Duration(us)": op_duration,
         "Task Start Time(us)": counter_start, **counter},
    ])
    step = {"run_id": run_id, "rank": rank, "step_id": 7, "bucket": bucket, "profiled": True,
            "host_ms": 5, "scheduled_requests": 4, "scheduled_decode_tokens": 16,
            "scheduled_prompt_tokens": 0, "scheduled_draft_tokens": 12,
            "context_min": 32770, "context_max": 32770,
            "scopes": [{"phase": "execute", "start_ns": 0, "end_ns": 5000000}]}
    path = root / f"timer-steps-rank{rank}-pid1.jsonl"
    with path.open("a") as handle:
        handle.write(json.dumps(step) + "\n")


def test_three_arms_control_vs_profile_and_no_rank_sum(tmp_path):
    batches = [batch("n4", "native", 4, "control", 10), batch("o4", "candidate", 4, "control", 12),
               batch("o8", "candidate", 8, "control", 18), batch("o4p", "candidate", 4, "profile", 100)]
    write_json(tmp_path / "run.json", {"run_id": "r", "status": "complete", "pair_hash": "pair",
                                       "services": [{"variant": "candidate", "batches": batches}]})
    add_window(tmp_path, "o4p", 0)
    add_window(tmp_path, "o4p", 1, op_duration=2000)
    report = summarize(tmp_path)
    assert len(report["comparisons"]) == 2
    same, capacity = report["comparisons"]
    assert same["output_throughput_ratio"] == pytest.approx(10 / 12)
    assert same["wall_ratio"] == 1.2
    assert capacity["output_throughput_ratio"] == pytest.approx(20 / 18)
    assert capacity["wall_ratio"] is None  # 4 and 8 requests have different work amounts.
    run = report["runs"][-1]
    assert not run["normal"]["eligible_for_throughput_comparison"]
    assert [w["task_sum_ms"] for w in run["device_windows"]] == [1.5, 2.5]
    assert [w["device_busy_union_ms"] for w in run["device_windows"]] == pytest.approx([1, 2])
    assert run["device_windows"][0]["bound"]["mte2_over_max_cube_vector"] == 2
    assert run["device_windows"][0]["step_alignment"] == "matched"
    lines = summary_lines(report)
    assert len(lines) <= 200
    assert any("RANK_SKEW" in line and "2" in line for line in lines)
    assert not any("native15" in line for line in lines)
    assert lines[-1] == "=== OSCAR_TIMER_SUMMARY_END ==="


def test_unaligned_pmu_is_retained_but_not_attached(tmp_path):
    write_json(tmp_path / "run.json", {"services": [{"batches": [batch("op", "candidate", 4, "profile")]}]})
    add_window(tmp_path, "op", 0, counter_start=3000)
    report = summarize(tmp_path)
    window = report["runs"][0]["device_windows"][0]
    assert not window["counters"]
    assert window["unaligned_counter_exports"]["rows"] == 1
    assert "aic_mac_ratio" in window["unaligned_counter_exports"]["counters"]
    assert window["bound"]["mte2_over_max_cube_vector"] is None
    assert any("PMU_UNALIGNED" in x for x in summary_lines(report))


def test_pair_hash_mismatch_rejects_ratio(tmp_path):
    native, other = batch("n", "native", 4, "control"), batch("o", "candidate", 4, "control")
    other["pair_hash"] = "another-model"
    write_json(tmp_path / "run.json", {"runs": [native, other]})
    compare = summarize(tmp_path)["comparisons"][0]
    assert compare["output_throughput_ratio"] is None
    assert compare["wall_ratio"] is None
    other["pair_hash"] = native["pair_hash"]
    other["prompt_manifest_sha256"] = "wrong-four-prompts"
    write_json(tmp_path / "run.json", {"runs": [native, other]})
    compare = summarize(tmp_path)["comparisons"][0]
    assert compare["output_throughput_ratio"] is None


def test_failed_and_profiled_runs_never_benchmark_baseline(tmp_path):
    native = batch("n", "native", 4, "profile")
    other = batch("o", "candidate", 4, "control")
    other["sample"]["failed_requests"] = 1
    write_json(tmp_path / "run.json", {"runs": [native, other]})
    report = summarize(tmp_path)
    assert not report["comparisons"]
    assert "completed_unprofiled_native_control_missing" in report["missing"]


def test_partial_and_unsupported_artifacts_are_visible(tmp_path):
    write_json(tmp_path / "run.json", {"runs": [batch("p", "candidate", 4, "profile")]})
    write_json(tmp_path / "windows" / "bad.json", {"run_id": "p", "rank": 0, "window_id": "bad", "profile_dir": "absent",
                                                 "status": "unsupported", "unsupported_reason": "Memory not supported"})
    report = summarize(tmp_path)
    missing = report["runs"][0]["missing"]
    assert any("window_unsupported" in value for value in missing)
    assert any("profile_dir_missing" in value for value in missing)
    assert any("pmu_counters_missing" in value for value in missing)


def test_summary_line_limit_with_many_windows(tmp_path):
    batches = [batch(f"p{i}", "candidate", 4, "profile") for i in range(10)]
    write_json(tmp_path / "run.json", {"runs": batches})
    for item in batches:
        for rank in range(4):
            add_window(tmp_path, item["run_id"], rank)
    lines = summary_lines(summarize(tmp_path))
    assert len(lines) <= 200
    assert len(("\n".join(lines) + "\n").encode("utf-8")) <= 50 * 1024
    assert lines[-1] == "=== OSCAR_TIMER_SUMMARY_END ==="


def test_no_nonfinite_output(tmp_path):
    path = tmp_path / "kernel_details.csv"
    write_csv(path, [{"Name": "bad", "Duration(us)": "nan", "aic_mac_ratio": "inf"}])
    assert parse_csv(path) == []


def test_partial_runner_failure_and_none_usage_are_reportable(tmp_path):
    write_json(tmp_path / "run.json", {"status": "failed", "error": "build failed", "services": []})
    report = summarize(tmp_path)
    assert report["evidence_status"] == "missing"
    assert any("build failed" in line for line in summary_lines(report))
    failed = {"run_id": "b4", "variant": "native", "concurrency": 4, "mode": "control",
              "status": "failed", "error": "request timeout", "sample": {"requests": [{"usage": None}]}}
    write_json(tmp_path / "run.json", {"status": "failed", "runs": [failed]})
    lines = summary_lines(summarize(tmp_path))
    assert any("request timeout" in line for line in lines)


def test_device_phase_comparison_requires_matching_scheduler_context(tmp_path):
    write_json(tmp_path / "run.json", {"runs": [batch("np", "native", 4, "profile"), batch("op", "candidate", 4, "profile")]})
    add_window(tmp_path, "np", 0, op_duration=1000)
    add_window(tmp_path, "op", 0, op_duration=2000)
    report = summarize(tmp_path)
    comparison = report["device_phase_comparisons"][0]
    assert comparison["alignment"] == "same_scheduler_shape"
    assert comparison["phase_task_ms"]["oscar.history_window_attention"]["delta"] == 1
    path = tmp_path / "timer-steps-rank0-pid1.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["context_max"] += 10
    path.write_text("\n".join(json.dumps(row) for row in rows))
    comparison = summarize(tmp_path)["device_phase_comparisons"][0]
    assert comparison["alignment"] == "not_aligned"
    assert comparison["phase_task_ms"]["oscar.history_window_attention"]["delta"] is None


def test_missing_mature_batch_coverage_is_reported(tmp_path):
    write_json(tmp_path / "run.json", {"runs": [batch("p", "candidate", 8, "profile")]})
    write_json(tmp_path / "windows" / "p" / "rank0" / "run-summary.json", {
        "t": "oscar-diagnostic-run", "run_id": "p", "rank": 0, "target_batch": 8,
        "max_decode_requests": 6, "missing_windows": {"decode": ["PipeUtilization", "Memory"]},
        "missing_reason": "mature_decode_not_observed"})
    report = summarize(tmp_path)
    lines = summary_lines(report)
    assert any("WINDOW_COVERAGE" in line and "mature_decode_not_observed" in line for line in lines)
    assert report["evidence_status"] == "missing"


def test_byte_budget_reserves_comparisons_errors_and_footer(tmp_path):
    from tools.summarize_timer import _bounded_lines

    lines = ["=== OSCAR_TIMER_SUMMARY_BEGIN ===", "RUN r"]
    lines += [f"PMU {i} " + "长名字🔥" * 500 for i in range(150)]
    lines += ["COMPARE ratio=0.8", "DEVICE_COMPARE rank0 native=10 oscar=20", "GLOBAL_MISSING missing_hbm",
              "FULL_REPORT /tmp/timer-summary.json", "=== OSCAR_TIMER_SUMMARY_END ==="]
    bounded = _bounded_lines(lines)
    assert len(bounded) <= 200
    assert len(("\n".join(bounded) + "\n").encode("utf-8")) <= 50 * 1024
    assert all(line in bounded for line in lines[-5:])
    assert bounded[0] == lines[0]
    assert bounded[-1] == lines[-1]
    assert any(line.startswith("SUMMARY_COMPACTED") for line in bounded)
