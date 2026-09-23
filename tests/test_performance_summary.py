"""Archive #70-#73/#94/#95/#125/#132-#139: compact, honest console evidence."""

from benchmarks.mixed import compare_mixed
from tools.performance_summary import format_performance_summary


def _report(kind, *, latency_scale=1, throughput_scale=1, tpot=10, failed=False):
    requests = []
    for index, length in enumerate((20000, 23000, 27000, 30000)):
        requests.append({"request_id": f"synthetic-{index}-{length}",
                         "status": "timeout" if failed and index == 3 else "completed",
                         "prompt_sha256": f"prompt-{index}", "prompt_tokens": length,
                         "ttft_ms": (100 + index * 10) * latency_scale,
                         "tpot_ms": tpot * latency_scale,
                         "e2e_ms": (200 + index * 10) * latency_scale})
    sample = {"status": "failed" if failed else "completed",
              "completed_requests": 3 if failed else 4,
              "failed_requests": 0, "timeouts": 1 if failed else 0,
              "requests": requests,
              "latency_ms": {name: {"p50": value * latency_scale, "p95": value * latency_scale}
                             for name, value in (("ttft_ms", 115), ("tpot_ms", tpot), ("e2e_ms", 215))},
              "throughput_tps": {"prompt": 5000 * throughput_scale,
                                  "generation": 50 * throughput_scale},
              "running_waiting_samples": [{"running": 18 if kind == "native" else 4,
                                           "waiting": 14 if kind == "native" else 28}],
              "repeat": 0}
    return {"mode": "synthetic_mixed", "variant": kind,
            "status": "failed" if failed else "measured",
            "synthetic_mixed": {"status": "failed" if failed else "measured",
                                "pair_sha256": "same-pair", "prompt_manifest": [
                                    {"id": row["request_id"], "prompt_tokens": row["prompt_tokens"],
                                     "prompt_sha256": row["prompt_sha256"]} for row in requests],
                                "prompt_lengths": [20000, 23000, 27000, 30000],
                                "sample": sample}}


def _lines(native, oscar):
    return format_performance_summary(native, oscar, compare_mixed(native, oscar),
                                      native_path="/e/native.json", oscar_path="/e/oscar.json",
                                      comparison_path="/e/paired.json")


def test_regression_is_copyable_and_never_claims_device_acceptance():
    lines = _lines(_report("native"), _report("oscar", latency_scale=2, throughput_scale=.5))
    assert len(lines) <= 8
    assert all(line.startswith("[oscar] PERF_") for line in lines)
    assert "native=4/4" in lines[0] and "oscar=4/4" in lines[0]
    assert "warmup_pairing=unpaired" in lines[0]
    assert any("PERF_QUEUE" in line and "peak_running_native=18" in line
               and "peak_running_oscar=4" in line for line in lines)
    assert any("PERF_TTFT_MS scope=client_SSE_ms" in line and "p50_ratio=2.00" in line for line in lines)
    assert any("PERF_TPOT_MS" in line and "zero=unresolved_SSE_burst" not in line for line in lines)
    assert any("PERF_THROUGHPUT scope=client_wall_tokens_per_s" in line
               and "prompt_ratio=0.50" in line and "generation_ratio=0.50" in line for line in lines)
    assert any("client_ratio_gate=regressed acceptance=not_established" in line for line in lines)
    assert any("device_cause=unknown" in line for line in lines)
    assert lines[-1] == "[oscar] PERF_EVIDENCE native=/e/native.json oscar=/e/oscar.json comparison=/e/paired.json"


def test_compact_queue_line_includes_mtp_acceptance_when_native_counters_exist():
    native, oscar = _report("native"), _report("oscar")
    names = ("vllm:spec_decode_num_drafts", "vllm:spec_decode_num_draft_tokens",
             "vllm:spec_decode_num_accepted_tokens")
    native["synthetic_mixed"]["mtp_counter_delta"] = {"status": "observed",
        "mtp_counter_delta": dict(zip(names, (10, 30, 24)))}
    oscar["synthetic_mixed"]["mtp_counter_delta"] = {"status": "observed",
        "mtp_counter_delta": dict(zip(names, (10, 30, 18)))}
    lines = _lines(native, oscar)
    assert len(lines) <= 8
    assert any("PERF_QUEUE" in line and "mtp_acceptance_native=0.800"
               in line and "mtp_acceptance_oscar=0.600" in line for line in lines)


def test_strict_client_gate_requires_every_request_and_both_throughputs():
    native = _report("native")
    oscar = _report("oscar", latency_scale=.8, throughput_scale=1.2)
    assert "client_ratio_gate=met acceptance=not_established" in _lines(native, oscar)[-2]
    oscar["synthetic_mixed"]["sample"]["requests"][0]["e2e_ms"] = 201
    assert "client_ratio_gate=regressed" in _lines(native, oscar)[-2]


def test_incomplete_requests_print_count_but_no_ratio_gate():
    lines = _lines(_report("native"), _report("oscar", failed=True))
    assert "pairing=incomplete" in lines[0] and "oscar=3/4" in lines[0]
    assert not any("PERF_TTFT" in line or "PERF_THROUGHPUT" in line for line in lines)
    assert "client_ratio_gate=unavailable acceptance=not_established" in lines[-2]


def test_zero_tpot_from_one_sse_burst_is_unresolved():
    native = _report("native", tpot=0)
    oscar = _report("oscar", tpot=0)
    lines = _lines(native, oscar)
    assert any("PERF_TPOT_MS" in line and "p50_native=0.00" in line
               and "p50_ratio=NA" in line and "zero=unresolved_SSE_burst" in line for line in lines)
    assert "client_ratio_gate=unavailable" in lines[-2]
    assert "tpot_unresolved_SSE_burst" in lines[-2]


def test_one_click_service_report_unwraps_performance():
    native = _report("native")
    oscar = _report("oscar")
    comparison = compare_mixed(native, oscar)
    full_service = {"mode": "probe", "status": "passed", "performance": oscar["synthetic_mixed"]}
    lines = format_performance_summary(native, full_service, comparison,
                                       native_path="native.json", oscar_path="service-probe-report.json",
                                       comparison_path="compare.json")
    assert "pairing=complete" in lines[0]
    assert "client_ratio_gate=met acceptance=not_established" in lines[-2]


def test_strict_comparator_failure_still_displays_measured_regression():
    native = _report("native")
    oscar = _report("oscar", latency_scale=2)
    comparison = compare_mixed(native, oscar)
    comparison.update(status="failed", diagnostic_status="diagnostic_measured",
                      issues=["ttft ratio exceeds 1.0"], performance_acceptance="not_run")
    lines = format_performance_summary(native, oscar, comparison,
                                       native_path="native.json", oscar_path="oscar.json",
                                       comparison_path="comparison.json")
    assert "pairing=complete" in lines[0]
    assert any("PERF_TTFT_MS" in line and "p50_ratio=2.00" in line for line in lines)
    assert "client_ratio_gate=regressed acceptance=not_established" in lines[-2]


def test_failed_policy_cannot_be_overridden_by_local_met_ratios():
    native = _report("native")
    oscar = _report("oscar", latency_scale=.8, throughput_scale=1.2)
    comparison = compare_mixed(native, oscar)
    comparison.update(status="failed", diagnostic_status="diagnostic_measured",
                      issues=["frozen policy unavailable"])
    lines = format_performance_summary(native, oscar, comparison,
                                       native_path="native.json", oscar_path="oscar.json",
                                       comparison_path="comparison.json")
    assert "client_ratio_gate=unavailable" in lines[-2]


def test_pair_fingerprint_mismatch_never_shows_ratios():
    native = _report("native")
    oscar = _report("oscar")
    comparison = compare_mixed(native, oscar)
    comparison["pair_sha256"] = "wrong-pair"
    lines = format_performance_summary(native, oscar, comparison,
                                       native_path="native.json", oscar_path="oscar.json",
                                       comparison_path="comparison.json")
    assert "pairing=incomplete" in lines[0]
    assert not any("PERF_THROUGHPUT" in line for line in lines)
    assert "client_ratio_gate=unavailable" in lines[-2]
