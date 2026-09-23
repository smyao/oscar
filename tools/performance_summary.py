"""Compact terminal view of a paired synthetic long-request diagnostic.

Archive #70-#73/#132-#134: HTTP/SSE wall time cannot identify device phases.
#94/#95/#125: keep a visible result and evidence paths while full logs retain
errors. #137-#139: show long-request concurrency and incomplete requests; a
single diagnostic does not establish the performance acceptance matrix (§10).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from benchmarks.compare import read_json


def _finite_positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _number(value, digits=2, *, allow_zero=False):
    valid = _finite_nonnegative(value) if allow_zero else _finite_positive(value)
    return f"{value:.{digits}f}" if valid else "NA"


def _ratio(candidate, baseline):
    if not _finite_positive(candidate) or not _finite_positive(baseline):
        return None
    result = candidate / baseline
    return result if math.isfinite(result) and result > 0 else None


def _sample(report):
    if report.get("mode") == "synthetic_mixed":
        section = report.get("synthetic_mixed")
        return section.get("sample") if isinstance(section, dict) else None
    batches = report.get("batches")
    return batches[0] if isinstance(batches, list) and len(batches) == 1 else None


def _variant(report):
    return report.get("variant", report.get("kind"))


def _pair_identity(report):
    section = report.get("synthetic_mixed")
    return section.get("pair_sha256") if isinstance(section, dict) else None


def _unwrap(report, variant):
    """Accept the one-click full-service report and the fast paired report."""
    if report.get("mode") == "probe" and isinstance(report.get("performance"), dict):
        performance = report["performance"]
        return {"variant": variant, "mode": "synthetic_mixed",
                "status": "measured" if (report.get("status") == "passed"
                                        and performance.get("status") == "measured") else "failed",
                "synthetic_mixed": performance}
    return report


def _completed(sample, expected):
    if not isinstance(sample, dict) or not isinstance(sample.get("requests"), list):
        return False
    requests = sample["requests"]
    return (type(sample.get("completed_requests")) is int
            and sample["completed_requests"] == expected
            and sample.get("failed_requests") == 0 and sample.get("timeouts") == 0
            and sample.get("status") == "completed" and len(requests) == expected
            and all(isinstance(row, dict) and row.get("status") == "completed"
                    for row in requests))


def _counts(sample, expected):
    if not isinstance(sample, dict):
        return f"0/{expected} failed=NA timeout=NA"
    return (f"{sample.get('completed_requests', 0)}/{expected} "
            f"failed={sample.get('failed_requests', 'NA')} "
            f"timeout={sample.get('timeouts', 'NA')}")


def _metric(sample, name, percentile):
    latency = sample.get("latency_ms") if isinstance(sample, dict) else None
    row = latency.get(name) if isinstance(latency, dict) else None
    value = row.get(percentile) if isinstance(row, dict) else None
    valid = _finite_nonnegative(value) if name == "tpot_ms" else _finite_positive(value)
    return value if valid else None


def _throughput(sample, name):
    throughput = sample.get("throughput_tps") if isinstance(sample, dict) else None
    value = throughput.get(name) if isinstance(throughput, dict) else None
    return value if _finite_positive(value) else None


def _queue_peaks(sample):
    rows = sample.get("running_waiting_samples") if isinstance(sample, dict) else None
    if not isinstance(rows, list):
        return None
    pairs = [(row.get("running"), row.get("waiting")) for row in rows if isinstance(row, dict)]
    pairs = [(running, waiting) for running, waiting in pairs
             if _finite_nonnegative(running) and _finite_nonnegative(waiting)]
    if not pairs:
        return None
    return max(r for r, _ in pairs), max(w for _, w in pairs), max(r + w for r, w in pairs)


def _mtp_acceptance(report):
    section = report.get("synthetic_mixed") if isinstance(report, dict) else None
    raw = section.get("mtp_counter_delta") if isinstance(section, dict) else None
    delta = raw.get("mtp_counter_delta") if isinstance(raw, dict) and raw.get("status") == "observed" else None
    if not isinstance(delta, dict):
        return None
    proposed = delta.get("vllm:spec_decode_num_draft_tokens")
    accepted = delta.get("vllm:spec_decode_num_accepted_tokens")
    drafts = delta.get("vllm:spec_decode_num_drafts")
    if (not _finite_positive(proposed) or not _finite_nonnegative(accepted)
            or accepted > proposed or not _finite_positive(drafts)):
        return None
    return accepted / proposed, 1 + accepted / drafts


def _latency_line(name, native, oscar):
    fields = []
    zero_burst = False
    for percentile in ("p50", "p95"):
        baseline, candidate = _metric(native, name, percentile), _metric(oscar, name, percentile)
        if name == "tpot_ms" and (baseline == 0 or candidate == 0):
            zero_burst = True
        fields.append(f"{percentile}_native={_number(baseline, allow_zero=name == 'tpot_ms')}")
        fields.append(f"{percentile}_oscar={_number(candidate, allow_zero=name == 'tpot_ms')}")
        fields.append(f"{percentile}_ratio={_number(_ratio(candidate, baseline))}")
    note = " zero=unresolved_SSE_burst" if zero_burst else ""
    return f"[oscar] PERF_{name.upper()} scope=client_SSE_ms" + note + " " + " ".join(fields)


def _hint(native, oscar, comparable):
    if not comparable:
        return "incomplete_or_unpaired; device_cause=unknown"
    ttft = _ratio(_metric(oscar, "ttft_ms", "p50"), _metric(native, "ttft_ms", "p50"))
    tpot = _ratio(_metric(oscar, "tpot_ms", "p50"), _metric(native, "tpot_ms", "p50"))
    if tpot is None and (_metric(native, "tpot_ms", "p50") == 0
                         or _metric(oscar, "tpot_ms", "p50") == 0):
        return ("queue_or_prefill; tpot_unresolved_SSE_burst; device_cause=unknown"
                if ttft is not None and ttft > 1
                else "tpot_unresolved_SSE_burst; device_cause=unknown")
    if ttft is None or tpot is None:
        return "missing_client_timing; device_cause=unknown"
    if ttft > 1 and tpot > 1:
        return "queue_or_prefill_plus_generation; device_cause=unknown"
    if ttft > 1:
        return "queue_or_prefill; device_cause=unknown"
    if tpot > 1:
        return "generation_or_interference; device_cause=unknown"
    return "no_client_latency_regression_seen; device_cause=unknown"


def _strict_client_gate(comparison, native, oscar):
    """A single-run client ratio screen, never an NPU acceptance decision."""
    rows = comparison.get("batches")
    if not isinstance(rows, list) or len(rows) != 1:
        return "unavailable"
    batch = rows[0]
    if not isinstance(batch, dict) or not isinstance(batch.get("requests"), list):
        return "unavailable"
    if len(batch["requests"]) != len(native["requests"]) or len(batch["requests"]) != len(oscar["requests"]):
        return "unavailable"
    native_requests = {row.get("request_id"): row for row in native["requests"]}
    oscar_requests = {row.get("request_id"): row for row in oscar["requests"]}
    comparison_requests = {row.get("request_id"): row for row in batch["requests"]}
    if (None in native_requests or None in oscar_requests or None in comparison_requests
            or len(native_requests) != len(native["requests"])
            or len(oscar_requests) != len(oscar["requests"])
            or len(comparison_requests) != len(batch["requests"])
            or native_requests.keys() != oscar_requests.keys()
            or native_requests.keys() != comparison_requests.keys()):
        return "unavailable"
    all_ratios = []
    for request_id, baseline in native_requests.items():
        candidate = oscar_requests[request_id]
        if (baseline.get("prompt_sha256") != candidate.get("prompt_sha256")
                or baseline.get("prompt_tokens") != candidate.get("prompt_tokens")):
            return "unavailable"
        for name in ("ttft_ms", "tpot_ms", "e2e_ms"):
            value = _ratio(candidate.get(name), baseline.get(name))
            if not _finite_positive(value):
                return "unavailable"
            all_ratios.append(value)
    if any(_metric(sample, name, percentile) is None
           for sample in (native, oscar)
           for name in ("ttft_ms", "tpot_ms", "e2e_ms")
           for percentile in ("p50", "p95")):
        return "unavailable"
    throughput_ratios = [_ratio(_throughput(oscar, name), _throughput(native, name))
                         for name in ("prompt", "generation")]
    if any(value is None for value in throughput_ratios):
        return "unavailable"
    measured = ("met" if all(value <= 1.0 for value in all_ratios)
                and all(value >= 1.0 for value in throughput_ratios) else "regressed")
    # A structurally valid comparator may still fail its ratio gate. Never
    # override a separate failed gate with a local "met" observation.
    return "unavailable" if measured == "met" and comparison.get("status") == "failed" else measured


def _diagnostic_status(comparison):
    return comparison.get("diagnostic_status", comparison.get("status"))


def format_performance_summary(native_report, oscar_report, comparison, *,
                               native_path, oscar_path, comparison_path):
    """Return <=8 copyable lines; full raw evidence remains at given paths.

    `met` means this one client burst meets strict 1.0x ratios on every
    request. It deliberately never means device, model, or overall acceptance.
    """
    if any(not isinstance(report, dict) for report in (native_report, oscar_report, comparison)):
        raise ValueError("native, oscar and comparison reports must be mappings")
    native_report = _unwrap(native_report, "native")
    oscar_report = _unwrap(oscar_report, "oscar")
    if _variant(native_report) != "native" or _variant(oscar_report) != "oscar":
        raise ValueError("reports require explicit native and oscar variants")
    native, oscar = _sample(native_report), _sample(oscar_report)
    section = oscar_report.get("synthetic_mixed") if oscar_report.get("mode") == "synthetic_mixed" else None
    lengths = section.get("prompt_lengths") if isinstance(section, dict) else None
    expected = len(lengths) if isinstance(lengths, list) and lengths else 4
    workload = ""
    if isinstance(section, dict):
        identity = section.get("pair_identity")
        output_tokens = identity.get("output_tokens") if isinstance(identity, dict) else None
        if isinstance(lengths, list) and all(type(length) is int and length > 0 for length in lengths):
            workload += " inputs=" + ",".join(str(length) for length in lengths)
        if type(output_tokens) is int and output_tokens > 0:
            workload += f" output_tokens={output_tokens}"
    complete = (_completed(native, expected) and _completed(oscar, expected)
                and _diagnostic_status(comparison) == "diagnostic_measured"
                and _pair_identity(native_report) is not None
                and _pair_identity(native_report) == comparison.get("pair_sha256")
                and _pair_identity(oscar_report) == comparison.get("pair_sha256")
                and native_report.get("status") in ("measured", "needs_evidence")
                and oscar_report.get("status") in ("measured", "needs_evidence"))
    lines = [f"[oscar] PERF_STATUS pairing={'complete' if complete else 'incomplete'} "
             f"native={_counts(native, expected)} oscar={_counts(oscar, expected)} "
             f"scope=synthetic_K{expected}{workload} warmup_pairing=unpaired"]
    if complete:
        native_queue, oscar_queue = _queue_peaks(native), _queue_peaks(oscar)
        if native_queue is not None or oscar_queue is not None:
            left = native_queue or ("NA", "NA", "NA")
            right = oscar_queue or ("NA", "NA", "NA")
            lines.append(f"[oscar] PERF_QUEUE peak_running_native={left[0]} peak_running_oscar={right[0]} "
                         f"peak_waiting_native={left[1]} peak_waiting_oscar={right[1]} "
                         f"peak_inflight_native={left[2]} peak_inflight_oscar={right[2]} "
                         "scope=sampled_1s")
        native_mtp, oscar_mtp = _mtp_acceptance(native_report), _mtp_acceptance(oscar_report)
        if native_mtp is not None or oscar_mtp is not None:
            left = native_mtp or (None, None)
            right = oscar_mtp or (None, None)
            mtp_text = (f" mtp_acceptance_native={_number(left[0], 3, allow_zero=True)}"
                        f" mtp_acceptance_oscar={_number(right[0], 3, allow_zero=True)}"
                        f" mtp_mean_length_native={_number(left[1], 2)}"
                        f" mtp_mean_length_oscar={_number(right[1], 2)}")
            if lines[-1].startswith("[oscar] PERF_QUEUE"):
                lines[-1] += mtp_text
            else:
                lines.append("[oscar] PERF_MTP scope=native_metrics" + mtp_text)
        lines.append(_latency_line("ttft_ms", native, oscar))
        lines.append(_latency_line("tpot_ms", native, oscar))
        lines.append(_latency_line("e2e_ms", native, oscar))
        parts = []
        for name in ("prompt", "generation"):
            baseline, candidate = _throughput(native, name), _throughput(oscar, name)
            parts.extend((f"{name}_native={_number(baseline, 1)}",
                          f"{name}_oscar={_number(candidate, 1)}",
                          f"{name}_ratio={_number(_ratio(candidate, baseline))}"))
        lines.append("[oscar] PERF_THROUGHPUT scope=client_wall_tokens_per_s " + " ".join(parts))
    gate = _strict_client_gate(comparison, native, oscar) if complete else "unavailable"
    lines.append(f"[oscar] PERF_VERDICT client_ratio_gate={gate} acceptance=not_established "
                 f"hint={_hint(native, oscar, complete)}")
    lines.append(f"[oscar] PERF_EVIDENCE native={native_path} oscar={oscar_path} "
                 f"comparison={comparison_path}")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--oscar", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    args = parser.parse_args(argv)
    for line in format_performance_summary(
        read_json(args.native), read_json(args.oscar), read_json(args.comparison),
        native_path=args.native, oscar_path=args.oscar, comparison_path=args.comparison
    ):
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
