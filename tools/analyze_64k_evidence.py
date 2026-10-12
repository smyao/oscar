"""Parse the 2026-10-09 64K/120/K32 evidence without running a benchmark.

档案 #137/#140: server interval counters and client SSE latency have different
denominators; HTTP 200 is not completion. #152/#155: capacity and a short timing
window cannot establish whole-model speed or an attention phase percentage.
References (read-only): vllm/v1/metrics/loggers.py LoggingStatLogger and
vllm/v1/spec_decode/metrics.py SpecDecodingLogging under references/vllm.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from fractions import Fraction
import hashlib
import json
from math import ceil, floor
from pathlib import Path
import re
import statistics


ENGINE_COLUMNS = ["source_line", "time", "prompt_tps", "generation_tps",
                  "running", "waiting", "kv_usage_pct", "prefix_hit_pct"]
SPEC_COLUMNS = ["source_line", "time", "mean_acceptance_length",
                "accepted_tps", "drafted_tps", "accepted_tokens",
                "drafted_tokens", "position_acceptance_rates", "acceptance_pct"]
TABLE_STATS = ["average", "min", "max", "median", "p75", "p90", "p99"]
HEADER = re.compile(r"\(APIServer pid=(\d+)\) INFO (\d\d-\d\d) (\d\d:\d\d:\d\d)")
ENGINE = re.compile(
    r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: "
    r"([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs, "
    r"GPU KV cache usage: ([\d.]+)%, Prefix cache hit rate: ([\d.]+)%"
)
SPEC = re.compile(
    r"Mean acceptance length: ([\d.]+), Accepted throughput: ([\d.]+) tokens/s, "
    r"Drafted throughput: ([\d.]+)(?: tokens/s, Accepted: (\d+) tokens, "
    r"Drafted: (\d+) tokens, Per-position acceptance rate: ([\d., ]+), "
    r"Avg Draft acceptance rate: ([\d.]+)%)?"
)
CLIENT_TIME = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})\]")
EVENT_MARKERS = {
    "warmup_start": "Start warmup, run with concurrency:",
    "warmup_end": "Warmup finished Total Count:",
    "traffic_start": "Traffic request rate:",
    "benchmark_dispatch": "Debug mode, run with concurrency:",
    "deserialization_end": "All subprocesses have finished deserializing",
    "progress_start": "Starting progress bar Total data num:",
    "api_task_end": "Api infer task time elapsed:",
    "inference_tasks_end": "Inference tasks completed.",
    "results_printed": "Performance Results of task",
}


def seconds(hms: str) -> int:
    h, m, s = map(int, hms.split(":"))
    return h * 3600 + m * 60 + s


def elapsed(a: str, b: str) -> float:
    fmt = "%Y-%m-%d %H:%M:%S,%f"
    return (datetime.strptime(b, fmt) - datetime.strptime(a, fmt)).total_seconds()


def distribution(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {"n": len(values), "min": min(values), "median": statistics.median(values),
            "mean": statistics.mean(values), "max": max(values)}


def ratio(a: float, b: float) -> float | None:
    return a / b if b else None


def rounded_usage_pool_bounds(used: int, displayed_pct: float) -> dict:
    """Invert one-decimal KV usage under an explicit used-block assumption.

    Reference block_pool.py get_usage excludes its one null block from both
    used and usable counts; configured num_gpu_blocks is usable + 1.
    """
    pct = Fraction(str(displayed_pct))
    half_unit = Fraction(1, 20)
    if used < 1 or pct <= half_unit:
        raise ValueError("Positive used blocks and nonzero rounded usage required")
    low, high = pct - half_unit, pct + half_unit
    # Include both tie boundaries, then reproduce the reference print format.
    candidates = [
        n for n in range(floor(100 * used / high), ceil(100 * used / low) + 1)
        if n >= used and format(100 * (1 - (n - used) / n), ".1f") == format(displayed_pct, ".1f")
    ]
    if not candidates:
        raise ValueError("No integer pool size matches the assumed usage")
    return {
        "assumed_request_used_blocks_excluding_null": used,
        "displayed_usage_pct": displayed_pct,
        "rounding_interval_pct": [float(low), float(high)],
        "rounding_boundary_policy": "Candidate ties are evaluated using the reference floating expression 100*(1-free/usable), then one-decimal formatting. Floating cancellation can retain the upper mathematical tie.",
        "usable_pool_blocks_range": [min(candidates), max(candidates)],
        "configured_num_gpu_blocks_range": [min(candidates) + 1, max(candidates) + 1],
    }


def conditional_warmup_pool_bounds(runs: dict) -> dict:
    """Attach a conditional source-layout inference, never an observed budget."""
    result = {
        "status": "conditional_inference_not_measured_pool_size",
        "reference": "references/vllm/vllm/v1/core/block_pool.py:get_usage (num_gpu_blocks minus one null block)",
        "assumptions": [
            "Reference model geometry supplied by the memory analysis applies to this run: 17 FULL, 48 GDN, rank Hkv=1/D256; not established by the performance log.",
            "Each GDN request retains four states; the native mixed pool charges 12 GDN blocks per request.",
            "The single warmup request has its complete 65100-token prompt resident, uses OSCAR 24 FULL + 12 GDN blocks, or native 85-86 FULL + 12 GDN blocks.",
            "No other allocated/cached workload contributes to pool usage; Running=1/Waiting=0 and prefix-hit=0 alone do not prove this allocator state.",
            "Server/client printed clocks are sufficiently aligned only to identify warmup before batch dispatch; no cross-clock duration or benchmark allocation is derived.",
        ],
        "ideal_real_arithmetic_formula": "pct = 100 * used_blocks / (num_gpu_blocks - 1)",
        "reference_float_formula": "displayed_pct = format(100 * (1 - free_blocks / (num_gpu_blocks - 1)), '.1f'); free_blocks = num_gpu_blocks - 1 - used_blocks; enumerate all consistent integers",
    }
    for variant, block_counts in (("oscar", [36]), ("native", [97, 98])):
        row = runs[variant]["engine"][0]
        if row[4] != 1 or row[5] != 0 or row[2] <= 0:
            raise ValueError(f"Expected first {variant} Engine sample to be warmup R1/W0 with prompt progress")
        spec = next((s for s in runs[variant]["spec"] if s[1] == row[1]), None)
        ranges = [rounded_usage_pool_bounds(used, row[6]) for used in block_counts]
        result[variant] = {
            "engine_line": row[0], "server_time": row[1],
            "prompt_tps": row[2], "generation_tps": row[3],
            "running": row[4], "waiting": row[5], "kv_usage_pct": row[6],
            "spec_line": spec[0] if spec else None,
            "accepted_tokens": spec[5] if spec else None,
            "drafted_tokens": spec[6] if spec else None,
            "warmup_start": runs[variant]["client"]["events"]["warmup_start"],
            "warmup_end": runs[variant]["client"]["events"]["warmup_end"],
            "batch_dispatch": runs[variant]["client"]["events"]["benchmark_dispatch"],
            "cases": ranges,
            "union_configured_num_gpu_blocks_range": [
                min(r["configured_num_gpu_blocks_range"][0] for r in ranges),
                max(r["configured_num_gpu_blocks_range"][1] for r in ranges),
            ],
        }
    return result


def summarize_server(engine: list[list], spec: list[list], fragments: list[dict]) -> dict:
    """Keep raw sample data and make only within-server-clock calculations."""
    if not engine:
        raise ValueError("Missing Engine samples")
    active = [r for r in engine if r[4] > 0]
    full_spec = [r for r in spec if r[5] is not None]
    accepted = sum(r[5] for r in full_spec)
    drafted = sum(r[6] for r in full_spec)
    # Three draft positions occur in this supplied trace. Check the conditional
    # D/3 inference against independently printed, rounded acceptance lengths.
    fixed_three_consistent = bool(full_spec) and all(
        len(r[7]) == 3 and r[6] > 0 and r[6] % 3 == 0
        and abs((1 + 3 * r[5] / r[6]) - r[2]) <= 0.00501 for r in full_spec
    )
    rounds = drafted // 3 if fixed_three_consistent else None
    intervals = []
    for previous, row in zip(engine, engine[1:]):
        dt = seconds(row[1]) - seconds(previous[1])
        if dt <= 0:
            raise ValueError("Engine timestamps must increase within one server PID")
        intervals.append((row, dt))
    zeros = [r for r in active if r[2] == 0 and r[3] == 0]
    groups = {}
    for name, predicate in {
        "prompt_counter_positive": lambda r: r[2] > 0,
        "prompt_counter_zero": lambda r: r[2] == 0,
        "generation_at_most_10_tps": lambda r: r[3] <= 10,
        "generation_at_least_100_tps": lambda r: r[3] >= 100,
        "kv_at_least_95_pct": lambda r: r[6] >= 95,
    }.items():
        selected = [r for r in active if predicate(r)]
        groups[name] = {
            "sample_count": len(selected),
            "sample_fraction_of_active": ratio(len(selected), len(active)),
            "generation_tps": distribution([r[3] for r in selected]),
            "running": distribution([r[4] for r in selected]),
        }
    peak_running = max(r[4] for r in engine)
    peak_first = next(r for r in engine if r[4] == peak_running)
    positive_waiting = [r for r in engine if r[5] > 0]
    last_busy = max((i for i, r in enumerate(engine) if r[4] > 0), default=-1)
    drained = engine[last_busy + 1] if last_busy + 1 < len(engine) else None
    last_wait_index = max((i for i, r in enumerate(engine) if r[5] > 0), default=-1)
    wait_zero = engine[last_wait_index + 1] if last_wait_index + 1 < len(engine) else None
    return {
        "sample_counts": {"engine": len(engine), "spec_complete": len(full_spec),
                          "spec_partial": len(spec) - len(full_spec),
                          "unparseable_fragments": len(fragments)},
        "server_clock": {"first_engine": engine[0][1], "last_engine": engine[-1][1],
                         "span_s": seconds(engine[-1][1]) - seconds(engine[0][1]),
                         "interval_seconds_histogram": dict(Counter(dt for _, dt in intervals))},
        "peak_running": peak_running, "peak_running_first": peak_first[1],
        "first_sample_to_peak_running_s": seconds(peak_first[1]) - seconds(engine[0][1]),
        "peak_waiting": max(r[5] for r in engine),
        "peak_kv_usage_pct": max(r[6] for r in engine),
        "running_histogram": dict(sorted(Counter(r[4] for r in engine).items())),
        "active_running": distribution([r[4] for r in active]),
        "active_generation_tps": distribution([r[3] for r in active]),
        "active_prompt_tps": distribution([r[2] for r in active]),
        "sample_groups_not_device_phases": groups,
        "both_reported_throughputs_zero_while_running": {
            "sample_count": len(zeros), "source_lines": [r[0] for r in zeros],
            "times": [r[1] for r in zeros],
            "associated_logging_interval_s": sum(dt for r, dt in intervals if r in zeros),
            "meaning": "No reported token-counter progress in these logging windows; not proof of a device stall or no prefill work.",
        },
        "queue_drain": {"last_waiting_positive": positive_waiting[-1][1] if positive_waiting else None,
                        "first_subsequent_waiting_zero": wait_zero[1] if wait_zero else None,
                        "first_subsequent_running_zero": drained[1] if drained else None,
                        "sampled_waiting_zero_to_running_zero_s":
                        seconds(drained[1]) - seconds(wait_zero[1]) if drained and wait_zero else None},
        "approx_counter_integrals_after_first_engine_sample": {
            "prompt_tokens": sum(r[2] * dt for r, dt in intervals),
            "generation_tokens": sum(r[3] * dt for r, dt in intervals),
            "rate_rounding_error_bound_each_tokens": 0.05 * sum(dt for _, dt in intervals),
            "scope": "Server-to-server timestamp deltas only; first interval unknown, warmup remainder included, integer-second clock and logger jitter unaccounted. Not client benchmark counts or phase time.",
        },
        "speculative_complete_logged_windows": {
            "accepted_tokens": accepted, "drafted_tokens": drafted,
            "accepted_over_drafted": ratio(accepted, drafted),
            "drafts_if_three_tokens_each": rounds,
            "fixed_three_draft_inference_matches_every_rounded_mean": fixed_three_consistent,
            "weighted_mean_acceptance_length_if_three": 1 + accepted / rounds if rounds else None,
            "weighted_position_acceptance_if_three":
            [sum(r[7][p] * (r[6] / 3) for r in full_spec) / rounds for p in range(3)] if rounds else None,
            "position_rate_rounding_bound": 0.0005,
            "scope": "Sums of complete printed windows, including warmup. Missing windows/counts excluded. D/3 counts per-request draft events, not global model steps or launches.",
            "missing_complete_counter_lines": [r[0] for r in spec if r[5] is None],
            "engine_generation_windows_without_spec": [
                {"line": r[0], "time": r[1], "generation_tps": r[3]}
                for r in engine if r[3] > 0 and r[1] not in {s[1] for s in spec}
            ],
            "absent_spec_note": "An absent Spec line alone does not prove truncation: windows without draft events are not logged. Consult recorded source fragments.",
        },
        "preemptions": {"count": None, "status": "not_measured",
                        "reason": "No preemption counter or scheduler events supplied. Running/Waiting changes do not establish preemption; reference LoggingStatLogger resets num_preemptions in _update_stats before log checks it."},
        "absolute_kv_bytes": None,
        "memory_scope": "KV usage is a percentage of each run's own allocated block pool. Running is a sampled scheduler state; neither establishes absolute HBM or 6-7x physical memory savings.",
    }


def parse_evidence(raw: bytes, source_path: str = "") -> dict:
    text = raw.decode("utf-8")
    lines = text.splitlines()
    runs = {v: {"pid": None, "engine": [], "spec": [], "fragments": [],
                "client": {"events": {}, "performance": {}, "common": {}},
                "http_200_lines": [], "warnings": []} for v in ("oscar", "native")}
    variant = "oscar"
    for number, line in enumerate(lines, 1):
        if "下面是原生的服务日志" in line:
            variant = "native"
        run = runs[variant]
        header = HEADER.search(line)
        if header:
            pid, _, hms = header.groups()
            if run["pid"] not in (None, int(pid)):
                raise ValueError(f"Multiple server PIDs inside {variant} at line {number}")
            run["pid"] = int(pid)
            if "Engine 000:" in line:
                match = ENGINE.search(line)
                if not match:
                    raise ValueError(f"Unparseable Engine row at line {number}")
                p, g, running, waiting, kv, prefix = match.groups()
                run["engine"].append([number, hms, float(p), float(g), int(running),
                                      int(waiting), float(kv), float(prefix)])
            elif "SpecDecoding metrics:" in line:
                match = SPEC.search(line)
                if not match:
                    raise ValueError(f"Unparseable Spec row at line {number}")
                mean, at, dt, a, d, positions, rate = match.groups()
                run["spec"].append([number, hms, float(mean), float(at), float(dt),
                                    int(a) if a else None, int(d) if d else None,
                                    list(map(float, positions.split(","))) if positions else None,
                                    float(rate) if rate else None])
                if a is None:
                    run["warnings"].append({"line": number, "kind": "truncated_spec_counts",
                                             "time": hms, "missing": ["accepted", "drafted", "per_position", "acceptance_pct"]})
        elif line.startswith("(APIServer") and "HTTP/1.1" not in line:
            run["fragments"].append({"line": number, "text": line,
                                     "preceding_engine_time": run["engine"][-1][1] if run["engine"] else None,
                                     "status": "Unrecoverable fragment; no continuation before next independent record."})
        if "HTTP/1.1\" 200 OK" in line:
            run["http_200_lines"].append(number)
        ct = CLIENT_TIME.search(line)
        if ct:
            for key, marker in EVENT_MARKERS.items():
                if marker in line:
                    run["client"]["events"][key] = {"time": ct[1], "line": number}
            if "Api infer task time elapsed:" in line:
                run["client"]["api_task_elapsed_s"] = float(re.search(r"elapsed: ([\d.]+)s", line)[1])
            if "run with concurrency:" in line:
                run["client"]["concurrency"] = int(re.search(r"concurrency: (\d+)", line)[1])
            if "Warmup finished" in line:
                run["client"]["warmup_counts"] = dict(zip(
                    ("total", "success", "failed"), map(int, re.search(
                        r"Total Count: (\d+) Success Count: (\d+) Failed Count: (\d+)", line).groups())))
        if line.startswith("│"):
            cells = [c.strip() for c in line.strip("│").split("│")]
            if len(cells) == 10 and cells[1] == "total":
                run["client"]["performance"][cells[0]] = {
                    "line": number, "stage": cells[1],
                    "unit": " ".join(cells[2].split()[1:]) or "count",
                    **dict(zip(TABLE_STATS, (float(c.split()[0]) for c in cells[2:9]))),
                    "n": int(cells[9]),
                }
            elif len(cells) == 3 and cells[1] == "total":
                run["client"]["common"][cells[0]] = {
                    "line": number, "stage": cells[1], "value": float(cells[2].split()[0]),
                    "unit": " ".join(cells[2].split()[1:]) or "count",
                }
    for run in runs.values():
        run["server_summary"] = summarize_server(run["engine"], run["spec"], run["fragments"])
        c = run["client"]
        c["warmup_elapsed_s"] = elapsed(c["events"]["warmup_start"]["time"], c["events"]["warmup_end"]["time"])
        c["dispatch_to_api_end_s"] = elapsed(c["events"]["benchmark_dispatch"]["time"], c["events"]["api_task_end"]["time"])
        c["progress_start_to_api_end_s"] = elapsed(c["events"]["progress_start"]["time"], c["events"]["api_task_end"]["time"])
        c["benchmark_duration_s"] = c["common"]["Benchmark Duration"]["value"] / 1000
        c["mean_after_first_token_s"] = (c["performance"]["E2EL"]["average"] - c["performance"]["TTFT"]["average"]) / 1000
        c["mean_tpot_from_mean_latencies_ms"] = c["mean_after_first_token_s"] * 1000 / (c["performance"]["OutputTokens"]["average"] - 1)
        c["recomputed_input_tps"] = c["common"]["Total Input Tokens"]["value"] / c["benchmark_duration_s"]
        c["recomputed_output_tps"] = c["common"]["Total Generated Tokens"]["value"] / c["benchmark_duration_s"]
        c["client_mean_inflight_from_summed_request_latencies"] = c["common"]["Success Requests"]["value"] * c["performance"]["E2EL"]["average"] / 1000 / c["benchmark_duration_s"]
    a, b = (runs[v]["client"] for v in ("oscar", "native"))
    comparisons = {
        "same_workload": all(a["common"][k]["value"] == b["common"][k]["value"] for k in (
            "Total Requests", "Success Requests", "Failed Requests", "Total Input Tokens", "Total Generated Tokens")),
        "benchmark_wall_oscar_over_native": a["benchmark_duration_s"] / b["benchmark_duration_s"],
        "benchmark_wall_difference_s": a["benchmark_duration_s"] - b["benchmark_duration_s"],
        "output_throughput_oscar_over_native": a["recomputed_output_tps"] / b["recomputed_output_tps"],
        "observed_peak_running_oscar_over_native": runs["oscar"]["server_summary"]["peak_running"] / runs["native"]["server_summary"]["peak_running"],
        "client_metric_ratios_oscar_over_native": {
            metric: {stat: ratio(a["performance"][metric][stat], b["performance"][metric][stat]) for stat in TABLE_STATS}
            for metric in ("E2EL", "TTFT", "TPOT", "ITL", "OutputTokenThroughput")},
        "mean_after_first_token_oscar_over_native": a["mean_after_first_token_s"] / b["mean_after_first_token_s"],
        "mean_after_first_token_difference_s": a["mean_after_first_token_s"] - b["mean_after_first_token_s"],
        "wall_targets_s": {"four_times_faster_than_current": a["benchmark_duration_s"] / 4,
                           "five_times_faster_than_current": a["benchmark_duration_s"] / 5,
                           "four_times_faster_than_native": b["benchmark_duration_s"] / 4,
                           "five_times_faster_than_native": b["benchmark_duration_s"] / 5},
    }
    return {
        "schema_version": 1,
        "source": {"path": source_path, "bytes": len(raw), "lines": len(lines),
                   "sha256": hashlib.sha256(raw).hexdigest(),
                   "archive_refs": [137, 140, 152, 155]},
        "columns": {"engine": ENGINE_COLUMNS, "spec": SPEC_COLUMNS},
        "scope": "Only supplied 2026-10-09 65100-input/512-output, 120-request, concurrency-32 comparison. No other performance workload used.",
        "boundaries": [
            "AISBench Benchmark Duration is the completed batch wall denominator. Warmup, API task elapsed, dispatch envelope and summary overhead remain separate.",
            "Server and client clocks are not independently synchronized by this evidence. No cross-clock subtraction or client-benchmark attribution of server counter windows is made.",
            "prompt_tps=0 labels only a prompt-counter logging window. It does not establish pure decode or a device phase.",
            "HTTP 200 counts are informational. Completion is established by client Success Requests=120, Failed Requests=0 and the output totals.",
            "ITL is the reported AISBench distribution; raw request/chunk JSON is absent. Its N=120 must not be interpreted as the number of token intervals or device kernels.",
            "Spec totals cover complete printed windows only, include warmup, and have missing counters. Fixed-three inferred draft events are not global iteration counts.",
            "The Prefill Token Throughput client row uses a different denominator from completed-batch Input Token Throughput; it is not device prefill throughput.",
            "KV block-pool percentage, sampled Running and process memory check are not per-request physical KV bytes or absolute NPU HBM savings.",
        ],
        "runs": runs,
        "comparison": comparisons,
        "conditional_single_request_warmup_pool_inference": conditional_warmup_pool_bounds(runs),
        "simulation_observables_required": [
            "Same 120*65100 input and 120*512 output workload with client concurrency cap 32; count completed requests and tokens separately from HTTP headers.",
            "Independent original-baseline fit to client wall/TTFT/TPOT/E2EL/ITL and server Engine+Spec trace; freeze calibration before evaluating implementation changes.",
            "Sampled residency grows to 27 versus 12 while each run approaches its own KV pool capacity; explain fixed memory, per-page layout, padding, GDN and scratch from source rather than assuming compression ratio.",
            "Slow admission/ramp, low-throughput long intervals, short generation bursts, waiting-queue drainage and P99 tails; prompt-counter-zero windows must allow prefill work.",
            "Per-request MTP draft acceptance and extra verification work reproduce the complete-window acceptance difference; preserve missing-record uncertainty and distinguish request-draft events from batch launches.",
            "Device phase attribution needs source-derived work and simulator/operator evidence; do not fit a single attention percentage from these scalar logs.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("reports/target_64k_120_20261009.txt"))
    parser.add_argument("--output", type=Path, default=Path("reports/evidence_64k_120_20261009.json"))
    args = parser.parse_args()
    result = parse_evidence(args.input.read_bytes(), args.input.as_posix())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"output": str(args.output), "source": result["source"],
                      "comparison": result["comparison"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
