"""Offline 2026-10-09 64K latency evidence and required budgets.

Archive #137/#140/#152/#155: server credit windows are not device phases;
no old LongBenchv2 records, timing fit, simulation or acceptance claim.
The latest human target (2026-10-10) is 2-3x the native 22.4-minute run.
Serving retains the native token budget; retired whole-prefill savings are
historical only and must not enter the current projected speedup.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def seconds(value):
    h, m, s = map(int, value.split(":"))
    return 3600 * h + 60 * m + s


def summary(values):
    return {"count": len(values), "min": min(values), "median": statistics.median(values),
            "mean": statistics.mean(values), "max": max(values)} if values else None


def draft_event_candidates(row):
    """Invert printed integer counters and rounded native mean/rates.

    Native metrics.py mean = 1+A/Q. D <= 3*Q remains valid after invalid-draft
    filtering; Q <= D is NOT assumed, since filtering may leave zero tokens.
    An all-rejected window has no finite upper Q from these fields alone.
    """
    line, time, mean, _, _, accepted, drafted, rates, _ = row
    if accepted is None or drafted is None:
        return {"line": line, "time": time, "status": "missing_integer_counts"}
    lower = (drafted + 2) // 3
    if accepted == 0 or mean - 1 - 0.005 <= 0:
        return {"line": line, "time": time, "status": "no_finite_upper_bound",
                "minimum": lower, "maximum": None}
    upper = math.floor(accepted / (mean - 1 - 0.005)) + 2
    candidates = []
    for q in range(max(1, lower), upper + 1):
        if format(1 + accepted / q, ".2f") != format(mean, ".2f"):
            continue
        counts = []
        for rate in rates:
            start = max(0, math.floor((rate - .0005) * q) - 1)
            end = min(q, math.ceil((rate + .0005) * q) + 1)
            counts.append([value for value in range(start, end + 1)
                           if format(value / q, ".3f") == format(rate, ".3f")])
        if any(a >= b >= c and a + b + c == accepted
               for a, b, c in itertools.product(*counts)):
            candidates.append(q)
    if not candidates:
        raise ValueError(f"native counter/rate constraints have no solution at source line {line}")
    return {"line": line, "time": time,
            "status": "unique_request_draft_event_count" if len(candidates) == 1 else "bounded_request_draft_event_count",
            "minimum": min(candidates), "maximum": max(candidates), "candidate_count": len(candidates)}


def analyze_run(run, prompt=65100, requests=120):
    engine, spec = run["engine"], run["spec"]
    times = [seconds(row[1]) for row in engine]
    if any(right - left != 10 for left, right in zip(times, times[1:])):
        raise ValueError("this evidence ledger requires the supplied uninterrupted 10-second Engine log grid")
    credits = []
    for row in engine:
        estimated = row[2] * 10 / prompt
        count = round(estimated)
        if abs(estimated - count) > .01:
            raise ValueError(f"prompt credits do not identify whole {prompt}-token requests at {row[0]}")
        if count:
            credits.append({"line": row[0], "time": row[1], "count": count})
    total_credits = sum(row["count"] for row in credits)
    if total_credits != requests + 1 or credits[0]["count"] != 1:
        raise ValueError("supplied warmup+cohort credit count differs from 121")
    cohort_credits = credits[1:]
    credit_times = [seconds(row["time"]) for row in cohort_credits for _ in range(row["count"])]
    last_credit = cohort_credits[-1]
    drained = next(row for row in engine if seconds(row[1]) >= seconds(last_credit["time"]) and row[4] == row[5] == 0)
    tail = [row for row in engine if seconds(last_credit["time"]) < seconds(row[1]) <= seconds(drained[1])]
    tail_spec = [row for row in spec if seconds(last_credit["time"]) < seconds(row[1]) <= seconds(drained[1])]
    candidates = [draft_event_candidates(row) for row in spec]
    known = [row for row in spec if row[5] is not None and row[6] is not None]
    tail_a = sum(row[5] for row in tail_spec if row[5] is not None)
    tail_d = sum(row[6] for row in tail_spec if row[6] is not None)
    bursts = []
    by_time = {row[1]: row for row in engine}
    for row in known:
        if row[4] >= 150:
            e = by_time[row[1]]
            bursts.append({"line": row[0], "time": row[1], "accepted": row[5], "drafted": row[6],
                "conditional_fixed_three_request_events": row[6] // 3,
                "sampled_running": e[4], "sampled_waiting": e[5], "engine_generation_tps": e[3],
                "global_verify_steps_lower_bound_at_client_K32": math.ceil((row[6] / 3) / 32)})
    finite = [value for value in candidates if value.get("maximum") is not None]
    return {
        "first_output_credit_reconstruction": {
            "status": "conditional_integer_reconstruction_from_10s_prompt_counter_grid",
            "assumptions": "Each credit is the supplied 65100-token uncached input; one initial warmup. Logger jitter is not recovered from integer timestamps.",
            "total_including_warmup": total_credits, "cohort_credits": sum(v["count"] for v in cohort_credits),
            "first_cohort_credit": cohort_credits[0], "last_cohort_credit": last_credit,
            "credit_bin_span_s": credit_times[-1] - credit_times[0],
            "credit_bin_gaps_s": summary([b - a for a, b in zip(credit_times, credit_times[1:])]),
            "windows": cohort_credits,
        },
        "drain_after_last_first_output_credit": {
            "start": last_credit["time"], "start_line": last_credit["line"],
            "end": drained[1], "end_line": drained[0],
            "sampled_span_s": seconds(drained[1]) - seconds(last_credit["time"]),
            "tail_engine_lines": [v[0] for v in tail],
            "tail_spec_lines": [v[0] for v in tail_spec],
            "generation_credit_integral_approx": sum(v[3] * 10 for v in tail),
            "accepted": tail_a, "drafted": tail_d,
            "conditional_fixed_three_request_events": tail_d / 3,
            "accepted_plus_conditional_bonus_tokens": tail_a + tail_d / 3,
            "device_pure_decode_certified": False,
            "limit": "No further first-output credits and waiting drains, but metrics omit scheduler preemption/recompute events and device timestamps. A no-prefill tail requires that extra no-recompute assumption.",
        },
        "draft_counter_identifiability": {
            "known_integer_windows": len(known),
            "known_accepted": sum(row[5] for row in known),
            "known_drafted": sum(row[6] for row in known),
            "request_draft_events_lower_bound": sum(row.get("minimum", 0) for row in candidates),
            "finite_windows_exact_count": sum(row["status"] == "unique_request_draft_event_count" for row in candidates),
            "finite_windows_ambiguous_count": sum(row["status"] == "bounded_request_draft_event_count" for row in candidates),
            "finite_windows_event_interval": [sum(v["minimum"] for v in finite), sum(v["maximum"] for v in finite)],
            "unknown_windows": [v for v in candidates if v["status"] in {"missing_integer_counts", "no_finite_upper_bound"}],
            "unrecoverable_source_fragments": run.get("fragments", []),
            "all_logged_window_results": candidates,
            "scope": "Request-level draft/verification events, including warmup. Not global model iterations, kernel counts or device duration. Clipping occurs after native observe_draft, so A+Q need not equal emitted tokens.",
        },
        "high_credit_windows_not_device_benchmarks": bursts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=ROOT / "reports/evidence_64k_120_20261009.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/latency_budget_64k_20261010.json")
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text())
    raw = ROOT / evidence["source"]["path"]
    if hashlib.sha256(raw.read_bytes()).hexdigest() != evidence["source"]["sha256"]:
        raise ValueError("raw evidence SHA256 changed")
    native = evidence["runs"]["native"]["client"]["benchmark_duration_s"]
    current = evidence["runs"]["oscar"]["client"]["benchmark_duration_s"]
    budget = {}
    for factor in (2, 3):
        target = native / factor
        budget[str(factor)] = {
            "target_wall_seconds": target,
            "current_oscar_required_speedup": current / target,
            "maximum_mean_client_E2E_s_at_K32": 32 * target / 120,
            "required_input_tps_over_completed_batch": 120 * 65100 / target,
            "required_output_tps_over_completed_batch": 120 * 512 / target,
            "all_native_chunks_prefill_P_upper_bound_if_decode_free_s": target / 120,
            "serialized_budget_examples_not_a_fit": [
                {"assumed_pure_decode_aggregate_tps": d,
                 "decode_seconds_for_61440_tokens": 61440 / d,
                 "remaining_prefill_seconds_per_request_all_native_chunks": (target - 61440 / d) / 120}
                for d in (300, 400, 500, 1000)
            ],
            "required_decode_tps_if_all_chunks_prefill_P_s": {
                str(p): (61440 / (target - 120 * p) if target > 120 * p else None)
                for p in (1, 1.5, 2, 2.5, 3)
            },
        }
    pair_count = 65100 * 65101 // 2
    moved_pairs = 16384 * 16384 + 32768 * 16384 + 49152 * 15948
    result = {
        "schema_version": 2, "kind": "offline_evidence_and_required_budget_not_performance_simulation",
        "source": evidence["source"],
        "target_basis": "User 2026-10-10 accepts 2-3x faster than measured native 1342.4437259 seconds; universal routes and omitted max-num-batched-tokens remain required.",
        "serving_token_budget": "native_default_observed_at_runtime_not_overridden",
        "budgets": budget,
        "runs": {name: analyze_run(run) for name, run in evidence["runs"].items()},
        "static_work_conditional_on_Q65100_Hq6_Hkv1_D256_16targetFULL_1draftFULL": {
            "full_prompt_causal_pairs_per_head": pair_count,
            "unchanged_attention_QK_PV_math_FLOPs_per_prompt_rank_17_FULL_layers": 4 * pair_count * 256 * 6 * 17,
            "target_only_16_FULL_attention_QK_PV_math_FLOPs_per_prompt_rank": 4 * pair_count * 256 * 6 * 16,
            "additional_one_draft_FULL_math_if_it_has_the_same_Q65100_shape": 4 * pair_count * 256 * 6,
            "unchanged_GDN_layer_token_positions_for_120_prompts": 48 * 65100 * 120,
            "unchanged_full_model_layer_token_positions_for_120_prompts_including_one_draft": 65 * 65100 * 120,
            "unchanged_full_INT2_store_token_heads_for_120_prompts_per_rank": 17 * 65100 * 120,
            "limit": "Ideal source shapes, not reconstructed actual per-step traces. GM bytes are source copy volume, not measured HBM transactions or saved wall time. Model dimensions remain conditional absent the target config snapshot.",
        },
        "retired_whole_prefill_historical_accounting": {
            "allowed_in_current_speedup_projection": False,
            "reason": "User requires native scheduling and no explicit token budget; no whole-prefill admission override.",
            "prior_cross_history_pairs_before_window_correction": moved_pairs,
            "prior_fraction_of_pairs_moved_to_native_current_FIA": moved_pairs / pair_count,
            "prior_removed_KV256_units_per_FULL_layer_rank": 28023,
        },
        "unidentified": ["Sum of native-chunk target+first-MTP prefill service time, and current/GDN/MLP/communication breakdown.",
                         "Sustained pure q4 plus eager three-MTP decode cycle time at K32/context65100.",
                         "Global verification batch sequence and per-step live request counts.",
                         "Preemption/recompute events and exact no-prefill device intervals.",
                         "Shared model-compute wall-time lower bound: positive source work exists, but no phase timing/achieved hardware rate is supplied."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "targets": budget,
                      "counter_summary": {k: {x: y for x, y in v["draft_counter_identifiability"].items()
                                                if x != "all_logged_window_results"}
                                          for k, v in result["runs"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
