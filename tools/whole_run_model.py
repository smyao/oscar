"""Bounded whole-run counterfactual from service samples; archive #137/#152/#155.

Two wall-clock anchors do not identify an attention-time fraction.  This model
keeps the work counters and unknown coefficients explicit; changing a scenario
never refits coefficients to make a target pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


def required_fraction(current_seconds: float, target_seconds: float, speedup: float) -> float | None:
    if speedup <= 1:
        return None
    return (1 - target_seconds / current_seconds) / (1 - 1 / speedup)


def counterfactual(current_seconds: float, affected_fraction: float, speedup: float) -> float:
    if not 0 <= affected_fraction <= 1 or speedup <= 0:
        raise ValueError("affected_fraction must be in [0,1] and speedup positive")
    return current_seconds * (1 - affected_fraction * (1 - 1 / speedup))


def previous_projection_backtest(previous_seconds: float, observed_seconds: float,
                                 local_simulator_speedup: float) -> dict:
    """Falsify the previous whole-run extrapolation, without fitting a cause.

    The effective share is conditional on the same operator speedup holding
    on NPU, equal workload and no other change. It is not an attention profile
    and cannot be transferred to the next optimization.
    """
    if previous_seconds <= 0 or observed_seconds <= 0 or local_simulator_speedup <= 1:
        raise ValueError("backtest requires positive times and a local speedup > 1")
    naive = previous_seconds / local_simulator_speedup
    return {
        "previous_user_wall_seconds": previous_seconds,
        "observed_user_wall_seconds": observed_seconds,
        "local_simulator_speedup": local_simulator_speedup,
        "observed_whole_run_speedup": previous_seconds / observed_seconds,
        "rejected_naive_whole_run_prediction_seconds": naive,
        "naive_prediction_error_seconds": observed_seconds - naive,
        "conditional_effective_affected_share":
            (1 - observed_seconds / previous_seconds) / (1 - 1 / local_simulator_speedup),
        "conditional_share_is_measured": False,
        "transfer_share_to_next_candidate": False,
        "interpretation": "Previous 16-to-14 minute feedback disproves treating the local simulator ratio as the whole-run ratio. This inversion cannot distinguish limited coverage, simulator/device mismatch, workload or other cost changes.",
    }


def _prompt_off_step_samples(trace: dict) -> list[dict]:
    """High-volume, nearly steady prompt-off windows; endpoint-based proxy.

    Waiting can still be positive, so these are not guaranteed pure decode.
    """
    samples = trace["samples"]
    selected = []
    for i in range(1, len(samples) - 1):
        sample = samples[i]
        if (sample["prompt_tps"] != 0 or sample["generation_tps"] < 150
                or sample["running"] < 20 or not sample.get("spec")):
            continue
        if abs(samples[i - 1]["running"] - sample["running"]) > 2:
            continue
        if abs(samples[i + 1]["running"] - sample["running"]) > 2:
            continue
        rounds = sample["spec"]["drafted"] / 3
        steps_proxy = rounds / sample["running"]
        selected.append({
            "line": sample["line"], "time": sample["time"],
            "running_endpoint": sample["running"], "waiting_endpoint": sample["waiting"],
            "generation_tps": sample["generation_tps"], "verification_rounds": rounds,
            "engine_steps_proxy": steps_proxy,
            "seconds_per_engine_step_proxy": 10 / steps_proxy,
        })
    return selected


def _replay(trace: dict, anchor_seconds: float) -> dict:
    """Fit only a prompt-accounting-correlated residual after a step proxy.

    Prompt accounting is delayed to first output, so the per-window fit is not
    a physical CPU/GPU timing breakdown or a prefill execution timeline.
    """
    pure = _prompt_off_step_samples(trace)
    if not pure:
        return {"status": "no_clean_prompt_off_reference"}
    b_log = min(x["seconds_per_engine_step_proxy"] for x in pure)
    samples = trace["samples"]
    active_seconds = trace["groups"]["active"]["nominal_sample_seconds"]
    anchor_scale = anchor_seconds / active_seconds
    p_chunks = trace["groups"]["all"]["prompt_tokens_rate_integral_approx"] / 16384
    rows = []
    for i, sample in enumerate(samples):
        if not sample.get("spec"):
            continue
        prev_running = samples[i - 1]["running"] if i else sample["running"]
        occupancy_proxy = max(sample["running"], prev_running, 1)
        step_proxy = (sample["spec"]["drafted"] / 3) / occupancy_proxy
        prompt_chunks = sample["prompt_tps"] * 10 / 16384
        rows.append({
            "line": sample["line"], "time": sample["time"], "running_endpoint": sample["running"],
            "previous_running_endpoint": prev_running, "prompt_chunks": prompt_chunks,
            "engine_steps_proxy": step_proxy,
            "unexplained_seconds_before_prompt_fit_log_clock": 10 - b_log * step_proxy,
        })
    step_total = sum(x["engine_steps_proxy"] for x in rows)
    b_anchor = b_log * anchor_scale
    step_anchor_seconds = b_anchor * step_total
    prompt_residual_seconds = anchor_seconds - step_anchor_seconds
    a_anchor = prompt_residual_seconds / p_chunks
    negative_mixed = [
        {"line": x["line"], "time": x["time"],
         "negative_seconds": round(x["unexplained_seconds_before_prompt_fit_log_clock"], 4)}
        for x in rows if x["prompt_chunks"] > 0 and x["unexplained_seconds_before_prompt_fit_log_clock"] < 0
    ]
    residual_errors = [
        10 - b_log * x["engine_steps_proxy"] - (a_anchor / anchor_scale) * x["prompt_chunks"]
        for x in rows if x["prompt_chunks"] > 0
    ]
    return {
        "status": "aggregate_arithmetic_calibrated_only_window_timing_invalid",
        "clock": "10s sampled-log rates rescaled to user-reported wall anchor; not physical event time",
        "active_nominal_seconds": active_seconds,
        "anchor_scale_from_log_to_user_wall": anchor_scale,
        "prompt_off_high_generation_reference_windows": pure,
        "step_cost_log_clock_seconds_range": [
            min(x["seconds_per_engine_step_proxy"] for x in pure),
            max(x["seconds_per_engine_step_proxy"] for x in pure),
        ],
        "step_cost_selected_fastest_log_clock_seconds": b_log,
        "step_cost_user_anchor_clock_seconds": b_anchor,
        "step_proxy_total": step_total,
        "prompt_16k_chunk_proxy_total": p_chunks,
        "step_proxy_anchor_seconds": step_anchor_seconds,
        "prompt_correlated_residual_anchor_seconds": prompt_residual_seconds,
        "prompt_correlated_residual_seconds_per_16k_chunk": a_anchor,
        "negative_mixed_windows_before_prompt_fit": negative_mixed,
        "mixed_window_fit_abs_error_median_log_seconds": median(abs(x) for x in residual_errors),
        "mixed_window_fit_abs_error_max_log_seconds": max(abs(x) for x in residual_errors),
        "interpretation": "Prompt counter is reported at first output rather than when chunks execute. This residual includes delayed accounting, prefill, overlap, MTP/model/host, scheduling and endpoint error; it is neither CV duration nor a true per-window fit.",
    }


def _shared_two_anchor_fit(candidate: dict, native: dict, current_seconds: float, target_seconds: float) -> dict:
    c = candidate["groups"]["all"]
    n = native["groups"]["all"]
    pc = c["prompt_tokens_rate_integral_approx"] / 16384
    pn = n["prompt_tokens_rate_integral_approx"] / 16384
    vc = c["mtp_verification_rounds_inferred"]
    vn = n["mtp_verification_rounds_inferred"]
    determinant = pc * vn - pn * vc
    if abs(determinant) < 1e-12:
        return {"status": "singular"}
    a = (current_seconds * vn - target_seconds * vc) / determinant
    b = (target_seconds * pc - current_seconds * pn) / determinant
    return {
        "equation": "T = alpha*(reported_local_prompt_counter/16384) + beta*(drafted/3)",
        "alpha_seconds_per_16k_prompt": a,
        "beta_seconds_per_verification_round": b,
        "status": "rejected_negative_coefficient" if a < 0 or b < 0 else "nonnegative_solution",
        "meaning": "A shared additive reported-P+MTP-round model must fit both user wall anchors with nonnegative coefficients; rejection requires variant-specific costs, uncounted work, or incomparable timing boundaries.",
    }


def model(candidate: dict, native: dict, current_seconds: float = 840, target_seconds: float = 480,
          hypothetical_cv_speedup: float = 2.0954974660993666,
          affected_fraction: float | None = None) -> dict:
    """Workload and speedup bounds; no per-op attribution is inferred."""
    c = candidate["groups"]["all"]
    n = native["groups"]["all"]
    ratio = {
        "prompt_tokens": c["prompt_tokens_rate_integral_approx"] / n["prompt_tokens_rate_integral_approx"],
        "generated_tokens": c["generation_tokens_rate_integral_approx"] / n["generation_tokens_rate_integral_approx"],
        "mtp_verification_rounds": c["mtp_verification_rounds_inferred"] / n["mtp_verification_rounds_inferred"],
    }
    # Conditional nonnegative additive surrogate: native time shares for
    # fixed, prompt, output, and MTP-round work sum to one.  With two anchors,
    # every point in the simplex remains admissible; no share is estimated.
    simplex_ratios = {"fixed": 1.0, **ratio}
    common_low = target_seconds * min(simplex_ratios.values())
    common_high = target_seconds * max(simplex_ratios.values())
    excess_low = current_seconds - common_high
    excess_high = current_seconds - common_low
    req = required_fraction(current_seconds, target_seconds, hypothetical_cv_speedup)
    active_samples = candidate["groups"]["active"]["samples"]
    prompt_samples = candidate["groups"]["prompt_visible"]["samples"]
    prompt_reported_fraction = prompt_samples / active_samples
    candidate_replay = _replay(candidate, current_seconds)
    native_replay = _replay(native, target_seconds)
    step_seconds = candidate_replay.get("step_proxy_anchor_seconds")
    prompt_seconds = candidate_replay.get("prompt_correlated_residual_anchor_seconds")
    if step_seconds is not None and prompt_seconds is not None and target_seconds > step_seconds:
        residual_target = {
            "step_proxy_held_fixed_seconds": step_seconds,
            "prompt_correlated_residual_current_seconds": prompt_seconds,
            "prompt_correlated_residual_budget_for_480_seconds": target_seconds - step_seconds,
            "prompt_correlated_residual_required_speedup_if_wholly_eligible": prompt_seconds / (target_seconds - step_seconds),
            "if_whole_prompt_residual_were_accelerated_by_simulator_ratio_seconds": step_seconds + prompt_seconds / hypothetical_cv_speedup,
            "required_fraction_of_prompt_residual_affected_at_simulator_ratio":
                (current_seconds - target_seconds) / (prompt_seconds * (1 - 1 / hypothetical_cv_speedup)),
            "valid_as_prefill_budget": False,
            "interpretation": "Only a conditional two-bucket arithmetic replay: reported prompt totals are delayed until first output; prompt residual is not a measured prefill/attention budget and simulator ratio is not NPU speedup.",
        }
    else:
        residual_target = {"status": "target_not_feasible_with_step_proxy_fixed"}
    sensitivity = [
        {"affected_fraction": f, "seconds": round(counterfactual(current_seconds, f, hypothetical_cv_speedup), 3)}
        for f in (0.25, 0.5, 0.75, 1.0)
    ]
    if affected_fraction is not None:
        sensitivity.append({"affected_fraction": affected_fraction,
                            "seconds": round(counterfactual(current_seconds, affected_fraction, hypothetical_cv_speedup), 3),
                            "user_supplied": True})
    return {
        "previous_projection_backtest": previous_projection_backtest(960, 840, 2.0954974660993666),
        "anchor_seconds_user_reported": {"candidate": current_seconds, "native": target_seconds},
        "anchor_difference_seconds": current_seconds - target_seconds,
        "anchor_ratio": current_seconds / target_seconds,
        "log_sample_timestamp_spans_seconds": {
            "candidate": candidate["spans"]["sample_timestamp_span_seconds"],
            "native": native["spans"]["sample_timestamp_span_seconds"],
        },
        "anchor_matches_sample_span": (
            current_seconds == candidate["spans"]["sample_timestamp_span_seconds"]
            and target_seconds == native["spans"]["sample_timestamp_span_seconds"]
        ),
        "reported_counter_ratios_candidate_div_native": ratio,
        "per_generated_token_mtp_rounds": {
            "candidate": c["mtp_verification_rounds_inferred"] / c["generation_tokens_rate_integral_approx"],
            "native": n["mtp_verification_rounds_inferred"] / n["generation_tokens_rate_integral_approx"],
        },
        "conditional_additive_simplex": {
            "meaning": "If both anchors were aligned and shared nonnegative per-counter coefficients, native 480s can have any fixed/reported-prompt/output/MTP-round share. Reported prompt tokens omit cached tokens and can miss repeated prefill work. Candidate-specific excess is a residual, not observed kernel time.",
            "native_time_share_coefficients": "w_fixed,w_prompt,w_output,w_mtp>=0; sum=1; unidentifiable",
            "common_candidate_seconds_range": [round(common_low, 3), round(common_high, 3)],
            "candidate_specific_residual_seconds_range": [round(excess_low, 3), round(excess_high, 3)],
            "if_only_residual_had_hypothetical_cv_speedup_seconds_range": [
                round(current_seconds - excess_high * (1 - 1 / hypothetical_cv_speedup), 3),
                round(current_seconds - excess_low * (1 - 1 / hypothetical_cv_speedup), 3),
            ],
            "valid_as_actual_component_attribution": False,
        },
        "hypothetical_full_cv_speedup_from_8k_B4_simulator": hypothetical_cv_speedup,
        "required_fraction_of_840s_critical_path_for_480s": req,
        "required_fraction_if_2x": required_fraction(current_seconds, target_seconds, 2),
        "required_fraction_if_1_5x": required_fraction(current_seconds, target_seconds, 1.5),
        "prompt_accounting_positive_sample_fraction_not_prefill_execution_bound": prompt_reported_fraction,
        "prompt_timing_caveat": "Prompt rate reports local-compute prompt tokens on first output. Prompt=0 may still execute unfinished prefill chunks; prompt>0 can account earlier chunk work. No prefill time eligibility bound follows from positive-window fraction.",
        "sensitivity_not_prediction": sensitivity,
        "source_proven_work_change": "striped reader removes packed Gather/plane/index preparation; exact per-run call count and NPU elapsed fraction are not in service logs",
        "source_proven_late_mtp_padding_opportunity": {
            "reference": "references/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py:1213,1253-1284,1558-1570",
            "formula_dense_tokens_avoidable_per_engine_step": "2*(num_input_tokens - batch_size) when num_input_tokens > batch_size",
            "whole_run_sum": None,
            "reason": "Engine logs lack per-step num_input_tokens and actual batch_size; drafted/3 counts request rounds, not engine step count.",
        },
        "shared_two_anchor_work_count_fit": _shared_two_anchor_fit(candidate, native, current_seconds, target_seconds),
        "coarse_window_replay": {"candidate": candidate_replay, "native": native_replay},
        "coarse_replay_target_sensitivity_not_prediction": residual_target,
        "prediction_status": "not_supported",
        "reason": "No NPU stripe speedup or whole-run affected critical-path fraction; anchors use different boundaries and reported counters differ. Prompt counters are delayed to first output and cannot time local prefill execution."
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("--current-seconds", type=float, default=840)
    parser.add_argument("--target-seconds", type=float, default=480)
    parser.add_argument("--affected-fraction", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = model(json.loads(args.candidate.read_text()), json.loads(args.native.read_text()),
                   args.current_seconds, args.target_seconds, affected_fraction=args.affected_fraction)
    output = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
