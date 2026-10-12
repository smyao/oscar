"""Integer-event admission replay; archive #137/#148/#152/#155.

This is a synthetic scheduler-work comparison, not an NPU timing simulator or
the user's LongBenchv2 request lengths.  It pins the native RUNNING-first,
WAITING-greedy token-budget order and the Mamba align split shape, including
the extra Eagle/MTP pruned block (scheduler.py:293-334,366-445,566-810;
speculative.py:1070).  It omits KV allocation, prefix reuse, preemption,
host arrival times and variable accepted speculative lengths.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path


MAX_MODEL = 262144
CLIENT_CONCURRENCY = 32
SERVER_MAX_SEQUENCES = 128
SPEC_TOKENS = 3
ROUND_TOTAL = 33005  # candidate log: Drafted 99,015 / three drafts
DECODE_Q = SPEC_TOKENS + 1
WHOLE_LIMIT = 32768 - SERVER_MAX_SEQUENCES * DECODE_Q  # 32,256
RECENT_AND_SINK = 256 + 64


def decode_routes(work):
    """Request-round exposure to real selectors; this is not wall-time data."""
    from oscar_ascend.ops.cv_dispatch import select_cv_op,ATTENTION_QUERY_ROWS
    from oscar_ascend.integration.current_only_plan import plan_current_suffix
    # Native GDN speculative metadata sets threshold4; the native runner
    # already puts these short-query requests before longer prefill requests.
    rows=sorted(work,key=lambda row:row['qlen']>DECODE_Q)
    def select(group,optimized):
        return select_cv_op(16,6,1,sum(r['qlen'] for r in group),max(r['qlen'] for r in group),
            fast_unpack=True,mixed_cv=True,striped_cache=True,decode_bundle=optimized)
    old=Counter();new=Counter()
    old[select(rows,False)]+=sum(r['decode'] for r in rows)
    starts=[0]
    for row in rows:starts.append(starts[-1]+row['qlen'])
    fresh=plan_current_suffix(starts,[r['context']+r['qlen'] for r in rows],
                              actual_tokens=starts[-1],padded_tokens=starts[-1])
    remaining=rows if fresh is None else rows[:fresh.prefix_requests]
    if remaining:
        split=0
        while split<len(remaining) and remaining[split]['qlen']<=DECODE_Q:split+=1
        threshold=4*(ATTENTION_QUERY_ROWS//6)
        partition=(0<split<len(remaining) and sum(r['qlen'] for r in remaining[:split])<=128
                   and max(r['qlen'] for r in remaining[split:])>=threshold)
        groups=(remaining[:split],remaining[split:]) if partition else (remaining,)
        for group in groups:new[select(group,True)]+=sum(r['decode'] for r in group)
    old=+old;new=+new
    assert sum(old.values())==sum(new.values())==sum(r['decode'] for r in rows)
    return old,new


@dataclass
class Request:
    ident: int
    prompt: int
    rounds_left: int
    prefilled: int = 0
    prefill_chunks: int = 0


def distribute_rounds(count: int, total: int, pattern: str) -> list[int]:
    if pattern == "uniform":
        per, remainder = divmod(total, count)
        rounds = [per + (i < remainder) for i in range(count)]
    elif pattern == "staggered":
        # An arbitrary fixed permutation of 159..359-ish rounds, centered on
        # the same total.  It is sensitivity only, never a dataset claim.
        rounds = [total // count + ((i * 37) % 201) - 100 for i in range(count)]
        difference = total - sum(rounds)
        step = 1 if difference > 0 else -1
        for i in range(abs(difference)):
            rounds[i % count] += step
    else:
        raise ValueError(f"unknown synthetic round pattern {pattern}")
    assert min(rounds) > 0 and sum(rounds) == total
    return rounds


def align_split(prompt: int, already_computed: int, proposed: int, block: int | None) -> int:
    """Native _mamba_block_aligned_split for a prompt-only chunk, use_eagle=True."""
    if block is None:
        return proposed
    if block <= 0:
        raise ValueError("block must be positive")
    if already_computed >= prompt:
        return proposed  # ordinary decode
    last_cache_position = max((prompt // block) * block - block, 0)
    after = already_computed + proposed
    if after < last_cache_position:
        return (proposed // block) * block
    if already_computed < last_cache_position < after:
        return last_cache_position - already_computed
    return proposed


def simulate(lengths: list[int], rounds: list[int], *, policy: str,
             align_block: int | None, client_concurrency: int = CLIENT_CONCURRENCY) -> dict:
    if policy not in ("old_greedy_16k", "budget_only_32k", "whole_32k"):
        raise ValueError(policy)
    if len(lengths) != len(rounds) or not lengths or not 1 <= client_concurrency <= len(lengths):
        raise ValueError("invalid synthetic request matrix")
    if any(length <= 0 or length + DECODE_Q * count >= MAX_MODEL for length, count in zip(lengths, rounds)):
        raise ValueError("synthetic request exceeds max_model_len with q4 rounds")
    budget_limit = 16384 if policy == "old_greedy_16k" else 32768
    requests = [Request(i, length, count) for i, (length, count) in enumerate(zip(lengths, rounds))]
    waiting: deque[Request] = deque(requests[:client_concurrency])
    running: list[Request] = []
    next_to_submit = client_concurrency
    finished = 0
    metrics = {
        "policy": policy, "align_block": align_block, "client_concurrency": client_concurrency,
        "budget": budget_limit, "whole_limit": WHOLE_LIMIT if policy == "whole_32k" else None,
        "requests": len(requests), "declared_prompt_tokens": sum(lengths),
        "declared_request_rounds": sum(rounds), "max_model_len": MAX_MODEL,
        "steps": 0, "steps_with_prefill": 0, "steps_with_decode": 0,
        "steps_with_prefill_and_decode": 0, "peak_scheduled_tokens_N": 0,
        "peak_running": 0, "peak_waiting": len(waiting),
        "prefill_chunk_events": 0, "partial_prefill_events": 0,
        "historical_query_prior_key_pairs": 0,
        "prefill_current_causal_pairs": 0,
        "fresh_prefill_current_causal_pairs": 0,
        "mature_history_query_prior_key_pairs_approx": 0,
        "requests_with_historical_prefill": 0,
        "total_scheduled_prompt_tokens": 0, "total_scheduled_decode_q4_tokens": 0,
        "total_scheduled_tokens": 0, "late_two_draft_input_token_work": 0,
    }
    historical_request_ids = set()
    routes_before=Counter();routes_after=Counter()
    step_work=[]

    def prefill(req: Request, amount: int) -> None:
        prefix = req.prefilled
        step_work.append({'qlen':amount,'context':prefix,'decode':False})
        req.prefilled += amount
        req.prefill_chunks += 1
        metrics["prefill_chunk_events"] += 1
        metrics["partial_prefill_events"] += req.prefilled < req.prompt
        metrics["total_scheduled_prompt_tokens"] += amount
        metrics["historical_query_prior_key_pairs"] += amount * prefix
        # Per FULL layer, splitting moves query/key pairs between history
        # and current; it does not delete the causal attention mathematics.
        # First MTP still executes current via FP32 CV without the new route.
        current_pairs = amount * (amount + 1) // 2
        metrics["prefill_current_causal_pairs"] += current_pairs
        if prefix == 0:
            metrics["fresh_prefill_current_causal_pairs"] += current_pairs
        metrics["mature_history_query_prior_key_pairs_approx"] += amount * max(0, prefix - RECENT_AND_SINK)
        if prefix:
            historical_request_ids.add(req.ident)

    while finished < len(requests):
        metrics["steps"] += 1
        if metrics["steps"] > sum(rounds) + sum(lengths) // 512 + len(lengths) * 3:
            raise RuntimeError("synthetic scheduling did not make progress")
        available = budget_limit
        step_prompt = 0
        step_decode = 0
        step_work=[]
        metrics["peak_running"] = max(metrics["peak_running"], len(running))
        metrics["peak_waiting"] = max(metrics["peak_waiting"], len(waiting))

        # Pinned native order: process RUNNING requests before WAITING.
        for req in running:
            if available <= 0:
                break
            if req.prefilled < req.prompt:
                proposed = min(req.prompt - req.prefilled, available)
                if policy == "whole_32k":
                    proposed = min(proposed, 16384)  # RUNNING long-prefill threshold
                proposed = align_split(req.prompt, req.prefilled, proposed, align_block)
                if proposed:
                    prefill(req, proposed)
                    step_prompt += proposed
                    available -= proposed
            elif req.rounds_left and available >= DECODE_Q:
                step_work.append({'qlen':DECODE_Q,'context':req.prompt,'decode':True})
                req.rounds_left -= 1
                step_decode += DECODE_Q
                available -= DECODE_Q

        while waiting and len(running) < client_concurrency and available > 0:
            req = waiting[0]
            proposed = req.prompt
            if policy == "whole_32k":
                if req.prompt <= WHOLE_LIMIT:
                    if proposed > available:
                        break  # native WAITING queue retains order; no partial
                else:
                    proposed = min(proposed, 16384)
            proposed = min(proposed, available)
            proposed = align_split(req.prompt, 0, proposed, align_block)
            if proposed <= 0:
                break
            waiting.popleft()
            running.append(req)
            prefill(req, proposed)
            step_prompt += proposed
            available -= proposed

        step_N = step_prompt + step_decode
        if step_N <= 0:
            raise RuntimeError("synthetic scheduler deadlocked without KV pressure")
        metrics["steps_with_prefill"] += step_prompt > 0
        metrics["steps_with_decode"] += step_decode > 0
        metrics["steps_with_prefill_and_decode"] += step_prompt > 0 and step_decode > 0
        metrics["peak_scheduled_tokens_N"] = max(metrics["peak_scheduled_tokens_N"], step_N)
        metrics["total_scheduled_decode_q4_tokens"] += step_decode
        metrics["total_scheduled_tokens"] += step_N
        metrics["late_two_draft_input_token_work"] += 2 * step_N
        before,after=decode_routes(step_work)
        routes_before.update(before);routes_after.update(after)

        survivors = []
        for req in running:
            if req.prefilled == req.prompt and req.rounds_left == 0:
                finished += 1
            else:
                survivors.append(req)
        running = survivors
        while next_to_submit < len(requests) and len(waiting) + len(running) < client_concurrency:
            waiting.append(requests[next_to_submit])
            next_to_submit += 1

    metrics["requests_with_historical_prefill"] = len(historical_request_ids)
    metrics['decode_request_rounds_by_original_route']=dict(routes_before)
    metrics['decode_request_rounds_with_proposed_partition']=dict(routes_after)
    metrics['route_scope']='synthetic request-round exposure to current selectors, main FULL attention; not kernel counts, device occupancy or seconds'
    assert sum(routes_before.values())==sum(routes_after.values())==sum(rounds)
    metrics["max_prefill_chunks_per_request"] = max(req.prefill_chunks for req in requests)
    metrics["mean_prefill_chunks_per_request"] = metrics["prefill_chunk_events"] / len(requests)
    metrics["prefill_causal_pairs_per_full_layer"] = sum(n * (n + 1) // 2 for n in lengths)
    metrics["remaining_first_draft_current_fp32_pairs_if_fresh_fia"] = (
        metrics["prefill_current_causal_pairs"] - metrics["fresh_prefill_current_causal_pairs"])
    metrics["first_draft_pair_work_scope"] = (
        "Logical causal pairs only, conditional on first MTP executing for each prefill chunk. "
        "Synthetic no-prefix-cache schedule makes new requests a contiguous fresh suffix. "
        "Not measured proposer calls, FP32 FLOPs or critical-path time; history and current cannot both claim the same removed pairs.")
    assert (metrics["historical_query_prior_key_pairs"] + metrics["prefill_current_causal_pairs"]
            == metrics["prefill_causal_pairs_per_full_layer"])
    assert metrics["total_scheduled_prompt_tokens"] == sum(lengths)
    assert metrics["total_scheduled_decode_q4_tokens"] == DECODE_Q * sum(rounds)
    assert metrics["total_scheduled_tokens"] == sum(lengths) + DECODE_Q * sum(rounds)
    assert metrics["late_two_draft_input_token_work"] == 2 * metrics["total_scheduled_tokens"]
    assert metrics["peak_scheduled_tokens_N"] <= budget_limit
    return metrics


def scenarios() -> dict[str, list[int]]:
    interleaved_mean_20k = []
    short_seen = 0
    for i in range(127):
        if i % 4 == 0:
            interleaved_mean_20k.append(40000)
        else:
            interleaved_mean_20k.append(13278 if short_seen == 94 else 13263)
            short_seen += 1
    assert sum(interleaved_mean_20k) == 127 * 20000
    return {
        "declared_20k_each_synthetic": [20000] * 127,
        "cyclic_20_23_27_30k_synthetic": [[20000, 23000, 27000, 30000][i % 4] for i in range(127)],
        # Same total and mean as 127*20K, but 32 requests exceed whole_limit.
        "same_mean_20k_with_32x40k_synthetic": interleaved_mean_20k,
    }


def run_matrix() -> dict:
    result = {
        "scope": "synthetic integer-event work replay only; no real request-length distribution, NPU timing, cache capacity, accuracy or graph evidence",
        "source": "native scheduler RUNNING-first / WAITING-greedy plus experimental AdmissionView; speculative.use_eagle() true for MTP",
        "client_concurrency": CLIENT_CONCURRENCY,
        "server_max_sequences_for_whole_limit": SERVER_MAX_SEQUENCES,
        "fixed_candidate_request_rounds": ROUND_TOTAL,
        "q4_tokens_per_request_round": DECODE_Q,
        "max_model_len": MAX_MODEL,
        "matrix": {},
    }
    for scenario_name, lengths in scenarios().items():
        result["matrix"][scenario_name] = {}
        for pattern in ("uniform", "staggered"):
            rounds = distribute_rounds(len(lengths), ROUND_TOTAL, pattern)
            case = {}
            for block in (None, 512, 1024):
                key = "none" if block is None else f"align_{block}"
                arms = {policy: simulate(lengths, rounds, policy=policy, align_block=block)
                        for policy in ("old_greedy_16k", "budget_only_32k", "whole_32k")}
                baseline_pairs = arms["old_greedy_16k"]["historical_query_prior_key_pairs"]
                for arm in arms.values():
                    arm["history_pair_reduction_vs_old16k"] = 1 - arm["historical_query_prior_key_pairs"] / baseline_pairs
                case[key] = arms
            result["matrix"][scenario_name][pattern] = case
    # Isolated long-request controls exercise >32,256 fallback and max-model
    # bounds; they are separate from the 127-request / 33,005-round matrix.
    result["long_request_controls"] = {}
    for prompt in (65000, 261000):
        result["long_request_controls"][str(prompt)] = {
            "none" if block is None else f"align_{block}": {
                policy: simulate([prompt], [16], policy=policy, align_block=block, client_concurrency=1)
                for policy in ("old_greedy_16k", "budget_only_32k", "whole_32k")
            } for block in (None, 512, 1024)
        }
    return result


def add_conditional_budget(replay: dict, whole_model: dict,
                           eligible_history_fraction: float | None = None) -> None:
    """Hold the prior coarse model fixed; vary only an explicit history share.

    `p` below is a logical Q*prior-K pair reduction. Converting it to time is
    a hypothetical linearity assumption, not device evidence.
    """
    current = whole_model["anchor_seconds_user_reported"]["candidate"]
    target = whole_model["anchor_seconds_user_reported"]["native"]
    residual = whole_model["coarse_window_replay"]["candidate"]["prompt_correlated_residual_anchor_seconds"]
    required_saving = current - target
    prior = 70.0  # older 835 log rough rotation/store/merge envelope; not this version
    prior_sensitivity = (35.0, 140.0)  # illustrative factor-two, not a CI
    if eligible_history_fraction is not None and not 0 <= eligible_history_fraction <= 1:
        raise ValueError("eligible history fraction must be in [0,1]")
    replay["conditional_budget"] = {
        "fixed_user_anchor_seconds": {"candidate": current, "native": target},
        "fixed_prompt_accounting_residual_seconds": residual,
        "required_saving_seconds": required_saving,
        "formula": "T_hypothetical = 840 - residual * f_eligible_main_history * p_logical_pair_reduction",
        "f_definition": "fraction of 605.2s arithmetic residual attributable specifically to main-prefill history of requests eligible for the whole policy; includes eligibility share and excludes first-MTP-draft CV",
        "p_definition": "simulated logical prior-history Q*K pair reduction relative to old greedy16k; time proportionality is unverified",
        "older_rotation_store_merge_seconds_prior": prior,
        "older_prior_sensitivity_seconds_not_confidence_interval": list(prior_sensitivity),
        "nonhistory_prior_warning": "Older 835 phase data from a different revision; 70s is neither an immutable floor nor current-run device measurement. GDN, first MTP draft, dense MTP and host are additional unknown residuals.",
        "prediction_status": "not_supported_without_eligible_history_time_fraction_and_NPU_phase_data",
        "user_supplied_eligible_history_fraction": eligible_history_fraction,
    }
    for scenario in replay["matrix"].values():
        for modes in scenario.values():
            for arms in modes.values():
                for arm in arms.values():
                    reduction = arm["history_pair_reduction_vs_old16k"]
                    if reduction <= 0:
                        arm["conditional_budget"] = {"required_f": None, "reason": "no logical history reduction"}
                        continue
                    needed_history_seconds = required_saving / reduction
                    max_nonhistory = residual - needed_history_seconds
                    arm["conditional_budget"] = {
                        "p_logical_pair_reduction": reduction,
                        "required_f": needed_history_seconds / residual,
                        "required_eligible_main_history_seconds": needed_history_seconds,
                        "maximum_other_residual_seconds": max_nonhistory,
                        "if_older_70s_prior_as_nonhistory_and_all_other_eligible_seconds":
                            current - max(0, residual - prior) * reduction,
                        "would_70s_prior_leave_enough_arithmetic_headroom": max_nonhistory >= prior,
                        "older_prior_sensitivity_conditional_seconds": [
                            current - max(0, residual - value) * reduction for value in prior_sensitivity
                        ],
                        "user_supplied_fraction_conditional_seconds": (
                            current - residual * eligible_history_fraction * reduction
                            if eligible_history_fraction is not None else None
                        ),
                        "status": "conditional_only_not_NPU_prediction",
                    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/admission_replay_20260930.json"))
    parser.add_argument("--whole-model", type=Path, default=Path("reports/whole_run_model_20260930.json"))
    parser.add_argument("--eligible-history-fraction", type=float,
                        help="Hypothetical f in the fixed 605s residual; not inferred from logs")
    args = parser.parse_args()
    result = run_matrix()
    add_conditional_budget(result, json.loads(args.whole_model.read_text()), args.eligible_history_fraction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"ADMISSION_REPLAY cases={sum(len(x)*len(next(iter(x.values())))*3 for x in result['matrix'].values())} output={args.output}")


if __name__ == "__main__":
    main()
