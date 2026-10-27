"""Archive #70-73/#129/#148/#150/#171: count work, never predict time.

D.4: fused bounded history/current FIA and online-state movement are counted
separately from quantization writes. No full-history restoration, accelerator
connection, timer, model or performance experiment is run. APC chunk records
come from the existing production-policy/pinned-native host replay. Their
remaining budget is explicit; they are not a BS32 service scheduling trace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from math import ceil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN_BUDGET = 16384
QUERY_ROWS = 128
KV_ROWS = 256
GROUP = 6
HEAD_DIM = 256
SINK = 64
RECENT = 256
SUM_FIELDS = (
    "new_tokens", "native_current_only_tokens", "warm_tokens",
    "history_qk_pv_tile_pairs", "history_unpack_tiles_after_reuse",
    "current_qk_pv_tile_pairs", "window_qk_pv_tile_pairs",
    "cluster_state_loop_gm_address_bytes",
)


def count_chunk(query_tokens: int, context_tokens: int) -> dict:
    """One request, Hq6/Hkv1/D256, S1, no holes/padding, fixed precision cut.

    A pair counts one QK and one PV for one query group and KV256 tile.
    GM bytes count Vector RestoreState+SpillState inside cluster loops only;
    exclude initialization/final restore, Cube atomic traffic and cache effects.
    Counts do not model per-request MTP acceptance or end-to-end latency.
    """
    if type(query_tokens) is not int or not 1 <= query_tokens <= TOKEN_BUDGET:
        raise ValueError("query_tokens must be an integer in 1..16384")
    if type(context_tokens) is not int or context_tokens < 0:
        raise ValueError("context_tokens must be a nonnegative integer")
    query_tile = QUERY_ROWS // GROUP
    groups = ceil(query_tokens / query_tile)
    # Every replayed chunk has >=1024 tokens, hence enough query groups for
    # the configured 20 Cubes to use S1. Reject silently mis-modeling q4/q1.
    if context_tokens and groups < 20:
        raise ValueError("this long-prefill counter requires the S1 long-query geometry")
    cluster_size = 16 if query_tokens >= 8192 else 4
    history_tiles = ceil(max(context_tokens - min(SINK, context_tokens), 0) / KV_ROWS)
    clusters = query_tokens // (cluster_size * query_tile) if history_tiles else 0
    clustered_groups = clusters * cluster_size
    unclustered_groups = groups - clustered_groups
    cut = max(min(SINK, context_tokens), context_tokens + 1 - RECENT)
    window_candidates = min(SINK, context_tokens) + context_tokens - cut
    current_tiles = sum(ceil(min(begin + query_tile, query_tokens) / KV_ROWS)
                        for begin in range(0, query_tokens, query_tile))
    fresh = context_tokens == 0
    state_bytes_per_group_tile = 2 * QUERY_ROWS * (HEAD_DIM + 2) * 4
    return {
        "context_tokens": context_tokens,
        "new_tokens": query_tokens,
        "end_tokens": context_tokens + query_tokens,
        "route": "native_rotated_current_only" if fresh else f"official_C{cluster_size}",
        "source_splits": 1,
        "query_groups": groups,
        "cluster_size": 0 if fresh else cluster_size,
        "history_clusters": clusters,
        "unclustered_history_groups": 0 if fresh else unclustered_groups,
        "history_tiles_per_group": history_tiles,
        "native_current_only_tokens": query_tokens if fresh else 0,
        "warm_tokens": 0 if fresh else query_tokens,
        "history_qk_pv_tile_pairs": groups * history_tiles,
        "history_unpack_tiles_after_reuse": (clusters + unclustered_groups) * history_tiles,
        "current_qk_pv_tile_pairs": 0 if fresh else current_tiles,
        "window_qk_pv_tile_pairs": 0 if fresh else groups * ceil(window_candidates / KV_ROWS),
        "cluster_state_loop_gm_address_bytes": clustered_groups * history_tiles * state_bytes_per_group_tile,
    }


def count_chunks(chunks: list[int], *, cached_prefix_tokens: int = 0) -> dict:
    if not chunks or type(cached_prefix_tokens) is not int or cached_prefix_tokens < 0:
        raise ValueError("nonempty chunks and explicit nonnegative cached prefix required")
    rows = []
    context = cached_prefix_tokens
    for length in chunks:
        row = count_chunk(length, context)
        rows.append(row)
        context += length
    return {
        "assumed_cached_prefix_tokens": cached_prefix_tokens,
        "prompt_tokens": context,
        "chunks": rows,
        "totals_per_request_per_full_layer_per_rank": {
            field: sum(row[field] for row in rows) for field in SUM_FIELDS
        },
    }


def cast_issue_counts(elements: int) -> dict:
    """Same 64-element RNE repeats, batched at the A2 uint8 repeat limit."""
    if type(elements) is not int or elements < 0:
        raise ValueError("elements must be a nonnegative integer")
    full, tail = divmod(elements, 64)
    return {"elements": elements, "old_single_repeat_instructions": ceil(elements / 64),
            "new_batched_repeat_instructions": ceil(full / 255) + bool(tail),
            "converted_elements": elements}


def build_report() -> dict:
    from tools.experiments.prefix_checkpoint_audit import run_case

    actual = {}
    for length in (32768, 65536):
        pair = {arm: run_case(prompt_tokens=length, mtp=True, arm=arm,
                              remaining_budgets=(TOKEN_BUDGET,))
                for arm in ("native", "oscar_fixed")}
        for row in pair.values():
            if row["status"] != "completed" or row["symbolic_state_identity_violations"]:
                raise RuntimeError("native/OSCAR host checkpoint replay did not complete cleanly")
        oscar = pair["oscar_fixed"]
        model = count_chunks([row["scheduled"] for row in oscar["chunks"]], cached_prefix_tokens=0)
        # This is conditional work after the synthetic repeated-prompt hit,
        # not an assumption about the user's traffic. This checkpoint occurs
        # exactly in the cold sequence; the splitter's subsequent input tuple
        # is identical and no intermediate state is manufactured.
        hit = oscar["repeat_prompt_hit_tokens"]
        suffix = [row["scheduled"] for row in oscar["chunks"] if row["before"] >= hit]
        if not suffix or sum(suffix) + hit != length:
            raise RuntimeError("joint-hit tail is not an exact existing replay suffix")
        actual[str(length)] = {
            "host_native_and_oscar_replays": pair,
            "explicit_cold_prefill_work_model": model,
            "conditional_retained_identical_prefix_tail": count_chunks(suffix, cached_prefix_tokens=hit),
        }
    sources = (
        "oscar_ascend/ops/cv_dispatch.py", "oscar_ascend/integration/impl.py",
        "oscar_ascend/integration/prefix_checkpoint.py", "oscar_ascend/runtime.py",
        "csrc/kernels/attention_official_tasks.cpp",
        "csrc/kernels/attention_cv_official_cluster16.cpp",
        "csrc/kernels/attention_cv_official_cluster4.cpp",
        "csrc/kernels/attention_cv_official_decode.cpp",
        "tools/experiments/prefix_checkpoint_audit.py", "tools/analyze_long_context_work.py",
    )
    return {
        "schema_version": 1,
        "evidence_kind": "source_work_counts_and_host_native_contract_replay_not_timing",
        "user_report": {"requests": 200, "client_batch_size": 32, "prefix_cache_enabled": True,
                        "token_budget": TOKEN_BUDGET, "32k_total_time_vs_native": "30% faster, user reported",
                        "64k_total_time_vs_native": "10% slower, user reported",
                        "actual_prefix_hit_tokens": None, "actual_phase_times": None,
                        "output_tokens_per_request": None},
        "scope": {"counts": "per request, per FULL layer, per TP rank; Hq6/Hkv1/D256",
                  "cold_samples": "declared zero-hit examples, not the observed run's hit rate",
                  "apc_replay": "APC on; isolated request receives full16384 remaining budget each step",
                  "bytes": "GM address bytes, not measured L2/HBM or elapsed time",
                  "not_modeled": ["BS32 arrival/admission/preemption trajectory", "MTP acceptance",
                                  "real prefix distribution/eviction", "GDN arithmetic", "device overlap/cache"]},
        "ideal_16k_cold_chunks": {
            str(length): count_chunks([TOKEN_BUDGET] * (length // TOKEN_BUDGET), cached_prefix_tokens=0)
            for length in (32768, 65536)
        },
        "actual_policy_host_replay": actual,
        "decode_publication_cast_issue_examples": [cast_issue_counts(x) for x in (1536, 4096, 16384)],
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(build_report(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
