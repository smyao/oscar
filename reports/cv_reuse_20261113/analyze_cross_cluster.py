#!/usr/bin/env python3
"""Offline source-bound accounting for a proposed single-first-KV256 cache.

Archive #126/#129/#145/#148-151/#154/#164/#172 and startup D.4 reviewed.
D.4(1): repeated bounded history unpack across existing C4/C16 query clusters.
D.4(2): old whole-history dequant 6499.8--6655.1 ms/card is rejected history,
not a measurement of this model or the current implementation.
D.4(3): this model changes no production source, task owner, source interval,
KV256 recurrence, arithmetic, error domain, graph state, or synchronization.
D.4(4): the proposed cache can occupy unused BF16 halves of existing slots;
there is no extra GM copy, but new predicates/state are not timed. This report
does not recommend promotion and does not predict latency/HBM bandwidth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "csrc/kernels/attention_cv_official_cluster4_active.cpp",
    "csrc/kernels/attention_cv_official_cluster16_active.cpp",
    "csrc/kernels/attention_official_tasks.cpp",
    "csrc/include/oscar_attention_launch.h",
    "csrc/include/oscar_bf16_cluster_launch.h",
    "oscar_ascend/ops/cv_dispatch.py",
    "configs/target.json",
)


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def domain(request, begin, head, split, context, lo, hi):
    # kind=0; qbegin/qcount are intentionally not the read-domain key. The
    # actual immutable pointer/geometry/dequantMode identity is launch-scoped.
    return (request, begin, head, split, context, lo, hi, 0)


def account(lengths, *, cluster, context=32768, splits=1, heads=2, cores=20):
    """Exact healthy contiguous-slot eligibility/owner model, GQA6/D256.

    Request offsets are global, anchor alignment is request-relative, buckets
    are global, and workId advances by cores exactly as Process pass two.
    First-pass ordinary history units are included in the denominator.
    """
    assert cluster in (4, 16) and all(n > 0 for n in lengths)
    assert context >= 0 and splits > 0 and heads > 0 and cores > 0
    qtile, kvrows, sink = 128 // 6, 256, 64
    width, tokens = cluster * qtile, sum(lengths)
    buckets = ceildiv(tokens, width)
    events = [[] for _ in range(cores)]
    ordinary_units = cluster_units = cluster_count = 0
    begin = 0
    for request, length in enumerate(lengths):
        for head in range(heads):
            first, last = min(sink, context), context
            split_width = ceildiv(last - first, splits)
            for split in range(splits):
                lo = min(last, first + split * split_width)
                hi = min(last, lo + split_width)
                tile_count = ceildiv(hi - lo, kvrows)
                grouped = set()
                # Exact IsClusterMember: complete cluster, nonempty source0,
                # identical bounds, and kvend == context. In split mode this
                # generally admits only the final nonempty split.
                if lo < hi and hi == context:
                    for relative in range(0, length - width + 1, width):
                        anchor = begin + relative
                        bucket = anchor // width
                        segment = head * splits + split
                        work_id = segment * buckets + bucket
                        owner = work_id % cores
                        key = domain(request, begin, head, split, context, lo, hi)
                        events[owner].append((work_id, anchor, key, min(256, hi - lo)))
                        grouped.update(range(relative // qtile,
                                             relative // qtile + cluster))
                        cluster_count += 1
                        cluster_units += tile_count
                ordinary_units += (ceildiv(length, qtile) - len(grouped)) * tile_count
        begin += length
    hits = saved_packed = 0
    core_counts = []
    for owner_events in events:
        previous = None
        owner_hits = 0
        for _, _, key, valid_rows in sorted(owner_events):
            if key == previous:
                hits += 1
                owner_hits += 1
                saved_packed += valid_rows * 144
            previous = key
        core_counts.append({"clusters": len(owner_events), "first_tile_hits": owner_hits})
    total = ordinary_units + cluster_units
    expected_route = ("decode" if max(lengths) <= 4 else
                      "C16" if max(lengths) >= 8192 else
                      "C4" if max(lengths) >= 84 else "ordinary")
    return {
        "request_query_tokens": lengths,
        "context_tokens_per_request": context,
        "gqa_ratio": 6, "dim": 256, "kv_heads_per_rank": heads,
        "splits": splits, "cores": cores, "modeled_cluster": cluster,
        "existing_route": expected_route,
        "matches_existing_route": expected_route == f"C{cluster}",
        "fits_current_16384_token_budget": tokens <= 16384,
        "eligible_clusters": cluster_count,
        "ordinary_history_KV256_units": ordinary_units,
        "cluster_history_KV256_units": cluster_units,
        "all_history_KV256_units": total,
        "first_tile_cache_hits": hits,
        "first_tile_hit_pct": 100 * hits / cluster_count if cluster_count else 0,
        "all_history_unpack_reduction_pct": 100 * hits / total if total else 0,
        "saved_packed_logical_GM_bytes": saved_packed,
        "saved_BF16_publish_logical_GM_bytes": hits * 2 * 256 * 256 * 2,
        "extra_cache_copy_GM_bytes": 0,
        "extra_workspace_bytes": 0,
        "extra_live_KV_address_set_bytes_per_used_core_upper_bound": 262144,
        "unchanged_cube_operand_loads": True,
        "per_core": core_counts,
    }


def workspace(cluster):
    # The offsets below use 4-byte units as actual SelectClusterSlots does.
    q, k, d = 128, 256, 256
    base = ((cluster * q + 2*k + q + cluster*q) * d + q*k + cluster*2*q)*4
    size = base + (2*k*d + (1 if cluster == 4 else 15)*q*k)*4
    base_k = cluster*q*d*4
    base_v = base_k + k*d*4
    bf16_tile = k*d*2
    slots = []
    live = [(0, base_k, "Q and P half-slots")]
    for label, start in (("base_K", base_k), ("base_V", base_v),
                         ("extra_K", base), ("extra_V", base + k*d*4)):
        live.append((start, start + bf16_tile, label))
        slots.append({"name": label, "active_BF16_byte_interval": [start, start+bf16_tile],
                      "unused_second_half_interval": [start+bf16_tile, start+2*bf16_tile]})
    live += [(base_v+2*bf16_tile, base, "score/PV/acc/stat"),
             (base+4*bf16_tile, size, "extra scores")]
    cache = [(base_k+bf16_tile, base_k+2*bf16_tile),
             (base_v+bf16_tile, base_v+2*bf16_tile)]
    for lo, hi in cache:
        assert 0 <= lo < hi <= size
        assert all(hi <= a or b <= lo for a, b, _ in live)
    return {"cluster": cluster, "workspace_bytes_per_core": size,
            "cache_K_and_V_byte_intervals": cache, "cache_payload_bytes": 262144,
            "extra_allocation_bytes": 0, "slots": slots,
            "extra_live_KV_address_set_bytes_upper_bound": 262144,
            "source_interval_nonoverlap_check": "passed",
            "sdk_actual_DMA_access_and_asynchronous_ownership": "not_device_certified"}


def source_checks():
    hashes = {}
    for name in SOURCES:
        data = (ROOT / name).read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
    for cluster in (4, 16):
        text = (ROOT / SOURCES[0 if cluster == 4 else 1]).read_text()
        assert f"const int64_t bucketWidth={cluster}*queryTile;" in text
        assert "const int64_t bucket=workId%buckets;" in text
        assert "const int64_t segment=workId/buckets;" in text
        assert "workId+=GetBlockNum()" in text
        assert "kvbegin!=original.kvbegin || kvend!=original.kvend" in text
        assert "kvend!=context || !TaskValid()" in text
        assert "LoadPacked(subStart);Unpack(false);PublishHistoryKv(false,subtile);" in text
        assert "Unpack(true);PublishHistoryKv(true,subtile);" in text
        assert "operands[2*(value?vOffset:kOffset)+slot*D]" in text
    prepare = (ROOT / SOURCES[2]).read_text()
    assert "if(kind==0) {first=Min(sink,context);last=context;}" in prepare
    dispatch = (ROOT / SOURCES[5]).read_text()
    assert "max_query_len >= 8192" in dispatch
    assert json.loads((ROOT / SOURCES[-1]).read_text())["max_num_batched_tokens"] == 16384
    return hashes


def report():
    hashes = source_checks()
    scenarios = [("single_16k", [16384]), ("eight_2k", [2048]*8),
                 ("four_4k", [4096]*4), ("eight_4k_budget_exceeded", [4096]*8)]
    scenarios += [(f"decode_bs{bs}_q{q}", [q]*bs) for bs in (16, 28) for q in (1, 4)]
    cases = []
    for name, lengths in scenarios:
        for cluster in (4, 16):
            cases.append({"name": name, **account(lengths, cluster=cluster)})
    # Bounds and negative controls: no healthy first tile cache can save more
    # than one unpacked KV256 per eligible cluster. Launch-boundary invalidation
    # and key equality are modeled as per-call local state, never persistent GM.
    for case in cases:
        assert case["first_tile_cache_hits"] <= case["eligible_clusters"]
        assert case["first_tile_cache_hits"] <= case["all_history_KV256_units"]
        if case["existing_route"] == "decode":
            assert case["eligible_clusters"] == case["first_tile_cache_hits"] == 0
    assert account([2048]*8, cluster=16)["first_tile_cache_hits"] == 0
    assert account([4096]*8, cluster=16)["first_tile_cache_hits"] == 0
    assert account([16384], cluster=16)["first_tile_cache_hits"] == 56
    assert account([16384], cluster=16, context=0)["all_history_KV256_units"] == 0
    sensitivity = [{"name": f"single16k_H{h}_S{s}_cores{n}",
                    **account([16384], cluster=16, context=h, splits=s, cores=n)}
                   for h in (128, 8192, 32768, 65536) for s in (1, 4) for n in (20, 24)]
    return {
        "scope": "offline exact healthy contiguous-task loop/byte model; not NPU time or HBM traffic",
        "source_sha256": hashes,
        "decision": "do_not_promote_single_first_tile_cache",
        "decision_reason": "existing-route H32768/S1/20-core examples save only 0.1202--0.4207% of all source0 unpack units; decode saves zero; scalar key/offset bookkeeping and protocol complexity are not cost-free",
        "actual_live_query_length_distribution": "not present in supplied profile aggregates; scenarios are synthetic, never observed frequencies",
        "assumptions": ["all listed requests have the same context and healthy contiguous slots",
                        "GQA6, D256, 2 local KV heads, 20 Cubes unless explicitly varied",
                        "kernel inputs and block table immutable for the duration of one launch",
                        "first-pass ordinary history units included; window/current/QK/PV/softmax unchanged",
                        "existing host route retained; no context threshold or task-owner change"],
        "cache_key": ["request", "requestBegin", "kvhead", "split", "context", "kvbegin", "kvend", "kind=0"],
        "cache_identity": "reset valid=false at each Init; raw/table pointers, strides, geometry and dequantMode fixed for one launch; no reuse across graph replays",
        "cache_error_contract": "each AIV separately caches final first-tile error and exact BF16 row bits; mark valid only after complete original Publish fences; error precedence retained",
        "copy_free_design": "first tile always publishes directly into unused second halves of base K/V slots and Cube consumes those addresses; other tiles retain both existing first-half buffers; no staging-copy detour",
        "working_set_caveat": "unchanged allocation does not mean unchanged active address set: retaining the cached tile beside two live ping-pong tiles can touch 256KiB more/core, up to 5MiB for 20 cores; no claim of free L2 residency",
        "added_costs_not_timed": ["one strict key comparison per eligible cluster", "per-AIV key/valid/error scalar state", "extra first-tile address selection bookkeeping", "new cache identity/lifetime proof and corresponding correctness tests"],
        "unchanged_costs": ["every query's QK/PV", "C16 GM-to-B1 K and V per cluster tile", "C4 per-member B operand consumption", "all online FP32 arithmetic/ordering", "all cross-core ready/score/P/rounddone/final-publication handshakes"],
        "bounds": "for L full KV256 history tiles per cluster, hits/cluster<=1 and unpack reduction within those clusters<=1/L; ordinary history lowers the total reduction further",
        "workspace": [workspace(4), workspace(16)],
        "cases": cases,
        "sensitivity_cases": sensitivity,
        "not_implemented": ["production cache", "CANN compile", "device numerical/graph validation", "performance test"],
        "excluded_prior_failures": ["B-NZ 8.40% regression", "V-transpose NZ 7.67% regression", "Kt/Vt NZ 6.88% regression and uint32 variant 6.05% regression"],
        "D4_four_answers": ["targets repeated history unpack between same-owner same-domain clusters",
                            "D.4 whole-history restore is forbidden and supplies no current timing percentage",
                            "bounded one-KV256 BF16 cache only; original codec, rounding, source/error domain, owner and recurrence preserved in proposed contract",
                            "zero extra GM copy/allocation is possible, but tiny modeled scope and unmeasured scalar costs do not establish a net improvement"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("cross_cluster_analysis.json"))
    args = parser.parse_args()
    result = report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for case in result["cases"]:
        if case["matches_existing_route"] or case["existing_route"] == "decode" and case["modeled_cluster"] == 4:
            print(f"{case['name']}: route={case['existing_route']} "
                  f"clusters={case['eligible_clusters']} hits={case['first_tile_cache_hits']} "
                  f"all_history_reduction={case['all_history_unpack_reduction_pct']:.6f}%")
