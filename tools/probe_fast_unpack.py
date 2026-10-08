# Archive #126/#129/#148-151 and #150/P0 startup D.4: an independent real-NPU gate
# for faster fused INT2 unpack. D.4's 6.5s full-history restore is forbidden;
# this probe tests the bounded CV tile, exact fe0 outputs, and measured speed.
"""Compare old and fast INT2 CV symbols on identical padded/long task tables."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
import math
from pathlib import Path
import statistics
import traceback

from . import probe_history_reuse as reuse
from .phase import atomic_json

ROOT = Path(__file__).resolve().parents[1]
FAST_BASE = "attention_cv_fast_out"
P0_WEIGHTED = "attention_cv_fast_weighted_out"
FAST_Q1 = "attention_cv_fast_q1_out"
FAST_C4 = "attention_cv_fast_cluster4_out"
_MODES = {
    "base": (reuse.FE0_CV_OP, FAST_BASE),
    "q1": (reuse.Q1_CV_OP, FAST_Q1),
    "c4": (reuse.CANDIDATE_OP, FAST_C4),
}
_TIMED = frozenset({"q4_decode32_n128_s3", "q1_decode32_n128_s3",
                    "q1_mixed32_n16384_s1", "mature_20k", "mixed_q1_q4_long"})
_GRAPHS = frozenset({"q4_decode32_n128_s3", "q1_decode32_n128_s3"})
_EXTRA_NUMERIC = frozenset({"base_frontier_d64", "base_mixed_hkv2_d128"})
_META_CASES = frozenset({"meta_half_base_d256", "meta_half_q1_d256",
                         "meta_half_c4_d256"})
_Q4_SPLIT_SCAN = (1, 2, 3, 4)
_VALID_HALF_BITS = (
    (64, "k", "scale", 0x0001), (64, "k", "zero", 0x8000),
    (65, "v", "scale", 0x03FF), (65, "v", "zero", 0x0001),
    (66, "k", "scale", 0x03FF), (66, "k", "zero", 0x8001),
    (67, "v", "scale", 0x0001), (67, "v", "zero", 0x8001),
    (68, "k", "scale", 0x3C00), (68, "k", "zero", 0xBC00),
    (69, "v", "scale", 0x3C00), (69, "v", "zero", 0x3C00),
)
_INVALID_HALF_BITS = (
    ("scale_pos_zero", "k", "scale", 0x0000),
    ("scale_neg_zero", "v", "scale", 0x8000),
    ("scale_nan", "k", "scale", 0x7E01),
    ("scale_pos_inf", "v", "scale", 0x7C00),
    ("scale_neg_inf", "k", "scale", 0xFC00),
    ("zero_nan", "v", "zero", 0x7E01),
    ("zero_pos_inf", "k", "zero", 0x7C00),
    ("zero_neg_inf", "v", "zero", 0xFC00),
)


class FastUnpackProbeError(RuntimeError):
    pass


def case_plan(target: dict, cores: int) -> tuple[tuple[reuse.Shape, str, bool], ...]:
    """#150/#151: keep every old C4 boundary, then measure five served shapes."""
    original = tuple((reuse.measurement_shape(shape, target, cores), "c4",
                      shape.name == "mature_20k") for shape in reuse.CASES)
    # C4's old nine cover D64/128/256. Exercise the separate fast base
    # template at D64 and D128 too, using the same frozen oracle shapes.
    frontier = next(shape for shape in reuse.CASES if shape.name == "frontier_511")
    hkv2 = next(shape for shape in reuse.CASES if shape.name == "mixed_hkv2_s3")
    base_numeric = ((replace(frontier, name="base_frontier_d64"), "base", False),
                    (replace(hkv2, name="base_mixed_hkv2_d128"), "base", False))
    q4 = reuse.measurement_shape(next(shape for shape in reuse.CASES
                                      if shape.name == "decode32"), target, cores)
    q4 = replace(q4, name="q4_decode32_n128_s3")
    if q4.splits != 3 or sum(q4.qlens) != 128:
        raise FastUnpackProbeError("q4 measurement does not match production N128/S3")
    q1 = tuple((shape, "q1", True) for shape in reuse.Q1_CASES)
    # One batch contains q1, q4, and a 385-token request. Recent=256 means
    # q128 cannot reach terminal history/C4 reuse; 385 crosses that boundary.
    # N390 uses S2 on the target's 20 Cubes, with no fake graph padding.
    mixed = reuse.Shape("mixed_q1_q4_long", (1, 4, 385),
                        (20000, 23000, 27000), 256, 1, 2, True)
    # D256 instantiates each new helper mode with live subnormal and signed
    # zero metadata. C4 q385 crosses the terminal history boundary.
    meta = ((reuse.Shape("meta_half_base_d256", (4,), (641,),
                         256, 1, 1, False), "base", False),
            (reuse.Shape("meta_half_q1_d256", (1,), (641,),
                         256, 1, 3, False, padded_tokens=128,
                         slot_context=True), "q1", False),
            (reuse.Shape("meta_half_c4_d256", (385,), (641,),
                         256, 1, 2, True), "c4", False))
    return (original + base_numeric + ((q4, "base", True),) + q1 +
            ((mixed, "c4", True),) + meta)


def _raw_pages(fixture: dict, raw):
    spec: reuse.Shape = fixture["spec"]
    slot_bytes = spec.dim // 2 + 8
    return raw[reuse.PREFIX:].view(fixture["blocks"], reuse.BLOCK_TOKENS,
                                   spec.kv_heads, slot_bytes)


def _metadata_bytes(fixture: dict, raw, position: int, kind: str,
                    field: str) -> tuple[int, int]:
    spec: reuse.Shape = fixture["spec"]
    if len(spec.qlens) != 1 or spec.kv_heads != 1:
        raise FastUnpackProbeError("special-half fixture requires one request/head")
    physical = fixture["page_assignments"][0][position // reuse.BLOCK_TOKENS]
    offset = ((spec.dim // 4 + 4) if kind == "v" else 0) + spec.dim // 4
    if field == "zero":
        offset += 2
    if kind not in {"k", "v"} or field not in {"scale", "zero"}:
        raise FastUnpackProbeError("invalid special-half field")
    return physical, offset


def _write_half_bits(fixture: dict, raw, position: int,
                     kind: str, field: str, bits: int) -> None:
    physical, offset = _metadata_bytes(fixture, raw, position, kind, field)
    pages = _raw_pages(fixture, raw)
    pages[physical, position % reuse.BLOCK_TOKENS, 0, offset] = bits & 255
    pages[physical, position % reuse.BLOCK_TOKENS, 0, offset + 1] = bits >> 8


def _read_half_bits(fixture: dict, raw, position: int,
                    kind: str, field: str) -> int:
    physical, offset = _metadata_bytes(fixture, raw, position, kind, field)
    pages = _raw_pages(fixture, raw)
    pair = pages[physical, position % reuse.BLOCK_TOKENS, 0, offset:offset + 2]
    return int(pair[0]) | (int(pair[1]) << 8)


def _recompute_oracle_after_metadata(torch, fixture: dict) -> None:
    """Independent PR decode and dense attention from the actual modified bytes."""
    from oscar_ascend.ops.reference import attention, decode_kv
    spec: reuse.Shape = fixture["spec"]
    context, length, dim = spec.contexts[0], spec.qlens[0], spec.dim
    tensors = fixture["cpu"]
    pages = _raw_pages(fixture, tensors["raw"])
    packed = torch.stack([pages[fixture["page_assignments"][0]
                                [position // reuse.BLOCK_TOKENS],
                                position % reuse.BLOCK_TOKENS, 0]
                          for position in range(context)])[:, None, :]
    rotated_k, rotated_v = decode_kv(packed, dim)
    rk = reuse._hadamard(torch, dim)
    rv = tensors["rv"]
    restored_k, restored_v = rotated_k @ rk.T, rotated_v @ rv.T
    exact_k, exact_v = restored_k.clone(), restored_v.clone()
    for position in sorted(set(range(min(reuse.SINK, context))) |
                           set(range(max(0, context - spec.recent_tokens), context))):
        physical = fixture["page_assignments"][0][position // reuse.BLOCK_TOKENS]
        in_page = position % reuse.BLOCK_TOKENS
        row = (position if position < reuse.SINK else
               reuse.SINK + in_page % (spec.recent_tokens + reuse.SPECULATIVE))
        exact_k[position] = tensors["wk"][physical, row].float()
        exact_v[position] = tensors["wv"][physical, row].float()
    expected = {}
    for local in reuse._samples(length):
        cut = min(context, max(reuse.SINK,
                               context + local + 1 - spec.recent_tokens))
        selected_k, selected_v = exact_k.clone(), exact_v.clone()
        selected_k[reuse.SINK:cut] = restored_k[reuse.SINK:cut]
        selected_v[reuse.SINK:cut] = restored_v[reuse.SINK:cut]
        keys = torch.cat((selected_k, tensors["ck"][:local + 1].float()))
        values = torch.cat((selected_v, tensors["cv"][:local + 1].float()))
        result = attention(tensors["q"][local:local + 1], keys, values,
                           scale=fixture["scale"], causal=False)
        expected[local] = (result.output[0], result.lse[0])
    fixture["expected"] = expected


def _inject_valid_metadata(torch, fixture: dict) -> list[dict]:
    raw = fixture["cpu"]["raw"]
    evidence = []
    # C4 only shares the terminal source0 split. With q385/context641/S2,
    # its shared split starts around 353; place special rows 400..405 inside
    # that grouped history, instead of testing only its independent split0.
    offset = 336 if fixture["spec"].name == "meta_half_c4_d256" else 0
    for base_position, kind, field, bits in _VALID_HALF_BITS:
        position = base_position + offset
        _write_half_bits(fixture, raw, position, kind, field, bits)
        if _read_half_bits(fixture, raw, position, kind, field) != bits:
            raise FastUnpackProbeError("special-half bytes did not survive fixture write")
        evidence.append({"position": position, "kind": kind,
                         "field": field, "half_bits": f"0x{bits:04x}"})
    _recompute_oracle_after_metadata(torch, fixture)
    fixture["input_sha256"] = reuse._fingerprint(fixture["cpu"])
    return evidence


def _buffers(torch, fixture: dict, device, cores: int, mode: str) -> dict:
    return reuse._allocate(torch, fixture, device, cores, candidate=mode == "c4")


def _launch(ops, tensors: dict, fixture: dict, buffers: dict,
            cores: int, mode: str, *, fast: bool) -> str:
    """The three fast ABI layouts are identical to their respective old op."""
    if mode not in _MODES:
        raise FastUnpackProbeError(f"unknown fast-unpack mode {mode}")
    old_name, fast_name = _MODES[mode]
    name = fast_name if fast else old_name
    spec: reuse.Shape = fixture["spec"]
    args = (tensors["q"], tensors["qr"], tensors["ck"], tensors["cv"],
            tensors["rv"], tensors["raw"], tensors["table"], tensors["wk"],
            tensors["wv"], tensors["tags"], buffers["tasks"],
            buffers["partial"], buffers["lse"], buffers["status"],
            buffers["workspace"])
    attrs = (reuse.BLOCK_TOKENS, fixture["blocks"], reuse.PREFIX,
             fixture["stride"], reuse.SINK, spec.recent_tokens,
             reuse.SPECULATIVE, spec.splits, fixture["scale"], cores)
    if mode == "c4":
        if buffers["cluster_stats"] is None:
            raise FastUnpackProbeError("C4 call lacks independent aligned cluster stats")
        getattr(ops, name)(*args, buffers["cluster_stats"], *attrs)
    else:
        if buffers["cluster_stats"] is not None:
            raise FastUnpackProbeError("base/q1 call unexpectedly carries C4 stats")
        getattr(ops, name)(*args, *attrs)
    return name


def _launch_weighted(ops, tensors: dict, fixture: dict, buffers: dict,
                     cores: int) -> str:
    """#150/P0: independent q4 symbol; never selected by production dispatch."""
    spec: reuse.Shape = fixture["spec"]
    args = (tensors["q"], tensors["qr"], tensors["ck"], tensors["cv"],
            tensors["rv"], tensors["raw"], tensors["table"], tensors["wk"],
            tensors["wv"], tensors["tags"], buffers["tasks"],
            buffers["partial"], buffers["lse"], buffers["status"],
            buffers["workspace"])
    attrs = (reuse.BLOCK_TOKENS, fixture["blocks"], reuse.PREFIX,
             fixture["stride"], reuse.SINK, spec.recent_tokens,
             reuse.SPECULATIVE, spec.splits, fixture["scale"], cores)
    getattr(ops, P0_WEIGHTED)(*args, *attrs)
    return P0_WEIGHTED


def predicted_kv256_weights(tasks, cores: int) -> list[int]:
    """Mirror the kernel's stable least-loaded assignment from actual task rows."""
    if cores <= 0 or cores > 32 or tasks.ndim != 2 or tasks.shape[1] != 16:
        raise FastUnpackProbeError("invalid P0 prediction geometry")
    loads = [0] * cores
    for task_id, row in enumerate(tasks.tolist()):
        qcount, begin, end, error = (int(row[1]), int(row[3]),
                                     int(row[4]), int(row[10]))
        weight = 0 if qcount <= 0 or error or begin < 0 or end < begin else \
            max(1, (end - begin + 255) // 256)
        if weight:
            owner = min(range(cores), key=lambda core: loads[core])
            loads[owner] += weight
        else:
            owner = task_id % cores
    return loads


def split_scan_decision(samples: dict[int, list[float]]) -> dict:
    """Choose only a repeatably faster split; the production S3 remains default."""
    if set(samples) != set(_Q4_SPLIT_SCAN):
        raise FastUnpackProbeError("q4 split scan is missing S1/S2/S3/S4")
    for split, values in samples.items():
        if (len(values) != 5 or any(type(value) not in (int, float) or
                not math.isfinite(value) or value <= 0 for value in values)):
            raise FastUnpackProbeError(f"q4 S{split} has invalid Event samples")
    medians = {split: statistics.median(values) for split, values in samples.items()}
    best = min(medians, key=lambda split: (medians[split], split))
    stable = best != 3 and all(candidate < baseline for candidate, baseline in
                               zip(samples[best], samples[3]))
    return {"measured_best_split": best,
            "recommended_split": best if stable else 3,
            "decision": f"promote_s{best}_after_graph_gate" if stable else "retain_s3",
            "stable_all_five_faster_than_s3": stable,
            "median_ms": {f"s{split}": medians[split] for split in _Q4_SPLIT_SCAN},
            "best_over_s3": medians[best] / medians[3]}


def _weighted_split_graph(torch, ops, tensors: dict, fixture: dict,
                          buffers: dict, cores: int) -> dict:
    """Capture S2 at fixed addresses and prove changed Q/task replay."""
    reuse._poison(torch, buffers)
    _launch_weighted(ops, tensors, fixture, buffers, cores)
    torch.npu.synchronize()
    _check_one(torch, fixture, buffers, invalid=False)
    original = reuse._snapshot(buffers)
    graph = torch.npu.NPUGraph()
    reuse._poison(torch, buffers)
    with torch.npu.graph(graph, capture_error_mode="thread_local",
                         auto_dispatch_capture=True):
        _launch_weighted(ops, tensors, fixture, buffers, cores)
    torch.npu.synchronize()
    reuse._poison(torch, buffers)
    graph.replay(); torch.npu.synchronize()
    _check_one(torch, fixture, buffers, invalid=False)
    _same_bits(torch, original, buffers, "weighted split initial graph replay")

    tensors["q"].copy_(-tensors["q"])
    tensors["qr"].copy_(-tensors["qr"])
    changed_tasks = buffers["tasks"].cpu()
    chosen = next((row for row in changed_tasks
                   if int(row[7]) == 0 and int(row[1]) > 0 and
                   int(row[4] - row[3]) >= 32), None)
    if chosen is None:
        raise FastUnpackProbeError("S2 graph fixture has no live source0 range")
    chosen[4] -= 16
    buffers["tasks"].copy_(changed_tasks)
    reuse._poison(torch, buffers)
    graph.replay(); torch.npu.synchronize()
    _check_one(torch, fixture, buffers, invalid=False)
    changed = reuse._snapshot(buffers)
    if all(reuse._bitwise_identical(torch, original[name], changed[name])
           for name in ("partial", "lse")):
        raise FastUnpackProbeError("S2 graph replay ignored changed input")
    changed_merge = reuse._merge_only(torch, ops, fixture, buffers)
    reuse._poison(torch, buffers)
    _launch_weighted(ops, tensors, fixture, buffers, cores)
    torch.npu.synchronize()
    _check_one(torch, fixture, buffers, invalid=False)
    _same_bits(torch, changed, buffers, "weighted S2 eager versus graph")
    _same_merge(torch, changed_merge, reuse._merge_only(torch, ops, fixture, buffers),
                "weighted S2 changed-input graph")
    return {"capture": "passed", "replay": "passed",
            "changed_query_and_task_same_addresses": True,
            "split": fixture["spec"].splits, "operator": P0_WEIGHTED}


def _q4_split_scan(torch, ops, target: dict, device, cores: int,
                   acceptance: dict) -> dict:
    """P1.5: compare S1-S4 on one exact q4 input and attest candidate routing."""
    route_enabled = True
    if type(route_enabled) is not bool:
        raise FastUnpackProbeError("weighted q4 S2 route flag must be explicit boolean")
    base = reuse.measurement_shape(next(shape for shape in reuse.CASES
                                        if shape.name == "decode32"), target, cores)
    base = replace(base, name="q4_split_scan", splits=3)
    fixture = reuse.make_fixture(torch, base)
    tensors = {name: tensor.to(device) for name, tensor in fixture["cpu"].items()}
    variants = {}
    input_hash = fixture["input_sha256"]
    for split in _Q4_SPLIT_SCAN:
        split_fixture = dict(fixture)
        split_fixture["spec"] = replace(base, splits=split)
        buffers = _buffers(torch, split_fixture, device, cores, "base")
        preparation = reuse._prepare(torch, ops, tensors, split_fixture, buffers)
        reuse._poison(torch, buffers)
        _launch_weighted(ops, tensors, split_fixture, buffers, cores)
        torch.npu.synchronize()
        _check_one(torch, split_fixture, buffers, invalid=False)
        oracle = reuse._merge_and_oracle(
            torch, ops, split_fixture, buffers, acceptance["fused_attention"])
        weights = predicted_kv256_weights(buffers["tasks"].cpu(), cores)
        variants[split] = {"fixture": split_fixture, "buffers": buffers,
                           "task_sha256": preparation["task_sha256"],
                           "weights": weights,
                           "oracle": {key: value for key, value in oracle.items()
                                      if key not in ("output", "lse")}}

    policy = acceptance["performance"]
    if (policy["warmup"], policy["repeats"]) != (2, 5):
        raise FastUnpackProbeError("q4 split scan requires frozen 2+5 policy")
    samples = {split: [] for split in _Q4_SPLIT_SCAN}
    stream = torch.npu.current_stream()
    for index in range(policy["warmup"] + policy["repeats"]):
        order = _Q4_SPLIT_SCAN if index < policy["warmup"] or \
            (index - policy["warmup"]) % 2 == 0 else tuple(reversed(_Q4_SPLIT_SCAN))
        for split in order:
            item = variants[split]
            buffers, split_fixture = item["buffers"], item["fixture"]
            reuse._poison(torch, buffers)
            if index >= policy["warmup"]:
                begin, end = (torch.npu.Event(enable_timing=True),
                              torch.npu.Event(enable_timing=True))
                begin.record(stream)
            _launch_weighted(ops, tensors, split_fixture, buffers, cores)
            if index >= policy["warmup"]:
                end.record(stream); end.synchronize()
                elapsed = float(begin.elapsed_time(end))
                if not math.isfinite(elapsed) or elapsed <= 0:
                    raise FastUnpackProbeError("invalid q4 split-scan Event duration")
                samples[split].append(elapsed)
            else:
                torch.npu.synchronize()
            _check_one(torch, split_fixture, buffers, invalid=False)
            reuse._merge_and_oracle(
                torch, ops, split_fixture, buffers, acceptance["fused_attention"])

    decision = split_scan_decision(samples)
    winner = decision["recommended_split"]
    graph = _weighted_split_graph(
        torch, ops, tensors, variants[winner]["fixture"],
        variants[winner]["buffers"], cores)
    rows = []
    for split in _Q4_SPLIT_SCAN:
        item = variants[split]
        mean = sum(item["weights"]) / cores
        rows.append({"split": split, "operator": P0_WEIGHTED,
                     "input_sha256": input_hash,
                     "task_sha256": item["task_sha256"],
                     "predicted_kv256_per_core": item["weights"],
                     "max_over_mean": max(item["weights"]) / mean,
                     "device_event_ms": samples[split],
                     "median_ms": statistics.median(samples[split]),
                     "precision": "passed", "frozen_oracle": "passed",
                     "oracle": item["oracle"]})
    return {"status": "passed", "shape": "32*q4/Hq6/Hkv1/D256/N128",
            "splits": rows, "warmup": 2, "repeats": 5,
            "order": "alternating_forward_reverse",
            "same_input_all_splits": len({row["input_sha256"] for row in rows}) == 1,
            **decision,
            "recommended_split_graph": graph,
            "production_promotion_allowed": graph["capture"] == graph["replay"] == "passed",
            "production_route_changed": route_enabled,
            "production_route": "weighted_s2" if route_enabled else "unchanged",
            "promotion_gate": "passed"}


def _check_one(torch, fixture: dict, buffers: dict, *, invalid: bool,
               expected_stats: dict | None = None,
               expect_clusters: bool | None = None) -> dict | None:
    reuse._check_status(torch, buffers, invalid=invalid)
    stats = buffers["cluster_stats"]
    if stats is None:
        return None
    values = reuse._check_cluster_stats(stats.cpu(),
        expect_clusters=fixture["spec"].expect_clusters if expect_clusters is None
                        else expect_clusters,
        expected_source0_leaders=expected_stats["source0_leaders"] if expected_stats else None)
    required = fixture["spec"].required_clusters if expect_clusters is None else None
    if required is not None and values["eligible_clusters"] != required:
        raise FastUnpackProbeError(
            f"{fixture['spec'].name}: C4 clusters {values['eligible_clusters']} != {required}")
    return values


def _same_bits(torch, expected: dict, buffers: dict, label: str) -> None:
    reuse._assert_same_bits(torch, expected, buffers, label)


def _same_merge(torch, expected: dict, actual: dict, label: str) -> None:
    for name in ("output", "lse"):
        if not reuse._bitwise_identical(torch, expected[name], actual[name]):
            raise FastUnpackProbeError(f"{label} merged {name} differs bitwise")


def _merge(torch, ops, fixture: dict, buffers: dict, acceptance: dict):
    return reuse._merge_and_oracle(torch, ops, fixture, buffers,
                                   acceptance["fused_attention"])


def _live_source_error_codes(tasks_cpu, status_cpu,
                             spec: reuse.Shape, tokens: int) -> list[int]:
    tasks = tasks_cpu.view(tokens, spec.kv_heads, 3, spec.splits, 16)
    status = status_cpu.view(tokens, spec.kv_heads, 3, spec.splits, 2)
    live_source0 = tasks[:, :, 0, :, 1] > 0
    final = status[:, :, 0, :, :][live_source0]
    codes = sorted({int(value) for value in final.flatten().tolist()
                    if int(value) != 0})
    if not codes:
        raise FastUnpackProbeError("invalid live metadata produced no final source0 error")
    return codes


def _invalid_metadata_matrix(torch, ops, tensors: dict, fixture: dict,
                             baseline: dict, old: dict, fast: dict,
                             cores: int, mode: str,
                             valid_reference: dict) -> dict:
    """#126/#148: each bad half value is isolated, then a dead tail stays inert."""
    valid_raw = fixture["cpu"]["raw"]
    live_position = 400 if fixture["spec"].name == "meta_half_c4_d256" else 64
    cases = []
    for name, kind, field, bits in _INVALID_HALF_BITS:
        mutated = valid_raw.clone()
        _write_half_bits(fixture, mutated, live_position, kind, field, bits)
        if _read_half_bits(fixture, mutated, live_position, kind, field) != bits:
            raise FastUnpackProbeError(f"{name}: invalid half bits were not written")
        tensors["raw"].copy_(mutated.to(tensors["raw"].device))
        outputs = []
        for label, buffers, fast_flag in (("fe0", baseline, None),
                                          ("old", old, False),
                                          ("fast", fast, True)):
            reuse._poison(torch, buffers)
            if fast_flag is None:
                reuse._launch(ops, tensors, fixture, buffers, cores, candidate=False)
            else:
                _launch(ops, tensors, fixture, buffers, cores, mode, fast=fast_flag)
            torch.npu.synchronize()
            _check_one(torch, fixture, buffers, invalid=True)
            # Unpack first marks bad metadata as 3, but a later non-finite
            # score/Softmax check may publish 2 for the same live task. The
            # final status code is the ABI: observe fe0 and require exact
            # old/fast parity, never hardcode an intermediate internal code.
            codes = _live_source_error_codes(
                buffers["tasks"].cpu(), buffers["status"].cpu(),
                fixture["spec"], fixture["tokens"])
            outputs.append((label, reuse._snapshot(buffers), codes))
        for label, _, codes in outputs[1:]:
            _same_bits(torch, outputs[0][1], old if label == "old" else fast,
                       f"{name} {label} versus fe0")
            if codes != outputs[0][2]:
                raise FastUnpackProbeError(f"{name}: {label} final error codes differ from fe0")
        cases.append({"name": name, "position": live_position,
                      "kind": kind, "field": field,
                      "half_bits": f"0x{bits:04x}",
                      "fe0_old_fast_final_nonzero_status_codes": outputs[0][2],
                      "partial_lse_status": "bitwise_passed"})
    # One poisoned future slot belongs to an allocated page but lies beyond
    # every actual causal task range. It may never produce a metadata error.
    dead_position = fixture["spec"].contexts[0] + fixture["spec"].qlens[0] + 1
    mutated = valid_raw.clone()
    _write_half_bits(fixture, mutated, dead_position, "k", "scale", 0x0000)
    tensors["raw"].copy_(mutated.to(tensors["raw"].device))
    for label, buffers, fast_flag in (("fe0", baseline, None),
                                      ("old", old, False),
                                      ("fast", fast, True)):
        reuse._poison(torch, buffers)
        if fast_flag is None:
            reuse._launch(ops, tensors, fixture, buffers, cores, candidate=False)
        else:
            _launch(ops, tensors, fixture, buffers, cores, mode, fast=fast_flag)
        torch.npu.synchronize()
        _check_one(torch, fixture, buffers, invalid=False)
        _same_bits(torch, valid_reference, buffers, f"{label} dead-tail metadata")
    tensors["raw"].copy_(valid_raw.to(tensors["raw"].device))
    torch.npu.synchronize()
    return {"status": "passed", "live_invalid_cases": cases,
            "dead_tail": {"position": dead_position, "half_bits": "0x0000",
                          "fe0_old_fast_output_status": "bitwise_passed"}}


def _terminal_counterfactual(torch, ops, tensors: dict, fixture: dict,
                             baseline: dict, old: dict, fast: dict,
                             cores: int) -> dict:
    """#148: disabling only the terminal split must remove sharing, not math."""
    original = baseline["tasks"].cpu()
    changed = original.clone()
    count = reuse._disable_terminal_source0_split(changed, fixture["spec"])
    if count < 4:
        raise FastUnpackProbeError("terminal split counterfactual changed too few leaders")
    baseline["tasks"].copy_(changed)
    try:
        reuse._poison(torch, baseline)
        reuse._launch(ops, tensors, fixture, baseline, cores, candidate=False)
        torch.npu.synchronize()
        reuse._check_status(torch, baseline, invalid=False)
        reference = reuse._snapshot(baseline)
        reference_merge = reuse._merge_only(torch, ops, fixture, baseline)
        counters = {}
        for label, buffers, fast_flag in (("old", old, False), ("fast", fast, True)):
            reuse._poison(torch, buffers)
            _launch(ops, tensors, fixture, buffers, cores, "c4", fast=fast_flag)
            torch.npu.synchronize()
            stats = _check_one(torch, fixture, buffers, invalid=False,
                               expect_clusters=False)
            _same_bits(torch, reference, buffers, f"nonterminal {label} C4 versus fe0")
            _same_merge(torch, reference_merge,
                        reuse._merge_only(torch, ops, fixture, buffers),
                        f"nonterminal {label} C4")
            if stats["eligible_clusters"] != 0:
                raise FastUnpackProbeError("nonterminal C4 retained a history-sharing cluster")
            counters[label] = stats
        if counters["old"] != counters["fast"]:
            raise FastUnpackProbeError("fast C4 changed nonterminal schedule statistics")
        return {"terminal_leaders_modified": count,
                "old_fast_stats_bitwise": True,
                "fe0_old_fast_partial_lse_status_merge": "bitwise_passed",
                "frozen_oracle": "not_applicable_changed_task_range"}
    finally:
        baseline["tasks"].copy_(original)


def _timed_pair(torch, ops, tensors: dict, fixture: dict, old: dict,
                fast: dict, cores: int, mode: str, reference_bits: dict,
                acceptance: dict) -> dict:
    policy = acceptance["performance"]
    if (policy["warmup"], policy["repeats"], policy["max_latency_ratio"]) != (2, 5, 1.0):
        raise FastUnpackProbeError("fast performance must retain frozen 2+5 and ratio<=1.0")
    samples: dict[str, list[float]] = {"old": [], "fast": []}
    stream = torch.npu.current_stream()
    for index in range(policy["warmup"] + policy["repeats"]):
        order = (False, True) if index < policy["warmup"] or \
            (index - policy["warmup"]) % 2 == 0 else (True, False)
        for use_fast in order:
            buffers = fast if use_fast else old
            label = "fast" if use_fast else "old"
            reuse._poison(torch, buffers)
            if index >= policy["warmup"]:
                begin = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                begin.record(stream)
            _launch(ops, tensors, fixture, buffers, cores, mode, fast=use_fast)
            if index >= policy["warmup"]:
                end.record(stream)
                end.synchronize()
                elapsed = float(begin.elapsed_time(end))
                if not math.isfinite(elapsed) or elapsed <= 0:
                    raise FastUnpackProbeError("invalid fast-unpack NPU Event duration")
                samples[label].append(elapsed)
            else:
                torch.npu.synchronize()
            _check_one(torch, fixture, buffers, invalid=False)
            _same_bits(torch, reference_bits, buffers,
                       f"{fixture['spec'].name} {label} timed repeat")
    old_median, fast_median = (statistics.median(samples[name])
                               for name in ("old", "fast"))
    ratio = fast_median / old_median
    return {"old": {"device_event_ms": samples["old"], "median_ms": old_median},
            "fast": {"device_event_ms": samples["fast"], "median_ms": fast_median},
            "fast_over_old": ratio,
            "gate": "passed" if ratio <= policy["max_latency_ratio"] else "failed",
            "warmup": 2, "repeats": 5, "order": "alternating_AB_BA",
            "scope": "distinct_old_vs_fast_operator_real_NPU_Event_no_profiler"}


def _timed_p0(torch, ops, tensors: dict, fixture: dict, fast: dict,
              weighted: dict, cores: int, reference_bits: dict,
              acceptance: dict) -> dict:
    policy = acceptance["performance"]
    samples = {"fast": [], "weighted": []}
    stream = torch.npu.current_stream()
    for index in range(policy["warmup"] + policy["repeats"]):
        order = ("fast", "weighted") if index < policy["warmup"] or \
            (index - policy["warmup"]) % 2 == 0 else ("weighted", "fast")
        for label in order:
            buffers = fast if label == "fast" else weighted
            reuse._poison(torch, buffers)
            if index >= policy["warmup"]:
                begin, end = (torch.npu.Event(enable_timing=True),
                              torch.npu.Event(enable_timing=True))
                begin.record(stream)
            if label == "fast":
                _launch(ops, tensors, fixture, buffers, cores, "base", fast=True)
            else:
                _launch_weighted(ops, tensors, fixture, buffers, cores)
            if index >= policy["warmup"]:
                end.record(stream);end.synchronize()
                elapsed = float(begin.elapsed_time(end))
                if not math.isfinite(elapsed) or elapsed <= 0:
                    raise FastUnpackProbeError("invalid P0 NPU Event duration")
                samples[label].append(elapsed)
            else:
                torch.npu.synchronize()
            _check_one(torch, fixture, buffers, invalid=False)
            _same_bits(torch, reference_bits, buffers, f"P0 {label} timed repeat")
    medians = {name: statistics.median(values) for name, values in samples.items()}
    ratio = medians["weighted"] / medians["fast"]
    return {"fast": {"device_event_ms": samples["fast"], "median_ms": medians["fast"]},
            "weighted": {"device_event_ms": samples["weighted"],
                         "median_ms": medians["weighted"]},
            "weighted_over_fast": ratio,
            "gate": "passed" if ratio < policy["max_latency_ratio"] else "failed",
            "warmup": 2, "repeats": 5, "order": "alternating_AB_BA",
            "scope": "independent_q4_fast_vs_weighted_real_NPU_Event"}


def _changed_input_graph(torch, ops, tensors: dict, fixture: dict,
                         old: dict, fast: dict, cores: int, mode: str,
                         original_bits: dict, weighted: dict | None = None) -> dict:
    """#129/#151: old and fast standalone graphs replay changed Q/tasks in place."""
    if mode not in {"base", "q1"} or fixture["tokens"] != 128:
        raise FastUnpackProbeError("fast graph gate requires q4/q1 N128")
    graph_old, graph_fast = torch.npu.NPUGraph(), torch.npu.NPUGraph()
    graph_weighted = torch.npu.NPUGraph() if weighted is not None else None
    reuse._poison(torch, old)
    reuse._poison(torch, fast)
    torch.npu.synchronize()
    with torch.npu.graph(graph_old, capture_error_mode="thread_local", auto_dispatch_capture=True):
        _launch(ops, tensors, fixture, old, cores, mode, fast=False)
    with torch.npu.graph(graph_fast, capture_error_mode="thread_local", auto_dispatch_capture=True):
        _launch(ops, tensors, fixture, fast, cores, mode, fast=True)
    if weighted is not None:
        reuse._poison(torch, weighted)
        with torch.npu.graph(graph_weighted, capture_error_mode="thread_local", auto_dispatch_capture=True):
            _launch_weighted(ops, tensors, fixture, weighted, cores)
    torch.npu.synchronize()
    reuse._poison(torch, old)
    reuse._poison(torch, fast)
    graph_old.replay()
    graph_fast.replay()
    if graph_weighted is not None:graph_weighted.replay()
    torch.npu.synchronize()
    _check_one(torch, fixture, old, invalid=False)
    _check_one(torch, fixture, fast, invalid=False)
    _same_bits(torch, original_bits, old, "old initial graph replay")
    _same_bits(torch, original_bits, fast, "fast initial graph replay")
    if weighted is not None:
        _check_one(torch, fixture, weighted, invalid=False)
        _same_bits(torch, original_bits, weighted, "weighted initial graph replay")
    tensors["q"].copy_(-tensors["q"])
    tensors["qr"].copy_(-tensors["qr"])
    changed_tasks = old["tasks"].cpu()
    chosen = next((row for row in changed_tasks
                   if int(row[7]) == 0 and int(row[1]) > 0 and
                   int(row[4] - row[3]) >= 32), None)
    if chosen is None:
        raise FastUnpackProbeError("fast graph fixture has no live source0 range")
    chosen[4] -= 16
    old["tasks"].copy_(changed_tasks)
    reuse._poison(torch, old)
    reuse._poison(torch, fast)
    if weighted is not None:reuse._poison(torch, weighted)
    graph_old.replay()
    graph_fast.replay()
    if graph_weighted is not None:graph_weighted.replay()
    torch.npu.synchronize()
    _check_one(torch, fixture, old, invalid=False)
    _check_one(torch, fixture, fast, invalid=False)
    changed_bits = reuse._snapshot(old)
    _same_bits(torch, changed_bits, fast, "fast changed-input graph replay")
    if weighted is not None:
        _check_one(torch, fixture, weighted, invalid=False)
        _same_bits(torch, changed_bits, weighted, "weighted changed-input graph replay")
    if all(reuse._bitwise_identical(torch, original_bits[name], changed_bits[name])
           for name in ("partial", "lse")):
        raise FastUnpackProbeError("fast graph replay ignored changed input")
    changed_merge = reuse._merge_only(torch, ops, fixture, old)
    _same_merge(torch, changed_merge, reuse._merge_only(torch, ops, fixture, fast),
                "fast changed-input graph")
    for label, buffers, fast_flag in (("old", old, False), ("fast", fast, True)):
        reuse._poison(torch, buffers)
        _launch(ops, tensors, fixture, buffers, cores, mode, fast=fast_flag)
        torch.npu.synchronize()
        _check_one(torch, fixture, buffers, invalid=False)
        _same_bits(torch, changed_bits, buffers,
                   f"{label} changed-input eager versus graph")
        _same_merge(torch, changed_merge,
                    reuse._merge_only(torch, ops, fixture, buffers),
                    f"{label} changed-input eager versus graph")
    return {"capture": "passed", "replay": "passed",
            "changed_query_and_task_same_addresses": True,
            "weighted_operator": P0_WEIGHTED if weighted is not None else "not_run",
            "scope": "standalone_N128_operator_graph_not_full_service"}


def run_case(torch, ops, spec: reuse.Shape, mode: str, timed: bool,
             device, cores: int, acceptance: dict) -> dict:
    """Check fe0 repeatability, old semantics, fast semantics, then speed."""
    fixture = reuse.make_fixture(torch, spec)
    valid_half_bits = (_inject_valid_metadata(torch, fixture)
                       if spec.name in _META_CASES else None)
    tensors = {name: tensor.to(device) for name, tensor in fixture["cpu"].items()}
    baseline = reuse._allocate(torch, fixture, device, cores, candidate=False)
    old = _buffers(torch, fixture, device, cores, mode)
    fast = _buffers(torch, fixture, device, cores, mode)
    weighted = (_buffers(torch, fixture, device, cores, "base")
                if spec.name == "q4_decode32_n128_s3" else None)
    preparation = reuse._prepare(torch, ops, tensors, fixture, baseline)
    for buffers in (old, fast):
        buffers["tasks"] = baseline["tasks"]
        buffers["positions"] = baseline["positions"]
    if weighted is not None:
        weighted["tasks"] = baseline["tasks"]
        weighted["positions"] = baseline["positions"]
    invalid = spec.corrupt_metadata or spec.corrupt_qr
    reuse._poison(torch, baseline)
    reuse._launch(ops, tensors, fixture, baseline, cores, candidate=False)
    torch.npu.synchronize()
    _check_one(torch, fixture, baseline, invalid=invalid)
    reference_bits = reuse._snapshot(baseline)
    first_merge = None if invalid else _merge(torch, ops, fixture, baseline, acceptance)
    reuse._poison(torch, baseline)
    reuse._launch(ops, tensors, fixture, baseline, cores, candidate=False)
    torch.npu.synchronize()
    _check_one(torch, fixture, baseline, invalid=invalid)
    _same_bits(torch, reference_bits, baseline, "fe0 repeat fast probe")
    repeat_merge = None if invalid else _merge(torch, ops, fixture, baseline, acceptance)
    if first_merge is not None:
        for name in ("output", "lse"):
            if not reuse._bitwise_identical(torch, first_merge[name], repeat_merge[name]):
                raise reuse.BaselineNondeterministic(
                    f"fe0 fast-probe repeat merged {name} differs bitwise")
    candidate_results = {}
    for label, buffers, fast_flag in (("old", old, False), ("fast", fast, True)):
        reuse._poison(torch, buffers)
        _launch(ops, tensors, fixture, buffers, cores, mode, fast=fast_flag)
        torch.npu.synchronize()
        stats = _check_one(torch, fixture, buffers, invalid=invalid,
                           expected_stats=preparation)
        _same_bits(torch, reference_bits, buffers, f"{spec.name} {label} versus fe0")
        candidate_results[label] = {"stats": stats,
            "merge": None if invalid else _merge(torch, ops, fixture, buffers, acceptance)}
        if repeat_merge is not None:
            _same_merge(torch, repeat_merge, candidate_results[label]["merge"],
                        f"{spec.name} {label} versus fe0")
    if candidate_results["old"]["stats"] != candidate_results["fast"]["stats"]:
        raise FastUnpackProbeError(f"{spec.name}: fast changed C4 schedule statistics")
    if not invalid:
        _same_merge(torch, candidate_results["old"]["merge"],
                    candidate_results["fast"]["merge"],
                    f"{spec.name} fast versus old")
    p0 = None
    if weighted is not None:
        reuse._poison(torch, weighted)
        _launch_weighted(ops, tensors, fixture, weighted, cores)
        torch.npu.synchronize()
        _check_one(torch, fixture, weighted, invalid=False)
        _same_bits(torch, reference_bits, weighted, "P0 weighted versus fe0")
        _same_merge(torch, repeat_merge,
                    reuse._merge_only(torch, ops, fixture, weighted),
                    "P0 weighted versus fe0")
        p0 = {"operator": P0_WEIGHTED,
              "predicted_kv256_per_core": predicted_kv256_weights(
                  baseline["tasks"].cpu(), cores)}
        p0["max_over_mean"] = max(p0["predicted_kv256_per_core"]) / (
            sum(p0["predicted_kv256_per_core"]) / cores)
        p0["performance"] = _timed_p0(
            torch, ops, tensors, fixture, fast, weighted, cores,
            reference_bits, acceptance)
    counterfactual = (_terminal_counterfactual(torch, ops, tensors, fixture,
                                               baseline, old, fast, cores)
                      if spec.name == "mixed_unaligned_mature_s3" else None)
    invalid_metadata = (_invalid_metadata_matrix(
        torch, ops, tensors, fixture, baseline, old, fast, cores, mode,
        reference_bits) if spec.name in _META_CASES else None)
    timing = (_timed_pair(torch, ops, tensors, fixture, old, fast, cores,
                          mode, reference_bits, acceptance) if timed else None)
    graph = (_changed_input_graph(torch, ops, tensors, fixture, old, fast,
                                  cores, mode, reference_bits, weighted)
             if spec.name in _GRAPHS else None)
    row = {"case": spec.name, "status": "passed" if timing is None or
           timing["gate"] == "passed" else "failed",
           "mode": mode, "old_operator": _MODES[mode][0],
           "fast_operator": _MODES[mode][1],
           "shape": {"qlens": spec.qlens, "contexts": spec.contexts,
                     "actual_tokens": fixture["actual_tokens"],
                     "padded_tokens": fixture["tokens"],
                     "padding_tokens": fixture["tokens"] - fixture["actual_tokens"],
                     "head_dim": spec.dim, "kv_heads": spec.kv_heads,
                     "splits": spec.splits, "cube_cores": cores,
                     "slot_context": spec.slot_context},
           "input_sha256": fixture["input_sha256"],
           "task_sha256": preparation["task_sha256"],
           "padding_task_gate": "passed" if fixture["tokens"] > fixture["actual_tokens"]
                                else "not_applicable",
           "poison_gate": "passed", "fe0_repeatability": "bitwise_passed",
           "precision": "bitwise_passed", "partial_lse_status": "bitwise_passed",
           "merged": "invalid_input_not_applicable" if invalid else "bitwise_passed",
           "frozen_oracle": "invalid_input_not_applicable" if invalid else "passed",
           "sampled_queries": len(fixture["expected"]) if not invalid else 0,
           "old_cluster_stats": candidate_results["old"]["stats"],
           "fast_cluster_stats": candidate_results["fast"]["stats"],
           "terminal_split_counterfactual": counterfactual,
           "fp16_metadata_valid_bits": valid_half_bits,
           "fp16_metadata_invalid_and_dead_tail": invalid_metadata,
           "graph_capture": graph["capture"] if graph else "not_run",
           "graph_replay": graph["replay"] if graph else "not_run",
           "graph": graph, "performance": timing, "p0_weighted": p0}
    del fixture, tensors, baseline, old, fast, weighted
    gc.collect()
    torch.npu.empty_cache()
    return row


def verdict(rows: list[dict]) -> dict:
    """Fail closed if any required numerical, graph, or distinct-op speed row is absent."""
    expected = {shape.name for shape in reuse.CASES} | _EXTRA_NUMERIC | _TIMED | _META_CASES
    if (len(rows) != len(expected) or {row.get("case") for row in rows} != expected):
        raise FastUnpackProbeError("fast-unpack result is missing a required case")
    invalid_cases = {shape.name for shape in reuse.CASES
                     if shape.corrupt_metadata or shape.corrupt_qr}
    def precision_ok(row: dict) -> bool:
        mode = row.get("mode")
        if mode not in _MODES:
            return False
        invalid = row.get("case") in invalid_cases
        return (row.get("precision") == "bitwise_passed" and
                row.get("partial_lse_status") == "bitwise_passed" and
                row.get("fe0_repeatability") == "bitwise_passed" and
                row.get("frozen_oracle") == (
                    "invalid_input_not_applicable" if invalid else "passed") and
                row.get("merged") == (
                    "invalid_input_not_applicable" if invalid else "bitwise_passed") and
                (row.get("old_operator"), row.get("fast_operator")) == _MODES[mode] and
                (row.get("case") != "mixed_unaligned_mature_s3" or
                 (isinstance(row.get("terminal_split_counterfactual"), dict) and
                  row["terminal_split_counterfactual"].get(
                      "fe0_old_fast_partial_lse_status_merge") == "bitwise_passed")) and
                (row.get("case") not in _META_CASES or
                 (isinstance(row.get("fp16_metadata_valid_bits"), list) and
                  len(row["fp16_metadata_valid_bits"]) == len(_VALID_HALF_BITS) and
                  isinstance(row.get("fp16_metadata_invalid_and_dead_tail"), dict) and
                  row["fp16_metadata_invalid_and_dead_tail"].get("status") == "passed" and
                  len(row["fp16_metadata_invalid_and_dead_tail"].get(
                      "live_invalid_cases", ())) == len(_INVALID_HALF_BITS) and
                  row["fp16_metadata_invalid_and_dead_tail"].get(
                      "dead_tail", {}).get("fe0_old_fast_output_status") ==
                      "bitwise_passed")) and
                (row.get("shape", {}).get("padding_tokens", 0) == 0 or
                 row.get("padding_task_gate") == "passed"))
    precision = "passed" if all(
        precision_ok(row) for row in rows) else "failed"
    by_name = {row["case"]: row for row in rows}
    graph_capture = "passed" if all(by_name[name].get("graph_capture") == "passed"
                                     for name in _GRAPHS) else "failed"
    graph_replay = "passed" if all(by_name[name].get("graph_replay") == "passed"
                                    for name in _GRAPHS) else "failed"
    def speed_ok(row: dict) -> bool:
        result = row.get("performance")
        if not isinstance(result, dict) or result.get("gate") != "passed" or \
                result.get("warmup") != 2 or result.get("repeats") != 5 or \
                result.get("order") != "alternating_AB_BA":
            return False
        try:
            old = result["old"]["device_event_ms"]
            new = result["fast"]["device_event_ms"]
            if (len(old) != 5 or len(new) != 5 or
                    any(type(value) not in (int, float) or not math.isfinite(value) or
                        value <= 0 for value in old + new)):
                return False
            old_median, new_median = statistics.median(old), statistics.median(new)
            ratio = new_median / old_median
            return (ratio <= 1.0 and
                    math.isclose(result["old"]["median_ms"], old_median, rel_tol=1e-12) and
                    math.isclose(result["fast"]["median_ms"], new_median, rel_tol=1e-12) and
                    math.isclose(result["fast_over_old"], ratio, rel_tol=1e-12))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return False
    performance = "passed" if all(speed_ok(by_name[name]) for name in _TIMED) else "failed"
    p0 = by_name["q4_decode32_n128_s3"].get("p0_weighted")
    def p0_ok() -> bool:
        try:
            timing = p0["performance"]
            old = timing["fast"]["device_event_ms"]
            new = timing["weighted"]["device_event_ms"]
            weights = p0["predicted_kv256_per_core"]
            old_median, new_median = statistics.median(old), statistics.median(new)
            ratio = new_median / old_median
            return (p0.get("operator") == P0_WEIGHTED and len(weights) == 20 and
                    all(type(value) is int and value >= 0 for value in weights) and
                    sum(weights) > 0 and len(old) == len(new) == 5 and
                    all(type(value) in (int, float) and math.isfinite(value) and
                        value > 0 for value in old + new) and
                    timing.get("gate") == "passed" and ratio < 1.0 and
                    math.isclose(timing["fast"]["median_ms"], old_median, rel_tol=1e-12) and
                    math.isclose(timing["weighted"]["median_ms"], new_median, rel_tol=1e-12) and
                    math.isclose(timing["weighted_over_fast"], ratio, rel_tol=1e-12) and
                    timing.get("warmup") == 2 and timing.get("repeats") == 5 and
                    timing.get("order") == "alternating_AB_BA")
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return False
    p0_performance = "passed" if isinstance(p0, dict) and p0_ok() else "failed"
    failures = [name for name, value in (("precision", precision),
                ("graph_capture", graph_capture), ("graph_replay", graph_replay),
                ("performance", performance), ("p0_performance", p0_performance))
                if value != "passed"]
    return {"status": "passed" if not failures else "failed",
            "precision": precision, "graph_capture": graph_capture,
            "graph_replay": graph_replay, "performance": performance,
            "p0_performance": p0_performance,
            "failed_gates": failures,
            "failed_cases": [row["case"] for row in rows if row.get("status") != "passed"]}


def probe(config_path: Path, acceptance_path: Path) -> dict:
    target = json.loads(config_path.read_text())
    acceptance = json.loads(acceptance_path.read_text())
    reuse._active_device(target)
    policy = acceptance.get("performance", {})
    if (acceptance.get("frozen_before_measurement") is not True or
            policy.get("warmup") != 2 or policy.get("repeats") != 5 or
            policy.get("statistic") != "median" or
            policy.get("max_latency_ratio") != 1.0 or
            policy.get("per_case_required") is not True or
            policy.get("noise_allowance") != 0.0):
        raise FastUnpackProbeError("frozen precision and 2+5 speed policy is required")
    fe0_source = reuse.assert_fe0_kernel_unchanged()
    import torch
    import torch_npu  # noqa: F401 - this command requires actual Ascend NPUs
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.npu.set_device(0)
    from .build_ops import normalize_soc
    if (not torch.npu.is_available() or
            normalize_soc(torch.npu.get_device_name(0)) != target["soc_version"]):
        raise FastUnpackProbeError("selected target NPU/SOC is unavailable")
    from oscar_ascend.ops.loader import require_capabilities, validate_build_artifacts
    manifest_path = ROOT / "build/ascendc/build_manifest.json"
    manifest = validate_build_artifacts(manifest_path)
    require_capabilities({"prepare_attention_tasks_out", "merge_lse_out",
                          reuse.FE0_CV_OP, reuse.Q1_CV_OP, reuse.CANDIDATE_OP,
                          FAST_BASE, P0_WEIGHTED, FAST_Q1, FAST_C4}, manifest_path)
    from .probe_native_current_fia import _target_geometry
    if _target_geometry(target) != (6, 1, 256):
        raise FastUnpackProbeError("target TP head geometry differs from Hq6/Hkv1/D256")
    cores = reuse._core_count(torch, target)
    if cores != 20:
        raise FastUnpackProbeError("target A2 geometry requires 20 Cube cores")
    plan = case_plan(target, cores)
    device = torch.device("npu:0")
    rows = []
    for spec, mode, timed in plan:
        row = run_case(torch, torch.ops.oscar_ascend_ops, spec, mode,
                       timed, device, cores, acceptance)
        rows.append(row)
        if timed:
            result = row["performance"]
            print("[oscar] PERF_FAST_UNPACK " + json.dumps({
                "case": row["case"], "mode": mode,
                "old_operator": row["old_operator"],
                "fast_operator": row["fast_operator"],
                "old_ms": result["old"]["median_ms"],
                "fast_ms": result["fast"]["median_ms"],
                "ratio": result["fast_over_old"],
                "precision": row["precision"],
                "graph": row["graph_replay"],
                "status": row["status"]}, sort_keys=True), flush=True)
            if row.get("p0_weighted") is not None:
                p0 = row["p0_weighted"]
                timing = p0["performance"]
                print("[oscar] PERF_P0_WEIGHTED " + json.dumps({
                    "case": row["case"], "operator": p0["operator"],
                    "predicted_kv256_per_core": p0["predicted_kv256_per_core"],
                    "max_over_mean": p0["max_over_mean"],
                    "fast_ms": timing["fast"]["median_ms"],
                    "weighted_ms": timing["weighted"]["median_ms"],
                    "ratio": timing["weighted_over_fast"],
                    "graph": row["graph_replay"], "gate": timing["gate"]},
                    sort_keys=True), flush=True)
    split_scan = _q4_split_scan(torch, torch.ops.oscar_ascend_ops, target,
                                device, cores, acceptance)
    print("[oscar] PERF_Q4_SPLIT_SCAN " + json.dumps({
        "status": split_scan["status"],
        "decision": split_scan["decision"],
        "measured_best_split": split_scan["measured_best_split"],
        "recommended_split": split_scan["recommended_split"],
        "best_over_s3": split_scan["best_over_s3"],
        "median_ms": split_scan["median_ms"],
        "stable_all_five_faster_than_s3":
            split_scan["stable_all_five_faster_than_s3"],
        "graph": split_scan["recommended_split_graph"],
        "promotion_gate": split_scan["promotion_gate"],
        "production_route_changed": split_scan["production_route_changed"],
        "production_route": split_scan["production_route"]}, sort_keys=True), flush=True)
    gates = verdict(rows)
    return {**gates, "q4_split_scan_gate": split_scan["status"],
            "q4_split_scan": split_scan,
            "scope": "independent_real_NPU_operator_precision_graph_and_Event_AB",
            "baseline": "fe0_old_C4_old_q1_mode_matched",
            "full_service_acceptance": "not_run",
            "default_route": "fe0",
            "candidate_evaluation_allowed": gates["status"] == "passed",
            "reference_commit": reuse.BASELINE_COMMIT,
            "fe0_source_sha256": fe0_source,
            "artifact_signature": manifest["signature"],
            "artifact_sha256": manifest["sha256"],
            "device": str(device), "device_name": torch.npu.get_device_name(0),
            "physical_devices": target["devices"], "cases": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/target.json")
    parser.add_argument("--acceptance", type=Path,
                        default=ROOT / "configs/acceptance.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.name in {"fast-unpack.json", "status.json"}:
        parser.error("#151: measurement output must differ from phase status JSON")
    try:
        report = probe(args.config, args.acceptance)
        atomic_json(args.output, report)
        print("[oscar] PERF_FAST_UNPACK_RESULT " + json.dumps({
            "status": report["status"], "precision": report["precision"],
            "graph_capture": report["graph_capture"],
            "graph_replay": report["graph_replay"],
            "performance": report["performance"],
            "q4_split_scan_gate": report["q4_split_scan_gate"],
            "failed_cases": report["failed_cases"],
            "report": str(args.output)}, sort_keys=True), flush=True)
        return 0 if report["status"] == "passed" else 2
    except Exception as error:
        report = {"status": "failed", "precision": "not_established",
                  "graph_capture": "not_established",
                  "graph_replay": "not_established",
                  "performance": "not_established",
                  "candidate_evaluation_allowed": False,
                  "error_type": type(error).__name__, "error": str(error),
                  "default_route": "fe0", "device_completion": "not_established"}
        atomic_json(args.output, report)
        traceback.print_exc()
        print("[oscar] PERF_FAST_UNPACK_RESULT " + json.dumps({
            "status": "failed", "error": str(error), "report": str(args.output)},
            sort_keys=True), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
