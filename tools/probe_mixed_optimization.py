# Archive #126/#129/#143-151 and startup D.4: compare complete bounded INT2
# CV paths on identical NPU inputs. D.4 dequant was 6.5s/card; this probe
# rejects numerical drift and any measured regression before service use.
"""Short NPU old-fast versus balanced/C16 mixed-attention gate."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import traceback

from . import probe_fast_unpack as fast
from . import probe_history_reuse as reuse
from . import probe_mixed_attention as mixed
from .phase import atomic_json

ROOT = Path(__file__).resolve().parents[1]
OLD_BASE = "attention_cv_fast_out"
NEW_BALANCED = "attention_cv_fast_balanced_out"
OLD_C4 = "attention_cv_fast_cluster4_out"
NEW_C16 = "attention_cv_fast_cluster16_out"
WARMUP = 2
REPEATS = 5


class MixedOptimizationProbeError(RuntimeError):
    pass


class _RedirectOps:
    """Only the candidate CV symbol changes; all other ops stay identical."""

    def __init__(self, ops, old: str, new: str):
        self.ops, self.old, self.new = ops, old, new

    def __getattr__(self, name: str):
        return getattr(self.ops, self.new if name == self.old else name)


def shape_plan() -> tuple[reuse.Shape, reuse.Shape, reuse.Shape]:
    cold, warm = mixed.shape_plan()
    long_context = 65100 - mixed.PREFILL_Q
    long = replace(warm, name="mixed_history_65k",
                   contexts=(65100,) * mixed.DECODE_REQUESTS + (long_context,))
    if (long.contexts[-1] + long.qlens[-1] != 65100 or
            any(context != 65100 for context in long.contexts[:-1]) or
            sum(long.qlens) != mixed.MIXED_TOKENS):
        raise MixedOptimizationProbeError("65K fixture lost the real mixed query geometry")
    return cold, warm, long


def _long_fixture(torch, spec: reuse.Shape) -> dict:
    logical = hashlib.sha256()

    def observe(request, _start, q, old_k, old_v, ck, cv):
        if request < mixed.DECODE_REQUESTS:
            logical.update(mixed._logical_request_hash(request, q, old_k, old_v, ck, cv))

    fixture = reuse.make_fixture(torch, spec, on_request=observe)
    fixture["logical_decode_input_sha256"] = logical.hexdigest()
    return fixture


def _c16_workspace_per_core(dim: int) -> int:
    """Mirror csrc/include/oscar_attention_launch.h's exact C16 bound."""
    if dim not in (64, 128, 256):
        raise MixedOptimizationProbeError("unsupported C16 head dimension")
    rows, kv_rows = 128, 256
    return ((16 * rows + 2 * kv_rows + rows + 16 * rows) * dim
            + rows * kv_rows + 16 * 2 * rows) * 4


def _allocate_pair(torch, fixture: dict, device, cores: int, *, is_mixed: bool):
    old = mixed._allocate(torch, fixture, device, cores, mixed=is_mixed)
    new = mixed._allocate(torch, fixture, device, cores, mixed=is_mixed)
    if is_mixed:
        new["cv"]["workspace"] = torch.empty(
            cores * _c16_workspace_per_core(fixture["spec"].dim),
            dtype=torch.uint8, device=device)
    # Both variants read the exact same prepared task/position tensors.
    new["cv"]["tasks"] = old["cv"]["tasks"]
    new["cv"]["positions"] = old["cv"]["positions"]
    return old, new


def _symbols(is_mixed: bool) -> tuple[str, str, str]:
    return (OLD_C4, NEW_C16, "c4") if is_mixed else (OLD_BASE, NEW_BALANCED, "base")


def _candidate_ops(ops, is_mixed: bool):
    old, new, _ = _symbols(is_mixed)
    return _RedirectOps(ops, old, new)


def _launch_cv(ops, tensors: dict, fixture: dict, cv: dict, cores: int,
               *, is_mixed: bool, candidate: bool,
               candidate_name: str | None = None) -> str:
    old_name, new_name, mode = _symbols(is_mixed)
    target = candidate_name or new_name
    selected_ops = (_RedirectOps(ops, old_name, target)
                    if candidate and target != old_name else ops)
    fast._launch(selected_ops, tensors, fixture, cv, cores, mode, fast=True)
    return target if candidate else old_name


def _check_c16_stats(torch, cv: dict, expected_leaders: int,
                     *, require_clusters: bool) -> dict:
    stats = cv.get("cluster_stats")
    if stats is None or tuple(stats.shape)[1:] != (8,):
        raise MixedOptimizationProbeError("C16 cluster stats buffer is missing")
    rows = stats.cpu().tolist()
    if any(len(row) != 8 or any(value < 0 for value in row) for row in rows):
        raise MixedOptimizationProbeError("C16 cluster stats are malformed")
    # #149: first-pass skips and second-pass leaders have different core
    # ownership. Their equality is checked only after summing all cores.
    for core, row in enumerate(rows):
        if row[1] != 16 * row[0] or row[4] != 15 * row[3] or row[7] != row[0]:
            raise MixedOptimizationProbeError(f"C16 per-core counters fail at core {core}")
    totals = [sum(row[index] for row in rows) for index in range(8)]
    if (totals[6] != totals[1] or totals[1] + totals[2] != expected_leaders or
            (require_clusters and (totals[0] <= 0 or totals[3] <= 0)) or
            (not require_clusters and totals[0] != 0)):
        raise MixedOptimizationProbeError("C16 global ownership counters fail")
    return dict(zip(reuse.STATS_FIELDS, totals))


def _fault_cluster(tasks_cpu, fixture: dict) -> tuple[int, int]:
    """Find one truly eligible sixteen-group source0 bucket on this fixture."""
    spec = fixture["spec"]
    if spec.kv_heads != 1 or spec.splits != 1 or len(spec.qlens) != 1:
        raise MixedOptimizationProbeError("C16 fault gate requires one request/head/split")
    tile = 128 // (spec.heads // spec.kv_heads)
    width = 16 * tile
    rows = tasks_cpu.view(fixture["tokens"], 1, 3, 1, 16)[:, 0, 0, 0]
    for anchor in range(0, fixture["tokens"] - width + 1, width):
        members = [rows[anchor + j * tile].tolist() for j in range(16)]
        first = members[0]
        if (first[1] != tile or first[3] < reuse.SINK or
                first[4] <= first[3] or first[4] != spec.contexts[0]):
            continue
        if all(row[0] == anchor + j * tile and row[1] == tile and
               row[3:6] == first[3:6] and row[8:11] == first[8:11]
               for j, row in enumerate(members)):
            position = first[3] + min(16, first[4] - first[3] - 1)
            return anchor, position
    raise MixedOptimizationProbeError("C16 graph fixture lacks a complete shared live cluster")


def _error_propagation_cases(torch, ops, tensors: dict, fixture: dict,
                             old: dict, new: dict, cores: int,
                             expected_leaders: int) -> list[dict]:
    """Compare final status for shared-scale and one-group QR faults.

    Invalid-input partials may contain NaN payloads; their bits are not an
    oracle. The final two-AIV status, its affected source0 leader, and a true
    C16 cluster are the error contract (#126/#145/#148).
    """
    old_cv, new_cv = old["cv"], new["cv"]
    anchor, position = _fault_cluster(old_cv["tasks"].cpu(), fixture)
    tile = 128 // (fixture["spec"].heads // fixture["spec"].kv_heads)
    one_group_token = anchor + 7 * tile
    original_raw = tensors["raw"].clone()
    original_qr = tensors["qr"].clone()
    rows = []
    try:
        for label in ("shared_live_scale_zero", "one_group_query_rot_nan"):
            tensors["raw"].copy_(original_raw)
            tensors["qr"].copy_(original_qr)
            if label == "shared_live_scale_zero":
                corrupted = fixture["cpu"]["raw"].clone()
                fast._write_half_bits(fixture, corrupted, position, "k", "scale", 0)
                tensors["raw"].copy_(corrupted.to(tensors["raw"].device))
            else:
                tensors["qr"][one_group_token, 0, 0] = float("nan")
            statuses = []
            for cv, candidate in ((old_cv, False), (new_cv, True)):
                reuse._poison(torch, cv)
                _launch_cv(ops, tensors, fixture, cv, cores,
                           is_mixed=True, candidate=candidate)
                torch.npu.synchronize()
                statuses.append(cv["status"].cpu().clone())
            old_status, new_status = statuses
            if not torch.equal(old_status, new_status) or not bool((old_status != 0).any()):
                raise MixedOptimizationProbeError(
                    f"{label}: C16 final status differs from fast C4 or misses fault")
            source0 = old_status.view(fixture["tokens"], 1, 3, 1, 2)[:, 0, 0, 0]
            if not bool((source0[anchor:anchor + 16 * tile] != 0).any()):
                raise MixedOptimizationProbeError(f"{label}: active C16 cluster missed fault")
            _check_c16_stats(torch, new_cv, expected_leaders,
                             require_clusters=True)
            rows.append({"case": label, "status": "passed",
                         "cluster_anchor": anchor,
                         "fault_position": position if label == "shared_live_scale_zero"
                                           else one_group_token,
                         "nonzero_status_words": int((old_status != 0).sum()),
                         "status_bitwise": "passed",
                         "partial_lse": "not_compared_invalid_NaN_domain"})
    finally:
        tensors["raw"].copy_(original_raw)
        tensors["qr"].copy_(original_qr)
        torch.npu.synchronize()
    return rows


def _same_bits(torch, old: dict, new: dict, label: str) -> None:
    for name in ("partial", "lse", "status"):
        if not reuse._bitwise_identical(torch, old["cv"][name], new["cv"][name]):
            raise MixedOptimizationProbeError(f"{label}: {name} changed bitwise")
    for name in ("output", "output_lse", "merge_status"):
        if not reuse._bitwise_identical(torch, old[name], new[name]):
            raise MixedOptimizationProbeError(f"{label}: merged {name} changed bitwise")


def _reference_bits(buffers: dict) -> dict:
    return {**reuse._snapshot(buffers["cv"]),
            **{name: buffers[name].clone() for name in
               ("output", "output_lse", "merge_status")}}


def _check_reference_bits(torch, reference: dict, buffers: dict,
                          label: str) -> None:
    for name in ("partial", "lse", "status"):
        actual = buffers["cv"][name]
        if not reuse._bitwise_identical(torch, reference[name], actual):
            raise MixedOptimizationProbeError(f"{label}: {name} changed on repeat")
    for name in ("output", "output_lse", "merge_status"):
        if not reuse._bitwise_identical(torch, reference[name], buffers[name]):
            raise MixedOptimizationProbeError(f"{label}: merged {name} changed on repeat")


def _verdict(samples: dict, old_name: str, new_name: str,
             max_ratio: float) -> dict:
    if (not 0 < max_ratio <= 1 or
            any(len(samples[label]) != REPEATS for label in ("old", "new"))):
        raise MixedOptimizationProbeError("comparison lacks 2+5 samples")
    medians = {label: {metric: statistics.median(row[metric] for row in samples[label])
                       for metric in ("cv_ms", "total_ms")}
               for label in ("old", "new")}
    if any(not math.isfinite(value) or value <= 0
           for values in medians.values() for value in values.values()):
        raise MixedOptimizationProbeError("invalid NPU Event median")
    ratios = {metric: medians["new"][metric] / medians["old"][metric]
              for metric in ("cv_ms", "total_ms")}
    same_op = old_name == new_name
    return {"old": medians["old"], "new": medians["new"],
            "ratios": ratios, "status": "passed" if (same_op or all(
                value <= max_ratio for value in ratios.values())) else "failed",
            "ratio_scope": "identical_operator_repeatability" if same_op else
                           "different_operators_strict_speed_gate",
            "warmup": WARMUP, "repeats": REPEATS,
            "order": "AB_BA_alternating", "max_latency_ratio": max_ratio,
            "scope": "one_NPU_same_stream_operator_and_attention_pipeline_not_full_model"}


def _graph_cv(torch, ops, tensors: dict, fixture: dict, old: dict,
              new: dict, cores: int, *, is_mixed: bool,
              expected_leaders: int, require_clusters: bool,
              candidate_name: str | None = None) -> dict:
    """Capture distinct ops, then change Q in place and compare graph/eager."""
    old_cv, new_cv = old["cv"], new["cv"]
    original_q, original_qr = tensors["q"].clone(), tensors["qr"].clone()
    try:
        reuse._poison(torch, old_cv)
        _launch_cv(ops, tensors, fixture, old_cv, cores,
                   is_mixed=is_mixed, candidate=False)
        torch.npu.synchronize()
        reuse._check_status(torch, old_cv, invalid=False)
        original = reuse._snapshot(old_cv)
        reuse._poison(torch, new_cv)
        _launch_cv(ops, tensors, fixture, new_cv, cores,
                   is_mixed=is_mixed, candidate=True,
                   candidate_name=candidate_name)
        torch.npu.synchronize()
        reuse._check_status(torch, new_cv, invalid=False)
        reuse._assert_same_bits(torch, original, new_cv, "new eager before graph")
        if is_mixed:
            _check_c16_stats(torch, new_cv, expected_leaders,
                             require_clusters=require_clusters)
        old_graph, new_graph = torch.npu.NPUGraph(), torch.npu.NPUGraph()
        reuse._poison(torch, old_cv)
        reuse._poison(torch, new_cv)
        torch.npu.synchronize()
        with torch.npu.graph(old_graph, capture_error_mode="thread_local",
                             auto_dispatch_capture=True):
            _launch_cv(ops, tensors, fixture, old_cv, cores,
                       is_mixed=is_mixed, candidate=False)
        with torch.npu.graph(new_graph, capture_error_mode="thread_local",
                             auto_dispatch_capture=True):
            _launch_cv(ops, tensors, fixture, new_cv, cores,
                       is_mixed=is_mixed, candidate=True,
                       candidate_name=candidate_name)
        torch.npu.synchronize()
        for changed in (False, True):
            if changed:
                tensors["q"].copy_(-original_q)
                tensors["qr"].copy_(-original_qr)
            reuse._poison(torch, old_cv)
            reuse._poison(torch, new_cv)
            old_graph.replay()
            new_graph.replay()
            torch.npu.synchronize()
            reuse._check_status(torch, old_cv, invalid=False)
            reuse._check_status(torch, new_cv, invalid=False)
            replay_bits = reuse._snapshot(old_cv)
            reuse._assert_same_bits(torch, replay_bits, new_cv,
                                    "changed-Q candidate graph replay")
            if not changed:
                reuse._assert_same_bits(torch, original, old_cv,
                                        "initial old graph replay")
            elif all(reuse._bitwise_identical(torch, original[name], replay_bits[name])
                     for name in ("partial", "lse")):
                raise MixedOptimizationProbeError("graph replay ignored changed Q")
            for candidate, cv in ((False, old_cv), (True, new_cv)):
                reuse._poison(torch, cv)
                _launch_cv(ops, tensors, fixture, cv, cores,
                           is_mixed=is_mixed, candidate=candidate,
                           candidate_name=candidate_name)
                torch.npu.synchronize()
                reuse._check_status(torch, cv, invalid=False)
                reuse._assert_same_bits(torch, replay_bits, cv,
                                        "graph versus eager changed-Q")
            if is_mixed:
                _check_c16_stats(torch, new_cv, expected_leaders,
                                 require_clusters=require_clusters)
        return {"graph_capture": "passed", "graph_replay": "passed",
                "changed_q_same_addresses": True,
                "old_operator": _symbols(is_mixed)[0],
                "new_operator": candidate_name or _symbols(is_mixed)[1],
                "scope": "standalone_CV_operator_graph_not_full_service"}
    finally:
        tensors["q"].copy_(original_q)
        tensors["qr"].copy_(original_qr)
        torch.npu.synchronize()


def _run_first_draft_once(torch, ops, fixture: dict, tensors: dict,
                          buffers: dict, cores: int) -> dict:
    """First MTP draft: full source2 CV followed by merge, no native FIA.

    This is the production C4/C16 ABI and task table. Prepare and rotation
    happened before the Event envelope; no source2 rewrite occurs here.
    """
    spec = fixture["spec"]
    n, h, d, splits = fixture["tokens"], spec.heads, spec.dim, spec.splits
    cv = buffers["cv"]
    reuse._poison(torch, cv)
    buffers["output"].fill_(float("nan"))
    buffers["output_lse"].fill_(float("nan"))
    buffers["merge_status"].fill_(-99)
    stream = torch.npu.current_stream()
    start = torch.npu.Event(enable_timing=True)
    after_cv = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record(stream)
    fast._launch(ops, tensors, fixture, cv, cores, "c4", fast=True)
    after_cv.record(stream)
    ops.merge_lse_out(cv["partial"].view(n * h, 3 * splits, d),
                      cv["lse"].view(n * h, 3 * splits),
                      buffers["output"], buffers["output_lse"],
                      buffers["merge_status"])
    end.record(stream)
    end.synchronize()
    if torch.npu.current_stream() != stream:
        raise MixedOptimizationProbeError("first-draft CV/merge changed NPU stream")
    result = {"cv_ms": float(start.elapsed_time(after_cv)),
              "merge_ms": float(after_cv.elapsed_time(end)),
              "total_ms": float(start.elapsed_time(end)),
              "suppress_ms": 0.0, "guard_ms": 0.0, "current_ms": 0.0}
    if any(not math.isfinite(value) or value < 0 for value in result.values()) or \
            result["cv_ms"] <= 0 or result["total_ms"] <= 0:
        raise MixedOptimizationProbeError("invalid first-draft Event duration")
    return result


def _run_case(torch, ops, fixture: dict, device, cores: int,
              acceptance: dict, *, is_mixed: bool,
              first_draft: bool = False) -> dict:
    if first_draft and not is_mixed:
        raise MixedOptimizationProbeError("first draft requires C4/C16 full-source ABI")
    spec = fixture["spec"]
    old_name, new_name, _ = _symbols(is_mixed)
    from oscar_ascend.ops.cv_dispatch import select_cv_op
    selected = select_cv_op(16, spec.heads, spec.kv_heads, fixture["tokens"],
                            max(spec.qlens), fast_unpack=True, mixed_cv=True)
    if selected not in ({new_name} if is_mixed else {old_name, new_name}):
        raise MixedOptimizationProbeError(
            f"{spec.name}: unexpected production candidate {selected}")
    tensors = {name: tensor.to(device) for name, tensor in fixture["cpu"].items()}
    old, new = _allocate_pair(torch, fixture, device, cores, is_mixed=is_mixed)
    prepared = mixed._prepare_rotation(torch, ops, fixture, tensors, old, cores)
    pristine = old["cv"]["tasks"].clone() if is_mixed and not first_draft else None
    cumulative = (tuple(int(v) for v in fixture["cpu"]["starts"][1:].tolist())
                  if is_mixed and not first_draft else None)
    new_ops = _RedirectOps(ops, old_name, selected) if selected != old_name else ops
    if is_mixed and not first_draft:
        mixed._verify_suppressed_cv(torch, ops, fixture, tensors, old, pristine, cores)
        mixed._verify_suppressed_cv(torch, new_ops, fixture, tensors, new, pristine, cores)
    # Numerical and independent-oracle checks precede every measured repeat.
    def run(buffers, chosen_ops):
        if first_draft:
            return _run_first_draft_once(torch, chosen_ops, fixture,
                                         tensors, buffers, cores)
        return mixed._run_once(torch, chosen_ops, fixture, tensors, buffers,
                               cores, mixed=is_mixed, pristine_tasks=pristine,
                               cumulative=cumulative)
    run(old, ops)
    old_oracle = mixed._oracle(torch, fixture, old, acceptance)
    old_reference = _reference_bits(old)
    old_stats = (fast._check_one(torch, fixture, old["cv"], invalid=False,
                                 expected_stats=prepared) if is_mixed else None)
    run(new, new_ops)
    new_oracle = mixed._oracle(torch, fixture, new, acceptance)
    _check_reference_bits(torch, old_reference, new, spec.name + " candidate")
    _same_bits(torch, old, new, spec.name)
    stats = (_check_c16_stats(torch, new["cv"], prepared["source0_leaders"],
                             require_clusters=spec.expect_clusters)
             if is_mixed else None)
    graph_result = ({"graph_capture": "reused", "graph_replay": "reused",
                     "scope": "identical_fast_base_N128_production_route_not_new_graph_evidence"}
                    if not is_mixed and fixture["tokens"] == 128 and
                    selected == old_name else None)
    samples = {"old": [], "new": []}
    for index in range(WARMUP + REPEATS):
        order = ("old", "new") if index < WARMUP or (index - WARMUP) % 2 == 0 \
            else ("new", "old")
        for label in order:
            row = run(old, ops) if label == "old" else run(new, new_ops)
            if index >= WARMUP:
                samples[label].append(row)
            mixed._oracle(torch, fixture, old if label == "old" else new,
                          acceptance)
            _check_reference_bits(torch, old_reference,
                                  old if label == "old" else new,
                                  spec.name + " " + label + " timed")
    performance = _verdict(samples, old_name, selected,
                           acceptance["performance"]["max_latency_ratio"])
    result = {"case": spec.name, "status": "passed" if performance["status"] == "passed"
              else "failed", "precision": "passed", "old_operator": old_name,
              "new_operator": selected, "shape": {"requests": len(spec.qlens),
              "total_n": fixture["tokens"], "active_n": fixture["actual_tokens"],
              "splits": spec.splits, "max_query_len": max(spec.qlens),
              "max_context": max(spec.contexts), "end_position": max(
                  a + b for a, b in zip(spec.contexts, spec.qlens))},
              "input_sha256": fixture["input_sha256"],
              "logical_decode_input_sha256": fixture.get("logical_decode_input_sha256"),
              "task_sha256": prepared["task_sha256"],
              "oracle": {"old": old_oracle, "new": new_oracle},
              "partial_lse_status_merged": "bitwise_passed",
              "c4_stats": old_stats, "c16_stats": stats, "graph": graph_result,
              "performance": performance,
              "event_samples_ms": samples,
              "source2_policy": ("full_CV_first_MTP_draft" if first_draft else
                                 "suppressed_native_FIA" if is_mixed else "CV_current"),
              "scope": ("first_MTP_draft_FULL_attention_CV_plus_merge_only_excludes_"
                        "GDN_and_other_MTP_model_work" if first_draft else
                        "one_NPU_FULL_attention_only_not_TP4_or_AISBench")}
    del tensors, old, new
    gc.collect()
    torch.npu.empty_cache()
    return result


def _balanced_graph_case(torch, ops, warm_fixture: dict, device,
                         cores: int, acceptance: dict) -> dict:
    """A real >128 production-route q4 graph, with one shared decode cohort."""
    fixture = mixed.derive_decode_fixture(torch, warm_fixture,
                                          padded_tokens=512, splits=1)
    spec = fixture["spec"]
    from oscar_ascend.ops.cv_dispatch import select_cv_op
    selected = select_cv_op(16, spec.heads, spec.kv_heads, fixture["tokens"],
                            max(spec.qlens), fast_unpack=True, mixed_cv=True)
    if selected != NEW_BALANCED:
        raise MixedOptimizationProbeError("N512/S1 production route did not select balanced")
    tensors = {name: tensor.to(device) for name, tensor in fixture["cpu"].items()}
    old, new = _allocate_pair(torch, fixture, device, cores, is_mixed=False)
    prepared = mixed._prepare_rotation(torch, ops, fixture, tensors, old, cores)
    mixed._run_once(torch, ops, fixture, tensors, old, cores, mixed=False)
    mixed._oracle(torch, fixture, old, acceptance)
    mixed._run_once(torch, _candidate_ops(ops, False), fixture,
                    tensors, new, cores, mixed=False)
    mixed._oracle(torch, fixture, new, acceptance)
    _same_bits(torch, old, new, "balanced N512 eager")
    graph = _graph_cv(torch, ops, tensors, fixture, old, new, cores,
                      is_mixed=False, expected_leaders=prepared["source0_leaders"],
                      require_clusters=False, candidate_name=NEW_BALANCED)
    result = {"case": spec.name, "status": "passed", "input_sha256":
              fixture["input_sha256"], "logical_decode_input_sha256":
              fixture["logical_decode_input_sha256"],
              "production_operator": selected, "oracle": "passed", **graph}
    del tensors, old, new, fixture
    gc.collect()
    torch.npu.empty_cache()
    return result


def _c16_graph_case(torch, ops, device, cores: int,
                    acceptance: dict) -> dict:
    # The first request-relative 336-token bucket can be frontier. A 1024
    # query/641-context fixture with recent=32 contains later mature C16
    # buckets and remains small enough for changed-Q graph diagnosis.
    spec = reuse.Shape("c16_active_graph", (1024,), (641,), 256, 1, 1,
                       True, recent_tokens=32)
    fixture = reuse.make_fixture(torch, spec)
    tensors = {name: tensor.to(device) for name, tensor in fixture["cpu"].items()}
    old, new = _allocate_pair(torch, fixture, device, cores, is_mixed=True)
    prepared = reuse._prepare(torch, ops, tensors, fixture, old["cv"])
    old_name, new_name, _ = _symbols(True)
    reuse._poison(torch, old["cv"])
    _launch_cv(ops, tensors, fixture, old["cv"], cores,
               is_mixed=True, candidate=False)
    torch.npu.synchronize()
    reuse._check_status(torch, old["cv"], invalid=False)
    old_oracle = reuse._merge_and_oracle(torch, ops, fixture, old["cv"],
                                          acceptance["fused_attention"])
    reuse._poison(torch, new["cv"])
    _launch_cv(ops, tensors, fixture, new["cv"], cores,
               is_mixed=True, candidate=True)
    torch.npu.synchronize()
    reuse._check_status(torch, new["cv"], invalid=False)
    reuse._assert_same_bits(torch, reuse._snapshot(old["cv"]), new["cv"],
                            "C16 graph fixture eager")
    new_oracle = reuse._merge_and_oracle(torch, ops, fixture, new["cv"],
                                          acceptance["fused_attention"])
    for name in ("output", "lse"):
        if not reuse._bitwise_identical(torch, old_oracle[name], new_oracle[name]):
            raise MixedOptimizationProbeError(f"C16 graph fixture merged {name} changed")
    stats = _check_c16_stats(torch, new["cv"], prepared["source0_leaders"],
                             require_clusters=True)
    faults = _error_propagation_cases(torch, ops, tensors, fixture,
                                      old, new, cores,
                                      prepared["source0_leaders"])
    graph = _graph_cv(torch, ops, tensors, fixture, old, new, cores,
                      is_mixed=True, expected_leaders=prepared["source0_leaders"],
                      require_clusters=True)
    result = {"case": spec.name, "status": "passed", "old_operator": old_name,
              "new_operator": new_name, "clusters": stats["eligible_clusters"],
              "input_sha256": fixture["input_sha256"], "oracle": "passed",
              "invalid_error_propagation": faults,
              "production_route": "standalone_direct_C16_below_host_selection_threshold",
              **graph}
    del tensors, old, new, fixture
    gc.collect()
    torch.npu.empty_cache()
    return result


def _print_case(row: dict) -> None:
    p = row["performance"]
    print("[oscar] PERF_MIXED_OPT " + json.dumps({
        "case": row["case"], "old_operator": row["old_operator"],
        "new_operator": row["new_operator"],
        "old_cv_ms": p["old"]["cv_ms"], "new_cv_ms": p["new"]["cv_ms"],
        "old_total_ms": p["old"]["total_ms"],
        "new_total_ms": p["new"]["total_ms"],
        "cv_ratio": p["ratios"]["cv_ms"],
        "total_ratio": p["ratios"]["total_ms"],
        "ratio_scope": p["ratio_scope"],
        "precision": row["precision"], "graph": (row["graph"] or {}).get(
            "graph_replay", "not_run"), "status": row["status"]},
        sort_keys=True), flush=True)


def probe(config_path: Path, acceptance_path: Path) -> dict:
    target = json.loads(config_path.read_text())
    acceptance = json.loads(acceptance_path.read_text())
    policy = acceptance.get("performance", {})
    if (acceptance.get("frozen_before_measurement") is not True or
            policy.get("warmup") != WARMUP or policy.get("repeats") != REPEATS or
            policy.get("statistic") != "median" or policy.get("max_latency_ratio") != 1.0):
        raise MixedOptimizationProbeError("frozen precision and strict 2+5 median policy required")
    from .probe_native_current_fia import _select_target_npu, _target_geometry
    _select_target_npu(target)
    if _target_geometry(target) != (6, 1, 256) or target.get("rotation_method") != "hadamard":
        raise MixedOptimizationProbeError("target head geometry or rotation changed")
    import torch
    import torch_npu  # noqa: F401 - actual target NPU only
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.npu.set_device(0)
    from .build_ops import normalize_soc
    if (not torch.npu.is_available() or
            normalize_soc(torch.npu.get_device_name(0)) != target["soc_version"]):
        raise MixedOptimizationProbeError("selected target NPU/SOC is unavailable")
    from oscar_ascend.ops.loader import require_capabilities, validate_build_artifacts
    manifest_path = ROOT / "build/ascendc/build_manifest.json"
    manifest = validate_build_artifacts(manifest_path)
    require_capabilities({"prepare_attention_tasks_out", "rotate_out",
                          OLD_BASE, OLD_C4, NEW_BALANCED, NEW_C16,
                          "merge_lse_out", "status_guard"}, manifest_path)
    ops = torch.ops.oscar_ascend_ops
    cores = reuse._core_count(torch, target)
    if cores != 20:
        raise MixedOptimizationProbeError("target A2 geometry requires 20 Cube cores")
    device = torch.device("npu:0")
    rows = []
    cold, warm, long = shape_plan()
    first_cohort_hash = None
    balanced_graph = None
    for spec in (cold, warm, long):
        fixture = (mixed.make_mixed_fixture(torch, spec)
                   if spec.name != long.name else _long_fixture(torch, spec))
        if spec.name != long.name:
            if first_cohort_hash is None:
                first_cohort_hash = fixture["logical_decode_input_sha256"]
            elif fixture["logical_decode_input_sha256"] != first_cohort_hash:
                raise MixedOptimizationProbeError("cold/warm decode cohort changed")
        row = _run_case(torch, ops, fixture, device, cores, acceptance, is_mixed=True)
        rows.append(row)
        _print_case(row)
        if row["status"] != "passed":
            break
        if spec.name == warm.name:
            draft_fixture = {**fixture, "spec": replace(
                spec, name="mtp_first_draft_warm30k")}
            draft = _run_case(torch, ops, draft_fixture, device, cores,
                              acceptance, is_mixed=True, first_draft=True)
            rows.append(draft)
            _print_case(draft)
            if draft["status"] != "passed":
                break
        if spec.name in {warm.name, long.name}:
            for n, splits in ((128, 3), (16384, 1)):
                pure = mixed.derive_decode_fixture(torch, fixture,
                                                   padded_tokens=n, splits=splits)
                control = _run_case(torch, ops, pure, device, cores,
                                    acceptance, is_mixed=False)
                rows.append(control)
                _print_case(control)
                if control["status"] != "passed":
                    break
                del pure
            if rows[-1]["status"] != "passed":
                break
            if spec.name == warm.name:
                balanced_graph = _balanced_graph_case(
                    torch, ops, fixture, device, cores, acceptance)
        del fixture
        gc.collect()
    graph_c16 = (_c16_graph_case(torch, ops, device, cores, acceptance)
                 if len(rows) == 8 and all(row["status"] == "passed" for row in rows)
                 else None)
    compact = next((row for row in rows if row["case"] ==
                    "decode_from_mixed_history_30k_n128_s3"), None)
    status = "passed" if (len(rows) == 8 and all(
        row["status"] == "passed" and row["precision"] == "passed" for row in rows)
        and compact and compact["graph"] and compact["graph"]["graph_replay"] == "reused"
        and balanced_graph and balanced_graph["graph_replay"] == "passed"
        and graph_c16 and graph_c16["graph_replay"] == "passed") else "failed"
    return {"status": status, "precision": "passed" if all(
                row["precision"] == "passed" for row in rows) else "failed",
            "graph_capture": "passed" if status == "passed" else "not_established",
            "graph_replay": "passed" if status == "passed" else "not_established",
            "performance": "passed" if status == "passed" else "failed",
            "full_model_quality": "not_run", "full_model_performance": "not_run",
            "device": str(device), "device_name": torch.npu.get_device_name(0),
            "physical_devices": target["devices"],
            "artifact_signature": manifest["signature"],
            "artifact_sha256": manifest["sha256"],
            "logical_decode_input_sha256": first_cohort_hash,
            "cases": rows, "balanced_graph": balanced_graph,
            "c16_active_graph": graph_c16,
            "first_error": next((f"{row['case']}: CV/total ratio exceeds 1.0"
                                 for row in rows if row["status"] != "passed"),
                                None if status == "passed" else "required evidence missing")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/target.json")
    parser.add_argument("--acceptance", type=Path,
                        default=ROOT / "configs/acceptance.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.name in {"mixed-optimization.json", "status.json"}:
        parser.error("#136/#151: report path must differ from phase state JSON")
    try:
        report = probe(args.config, args.acceptance)
        atomic_json(args.output, report)
        print("[oscar] PERF_MIXED_OPT_RESULT " + json.dumps({
            "status": report["status"], "precision": report["precision"],
            "graph_capture": report["graph_capture"],
            "graph_replay": report["graph_replay"],
            "performance": report["performance"],
            "cases": len(report["cases"]), "first_error": report["first_error"],
            "report": str(args.output)}, sort_keys=True), flush=True)
        return 0 if report["status"] == "passed" else 2
    except Exception as error:
        atomic_json(args.output, {"status": "failed", "precision": "not_established",
                                  "graph_capture": "not_established",
                                  "graph_replay": "not_established",
                                  "performance": "not_established",
                                  "error_type": type(error).__name__,
                                  "first_error": str(error),
                                  "device_completion": "not_established"})
        traceback.print_exc()
        print("[oscar] PERF_MIXED_OPT_RESULT " + json.dumps({
            "status": "failed", "first_error": str(error),
            "report": str(args.output)}, sort_keys=True), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
