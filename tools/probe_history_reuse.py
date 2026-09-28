# Archive #126/#129/#140-145 and startup D.4: fe0 is the numerical baseline.
# D.4 four questions: (1) this checks fused history dequant+FIA only;
# (2) the failed implementation restored full 32K history for 6.5s/device;
# (3) the experimental C4 op shares one bounded INT2 tile only among four
# identical-domain source0 query groups, retaining fe0 FP32/QK/PV/order and
# exact sink/recent/current sources; (4) only same-input NPU numerical and
# Event A/B measurements can establish gain, never loop counts or CPU tests.
"""Fail-closed real-NPU fe0-versus-C4 diagnostic; never switches production."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback

from .phase import atomic_json

ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "fe0e925e7ef78bfb64217a300031502fc4a7b7bc"
CANDIDATE_OP = "attention_cv_cluster4_out"
BLOCK_TOKENS, SINK, RECENT, SPECULATIVE, PREFIX = 512, 64, 256, 3, 64
SEED = 46817
STATS_FIELDS = ("eligible_clusters", "grouped_leaders", "independent_leaders",
                "shared_history_kv_tiles", "avoided_history_kv_loads",
                "independent_history_kv_tiles", "original_schedule_skips",
                "cluster_anchors")


class HistoryReuseProbeError(RuntimeError):
    pass


class BaselineNondeterministic(HistoryReuseProbeError):
    """Identical fe0 inputs differed; candidate bitwise parity cannot be judged."""


@dataclass(frozen=True)
class Shape:
    name: str
    qlens: tuple[int, ...]
    contexts: tuple[int, ...]
    dim: int
    kv_heads: int
    splits: int
    expect_clusters: bool
    speed_gate: bool = True
    corrupt_metadata: bool = False
    corrupt_dead_tail: bool = False
    corrupt_qr: bool = False
    recent_tokens: int = RECENT
    corrupt_position: int = 127
    corrupt_qr_token: int = 84
    required_clusters: int | None = None

    @property
    def heads(self) -> int:
        return self.kv_heads * 6

    def validate(self) -> None:
        if (len(self.qlens) != len(self.contexts) or not self.qlens or
                any(type(x) is not int or x <= 0 for x in self.qlens) or
                any(type(x) is not int or x < 0 for x in self.contexts) or
                self.dim not in (64, 128, 256) or self.kv_heads <= 0 or
                self.splits <= 0 or self.splits > 32 or self.recent_tokens <= 0):
            raise HistoryReuseProbeError(f"invalid diagnostic shape {self.name}")


CASES = (
    Shape("mature_20k", (4096,), (20000,), 256, 1, 1, True),
    Shape("frontier_511", (85,), (511,), 64, 1, 1, False),
    Shape("mixed_hkv2_s3", (33, 17, 49), (511, 769, 1025), 128, 2, 3, False),
    Shape("mixed_unaligned_mature_s3", (3, 385), (641, 769),
          128, 2, 3, True, speed_gate=False, required_clusters=2),
    Shape("decode32", (4,) * 32, (20000, 23000, 27000, 30000) * 8,
          256, 1, 1, False),
    Shape("invalid_live_metadata", (85,), (511,), 64, 1, 1, False,
          speed_gate=False, corrupt_metadata=True),
    Shape("invalid_dead_tail", (85,), (511,), 64, 1, 1, False,
          speed_gate=False, corrupt_dead_tail=True),
    Shape("mature_bad_metadata", (168,), (641,), 64, 1, 1, True,
          speed_gate=False, corrupt_metadata=True, recent_tokens=32,
          corrupt_position=100),
    Shape("mature_bad_query", (168,), (641,), 64, 1, 1, True,
          speed_gate=False, corrupt_qr=True, recent_tokens=32,
          corrupt_qr_token=105),
)


def candidate_workspace_per_core(dim: int) -> int:
    if dim not in (64, 128, 256):
        raise ValueError("unsupported C4 head dimension")
    # Mirror the independent test-only C++ helper; binding rejects undersize.
    return (1664 * dim + 33792) * 4


def baseline_workspace_per_core(dim: int) -> int:
    return ((2 * 128 + 2 * 256) * dim + 128 * 256) * 4


def assert_fe0_kernel_unchanged(root: Path = ROOT) -> dict:
    """Require the original fe0 kernel body and shared math header bytewise."""
    hashes = {}
    for relative in ("csrc/kernels/attention_cv.cpp", "csrc/kernels/oscar_common.h"):
        expected = subprocess.run(["git", "show", f"{BASELINE_COMMIT}:{relative}"],
            cwd=root, capture_output=True, check=True).stdout
        actual = (root / relative).read_bytes()
        if actual != expected:
            raise HistoryReuseProbeError(f"fe0 baseline source changed: {relative}")
        hashes[relative] = hashlib.sha256(actual).hexdigest()
    return hashes


def _hadamard(torch, dim: int):
    matrix = torch.ones((1, 1), dtype=torch.float32)
    while matrix.shape[0] < dim:
        matrix = torch.cat((torch.cat((matrix, matrix), 1),
                            torch.cat((matrix, -matrix), 1)), 0) / math.sqrt(2)
    return matrix.contiguous()


def _samples(length: int) -> tuple[int, ...]:
    if length <= 128:
        return tuple(range(length))
    points = (0, 1, 20, 21, 22, 41, 42, 62, 63, 83, 84, 125, 126,
              255, 256, 257, 511, 512, 1023, length // 2, length - 1)
    return tuple(sorted({point for point in points if point < length}))


def _fingerprint(named: dict) -> str:
    # Reuse the tested CPU byte-hash helper, not an output-derived signature.
    from .probe_cv_hotshape import _tensor_hash
    return _tensor_hash(named)


def make_fixture(torch, shape: Shape) -> dict:
    """Independent CPU oracle input; production never imports this module."""
    from oscar_ascend.ops.reference import attention, decode_kv, encode_kv
    shape.validate()
    qlens, contexts, dim, hk, hq = shape.qlens, shape.contexts, shape.dim, shape.kv_heads, shape.heads
    total = sum(qlens)
    generator = torch.Generator(device="cpu").manual_seed(SEED + total + dim + hk)
    q = torch.randn((total, hq, dim), generator=generator).to(torch.bfloat16)
    ck = torch.randn((total, hk, dim), generator=generator).to(torch.bfloat16)
    cv = torch.randn((total, hk, dim), generator=generator).to(torch.bfloat16)
    rk = _hadamard(torch, dim)
    rv = rk.flip(1).contiguous()
    qr = (q.float() @ rk).contiguous()
    budgets = [math.ceil((context + length) / BLOCK_TOKENS)
               for context, length in zip(contexts, qlens)]
    blocks = sum(budgets)
    pages = list(range(blocks))
    random.Random(SEED + total + dim).shuffle(pages)
    assignments = []
    offset = 0
    for budget in budgets:
        assignments.append(tuple(pages[offset:offset + budget]))
        offset += budget
    slot_bytes = dim // 2 + 8
    stride = BLOCK_TOKENS * hk * slot_bytes
    raw = torch.full((PREFIX + blocks * stride,), 0xA5, dtype=torch.uint8)
    raw_pages = raw[PREFIX:].view(blocks, BLOCK_TOKENS, hk, slot_bytes)
    recent = shape.recent_tokens
    window_rows = SINK + recent + SPECULATIVE
    wk = torch.full((blocks, window_rows, hk, dim), float("nan"), dtype=torch.bfloat16)
    wv = torch.full_like(wk, float("nan"))
    tags = torch.full((blocks, window_rows), -1, dtype=torch.int64)
    columns = math.ceil(max(context + length for context, length in zip(contexts, qlens)) / 128)
    table = torch.full((len(qlens), columns), -1, dtype=torch.int32)
    slots = torch.empty(total, dtype=torch.int64)
    starts = [0]
    lens = []
    expected = {}
    query_begin = 0
    scale = dim ** -0.5
    for request, (length, context, request_pages) in enumerate(zip(qlens, contexts, assignments)):
        old_k = torch.randn((context, hk, dim), generator=generator).to(torch.bfloat16)
        old_v = torch.randn((context, hk, dim), generator=generator).to(torch.bfloat16)
        packed = encode_kv(old_k.float() @ rk, old_v.float() @ rv)
        restored_k, restored_v = decode_kv(packed, dim)
        restored_k, restored_v = restored_k @ rk.T, restored_v @ rv.T
        for logical_page, physical in enumerate(request_pages):
            first = logical_page * BLOCK_TOKENS
            count = min(BLOCK_TOKENS, context - first)
            if count > 0:
                raw_pages[physical, :count] = packed[first:first + count]
            for quarter in range(BLOCK_TOKENS // 128):
                column = logical_page * (BLOCK_TOKENS // 128) + quarter
                if column < columns:
                    table[request, column] = physical * (BLOCK_TOKENS // 128) + quarter
        for position in sorted(set(range(min(SINK, context))) |
                               set(range(max(0, context - recent), context))):
            physical = request_pages[position // BLOCK_TOKENS]
            in_page = position % BLOCK_TOKENS
            row = position if position < SINK else SINK + in_page % (recent + SPECULATIVE)
            wk[physical, row] = old_k[position]
            wv[physical, row] = old_v[position]
            tags[physical, row] = in_page
        for local in range(length):
            position = context + local
            physical = request_pages[position // BLOCK_TOKENS]
            slots[query_begin + local] = physical * BLOCK_TOKENS + position % BLOCK_TOKENS
        actual_k, actual_v = old_k.float(), old_v.float()
        last_cut = None
        selected_k = selected_v = None
        for local in _samples(length):
            cut = min(context, max(SINK, context + local + 1 - recent))
            if cut != last_cut:
                selected_k, selected_v = actual_k.clone(), actual_v.clone()
                selected_k[SINK:cut] = restored_k[SINK:cut]
                selected_v[SINK:cut] = restored_v[SINK:cut]
                last_cut = cut
            index = query_begin + local
            keys = torch.cat((selected_k, ck[query_begin:index + 1].float()))
            values = torch.cat((selected_v, cv[query_begin:index + 1].float()))
            result = attention(q[index:index + 1], keys, values,
                               scale=scale, causal=False)
            expected[index] = (result.output[0], result.lse[0])
        query_begin += length
        starts.append(query_begin)
        lens.append(context + length)
    if sorted(page for request_pages in assignments for page in request_pages) != list(range(blocks)):
        raise HistoryReuseProbeError("fixture physical page assignments overlap")
    if shape.corrupt_metadata or shape.corrupt_dead_tail:
        request = 0
        position = (shape.corrupt_position if shape.corrupt_metadata else
                    contexts[request] + qlens[request] + 1)
        if shape.corrupt_metadata and not 0 <= position < contexts[request]:
            raise HistoryReuseProbeError("invalid metadata case does not hit live old history")
        if position // BLOCK_TOKENS >= len(assignments[request]):
            raise HistoryReuseProbeError("invalid-tail fixture lacks a physical page")
        physical = assignments[request][position // BLOCK_TOKENS]
        in_page = position % BLOCK_TOKENS
        # Corrupt K scale. A live zero is rejected; a never-addressed future
        # slot must leave every valid output unchanged.
        raw_pages[physical, in_page, hk - 1, dim // 4:dim // 4 + 2] = torch.tensor(
            (0, 0), dtype=torch.uint8)
    if shape.corrupt_qr:
        # The selected token is inside a mature full-21-token group.
        if total <= shape.corrupt_qr_token:
            raise HistoryReuseProbeError("mature QR fault fixture is too short")
        qr[shape.corrupt_qr_token, 0, 0] = float("nan")
    tensors = {"q": q, "qr": qr, "ck": ck, "cv": cv, "rv": rv,
               "raw": raw, "table": table, "wk": wk, "wv": wv, "tags": tags,
               "starts": torch.tensor(starts, dtype=torch.int32),
               "lens": torch.tensor(lens, dtype=torch.int32), "slots": slots}
    return {"spec": shape, "cpu": tensors, "expected": expected,
            "input_sha256": _fingerprint(tensors), "blocks": blocks,
            "stride": stride, "tokens": total, "page_assignments": assignments,
            "scale": scale}


def _active_device(target: dict):
    if target.get("devices") != [0, 1, 2, 3] or target.get("soc_version") != "ascend910b4":
        raise HistoryReuseProbeError("explicit target devices 0,1,2,3 and ascend910b4 are required")
    selection = "0,1,2,3"
    inherited = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if inherited is not None and inherited != selection:
        raise HistoryReuseProbeError(f"physical device selection {inherited!r} differs from target")
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = selection


def _core_count(torch, target: dict) -> int:
    from .probe_cv_hotshape import _actual_cores
    return _actual_cores(torch, target)


def _allocate(torch, fixture: dict, device, cores: int, *, candidate: bool):
    spec: Shape = fixture["spec"]
    n, h, hk, d, splits = fixture["tokens"], spec.heads, spec.kv_heads, spec.dim, spec.splits
    tasks = torch.empty((n * hk * 3 * splits, 16), dtype=torch.int64, device=device)
    positions = torch.empty(n, dtype=torch.int64, device=device)
    partial = torch.empty((n, h, 3 * splits, d), dtype=torch.float32, device=device)
    lse = torch.empty((n, h, 3 * splits), dtype=torch.float32, device=device)
    status = torch.empty((n * hk * 3 * splits, 2), dtype=torch.int32, device=device)
    per_core = candidate_workspace_per_core(d) if candidate else baseline_workspace_per_core(d)
    workspace = torch.empty(cores * per_core, dtype=torch.uint8, device=device)
    stats = None
    if candidate:
        count = cores * len(STATS_FIELDS)
        storage = torch.empty(count + 8, dtype=torch.int64, device=device)
        offset = ((-storage.data_ptr()) % 64) // 8
        stats = storage[offset:offset + count].view(cores, len(STATS_FIELDS))
        if stats.data_ptr() % 64 or not stats.is_contiguous():
            raise HistoryReuseProbeError("cluster_stats must be 64-byte aligned and contiguous")
        stats.zero_()
    return {"tasks": tasks, "positions": positions, "partial": partial, "lse": lse,
            "status": status, "workspace": workspace, "cluster_stats": stats}


def _prepare(torch, ops, tensors: dict, fixture: dict, buffers: dict) -> dict:
    spec: Shape = fixture["spec"]
    ops.prepare_attention_tasks_out(tensors["starts"], tensors["lens"], tensors["slots"],
        buffers["tasks"], buffers["positions"], spec.heads, spec.kv_heads,
        SINK, spec.recent_tokens, spec.splits)
    torch.npu.synchronize()
    expected = torch.cat([torch.arange(context, context + length)
                          for context, length in zip(spec.contexts, spec.qlens)])
    torch.testing.assert_close(buffers["positions"].cpu(), expected, atol=0, rtol=0)
    task_cpu = buffers["tasks"].cpu()
    source0 = task_cpu.view(fixture["tokens"], spec.kv_heads, 3, spec.splits, 16)[:, :, 0]
    leaders = source0[..., 1] > 0
    unique_ends = torch.unique(source0[..., 4][leaders])
    return {"task_sha256": _fingerprint({"tasks": task_cpu}),
            "source0_leaders": int(leaders.sum()),
            "source0_distinct_kvend": int(unique_ends.numel())}


def _poison(torch, buffers: dict):
    buffers["partial"].fill_(float("nan"))
    buffers["lse"].fill_(float("nan"))
    buffers["status"].fill_(-99)
    buffers["workspace"].view(torch.float32).fill_(float("nan"))
    if buffers["cluster_stats"] is not None:
        buffers["cluster_stats"].zero_()


def _launch(ops, tensors: dict, fixture: dict, buffers: dict, cores: int, *, candidate: bool):
    spec: Shape = fixture["spec"]
    args = (tensors["q"], tensors["qr"], tensors["ck"], tensors["cv"], tensors["rv"],
        tensors["raw"], tensors["table"], tensors["wk"], tensors["wv"], tensors["tags"],
        buffers["tasks"], buffers["partial"], buffers["lse"], buffers["status"],
        buffers["workspace"])
    attrs = (BLOCK_TOKENS, fixture["blocks"], PREFIX, fixture["stride"],
             SINK, spec.recent_tokens, SPECULATIVE, spec.splits, fixture["scale"], cores)
    if candidate:
        getattr(ops, CANDIDATE_OP)(*args, buffers["cluster_stats"], *attrs)
    else:
        ops.attention_cv_out(*args, *attrs)


def _bitwise_identical(torch, left, right) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.dtype == torch.float32:
        return bool(torch.equal(left.view(torch.int32), right.view(torch.int32)))
    return bool(torch.equal(left, right))


def _bitwise_mismatch_count(torch, left, right) -> int:
    if left.shape != right.shape or left.dtype != right.dtype:
        raise HistoryReuseProbeError("bitwise comparison tensor geometry differs")
    bits_left = left.view(torch.int32) if left.dtype == torch.float32 else left
    bits_right = right.view(torch.int32) if right.dtype == torch.float32 else right
    return int(torch.count_nonzero(bits_left != bits_right))


def _assert_same_bits(torch, expected: dict, actual: dict, label: str) -> None:
    for name in ("partial", "lse", "status"):
        if not _bitwise_identical(torch, expected[name], actual[name]):
            error = BaselineNondeterministic if label.startswith("fe0 repeat") else HistoryReuseProbeError
            raise error(f"{label} differs bitwise in {name}: "
                        f"mismatched_elements={_bitwise_mismatch_count(torch, expected[name], actual[name])}")


def _snapshot(buffers: dict) -> dict:
    return {name: buffers[name].clone() for name in ("partial", "lse", "status")}


def _check_status(torch, buffers: dict, *, invalid: bool):
    status = buffers["status"].cpu()
    if invalid:
        if not bool((status != 0).any()) or not bool(((status == 0) | (status == 2) | (status == 3)).all()):
            raise HistoryReuseProbeError("invalid live metadata did not preserve fe0 status domain")
    elif not bool((status == 0).all()):
        raise HistoryReuseProbeError("valid CV case produced nonzero or unwritten status")
    if not invalid:
        if bool(torch.isnan(buffers["lse"]).any()) or not bool(torch.isfinite(buffers["partial"]).all()):
            raise HistoryReuseProbeError("valid CV case left NaN LSE or non-finite partial")


def _source0_leader_status(fixture: dict, buffers: dict) -> dict:
    spec: Shape = fixture["spec"]
    n, hk, splits = fixture["tokens"], spec.kv_heads, spec.splits
    tasks = buffers["tasks"].cpu().view(n, hk, 3, splits, 16)
    status = buffers["status"].cpu().view(n, hk, 3, splits, 2)
    leaders = []
    nonzero = []
    for token in range(n):
        for head in range(hk):
            for split in range(splits):
                if int(tasks[token, head, 0, split, 1]) <= 0:
                    continue
                codes = [int(x) for x in status[token, head, 0, split]]
                record = {"token": token, "kv_head": head, "split": split,
                          "qcount": int(tasks[token, head, 0, split, 1]),
                          "status_aiv": codes}
                leaders.append(record)
                if any(codes):
                    nonzero.append(record)
    return {"leaders": len(leaders), "nonzero_leaders": nonzero}


def _disable_terminal_source0_split(task_cpu, spec: Shape) -> int:
    """Counterfactual: remove only terminal split's `kvend=context` flag."""
    n = sum(spec.qlens)
    rows = task_cpu.view(n, spec.kv_heads, 3, spec.splits, 16)[:, :, 0, -1]
    qcount, begin, end, context = (rows[..., column] for column in (1, 3, 4, 8))
    selected = (qcount > 0) & (end == context) & (end > begin + 1)
    changed = int(selected.sum())
    end[selected] -= 1
    return changed


def _counterfactual_no_terminal_share(torch, ops, tensors: dict, fixture: dict,
                                      baseline: dict, candidate: dict,
                                      cores: int, expected_leaders: int) -> dict:
    original = baseline["tasks"].cpu()
    changed = original.clone()
    changed_leaders = _disable_terminal_source0_split(changed, fixture["spec"])
    if changed_leaders < 4:
        raise HistoryReuseProbeError("mixed mature fixture has too few terminal split leaders")
    baseline["tasks"].copy_(changed.to(baseline["tasks"].device))
    _poison(torch, baseline)
    _launch(ops, tensors, fixture, baseline, cores, candidate=False)
    torch.npu.synchronize()
    _check_status(torch, baseline, invalid=False)
    changed_fe0 = _snapshot(baseline)
    merged_fe0 = _merge_only(torch, ops, fixture, baseline)
    _poison(torch, candidate)
    _launch(ops, tensors, fixture, candidate, cores, candidate=True)
    torch.npu.synchronize()
    _check_status(torch, candidate, invalid=False)
    _assert_same_bits(torch, changed_fe0, candidate, "nonterminal C1 versus fe0")
    merged_candidate = _merge_only(torch, ops, fixture, candidate)
    for name in ("output", "lse"):
        if not _bitwise_identical(torch, merged_fe0[name], merged_candidate[name]):
            raise HistoryReuseProbeError(f"nonterminal C1 merged {name} differs from fe0")
    stats = _check_cluster_stats(candidate["cluster_stats"].cpu(),
        expect_clusters=False, expected_source0_leaders=expected_leaders)
    baseline["tasks"].copy_(original.to(baseline["tasks"].device))
    return {"terminal_leaders_modified": changed_leaders,
            "clusters_after_terminal_disabled": stats["eligible_clusters"],
            "fe0_vs_c4": "bitwise_passed", "statuses": "all_zero",
            "oracle_scope": "counterfactual_task_range_fe0_eager_reference"}


def _merge_only(torch, ops, fixture: dict, buffers: dict) -> dict:
    spec: Shape = fixture["spec"]
    n, h, d, s = fixture["tokens"], spec.heads, spec.dim, 3 * spec.splits
    device = buffers["partial"].device
    out = torch.empty((n * h, d), dtype=torch.float32, device=device)
    out_lse = torch.empty((n * h,), dtype=torch.float32, device=device)
    merge_status = torch.full((n * h,), -99, dtype=torch.int32, device=device)
    ops.merge_lse_out(buffers["partial"].view(n * h, s, d),
        buffers["lse"].view(n * h, s), out, out_lse, merge_status)
    torch.npu.synchronize()
    if not bool((merge_status.cpu() == 0).all()):
        raise HistoryReuseProbeError("merge status did not complete")
    return {"output": out, "lse": out_lse}


def _merge_and_oracle(torch, ops, fixture: dict, buffers: dict, tolerance: dict) -> dict:
    merged = _merge_only(torch, ops, fixture, buffers)
    spec: Shape = fixture["spec"]
    n, h, d = fixture["tokens"], spec.heads, spec.dim
    out, out_lse = merged["output"], merged["lse"]
    indexes = sorted(fixture["expected"])
    observed = out.view(n, h, d)[indexes].cpu()
    observed_lse = out_lse.view(n, h)[indexes].cpu()
    expected = torch.stack([fixture["expected"][index][0] for index in indexes])
    expected_lse = torch.stack([fixture["expected"][index][1] for index in indexes])
    torch.testing.assert_close(observed, expected, **tolerance)
    torch.testing.assert_close(observed_lse, expected_lse, **tolerance)
    return {**merged, "sampled_queries": len(indexes),
            "max_output_abs": float((observed - expected).abs().max()),
            "max_lse_abs": float((observed_lse - expected_lse).abs().max())}


def _check_cluster_stats(stats_cpu, *, expect_clusters: bool,
                         expected_source0_leaders: int | None = None) -> dict:
    if stats_cpu.ndim != 2 or stats_cpu.shape[1] != len(STATS_FIELDS):
        raise HistoryReuseProbeError("candidate cluster_stats shape is invalid")
    # Archive #129/#148/#149: pass one charges each deferred member's skip to
    # the core owning its query tile, while pass two charges the four grouped
    # leaders to the single core owning the remapped bucket. skips == grouped
    # is therefore a global identity only; per core just the same-site
    # identities hold, matching the CPU-debug totals oracle.
    for core, row in enumerate(stats_cpu.tolist()):
        if row[1] != 4 * row[0] or row[4] != 3 * row[3] or row[7] != row[0]:
            raise HistoryReuseProbeError(f"candidate cluster owner {core} counters are inconsistent")
    values = [int(x) for x in stats_cpu.sum(0).tolist()]
    if any(value < 0 for value in values):
        raise HistoryReuseProbeError("candidate cluster_stats contains negative counts")
    if values[1] != 4 * values[0] or values[4] != 3 * values[3] or \
            values[6] != values[1] or values[7] != values[0]:
        raise HistoryReuseProbeError("candidate cluster ownership counters are inconsistent")
    if expect_clusters and (values[0] <= 0 or values[3] <= 0):
        raise HistoryReuseProbeError("mature long prefill did not activate C4 history reuse")
    if not expect_clusters and (values[0] or values[1] or values[3] or values[4]):
        raise HistoryReuseProbeError("frontier/decode/ineligible case unexpectedly activated C4")
    if not expect_clusters and (values[2] <= 0 or values[5] <= 0):
        raise HistoryReuseProbeError("ineligible case recorded no fe0 C1 history work")
    if expected_source0_leaders is not None and values[1] + values[2] != expected_source0_leaders:
        raise HistoryReuseProbeError("candidate lost or duplicated source0 leader ownership")
    return dict(zip(STATS_FIELDS, values))


def _timed_pair(torch, ops, tensors, fixture, baseline, candidate_buffers, cores, *,
                warmup: int, repeats: int, reference_bits: dict) -> tuple[dict, dict]:
    durations: dict[str, list[float]] = {"fe0": [], "candidate": []}
    stream = torch.npu.current_stream()
    for index in range(warmup + repeats):
        # Both kernels see a warm cache. Reverse order on alternating measured
        # repetitions so a later invocation is not consistently favored.
        order = (False, True) if index < warmup or (index - warmup) % 2 == 0 else (True, False)
        for is_candidate in order:
            name = "candidate" if is_candidate else "fe0"
            buffers = candidate_buffers if is_candidate else baseline
            _poison(torch, buffers)
            if index >= warmup:
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                start.record(stream)
            _launch(ops, tensors, fixture, buffers, cores, candidate=is_candidate)
            if index >= warmup:
                end.record(stream)
                end.synchronize()
                duration = float(start.elapsed_time(end))
                if not math.isfinite(duration) or duration <= 0:
                    raise HistoryReuseProbeError("invalid NPU Event duration")
                durations[name].append(duration)
            else:
                torch.npu.synchronize()
            _check_status(torch, buffers, invalid=(fixture["spec"].corrupt_metadata or
                                                  fixture["spec"].corrupt_qr))
            if index >= warmup:
                _assert_same_bits(torch, reference_bits, buffers, f"{name} timed repeat")
    evidence = {name: {"device_event_ms": values,
                       "median_ms": statistics.median(values), "warmup": warmup,
                       "repeats": repeats, "order": "alternating_AB_BA",
                       "scope": "operator_only_NPU_Event_no_kernel_profiler"}
                for name, values in durations.items()}
    return evidence["fe0"], evidence["candidate"]


def run_case(torch, ops, spec: Shape, device, cores: int, acceptance: dict) -> dict:
    started = time.monotonic()
    fixture = make_fixture(torch, spec)
    build_seconds = time.monotonic() - started
    tensors = {name: tensor.to(device) for name, tensor in fixture["cpu"].items()}
    baseline = _allocate(torch, fixture, device, cores, candidate=False)
    candidate = _allocate(torch, fixture, device, cores, candidate=True)
    preparation = _prepare(torch, ops, tensors, fixture, baseline)
    if spec.name == "frontier_511" and preparation["source0_distinct_kvend"] < 2:
        raise HistoryReuseProbeError("frontier fixture did not produce different history kvend values")
    # Both kernels read identical task bytes from the same NPU address.
    candidate["tasks"] = baseline["tasks"]
    candidate["positions"] = baseline["positions"]
    _poison(torch, baseline)
    _launch(ops, tensors, fixture, baseline, cores, candidate=False)
    torch.npu.synchronize()
    invalid = spec.corrupt_metadata or spec.corrupt_qr
    _check_status(torch, baseline, invalid=invalid)
    reference_bits = _snapshot(baseline)
    baseline_first_merge = None if invalid else _merge_and_oracle(
        torch, ops, fixture, baseline, acceptance["fused_attention"])
    _poison(torch, baseline)
    _launch(ops, tensors, fixture, baseline, cores, candidate=False)
    torch.npu.synchronize()
    _check_status(torch, baseline, invalid=invalid)
    _assert_same_bits(torch, reference_bits, baseline, "fe0 repeat")
    baseline_merge = None if invalid else _merge_and_oracle(
        torch, ops, fixture, baseline, acceptance["fused_attention"])
    if baseline_first_merge is not None and baseline_merge is not None:
        for name in ("output", "lse"):
            if not _bitwise_identical(torch, baseline_first_merge[name], baseline_merge[name]):
                raise BaselineNondeterministic(f"fe0 repeat differs bitwise in merged {name}: "
                    f"mismatched_elements={_bitwise_mismatch_count(torch, baseline_first_merge[name], baseline_merge[name])}")
    _poison(torch, candidate)
    _launch(ops, tensors, fixture, candidate, cores, candidate=True)
    torch.npu.synchronize()
    _check_status(torch, candidate, invalid=invalid)
    _assert_same_bits(torch, reference_bits, candidate, "C4 versus fe0")
    candidate_merge = None if invalid else _merge_and_oracle(
        torch, ops, fixture, candidate, acceptance["fused_attention"])
    if baseline_merge is not None and candidate_merge is not None:
        for name in ("output", "lse"):
            if not _bitwise_identical(torch, baseline_merge[name], candidate_merge[name]):
                raise HistoryReuseProbeError(f"C4 merged {name} differs bitwise from fe0")
    stats = _check_cluster_stats(candidate["cluster_stats"].cpu(),
        expect_clusters=spec.expect_clusters,
        expected_source0_leaders=preparation["source0_leaders"])
    if spec.required_clusters is not None and stats["eligible_clusters"] != spec.required_clusters:
        raise HistoryReuseProbeError(
            f"{spec.name} expected {spec.required_clusters} terminal-split C4 clusters, "
            f"got {stats['eligible_clusters']}")
    row = {"status": "passed", "case": spec.name, "input_sha256": fixture["input_sha256"],
           "task_sha256": preparation["task_sha256"], "shape": {"qlens": spec.qlens,
           "contexts": spec.contexts, "head_dim": spec.dim, "kv_heads": spec.kv_heads,
           "splits": spec.splits, "cube_cores": cores,
           "recent_tokens": spec.recent_tokens},
           "source0_distinct_kvend": preparation["source0_distinct_kvend"],
           "fixture_build_seconds": build_seconds,
           "fe0_repeatability": "bitwise_passed", "fe0_vs_c4": "bitwise_passed",
           "frozen_oracle": "passed" if baseline_merge is not None else "invalid_input_not_applicable",
           "baseline_oracle": ({key: value for key, value in baseline_merge.items()
                                if key not in ("output", "lse")} if baseline_merge else None),
           "candidate_oracle": ({key: value for key, value in candidate_merge.items()
                                 if key not in ("output", "lse")} if candidate_merge else None),
           "cluster_stats": stats,
           "source0_leader_status": _source0_leader_status(fixture, baseline),
           "graph_capture": "not_run", "graph_replay": "not_run"}
    if spec.name == "mixed_unaligned_mature_s3":
        row["nonterminal_split_counterfactual"] = _counterfactual_no_terminal_share(
            torch, ops, tensors, fixture, baseline, candidate, cores,
            preparation["source0_leaders"])
    if spec.speed_gate:
        before, after = _timed_pair(torch, ops, tensors, fixture, baseline, candidate, cores,
            warmup=acceptance["performance"]["warmup"],
            repeats=acceptance["performance"]["repeats"], reference_bits=reference_bits)
        ratio = after["median_ms"] / before["median_ms"]
        row.update(baseline_timing=before, candidate_timing=after,
                   candidate_over_fe0=ratio,
                   speed_gate="passed" if ratio <= acceptance["performance"]["max_latency_ratio"]
                   else "failed")
    else:
        row.update(speed_gate="not_applicable_invalid_or_padding_case")
    del fixture, tensors, baseline, candidate
    gc.collect()
    torch.npu.empty_cache()
    return row


def graph_changed_inputs(torch, ops, device, cores: int, acceptance: dict) -> dict:
    """Capture both standalone ops, mutate buffers in place, then replay.

    The original graph input has a frozen dense oracle. For the deliberate
    changed-task variant, fe0 eager is the exact reference; its task range
    differs from the natural dense-attention oracle by construction.
    """
    spec = Shape("graph_changed_inputs", (4,), (511,), 64, 1, 1, False,
                 speed_gate=False)
    fixture = make_fixture(torch, spec)
    tensors = {name: tensor.to(device) for name, tensor in fixture["cpu"].items()}
    baseline = _allocate(torch, fixture, device, cores, candidate=False)
    candidate = _allocate(torch, fixture, device, cores, candidate=True)
    _prepare(torch, ops, tensors, fixture, baseline)
    candidate["tasks"] = baseline["tasks"]
    candidate["positions"] = baseline["positions"]
    _poison(torch, baseline)
    _launch(ops, tensors, fixture, baseline, cores, candidate=False)
    torch.npu.synchronize()
    _check_status(torch, baseline, invalid=False)
    _merge_and_oracle(torch, ops, fixture, baseline, acceptance["fused_attention"])
    original_bits = _snapshot(baseline)
    _poison(torch, candidate)
    _launch(ops, tensors, fixture, candidate, cores, candidate=True)
    torch.npu.synchronize()
    _check_status(torch, candidate, invalid=False)
    _assert_same_bits(torch, original_bits, candidate, "graph eager initial C4")
    graph_fe0 = torch.npu.NPUGraph()
    graph_c4 = torch.npu.NPUGraph()
    torch.npu.synchronize()
    with torch.npu.graph(graph_fe0, capture_error_mode="thread_local", auto_dispatch_capture=True):
        _launch(ops, tensors, fixture, baseline, cores, candidate=False)
    with torch.npu.graph(graph_c4, capture_error_mode="thread_local", auto_dispatch_capture=True):
        _launch(ops, tensors, fixture, candidate, cores, candidate=True)
    torch.npu.synchronize()
    _poison(torch, baseline)
    _poison(torch, candidate)
    graph_fe0.replay()
    graph_c4.replay()
    torch.npu.synchronize()
    _check_status(torch, baseline, invalid=False)
    _check_status(torch, candidate, invalid=False)
    _assert_same_bits(torch, original_bits, baseline, "fe0 initial graph replay")
    _assert_same_bits(torch, original_bits, candidate, "C4 initial graph replay")
    # Change both query values and a source0 history range while preserving
    # every captured pointer and tensor shape. The smaller kvend stays valid.
    tensors["q"].copy_(-tensors["q"])
    tensors["qr"].copy_(-tensors["qr"])
    changed_tasks = baseline["tasks"].cpu()
    chosen = next((row for row in changed_tasks
                   if int(row[7]) == 0 and int(row[1]) > 0 and
                   int(row[4] - row[3]) >= 32), None)
    if chosen is None:
        raise HistoryReuseProbeError("graph fixture has no live source0 range to mutate")
    chosen[4] -= 16
    baseline["tasks"].copy_(changed_tasks)
    _poison(torch, baseline)
    _poison(torch, candidate)
    graph_fe0.replay()
    graph_c4.replay()
    torch.npu.synchronize()
    _check_status(torch, baseline, invalid=False)
    _check_status(torch, candidate, invalid=False)
    changed_graph = _snapshot(baseline)
    _assert_same_bits(torch, changed_graph, candidate, "C4 changed-input graph replay")
    changed_merged_fe0 = _merge_only(torch, ops, fixture, baseline)
    changed_merged_c4 = _merge_only(torch, ops, fixture, candidate)
    for name in ("output", "lse"):
        if not _bitwise_identical(torch, changed_merged_fe0[name], changed_merged_c4[name]):
            raise HistoryReuseProbeError(f"C4 changed-input graph merged {name} differs from fe0")
    if all(_bitwise_identical(torch, original_bits[name], changed_graph[name])
           for name in ("partial", "lse")):
        raise HistoryReuseProbeError("graph replay ignored changed input buffers")
    _poison(torch, baseline)
    _launch(ops, tensors, fixture, baseline, cores, candidate=False)
    torch.npu.synchronize()
    _check_status(torch, baseline, invalid=False)
    _assert_same_bits(torch, changed_graph, baseline, "fe0 changed-input eager versus graph")
    eager_merged_fe0 = _merge_only(torch, ops, fixture, baseline)
    for name in ("output", "lse"):
        if not _bitwise_identical(torch, changed_merged_fe0[name], eager_merged_fe0[name]):
            raise HistoryReuseProbeError(f"fe0 changed-input eager merged {name} differs from graph")
    _poison(torch, candidate)
    _launch(ops, tensors, fixture, candidate, cores, candidate=True)
    torch.npu.synchronize()
    _check_status(torch, candidate, invalid=False)
    _assert_same_bits(torch, changed_graph, candidate, "C4 changed-input eager versus graph")
    eager_merged_c4 = _merge_only(torch, ops, fixture, candidate)
    for name in ("output", "lse"):
        if not _bitwise_identical(torch, changed_merged_fe0[name], eager_merged_c4[name]):
            raise HistoryReuseProbeError(f"C4 changed-input eager merged {name} differs from fe0 graph")
    stats = _check_cluster_stats(candidate["cluster_stats"].cpu(), expect_clusters=False)
    return {"case": spec.name, "status": "passed", "input_sha256": fixture["input_sha256"],
            "graph_capture": "passed", "graph_replay": "passed",
            "graph_scope": "standalone_operator_graph_not_full_TP4_service",
            "changed_query_and_task_same_addresses": True,
            "initial_frozen_oracle": "passed",
            "changed_input_reference": "fe0_eager_bitwise",
            "fe0_vs_c4": "bitwise_passed", "cluster_stats": stats,
            "speed_gate": "not_measured_graph_contract"}


def probe(config_path: Path, acceptance_path: Path) -> dict:
    target = json.loads(config_path.read_text())
    acceptance = json.loads(acceptance_path.read_text())
    _active_device(target)
    fe0_source = assert_fe0_kernel_unchanged()
    policy = acceptance.get("performance", {})
    if (acceptance.get("frozen_before_measurement") is not True or
            policy.get("warmup") != 2 or policy.get("repeats") != 5 or
            policy.get("statistic") != "median" or
            policy.get("max_latency_ratio") != 1.0):
        raise HistoryReuseProbeError("frozen precision and 2+5 median performance policy is required")
    import torch
    import torch_npu  # noqa: F401 - real NPU only
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.npu.set_device(0)
    from .build_ops import normalize_soc
    if not torch.npu.is_available() or normalize_soc(torch.npu.get_device_name(0)) != target["soc_version"]:
        raise HistoryReuseProbeError("selected target NPU/SOC is unavailable")
    from oscar_ascend.ops.loader import require_capabilities, validate_build_artifacts
    manifest_path = ROOT / "build/ascendc/build_manifest.json"
    manifest = validate_build_artifacts(manifest_path)
    require_capabilities({"prepare_attention_tasks_out", "attention_cv_out",
                          CANDIDATE_OP, "merge_lse_out"}, manifest_path)
    from .probe_native_current_fia import _target_geometry
    if _target_geometry(target) != (6, 1, 256):
        raise HistoryReuseProbeError("target TP head geometry differs from signed Hq6/Hkv1/D256 shape")
    device = torch.device("npu:0")
    cores = _core_count(torch, target)
    cases = []
    for shape in CASES:
        row = run_case(torch, torch.ops.oscar_ascend_ops, shape, device, cores, acceptance)
        cases.append(row)
        if row["case"] in ("mature_20k", "decode32"):
            print("[oscar] PERF_HISTORY_REUSE_A_B " + json.dumps({"case": row["case"],
                "accuracy": row["fe0_vs_c4"],
                "clusters": row["cluster_stats"]["eligible_clusters"],
                "baseline_ms": row["baseline_timing"]["median_ms"],
                "candidate_ms": row["candidate_timing"]["median_ms"],
                "ratio": row["candidate_over_fe0"], "gate": row["speed_gate"]},
                sort_keys=True), flush=True)
    print("[oscar] PERF_HISTORY_REUSE_ACCURACY " + json.dumps({
        "cases_bitwise": sum(row["fe0_vs_c4"] == "bitwise_passed" for row in cases),
        "cases_total": len(cases),
        "frozen_oracle_valid_cases": sum(row["frozen_oracle"] == "passed" for row in cases),
        "invalid_live_status": next(row["status"] for row in cases
                                    if row["case"] == "invalid_live_metadata"),
        "frontier_distinct_kvend": next(row["source0_distinct_kvend"] for row in cases
                                         if row["case"] == "frontier_511"),
        "unaligned_mature_clusters": next(row["cluster_stats"]["eligible_clusters"]
                                         for row in cases if row["case"] == "mixed_unaligned_mature_s3"),
        "nonterminal_clusters": next(row["nonterminal_split_counterfactual"]
                                      ["clusters_after_terminal_disabled"]
                                      for row in cases if row["case"] == "mixed_unaligned_mature_s3")},
        sort_keys=True), flush=True)
    graph = graph_changed_inputs(torch, torch.ops.oscar_ascend_ops, device, cores, acceptance)
    cases.append(graph)
    print("[oscar] PERF_HISTORY_REUSE_GRAPH " + json.dumps({"case": graph["case"],
        "accuracy": graph["fe0_vs_c4"], "oracle": graph["initial_frozen_oracle"],
        "graph_capture": graph["graph_capture"], "graph_replay": graph["graph_replay"],
        "clusters": graph["cluster_stats"]["eligible_clusters"]}, sort_keys=True), flush=True)
    failures = [row["case"] for row in cases if row["speed_gate"] == "failed"]
    return {"status": "passed" if not failures else "failed", "reason": (
            None if not failures else "candidate slower on " + ",".join(failures)),
            "scope": "experimental_operator_only_real_NPU; no_production_route_change",
            "candidate_timing_includes_counter_output": True,
            "default_route": "fe0",
            "production_promotion": "blocked_pending_full_model_quality_and_service_performance",
            "candidate_evaluation_allowed": not failures,
            "reference_commit": BASELINE_COMMIT, "fe0_source_sha256": fe0_source,
            "artifact_signature": manifest["signature"],
            "artifact_sha256": manifest["sha256"], "device": str(device),
            "device_name": torch.npu.get_device_name(0), "physical_devices": target["devices"],
            "cases": cases, "graph_capture": graph["graph_capture"],
            "graph_replay": graph["graph_replay"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/target.json")
    parser.add_argument("--acceptance", type=Path, default=ROOT / "configs/acceptance.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = probe(args.config, args.acceptance)
        atomic_json(args.output, report)
        print("[oscar] PERF_HISTORY_REUSE_RESULT " + json.dumps({"status": report["status"],
            "reason": report["reason"],
            "candidate_evaluation_allowed": report["candidate_evaluation_allowed"],
            "production_promotion": report["production_promotion"],
            "report": str(args.output)}, sort_keys=True), flush=True)
        return 0 if report["status"] == "passed" else 2
    except BaselineNondeterministic as exc:
        report = {"status": "needs_evidence", "default_route": "fe0",
                  "production_promotion": "blocked", "candidate_evaluation_allowed": False,
                  "reason": "fe0_same_input_not_bitwise_repeatable", "error": str(exc),
                  "device_completion": "observed_but_numerical_repeatability_failed"}
        atomic_json(args.output, report)
        print("[oscar] PERF_HISTORY_REUSE_RESULT " + json.dumps({"status": report["status"],
            "reason": report["reason"], "error": str(exc), "report": str(args.output)},
            sort_keys=True), flush=True)
        return 2
    except Exception as exc:
        report = {"status": "failed", "default_route": "fe0",
                  "production_promotion": "blocked", "candidate_evaluation_allowed": False,
                  "error_type": type(exc).__name__,
                  "error": str(exc), "device_completion": "not_established"}
        atomic_json(args.output, report)
        traceback.print_exc()
        print("[oscar] PERF_HISTORY_REUSE_RESULT " + json.dumps({"status": "failed", "error": str(exc),
            "report": str(args.output)}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
