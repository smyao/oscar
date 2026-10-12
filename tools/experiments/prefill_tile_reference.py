"""Bounded CPU experiment for PR 46774's continuation-prefill dtype order.

This is a numerical hypothesis test, not an Ascend implementation or device
acceptance. Only compressed history persists. Decoded history and exact current
K/V are constructed one tile at a time. The candidate path applies the PR's
rotated FP16 dequant output, FP32 inverse rotation, BF16 cast, and BF16 partial
output before an FP32 LSE merge. Sink, query-dependent frontier, recent and
current segments use the frozen FP32 reference in both arms.

Archive #126/#129/#148 and startup D.4: never materialize a full decoded
history in a serving path. The failed D.4 dequant took 6499.8-6655.1 ms per
card while FIA took about 18.7 ms. This CPU experiment makes no speed claim.
The #148 model-quality rollback means an operator tolerance result cannot
establish full-model quality or justify changing production dtype.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch

from oscar_ascend.ops import reference
from tools.probe_history_reuse import _hadamard


SEED = 46774
SINK = 64
RECENT = 256


@dataclass(frozen=True)
class Case:
    history: int
    query_length: int
    dim: int
    tile: int = 1024
    samples: int = 21
    query_heads: int = 6

    def validate(self) -> None:
        if self.history <= SINK + RECENT or self.query_length < 2:
            raise ValueError("history must exceed sink+recent and query_length must be >=2")
        if self.dim not in (64, 256) or self.tile <= 0 or self.samples <= 0:
            raise ValueError("invalid CPU experiment geometry")
        if self.query_heads <= 0:
            raise ValueError("query_heads must be positive")


def sampled_offsets(query_length: int, count: int = 21) -> tuple[int, ...]:
    """Include the first causal frontier, central rows, and the final row."""
    if query_length < 2 or count < 2:
        raise ValueError("need at least two positions and samples")
    points = {(i * (query_length - 1)) // (count - 1) for i in range(count)}
    points.update((0, 1, query_length // 2, query_length - 2, query_length - 1))
    return tuple(sorted(points))


def _random_bf16(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=gen, dtype=torch.float32).to(torch.bfloat16)


def _history(case: Case, rk: torch.Tensor, rv: torch.Tensor) -> dict[str, Any]:
    """Retain packed INT2 and only the exact sink/recent windows."""
    n, d, tile = case.history, case.dim, case.tile
    slot_bytes = d // 2 + 8
    slots = torch.empty((n, 1, slot_bytes), dtype=torch.uint8)
    exact_sink = torch.empty((SINK, 1, d), dtype=torch.bfloat16)
    tail_start = n - RECENT
    exact_tail = torch.empty((RECENT, 1, d), dtype=torch.bfloat16)
    for begin in range(0, n, tile):
        end = min(n, begin + tile)
        length = end - begin
        k = _random_bf16((length, 1, d), SEED + n * 3 + d * 7 + begin * 2)
        v = _random_bf16((length, 1, d), SEED + n * 5 + d * 11 + begin * 2 + 1)
        slots[begin:end] = reference.encode_kv(k.float() @ rk, v.float() @ rv)
        if begin < SINK:
            stop = min(end, SINK)
            exact_sink[begin:stop] = k[:stop - begin]
        overlap = max(begin, tail_start)
        if overlap < end:
            exact_tail[overlap - tail_start:end - tail_start] = k[overlap - begin:]
        if begin < SINK:
            stop = min(end, SINK)
            # V has a separate exact window and cannot reuse the K tensor.
            # It is assigned below through the second pair in the return value.
        if overlap < end:
            pass
    # Build the exact V windows with the same bounded deterministic tiles.
    exact_sink_v = torch.empty_like(exact_sink)
    exact_tail_v = torch.empty_like(exact_tail)
    for begin in range(0, n, tile):
        end = min(n, begin + tile)
        if begin >= SINK and end <= tail_start:
            continue
        v = _random_bf16((end - begin, 1, d), SEED + n * 5 + d * 11 + begin * 2 + 1)
        if begin < SINK:
            stop = min(end, SINK)
            exact_sink_v[begin:stop] = v[:stop - begin]
        overlap = max(begin, tail_start)
        if overlap < end:
            exact_tail_v[overlap - tail_start:end - tail_start] = v[overlap - begin:]
    return {"slots": slots, "sink_k": exact_sink, "sink_v": exact_sink_v,
            "tail_k": exact_tail, "tail_v": exact_tail_v,
            "tail_start": tail_start, "stored_bytes": slots.numel()}


def _candidate_common(query: torch.Tensor, slots: torch.Tensor,
                      rk: torch.Tensor, rv: torch.Tensor, scale: float
                      ) -> reference.AttentionResult:
    # PR triton_oscar_decode.py:154-240 writes the rotated dequant to FP16.
    k_rot, v_rot = reference.decode_kv(slots, query.shape[-1])
    k = (k_rot.to(torch.float16).float() @ rk.T).to(torch.bfloat16)
    v = (v_rot.to(torch.float16).float() @ rv.T).to(torch.bfloat16)
    # PR oscar_attn.py:541-589 sends BF16 Q/K/V to SDPA. A native partial
    # produces BF16 output and FP32 LSE; CPU attention simulates its FP32
    # accumulation but does not claim vendor kernel parity.
    partial = reference.attention(query, k, v, scale=scale, causal=False)
    return reference.AttentionResult(partial.output.to(torch.bfloat16).float(),
                                     partial.lse.float())


def _slice(part: reference.AttentionResult, index: int) -> reference.AttentionResult:
    return reference.AttentionResult(part.output[index:index + 1],
                                     part.lse[index:index + 1])


def _compare(actual: torch.Tensor, expected: torch.Tensor, atol: float,
             rtol: float) -> dict[str, Any]:
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    diff = (actual.float() - expected.float()).abs()
    allowed = atol + rtol * expected.float().abs()
    return {"passed": finite and bool(torch.all(diff <= allowed)),
            "max_abs": float(diff.max()),
            "max_excess": float((diff - allowed).max().clamp_min(0)),
            "failed_elements": int((diff > allowed).sum()),
            "elements": expected.numel()}


def _current_tile(case: Case, begin: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
    d = case.dim
    k = _random_bf16((end - begin, 1, d), SEED + case.history * 13 + d * 17 + begin * 2)
    v = _random_bf16((end - begin, 1, d), SEED + case.history * 19 + d * 23 + begin * 2 + 1)
    return k, v


def run_case(case: Case, *, atol: float, rtol: float) -> dict[str, Any]:
    case.validate()
    if not (math.isfinite(atol) and math.isfinite(rtol) and atol >= 0 and rtol >= 0):
        raise ValueError("invalid frozen tolerance")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    rk = _hadamard(torch, case.dim)
    rv = rk.flip(1).contiguous()
    storage = _history(case, rk, rv)
    offsets = sampled_offsets(case.query_length, case.samples)
    q = _random_bf16((len(offsets), case.query_heads, case.dim),
                     SEED + case.history * 29 + case.dim * 31)
    scale = case.dim ** -0.5
    common_end = max(SINK, case.history + 1 - RECENT)
    reference_common: list[reference.AttentionResult] = []
    candidate_common: list[reference.AttentionResult] = []
    for begin in range(SINK, common_end, case.tile):
        end = min(common_end, begin + case.tile)
        tile_slots = storage["slots"][begin:end]
        reference_common.append(reference.compressed_attention(
            q, tile_slots, rk, rv, scale=scale, causal=False))
        candidate_common.append(_candidate_common(q, tile_slots, rk, rv, scale))
    ref_common = reference.merge_attention(reference_common)
    alt_common = reference.merge_attention(candidate_common)
    ref_outputs: list[torch.Tensor] = []
    alt_outputs: list[torch.Tensor] = []
    ref_lses: list[torch.Tensor] = []
    alt_lses: list[torch.Tensor] = []
    for row, offset in enumerate(offsets):
        query = q[row:row + 1]
        # For each later query, a portion of the old recent window crosses
        # the causal frontier and re-enters compressed history. Both arms use
        # the same frozen FP32 reference for this query-dependent portion.
        cut = min(case.history, max(SINK, case.history + offset + 1 - RECENT))
        shared: list[reference.AttentionResult] = []
        shared.append(reference.attention(query, storage["sink_k"], storage["sink_v"],
                                          scale=scale, causal=False))
        for begin in range(common_end, cut, case.tile):
            end = min(cut, begin + case.tile)
            shared.append(reference.compressed_attention(
                query, storage["slots"][begin:end], rk, rv,
                scale=scale, causal=False))
        if cut < case.history:
            begin = cut - storage["tail_start"]
            shared.append(reference.attention(
                query, storage["tail_k"][begin:], storage["tail_v"][begin:],
                scale=scale, causal=False))
        for begin in range(0, offset + 1, case.tile):
            end = min(offset + 1, begin + case.tile)
            ck, cv = _current_tile(case, begin, min(begin + case.tile, case.query_length))
            shared.append(reference.attention(query, ck[:end - begin], cv[:end - begin],
                                              scale=scale, causal=False))
        ref_parts = [_slice(ref_common, row), *shared]
        alt_parts = [_slice(alt_common, row), *shared]
        merged_ref = reference.merge_attention(ref_parts)
        merged_alt = reference.merge_attention(alt_parts)
        ref_outputs.append(merged_ref.output)
        alt_outputs.append(merged_alt.output.to(torch.bfloat16).float())
        ref_lses.append(merged_ref.lse)
        alt_lses.append(merged_alt.lse)
    expected = torch.cat(ref_outputs, dim=0)
    actual = torch.cat(alt_outputs, dim=0)
    lse_ref = torch.cat(ref_lses, dim=0)
    lse_alt = torch.cat(alt_lses, dim=0)
    output_check = _compare(actual, expected, atol, rtol)
    lse_check = _compare(lse_alt, lse_ref, atol, rtol)
    return {"status": "passed" if output_check["passed"] and lse_check["passed"] else "failed",
            "history": case.history, "query_length": case.query_length,
            "dimension": case.dim, "query_heads": case.query_heads,
            "tile": case.tile, "sampled_rows": list(offsets),
            "common_quantized_range": [SINK, common_end],
            "frontier_range": [common_end, case.history],
            "stored_compressed_bytes": storage["stored_bytes"],
            "max_materialized_kv_rows_each": case.tile,
            "output": output_check, "lse": lse_check}


def run(output: Path, *, tile: int = 1024,
        cases: tuple[Case, ...] | None = None) -> dict[str, Any]:
    acceptance_path = Path(__file__).resolve().parents[2] / "configs/acceptance.json"
    acceptance = json.loads(acceptance_path.read_text())
    frozen = acceptance["fused_attention"]
    atol, rtol = float(frozen["atol"]), float(frozen["rtol"])
    shapes = cases or (Case(20_000, 8192, 64, tile),
                       Case(20_000, 8192, 256, tile),
                       Case(65_536, 16384, 64, tile),
                       Case(65_536, 16384, 256, tile))
    report: dict[str, Any] = {
        "status": "passed", "mode": "cpu_experiment_only", "npu": "not_run",
        "model_quality": "not_established", "production_route": "unchanged",
        "acceptance_path": str(acceptance_path),
        "frozen_atol": atol, "frozen_rtol": rtol,
        "pr_dtype_order": ["INT2 affine dequant FP32", "rotated storage FP16",
                           "inverse rotation FP32", "K/V cast query BF16",
                           "BF16 attention partial output", "FP32 LSE merge"],
        "scope": "common historical INT2 prefix only; sink/frontier/recent/current use FP32 reference",
        "cases": [], "not_run_after_first_failure": [],
        "archive": ["#126", "#129", "#148", "startup D.4"],
    }
    for index, shape in enumerate(shapes):
        case = run_case(shape, atol=atol, rtol=rtol)
        report["cases"].append(case)
        print("PREFILL_TILE_REFERENCE " + json.dumps({
            "history": shape.history, "q": shape.query_length, "d": shape.dim,
            "status": case["status"], "output_max_abs": case["output"]["max_abs"],
            "output_failed_elements": case["output"]["failed_elements"],
            "lse_max_abs": case["lse"]["max_abs"]}, sort_keys=True), flush=True)
        if case["status"] != "passed":
            report["status"] = "failed"
            report["not_run_after_first_failure"] = [
                {"history": c.history, "query_length": c.query_length, "dimension": c.dim}
                for c in shapes[index + 1:]]
            break
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("reports/prefill_tile_reference_20260929.json"))
    parser.add_argument("--tile", type=int, default=1024)
    args = parser.parse_args(argv)
    try:
        report = run(args.output, tile=args.tile)
    except Exception as exc:
        print(f"PREFILL_TILE_REFERENCE_ERROR {type(exc).__name__}: {exc}", flush=True)
        return 2
    print(f"PREFILL_TILE_REFERENCE_RESULT status={report['status']} output={args.output}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
