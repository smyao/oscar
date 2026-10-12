"""Bounded CPU falsification of the PR continuation-prefill dtype route.

The pinned PR (references/oscar-vllm-pr46774/vllm/v1/attention/ops/
triton_oscar_decode.py:154-240 and backends/oscar_attn.py:541-589) decodes
historical INT2 K/V into rotated FP16, applies the inverse rotation in FP32,
casts K/V to query BF16, then uses BF16 attention partials. This file tests
that order against the frozen FP32 OSCAR oracle.

All compressed slots are produced by the normal BF16 input -> rotation ->
reference.encode_kv store path. A two-token cancellation case is embedded in
20K/65K history with only tile-sized decoded K/V. No vendor FIA, AscendC,
service, performance, or model-quality result is inferred from this CPU test.

Archive #126/#129/#148 and startup D.4 apply: full-history materialization
caused the original multi-second dequant regression; this test must never be
used as a serving path or to relax configs/acceptance.json.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from oscar_ascend.ops import reference
from tools.experiments.prefill_tile_reference import _candidate_common
from tools.probe_history_reuse import _hadamard


SEED_HADAMARD = 46801
TILE_ROWS = 1024


def _signal(d: int, rotation: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if d not in (64, 256):
        raise ValueError("only production D64/D256 dimensions are admitted")
    matrix = torch.eye(d) if rotation == "identity" else _hadamard(torch, d)
    if rotation == "identity":
        # Both raw query-facing K values are the same BF16 1/3. Different
        # per-vector maxima make the old INT2 reconstructions differ, while
        # the later BF16 K cast collapses them to one value.
        k = torch.zeros((2, 1, d), dtype=torch.bfloat16)
        k[:, 0, 0] = torch.tensor(1 / 3, dtype=torch.float32).to(torch.bfloat16)
        k[0, 0, 1] = 1
        v = torch.zeros_like(k)
        v[0, 0, 0], v[1, 0, 0] = 8, -8
        v[0, 0, 1], v[1, 0, 1] = -8, 8
        q = torch.zeros((1, 1, d), dtype=torch.bfloat16)
        q[0, 0, 0] = 32 if d == 64 else 64
    elif rotation == "hadamard":
        generator = torch.Generator(device="cpu").manual_seed(SEED_HADAMARD)
        k = torch.randn((2, 1, d), generator=generator).to(torch.bfloat16)
        v = (torch.randn((2, 1, d), generator=generator).to(torch.bfloat16) * 0.5)
        v[0, 0, 0], v[1, 0, 0] = 16, -16
        q = torch.zeros((1, 1, d), dtype=torch.bfloat16)
        q[0, 0, 4], q[0, 0, 24] = 4, 3.046875
    else:
        raise ValueError("rotation must be identity or hadamard")
    return q, k, v, matrix


def _slots(k: torch.Tensor, v: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    # Same quantizer and FP16 metadata rounding as the frozen store oracle.
    return reference.encode_kv(reference.rotate_clip(k.float(), matrix),
                               reference.rotate_clip(v.float(), matrix))


def _check(actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float
           ) -> dict[str, Any]:
    if actual.shape != expected.shape:
        raise ValueError("comparison geometry differs")
    if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        return {"passed": False, "reason": "nonfinite"}
    difference = (actual.float() - expected.float()).abs()
    allowance = atol + rtol * expected.float().abs()
    excess = difference - allowance
    index = int(excess.flatten().argmax())
    return {"passed": bool(torch.all(excess <= 0)), "max_abs": float(difference.max()),
            "max_excess": float(excess.max()), "failed_elements": int((excess > 0).sum()),
            "worst_flat_index": index,
            "worst_expected": float(expected.flatten()[index]),
            "worst_actual": float(actual.flatten()[index]),
            "worst_allowed": float(allowance.flatten()[index])}


def _ablation(q: torch.Tensor, slots: torch.Tensor, matrix: torch.Tensor,
              scale: float) -> dict[str, float]:
    rotated_k, rotated_v = reference.decode_kv(slots, q.shape[-1])
    k_ref = rotated_k @ matrix.T
    v_ref = rotated_v @ matrix.T
    k_cast = (rotated_k.to(torch.float16).float() @ matrix.T).to(torch.bfloat16)
    v_cast = (rotated_v.to(torch.float16).float() @ matrix.T).to(torch.bfloat16)
    base = reference.attention(q, k_ref, v_ref, scale=scale, causal=False).output
    k_only = reference.attention(q, k_cast, v_ref, scale=scale, causal=False).output
    v_only = reference.attention(q, k_ref, v_cast, scale=scale, causal=False).output
    both = reference.attention(q, k_cast, v_cast, scale=scale, causal=False).output
    return {"key_cast_max_abs": float((k_only - base).abs().max()),
            "value_cast_max_abs": float((v_only - base).abs().max()),
            "both_cast_before_partial_bf16_max_abs": float((both - base).abs().max()),
            "partial_bf16_extra_max_abs": float((both.to(torch.bfloat16).float() - both).abs().max())}


def run_pair(d: int, rotation: str, atol: float, rtol: float) -> dict[str, Any]:
    q, k, v, matrix = _signal(d, rotation)
    slots = _slots(k, v, matrix)
    decoded_k, decoded_v = reference.decode_kv(slots, d)
    assert bool(torch.isfinite(decoded_k).all() and torch.isfinite(decoded_v).all())
    scale = d ** -0.5
    expected = reference.compressed_attention(q, slots, matrix, matrix,
                                              scale=scale, causal=False)
    actual = _candidate_common(q, slots, matrix, matrix, scale)
    output = _check(actual.output, expected.output, atol, rtol)
    lse = _check(actual.lse, expected.lse, atol, rtol)
    return {"dimension": d, "rotation": rotation, "store_reachable": True,
            "query_bf16_nonzero": [[i, float(value)] for i, value in enumerate(q[0, 0])
                                   if value != 0],
            "raw_key_max_abs": float(k.float().abs().max()),
            "raw_value_max_abs": float(v.float().abs().max()),
            "output": output, "lse": lse,
            "ablation": _ablation(q, slots, matrix, scale),
            "status": "passed" if output["passed"] and lse["passed"] else "failed"}


def _filler(d: int, rotation: str, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    k = torch.zeros((1, 1, d), dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    # Downweight filler under every sampled Q. Another K/V component keeps
    # per-vector scale positive, including after Hadamard rotation.
    for index in torch.nonzero(q[0, 0], as_tuple=False).flatten().tolist():
        k[0, 0, index] = -64 if q[0, 0, index] > 0 else 64
    k[0, 0, 1] = 1
    v[0, 0, 1] = 1
    v[0, 0, 2] = -1
    return k, v


def run_long(d: int, rotation: str, history: int, atol: float, rtol: float
             ) -> dict[str, Any]:
    """Only a tile of history is ever decoded; 21 queries share the prefix."""
    if history not in (20_000, 65_536):
        raise ValueError("long-history stress only admits specified lengths")
    q_base, k_signal, v_signal, matrix = _signal(d, rotation)
    filler_k, filler_v = _filler(d, rotation, q_base)
    signal_slots = _slots(k_signal, v_signal, matrix)
    filler_slot = _slots(filler_k, filler_v, matrix)
    common = filler_slot.expand(history - 320, -1, -1).clone()
    common[:2] = signal_slots
    # 21 BF16 query rows model samples from a long continuation chunk. Their
    # magnitudes vary but exact sink/recent/current rows remain downweighted.
    factors = torch.linspace(0.75, 1.25, 21)
    queries = (q_base.float() * factors[:, None, None]).to(torch.bfloat16)
    scale = d ** -0.5
    ref_parts: list[reference.AttentionResult] = []
    alt_parts: list[reference.AttentionResult] = []
    for start in range(0, common.shape[0], TILE_ROWS):
        tile = common[start:start + TILE_ROWS]
        ref_parts.append(reference.compressed_attention(
            queries, tile, matrix, matrix, scale=scale, causal=False))
        alt_parts.append(_candidate_common(queries, tile, matrix, matrix, scale))
    exact_k = filler_k.expand(341, -1, -1)
    exact_v = filler_v.expand(341, -1, -1)
    exact = reference.attention(
        queries, exact_k, exact_v, scale=scale, causal=True,
        query_positions=torch.arange(320, 341), key_positions=torch.arange(341))
    expected = reference.merge_attention([*ref_parts, exact])
    actual = reference.merge_attention([*alt_parts, exact])
    # The PR continuation path returns BF16 attention output to the model.
    output = _check(actual.output.to(torch.bfloat16).float(), expected.output, atol, rtol)
    lse = _check(actual.lse, expected.lse, atol, rtol)
    return {"dimension": d, "rotation": rotation, "history": history,
            "sampled_query_rows": 21, "compressed_history_rows": history - 320,
            "exact_sink_recent_current_rows": 341,
            "maximum_decoded_history_rows_at_once": TILE_ROWS,
            "store_reachable": True, "output": output, "lse": lse,
            "status": "passed" if output["passed"] and lse["passed"] else "failed"}


def run(output: Path) -> dict[str, Any]:
    torch.set_num_threads(min(4, torch.get_num_threads()))
    acceptance_path = Path(__file__).resolve().parents[2] / "configs/acceptance.json"
    frozen = json.loads(acceptance_path.read_text())["fused_attention"]
    atol, rtol = float(frozen["atol"]), float(frozen["rtol"])
    if not (math.isfinite(atol) and math.isfinite(rtol) and atol >= 0 and rtol >= 0):
        raise ValueError("invalid frozen fused-attention tolerance")
    pairs = [run_pair(d, rotation, atol, rtol)
             for d, rotation in ((64, "identity"), (256, "identity"),
                                 (64, "hadamard"), (256, "hadamard"))]
    long_cases = [run_long(d, rotation, history, atol, rtol)
                  for d, rotation, history in ((64, "identity", 20_000),
                                               (64, "hadamard", 20_000),
                                               (256, "hadamard", 65_536))]
    failures = sum(case["status"] == "failed" for case in [*pairs, *long_cases])
    report = {"status": "falsified" if failures else "inconclusive",
              "mode": "cpu_numerical_stress_only", "vendor_fia": "not_run",
              "model_quality": "not_established", "production_route": "unchanged",
              "frozen_atol": atol, "frozen_rtol": rtol,
              "reference": "PR46774@57286d5d continuation-prefill dtype sequence",
              "long_case_scope": "21 synthetic continuation queries, shared quantized old prefix, exact downweighted sink/recent/current with causal current masks; no full window policy or vendor FIA simulation",
              "archive": ["#126", "#129", "#148", "startup D.4"],
              "pair_cases": pairs, "long_cases": long_cases,
              "failed_cases": failures,
              "conclusion": "Unconditional tiled PR dtype substitution fails the frozen FP32 operator tolerance on valid store outputs" if failures else "No counterexample in this bounded CPU stress"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("PREFILL_TILE_STRESS " + json.dumps({"status": report["status"],
          "failed_cases": failures, "pair_cases": len(pairs),
          "long_cases": len(long_cases), "report": str(output)}, sort_keys=True), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("reports/prefill_tile_stress_20260930.json"))
    args = parser.parse_args()
    return 0 if run(args.output)["status"] == "falsified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
