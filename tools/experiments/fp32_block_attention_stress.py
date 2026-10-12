"""Bounded CPU falsification of an 8192-KV all-FP32 OSCAR attention block.

Archive #126/#129/#145/#148 and startup D.4: this compares the current
KV256 online FP32 recurrence with a hypothetical KV8192 block that computes
one FP32 QK and one FP32 PV per block. Store bytes are produced by the pinned
INT2/FP16-metadata oracle; decoded K/V and rotated Q remain FP32 throughout.
The 8192 block rescales each 256-score piece to the block maximum before one
large PV, then merges raw accumulator/max/sum across blocks and applies RvT.

Both arms must independently pass the frozen FP32 reference tolerance. A
case where the old arm fails is inconclusive; only old-pass/new-fail rejects
the new grouping. CPU PyTorch GEMM order is not an AscendC numerical, memory,
graph or speed result. No history-wide K/V tensor is constructed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from oscar_ascend.ops import reference
from tools.experiments.prefill_tile_stress import _check, _filler, _signal, _slots


OLD_KV = 256
BLOCK_KV = 8192
ORACLE_KV = 1024


def _finish(acc: torch.Tensor, maximum: torch.Tensor, total: torch.Tensor,
            value_rotation: torch.Tensor) -> reference.AttentionResult:
    if not bool((total > 0).all()):
        raise AssertionError("a nonempty valid history needs a positive FP32 denominator")
    output = (acc / total.unsqueeze(-1)) @ value_rotation.T
    return reference.AttentionResult(output, maximum + torch.log(total))


def _old_kv256(query_rot: torch.Tensor, slots: torch.Tensor,
               value_rotation: torch.Tensor, scale: float
               ) -> tuple[reference.AttentionResult, dict[str, int]]:
    shape = query_rot.shape
    maximum = torch.full(shape[:-1], -math.inf, dtype=torch.float32)
    total = torch.zeros_like(maximum)
    acc = torch.zeros(shape, dtype=torch.float32)
    calls = 0
    for begin in range(0, slots.shape[0], OLD_KV):
        key, value = reference.decode_kv(slots[begin:begin + OLD_KV], shape[-1])
        scores = torch.einsum("qhd,khd->qhk", query_rot, key) * scale
        next_max = torch.maximum(maximum, scores.amax(-1))
        old_alpha = torch.where(total > 0, torch.exp(maximum - next_max),
                                torch.zeros_like(total))
        probability = torch.exp(scores - next_max.unsqueeze(-1))
        acc = acc * old_alpha.unsqueeze(-1) + torch.einsum(
            "qhk,khd->qhd", probability, value)
        total = total * old_alpha + probability.sum(-1)
        maximum = next_max
        calls += 1
    return _finish(acc, maximum, total, value_rotation), {"qk": calls, "pv": calls}


def _new_kv8192(query_rot: torch.Tensor, slots: torch.Tensor,
                value_rotation: torch.Tensor, scale: float
                ) -> tuple[reference.AttentionResult, dict[str, int]]:
    shape = query_rot.shape
    maximum = torch.full(shape[:-1], -math.inf, dtype=torch.float32)
    total = torch.zeros_like(maximum)
    acc = torch.zeros(shape, dtype=torch.float32)
    calls = 0
    for begin in range(0, slots.shape[0], BLOCK_KV):
        key, value = reference.decode_kv(slots[begin:begin + BLOCK_KV], shape[-1])
        # One large FP32 QK for the bounded 8K block. The local 256 pieces
        # rescale P to this block's maximum; no BF16/FP16 GEMM input is used.
        scores = torch.einsum("qhd,khd->qhk", query_rot, key) * scale
        block_max = scores.amax(-1)
        probability = torch.empty_like(scores)
        for offset in range(0, scores.shape[-1], OLD_KV):
            local = scores[..., offset:offset + OLD_KV]
            local_max = local.amax(-1)
            probability[..., offset:offset + OLD_KV] = (
                torch.exp(local - local_max.unsqueeze(-1))
                * torch.exp(local_max - block_max).unsqueeze(-1))
        block_sum = probability.sum(-1)
        # One large FP32 PV for the same bounded 8K block.
        block_acc = torch.einsum("qhk,khd->qhd", probability, value)
        next_max = torch.maximum(maximum, block_max)
        old_alpha = torch.where(total > 0, torch.exp(maximum - next_max),
                                torch.zeros_like(total))
        block_alpha = torch.exp(block_max - next_max)
        acc = acc * old_alpha.unsqueeze(-1) + block_acc * block_alpha.unsqueeze(-1)
        total = total * old_alpha + block_sum * block_alpha
        maximum = next_max
        calls += 1
    return _finish(acc, maximum, total, value_rotation), {"qk": calls, "pv": calls}


def _oracle(query: torch.Tensor, slots: torch.Tensor, key_rotation: torch.Tensor,
            value_rotation: torch.Tensor, scale: float) -> reference.AttentionResult:
    parts = [reference.compressed_attention(
        query, slots[begin:begin + ORACLE_KV], key_rotation, value_rotation,
        scale=scale, causal=False)
        for begin in range(0, slots.shape[0], ORACLE_KV)]
    return reference.merge_attention(parts)


def _evaluate(name: str, query: torch.Tensor, slots: torch.Tensor,
              key_rotation: torch.Tensor, value_rotation: torch.Tensor,
              exact: reference.AttentionResult | None, atol: float, rtol: float
              ) -> dict[str, Any]:
    d = query.shape[-1]
    scale = d ** -0.5
    query_rot = query.float() @ key_rotation
    old, old_calls = _old_kv256(query_rot, slots, value_rotation, scale)
    new, new_calls = _new_kv8192(query_rot, slots, value_rotation, scale)
    independent = _oracle(query, slots, key_rotation, value_rotation, scale)
    if exact is not None:
        old = reference.merge_attention([old, exact])
        new = reference.merge_attention([new, exact])
        independent = reference.merge_attention([independent, exact])
    old_checks = {"output": _check(old.output, independent.output, atol, rtol),
                  "lse": _check(old.lse, independent.lse, atol, rtol)}
    new_checks = {"output": _check(new.output, independent.output, atol, rtol),
                  "lse": _check(new.lse, independent.lse, atol, rtol)}
    old_pass = all(item["passed"] for item in old_checks.values())
    new_pass = all(item["passed"] for item in new_checks.values())
    verdict = ("old_baseline_failed_inconclusive" if not old_pass else
               "new_grouping_rejected" if not new_pass else "cpu_tolerance_passed")
    return {"name": name, "dimension": d, "query_rows": query.shape[0],
            "history_rows": slots.shape[0], "old_calls": old_calls,
            "new_calls": new_calls, "old_vs_oracle": old_checks,
            "new_vs_oracle": new_checks,
            "new_vs_old": {"output": _check(new.output, old.output, atol, rtol),
                           "lse": _check(new.lse, old.lse, atol, rtol)},
            "verdict": verdict}


def _case(d: int, rotation: str, history: int, placement: str,
          atol: float, rtol: float) -> dict[str, Any]:
    query_base, signal_k, signal_v, rk = _signal(d, rotation)
    rv = rk if rotation == "identity" else rk.flip(1).contiguous()
    signal = reference.encode_kv(reference.rotate_clip(signal_k.float(), rk),
                                 reference.rotate_clip(signal_v.float(), rv))
    if history == 2:
        query, slots, exact = query_base, signal, None
    else:
        filler_k, filler_v = _filler(d, rotation, query_base)
        filler = reference.encode_kv(reference.rotate_clip(filler_k.float(), rk),
                                     reference.rotate_clip(filler_v.float(), rv))
        common_rows = history - 320
        slots = filler.expand(common_rows, -1, -1).clone()
        if placement == "same_8k":
            slots[:2] = signal
        elif placement == "cross_8k":
            slots[0] = signal[0]
            slots[BLOCK_KV] = signal[1]
        else:
            raise ValueError("unknown signal placement")
        factors = torch.linspace(0.75, 1.25, 21)
        query = (query_base.float() * factors[:, None, None]).to(torch.bfloat16)
        exact_key = filler_k.expand(341, -1, -1)
        exact_value = filler_v.expand(341, -1, -1)
        exact = reference.attention(
            query, exact_key, exact_value, scale=d ** -0.5, causal=True,
            query_positions=torch.arange(320, 341),
            key_positions=torch.arange(341))
    result = _evaluate(f"{rotation}_d{d}_h{history}_{placement}", query, slots,
                       rk, rv, exact, atol, rtol)
    result.update({"rotation": rotation, "signal_placement": placement,
                   "store_reachable": True,
                   "max_staged_history_rows": min(BLOCK_KV, slots.shape[0]),
                   "exact_rows": 0 if exact is None else 341})
    return result


def _dense_case(d: int, atol: float, rtol: float) -> dict[str, Any]:
    """All 16K rows remain live, with signed V cancellation across tiles."""
    base_q, base_k, base_v, rk = _signal(d, "hadamard")
    rv = rk.flip(1).contiguous()
    count = 16_384
    position = torch.arange(count)
    tile = position // OLD_KV
    index = (position % 2).long()
    key = (base_k.index_select(0, index).float()
           * (1 + (tile % 8).float()[:, None, None] / 32)).to(torch.bfloat16)
    value = (base_v.index_select(0, index).float()
             * (1 + (tile % 4).float()[:, None, None] / 8)).to(torch.bfloat16)
    slots = reference.encode_kv(reference.rotate_clip(key.float(), rk),
                                reference.rotate_clip(value.float(), rv))
    query = (base_q.float() * torch.linspace(0.75, 1.25, 21)[:, None, None]
             ).to(torch.bfloat16)
    result = _evaluate(f"dense_hadamard_d{d}_h{count}", query, slots,
                       rk, rv, None, atol, rtol)
    result.update({"rotation": "hadamard", "store_reachable": True,
                   "signed_value_cancellation": True,
                   "scaled_kv_groups": "K varies every KV256 among 8 scales; V varies among 4 scales",
                   "max_staged_history_rows": BLOCK_KV})
    return result


def run(output: Path) -> dict[str, Any]:
    torch.set_num_threads(min(4, torch.get_num_threads()))
    acceptance = Path(__file__).resolve().parents[2] / "configs/acceptance.json"
    frozen = json.loads(acceptance.read_text())["fused_attention"]
    atol, rtol = float(frozen["atol"]), float(frozen["rtol"])
    if not (math.isfinite(atol) and math.isfinite(rtol) and atol >= 0 and rtol >= 0):
        raise ValueError("invalid frozen fused-attention tolerance")
    shapes = ((64, "identity", 2, "same_8k"),
              (64, "hadamard", 2, "same_8k"),
              (256, "hadamard", 2, "same_8k"),
              (64, "identity", 20_000, "same_8k"),
              (64, "hadamard", 20_000, "cross_8k"),
              (256, "hadamard", 65_536, "cross_8k"))
    cases = []
    for shape in shapes:
        case = _case(*shape, atol, rtol)
        cases.append(case)
        print("FP32_BLOCK_STRESS " + json.dumps({"case": case["name"],
              "verdict": case["verdict"],
              "old_output_excess": case["old_vs_oracle"]["output"]["max_excess"],
              "new_output_excess": case["new_vs_oracle"]["output"]["max_excess"]},
              sort_keys=True), flush=True)
    for d in (64, 256):
        case = _dense_case(d, atol, rtol)
        cases.append(case)
        print("FP32_BLOCK_STRESS " + json.dumps({"case": case["name"],
              "verdict": case["verdict"],
              "old_output_excess": case["old_vs_oracle"]["output"]["max_excess"],
              "new_output_excess": case["new_vs_oracle"]["output"]["max_excess"]},
              sort_keys=True), flush=True)
    rejected = sum(case["verdict"] == "new_grouping_rejected" for case in cases)
    inconclusive = sum(case["verdict"] == "old_baseline_failed_inconclusive"
                       for case in cases)
    report = {"status": "rejected_cpu" if rejected else
              "inconclusive_cpu" if inconclusive else "cpu_tolerance_passed",
              "mode": "bounded_cpu_fp32_numerical_experiment",
              "frozen_atol": atol, "frozen_rtol": rtol,
              "reference": "independent original-space tilewise compressed_attention + merge_attention",
              "math_dtype": "FP32 Qrot/Krot/Vrot/QK/PV/online state and RvT; INT2 metadata FP16 store rounding retained",
              "block_kv": BLOCK_KV, "old_kv": OLD_KV,
              "cases": cases, "new_grouping_rejected": rejected,
              "old_baseline_inconclusive": inconclusive,
              "npu_precision": "not_run", "graph": "not_run",
              "device_speed": "not_run", "model_quality": "not_established",
              "production_route": "unchanged",
              "archive": ["#126", "#129", "#145", "#148", "startup D.4"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print("FP32_BLOCK_RESULT " + json.dumps({"status": report["status"],
          "rejected": rejected, "old_inconclusive": inconclusive,
          "report": str(output)}, sort_keys=True), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("reports/fp32_block_attention_stress_20260930.json"))
    args = parser.parse_args()
    return 0 if run(args.output)["status"] == "cpu_tolerance_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
