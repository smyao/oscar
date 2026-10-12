"""Archive #13-22/#126/#148/#154-156; D.4 CPU-only numerical screen.

Unlike the affine-code route, first obtain actual FP32 dequantized values using
the unchanged oracle Mul-then-Add order. Split those values and Q/P into BF16
residual planes. Multiply every plane pair and sum FP32 partial matrices in
lexicographic (A-plane, B-plane) order. No affine redistribution, no relaxed
tolerance, no production route, no VM, and no hardware accumulation claim.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oscar_ascend.ops.reference import (attention, compressed_attention, decode_kv, encode_kv,
                                       quantize_int2, dequantize_int2)
from tools.experiments.affine_bf16_residual_screen import cases, gate, split_bf16


def matrix_product(left, right, equation, left_count, right_count):
    a, ar = split_bf16(left, left_count)
    b, br = split_bf16(right, right_count)
    result = None
    for ap in a:
        for bp in b:
            partial = torch.einsum(equation, ap, bp)
            result = partial if result is None else result + partial
    return result, {"left_remainder_max": float(ar.abs().max()),
                    "right_remainder_max": float(br.abs().max()),
                    "left_exact_reconstruction": bool((ar == 0).all()),
                    "right_exact_reconstruction": bool((br == 0).all()),
                    "left_nonzero_planes": sum(bool((p != 0).any()) for p in a),
                    "right_nonzero_planes": sum(bool((p != 0).any()) for p in b),
                    "matrix_products_executed": left_count * right_count,
                    "partial_combine_order": "left_plane_outer_right_plane_inner_FP32_add"}


def candidate(query, key, value, rotation, left_count, right_count):
    # Actual dequant output is consumed without factoring scale/zero out.
    q = query.float() @ rotation
    repeated_key, repeated_value = key.expand(-1, 6, -1), value.expand(-1, 6, -1)
    dot, qdetail = matrix_product(q, repeated_key, "qhd,khd->qhk", left_count, right_count)
    scores = dot * (1 / 16)
    maximum = scores.amax(-1, keepdim=True)
    weights = torch.exp(scores - maximum)
    value_sum, pdetail = matrix_product(weights, repeated_value, "qhk,khd->qhd", left_count, right_count)
    output = (value_sum / weights.sum(-1, keepdim=True)) @ rotation.T
    lse = maximum.squeeze(-1) + torch.log(weights.sum(-1))
    return output, lse, {"qk": qdetail, "pv": pdetail}


def work_ledger(a, b):
    m, n, k = 24, 256, 256
    # Ideal single grouped launch per operation, every distinct operand plane
    # published/read once; partial pair results still FP32 and need combining.
    f32 = (m*k*4 + k*n*4 + m*n*4) * 2 * 2
    expanded = (a*m*k*2 + b*k*n*2 + a*b*m*n*4) * 2 * 2
    return {"A_planes": a, "B_planes": b, "products_per_original_matmul": a*b,
            "QK_plus_PV_MACs": a*b*2*m*n*k, "FP32_baseline_MACs": 2*m*n*k,
            "FP32_KV_still_materialized_in_UB": True,
            "KV_plane_GM_payload_bytes": b*2*k*n*2,
            "FP32_KV_payload_bytes": 2*k*n*4,
            "logical_operand_and_partial_publish_read_bytes": expanded,
            "baseline_logical_bytes": f32, "ratio": expanded/f32,
            "separate_Matmul_calls_per_KV_if_unfused": 2*a*b,
            "baseline_calls_per_KV": 2,
            "ideal_fused_minimum_launches_per_KV": 2,
            "large_compute_only_gain_requirements": {
                "any_gain": f"r>{a*b}/(1-E/C)",
                "1p5x": f"r>={a*b}/(2/3-E/C)",
                "2x": f"r>={a*b}/(1/2-E/C)"},
            "zero_extra_cost_rate_ratio_for_1p5x": a*b*1.5,
            "zero_extra_cost_rate_ratio_for_2x": a*b*2,
            "boundary": "r=measured sustainable BF16/FP32 compute rate; E=additional exposed split/recombine/control; not whole-op dtype ratio",
            "additional_not_charged_bytes": ["plane extraction scratch", "final combined output write",
                "SDK rereads/padding", "metadata and flags", "same softmax and final Rv"],
            "fusion_caveat": "retaining partials in L0C could avoid some GM but changes combine order; this screen does not qualify that algorithm"}


def metadata_plane_bound():
    # A legal finite FP16 scale/zero pair can produce more than two BF16
    # residual planes. This is an exact representation counterexample, NOT
    # an attention failure or a proved BF16 raw/Hadamard-reachable input.
    scale = torch.tensor((1 + 2**-10) * 2**-12, dtype=torch.float16)
    zero = torch.tensor(1 + 2**-10, dtype=torch.float16)
    values = torch.arange(4).repeat(64).float() * scale.float() + zero.float()
    quantized = quantize_int2(values)
    assert torch.equal(quantized.scale, scale.reshape(1))
    assert torch.equal(quantized.zero, zero.reshape(1))
    restored = dequantize_int2(quantized)
    _, two = split_bf16(restored, 2)
    _, three = split_bf16(restored, 3)
    assert bool((two != 0).any()) and bool((three == 0).all())
    return {"status": "two_B_planes_not_universally_exact_in_valid_FP16_metadata_domain",
            "scale_fp16": float(scale), "zero_fp16": float(zero),
            "decoded_levels": restored[:4].tolist(), "two_plane_residuals": two[:4].tolist(),
            "three_plane_residual_max": float(three.abs().max()),
            "quantizer_roundtrip": "same_valid_scale_zero_and_values",
            "scope": "representation_only_not_full_attention_failure; raw_BF16_Hadamard_reachability_not_proved"}


def run():
    torch.set_num_threads(1)
    tolerance = json.loads((ROOT / "configs/acceptance.json").read_text())["fused_attention"]
    results = []
    for name, query, kr, vr, rotation in cases():
        slots = encode_kv(kr, vr)
        key, value = decode_kv(slots, 256)
        gold = compressed_attention(query, slots, rotation, rotation, causal=False)
        baseline = attention(query.float() @ rotation, key, value, causal=False)
        row = {"case": name, "query_shape": list(query.shape), "kv_rows": len(key),
               "CPU_FP32_baseline_output": gate(baseline.output @ rotation.T, gold.output, tolerance),
               "CPU_FP32_baseline_lse": gate(baseline.lse, gold.lse, tolerance), "candidates": []}
        for a, b in ((2, 2), (3, 2)):
            output, lse, detail = candidate(query, key, value, rotation, a, b)
            row["candidates"].append({"A_planes": a, "B_planes": b, **detail,
                "output": gate(output, gold.output, tolerance), "lse": gate(lse, gold.lse, tolerance)})
        results.append(row)
    failed = any(c["output"]["failed_elements"] or c["lse"]["failed_elements"]
                 for row in results for c in row["candidates"])
    return {"status": "screen_failed" if failed else "screen_passed_not_qualified",
            "scope": "CPU BF16-valued products accumulated by torch FP32; not A2 K128/FMA emulation",
            "environment": {"torch": torch.__version__, "system": platform.system()},
            "frozen_gate": tolerance, "cases": results,
            "metadata_plane_representation_bound": metadata_plane_bound(),
            "work_per_M24_KV256_D256": [work_ledger(a,b) for a,b in ((2,2),(3,2))],
            "not_inferences": ["does_not_reject_all_decompositions_or_combine_orders", "not_A2_accuracy_or_speed",
                "two_BF16_planes_exact_on_saved_cases_is_not_a_universal_metadata_domain_proof"],
            "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in (
                "tools/experiments/dequant_bf16_product_screen.py", "tools/experiments/affine_bf16_residual_screen.py",
                "oscar_ascend/ops/reference.py", "configs/acceptance.json")}}


if __name__ == "__main__":
    result = run()
    target = ROOT / "reports/dequant_bf16_product_screen_20261012.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "report": str(target),
        "failures": [(r["case"], c["A_planes"], c["output"]["failed_elements"])
                     for r in result["cases"] for c in r["candidates"]]}))
