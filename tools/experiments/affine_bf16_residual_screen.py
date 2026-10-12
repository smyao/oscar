"""Archive #13-22/#126/#148/#154-156 and startup D.4: CPU-only falsification.

Screen one-sided BF16 residual planes against the unchanged ORIGINAL-space
compressed_attention oracle. Exact INT2 codes become BF16 B; Q and P*Vscale
become stacked BF16 A planes. This changes affine/reduction order and is NOT
claimed bit-exact. No native/GDN/kernel change, VM, NPU, or speed claim.
The bounded KV tile proposal avoids full-history restore, but charges every
plane MAC, partial result and affine correction before considering a kernel.
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

from oscar_ascend.ops.reference import (attention, compressed_attention, decode_kv,
                                       encode_kv, QuantizedVector, _le_bytes_to_half,
                                       rotate_clip, unpack_int2)
from tools.experiments.int8_planes_counterexample import hadamard


def split_bf16(value, count):
    remainder = value.float().clone()
    parts = []
    for _ in range(count):
        part = remainder.bfloat16().float()
        parts.append(part)
        remainder = remainder - part
    return parts, remainder


def candidate(query, slots, rotation, count):
    # Read the original serialized codes/metadata, never requantize recovered K/V.
    key, value = [QuantizedVector(slots[..., start:start + 64],
                  _le_bytes_to_half(slots[..., start + 64:start + 66]),
                  _le_bytes_to_half(slots[..., start + 66:start + 68]), 256)
                  for start in (0, 68)]
    kc = unpack_int2(key.packed, 256).float().expand(-1, 6, -1)
    vc = unpack_int2(value.packed, 256).float().expand(-1, 6, -1)
    q = query.float() @ rotation
    qparts, qremaining = split_bf16(q, count)
    dot = torch.einsum("qhd,khd->qhk", qparts[0], kc)
    for part in qparts[1:]:
        dot = dot + torch.einsum("qhd,khd->qhk", part, kc)
    ks, kz = key.scale.flatten().float(), key.zero.flatten().float()
    scores = (dot * ks + q.sum(-1, keepdim=True) * kz) * (1 / 16)
    maximum = scores.amax(-1, keepdim=True)
    weights = torch.exp(scores - maximum)
    vs, vz = value.scale.flatten().float(), value.zero.flatten().float()
    pparts, premaining = split_bf16(weights * vs, count)
    result = torch.einsum("qhk,khd->qhd", pparts[0], vc)
    for part in pparts[1:]:
        result = result + torch.einsum("qhk,khd->qhd", part, vc)
    result = result + (weights * vz).sum(-1, keepdim=True)
    result = (result / weights.sum(-1, keepdim=True)) @ rotation.T
    lse = maximum.squeeze(-1) + torch.log(weights.sum(-1))
    return result, lse, {"q_component_remainder_max": float(qremaining.abs().max()),
                         "p_times_scale_component_remainder_max": float(premaining.abs().max()),
                         "components_finite": bool(torch.isfinite(result).all()),
                         "score_abs_max": float(scores.abs().max())}


def gate(result, expected, tolerance):
    difference = (result - expected).abs()
    allowed = tolerance["atol"] + tolerance["rtol"] * expected.abs()
    failed = (~torch.isfinite(result)) | (difference > allowed)
    selected = torch.where(failed, difference, -1).flatten().argmax() if bool(failed.any()) else difference.flatten().argmax()
    index = int(selected)
    return {"failed_elements": int(failed.sum()), "elements": failed.numel(),
            "max_abs": float(difference.max()), "selected_flat_index": index,
            "reference": float(expected.flatten()[index]), "candidate": float(result.flatten()[index]),
            "difference": float(difference.flatten()[index]), "allowed": float(allowed.flatten()[index])}


def identity_counterexample():
    generator = torch.Generator().manual_seed(1753)
    half = torch.randn((24, 128), generator=generator).bfloat16().float()
    exponents = torch.randint(-16, 17, (24, 128), generator=generator)
    half = (torch.ldexp(half, exponents) / 16384).bfloat16().float()
    query = torch.cat((half, -half), -1).reshape(4, 6, 256)
    base = torch.randint(0, 4, (128,), generator=generator).float()
    base[0], base[1] = 0, 3
    codes = torch.cat((base, base)).repeat(2, 1)
    keys = (codes - 1) * torch.tensor([[16.0], [1.0]])
    vc = torch.ones((2, 256)); vc[:, [0, 2]] = 0; vc[:, [1, 3]] = 3
    values = vc * 2048 + torch.tensor([[0.0], [-4096.0]])
    assert all(torch.equal(x.bfloat16().float(), x) for x in (query, keys, values))
    return "identity_cancellation_seed1753", query, keys[:, None], values[:, None], torch.eye(256)


def cases():
    yield identity_counterexample()
    rotation = hadamard(torch.eye(256))
    for seed, length, amplitude in ((1753, 257, 1), (1931, 257, 128), (2923, 1025, 1)):
        generator = torch.Generator().manual_seed(seed)
        q = torch.randn((4, 6, 256), generator=generator).bfloat16()
        k = torch.randn((length, 1, 256), generator=generator).bfloat16()
        v = (torch.randn((length, 1, 256), generator=generator) * amplitude).bfloat16()
        yield f"hadamard_seed{seed}_kv{length}_v{amplitude}", q, rotate_clip(k, rotation), rotate_clip(v, rotation), rotation


def run():
    torch.set_num_threads(1)
    tolerance = json.loads((ROOT / "configs/acceptance.json").read_text())["fused_attention"]
    results = []
    for name, query, kr, vr, rotation in cases():
        slots = encode_kv(kr, vr)
        golden = compressed_attention(query, slots, rotation, rotation, causal=False)
        dk, dv = decode_kv(slots, 256)
        baseline = attention(query.float() @ rotation, dk, dv, causal=False)
        baseline_output = baseline.output @ rotation.T
        baseline_gate = gate(baseline_output, golden.output, tolerance)
        row = {"case": name, "query_shape": list(query.shape), "kv_rows": len(kr),
               "original_space_oracle": "unchanged_reference.compressed_attention",
               "CPU_rotated_FP32_baseline": baseline_gate, "planes": []}
        for count in (2, 3):
            output, lse, detail = candidate(query, slots, rotation, count)
            row["planes"].append({"count": count, **detail,
                                  "output": gate(output, golden.output, tolerance),
                                  "lse": gate(lse, golden.lse, tolerance)})
        results.append(row)
    # No cost-free pre-extraction, no assumption three separate API calls.
    m, n, k = 24, 256, 256
    work = []
    for count in (2, 3):
        f32 = (m*k*4 + k*n*4 + m*n*4) * 2 * 2
        planes = (m*count*k*2 + k*n*2 + m*count*n*4) * 2 * 2
        work.append({"planes": count, "qk_and_pv_MACs": 2*m*n*k*count,
                     "FP32_baseline_MACs": 2*m*n*k, "stacked_M": m*count,
                     "logical_GM_publish_and_read_bytes": planes, "FP32_baseline_bytes": f32,
                     "nominal_byte_reduction": 1-planes/f32,
                     "Cube_calls_per_KV": 2, "FP32_baseline_Cube_calls_per_KV": 2,
                     "fixed_calls_removed": 0,
                     "unmeasured_additional_work": ["BF16 split/subtract of Q and every P*scale tile",
                         "stacked-M layout and FP32 partial regroup", "zero/scale affine correction",
                         "new error/finite handling", "same softmax, windows, final Rv and flags"],
                     "speed_condition": f"{count}*C/r + exposed_extra < C; r is measured sustainable compute rate, not whole micro ratio"})
    failed = any(p["output"]["failed_elements"] or p["lse"]["failed_elements"] for r in results for p in r["planes"])
    return {"status": "screen_failed" if failed else "screen_passed_not_qualified",
            "scope": "CPU mathematics; FP32 accumulation of BF16-valued operands; not A2 instruction emulation",
            "environment": {"torch": torch.__version__, "system": platform.system()},
            "frozen_gate": tolerance, "cases": results, "work_per_M24_KV256_D256": work,
            "not_inferences": ["does_not_reject_all_BF16_algorithms", "not_device_precision_or_performance",
                "does_not_reproduce_K128_hardware_accumulation_order", "not_graph_or_whole_service"],
            "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in (
                "tools/experiments/affine_bf16_residual_screen.py", "oscar_ascend/ops/reference.py",
                "configs/acceptance.json", "tools/experiments/int8_planes_counterexample.py")}}


if __name__ == "__main__":
    result = run()
    path = ROOT / "reports/affine_bf16_residual_screen_20261012.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "report": str(path),
                      "failures": [(r["case"], p["count"], p["output"]["failed_elements"])
                                   for r in result["cases"] for p in r["planes"]]}))
