"""CPU-only feasibility screen; does not change production or the oracle.

Archive #126/#145/#148 and startup D.4 were consulted. The hypothetical
implementation must cast within bounded KV tiles, never restore all history
in production. This CPU oracle deliberately reconstructs tensors for testing.
BF16 operands are widened to FP32 for CPU multiplication: this isolates input
rounding, not Ascend MMAD rounding, asynchronous execution, or performance.
"""
from pathlib import Path
import hashlib
import json
import platform
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from oscar_ascend.ops.reference import (
    AttentionResult, attention, compressed_attention, decode_kv, encode_kv,
    merge_attention, rotate_clip,
)
from tools.experiments.affine_bf16_residual_screen import cases, gate
from tools.experiments.int8_planes_counterexample import hadamard


def rounded(x):
    return x.bfloat16().float()


def online(query, key, value, mode):
    """KV256 online softmax, FP32 denominator and numerator accumulation."""
    q = query.float()
    k, v = key[:, 0].float(), value[:, 0].float()
    if mode in ("qk_only", "qk_pv"):
        q, k = rounded(q), rounded(k)
    if mode == "kv_only":
        k, v = rounded(k), rounded(v)
    if mode in ("pv_only", "qk_pv"):
        v = rounded(v)
    maximum = torch.full(q.shape[:-1], -float("inf"))
    denominator = torch.zeros_like(maximum)
    numerator = torch.zeros_like(q)
    for start in range(0, len(k), 256):
        scores = torch.einsum("qhd,kd->qhk", q, k[start:start + 256]) / 16
        new_max = torch.maximum(maximum, scores.amax(-1))
        alpha = torch.exp(maximum - new_max)
        weights = torch.exp(scores - new_max[..., None])
        denominator = denominator * alpha + weights.sum(-1)
        operands = rounded(weights) if mode in ("pv_only", "qk_pv") else weights
        numerator = numerator * alpha[..., None] + torch.einsum(
            "qhk,kd->qhd", operands, v[start:start + 256])
        maximum = new_max
    return AttentionResult(numerator / denominator[..., None],
                           maximum + denominator.log())


def extended_cases():
    for name, q, kr, vr, rotation in cases():
        yield name, q, kr, vr, rotation, None, None, None
    for amplitude in (1, 128):
        g = torch.Generator().manual_seed(20261015)
        rotation = hadamard(torch.eye(256))
        q = torch.randn((1, 6, 256), generator=g).bfloat16()
        k = torch.randn((32768, 1, 256), generator=g).bfloat16()
        v = (torch.randn((32768, 1, 256), generator=g) * amplitude).bfloat16()
        mask = torch.zeros(32768, dtype=torch.bool)
        mask[:64] = True
        mask[-256:] = True
        yield (f"hadamard_32k_q1_v{amplitude}_exact320", q,
               rotate_clip(k, rotation), rotate_clip(v, rotation), rotation,
               k, v, mask)


def evaluate():
    torch.set_num_threads(1)
    tolerance = json.loads((ROOT / "configs/acceptance.json").read_text())["fused_attention"]
    rows = []
    for name, q, kr, vr, rotation, raw_k, raw_v, mask in extended_cases():
        slots = encode_kv(kr, vr)
        key, value = decode_kv(slots, 256)
        oracle = compressed_attention(q, slots, rotation, rotation, raw_key=raw_k,
            raw_value=raw_v, bf16_mask=mask, causal=False)
        history = ~mask if mask is not None else torch.ones(len(key), dtype=torch.bool)
        exact = (attention(q, raw_k[mask], raw_v[mask], causal=False)
                 if mask is not None else None)
        row = {"case": name, "query_shape": list(q.shape), "kv_tokens": len(key),
               "exact_tokens": int(mask.sum()) if mask is not None else 0,
               "input_scope": "synthetic; not captured model activations",
               "history_input_rounding": {
                   "key_max_abs": float((key[history] - rounded(key[history])).abs().max()),
                   "value_max_abs": float((value[history] - rounded(value[history])).abs().max())},
               "modes": {}}
        for mode in ("fp32_control", "kv_only", "qk_only", "pv_only", "qk_pv"):
            part = online(q.float() @ rotation, key[history], value[history], mode)
            result = AttentionResult(part.output @ rotation.T, part.lse)
            if exact is not None:
                result = merge_attention([result, exact])
            output = gate(result.output, oracle.output, tolerance)
            lse = gate(result.lse, oracle.lse, tolerance)
            # The actual serving output cast does not automatically erase errors.
            serving = gate(rounded(result.output), rounded(oracle.output), tolerance)
            row["modes"][mode] = {"output": output, "lse": lse,
                                   "both_outputs_cast_to_bf16": serving}
            if mode == "fp32_control":
                assert output["failed_elements"] == lse["failed_elements"] == 0, row
        rows.append(row)
        print(json.dumps({"case": name, "failed_output_elements": {
            m: v["output"]["failed_elements"] for m, v in row["modes"].items()}}), flush=True)
    sources = ["oscar_ascend/ops/reference.py", "configs/acceptance.json",
               "tools/experiments/affine_bf16_residual_screen.py",
               "tools/experiments/int8_planes_counterexample.py",
               "csrc/kernels/attention_cv_mdl_decode.cpp",
               str(Path(__file__).resolve().relative_to(ROOT))]
    return {"scope": "host numerical feasibility only; no NPU kernel or performance measurement",
            "environment": {"torch": torch.__version__, "system": platform.system()},
            "tolerance": tolerance, "modes": {
                "kv_only": "only reconstructed history K/V rounded; Q/P stay FP32 (diagnostic)",
                "qk_only": "history Q/K operands BF16; PV operands FP32",
                "pv_only": "history QK operands FP32; P/V operands BF16",
                "qk_pv": "history Q/K/P/V operands BF16; all accumulation/softmax/rotation/merge FP32"},
            "limitations": ["CPU reduction order is not Ascend MMAD",
                "synthetic samples do not predict model task accuracy or MTP acceptance",
                "32K checks use one independent request, not a TP4 batch32 run",
                "short history-only fixtures are numeric unit cases, not whole serving requests",
                "one split; no asynchronous, graph, prefix-cache or paging validation",
                "exact-window contribution uses unchanged FP32 reference with BF16 inputs"],
            "cases": rows,
            "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}}


if __name__ == "__main__":
    report = evaluate()
    target = Path(__file__).with_name("result.json")
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(f"CPU_SCREEN_COMPLETE report={target}; not NPU qualification", flush=True)
