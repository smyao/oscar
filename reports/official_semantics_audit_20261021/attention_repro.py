"""CPU-only mathematical counterexamples, not model/NPU acceptance.

Archive #13-20: matching LSE does not prove matching output; #126/#145:
CPU arithmetic cannot establish asynchronous device buffer correctness;
#148: an operator oracle pass does not establish LongBench accuracy;
#164: BF16 cast placement is part of the declared numerical contract.
D.4: this is offline fixed-size analysis of fused FIA/store semantics. The
rejected 6499.8-6655.1ms full-history restore is not a proposed serving path;
no production memory allocation, timing, NPU run, or performance claim here.
Upstream pin: 38680852e52c10bd6b9e9a64ed9bc815e742859b.
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.aligned_reference import pack_kv, unpack_kv, rotate_bf16
from tools.bf16_reference import attention_tiles, merge_parts, round_bf16

HERE = Path(__file__).resolve().parent
UP = HERE / "upstream/python/sglang/srt/layers/attention"


def original_rotation_functions():
    """Execute unchanged pure-torch upstream functions, without importing GPU code."""
    path = UP / "quantized_kv_prefill.py"
    names = {"_pool_uses_oscar_rotation", "oscar_q_rotation", "oscar_o_rotation",
             "oscar_k_mean", "_apply_oscar_rotation", "prepare_quantized_extend_qkv",
             "apply_inverse_v_rotation"}
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    ns = {"torch": torch, "Optional": Optional, "Tuple": Tuple}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    return ns


def pair_rotation():
    original = torch.eye(256)
    original[:2, :2] = torch.tensor([[0.6, -0.8], [0.8, 0.6]])
    return original, original.bfloat16()


def source_provenance():
    paths = [
        UP / "quantized_kv_prefill.py", UP / "triton_backend.py",
        UP / "triton_ops/decode_attention.py",
        ROOT / "csrc/kernels/store_aligned_int2.cpp",
        ROOT / "csrc/kernels/attention_cv_aligned_decode.cpp",
        ROOT / "oscar_ascend/integration/current_only_dispatch.py",
        ROOT / "tools/bf16_reference.py",
        HERE / "upstream/python/sglang/QuantKernel/oscar_rotate_rows.py",
    ]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def main():
    torch.set_num_threads(1)
    upstream = original_rotation_functions()
    identity = torch.eye(256, dtype=torch.bfloat16)
    original, rotation = pair_rotation()
    layer = SimpleNamespace(layer_id=0, oscar_v_rotation_absorbed=False)
    pool = SimpleNamespace(dtype="int2", start_layer=0, _R_k=rotation[None], _R_v=rotation[None])
    q = torch.zeros(1, 1, 256, dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    v[..., :2] = 1
    oq, ok, ov, inverse = upstream["prepare_quantized_extend_qkv"](pool, layer, q, k, v)
    # For a single visible exact token attention returns its V exactly.
    official = upstream["apply_inverse_v_rotation"](ov, pool, layer, inverse).float()
    exact_case = {
        "shape": {"queries": 1, "kv_tokens": 1, "head_dim": 256},
        "input_v_first2": v.flatten()[:2].tolist(),
        "local_original_space_output_first2": v.float().flatten()[:2].tolist(),
        "official_rotated_space_output_first2": official.flatten()[:2].tolist(),
        "max_abs_difference": (official - v.float()).abs().max().item(),
        "original_fp32_rotation_orthogonality_error": (original @ original.T - torch.eye(256)).abs().max().item(),
        "rounded_bf16_rotation_orthogonality_error": (rotation.float() @ rotation.float().T - torch.eye(256)).abs().max().item(),
        "compression_used": False,
    }
    assert exact_case["max_abs_difference"] == 0.0078125

    # Isolate key/query rotation from V/output rotation.
    pool._R_v = identity[None]
    q[..., :2] = 4
    k = torch.zeros(2, 1, 256, dtype=torch.bfloat16)
    k[0, 0, :2] = torch.tensor([4, 4], dtype=torch.bfloat16)
    k[1, 0, :2] = torch.tensor([4, -4], dtype=torch.bfloat16)
    v = torch.stack([torch.ones(1, 256), -torch.ones(1, 256)]).bfloat16()
    oq, ok, ov, _ = upstream["prepare_quantized_extend_qkv"](pool, layer, q, k, v)
    raw_logits = (q[0, 0].float() @ k[:, 0].float().T) / 16
    rotated_logits = (oq[0, 0].float() @ ok[:, 0].float().T) / 16
    raw_answer = attention_tiles(q, k, v)
    rotated_answer = attention_tiles(oq, ok, ov)
    qk_case = {
        "raw_logits": raw_logits.tolist(), "rotated_logits": rotated_logits.tolist(),
        "raw_attention_output": raw_answer.output[0, 0, 0].item(),
        "rotated_attention_output": rotated_answer.output[0, 0, 0].item(),
        "raw_lse": raw_answer.lse.item(), "rotated_lse": rotated_answer.lse.item(),
        "compression_used": False,
        "attention_implementation": "same declared BF16 online-tile oracle on both sides; not GPU or NPU execution",
    }
    assert qk_case["raw_logits"] != qk_case["rotated_logits"]

    # Identical true keys receive different logits solely because one is
    # history (rotated) and one is exact/current (original frame). This is
    # stronger than a global rotation-roundtrip difference: the INT2 history
    # roundtrip is exact for these two-level K rows in BOTH codec modes.
    repeated_rotation = torch.block_diag(*([original[:2, :2]] * 128)).bfloat16()
    same_q = torch.ones(1, 1, 256, dtype=torch.bfloat16)
    same_k = same_q.clone()
    same_qr = rotate_bf16(same_q, repeated_rotation)
    same_kr = rotate_bf16(same_k, repeated_rotation)
    positive_v, negative_v = torch.ones_like(same_k), -torch.ones_like(same_k)
    domain_bias = []
    for mode in (0, 1):
        hist_k, hist_v = unpack_kv(pack_kv(same_kr, positive_v), 256, mode)
        assert torch.equal(hist_k, same_kr.float())
        history = attention_tiles(same_qr, hist_k, hist_v, value_rotation=identity)
        exact_local = attention_tiles(same_q, same_k, negative_v)
        exact_official = attention_tiles(same_qr, same_kr, negative_v)
        local = round_bf16(merge_parts([history, exact_local]).output)
        official = round_bf16(merge_parts([history, exact_official]).output)
        domain_bias.append({
            "mode": mode, "same_original_keys": True,
            "int2_kv_roundtrip_exact": bool(torch.equal(hist_k, same_kr.float()) and torch.equal(hist_v, positive_v.float())),
            "history_logit": history.lse.item(), "local_exact_logit": exact_local.lse.item(),
            "official_exact_logit": exact_official.lse.item(),
            "local_output": local[0, 0, 0].item(), "official_output": official[0, 0, 0].item(),
            "max_abs_difference": (local - official).abs().max().item(),
        })
        assert domain_bias[-1]["official_output"] == 0
        assert domain_bias[-1]["local_output"] == 0.06884765625

    # A real five-token INT2 fixture with two ceil-divided source splits 3+2.
    # Uniform logits remove softmax-kernel variation; all V values survive
    # the constant-row INT2 codec. Only partial-cast/rotation placement differs.
    q = torch.zeros(1, 1, 256, dtype=torch.bfloat16)
    k = torch.zeros(5, 1, 256, dtype=torch.bfloat16)
    v = torch.ones_like(k)
    v[2].fill_(1.015625)
    packed = pack_kv(k, v)
    modes = []
    for mode in (0, 1):
        decoded_k, decoded_v = unpack_kv(packed, 256, mode)
        assert torch.equal(decoded_v, v.float())
        local_parts = [attention_tiles(q, decoded_k[a:b], decoded_v[a:b], value_rotation=identity)
                       for a, b in ((0, 3), (3, 5))]
        official_parts = [attention_tiles(q, decoded_k[a:b], decoded_v[a:b])
                          for a, b in ((0, 3), (3, 5))]
        local = round_bf16(merge_parts(local_parts).output)
        # Official stage-2 merges rotated FP32 partials into BF16 O, followed
        # by one BF16 inverse rotation. R=I makes the final GEMM exact here.
        official_rotated = round_bf16(merge_parts(official_parts).output)
        official = round_bf16(official_rotated @ identity.float().T)
        single_local = round_bf16(attention_tiles(q, decoded_k, decoded_v, value_rotation=identity).output)
        modes.append({
            "mode": mode, "split_lengths": [3, 2],
            "unrounded_partial_first_component": [p.output[0, 0, 0].item() for p in official_parts],
            "local_split_output": local[0, 0, 0].item(),
            "official_merge_then_inverse_output": official[0, 0, 0].item(),
            "local_one_split_output": single_local[0, 0, 0].item(),
            "max_abs_difference": (local - official).abs().max().item(),
        })
        assert modes[-1]["max_abs_difference"] == 0.0078125

    # The local GQA head mapping agrees for the V1 shared matrix; a V2 matrix
    # is a different optional artifact capability, not evidence of bad V1 GQA.
    qmulti = torch.arange(1 * 12 * 256).reshape(1, 12, 256).remainder(17).bfloat16()
    q_shared = upstream["_apply_oscar_rotation"](qmulti, rotation)
    q_repeated = upstream["_apply_oscar_rotation"](qmulti, rotation.expand(2, 256, 256).contiguous(), 6)
    assert torch.equal(q_shared, q_repeated)

    # Fixed upstream prefix-slot tier versus per-query local recent cut.
    # Upstream allocation/flush chooses the fixed tier before this forward;
    # this case enumerates BOTH possible tiers, without guessing its live plan.
    context, target, sink, recent = 512, 300, 64, 256
    special_v = torch.zeros(1, 1, 256, dtype=torch.bfloat16)
    special_v[..., 0] = 1
    _, compressed_v = unpack_kv(pack_kv(torch.zeros_like(special_v), special_v), 256, 0)
    assert compressed_v.abs().max().item() == 0
    cut_cases = []
    for qpos in (512, 612):
        cut = max(sink, qpos + 1 - recent)
        is_exact = target < sink or target >= cut
        local_value = special_v.float() if is_exact else compressed_v
        # With every Q/K zero, all qpos+1 causal keys have equal weight.
        cut_cases.append({"query_position": qpos, "prefix_token": target,
                          "cut": cut, "local_tier": "exact" if is_exact else "int2",
                          "local_output_first_component": round_bf16(local_value / (qpos + 1)).flatten()[0].item(),
                          "upstream_if_fixed_hp": round_bf16(special_v.float() / (qpos + 1)).flatten()[0].item(),
                          "upstream_if_fixed_int2": 0.0})

    report = {
        "classification": "artificial CPU mathematical counterexamples; no attribution of the reported 5 LongBench points",
        "upstream_commit": "38680852e52c10bd6b9e9a64ed9bc815e742859b",
        "production_edits": False, "npu_execution": False, "performance_measured": False,
        "exact_token_rotation_domain": exact_case,
        "query_key_rotation_domain": qk_case,
        "same_keys_history_exact_domain_bias": domain_bias,
        "partial_inverse_before_merge": modes,
        "v1_shared_rotation_gqa_12q_2kv_ratio6": "same output as repeated per-head matrices",
        "prefill_prefix_precision_cut": {"context": context, "sink": sink, "recent": recent,
            "cases": cut_cases, "upstream_live_tier": "not inferred; both possible fixed classifications shown",
            "meaning": "within one forward upstream dequantizes prefix once; local chooses prefix precision separately for each query"},
        "sources_sha256": source_provenance(),
    }
    out = HERE / "attention_repro.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(out), "exact_domain_delta": exact_case["max_abs_difference"],
                      "split_deltas": [x["max_abs_difference"] for x in modes]}))


if __name__ == "__main__":
    main()
