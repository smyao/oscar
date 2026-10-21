"""Archive G25-G34/#13-20/#126/#145/#164 and startup D.4: CPU semantics only.

No production imports, NPU, performance test, or model-quality claim. The
official Torch quantizer and clip-index function are extracted from the pinned
upstream files; remaining formulas isolate documented floating-point order.
"""
import ast
import json
import math
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parent
UP = OUT / "upstream"


def upstream_function(relative, name):
    path = UP / relative
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def bf16(x):
    return x.to(torch.bfloat16).float()


def local_writer_formula(rows, ratio):
    # Transcribed from store_aligned_int2.cpp:39-79, not aligned_reference.py.
    x = rows.float()
    if ratio > 0:
        idx = min(int(ratio * x.shape[-1]), x.shape[-1] - 1)
        threshold = x.abs().sort(-1).values[..., idx, None]
        x = torch.minimum(torch.maximum(x, -threshold), threshold)
    low, high = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
    scale = (high - low).clamp_min(1e-8) / 3
    zero = -low / scale
    code = (x / scale + zero + 0.5).clamp(0, 3).to(torch.uint8)
    return code, scale, zero


def codec_checks():
    quantize = upstream_function(
        "python/sglang/srt/mem_cache/kv_quant_kernels.py", "_groupwise_affine_quantize")
    index = upstream_function(
        "python/sglang/QuantKernel/oscar_rotation_clip_int2_kv.py", "_clip_index")
    g = torch.Generator().manual_seed(20261021)
    rows = torch.randn((8, 2, 256), generator=g).bfloat16()
    cases = []
    for ratio in [0, .92, .96, 1]:
        idx = index(ratio, 256)
        clipped = rows.float()
        if idx >= 0:
            threshold = clipped.abs().sort(-1).values[..., idx, None]
            clipped = clipped.clamp(-threshold, threshold)
        official = quantize(clipped, 256, 3)
        local = local_writer_formula(rows, ratio)
        equal = [torch.equal(a, b) for a, b in zip(official, local)]
        assert all(equal)
        cases.append({"clip_ratio": ratio, "sort_index": idx,
                      "codes_scale_zero_equal": equal})
    ties = torch.zeros((1, 1, 256), dtype=torch.bfloat16)
    ties[0, 0, :5] = torch.tensor([0, .5, 1.5, 2.5, 3])
    actual, _, _ = quantize(ties, 256, 3)
    assert actual[0, 0, :5].tolist() == [0, 1, 2, 3, 3]
    return {"scope": "uniform_group256_given_same_rotated_BF16_input",
            "checks": cases, "half_ties_inputs": ties[0, 0, :5].float().tolist(),
            "official_half_ties_codes": actual[0, 0, :5].tolist(),
            "not_a_test_of": "AscendC or Triton GPU instruction execution"}


def inverse_placement():
    # Two rotated-domain split outputs. D2 is an isolated subspace example;
    # embedding this block in D256 preserves the same mismatch.
    partial = torch.tensor([[-.5390625, .99609375], [.890625, -1.328125]])
    rotation = bf16(torch.tensor([[1., 1.], [1., -1.]]) / math.sqrt(2))
    weights = torch.tensor([.3, .7])
    local = bf16(((bf16(partial) @ rotation.T) * weights[:, None]).sum(0))
    official = bf16(bf16((partial * weights[:, None]).sum(0)) @ rotation.T)
    assert not torch.equal(local, official)
    return {"partial": partial.tolist(), "rotation_bf16": rotation.tolist(),
            "normalized_LSE_weights": weights.tolist(),
            "local_inverse_each_partial_then_merge": local.tolist(),
            "upstream_merge_then_inverse": official.tolist(),
            "max_abs": float((local - official).abs().max()),
            "scope": "isolated_order_counterexample_no_claim_of_model_loss"}


def tile_rounding():
    g = torch.Generator().manual_seed(1)
    scores = bf16(torch.randn(128, generator=g)) * 2
    values = bf16(torch.randn((128, 4), generator=g))

    def online(tile):
        maximum, total = torch.tensor(-torch.inf), torch.tensor(0.)
        output = torch.zeros(4)
        for begin in range(0, len(scores), tile):
            score = scores[begin:begin + tile]
            new_max = torch.maximum(maximum, score.max())
            alpha = torch.exp(maximum - new_max)
            probability = torch.exp(score - new_max)
            total = total * alpha + probability.sum()
            output = output * alpha + bf16(probability) @ values[begin:begin + tile]
            maximum = new_max
        return output / total

    local, official = online(256), online(64)
    assert not torch.equal(bf16(local), bf16(official))
    return {"seed": 1, "scores": scores.tolist(), "values": values.tolist(),
            "local_tile256": bf16(local).tolist(), "upstream_tile64": bf16(official).tolist(),
            "FP32_partial_max_abs": float((local - official).abs().max()),
            "BF16_output_max_abs": float((bf16(local) - bf16(official)).abs().max()),
            "scope": "same_QK_scores_and_V_isolated_online_probability_rounding"}


if __name__ == "__main__":
    report = {"status": "cpu_formula_checks_complete", "device": "cpu",
              "upstream_commit": json.loads((OUT / "upstream_manifest.json").read_text())["resolved_commit"],
              "quality_acceptance": "not_measured", "performance": "not_measured",
              "uniform_codec": codec_checks(), "inverse_placement": inverse_placement(),
              "online_tile_rounding": tile_rounding()}
    output = OUT / "codec_counterexamples.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "status": report["status"],
        "inverse_order_max_abs": report["inverse_placement"]["max_abs"],
        "tile_BF16_max_abs": report["online_tile_rounding"]["BF16_output_max_abs"]}))
