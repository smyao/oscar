"""Archive #13-22/#127-128/#157/#170/#172-177; startup D.4.

Independent CPU evidence for the active OSCAR recipe3 path (OPTR disabled).
These deliberately constructed examples distinguish finite-precision/INT2
error from a kernel defect; they do not measure model quality or NPU speed.
No acceptance tolerance or production algorithm is changed.
"""
from types import SimpleNamespace

import pytest
import torch

from oscar_ascend.integration.official_math import _center_key
from tools import official_reference as ref


def _dense(q, k, v):
    """Independent FP64 softmax for the supplied already-BF16 operands."""
    score = torch.einsum("qhd,khd->qhk", q.double(), k.double()) / 16
    return torch.einsum("qhk,khd->qhd", score.softmax(-1), v.double()).float()


def centered_collision_evidence():
    q = torch.zeros((1, 1, 256), dtype=torch.bfloat16)
    q[..., 0] = 64
    k = torch.zeros((2, 1, 256), dtype=torch.bfloat16)
    k[:, 0, 0] = torch.tensor([1., 1.0078125])
    v = torch.ones_like(k)
    v[0].neg_()
    mean = torch.zeros(256, dtype=torch.bfloat16)
    mean[0] = -1
    state = SimpleNamespace(key_mean_bf16=mean, workspace=SimpleNamespace(
        centered_key_bf16=torch.empty_like(k)))
    centered = _center_key(state, k, torch)
    original = _dense(q, k, v)
    exact_centered = _dense(q, k.float() - mean.float(), v)
    bf16_centered = _dense(q, centered, v)
    return {
        "key_before": k[:, 0, 0].float().tolist(),
        "key_after_bf16_center": centered[:, 0, 0].float().tolist(),
        "mean": float(mean[0]),
        "original_output": float(original[0, 0, 0]),
        "exact_centered_output": float(exact_centered[0, 0, 0]),
        "bf16_centered_output": float(bf16_centered[0, 0, 0]),
        "max_abs_output_error": float((bf16_centered - original).abs().max()),
        "int2_used": False, "value_rotation": "identity",
    }


def test_bf16_centering_alone_can_erase_a_real_logit_difference():
    evidence = centered_collision_evidence()
    assert evidence["key_before"][0] != evidence["key_before"][1]
    assert evidence["key_after_bf16_center"] == [2., 2.]
    assert evidence["exact_centered_output"] == evidence["original_output"]
    assert .015 < evidence["max_abs_output_error"] < .016
    assert evidence["bf16_centered_output"] == 0


def long_history_evidence(context):
    sink, recent = 64, 256
    cut = max(min(sink, context), context + 1 - recent)
    q = torch.zeros((1, 1, 256), dtype=torch.bfloat16)
    q[..., 0] = 16
    k = torch.zeros((context + 1, 1, 256), dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    rows = torch.arange(sink, cut)
    k[rows, 0, 128:] = 1
    k[rows, 0, 0] = torch.where(rows % 2 == 0, .4, .6).bfloat16()
    v[rows, 0] = torch.where(rows % 2 == 0, -1., 1.)[:, None].bfloat16()
    hk, hv = ref.unpack_kv(ref.pack_kv(k[sink:cut], v[sink:cut], .96, .92), 256, 0)
    deployed_k, deployed_v = k.float().clone(), v.float().clone()
    deployed_k[sink:cut], deployed_v[sink:cut] = hk, hv
    source_parts = [
        ref.attention_tiles(q, hk, hv),
        ref.attention_tiles(q, torch.cat((k[:sink], k[cut:context])),
                           torch.cat((v[:sink], v[cut:context]))),
        ref.attention_tiles(q, k[context:], v[context:]),
    ]
    tiled = ref.merge_rotated_parts(source_parts).output
    dense_original = _dense(q, k, v)
    dense_deployed = _dense(q, deployed_k, deployed_v)
    final = tiled.bfloat16().float()  # Identity Rv isolates the KV/PV errors.
    return {
        "context": context, "sink": sink, "recent": recent,
        "quantized_rows": len(rows), "history_tiles": (len(rows) + 255) // 256,
        "dense_original_output": float(dense_original[0, 0, 0]),
        "dense_quantized_output": float(dense_deployed[0, 0, 0]),
        "kv256_three_source_output": float(tiled[0, 0, 0]),
        "final_bf16_output": float(final[0, 0, 0]),
        "max_key_coordinate_error": float((hk - k[sink:cut].float()).abs().max()),
        "max_value_coordinate_error": float((hv - v[sink:cut].float()).abs().max()),
        "int2_output_error": float((dense_deployed - dense_original).abs().max()),
        "tile_and_merge_output_error": float((tiled - dense_deployed).abs().max()),
        "total_output_error": float((final - dense_original).abs().max()),
    }


@pytest.mark.parametrize("context", [1024, 4096, 16384])
def test_correlated_int2_error_can_dominate_kv256_rounding(context):
    evidence = long_history_evidence(context)
    assert evidence["history_tiles"] >= 3
    assert evidence["max_value_coordinate_error"] == 0
    assert evidence["int2_output_error"] > .04
    assert evidence["tile_and_merge_output_error"] < .002
    assert evidence["int2_output_error"] > 50 * evidence["tile_and_merge_output_error"]


def boundary_evidence():
    context, sink, recent, width = 4096, 64, 256, 4
    q = torch.zeros((1, 1, 256), dtype=torch.bfloat16)
    q[..., 0] = 16
    k = torch.zeros((context + width, 1, 256), dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    cut = context + 1 - recent
    k[cut] = 16  # Constant K is exactly representable by the affine codec.
    v[cut, 0, 0] = 1  # Fixed .92 clipping removes this isolated coordinate.
    decoded_k, decoded_v = ref.unpack_kv(ref.pack_kv(k, v, .96, .92), 256, 0)

    def evaluate(forward_context):
        boundary = max(sink, forward_context + 1 - recent)
        query_position = context + width - 1
        indices = (torch.arange(sink, boundary),
                   torch.cat((torch.arange(sink), torch.arange(boundary, forward_context))),
                   torch.arange(forward_context, query_position + 1))
        all_indices = torch.cat(indices)
        assert torch.equal(all_indices.sort().values, torch.arange(query_position + 1))
        parts = [ref.attention_tiles(q, decoded_k[indices[0]], decoded_v[indices[0]]),
                 ref.attention_tiles(q, k[indices[1]], v[indices[1]]),
                 ref.attention_tiles(q, k[indices[2]], v[indices[2]])]
        out = ref.merge_rotated_parts(parts).output.bfloat16().float()
        return boundary, float(out[0, 0, 0])

    q4_cut, q4 = evaluate(context)
    q1_cut, q1 = evaluate(context + width - 1)
    return {"context": context, "query_position": context + width - 1,
            "q4_cut": q4_cut, "last_q1_cut": q1_cut,
            "distinguished_position": cut,
            "q4_output": q4, "last_q1_output": q1,
            "same_visible_positions_exactly_once": True,
            "decoded_key_error": float((decoded_k[cut] - k[cut]).abs().max()),
            "decoded_value": float(decoded_v[cut, 0, 0])}


def test_q4_fixed_window_and_four_q1_steps_have_different_precision_membership():
    evidence = boundary_evidence()
    assert evidence["decoded_key_error"] == 0
    assert evidence["decoded_value"] == 0
    assert evidence["last_q1_cut"] - evidence["q4_cut"] == 3
    assert evidence["q4_output"] > .99
    assert evidence["last_q1_output"] == 0
