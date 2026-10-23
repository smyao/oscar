"""Archive #13-20/#126/#145/#148/#164: stored-level reconstruction evidence.

Real BF16 rows traverse the existing FP32 quantizer and 144-byte codec. The
small CPU examples establish arithmetic only, not NPU or model acceptance.
The v4 oracle and its mode-1 contract remain unchanged.
"""
import json
import math
from pathlib import Path

import pytest
import torch

from tools import official_reference as ref

ROOT = Path(__file__).resolve().parents[1]


def test_overlay_is_distinct_and_inherits_unrelaxed_v4_protocol():
    base = ref.load_acceptance()
    overlay = json.loads((ROOT / "configs/acceptance_fp32_dequant_bf16.json").read_text())
    assert overlay["contract"] == "oscar_bf16_runtime_fp32_dequant_v1"
    assert overlay["kernel_contract"] == base["contract"] == ref.CONTRACT
    assert ROOT / overlay["kernel_acceptance"] == ref.ACCEPTANCE
    assert overlay["frozen_before_measurement"] is True
    assert overlay["dequant_modes"] == dict.fromkeys(("prefill", "decode", "mtp", "capture"), 0)
    assert overlay["rotation_recipe_version"] == 3
    for field in ("cache_format", "slot_bytes_D256", "fused_attention", "protocol"):
        assert overlay[field] == base[field]
    assert base["decode_dequant"].startswith("mode1:")
    assert overlay["model_quality"] == "not_established"


@pytest.mark.parametrize("low,high,rounded_low,rounded_high", [
    (1.0, 1.0078125, 1.0, 1.015625),
    (100.0, 100.5, 100.0, 101.0),
    (-100.5, -100.0, -101.0, -100.0),
])
def test_real_bf16_rows_exact_in_stored_levels_gain_error_from_mode1(
        low, high, rounded_low, rounded_high):
    row = torch.tensor([low] * 128 + [high] * 128, dtype=torch.bfloat16).view(1, 1, 256)
    before = row.clone()
    packed = ref.pack_kv(row, row, k_clip=.96, v_clip=.92)
    assert packed.shape == (1, 1, 144)
    metadata = packed[..., 128:].contiguous().view(torch.float32)
    decoded0 = ref.unpack_kv(packed, 256, 0)
    decoded1 = ref.unpack_kv(packed, 256, 1)
    for side, ratio in enumerate((.96, .92)):
        codes, scale, zero = ref.quantize_rows(row, ratio)
        assert codes.unique().tolist() == [0, 3]
        assert torch.equal(metadata[..., 2 * side], scale)
        assert torch.equal(metadata[..., 2 * side + 1], zero)
        # FP64 here independently evaluates the levels DEFINED BY the stored
        # FP32 metadata; no invented scale/zero and no replacement quantizer.
        levels = (codes.double() - zero.double()[..., None]) * scale.double()[..., None]
        assert torch.equal(levels.bfloat16(), row)
        assert torch.equal(decoded0[side].bfloat16(), row)
        assert decoded1[side].unique().tolist() == [rounded_low, rounded_high]
        assert (decoded1[side] - row.float()).abs().amax().item() == high - low
    assert torch.equal(row, before)


def test_two_key_attention_tracks_stored_levels_without_extra_mode1_error():
    keys = torch.zeros((2, 1, 256), dtype=torch.bfloat16)
    keys[0, 0, :128] = 1.0
    keys[0, 0, 128:] = 1.0078125
    values = torch.ones_like(keys)
    values[1].neg_()
    packed = ref.pack_kv(keys, values, .96, .92)
    codes, scale, zero = ref.quantize_rows(keys, .96)
    stored_keys = ((codes.double() - zero.double()[..., None]) * scale.double()[..., None]).bfloat16()
    # Q has only one nonzero BF16 coordinate; the model's D**-.5 scaling
    # makes the logits exactly [that key coordinate, 0]. Scalar softmax is
    # independent of the CV/merge oracle and deliberately isolates read error.
    query = torch.zeros(256, dtype=torch.bfloat16)
    query[-1] = 16

    def scalar_attention(k, v):
        logits = [sum(float(q) * float(x) for q, x in zip(query, row)) / 16 for row in k[:, 0]]
        exponentials = [math.exp(x - max(logits)) for x in logits]
        return sum(p * float(v[i, 0, 0]) for i, p in enumerate(exponentials)) / sum(exponentials)

    expected = scalar_attention(stored_keys, values)
    key0, value0 = ref.unpack_kv(packed, 256, 0)
    key1, value1 = ref.unpack_kv(packed, 256, 1)
    assert torch.equal(value0.bfloat16(), values)
    assert torch.equal(value1.bfloat16(), values)
    assert scalar_attention(key0, value0) == expected
    assert expected == pytest.approx(math.tanh(1.0078125 / 2), abs=1e-15)
    assert scalar_attention(key1, value1) == pytest.approx(math.tanh(1.015625 / 2), abs=1e-15)
    assert abs(scalar_attention(key1, value1) - expected) > .003
