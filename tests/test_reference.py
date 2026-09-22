"""CPU oracle tests, not target-NPU acceptance.

Archive G26/G27,#13-#20: independent PR byte and fp16 metadata expectations.
G30-G34,#6-#12: causal/GQA LSE and empty-row coverage. PR 46774 test_oscar.py
provides store 2e-3, decode 5e-3 and prefill/window relative-L2 2e-2 gates.
"""

import math
import struct

import pytest
import torch

from oscar_ascend.ops.reference import (
    AttentionResult, attention, compressed_attention, decode_kv, dequantize_int2,
    encode_kv, merge_attention, pack_int2, quantize_int2, rotate_clip,
    unpack_int2, window_ranges,
)


def _scalar_pr_quantizer(values):
    """Independent scalar transcription of PR store:41-77, using struct fp16."""
    low, high = min(values), max(values)
    scale = struct.unpack("<e", struct.pack("<e", max((high - low) / 3, 1e-8)))[0]
    zero = struct.unpack("<e", struct.pack("<e", low))[0]
    codes = [min(3, max(0, int((x - zero) / scale + 0.5))) for x in values]
    encoded = bytearray()
    for base in range(0, len(codes), 4):
        encoded.append(sum(code << (2 * offset) for offset, code in enumerate(codes[base:base + 4])))
    encoded.extend(struct.pack("<ee", scale, zero))
    return list(encoded), [code * scale + zero for code in codes]


@pytest.mark.parametrize("dimension", [1, 3, 4, 5, 63, 64, 128, 256])
def test_lsb_pack_unpack_with_zero_tail_bits(dimension):
    indices = (torch.arange(2 * dimension).reshape(2, dimension) % 4).to(torch.int32)
    packed = pack_int2(indices)
    assert packed.dtype == torch.uint8
    torch.testing.assert_close(unpack_int2(packed, dimension).int(), indices, atol=0, rtol=0)
    if dimension >= 4:
        assert packed[0, 0].item() == 0b11100100
    if dimension % 4:
        assert bool((packed[..., -1].int() >> (2 * (dimension % 4)) == 0).all())


def test_pack_rejects_out_of_range_and_floating_codes():
    with pytest.raises(ValueError, match="integer"):
        pack_int2(torch.tensor([0.0, 1.0]))
    with pytest.raises(ValueError, match=r"\[0, 3\]"):
        pack_int2(torch.tensor([-1, 4]))


@pytest.mark.parametrize("values", [
    [-0.7, -0.02, 0.17, 0.41, 1.11],
    [0.0, 0.5, 1.5, 2.5, 3.0],
    [-1.001, -0.7005, -0.4001, -0.1000],
])
def test_store_is_scalar_pr_exact_with_fp16_metadata_first(values):
    k = torch.tensor(values, dtype=torch.float32)
    v = torch.tensor(list(reversed(values)), dtype=torch.float32)
    # Read fp32-rounded inputs in the independent oracle, matching PR loads.
    kb, kd = _scalar_pr_quantizer(k.tolist())
    vb, vd = _scalar_pr_quantizer(v.tolist())
    slots = encode_kv(k, v)
    assert slots.tolist() == kb + vb
    out_k, out_v = decode_kv(slots, len(values))
    torch.testing.assert_close(out_k, torch.tensor(kd), atol=0, rtol=0)
    torch.testing.assert_close(out_v, torch.tensor(vd), atol=0, rtol=0)


def test_half_integer_bins_are_not_round_to_even():
    result = quantize_int2(torch.tensor([0.0, 0.5, 1.5, 2.5, 3.0]))
    assert unpack_int2(result.packed, 5).tolist() == [0, 1, 2, 3, 3]


@pytest.mark.parametrize("value", [torch.zeros(8), torch.full((8,), 1.0),
                                   torch.tensor([0.0, 1e-9]),
                                   torch.tensor([0.0, float("nan")]),
                                   torch.tensor([-1e10, 1e10])])
def test_undefined_or_nonfinite_metadata_domain_fails(value):
    with pytest.raises(ValueError):
        quantize_int2(value)


@pytest.mark.parametrize("dimension", [64, 128, 256])
def test_store_dequant_matches_pr_dense_oracle(dimension):
    generator = torch.Generator().manual_seed(93)
    x = torch.randn(7, 2, dimension, generator=generator)
    low = x.amin(-1, keepdim=True).half().float()
    scale = ((x.amax(-1, keepdim=True) - x.amin(-1, keepdim=True)) / 3).clamp_min(1e-8).half().float()
    expected = ((x - low) / scale + 0.5).int().clamp(0, 3).float() * scale + low
    torch.testing.assert_close(dequantize_int2(quantize_int2(x)), expected, atol=2e-3, rtol=2e-3)


def test_rotation_then_abs_percentile_clip():
    x = torch.tensor([[1.0, 2.0, 4.0, 20.0]])
    rotation = torch.tensor([[0., 1., 0., 0.], [1., 0., 0., 0.],
                             [0., 0., 0., -1.], [0., 0., 1., 0.]])
    # Rotation gives [2,1,20,-4]; median absolute threshold is (2+4)/2.
    torch.testing.assert_close(rotate_clip(x, rotation, .5), torch.tensor([[2., 1., 3., -3.]]))
    torch.testing.assert_close(rotate_clip(x, rotation, 0), x @ rotation)


@pytest.mark.parametrize("length", [0, 1, 63, 64, 65, 319, 320, 321, 16384, 32768, 50000, 262144])
def test_request_windows_partition_without_negative_or_overlap(length):
    sink, history, recent = window_ranges(length)
    assert sink == (0, min(64, length))
    assert sink[1] == history[0] <= history[1] == recent[0] <= recent[1] == length
    assert recent[1] - recent[0] == min(256, length - sink[1])


@pytest.mark.parametrize("queries,keys,hq,hk", [(1, 17, 4, 1), (4, 31, 8, 2), (11, 11, 4, 2)])
def test_causal_gqa_matches_independent_sdpa(queries, keys, hq, hk):
    torch.manual_seed(17)
    query = torch.randn(queries, hq, 8)
    key, value = torch.randn(keys, hk, 8), torch.randn(keys, hk, 8)
    mask = torch.arange(keys)[None, :] <= torch.arange(keys - queries, keys)[:, None]
    expected = torch.nn.functional.scaled_dot_product_attention(
        query.transpose(0, 1), key.repeat_interleave(hq // hk, 1).transpose(0, 1),
        value.repeat_interleave(hq // hk, 1).transpose(0, 1), attn_mask=mask).transpose(0, 1)
    actual = attention(query, key, value)
    torch.testing.assert_close(actual.output, expected, atol=5e-3, rtol=5e-3)
    scores = torch.einsum("qhd,khd->qhk", query, key.repeat_interleave(hq // hk, 1)) / math.sqrt(8)
    scores.masked_fill_(~mask[:, None, :], -math.inf)
    torch.testing.assert_close(actual.lse, scores.logsumexp(-1), atol=1e-6, rtol=1e-6)


def test_segment_merge_matches_dense_and_handles_empty_rows():
    torch.manual_seed(19)
    q, k, v = torch.randn(4, 4, 8), torch.randn(17, 2, 8), torch.randn(17, 2, 8)
    positions = torch.tensor([-1, 1, 5, 16])
    full = attention(q, k, v, query_positions=positions)
    parts = [attention(q, k[a:b], v[a:b], query_positions=positions,
                       key_positions=torch.arange(a, b)) for a, b in ((0, 3), (3, 11), (11, 17))]
    merged = merge_attention(parts)
    torch.testing.assert_close(merged.output, full.output, atol=5e-6, rtol=5e-6)
    torch.testing.assert_close(merged.lse, full.lse, atol=1e-6, rtol=1e-6)
    assert bool((merged.output[0] == 0).all())
    assert bool(torch.isneginf(merged.lse[0]).all())
    empty = attention(q, k[:0], v[:0])
    torch.testing.assert_close(merge_attention([empty, full]).output, full.output)


def test_merge_never_hides_nonfinite_numerics():
    with pytest.raises(ValueError, match="LSE"):
        merge_attention([AttentionResult(torch.zeros(1, 1, 2), torch.tensor([[float("nan")]]))])


def test_compressed_rotated_decode_and_prefill_with_exact_windows():
    torch.manual_seed(27)
    dim, count = 8, 33
    rk = torch.linalg.qr(torch.randn(dim, dim)).Q
    rv = torch.linalg.qr(torch.randn(dim, dim)).Q
    key, value = torch.randn(count, 2, dim).bfloat16(), torch.randn(count, 2, dim).bfloat16()
    slots = encode_kv(rotate_clip(key, rk, .96), rotate_clip(value, rv, .92))
    kd, vd = decode_kv(slots, dim)
    mask = torch.arange(count) < 3
    mask |= torch.arange(count) >= count - 5
    expected_k = torch.where(mask[:, None, None], key.float(), kd @ rk.T)
    expected_v = torch.where(mask[:, None, None], value.float(), vd @ rv.T)
    for nq in (1, 4, count):
        query = torch.randn(nq, 4, dim)
        result = compressed_attention(query, slots, rk, rv, raw_key=key, raw_value=value, bf16_mask=mask)
        expected = attention(query, expected_k, expected_v)
        torch.testing.assert_close(result.output, expected.output, atol=5e-3, rtol=5e-3)


def test_query_rotation_and_value_inverse_are_equivalent():
    torch.manual_seed(7)
    rk, rv = torch.linalg.qr(torch.randn(8, 8)).Q, torch.linalg.qr(torch.randn(8, 8)).Q
    q, k, v = torch.randn(4, 4, 8), torch.randn(17, 2, 8), torch.randn(17, 2, 8)
    slots = encode_kv(k @ rk, v @ rv)
    kd, vd = decode_kv(slots, 8)
    rotated = attention(q @ rk, kd, vd)
    actual = compressed_attention(q, slots, rk, rv)
    torch.testing.assert_close(rotated.output @ rv.T, actual.output, atol=2e-6, rtol=2e-6)
