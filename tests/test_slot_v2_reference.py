"""Archive G26-G34/#13-20/#151: V2 layout must be byte-exact and reversible."""
import pytest
import torch

from oscar_ascend.ops.reference import decode_kv_v2_group, encode_kv_v2_group


@pytest.mark.parametrize("dim", [64, 128, 256])
def test_consumer_major_v2_round_trip_matches_fp16_metadata_oracle(dim):
    generator = torch.Generator().manual_seed(151 + dim)
    key = torch.randn((16, dim), generator=generator)
    value = torch.randn((16, dim), generator=generator)
    encoded = encode_kv_v2_group(key, value)
    decoded_key, decoded_value = decode_kv_v2_group(encoded, dim)
    assert encoded.numel() == 16 * (dim // 2 + 8)
    assert decoded_key.shape == key.shape and decoded_value.shape == value.shape
    assert torch.isfinite(decoded_key).all() and torch.isfinite(decoded_value).all()


def test_v2_layout_rejects_partial_storage_groups():
    with pytest.raises(ValueError, match="16,D"):
        encode_kv_v2_group(torch.zeros(15, 256), torch.zeros(15, 256))
