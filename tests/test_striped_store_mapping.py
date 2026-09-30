"""Archive G27/#13-22/#37-49/#126/#154, D.4: exhaustive INT2 slot permutation."""

import pytest


def _canonical_pack(codes: list[int]) -> bytes:
    return bytes(sum(codes[4 * i + lane] << (2 * lane) for lane in range(4))
                 for i in range(len(codes) // 4))


def _striped_pack(codes: list[int]) -> bytes:
    width = len(codes) // 8
    out = bytearray()
    for i in range(width):
        word = sum(codes[b * width + i] << (2 * b) for b in range(8))
        out.extend(word.to_bytes(2, "little"))
    return bytes(out)


def _inverse_striped(data: bytes, dim: int) -> bytes:
    width = dim // 8
    codes = [0] * dim
    for i in range(width):
        word = int.from_bytes(data[2 * i:2 * i + 2], "little")
        for b in range(8):
            codes[b * width + i] = (word >> (2 * b)) & 3
    return _canonical_pack(codes)


@pytest.mark.parametrize("dim", [64, 128, 256])
def test_all_65536_code_words_roundtrip_to_canonical(dim: int):
    width = dim // 8
    for word in range(1 << 16):
        codes = [0] * dim
        for b in range(8):
            codes[b * width + width - 1] = (word >> (2 * b)) & 3
        striped = _striped_pack(codes)
        assert striped[-2:] == word.to_bytes(2, "little")
        assert _inverse_striped(striped, dim) == _canonical_pack(codes)


@pytest.mark.parametrize("dim", [64, 128, 256])
def test_slot_layout_keeps_all_metadata_and_byte_count(dim: int):
    k = [(i * 3 + (i // 7)) & 3 for i in range(dim)]
    v = [(i * 5 + (i // 11)) & 3 for i in range(dim)]
    k_meta = bytes([0x01, 0x3C, 0x00, 0xBC])
    v_meta = bytes([0xFF, 0x7B, 0x01, 0x80])
    old = _canonical_pack(k) + k_meta + _canonical_pack(v) + v_meta
    new = _striped_pack(k) + _striped_pack(v) + k_meta + v_meta
    packed = dim // 4
    assert len(old) == len(new) == dim // 2 + 8
    assert new[dim // 2:dim // 2 + 8] == k_meta + v_meta
    restored = (_inverse_striped(new[:packed], dim) + new[dim // 2:dim // 2 + 4]
                + _inverse_striped(new[packed:2 * packed], dim)
                + new[dim // 2 + 4:dim // 2 + 8])
    assert restored == old
