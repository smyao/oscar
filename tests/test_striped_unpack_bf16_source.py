"""Archive #13-22/#126/#145/#148/#151/#154/#155/#157: direct BF16 decode.

CPU address/guard models and source contracts only. The separate direct-decode
experiment checks arithmetic against an independent RNE oracle; neither proves
CANN API acceptance or asynchronous NPU execution on the unavailable device.
"""
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "csrc/kernels/attention_striped_unpack_bf16.h"


def test_raw_half_guard_and_error_override_are_identical_to_current_h32_helper():
    source = HELPER.read_text()
    mother = (ROOT / "csrc/kernels/attention_striped_unpack32.h").read_text()
    start = "  auto gt0=wordBuf.Get<uint8_t>();"
    end = "  if((~(validScale&validZero)&liveMask)!=0)error=3;"
    def guard(text):
        return text[text.index(start):text.index(end) + len(end)]
    assert guard(source) == guard(mother)
    # All 65536 raw half encodings, including +/-0, subnormals and NaN
    # payloads, against a numerical definition independent of packed masks.
    bits = np.arange(65536, dtype=np.uint16)
    raw = bits.view(np.int16).astype(np.int32)
    with np.errstate(invalid="ignore"):
        values = bits.view(np.float16).astype(np.float32)
    np.testing.assert_array_equal((raw > 0) & (raw < 31744),
                                  np.isfinite(values) & (values > 0))
    np.testing.assert_array_equal((raw < -1024) | ((raw >= 0) & (raw < 31744)),
                                  np.isfinite(values))


@pytest.mark.parametrize("rows", [16, 32, 64])
def test_vector_repeat_mapping_covers_natural_dimensions_without_touching_padding(rows):
    # Decode the actual block/repeat address equations; each 128-lane
    # instruction handles eight rows x16 words, then the second word block.
    seen = {}
    for bit in range(8):
        for word_block in range(2):
            for repeat in range(rows // 8):
                for block in range(8):
                    for lane in range(16):
                        src = word_block * 16 + repeat * 8 * 80 + block * 80 + lane
                        dst = bit * 32 + word_block * 16 + repeat * 8 * 272 + block * 272 + lane
                        row, dim = divmod(dst, 272)
                        assert dim < 256 and dst not in seen
                        assert src == row * 80 + dim % 32
                        seen[dst] = (src, 2 * bit)
    assert sorted(seen) == [row * 272 + dim for row in range(rows) for dim in range(256)]


@pytest.mark.parametrize("rows", [16, 32, 64])
def test_code_major_table_byte_gather_preserves_bf16_bits_and_row_domain(rows):
    rng = np.random.default_rng(rows)
    codes = rng.integers(0, 4, size=(rows, 256), dtype=np.int16)
    # Arbitrary 16-bit words test copying, including NaN payloads and signed
    # zero; interpreting the table as half must perform no numeric conversion.
    table = rng.integers(0, 65536, size=(4, rows), dtype=np.uint16)
    table[0, :4] = [0x0000, 0x8000, 0x7FC1, 0xFFFF]
    codes[:4, 0] = 0
    result = np.empty_like(codes, dtype=np.uint16)
    for begin in range(0, rows, 8):
        code_float = codes[begin:begin + 8].astype(np.float32)
        offsets = code_float.astype(np.int32) * (2 * rows)
        offsets += 2 * np.arange(begin, begin + 8, dtype=np.int32)[:, None]
        assert int(offsets.min()) >= 0 and int(offsets.max()) < rows * 4 * 2
        assert np.all(offsets % 2 == 0)
        result[begin:begin + 8] = table.reshape(-1)[offsets // 2]
    expected = table[codes, np.arange(rows)[:, None]]
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(result[:4, 0], table[0, :4])


@pytest.mark.parametrize("rows", [16, 32, 64])
def test_live_mask_boundary_preserves_incoming_error_until_a_live_invalid_row(rows):
    for live in (0, 1, rows - 1, rows):
        live_mask = (1 << live) - 1
        for invalid_row in (0, rows - 1):
            valid = ((1 << rows) - 1) ^ (1 << invalid_row)
            for incoming in (0, 1, 2, 4, 5):
                actual = 3 if (~valid & live_mask) else incoming
                assert actual == (3 if invalid_row < live else incoming)


@pytest.mark.parametrize("rows", [16, 32, 64])
def test_small_table_and_eight_row_index_scratch_have_disjoint_live_regions(rows):
    scratch_bytes = 32 * rows
    metadata = (0, 8 * rows)
    raw_compare = (8 * rows, (2 * rows + (64 if rows <= 32 else 128)) * 4)
    fp32_table = (16 * rows, 32 * rows)
    bf16_table = (0, 8 * rows)
    assert metadata[1] <= raw_compare[0]
    assert raw_compare[1] <= scratch_bytes
    # raw_compare is dead before table starts; metadata survives until all
    # four table planes are complete and only then becomes the BF16 table.
    assert metadata[1] <= fp32_table[0]
    assert bf16_table[1] <= fp32_table[0]
    assert fp32_table[1] == scratch_bytes
    float_codes = (0, 8 * 256 * 4)
    integer_offsets = (float_codes[1], 2 * 8 * 256 * 4)
    assert float_codes[1] == integer_offsets[0]
    assert integer_offsets[1] == 16384


def test_cpp_does_not_expand_fp32_kv_or_reinterpret_cast_bf16_as_half_arithmetic():
    source = HELPER.read_text()
    assert "dequantBuf.Get<float>()" not in source
    assert "auto values=dequantBuf.Get<bfloat16_t>();" in source
    assert "auto table=metadata[4*HalfKv];" in source
    mul = source.index("Mul(table[code*HalfKv]")
    add = source.index("Add(table[code*HalfKv]")
    cast = source.index("Cast(tableBf16,table,RoundMode::CAST_RINT,4*HalfKv);")
    gather = source.index("Gather(values.ReinterpretCast<half>()")
    assert mul < add < cast < gather
    assert "PipeBarrier<PIPE_V>();" in source[mul:add]
    assert "PipeBarrier<PIPE_V>();" in source[add:cast]
    assert "PipeBarrier<PIPE_V>();" in source[gather:]
    assert source.count("values.ReinterpretCast<half>()") == 1
    assert source.count("tableBf16.ReinterpretCast<half>()") == 1
    assert "Gather(halfMetadata,packedBuf.Get<half>()" in source
    assert "static_assert(indexScratchBytes==16384 && scratchBytes<=2048);" in source
    guard_end = source.index("if((~(validScale&validZero)&liveMask)!=0)error=3;")
    metadata_cast = source.index("Cast(metadata,halfMetadata,RoundMode::CAST_NONE,2*HalfKv);")
    assert "Fence<HardEvent::S_V>();" in source[guard_end:metadata_cast]
