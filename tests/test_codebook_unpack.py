"""Actual old/new helper execution plus independent v4 and bit-domain checks.

Archive #126/#145/#154/#155/#164/#172 and complete startup D.4 reviewed.
This bounded host Vector/Gather model is not AscendC CPU-debug, CANN
compilation, device instruction timing, NPU correctness, or throughput.
Its NaN Cast rule is explicit; matching modeled payloads is not an assertion
about an unmeasured hardware NaN canonicalization rule.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess

import pytest
import torch

from tools import official_reference


ROOT = Path(__file__).resolve().parents[1]
H, D, STRIDE = 64, 256, 160


def _without_includes(text):
    return re.sub(r"^#(?:include[^\n]*|pragma once)\n", "", text, flags=re.MULTILINE)


def build_control(build, replacement=None):
    compiler = shutil.which("clang++")
    if compiler is None:
        pytest.fail("clang++ required for actual helper host semantic checks")
    harness = (ROOT / "tests/cpp/codebook_unpack_control.cpp").read_text()
    for marker, path in (
        ("@@CAST_HEADER@@", "csrc/kernels/attention_bf16_cast.h"),
        ("@@OLD_HEADER@@", "csrc/kernels/attention_aligned_unpack.h"),
        ("@@NEW_HEADER@@", "csrc/kernels/attention_codebook_unpack.h"),
    ):
        contents = (ROOT / path).read_text()
        if marker == "@@NEW_HEADER@@" and replacement:
            before, after = replacement
            assert contents.count(before) == 1
            contents = contents.replace(before, after, 1)
        harness = harness.replace(marker, _without_includes(contents))
    source = build / "actual_helpers.cpp"
    source.write_text(harness)
    binary = build / "codebook_unpack"
    env = os.environ.copy()
    env["DEVELOPER_DIR"] = "/Applications/Xcode.app/Contents/Developer"
    result = subprocess.run(
        [compiler, "-std=c++17", "-O0", "-ffp-contract=off", "-Wall", "-Wextra",
         "-Werror", "-Wno-unused-parameter", str(source), "-o", str(binary)],
        text=True, capture_output=True, timeout=60, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    return binary


@pytest.fixture(scope="module")
def control_binary(tmp_path_factory):
    return build_control(tmp_path_factory.mktemp("codebook-unpack"))


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def record(seed=0, *, scales=None, zeros=None, constant_code=None):
    """Write physical striped bytes independently, including poison padding."""
    result = bytearray([0xD7] * (H * STRIDE))
    for row in range(H):
        for word in range(32):
            k_word = (seed + row * 32 + word) & 0xFFFF
            v_word = 65535 - k_word
            if constant_code is not None:
                k_word = v_word = sum(constant_code << (2 * p) for p in range(8))
            struct.pack_into("<H", result, row * STRIDE + 2 * word, k_word)
            struct.pack_into("<H", result, row * STRIDE + 64 + 2 * word, v_word)
        for side in range(2):
            scale = scales[row] if scales is not None else (1 + row % 7) / 128
            zero = zeros[row] if zeros is not None else (row % 11 - 5) / 4
            struct.pack_into("<ff", result, row * STRIDE + 128 + 8 * side, scale, zero)
    return result


def set_metadata(data, row, side, *, scale=None, zero=None):
    if scale is not None:
        struct.pack_into("<I", data, row * STRIDE + 128 + side * 8, scale)
    if zero is not None:
        struct.pack_into("<I", data, row * STRIDE + 132 + side * 8, zero)


def run_helpers(binary, tmp_path, records, *, mode=0, live=H, initial=0):
    source, output = tmp_path / "packed.bin", tmp_path / "bf16.bin"
    source.write_bytes(b"".join(records))
    result = subprocess.run(
        [str(binary), str(mode), str(live), str(initial), str(source), str(output)],
        text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    info = json.loads(result.stdout)
    bits = torch.frombuffer(bytearray(output.read_bytes()), dtype=torch.int16).reshape(len(records), 2, H, D)
    assert info["records"] == len(records)
    assert info["host_model_only"] is True
    # This is the selected bounded call size, not a claimed SDK maximum.
    assert info["gather_largest"] == 2048
    assert info["raw_gather_calls"] == 16 * len(records)
    assert 0 <= info["raw_gather_index_max"] <= 510
    return info, bits


@pytest.mark.parametrize("mode", [0, 1])
def test_actual_helpers_match_independent_v4_oracle_and_fixed_address_replay(control_binary, tmp_path, mode):
    cases = [record(0), record(23421), record(0)]
    info, actual = run_helpers(control_binary, tmp_path, cases, mode=mode)
    assert info["statuses"] == [0] * 6
    for index, data in enumerate(cases):
        physical = torch.frombuffer(data, dtype=torch.uint8).reshape(H, STRIDE)
        k, v = official_reference.unpack_kv(physical[:, :144].contiguous(), D, mode)
        expected = torch.stack((k, v)).bfloat16().view(torch.int16)
        assert torch.equal(actual[index], expected)
    assert torch.equal(actual[0], actual[2])
    assert not torch.equal(actual[0], actual[1])


@pytest.mark.parametrize("mode", [0, 1])
def test_all_65536_packed_words_both_sides_and_all_eight_planes(control_binary, tmp_path, mode):
    cases = [record(index * 2048) for index in range(32)]
    info, actual = run_helpers(control_binary, tmp_path, cases, mode=mode)
    assert info["statuses"] == [0] * 64
    # The C++ oracle independently decodes each raw word/plane. Also compare
    # this exhaustive corpus to the pinned torch v4 test-only implementation.
    for index, data in enumerate(cases):
        physical = torch.frombuffer(data, dtype=torch.uint8).reshape(H, STRIDE)
        expected = official_reference.unpack_kv(physical[:, :144].contiguous(), D, mode)
        assert torch.equal(actual[index], torch.stack(expected).bfloat16().view(torch.int16))


@pytest.mark.parametrize("live", [0, 1, 7, 8, 16, 31, 32, 63, 64])
@pytest.mark.parametrize("mode", [0, 1, -1, 2])
def test_tail_metadata_poison_and_error_overwrite_order(control_binary, tmp_path, live, mode):
    data = record(37419)
    for row in range(live, H):
        set_metadata(data, row, 0, scale=0x7FC12345, zero=0xFF800000)
        set_metadata(data, row, 1, scale=0, zero=0xFFC23456)
    info, _ = run_helpers(control_binary, tmp_path, [data], mode=mode, live=live, initial=5)
    assert info["statuses"] == ([5, 5] if mode in (0, 1) else [1, 1])


@pytest.mark.parametrize("field,bits", [
    ("scale", 0), ("scale", 0x80000000), ("scale", 0xBF800000),
    ("scale", 0x7F800000), ("scale", 0xFF800000),
    ("scale", 0x7FC12345), ("scale", 0x7F812345),
    ("scale", 0x00000001), ("scale", 0x007FFFFF),
    ("scale", 0x00800000), ("scale", 0x7F7FFFFF),
    ("zero", 0x7FC23456), ("zero", 0xFFC23456),
    ("zero", 0x7F800000), ("zero", 0xFF800000),
    ("zero", 0), ("zero", 0x80000000),
    ("zero", 0x00000001), ("zero", 0x80000001),
])
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_fp32_metadata_special_values_and_rounding_error_domains(control_binary, tmp_path, field, bits, mode):
    data = record(17379)
    set_metadata(data, 0, 0, **{field: bits})
    # Test the last live V row too, after K may already have changed error.
    set_metadata(data, 63, 1, **{field: bits})
    info, _ = run_helpers(control_binary, tmp_path, [data], mode=mode, initial=4)
    if mode == 2:
        assert info["statuses"] == [1, 1]
    else:
        value = struct.unpack("<f", struct.pack("<I", bits))[0]
        rounded = torch.tensor(value, dtype=torch.float32).bfloat16().float().item()
        invalid = not bool(torch.isfinite(torch.tensor(value))) or (field == "scale" and value <= 0)
        if mode == 1:
            invalid |= not bool(torch.isfinite(torch.tensor(rounded))) or (field == "scale" and rounded <= 0)
        assert info["statuses"] == ([3, 3] if invalid else [4, 4])


@pytest.mark.parametrize("mode", [0, 1])
def test_unused_overflowing_levels_do_not_add_errors(control_binary, tmp_path, mode):
    data = record(constant_code=0)
    for row in range(H):
        # The used level is exact zero. Level3 exceeds FP32 range, but is
        # unused. This must not introduce a reconstruction-wide finite guard.
        for side in range(2):
            set_metadata(data, row, side, scale=0x7F000000, zero=0)
    info, actual = run_helpers(control_binary, tmp_path, [data], mode=mode)
    assert info["statuses"] == [0, 0]
    assert torch.count_nonzero(actual).item() == 0


@pytest.mark.parametrize("mode", [0, 1])
def test_bf16_halfway_ties_signed_zero_and_exact_subnormal_bits(control_binary, tmp_path, mode):
    scales = [struct.unpack("<f", struct.pack("<I", bits))[0] for bits in
              (0x3F808000, 0x3F818000, 0x00010000, 0x00808000) * 16]
    zeros = [-0.0, 0.0, 1.0, 3.0] * 16
    data = record(42231, scales=scales, zeros=zeros)
    _, actual = run_helpers(control_binary, tmp_path, [data], mode=mode)
    physical = torch.frombuffer(data, dtype=torch.uint8).reshape(H, STRIDE)
    expected = official_reference.unpack_kv(physical[:, :144].contiguous(), D, mode)
    assert torch.equal(actual[0], torch.stack(expected).bfloat16().view(torch.int16))


@pytest.mark.parametrize("before,after,message", [
    (
        "static_cast<float>(2*HalfKv),",
        "static_cast<float>(2*HalfKv+2),",
        "UB access outside allocation",
    ),
    (
        "auto output=naturalBuf.Get<uint16_t>();",
        "auto output=dequantBuf.Get<uint16_t>();",
        "Gather destination aliases offsets",
    ),
    (
        "auto output=naturalBuf.Get<uint16_t>();",
        "auto output=scratchBuf.Get<uint16_t>();",
        "Gather destination aliases source",
    ),
    (
        "ArithProgression<int32_t>(rowIntegers,0,2,HalfKv);",
        "ArithProgression<int32_t>(rowIntegers,0,1,HalfKv);",
        "Gather byte offset misaligned",
    ),
    (
        "value?8:0,metadataCount);\n  Fence<HardEvent::V_S>();",
        "value?8:0,metadataCount);\n  PipeBarrier<PIPE_V>();",
        "scalar read before V_S handoff",
    ),
])
def test_negative_controls_detect_index_alias_and_scalar_handoff_errors(tmp_path, before, after, message):
    binary = build_control(tmp_path, (before, after))
    source, output = tmp_path / "packed.bin", tmp_path / "bf16.bin"
    source.write_bytes(record(23141))
    result = subprocess.run([str(binary), "0", "64", "0", str(source), str(output)],
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 1
    assert message in result.stderr
