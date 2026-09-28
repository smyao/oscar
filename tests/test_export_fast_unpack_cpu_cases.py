"""Fast-unpack CPU goldens retain fe0 bytes and explicit FP16 edge domains.

Archive #126/#129/#143/#148-151 and startup D.4: host oracle construction
cannot establish target NPU Cast/Gather precision or any performance gain.
"""

import importlib.util
import json
from pathlib import Path

import torch

from tools import export_fast_unpack_cpu_cases as fast


ROOT = Path(__file__).resolve().parents[1]


def _read(path, dtype, shape):
    return torch.frombuffer(bytearray(path.read_bytes()), dtype=dtype).reshape(shape)


def test_base_golden_matches_existing_independent_cv_case():
    spec = importlib.util.spec_from_file_location("cv_contracts_for_fast_oracle",
                                                  ROOT / "tests/test_cv_contracts.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    existing, old_output, old_lse = module._case(64, 4, 511)
    fixture = fast._single_case(torch, 64, 4, 511)
    for name in ("q", "qr", "ck", "cv", "rv", "raw", "table", "wk", "wv",
                 "tags", "starts", "lens", "slots"):
        assert torch.equal(fixture["tensors"][name].contiguous().view(torch.uint8),
                           existing[name].contiguous().view(torch.uint8)), name
    assert torch.equal(fixture["expected_output"], old_output)
    assert torch.equal(fixture["expected_lse"], old_lse)


def test_fast_cases_cover_live_subnormal_invalid_and_dead_padding(tmp_path):
    directory = fast.export_fast_unpack_cpu_cases(tmp_path / "goldens")
    cases = json.loads((directory / "cases.json").read_text())
    counts = {name: sum(row["op"] == name for row in cases)
              for name in {row["op"] for row in cases}}
    assert counts == {"fast_fe0": 4, "fast_fe0_error": 8,
                      "fast_fe0_dead": 1, "fast_q1": 3,
                      "fast_cluster4": 3, "fast_cluster4_poison": 1,
                      "fast_cluster4_error": 1}
    for row in cases:
        assert Path(row["path"]).is_dir()
        assert (Path(row["path"]) / "shape.txt").is_file()
    legal = directory / "d64_q4_c511_legal_fp16_edges"
    raw = (legal / "raw.bin").read_bytes()
    for position, value, zero, bits in (
            (100, False, False, 0x0001), (101, True, False, 0x03FF),
            (102, False, False, 0x0400), (103, False, True, 0x8000),
            (104, True, True, 0x0000), (105, False, True, 0x0001)):
        offset = fast._metadata_offset(64, position, value=value, zero=zero)
        assert raw[offset:offset + 2] == bits.to_bytes(2, "little")
    legal_output = _read(legal / "expected_output.bin", torch.float32, (4, 6, 64))
    baseline_output = _read(directory / "d64_q4_c511/expected_output.bin",
                            torch.float32, (4, 6, 64))
    assert bool(torch.isfinite(legal_output).all())
    assert not torch.equal(legal_output, baseline_output)

    dead = directory / "d64_q4_c511_dead_tail_nan"
    dead_raw = (dead / "raw.bin").read_bytes()
    dead_offset = fast._metadata_offset(64, 520, value=False, zero=False)
    assert dead_raw[dead_offset:dead_offset + 2] == bytes.fromhex("007e")
    dead_output = _read(dead / "expected_output.bin", torch.float32, (4, 6, 64))
    assert torch.equal(dead_output, baseline_output)

    expected_invalid = {
        "scale_pos_zero": (False, False, 0x0000),
        "scale_neg_zero": (False, False, 0x8000),
        "scale_pos_inf": (True, False, 0x7C00),
        "scale_neg_inf": (True, False, 0xFC00),
        "scale_nan": (False, False, 0x7E00),
        "zero_pos_inf": (False, True, 0x7C00),
        "zero_neg_inf": (True, True, 0xFC00),
        "zero_nan": (True, True, 0x7E00),
    }
    assert fast.LIVE_POSITION == 100
    for label, (value, zero, bits) in expected_invalid.items():
        case = directory / f"d64_q4_c511_{label}"
        raw = (case / "raw.bin").read_bytes()
        offset = fast._metadata_offset(64, fast.LIVE_POSITION, value=value, zero=zero)
        assert raw[offset:offset + 2] == bits.to_bytes(2, "little")
    assert fast.SINK <= fast.LIVE_POSITION < 511 + 1 - fast.RECENT
    assert fast.DEAD_POSITION > 511 + 4


def test_q1_and_cluster_cases_reuse_one_directory_per_input(tmp_path):
    directory = fast.export_fast_unpack_cpu_cases(tmp_path / "goldens")
    cases = json.loads((directory / "cases.json").read_text())
    q1 = [row for row in cases if row["op"] == "fast_q1"]
    assert {Path(row["path"]).name for row in q1} == {
        "q1_32_s3_d64", "q1_32_s1_d64", "q1_32_empty_s3_d64"}
    mature = [row for row in cases if Path(row["path"]).name == "d64_q168_c641"]
    assert [row["op"] for row in mature] == ["fast_cluster4", "fast_cluster4_poison"]
    assert mature[0]["path"] == mature[1]["path"]
    assert tuple(map(int, (Path(mature[0]["path"]) / "shape.txt").read_text().split())) == (
        168, 6, 1, 64, 641, 4, 32, 3, 512, 2, 64, 20480, 2, 1, 1)
    legal = directory / "d64_q168_c641_legal_fp16_edges"
    invalid = directory / "d64_q168_c641_invalid_live_scale"
    legal_raw, invalid_raw = (legal / "raw.bin").read_bytes(), (invalid / "raw.bin").read_bytes()
    assert legal_raw[fast._metadata_offset(64, 400, value=False, zero=False):
                     fast._metadata_offset(64, 400, value=False, zero=False) + 2] == bytes.fromhex("0100")
    assert legal_raw[fast._metadata_offset(64, 403, value=False, zero=True):
                     fast._metadata_offset(64, 403, value=False, zero=True) + 2] == bytes.fromhex("0080")
    assert invalid_raw[fast._metadata_offset(64, 400, value=False, zero=False):
                       fast._metadata_offset(64, 400, value=False, zero=False) + 2] == bytes.fromhex("007e")
    assert invalid_raw[fast._metadata_offset(64, 401, value=True, zero=True):
                       fast._metadata_offset(64, 401, value=True, zero=True) + 2] == bytes.fromhex("007c")
    legal_output = _read(legal / "expected_output.bin", torch.float32, (168, 6, 64))
    assert bool(torch.isfinite(legal_output).all())
