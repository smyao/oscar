"""Archive #145/#164: independent fixture coverage and paired CPU harness.

These tests qualify the test inputs/harness, not execution on an Ascend device.
The original C16 and frozen v3 oracle remain separate from tuned generation.
"""
import json
from pathlib import Path

import pytest
import torch

from reports.c16_targeted_opt_20261020 import cpu_fixture, make_cpu_probe


def test_cpu_pair_compares_all_outputs_after_distinct_poison():
    source = make_cpu_probe.generate_probe()
    old = source.index("    CV_CALL(oscar_attention_cv_aligned_cluster16_control_kernel);")
    poison = source.index("std::memset(gm->ptr,0xC6,gm->size)", old)
    new = source.index("    CV_CALL(oscar_attention_cv_aligned_cluster16_kernel);", poison)
    check = source.index("tuned output differs bitwise", new)
    assert old < poison < new < check
    assert "{&partial,&lse,&status,&stats}" in source
    assert "if(dequantMode!=0)" in source
    assert 'bf16_expected_partial.bin' in source
    assert 'bf16_expected_lse.bin' in source
    assert 'bf16_expected_output.bin' in source
    assert "guards();immutable();" in source


def test_probe_refuses_stale_source_anchors_and_invalid_symbol():
    with pytest.raises(ValueError, match="anchor changed"):
        make_cpu_probe.replace_once("same same", "same", "replacement")
    with pytest.raises(ValueError, match="C identifier"):
        make_cpu_probe.generate_probe(symbol="kernel(); another()")
    with pytest.raises(ValueError, match="preserved control"):
        make_cpu_probe.generate_probe(symbol=make_cpu_probe.CONTROL_SYMBOL)


def test_cpu_single_arm_source_and_symbol_override_preserve_control(tmp_path):
    symbol = "oscar_attention_cv_aligned_cluster16_tuned_kernel"
    source = make_cpu_probe.generate_probe(symbol=symbol)
    assert source.count(f'extern "C" void {symbol}(CV_ARGS);') == 1
    assert source.count(f'extern "C" void {make_cpu_probe.CONTROL_SYMBOL}(CV_ARGS);') == 1
    assert f"CV_CALL({make_cpu_probe.CONTROL_SYMBOL})" in source
    assert f"CV_CALL({symbol})" in source
    cmake = make_cpu_probe.generate_cmake(tmp_path / "single_arm.cpp")
    assert "attention_cv_aligned_cluster16_control.cpp" in cmake
    assert str(tmp_path / "single_arm.cpp") in cmake


def test_matrix_reaches_live_history_fast_path_and_unaligned_tail():
    cases = {case.name: case for case in cpu_fixture.CASES}
    # Mature C16 needs sixteen 21-query groups. A history of 96 never reaches
    # a full 256-column tile; 577 requires full tiles plus one live tail.
    mature = cases["mature_multiple_kv_tiles"]
    assert mature.query >= 16 * 21 and mature.context - 64 > 2 * 256
    assert (mature.context - 64) % 256 != 0
    assert cases["cold_second_score_block"].query * 6 > 32
    assert cases["hkv2_s3_window_current"].heads == 2
    assert cases["hkv2_s3_window_current"].splits == 3
    assert all(case.padding > 0 for case in cases.values())
    assert {case.qr_value for case in cases.values() if case.bad_qr >= 0} == {"nan", "inf"}


def test_exported_fixture_padding_faults_and_reference_inputs(tmp_path):
    root = tmp_path / "inputs"
    cpu_fixture.make(root, ["metadata_lane1", "qr_last_member_inf", "unread_dead_metadata"])
    summary = json.loads((root / "cases.json").read_text())
    assert len(summary["cases"]) == 3
    for record in summary["cases"]:
        folder = root / record["name"]
        fixture = torch.load(folder / "fixture.pt", weights_only=False)
        data = fixture["cpu"]
        live = record["query"]
        assert bool(torch.isnan(data["q"][live:]).all())
        assert bool((data["slots"][live:] == -1).all())
        assert bool((data["expected_positions"][live:] == -1).all())
        assert data["qr"].dtype == torch.bfloat16
        assert fixture["cache_format"] == "striped_fp32_v2"
        assert record["dequant_mode"] == 0 and record["performance"] == "not_run"
        assert not list(folder.glob("bf16_expected_*.bin"))
    meta = torch.load(root / "metadata_lane1/fixture.pt", weights_only=False)
    checks = meta["cpu"]["required_status_checks"]
    assert checks.shape == (17, 3) and checks[0].tolist() == [0, 1, 3]
    qr = torch.load(root / "qr_last_member_inf/fixture.pt", weights_only=False)
    assert torch.isposinf(qr["cpu"]["qr"][315, 0, 0])
    assert qr["cpu"]["required_status_checks"].tolist() == [[945, 0, 2]]
    dead = json.loads((root / "unread_dead_metadata/case.json").read_text())
    assert dead["fault"] is False


def test_golden_rejects_missing_actual_prepare_table(tmp_path):
    folder = tmp_path / "input"
    cpu_fixture.make(folder, ["empty_history_window"])
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(FileNotFoundError, match="tasks.bin"):
        cpu_fixture.golden(folder / "empty_history_window", output)
