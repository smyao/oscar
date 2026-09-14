import pytest

from oscar_ascend.kernels import ascendc_attention


def test_ascendc_is_opt_in(monkeypatch):
    monkeypatch.delenv("OSCAR_ASCEND_USE_ASCENDC", raising=False)
    monkeypatch.setattr(ascendc_attention, "_resolve_op", lambda: object())
    assert ascendc_attention.ascendc_mode() == "0"
    assert not ascendc_attention.ascendc_enabled()


def test_required_ascendc_fails_closed(monkeypatch):
    monkeypatch.setenv("OSCAR_ASCEND_USE_ASCENDC", "required")
    monkeypatch.setattr(ascendc_attention, "_resolve_op", lambda: None)
    with pytest.raises(RuntimeError, match="required"):
        ascendc_attention.ascendc_enabled()


def test_invalid_ascendc_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("OSCAR_ASCEND_USE_ASCENDC", "maybe")
    with pytest.raises(ValueError, match="OSCAR_ASCEND_USE_ASCENDC"):
        ascendc_attention.ascendc_mode()


def test_required_mode_helper(monkeypatch):
    monkeypatch.setenv("OSCAR_ASCEND_USE_ASCENDC", "required")
    assert ascendc_attention.ascendc_required()
    monkeypatch.setenv("OSCAR_ASCEND_USE_ASCENDC", "1")
    assert not ascendc_attention.ascendc_required()


def test_reference_kernel_has_conservative_length_gate(monkeypatch):
    monkeypatch.delenv("OSCAR_ASCEND_ASCENDC_MAX_SEQ_LEN", raising=False)
    assert ascendc_attention.ascendc_max_seq_len() == 256
    monkeypatch.setenv("OSCAR_ASCEND_ASCENDC_MAX_SEQ_LEN", "0")
    with pytest.raises(ValueError, match="must be positive"):
        ascendc_attention.ascendc_max_seq_len()


def test_source_contract_is_packed_int8_only():
    from pathlib import Path

    root = Path(__file__).parents[1]
    definition = (root / "ascendc/oscar_int2_paged_attention/op_host/"
                  "oscar_int2_paged_attention_def.cpp").read_text()
    tiling = (root / "ascendc/oscar_int2_paged_attention/op_host/"
              "oscar_int2_paged_attention_tiling.cpp").read_text()
    assert definition.count(".DataType({ge::DT_INT8})") >= 2
    assert "kHeadDim = 256" in tiling
    assert "hq % hk" in tiling
    assert "GetLibApiWorkSpaceSize" in tiling
    assert "ge::DT_BF16" not in definition
    assert "DataType(fp)" not in definition
    assert "Format(nd)" not in definition
    assert "SetTilingKey(0)" in tiling


def test_kvcache_loader_matches_oscar_split_layout_contract():
    from pathlib import Path

    root = Path(__file__).parents[1]
    source = (root / "ascendc/oscar_int2_paged_attention/op_kernel/"
              "oscar_int2_paged_attention_kvcache.h").read_text()
    common = (root / "ascendc/oscar_int2_paged_attention/op_kernel/"
              "oscar_int2_paged_attention_common.h").read_text()
    assert "K_PACKED_OFFSET = 32" in common
    assert "V_PACKED_OFFSET = 0" in common
    assert "V_SCALE_OFFSET = 4" in common
    assert "V_ZERO_OFFSET = 6" in common
    assert "block % shape_.stageRows" in source
    assert "owner_.GetValue" in source
    assert "(kb >> 6) & 3U" in source
    assert "(vb >> 6) & 3U" in source
    assert "../op_host" not in common


def test_kernel_uses_aicore_math_and_qualified_tensor_types():
    from pathlib import Path

    root = Path(__file__).parents[1]
    source = (root / "ascendc/oscar_int2_paged_attention/op_kernel/"
              "oscar_int2_paged_attention.cpp").read_text()
    cache = (root / "ascendc/oscar_int2_paged_attention/op_kernel/"
             "oscar_int2_paged_attention_kvcache.h").read_text()
    assert "expf(" not in source
    assert "AscendC::Exp(" in source
    assert "bfloat16_t" not in source
    assert " LocalTensor<" not in cache
    assert "\n  GlobalTensor<" not in cache


def test_standalone_torch_binding_contract_is_present():
    from pathlib import Path

    root = Path(__file__).parents[1]
    binding = (root / "ascendc/torch_binding/oscar_ascend_torch.cpp").read_text()
    script = (root / "delivery/build_ascendc_torch_binding.sh").read_text()
    assert "TORCH_LIBRARY(oscar_ascend" in binding
    assert "aclnnOscarInt2PagedAttention" in binding
    assert "liboscar_ascend_torch.so" in script
