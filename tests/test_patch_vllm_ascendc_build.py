from pathlib import Path

import pytest

from tools.patch_vllm_ascendc_build import patch_build_script


def test_patch_adds_operator_to_both_soc_arrays(tmp_path: Path):
    path = tmp_path / "build_aclnn.sh"
    marker = '        "kv_quant_sparse_flash_attention"\n'
    path.write_text(marker + "middle\n" + marker)
    patch_build_script(path, "oscar_int2_paged_attention")
    text = path.read_text()
    assert text.count('        "oscar_int2_paged_attention"') == 2


def test_patch_is_idempotent(tmp_path: Path):
    path = tmp_path / "build_aclnn.sh"
    marker = '        "kv_quant_sparse_flash_attention"\n'
    path.write_text(marker + marker)
    patch_build_script(path, "oscar_int2_paged_attention")
    first = path.read_text()
    patch_build_script(path, "oscar_int2_paged_attention")
    assert path.read_text() == first


def test_patch_rejects_unknown_layout(tmp_path: Path):
    path = tmp_path / "build_aclnn.sh"
    path.write_text("CUSTOM_OPS_ARRAY=()\n")
    with pytest.raises(RuntimeError, match="refusing"):
        patch_build_script(path, "oscar_int2_paged_attention")
