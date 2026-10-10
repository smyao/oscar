"""Actual production selector, ABI/capability and bounded-arena contracts."""
from pathlib import Path
import pytest

from oscar_ascend.ops.cv_dispatch import select_cv_op,DECODE_BUNDLE_OPS,CLUSTER_CV_OPS
from oscar_ascend.ops.contracts import SOURCE_CAPABILITIES
from oscar_ascend.runtime import WorkspaceGeometry


@pytest.mark.parametrize('tokens,query,q1,expected',[
    (128,4,False,'attention_cv_bundle_decode_out'),
    (128,1,True,'attention_cv_bundle_q1_out'),
    (16384,1,True,'attention_cv_bundle_q1_out'),
    (16384,8192,False,'attention_cv_window_range_cluster16_out'),
    (388,385,False,'attention_cv_window_range_cluster4_out'),
    (36,33,False,'attention_cv_window_range_out'),
    (512,33,False,'attention_cv_window_range_balanced_out'),
])
def test_actual_selector_covers_decode_and_mixed_routes(tokens,query,q1,expected):
    assert select_cv_op(16,6,1,tokens,query,q1_draft=q1,fast_unpack=True,
        mixed_cv=True,striped_cache=True,decode_bundle=True)==expected
    if 'cluster' in expected:assert expected in CLUSTER_CV_OPS


def test_bundle_cannot_read_a_canonical_cache():
    with pytest.raises(ValueError,match='striped'):
        select_cv_op(16,6,1,128,4,fast_unpack=True,mixed_cv=True,decode_bundle=True)


def test_bundle_exact_window_dma_is_validated_per_physical_row():
    """Archive #156: prefix B=2304 must not reuse an unproved batched address run."""
    source = Path('csrc/kernels/attention_cv_decode_bundle.cpp').read_text()
    begin = source.index("__aicore__ void LoadWindowRuns")
    end = source.index("__aicore__ void LoadPrecise", begin)
    body = source[begin:end]
    assert "for(int32_t j=0;j<live;++j)" in body
    assert "if(!Physical(position,block,inpage))continue" in body
    assert "if(tag.GetValue(0)!=inpage)" in body
    assert "DataCopyExtParams copy{1,static_cast<uint32_t>(D*2)" in body
    assert "rows*8" not in body


def test_capability_schema_and_registration_are_complete():
    assert DECODE_BUNDLE_OPS<=SOURCE_CAPABILITIES
    binding=Path('csrc/striped_attention_bindings.cpp').read_text()
    extension=Path('csrc/torch_bindings.cpp').read_text()
    cmake=Path('csrc/CMakeLists.txt').read_text()
    meta=Path('oscar_ascend/ops/meta.py').read_text()
    for name in DECODE_BUNDLE_OPS:
        assert f'm.def("{name}(' in binding
        assert f'm.impl("{name}"' in binding
        assert name in extension and name in cmake and name in meta


def test_existing_c16_arena_covers_the_two_tile_decode_workspace():
    geometry=WorkspaceGeometry(16384,6,1,256,splits=1,cube_cores=20,history_cluster_size=16)
    assert geometry.cv_bytes>=20*1146880
