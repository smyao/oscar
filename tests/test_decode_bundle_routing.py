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
    """Archive #156: every candidate route validates B=2304 rows before DMA."""
    files = ["attention_cv_decode_bundle.cpp", "attention_cv_window_range.cpp",
             "attention_cv_window_range_balanced.cpp",
             "attention_cv_window_range_cluster4.cpp",
             "attention_cv_window_range_cluster16.cpp"]
    for name in files:
        source = (Path('csrc/kernels') / name).read_text()
        begin = source.index("__aicore__ void LoadWindowRuns")
        end = source.index("__aicore__ void LoadPrecise", begin)
        body = source[begin:end]
        assert "for(int32_t j=0;j<live;++j)" in body, name
        assert "if(!Physical(position,block,inpage))continue" in body, name
        assert "if(tag.GetValue(0)!=inpage)" in body, name
        assert "DataCopyExtParams copy{1,static_cast<uint32_t>(D*2)" in body, name
        assert "rows*8" not in body, name

    tree = "\n".join(path.read_text() for path in Path('csrc/kernels').glob('attention_cv*.cpp'))
    assert "static_cast<uint32_t>(rows*8)" not in tree


def test_store_slots_are_owned_by_the_native_request_table_before_publication():
    """Archive #156: native COW/table ownership is enforced on device."""
    prepare = Path('csrc/kernels/attention_tasks.cpp').read_text()
    binding = Path('csrc/attention_bindings.cpp').read_text()
    forward = Path('oscar_ascend/integration/impl.py').read_text()
    assert "tableColumns>0" in prepare
    assert "tableLocal.GetValue(0))!=physicalSlot/128" in prepare
    assert "row.SetValue(0,padding || invalid ? -1 : logicalPosition)" in prepare
    assert "if(blockTable.has_value())" in binding
    assert "blockTable?blockTable->data_ptr():nullptr" in binding
    assert "attn_metadata.block_tables[:attn_metadata.num_reqs]," in forward


def test_prefix_layout_and_feature_selection_cannot_be_silently_rewritten():
    runtime = Path('oscar_ascend/runtime.py').read_text()
    backend = Path('oscar_ascend/integration/backend.py').read_text()
    scheduler = Path('oscar_ascend/integration/whole_prefill_scheduler.py').read_text()
    assert 'mode != "align"' in runtime
    assert 'mamba_block != native_full_block' in runtime
    for field in ("prefix_caching", "mamba_mode", "physical_full_block_size",
                  "scheduler_lcm_block_size", "hash_block_size"):
        assert f'"{field}"' in runtime
    assert "self.mixed_decode_split = mixed_split" in backend
    assert 'emit_once("prefix_metadata"' in backend
    assert 'common_prefix_len=common_prefix_len' in backend
    feature_selection = backend[backend.index("mixed_split ="):backend.index("from ..telemetry")]
    assert "enable_prefix_caching" not in feature_selection
    assert "total - computed <= self.whole_limit" in scheduler


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
