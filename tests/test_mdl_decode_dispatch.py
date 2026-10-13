"""Archive #126/#129/#145/#148/#150/#153/#154: MDL q4/q1 production seams.

Host ABI/shape/source checks are not numeric or target NPU acceptance.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from oscar_ascend.ops.cv_dispatch import select_cv_op, MDL_DECODE_CV_OP, MDL_DECODE_Q1_CV_OP
from oscar_ascend.runtime import AscendRuntimeProvider, WorkspaceGeometry
from oscar_ascend.integration.runtime_api import OscarReadinessError
from oscar_ascend.integration.cache_storage_ops import storage_reader
from tools.serving_variants import variant_config
from tools.package_mdl_decode import package

FLAGS = dict(fast_unpack=True, mixed_cv=True, striped_cache=True, decode_bundle=True)

@pytest.mark.parametrize('tokens,query,later_q1,expected', [
    (8,4,False,MDL_DECODE_CV_OP), (128,4,False,MDL_DECODE_CV_OP),
    (2048,4,False,MDL_DECODE_CV_OP), (128,1,True,MDL_DECODE_Q1_CV_OP),
    (2048,1,True,MDL_DECODE_Q1_CV_OP),
    (256,128,False,'attention_cv_window_range_cluster4_out'),
    (8192,8192,False,'attention_cv_window_range_cluster16_out'),
    (512,33,False,'attention_cv_window_range_balanced_out'),
])
def test_mdl_replaces_only_qualified_q4_q1_routes(tokens,query,later_q1,expected):
    chosen=select_cv_op(16,6,1,tokens,query,q1_draft=later_q1,mdl_decode=True,**FLAGS)
    old=select_cv_op(16,6,1,tokens,query,q1_draft=later_q1,**FLAGS)
    assert chosen==expected
    assert (chosen!=old)==(old in {'attention_cv_bundle_decode_out','attention_cv_bundle_q1_out'})
    assert storage_reader(SimpleNamespace(cache_format='striped_v1'),chosen)==chosen

@pytest.mark.parametrize('bad',[None,1,'true'])
def test_flag_is_strict_at_three_boundaries(bad):
    with pytest.raises(ValueError,match='explicit boolean'):
        variant_config({'experimental_mdl_decode':bad},'candidate')
    with pytest.raises(OscarReadinessError,match='explicit boolean'):
        AscendRuntimeProvider({'experimental_mdl_decode':bad})
    with pytest.raises(ValueError,match='explicit boolean'):
        select_cv_op(16,6,1,128,4,mdl_decode=bad,**FLAGS)

def test_mdl_requires_complete_base():
    with pytest.raises(ValueError,match='MDL decode requires'):
        variant_config({'experimental_mdl_decode':True},None)
    with pytest.raises(OscarReadinessError,match='MDL decode requires'):
        AscendRuntimeProvider({'experimental_mdl_decode':True})
    with pytest.raises(ValueError,match='MDL decode requires'):
        select_cv_op(16,6,1,128,4,mdl_decode=True)

def test_packaged_source_is_reversible_and_old_source_frozen():
    report=package(check=True)
    assert report['inverse_full_source_bytes'] is True
    assert report['workspace_bytes_per_core_D256']==1146880
    header=Path('csrc/include/oscar_mdl_decode_launch.h').read_text()
    assert 'attention_decode_bundle_workspace_per_core(256)==1146880' in header
    assert WorkspaceGeometry(128,6,1,256,history_cluster_size=16).cv_bytes>=1146880
    binding=Path('csrc/striped_attention_bindings.cpp').read_text()
    assert 'AttentionStriped<oscar_ascend::attention_cv_mdl_decode_launch,true,oscar_ascend::attention_decode_bundle_workspace_per_core(256)>' in binding
    assert 'AttentionStriped<oscar_ascend::attention_cv_mdl_decode_q1_launch,true,oscar_ascend::attention_decode_bundle_workspace_per_core(256)>' in binding
    assert report['q1_operator']==MDL_DECODE_Q1_CV_OP

@pytest.mark.parametrize('missing', [MDL_DECODE_CV_OP, MDL_DECODE_Q1_CV_OP])
def test_missing_mdl_capability_fails_before_npu(monkeypatch, missing):
    from oscar_ascend.ops import loader
    config=variant_config(json.loads(Path('configs/target.json').read_text()),'candidate')
    provider=AscendRuntimeProvider(config)
    monkeypatch.setenv('ASCEND_RT_VISIBLE_DEVICES',','.join(map(str,config['devices'])))
    monkeypatch.setattr(loader,'require_production_ops',lambda:None)
    def require(names):
        if missing in names:
            raise RuntimeError('MDL extension missing')
    monkeypatch.setattr(loader,'require_capabilities',require)
    with pytest.raises(RuntimeError,match='MDL extension missing'):
        provider.assert_ready()

@pytest.mark.parametrize('target,later_q1', [(True,False),(False,False),(False,True)])
def test_shared_candidate_reaches_actual_forward_target_first_and_later_draft(target,later_q1):
    from test_runtime import test_forward_dispatches_native_draft_strides_and_int32_slots_in_order
    test_forward_dispatches_native_draft_strides_and_int32_slots_in_order(
        128,20,16,8,16,later_q1,True,True,True,_mdl_variant='candidate',_mdl_target=target)


def test_unqualified_request_window_mdl_combination_is_explicit():
    with pytest.raises(ValueError,match="request-window MDL reader is not qualified"):
        variant_config({'experimental_request_windows':True},'candidate')
    config=variant_config({},'candidate')
    config['experimental_request_windows']=True
    with pytest.raises(OscarReadinessError,match="request-window MDL reader is not qualified"):
        AscendRuntimeProvider(config)
    assert variant_config({},'candidate')['enable_prefix_caching'] is True


def test_qualified_mdl_q1_q4_and_c4_pipeline_compose_in_shared_preset():
    """Archive #150/#153: combined flags must reach all three real readers."""
    config=variant_config({},'candidate')
    flags={name:config['experimental_'+name] for name in
           ('fast_unpack','mixed_cv','striped_cache','decode_bundle','c4_member_pipeline','mdl_decode')}
    assert config['enable_prefix_caching'] is True
    assert 'max_num_batched_tokens' not in config and 'scheduler_cls' not in config
    cases=((128,4,False,MDL_DECODE_CV_OP),(128,1,True,MDL_DECODE_Q1_CV_OP),
           (2048,2048,False,'attention_cv_member_pipeline_cluster4_out'),
           (8192,8192,False,'attention_cv_window_range_cluster16_out'))
    for tokens,length,later,expected in cases:
        assert select_cv_op(16,6,1,tokens,length,q1_draft=later,**flags)==expected
