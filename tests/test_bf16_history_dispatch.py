"""Archive #126/#148/#150/#153/#157-163: research host seams, not NPU acceptance.

All six selectors retain old eligibility. Public FP32 state, fixed workspace,
source guards and explicit precision remain separate from new operand math.
"""
from dataclasses import replace
import hashlib
from pathlib import Path
import re

import pytest

from oscar_ascend.ops.contracts import (
    BF16_HISTORY_CAPABILITIES, PRODUCTION_CAPABILITIES, SOURCE_CAPABILITIES,
    source_capabilities,
)
from oscar_ascend.ops.cv_dispatch import (
    BF16_HISTORY_OP_MAP, BF16_HISTORY_CLUSTER_OPS, CLUSTER_CV_OPS, select_cv_op,
)
from oscar_ascend.runtime import WorkspaceGeometry
from tools.serving_variants import research_variant_config

ROOT = Path(__file__).resolve().parents[1]
FLAGS = dict(fast_unpack=True, mixed_cv=True, striped_cache=True,
             decode_bundle=True, c4_member_pipeline=True, mdl_decode=True)


@pytest.mark.parametrize("heads,tokens,length,later,old", [
    (6,128,4,False,"attention_cv_mdl_decode_out"),
    (6,16384,1,True,"attention_cv_mdl_decode_q1_out"),
    (6,84,84,False,"attention_cv_member_pipeline_cluster4_out"),
    (6,512,None,False,"attention_cv_member_pipeline_cluster4_out"),
    (6,8192,8192,False,"attention_cv_window_range_cluster16_out"),
    (6,83,83,False,"attention_cv_window_range_out"),
    (6,512,33,False,"attention_cv_window_range_balanced_out"),
    (16,128,1,True,"attention_cv_window_range_out"),
    (16,512,1,True,"attention_cv_window_range_balanced_out"),
])
def test_all_old_eligibility_results_map_explicitly(heads,tokens,length,later,old):
    assert select_cv_op(16,heads,1,tokens,length,q1_draft=later,**FLAGS) == old
    assert select_cv_op(16,heads,1,tokens,length,q1_draft=later,
                        bf16_history=True,**FLAGS) == BF16_HISTORY_OP_MAP[old]
    assert (old in CLUSTER_CV_OPS) == (BF16_HISTORY_OP_MAP[old] in BF16_HISTORY_CLUSTER_OPS)


@pytest.mark.parametrize("flag", tuple(FLAGS))
def test_incomplete_bundle_cannot_silently_use_an_old_reader(flag):
    with pytest.raises(ValueError, match="complete C16 candidate bundle"):
        select_cv_op(16,6,1,128,4,bf16_history=True,**{**FLAGS,flag:False})


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_selector_requires_strict_boolean(value):
    with pytest.raises(ValueError, match="explicit booleans"):
        select_cv_op(16,6,1,128,4,bf16_history=value,**FLAGS)


def test_missing_mapping_is_an_error_not_a_fallback(monkeypatch):
    monkeypatch.delitem(BF16_HISTORY_OP_MAP,"attention_cv_mdl_decode_out")
    with pytest.raises(ValueError, match="no registered reader"):
        select_cv_op(16,6,1,128,4,bf16_history=True,**FLAGS)


@pytest.mark.parametrize("cluster,pipeline,expected", [(1,False,1212416),
    (4,True,2887680),(16,True,5259264)])
def test_workspace_uses_bounded_residual_typed_arenas(cluster,pipeline,expected):
    old=WorkspaceGeometry(16384,6,1,256,cube_cores=20,
                          history_cluster_size=cluster,c4_member_pipeline=pipeline)
    new=replace(old,bf16_history=True)
    assert new.cv_bytes == 20*expected > old.cv_bytes
    assert replace(new,tokens=128).cv_bytes == new.cv_bytes
    old_bytes,new_bytes=old.buffer_bytes,new.buffer_bytes
    changed={name for name in old_bytes if old_bytes[name]!=new_bytes[name]}
    assert changed == {"cv"}
    assert new.buffer_bytes["query_rot"] == new.history_tokens*6*256*4
    assert new.buffer_bytes["query_rot"] == old.buffer_bytes["query_rot"]
    with pytest.raises(ValueError,match="D256"):
        replace(new,head_dim=128)
    with pytest.raises(ValueError,match="explicit boolean"):
        replace(new,bf16_history=1)


@pytest.mark.parametrize("later,name", [
    (False, "attention_bf16_history_mdl_out"),
    (True, "attention_bf16_history_mdl_q1_out"),
])
def test_single_cluster_arena_reserves_mdl_double_kv_storage(later, name):
    # #173: both MDL readers have M32 but two K/V tiles. The generic M128
    # single-buffer family is 1179648 B and cannot cover 1212416 B MDL.
    geometry = WorkspaceGeometry(128,6,1,256,cube_cores=20,
        history_cluster_size=1,bf16_history=True)
    rows, kv, dim = 32, 256, 256
    required = ((2*rows*dim + rows*kv)*4 +
                (rows*dim + 4*kv*dim + rows*kv)*2*2)
    assert required == 1212416
    assert geometry.buffer_bytes["cv"] == 20*required
    # Production still requires the complete C16 bundle; do not relax that
    # guard just to exercise the allocator's general capacity contract.
    assert select_cv_op(16,6,1,128,1 if later else 4,q1_draft=later,
                        bf16_history=True,**FLAGS) == name
    with pytest.raises(ValueError,match="complete C16 candidate bundle"):
        select_cv_op(1,6,1,128,1 if later else 4,q1_draft=later,
                     bf16_history=True,**FLAGS)


@pytest.mark.parametrize("n,length,later,target", [
    (8,4,False,True),(8,4,False,False),(8,1,True,False),
    (256,128,False,False),(8192,8192,False,False),
    (8,8,False,False),(256,33,False,False),
])
def test_research_config_reaches_actual_forward_abi(n,length,later,target):
    from test_runtime import test_forward_dispatches_native_draft_strides_and_int32_slots_in_order
    test_forward_dispatches_native_draft_strides_and_int32_slots_in_order(
        n,1,1,n,16,later,True,True,True,_member_pipeline=True,
        _bf16_history=True,_bf16_max_query=length,_mdl_target=target)


def test_default_build_and_production_require_every_bf16_reader():
    assert BF16_HISTORY_CAPABILITIES == frozenset(BF16_HISTORY_OP_MAP.values())
    assert BF16_HISTORY_CAPABILITIES <= SOURCE_CAPABILITIES
    assert BF16_HISTORY_CAPABILITIES <= PRODUCTION_CAPABILITIES
    assert "attention_cv_out" not in PRODUCTION_CAPABILITIES
    with pytest.raises(ValueError, match="required"):
        source_capabilities(bf16_history=False)
    assert source_capabilities() == SOURCE_CAPABILITIES
    assert source_capabilities(bf16_history=True) == SOURCE_CAPABILITIES|BF16_HISTORY_CAPABILITIES
    with pytest.raises(ValueError,match="explicit boolean"):
        source_capabilities(bf16_history=1)
    cpp=(ROOT/"csrc/torch_bindings.cpp").read_text()
    body=re.search(r'std::vector<std::string>\{(.*?)\};\}\);',cpp,re.S)[1]
    conditional=re.search(r'#ifdef OSCAR_BUILD_BF16_HISTORY(.*?)#endif',body,re.S)[1]
    assert set(re.findall(r'"([a-z0-9_]+)"',conditional)) == BF16_HISTORY_CAPABILITIES
    assert set(re.findall(r'"([a-z0-9_]+)"',body)) == source_capabilities(bf16_history=True)
    cmake=(ROOT/"csrc/CMakeLists.txt").read_text()
    conditional=re.search(r'set\(OSCAR_BF16_CAPABILITIES_JSON ", (.*?)\)',cmake,re.S)[1]
    assert set(re.findall(r'\\"([a-z0-9_]+)\\"',conditional)) == BF16_HISTORY_CAPABILITIES
    assert '${OSCAR_BF16_CAPABILITIES_JSON}' in cmake


def test_frozen_guard_and_existing_numerical_kernels_are_unchanged():
    from tools.observe_serve import FE0_KERNEL_SHA256
    for path,expected in FE0_KERNEL_SHA256.items():
        assert hashlib.sha256((ROOT/"csrc/kernels"/path).read_bytes()).hexdigest()==expected


def test_research_config_preserves_mtp_graph_apc_and_native_scheduler():
    import json
    original=json.loads((ROOT/"configs/target.json").read_text())
    selected=research_variant_config({**original,"experimental_bf16_history":True},"candidate")
    for key in ("speculative_config","compilation_config","devices","port"):
        assert selected[key]==original[key]
    assert selected["enable_prefix_caching"] is False
    assert selected["max_num_batched_tokens"]==16384
    assert selected.get("scheduler_cls") is None
    assert selected.get("experimental_request_windows",False) is False
