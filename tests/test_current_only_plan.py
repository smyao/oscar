"""Archive #126/#129/#155: no fresh-current plan may drop existing history."""
import pytest
from oscar_ascend.integration.current_only_plan import plan_current_suffix


def test_decode_prefix_and_20k_fresh_suffix_with_padding():
    plan=plan_current_suffix((0,4,8,20008,20016),(25004,27004,20000),
                             actual_tokens=20008,padded_tokens=20016)
    assert (plan.prefix_requests,plan.prefix_tokens)==(2,8)
    assert (plan.fresh_requests,plan.fresh_tokens,plan.padded_tokens)==(1,20000,8)
    assert plan.cumulative==(20000,)


def test_any_positive_history_bound_keeps_full_cv_route():
    assert plan_current_suffix((0,4,20004),(27004,20001),actual_tokens=20004,
                               padded_tokens=20004) is None
    # A fresh request before a continuing request is not a contiguous suffix.
    assert plan_current_suffix((0,20000,20004),(20000,27004),actual_tokens=20004,
                               padded_tokens=20004) is None


def test_all_fresh_multiple_requests_and_reject_invalid_bound():
    plan=plan_current_suffix((0,1000,2000),(1000,1000),actual_tokens=2000,padded_tokens=2000)
    assert plan.prefix_tokens==0 and plan.cumulative==(1000,2000)
    with pytest.raises(ValueError,match='below query'):
        plan_current_suffix((0,1000),(999,),actual_tokens=1000,padded_tokens=1000)


def test_native_metadata_plan_is_disabled_for_draft_and_capture():
    import torch
    from enum import Enum
    from types import SimpleNamespace
    from oscar_ascend.integration.metadata import from_common, OscarMetadataError
    class AscendAttentionState(Enum):
        ChunkedPrefill=1
    common=SimpleNamespace(causal=True,
        query_start_loc=torch.tensor([0,4,14],dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0,4,14],dtype=torch.int32),
        seq_lens=torch.tensor([100,10],dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([100,10],dtype=torch.int32),
        slot_mapping=torch.arange(16,dtype=torch.int64),
        block_table_tensor=torch.zeros(2,8,dtype=torch.int32),
        num_reqs=2,num_actual_tokens=14,num_input_tokens=16,max_query_len=10,
        max_seq_len=100,attn_state=AscendAttentionState.ChunkedPrefill)
    plan=from_common(common,current_only=True).current_only_plan
    assert plan.prefix_tokens==4 and plan.fresh_tokens==10
    assert from_common(common,current_only=True,is_draft=True).current_only_plan is None
    assert from_common(common,current_only=True,capture_origin=True).current_only_plan is None
    del common.seq_lens_cpu_upper_bound
    with pytest.raises(OscarMetadataError,match='CPU sequence upper bounds'):
        from_common(common,current_only=True)
