"""Archive #148/#155: shared-history accounting must preserve physical work bounds."""
import pytest
from tools.experiments.shared_history_work import Shape, account


def test_cold_context_has_no_history_or_spill_savings():
    row=account(Shape(16260,0,16))
    assert row['eligible_clusters']==0
    assert row['legacy']['expanded_logical_rows']==row['shared']['expanded_logical_rows']==0
    assert row['row_reuse_factor'] is None
    assert row['unchanged']['QK_calls']==0


def test_original_query_task_and_kv_boundaries_are_retained():
    row=account(Shape(385,1025,4))
    # 19 request-relative groups; 7 complete groups after the first 12 have
    # terminal domains, but only the aligned group12..15 is a complete C4.
    assert row['query_groups']==19
    assert row['eligible_clusters']==1
    assert row['shared']['cache_K_plus_V_bytes']==16*1024*1024
    assert row['shared']['expanded_logical_rows']==961
    assert row['unchanged']['QK_calls']==row['unchanged']['PV_calls']
    assert row['unchanged']['online_softmax_KV_width']==256


def test_long_query_reuse_does_not_claim_reduced_cube_flops_or_HBM():
    small=account(Shape(3740,16260,4))
    large=account(Shape(16260,48840,16))
    assert small['row_reuse_factor']>50
    assert large['row_reuse_factor']>50
    assert large['legacy']['acc_max_sum_logical_GM_bytes']>20*large['shared']['acc_max_sum_boundary_GM_bytes']
    assert small['shared']['cache_K_plus_V_bytes']==large['shared']['cache_K_plus_V_bytes']
    assert large['speed_gate']=='not_measured'


def test_history_chunk_must_preserve_KV256_recurrence_boundaries():
    with pytest.raises(ValueError,match='aligned'):
        account(Shape(4096,20000,16,history_tile=8191))
