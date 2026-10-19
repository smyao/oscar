"""Archive #127/#128/#140/#148/#163/#166/#167: source-derived capacity boundaries.

Native planner execution is host arithmetic, never target memory profiling.
"""
import pytest

from tools.experiments.batch_capacity_20261019 import build_report, GIB


@pytest.fixture(scope='module')
def report():
    return build_report()


def test_native_shape_pool_and_exact_snapshot_bytes_are_not_encoding_ratio(report):
    assert report['model']['status']=='conditional_existing_native_shape_fixture'
    assert report['model']['gdn_shapes']==((6,2560),(12,128,128))
    assert (report['model']['conv_bytes'],report['model']['ssm_bytes'])==(30720,393216)
    for mode in report['modes']:
        assert mode['page_bytes_per_layer']==2*768*256*2+30720==817152
        assert mode['group_layer_counts']==[17,16,16,16]
        assert mode['gdn_padding_slots']==3
        assert mode['one_pool_block_and_null_bytes']==17*817152
        parts=mode['oscar_full_page_components']
        assert parts['precise_snapshot']==(64+256+3)*(2*256*2+8)
        assert sum(parts.values())==817152
        assert mode['oscar_full_tokens_per_page']==(2816 if mode['mode']=='none' else 2304)


def test_latest_observed_budget_is_not_total_device_memory_or_native_profile(report):
    budget=report['budgets']
    assert budget['current_oscar_budget_interval_bytes']==[int(14.525*GIB),int(14.535*GIB)]
    assert budget['scenarios']['suggested_fit_0_9_min_rank']==15035635302
    assert budget['native_same_run_profiled_budget_bytes'] is None
    assert report['fixed_oscar_allocations']['double_subtraction_from_profiled_budget'] is False
    assert report['fixed_oscar_allocations']['workspace_instances_per_rank_if_uniform_target_and_draft']==1
    assert report['fixed_oscar_allocations']['workspace_total_bytes']==610272512
    assert report['fixed_oscar_allocations']['persistent_rotations_bytes']==12845056


@pytest.mark.parametrize('mode_name,expected',[('none',[(20,46),(11,31),(6,19)]),('align',[(19,37),(11,25),(6,15)])])
def test_native_planner_capacity_includes_one_null_and_mtp_state_reservation(report,mode_name,expected):
    mode=next(item for item in report['modes'] if item['mode']==mode_name)
    rows=[row for row in mode['requests'] if row['additional_output_reserve']==0]
    available=mode['native_planner_allocations']['observed_oscar_rounding_lower']['num_pool_blocks']-1
    for row,pair in zip(rows,expected,strict=True):
        assert row['reserved_full_tokens']==row['context_tokens']+3
        cap=row['capacity']['observed_oscar_rounding_lower']
        assert tuple(cap[v]['max_B_by_peak_reservation'] for v in ('native','oscar'))==pair
        for variant in ('native','oscar'):
            per_request=row['per_request_bytes_per_rank'][variant]['total_peak_reserved_bytes']//(17*817152)
            assert pair[('native','oscar').index(variant)]*per_request<=available
            assert (pair[('native','oscar').index(variant)]+1)*per_request>available
        assert all(item['within_max_seqs_and_token_budget'] for item in row['batch_sweep'])


def test_real_selectors_keep_padded_later_q1_and_separate_first_current(report):
    expected_splits=[20,20,10,5,3,2,1]
    for item,split in zip(report['pure_decode_work'],expected_splits,strict=True):
        target,first,later=item['stages'];batch=item['batch']
        assert target['source_splits']==first['source_splits']==later['source_splits']==split
        assert target['operator']==first['operator']=='attention_bf16_history_mdl_out'
        assert later['operator']=='attention_bf16_history_mdl_q1_out'
        assert first['native_current_source2'] and not target['native_current_source2']
        assert later['calls_per_step']==2 and later['live_rows']==batch
        assert later['input_rows']==4*batch and later['padding_rows']==3*batch
        assert not later['compact_later_applies']
        assert later['task_descriptors_per_layer']==4*batch*3*split
    # B64's target graph pads 256 to its configured capture size 272; eager
    # MTP retains 256. This differs from assuming every stage shares N.
    target,first,later=report['pure_decode_work'][5]['stages']
    assert (target['input_rows'],first['input_rows'],later['input_rows'])==(272,256,256)


def test_report_is_source_bound_and_does_not_include_request_window_projection(report):
    paths={item['path'] for item in report['sources']}
    assert 'oscar_ascend/runtime.py' in paths
    assert 'references/vllm/vllm/v1/core/kv_cache_utils.py' in paths
    assert all(len(item['sha256'])==64 for item in report['sources'])
    assert not any('request_window' in path for path in paths)


def test_mixed_q4_prefix_boundary_uses_real_planner_and_original_split_capacity(report):
    mixed=report['mixed_continued_prefill_work']
    for item in mixed:
        assert item['within_max_seqs_and_token_budget']
        batch=item['q4_prefix_requests']
        assert (item['plan'] is not None)==(batch<=32)
        assert all(part['source_splits']==1 for part in item['launches'])
        assert all(part['cv_shape_tokens']==item['actual_tokens'] for part in item['launches'])
        if batch<=32:
            assert item['plan']['prefix_tokens']==4*batch
            assert item['launches'][0]['operator']=='attention_bf16_history_mdl_out'
        else:
            assert len(item['launches'])==1
            assert item['launches'][0]['operator']==item['whole_batch_operator']
        suffix=item['launches'][-1]
        expected=('attention_bf16_history_member_pipeline_cluster4_out'
                  if item['continuation_query_tokens']==512 else
                  'attention_bf16_history_window_range_cluster16_out')
        assert suffix['operator']==expected
    pure32=next(row for row in report['pure_decode_work'] if row['batch']==32)
    assert pure32['stages'][0]['source_splits']==3
    assert any(row['q4_prefix_requests']==33 and row['q4_prefix_tokens']==132 and
               row['plan'] is None for row in mixed)
