"""Archive #34/#36/#126/#145/#148/#164: host v4 lifecycle wiring gates.

CPU doubles check which real-device entrypoints would execute, not NPU,
graph or model quality. Legacy v2/v3 reports cannot satisfy the v4 witnesses.
"""
from dataclasses import replace
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from tools import probe_current_only as shared
from tools import probe_prefix_cache as prefix
from tools import probe_mixed_prepare_reuse as prepare
from tools import official_reference as ref
from tools import official_runtime_oracle as sampled
from tools.serving_variants import variant_config


def target():
    return variant_config(json.loads((shared.ROOT / 'configs/target.json').read_text()), 'candidate')


def test_official_shared_selection_and_contract_are_not_legacy_aligned():
    config = target()
    assert shared.math_contract(config) == ref.CONTRACT
    assert shared.selected_prepare(config) == 'prepare_official_attention_tasks_out'
    assert set(shared.selected_operators(config).values()) == {
        'attention_cv_official_decode_out', 'attention_cv_official_decode_q1_out',
        'attention_cv_official_cluster4_out', 'attention_cv_official_cluster16_out'}
    fields = shared.precision_fields(config)
    assert fields['official_math'] and fields['cache_format'] == ref.CACHE_FORMAT
    assert fields['performance'] == 'not_run_user_requested'
    assert shared.selected_operators(fields) == shared.selected_operators(config)
    assert prefix.selected_reader({**config, 'enable_prefix_caching': True}) == 'attention_cv_official_decode_out'


@pytest.mark.parametrize('fault', ['prepare', 'writer', 'final', 'old_reader', 'old_prepare'])
def test_official_call_evidence_rejects_old_or_partial_routes(fault):
    config = target()
    calls = {'prepare_official_attention_tasks_out': 1, 'store_aligned_int2_out': 1,
             'copy_validate_current_out': 1, 'attention_cv_official_decode_q1_out': 1}
    shared.validate_aligned_calls(calls, config)
    if fault == 'prepare': calls.pop('prepare_official_attention_tasks_out')
    elif fault == 'writer': calls.pop('store_aligned_int2_out')
    elif fault == 'final': calls.pop('copy_validate_current_out')
    elif fault == 'old_reader': calls['attention_cv_aligned_decode_q1_out'] = 1
    else: calls['prepare_attention_tasks_out'] = 1
    with pytest.raises(RuntimeError):
        shared.validate_aligned_calls(calls, config)


def test_runtime_receives_original_projections_and_oracle_retains_rotated_inputs(monkeypatch):
    fixture = shared.make_fixture(torch, shared.reuse.Shape('official_lifecycle', (4,), (511,),
        256, 1, 1, False), target(), rotation_mode='nontrivial')
    tensors = shared.device_tensors(torch, fixture, 'cpu')
    for name in ('q', 'ck', 'cv'):
        assert torch.equal(tensors[name], fixture['original_inputs'][name])
    assert torch.equal(fixture['cpu']['q'], fixture['cpu']['qr'])
    assert not torch.equal(tensors['q'], fixture['cpu']['q'])
    # Model/backend construction is outside this host allocation test.
    class HostAttentionImpl:
        pass
    monkeypatch.setitem(sys.modules, 'oscar_ascend.integration.impl',
                        SimpleNamespace(OscarAttentionImpl=HostAttentionImpl))
    from oscar_ascend import runtime
    monkeypatch.setattr(runtime, 'GraphWorkspace', lambda geometry, device: SimpleNamespace(
        geometry=geometry, final_status=torch.empty((geometry.tokens, geometry.query_heads), dtype=torch.int32)))
    impl, state = shared._runtime(torch, fixture, tensors, 'cpu', 1, target(), True)
    assert state.cache_format == ref.CACHE_FORMAT
    assert state.workspace.geometry.official_math
    assert state.workspace.final_status.shape == (fixture['tokens'], fixture['spec'].heads)
    result = shared._expected(fixture, target(), native_current=False)
    assert set(result) == set(fixture['expected'])
    assert all(torch.isfinite(value[0]).all() for value in result.values())


def test_sampled_oracle_preserves_fixed_map_and_final_inverse():
    from test_official_reference import fixed_map_fixture
    fixture, tasks = fixed_map_fixture()
    fixture['spec'] = SimpleNamespace(dim=256, kv_heads=1, splits=1, recent_tokens=3,
                                      qlens=(2,), contexts=(70,))
    fixture['page_assignments'] = [(0,)]
    fixture['expected'] = {0: None, 1: None}
    sampled_result = sampled.sampled_fixture_expected(fixture, native_current=False, dequant_mode=0)
    full = ref.fixture_tasks_reference(fixture, tasks, 0)
    for index in sampled_result:
        assert torch.equal(sampled_result[index][0], full['output'][index].float())
        torch.testing.assert_close(sampled_result[index][1], full['output_lse'][index], atol=1e-6, rtol=0)
    broken = {**fixture, 'cache_format': 'striped_fp32_v2'}
    with pytest.raises(ValueError, match='format'):
        sampled.sampled_fixture_expected(broken, native_current=False, dequant_mode=0)


def test_official_prepare_reuse_exercises_new_prepare_without_suppressing_current(monkeypatch):
    from test_mixed_prepare_reuse import HostOps, pinned_context
    pinned_context(monkeypatch)
    class OfficialOps(HostOps):
        def prepare_attention_tasks_out(self, *args):
            pytest.fail('v4 selected the old moving-tier prepare')
        def prepare_official_attention_tasks_out(self, *args):
            return HostOps.prepare_attention_tasks_out(self, *args)
    workspace, audit = prepare.allocate(torch, 'cpu')
    metadata, _ = prepare.parent_metadata(torch, prepare.input_values(torch, 32768), 'cpu')
    def forbidden_suppression(*args):
        pytest.fail('v4 must retain FP32 current CV tasks')
    snapshots = []
    for enabled in (False, True):
        receipt = prepare.execute_epoch(torch, OfficialOps(), workspace, audit, metadata,
            enabled=enabled, epoch=enabled, official=True, suppress=forbidden_suppression)
        assert prepare._counts_valid(receipt, 'on' if enabled else 'off', official=True)
        assert receipt['suppress'] == 0
        assert receipt['prepare_operator'] == 'prepare_official_attention_tasks_out'
        snapshots.append(prepare._snapshot(audit))
    assert all(torch.equal(snapshots[0][name], snapshots[1][name]) for name in snapshots[0])


def test_prefix_execute_runs_official_prepare_reader_and_final_inverse_on_fixed_buffers():
    config = {**target(), 'enable_prefix_caching': True}
    state = prefix.allocate(torch, 'cpu', 1, bf16=True, aligned=True, official=True)
    logical = tuple((torch.zeros((512, 1, 256), dtype=torch.bfloat16),
                     torch.zeros((512, 1, 256), dtype=torch.bfloat16)) for _ in range(2))
    inputs = prefix.make_inputs(torch, logical, ((0,), (1,)), 2)
    calls = []
    def prepare_op(*args):
        calls.append('prepare_official_attention_tasks_out')
        args[3].zero_(); args[4].zero_()
    def qcopy(q, slots, out, status, cores): out.copy_(q); status.zero_()
    def reader(*args):
        calls.append('attention_cv_official_decode_out')
        args[10].zero_(); args[11].zero_(); args[12].zero_()
    def merge(partial, lse, output, output_lse, status):
        output.fill_(1.0052083730697632); output_lse.zero_(); status.zero_()
    def final_copy(value, lse, output, output_lse, status):
        calls.append('copy_validate_current_out')
        output.copy_(value); output_lse.copy_(lse.squeeze(-1)); status.zero_()
    def store(*args): calls.append('store_aligned_int2_out')
    ops = SimpleNamespace(prepare_official_attention_tasks_out=prepare_op,
        copy_validate_bf16_query_out=qcopy, attention_cv_official_decode_out=reader,
        merge_lse_out=merge, copy_validate_current_out=final_copy, store_aligned_int2_out=store)
    pointers = {name: state[name].data_ptr() for name in ('output', 'rotated_bf16', 'inverse_bf16', 'bf16')}
    prefix.execute(ops, state, inputs, prefix.selected_reader(config), 'unused_legacy_rotate', 1)
    assert calls == ['prepare_official_attention_tasks_out', 'attention_cv_official_decode_out',
                     'copy_validate_current_out', 'store_aligned_int2_out']
    assert pointers == {name: state[name].data_ptr() for name in pointers}
    assert torch.equal(state['bf16'], torch.full_like(state['bf16'], 1.0078125))


def test_official_current_only_reports_require_new_readers_and_final_copy_counts():
    from test_probe_current_only import complete_report
    config = target()
    report = complete_report()
    for row, case in zip(report['cases'], shared.case_plan()):
        paired = case[3]
        row.update(shared.precision_fields(config))
        row['real_operator_calls'] = {'prepare_official_attention_tasks_out': 3 if paired else 1,
            'copy_validate_current_out': 3 if paired else 1, 'store_aligned_int2_out': 3 if paired else 1}
        if paired:
            row.update(ab_output='separate_BF16_native_contracts_passed', baseline='complete_rotated_current_CV')
            row['real_operator_calls'].update(attention_cv_official_decode_out=1,
                                              attention_cv_official_decode_q1_out=1)
    acceptance = {'performance': {'warmup': 2, 'repeats': 5}}
    shared.validate_case_evidence(report, config, acceptance)
    report['cases'][0]['real_operator_calls']['copy_validate_current_out'] = 1
    with pytest.raises(RuntimeError, match='copy/writer'):
        shared.validate_case_evidence(report, config, acceptance)


def test_official_first_mtp_reports_require_current_cv_not_native_partial():
    from test_probe_first_mtp_current import evidence
    from tools import probe_first_mtp_current
    config = target()
    report, acceptance = evidence()
    report.update(shared.precision_fields(config))
    for row in report['cases']:
        row.update(ab_output='official_rotated_FP32_partials_passed',
            ab_lse='official_rotated_FP32_partials_passed', performance='not_run_user_requested',
            native_current_calls={'baseline': 0, 'candidate': 0}, final_inverse_calls=4,
            bf16_cv_calls={side: {'attention_cv_official_cluster16_out': 1,
                                 'attention_cv_official_decode_q1_out': 1}
                           for side in ('baseline', 'candidate')},
            real_operator_calls={side: {'prepare_official_attention_tasks_out': 2,
                'store_aligned_int2_out': 2, 'copy_validate_current_out': 2,
                'attention_cv_official_cluster16_out': 1, 'attention_cv_official_decode_q1_out': 1}
                for side in ('baseline', 'candidate')})
        row['decode_after_store']['operator'] = 'attention_cv_official_decode_q1_out'
    probe_first_mtp_current.validate_case_evidence(report, acceptance)
    report['cases'][0]['native_current_calls']['candidate'] = 1
    with pytest.raises(RuntimeError, match='route witnesses'):
        probe_first_mtp_current.validate_case_evidence(report, acceptance)
