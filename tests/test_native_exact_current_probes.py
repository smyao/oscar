"""Archive #126/#145/#173: raw-source oracle eligibility and stale-report gates.

Host contracts only: no simulated NPU acceptance, timing, or tolerance changes.
The cached reader remains v4, even when the fresh single-source policy changes.
"""
import pytest
import torch

from tools import probe_current_only as shared
from tools import probe_official_attention as official
from tools import official_runtime_oracle as v4
from tools import native_exact_current_reference as exact
from tools.official_fixture import make_fixture
from tools.probe_history_reuse import Shape


def target(policy='native_exact'):
    return {'experimental_official_math': True, 'experimental_aligned_math': True,
            'experimental_bf16_attention': True, 'experimental_current_only': True,
            'experimental_history_reuse': True, 'experimental_fast_unpack': True,
            'experimental_mixed_cv': True, 'experimental_striped_cache': True,
            'oscar_dequant_policy': 'fp32_metadata', 'oscar_current_only_policy': policy}


def assert_expected_equal(left, right):
    assert set(left) == set(right)
    for index in left:
        assert torch.equal(left[index][0], right[index][0])
        assert torch.equal(left[index][1], right[index][1])


def fixture():
    return make_fixture(torch, Shape('native_exact_oracle', (2, 1), (511, 0),
                        256, 1, 1, False), rotation_mode='nontrivial')


def test_only_qualified_zero_history_uses_original_projection_oracle():
    data = fixture()
    old = v4.sampled_fixture_expected(data, native_current=(True, True), dequant_mode=0)
    actual = shared._expected(data, target(), native_current=(True, True))
    # Request0 has history and must retain every cached-reader v4 value.
    assert_expected_equal({i: actual[i] for i in (0, 1)}, {i: old[i] for i in (0, 1)})
    # Request1 is a singleton fresh source: its raw answer is exactly V.
    assert torch.equal(actual[2][0], data['original_inputs']['cv'][2].expand(6, 256).float())
    expected = exact.sampled_fixture_expected(data, native_current=(True, True), dequant_mode=0)
    assert_expected_equal(actual, expected)


@pytest.mark.parametrize('native_current', [False, (False, False), (True, False)])
def test_disabled_fresh_qualification_keeps_the_original_v4_oracle(native_current):
    data = fixture()
    expected = v4.sampled_fixture_expected(data, native_current=native_current, dequant_mode=0)
    assert_expected_equal(shared._expected(data, target(), native_current=native_current), expected)


def test_explicit_rotated_policy_uses_old_oracle_even_with_a_fresh_suffix():
    data = fixture()
    expected = v4.sampled_fixture_expected(data, native_current=True, dequant_mode=0)
    assert_expected_equal(shared._expected(data, target('rotated_v4'), native_current=True), expected)


def test_new_oracle_cannot_reconstruct_raw_inputs_from_rotated_candidate_data():
    data = fixture()
    data.pop('original_inputs')
    with pytest.raises(ValueError, match='original projection inputs'):
        shared._expected(data, target(), native_current=(True, True))
    # When no fresh request is qualified, the unchanged v4 data is sufficient.
    expected = v4.sampled_fixture_expected(data, native_current=False, dequant_mode=0)
    assert_expected_equal(shared._expected(data, target(), native_current=False), expected)


def test_disabled_ab_side_has_complete_v4_policy_without_mutating_candidate():
    config = target()
    baseline = shared._configuration(config, False)
    candidate = shared._configuration(config, True)
    assert baseline['experimental_current_only'] is False
    assert baseline['oscar_current_only_policy'] == 'rotated_v4'
    assert baseline['oscar_dequant_policy'] == config['oscar_dequant_policy']
    assert candidate == config


def complete_current_report():
    config = target()
    fields = shared.precision_fields(config)
    rows = []
    for case in shared.case_plan():
        fresh, draft, slot_type, paired = case
        count = 3 if paired else 1
        row = {**fields, 'case': shared._case_name(case), 'status': 'passed',
            'first_draft': draft, 'slot_dtype': slot_type,
            'rotation': 'draft_identity' if draft else 'nontrivial_dense_fixture',
            'independent_oracle': 'passed', 'copy_lse_oracle': 'passed', 'max_output_abs': .001,
            'sampled_queries': len(shared.reuse._samples(fresh)) + (4 if paired else 0),
            'real_operator_calls': {'prepare_official_attention_tasks_out': count,
                'copy_validate_current_out': count, 'copy_validate_bf16_query_out': count,
                'store_aligned_int2_out': count}}
        if paired:
            row.update(ab_output='separate_BF16_native_contracts_passed', store_bitwise='passed',
                baseline='complete_rotated_current_CV', decode_after_store={'status': 'passed',
                    'reader': 'production_subsequent_q1', 'output_bitwise': 'passed',
                    'independent_oracle': 'passed', 'subsequent_store_bitwise': 'passed', 'sampled_queries': 2})
            row['real_operator_calls'].update(attention_cv_official_decode_out=1,
                                              attention_cv_official_decode_q1_out=1)
        rows.append(row)
    return {'cases': rows, **fields}


@pytest.mark.parametrize('field', ['current_only_policy', 'current_only_contract',
    'current_only_acceptance_sha256', 'current_only_oracle_sha256'])
@pytest.mark.parametrize('scope', ['report', 'case'])
def test_new_policy_rejects_missing_or_stale_precision_receipts(field, scope):
    report = complete_current_report()
    acceptance = {'performance': {'warmup': 2, 'repeats': 5}}
    shared.validate_case_evidence(report, target(), acceptance)
    record = report if scope == 'report' else report['cases'][0]
    record.pop(field)
    with pytest.raises(RuntimeError, match='precision policy/contract'):
        shared.validate_case_evidence(report, target(), acceptance)


def test_raw_query_guard_witness_is_required_and_old_stores_decode_gates_survive():
    report = complete_current_report()
    acceptance = {'performance': {'warmup': 2, 'repeats': 5}}
    report['cases'][0]['real_operator_calls'].pop('copy_validate_bf16_query_out')
    with pytest.raises(RuntimeError, match='raw query validation'):
        shared.validate_case_evidence(report, target(), acceptance)
    report = complete_current_report()
    report['cases'][0]['decode_after_store']['output_bitwise'] = 'failed'
    with pytest.raises(RuntimeError, match='store or existing q1'):
        shared.validate_case_evidence(report, target(), acceptance)


def test_runtime_receipt_adds_current_overlay_without_rewriting_dequant_fields():
    old = official.runtime_precision(target('rotated_v4'))
    new = official.runtime_precision(target())
    for field in ('policy', 'contract', 'dequant_modes', 'acceptance_sha256'):
        assert new[field] == old[field]
    assert new['dequant_modes'] == dict(prefill=0, decode=0, mtp=0, capture=0)
    assert new['current_only_policy'] == 'native_exact'
    assert new['current_only_contract'] == exact.CONTRACT
    assert new['current_only_acceptance_sha256'] == official.digest(exact.ACCEPTANCE)
    assert new['current_only_oracle_sha256'] == official.digest(exact.__file__)
    sources = official.source_binding()
    for name in ('oscar_ascend/integration/native_exact_current.py',
                 'oscar_ascend/integration/current_only_dispatch.py',
                 'oscar_ascend/integration/current_only_plan.py',
                 'oscar_ascend/integration/current_attention.py',
                 'tools/native_exact_current_reference.py', 'configs/acceptance_native_exact_current.json'):
        assert sources[name] == official.digest(official.ROOT / name)


def test_official_gate_rejects_old_runtime_receipt_for_new_policy():
    from test_probe_official_attention import complete_receipt
    report = complete_receipt()
    report['runtime_precision'] = official.runtime_precision(target('rotated_v4'))
    assert 'runtime_precision' in official.validate_evidence(report, target())
    report['runtime_precision'] = official.runtime_precision(target())
    assert official.validate_evidence(report, target()) == []
