"""Limiting cases and independently substituted break-even of offline model."""
import json

import pytest

from tools.experiments.batch_throughput_simulation_20261019 import (
    ASSUMPTIONS, ROOT, evaluate, history_and_exact_work, parameters,
)


@pytest.fixture
def settings():
    return json.loads((ROOT/ASSUMPTIONS).read_text())


def work(batch, persistent=100, other=0):
    return {'batch': batch, 'context': 1024,
            'native_ideal_kv_read_bytes': batch*100,
            'mandatory_cold_kv_page_tag_bytes': batch*persistent,
            'other_gm_request_bytes': batch*other}


def test_batch_cannot_improve_linear_memory_work_without_fixed_cost(settings):
    settings.update(common_fixed_cost_in_native_batch_history_units=0,
                    common_linear_cost_in_native_batch_history_units=1,
                    additional_serial_cost_per_request_in_native_history_units=0)
    result = evaluate(work(4), work(16), settings)
    assert result['oscar_larger_batch_over_native_throughput'] == 1
    assert result['oscar_larger_over_same_oscar_throughput'] == 1


def test_fixed_cost_amortization_matches_independent_request_accounting(settings):
    settings.update(common_fixed_cost_in_native_batch_history_units=1,
                    common_linear_cost_in_native_batch_history_units=0,
                    additional_serial_cost_per_request_in_native_history_units=0)
    # 4 requests take 2 units; 16 requests take 5 units. 8/5 throughput gain.
    result = evaluate(work(4), work(16), settings)
    assert result['oscar_larger_batch_over_native_throughput'] == 1.6


def test_break_even_substitutes_back_to_equal_throughput(settings):
    result = evaluate(work(4, 14, 260), work(12, 14, 260), settings)
    gamma = result['break_even_other_gm_hbm_fraction_unclamped']
    settings['temporary_equivalent_hbm_fraction'] = gamma
    assert evaluate(work(4, 14, 260), work(12, 14, 260), settings)[
        'oscar_larger_batch_over_native_throughput'] == pytest.approx(1)


def test_acceptance_loss_changes_tps_not_latency_or_capacity(settings):
    before = evaluate(work(4, 14, 260), work(12, 14, 260), settings)
    settings['oscar_over_native_emitted_tokens_per_request'] = 0.8
    after = evaluate(work(4, 14, 260), work(12, 14, 260), settings)
    assert after['oscar_larger_batch_step_time_units'] == before['oscar_larger_batch_step_time_units']
    assert after['capacity_ratio'] == before['capacity_ratio']
    assert after['oscar_larger_batch_over_native_throughput'] == pytest.approx(
        before['oscar_larger_batch_over_native_throughput']*0.8)
    assert after['oscar_larger_batch_over_native_throughput'] < before['oscar_larger_batch_over_native_throughput']


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -0.1, 1.1, True])
def test_invalid_miss_assumption_is_rejected(settings, value):
    settings['temporary_equivalent_hbm_fraction'] = value
    with pytest.raises(ValueError):
        parameters(settings)


def test_current_source_is_not_double_counted_as_first_draft_cv():
    source = {'packed_payload_read': 100}
    stage = {'per_request_source_counts': [source, {}, {'packed_payload_read': 10**15}]}
    step = {'batch': 3, 'context': 1024, 'target_q4': stage, 'first_q4': stage,
            'later_q1_retained_eager_first_n': stage}
    counted = history_and_exact_work(step)
    assert counted['mandatory_cold_kv_page_tag_bytes'] == 19*3*100
