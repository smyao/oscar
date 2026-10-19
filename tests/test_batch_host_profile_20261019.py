"""Exercise real planner boundaries used by the offline profile, not serving."""
import json

import pytest

from tools.experiments.batch_host_profile_20261019 import ROOT, fixture, run_step
from tools.serving_variants import variant_config
from tools.experiments import batch_host_profile_20261019 as profile


@pytest.fixture(scope='module')
def candidate():
    return variant_config(json.loads((ROOT/'configs/target.json').read_text()), 'candidate')


@pytest.mark.parametrize('batch,split', [(31, True), (32, True), (33, False), (46, False)])
def test_continued_prefill_boundary_selects_actual_operator(candidate, batch, split):
    result = run_step(fixture(batch, 'continued_mixed'), 'continued_mixed', candidate)
    assert result['mixed_split'] is split
    assert result['current_suffix'] is False
    assert result['short_request_operator'] == (
        'attention_bf16_history_mdl_out' if split else
        'attention_bf16_history_window_range_cluster16_out')


def test_fresh_suffix_priority_preserves_short_route_past_32(candidate):
    result = run_step(fixture(46, 'fresh_mixed'), 'fresh_mixed', candidate)
    assert result['current_suffix'] and not result['mixed_split']
    assert result['short_prefix_tokens'] == 184
    assert result['short_request_operator'] == 'attention_bf16_history_mdl_out'


def test_capture_path_does_not_read_cpu_mirrors(candidate):
    common = fixture(46, 'pure_target')
    common.query_start_loc_cpu = None
    common.seq_lens_cpu_upper_bound = None
    result = run_step(common, 'pure_target', candidate)
    assert result['cpu_cumulative_cached'] is False


def test_fresh_step_does_not_reuse_stale_cumulative(candidate):
    common = fixture(20, 'continued_mixed')
    common._oscar_current_cumulative = (17,)
    assert run_step(common, 'continued_mixed', candidate)['mixed_split']
    assert common._oscar_current_cumulative == (17,)


def test_invalid_synthetic_request_count_rejected():
    with pytest.raises(ValueError, match='request count'):
        fixture(128, 'continued_mixed')


def test_native_group_build_is_shared_by_sixteen_layer_route_decisions(candidate, monkeypatch):
    original = profile.from_common
    calls = []

    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(profile, 'from_common', count)
    run_step(fixture(46, 'continued_mixed'), 'continued_mixed', candidate, layers=16)
    assert len(calls) == 1
