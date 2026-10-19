"""Offline source-counted sensitivity experiment; archive #148/#154/#155/#156/#167.

It evaluates an explicitly supplied service-cost model, not accelerator ISA,
CAModel, a calibrated scheduler, or a measured whole-model performance claim.
No production behavior changes. No fitted fractions or CPU-to-NPU conversion.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import product
import json
import math
from pathlib import Path

from tools.experiments.batch_kernel_cost_20261019 import data_bytes

ROOT = Path(__file__).resolve().parents[2]
ASSUMPTIONS = 'reports/batch_simulation_assumptions_20261019.json'
KEYS = (
    'temporary_equivalent_hbm_fraction',
    'additional_serial_cost_per_request_in_native_history_units',
    'common_fixed_cost_in_native_batch_history_units',
    'common_linear_cost_in_native_batch_history_units',
    'oscar_over_native_emitted_tokens_per_request',
)


def parameters(values):
    row = tuple(values[key] for key in KEYS)
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in row):
        raise ValueError('finite numeric assumptions required')
    gamma, extra, fixed, linear, acceptance = row
    if not (0 <= gamma <= 1 and min(extra, fixed, linear) >= 0 and 0 < acceptance <= 4):
        raise ValueError('invalid physical/sensitivity parameter')
    return row


def history_and_exact_work(step):
    """Exclude current source in BOTH variants, including first-MTP native FIA.

    Original source0+source1 counts retain precise reads, tag checks, Q/Rv
    requests, tail padding, scores/P and partial outputs. Only KV+page+tag
    reads are assigned mandatory cold HBM in this scenario. Rv/Q and all
    other GM requests have the explicitly variable equivalent HBM fraction.
    """
    batch = step['batch']
    counts = Counter()
    later_key = next(key for key in step if key.startswith('later_q1'))
    for stage, calls in ((step['target_q4'], 16), (step['first_q4'], 1), (step[later_key], 2)):
        for source in stage['per_request_source_counts'][:2]:
            counts.update({key: value * batch * calls for key, value in source.items()})
    mandatory = sum(counts[key] for key in ('packed_payload_read', 'page_table_read',
                                          'exact_bf16_kv_read', 'exact_tag_read'))
    total = data_bytes(counts)
    return {'batch': batch, 'context': step['context'], 'counts': dict(counts),
            'mandatory_cold_kv_page_tag_bytes': mandatory,
            'other_gm_request_bytes': total-mandatory,
            'total_source_gm_request_bytes': total,
            'native_ideal_kv_read_bytes': batch * step['context'] * 1024 * 19}


def evaluate(small, large, assumptions):
    gamma, extra, fixed, linear, acceptance = parameters(assumptions)
    native_history = small['native_ideal_kv_read_bytes']
    scale = large['batch']/small['batch']
    if small['context'] != large['context'] or scale < 1 or native_history <= 0:
        raise ValueError('matched context and ordered batch sizes required')
    native = 1 + fixed + linear

    def oscar(work):
        bscale = work['batch']/small['batch']
        byte_service = (work['mandatory_cold_kv_page_tag_bytes'] +
                        gamma*work['other_gm_request_bytes'])/native_history
        return fixed + bscale*(linear+extra) + byte_service

    old, new = oscar(small), oscar(large)
    denominator = large['other_gm_request_bytes']/native_history
    threshold = ((scale*acceptance*native-fixed-scale*(linear+extra)-
                  large['mandatory_cold_kv_page_tag_bytes']/native_history)/denominator
                 if denominator else None)
    return {'native_batch': small['batch'], 'oscar_batch': large['batch'],
            'capacity_ratio': scale, 'native_step_time_units': native,
            'oscar_same_batch_step_time_units': old, 'oscar_larger_batch_step_time_units': new,
            'oscar_same_batch_over_native_throughput': acceptance*native/old,
            'oscar_larger_batch_over_native_throughput': scale*acceptance*native/new,
            'oscar_larger_over_same_oscar_throughput': scale*old/new,
            'oscar_larger_over_same_oscar_step_latency': new/old,
            'break_even_other_gm_hbm_fraction_unclamped': threshold,
            'acceptance_ratio_required_to_match_native': new/(scale*native),
            'model_kind': 'synthetic_steady_decode_resource_sensitivity_not_current_device_prediction'}


def run(assumption_path, *, capacity_path=None, kernel_path=None):
    assumptions = json.loads(assumption_path.read_text())
    parameters(assumptions)
    capacity_path = capacity_path or ROOT/'reports/batch_capacity_20261019.json'
    kernel_path = kernel_path or ROOT/'reports/batch_kernel_cost_20261019.json'
    capacity, kernel = (json.loads(path.read_text()) for path in (capacity_path, kernel_path))
    if kernel['status'] != 'host_source_count_profile_completed' or not kernel['self_checks']['passed']:
        raise ValueError('executed current source-count report required')
    # Verify relevant production source identities before consuming cached work.
    for name, expected in kernel['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'stale source-count report: {name}')
    mode = next(item for item in capacity['modes'] if item['production_default'])
    steps = {(s['context'], s['batch']): s for s in kernel['capacity_matched_full_layer_steps']}
    rows, grid = [], []
    for request in mode['requests']:
        if request['additional_output_reserve'] != 512:
            continue
        cap = request['capacity']['observed_oscar_rounding_lower']
        length = request['context_tokens']
        small = history_and_exact_work(steps[length, cap['native']['max_B_by_peak_reservation']])
        large = history_and_exact_work(steps[length, cap['oscar']['max_B_by_peak_reservation']])
        center = evaluate(small, large, assumptions)
        sensitivity = []
        for gamma in (0, 0.25, 0.5, 1):
            changed = dict(assumptions, temporary_equivalent_hbm_fraction=gamma)
            sensitivity.append({'other_gm_hbm_fraction': gamma, **evaluate(small, large, changed)})
        per_acceptance = evaluate(small, large, dict(assumptions,
            oscar_over_native_emitted_tokens_per_request=0.8))
        for values in product(*(assumptions['predeclared_grid'][key] for key in KEYS)):
            params = dict(zip(KEYS, values))
            grid.append({'context': length, 'parameters': params, **evaluate(small, large, params)})
        rows.append({'context': length, 'same_oscar_at_native_capacity': small,
                     'oscar_at_larger_capacity': large, 'selected_scenario': center,
                     'gm_fraction_sensitivity': sensitivity,
                     'selected_scenario_if_emitted_tokens_ratio_0_8': per_acceptance})
    return {'schema': 1, 'evidence_kind': 'executed_deterministic_sensitivity_simulation',
            'assumptions': assumptions, 'rows': rows, 'parameter_grid': grid,
            'grid_cases': len(grid), 'absolute_time_and_tps': None,
            'formula': 'T_native=1+F+L; T_oscar(B)=F+(B/Bnative)*(L+U)+(KV_oscar+gamma*other_GM)/KV_native(Bnative). TPS ratio=(B/Bnative)*accepted_ratio*T_native/T_oscar.',
            'scope': 'Stationary pure decode, no APC, same net KV budget, conditional local model shape, same KV length each draft call. 17 q4+2 q1 history/precise-window sources. Assumed service rate and additive unmodelled costs; no resource overlap fitted.',
            'not_calibrated': ['F and L common cost shares', 'U net extra serial cost',
                               'gamma equivalent HBM fraction (not raw L2 hit rate)',
                               'current MTP emitted-token ratio', 'actual admitted B',
                               'prefill/mixed/ramp/drain phase distribution'],
            'excluded_from_source_bytes': ['current CV/native FIA', 'Q rotation producer',
                'quantization/store', 'prepare', 'guards', 'merge/final copy',
                'GDN/MLP/TP/logits/host/framework'],
            'exclusion_accounting': 'Common stages are only represented by assumed F/L; OSCAR-exclusive missing stages are only represented by assumed U. No claim these assumptions bound or fit actual cost.',
            'sources': [{'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                        for path in (capacity_path, kernel_path, assumption_path, Path(__file__))]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assumptions', type=Path, default=ROOT/ASSUMPTIONS)
    parser.add_argument('--capacity-report', type=Path)
    parser.add_argument('--kernel-report', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/batch_throughput_simulation_20261019.json')
    args = parser.parse_args()
    report = run(args.assumptions, capacity_path=args.capacity_report, kernel_path=args.kernel_report)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'grid_cases': report['grid_cases'], 'output': str(args.output),
                      'selected_scenarios': [row['selected_scenario'] for row in report['rows']]}))


if __name__ == '__main__':
    main()
