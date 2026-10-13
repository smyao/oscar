"""Archive #129/#140/#148/#154-156: source work and conditional C4 gain only.

Measured closed C769 periods constrain a hypothesis, not full-kernel timing.
Neither the H32 helper gain nor C337 full ratio is multiplied into this model.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from csrc.experiments.c4_phase_profile.full_partition_budget import ownership
from csrc.experiments.c4_composed_long.generate import pair_sources, sha


def analyze():
    provenance = ROOT / 'reports/c4_phase_profile_q336_20261009/full_partition_conditional_budget.json'
    evidence = json.loads(provenance.read_text())['coefficient_evidence']
    c1 = evidence['C1_full256_ack_period']
    c4 = evidence['C4_full256_ack_period']
    proposed = evidence['proposed_C4_full256_pipeline_prefetch_budget']
    work = ownership(336, 1341)
    rows = []
    for owner in work['owners']:
        c1_full = sum(j['full_kv_units'] for j in owner['history'] if j['family'] != 'C4_mature')
        c4_full = sum(j['full_kv_units'] for j in owner['history'] if j['family'] == 'C4_mature')
        rows.append({'core': owner['core'], 'C1_full256': c1_full, 'C4_full256': c4_full,
            'old_full_history_proxy': c1_full * c1 + c4_full * c4,
            'proposed_full_history_proxy': c1_full * c1 + c4_full * proposed,
            'unknown_tails': [{'family': j['family'], 'rows': j['tail_kv_rows']} for j in owner['history']],
            'window_groups': owner['window_groups'], 'empty_current_groups': owner['empty_current_groups']})
    old = max(r['old_full_history_proxy'] for r in rows)
    new = max(r['proposed_full_history_proxy'] for r in rows)
    target = 1.15
    return {'status': 'host_source_preparation_and_conditional_budget_only',
        'shape': {'Q': 336, 'context': 1341, 'requests': 1, 'splits': 1, 'Cube_cores': 20, 'tasks': 1008},
        'pair_sources': pair_sources(), 'denominator': 'current_production_H16',
        'candidate': 'existing_H32_plus_member_pipeline', 'minimum_complete_CV_speedup': target,
        'source_work': work, 'owners_budget': rows,
        'coefficient_source': str(provenance.relative_to(ROOT)), 'coefficient_source_sha256': sha(provenance.read_bytes()),
        'coefficients': {'measured_C1_full_period': c1, 'measured_H16_C4_full_period': c4,
                         'conditional_member_pipeline_period_NOT_measured': proposed},
        'full_history_only_proxy': {'old': old, 'candidate': new, 'ratio': old / new,
            'old_critical_cores': [r['core'] for r in rows if r['old_full_history_proxy'] == old],
            'candidate_critical_cores': [r['core'] for r in rows if r['proposed_full_history_proxy'] == new]},
        'common_cost_condition': {'formula': '(A+F)/(B+F+P)>=1.15 requires F+(1.15/0.15)*P <= (A-1.15*B)/0.15',
            'ceiling_cycles_F_plus_7p6667P': (old - target * new) / (target - 1),
            'assumptions': 'same critical owner, common F, transferred stationary full256 periods; all unmeasured',
            'F': 'all partial KV rounds, task scan/followers, query loads, exact windows, current empties, normalize/Rv/output, startup/drain',
            'P': 'candidate-only costs not already included in proposed period; atomic/state/two-buffer costs are not free'},
        'real_wall_formula': 'max_i(old_full_i+F_old_i+tail_old_i) / max_i(new_full_i+F_new_i+tail_new_i)',
        'why_reopen': 'C337 old 1.148704 compared H32 against H32+pipeline and had one full mature round; C1341 retains four full mature rounds, directly measured against shipped H16.',
        'not_a_prediction': 'No measured complete C1341 kernel or gain yet; passed host source/oracle preparation never passes the device speed gate.',
        'not_included': ['native current FIA', 'prepare/rotate/store wrapper kernels', 'GDN/model/MTP', 'whole service'],
        'validation': {k: 'not_run' for k in ('CPU', 'CANN', 'CAModel', 'NPU', 'graph', 'whole_service')}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    result = analyze(); args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'conditional_full_history_proxy': result['full_history_only_proxy'],
                      'complete_gate': 'not_established'}))
