"""Archive #129/#148/#154-157: same-library stable/pipeline/B1 full pairing.

Do not multiply prior-library ratios. Stable->B1 is the composed result;
pipeline->B1 is an incremental diagnostic, not another multiplicative gain.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.c4_composed_long.qualify import bound,save,digest
from csrc.experiments.c16_l1_reuse_20261015 import generate,cv_compare


def compare(a,controls):
    rows={};hashes={}
    for arm in ('stable','baseline','candidate'):
        report,host,output_hashes=cv_compare.load(a.output/arm,controls[arm],arm)
        rows[arm]=report;hashes[arm]=output_hashes
    if hashes['stable']!=hashes['baseline'] or hashes['stable']!=hashes['candidate']:
        raise ValueError('three-arm partial/LSE/status/counters differ')
    for field in ('attrs','cores','input_sha256','oracle_sha256','library_sha256','simulator_config'):
        if any(rows[arm][field]!=rows['stable'][field] for arm in ('baseline','candidate')):
            raise ValueError('same-library/fixture/config mismatch '+field)
    ticks={arm:rows[arm]['task_ticks'] for arm in rows}
    composed=ticks['stable']/ticks['candidate'];incremental=ticks['baseline']/ticks['candidate']
    return {'status':'complete_same_library_three_arm_pair','task_ticks':ticks,
        'stable_over_B1':composed,'pipeline_over_B1':incremental,
        'stable_over_pipeline':ticks['stable']/ticks['baseline'],
        'selection_gate':'passed' if composed>=1.15 else 'failed',
        'incremental_gate':'passed' if incremental>=1.15 else 'failed',
        'minimum_local_speedup':1.15,'all_output_hashes':hashes['stable'],
        'all_sources_tasks_oracle':'passed_all_three','scope':'Q672/C337/recent256; all2016 tasks,20Cube',
        'Q672_CPU':'not_established','NPU':'not_run','graph':'not_run','whole_service':'not_established'}


def run(a):
    a.output.mkdir(parents=True,exist_ok=False)
    save(a.output/'plan.json',{'arms':['stable','baseline','candidate'],'deadline_each_arm_s':900,
        'source_manifest_sha256':digest(generate.HERE/'manifest.json'),'library_sha256':digest(a.library),
        'composed_gate':'stable/B1>=1.15','incremental_gate':'pipeline/B1>=1.15',
        'model_target':'separate 32K/64K native comparison, not proven by this local gate'})
    controls={}
    for arm in ('stable','baseline','candidate'):
        controls[arm]=bound(ROOT,a.output,arm,900,[sys.executable,str(generate.HERE/'cv_replay_backend_split.py'),
            '--variant',arm,'--fixture',str(a.fixture),'--prepared',str(a.prepared),
            '--small-cpu',str(a.small_cpu),'--full-cpu-control',str(a.full_cpu_control),
            '--library',str(a.library),'--output',str(a.output/arm)],a.env)
    result=compare(a,controls);save(a.output/'comparison.json',result)
    print('C16_B1_THREE_ARM',json.dumps(result),flush=True)
    return 0 if result['selection_gate']=='passed' else 2


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('output','fixture','prepared','small-cpu','full-cpu-control','library','env'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    for key,value in vars(a).items():setattr(a,key,value.resolve())
    if os.environ.get('LD_PRELOAD'):raise ValueError('clean supervisor environment required')
    raise SystemExit(run(a))
