"""Archive #129/#145/#148/#154: bounded complete C16 mechanism fixture.

After full Q672 timed out, use independently CPU-completed Q336/recent1/C337.
Every1008 task and source0+1 remains; this cannot certify the original shape.
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
from csrc.experiments.c16_l1_reuse_20261015 import generate


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);reports={};controls={}
    save(a.output/'plan.json',{'arms':['stable','baseline','candidate'],'deadline_each_arm_s':600,
        'source_manifest_sha256':digest(generate.HERE/'manifest.json'),'library_sha256':digest(a.library),
        'scope':f'Q336/recent1/C{a.context},1008 tasks,20Cube; complete mechanism fixture, not Q672',
        'history_KV_rounds':1 if a.context==65 else 2,
        'minimum_composed_speedup':1.15,'prior_Q672_CAModel':'900s_timeout_preserved_not_retried'})
    for arm in ('stable','baseline','candidate'):
        prepared=a.stable_cpu if arm=='stable' else a.small_cpu/(f'small{a.context}-'+arm)
        controls[arm]=bound(ROOT,a.output,arm,600,[sys.executable,str(generate.HERE/'cv_replay_small.py'),
            '--variant',arm,'--fixture',str(a.fixture),'--prepared',str(prepared),
            '--library',str(a.library),'--output',str(a.output/arm)],a.env)
        report=json.loads((a.output/arm/'report.json').read_text())
        complete=json.loads((a.output/arm/'completion.json').read_text())
        if report['status']!='passed' or report['failures'] or complete['failures']:
            raise ValueError('full arm completion/oracle failed')
        if complete['status']!='device_complete_outputs_saved':raise ValueError('missing completion')
        for name in ('partial','lse','status','cluster_stats'):
            if digest(a.output/arm/(name+'.bin'))!=report['outputs'][name]['sha256']:
                raise ValueError('saved bytes drift')
        reports[arm]=report
    for field in ('attrs','cores','input_sha256','oracle_sha256','library_sha256','simulator_config'):
        if any(reports[arm][field]!=reports['stable'][field] for arm in ('baseline','candidate')):
            raise ValueError('three-arm common input/library/config identity mismatch '+field)
    for name in ('partial','lse','status','cluster_stats'):
        if any(reports[arm]['outputs'][name]!=reports['stable']['outputs'][name] for arm in ('baseline','candidate')):
            raise ValueError('full three-arm bitwise mismatch '+name)
    ticks={arm:reports[arm]['task_ticks'] for arm in reports}
    result={'status':'complete_same_library_three_arm_pair','task_ticks':ticks,
        'stable_over_B1':ticks['stable']/ticks['candidate'],
        'pipeline_over_B1':ticks['baseline']/ticks['candidate'],
        'stable_over_pipeline':ticks['stable']/ticks['baseline'],
        'scope':f'Q336/recent1/C{a.context} full1008 tasks and allsources,20Cube; not Q672/recent256',
        'history_KV_rounds':1 if a.context==65 else 2,
        'all_partial_lse_status_counters_bitwise':True,'independent_oracle':'passed_three_arms',
        'selection_gate':'passed' if ticks['stable']/ticks['candidate']>=1.15 else 'failed',
        'Q672_CPU':'timed_out','Q672_CAModel':'timed_out','NPU':'not_run','graph':'not_run','whole_service':'not_established'}
    save(a.output/'comparison.json',result);print('C16_B1_SMALL_THREE_ARM',json.dumps(result),flush=True)
    return 0 if result['selection_gate']=='passed' else 2


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('output','fixture','stable-cpu','small-cpu','library','env'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--context',type=int,choices=(65,337),default=337)
    a=p.parse_args()
    for key,value in vars(a).items():
        if isinstance(value,Path):setattr(a,key,value.resolve())
    if os.environ.get('LD_PRELOAD'):raise ValueError('clean supervisor required')
    raise SystemExit(run(a))
