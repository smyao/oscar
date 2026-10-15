"""Archive #94/#126/#129/#145/#154-157: bounded C16 qualification, no retries.

Full Q672 CPU timeout is retained; small CPU and full CAModel are separate.
Only local VM simulator environment is supported, never a hardware launch.
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


def cpu(a):
    a.output.mkdir(parents=True,exist_ok=False);results={}
    cases=('bad80','bad96','bad336') if a.mode=='faults' else ('small65','small337')
    save(a.output/'plan.json',{'deadline_each_arm_s':120,'full_CPU672':'timed_out_not_retried',
        'cases':cases,'source':json.loads((generate.HERE/'manifest.json').read_text()),
        'executable_sha256':digest(a.executable)})
    for case in cases:
        controls={};reports={};hashes={}
        for arm in ('baseline','candidate'):
            label=case+'-'+arm
            controls[arm]=bound(ROOT,a.output,label,120,[str(a.executable),str(a.fixtures/case),str(a.output/label),arm])
            report=json.loads((a.output/label/'report.json').read_text())
            oracle='invalid_metadata_status' if a.mode=='faults' else 'numeric'
            if report['status']!='passed' or report['clusters']!=1 or report['oracle']!=oracle:
                raise ValueError('complete small CPU oracle/cluster required')
            reports[arm]=report
            hashes[arm]={name:digest(a.output/label/(name+'.bin')) for name in ('partial','lse','status','cluster_stats','tasks')}
        if hashes['baseline']!=hashes['candidate']:raise ValueError('small CPU full bitwise mismatch')
        results[case]={'controls':{arm:str(path) for arm,path in controls.items()},
            'reports':reports,'equal_hashes':hashes['baseline']}
        print('C16_CPU_CASE',case,'passed',flush=True)
    save(a.output/'comparison.json',{'status':'complete_small_CPU_pair','tokens':336,'recent':1,
        'cases':results,'manifest_sha256':digest(generate.HERE/'manifest.json'),
        'Q672_CPU':'not_established','NPU':'not_run','graph':'not_run'})


def camodel(a):
    a.output.mkdir(parents=True,exist_ok=False);controls={}
    for arm in ('baseline','candidate'):
        controls[arm]=bound(ROOT,a.output,arm,900,[sys.executable,str(generate.HERE/'cv_replay_backend_split.py'),
            '--variant',arm,'--fixture',str(a.fixture),'--prepared',str(a.prepared),
            '--small-cpu',str(a.small_cpu),'--full-cpu-control',str(a.full_cpu_control),
            '--library',str(a.library),'--output',str(a.output/arm)],a.env)
    result=cv_compare.compare(a.output/'baseline',a.output/'candidate',controls['baseline'],controls['candidate'])
    result.update(minimum_complete_local_speedup=1.15,
        performance_gate='passed' if result['old_over_new']>=1.15 else 'failed',
        Q672_CPU='not_established',NPU='not_run',graph='not_run',whole_service='not_established')
    save(a.output/'comparison.json',result);print('C16_CAMODEL_PAIR',json.dumps(result),flush=True)
    return 0 if result['performance_gate']=='passed' else 2


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
    for mode in ('cpu','faults'):
        c=sub.add_parser(mode)
        for name in ('output','executable','fixtures'):c.add_argument('--'+name,type=Path,required=True)
    c=sub.add_parser('camodel')
    for name in ('output','fixture','prepared','small-cpu','full-cpu-control','library','env'):
        c.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    for key,value in vars(a).items():
        if isinstance(value,Path):setattr(a,key,value.resolve())
    if os.environ.get('LD_PRELOAD'):raise ValueError('supervisor must not preload simulator')
    raise SystemExit((cpu(a) if a.mode in ('cpu','faults') else camodel(a)) or 0)
