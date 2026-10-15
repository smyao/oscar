"""Archive #94/#126/#145/#148/#154: complete fixed-bound Rv pairs, no retry.

CPU multi-KV and per-member QR isolation precede the C65 CAModel pair.
Only the official simulator preload is accepted for CAModel execution.
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
from csrc.experiments.c16_rv_reuse_20261015 import generate


def run_cpu(a):
    a.output.mkdir(parents=True,exist_ok=False);reports={};hashes={};controls={}
    save(a.output/'plan.json',{'case':a.case,'deadline_each_arm_s':120,'source_manifest_sha256':digest(generate.HERE/'manifest.json'),
        'executable_sha256':digest(a.executable),'production_changed':False})
    for arm in ('baseline','candidate'):
        controls[arm]=bound(ROOT,a.output,arm,120,[str(a.executable),str(a.fixture),str(a.output/arm),arm])
        reports[arm]=json.loads((a.output/arm/'report.json').read_text())
        if reports[arm]['status']!='passed' or reports[arm]['clusters']!=1:raise ValueError('full CPU completion/oracle required')
        if a.case=='qr' and reports[arm]['fault_statuses']!=1:raise ValueError('exact QR member/lane isolation required')
        if a.case=='numeric' and (reports[arm]['context']!=337 or reports[arm]['oracle']!='numeric'):raise ValueError('declared two-KV numeric gate required')
        hashes[arm]={name:digest(a.output/arm/(name+'.bin')) for name in ('partial','lse','status','cluster_stats','tasks')}
    if hashes['baseline']!=hashes['candidate']:raise ValueError('full CPU pair bitwise mismatch')
    result={'status':'complete_'+a.case+'_CPU_pair','reports':reports,'equal_hashes':hashes['baseline'],
        'controls':{arm:str(p) for arm,p in controls.items()},'scope':'Q336/recent1/C337,all1008tasks,16member cluster,twohistoryKV rounds',
        'source_manifest_sha256':digest(generate.HERE/'manifest.json'),'NPU':'not_run','performance':'not_measured'}
    save(a.output/'comparison.json',result);print(json.dumps({'status':result['status']}),flush=True)


def run_camodel(a):
    for path,status in ((a.cpu_numeric,'complete_numeric_CPU_pair'),(a.cpu_qr,'complete_qr_CPU_pair')):
        if json.loads(path.read_text())['status']!=status:raise ValueError('prior complete CPU gates required')
    env=json.loads(a.env.read_text())
    if env.get('LD_PRELOAD')!='/usr/local/Ascend/cann-9.1.0/aarch64-linux/simulator/dav_2201/lib/libruntime_camodel.so':raise ValueError('simulator only')
    a.output.mkdir(parents=True,exist_ok=False);reports={};controls={}
    save(a.output/'plan.json',{'arms':['baseline','candidate'],'deadline_each_arm_s':600,'source_manifest_sha256':digest(generate.HERE/'manifest.json'),
        'library_sha256':digest(a.library),'scope':'Q336/recent1/C65 all1008tasks,one16-member mature cluster,onehistoryKV,20AIC40AIV',
        'CPU_prepared_reuse':'qualified unchanged B1 C65 real CPU prepare; new kernels qualified separately on C337 and QR cases',
        'CPU_numeric':str(a.cpu_numeric),'CPU_QR':str(a.cpu_qr),'selection_rule':'report actual positive or negative complete incremental contribution; user waived15percent minimum',
        'NPU':'not_run','whole_model':'not_established'})
    for arm in ('baseline','candidate'):
        controls[arm]=bound(ROOT,a.output,arm,600,[sys.executable,str(generate.HERE/'cv_replay_small.py'),
            '--variant',arm,'--fixture',str(a.fixture),'--prepared',str(a.prepared),'--library',str(a.library),
            '--output',str(a.output/arm)],a.env)
        r=json.loads((a.output/arm/'report.json').read_text());c=json.loads((a.output/arm/'completion.json').read_text())
        if r['status']!='passed' or r['failures'] or c['failures'] or c['status']!='device_complete_outputs_saved':raise ValueError('full numeric/profile completion required')
        for name in ('partial','lse','status','cluster_stats'):
            if digest(a.output/arm/(name+'.bin'))!=r['outputs'][name]['sha256']:raise ValueError('saved output drift')
        reports[arm]=r
    old,new=reports['baseline'],reports['candidate']
    for field in ('attrs','cores','input_sha256','oracle_sha256','library_sha256','simulator_config'):
        if old[field]!=new[field]:raise ValueError('same pair identity '+field)
    for name in ('partial','lse','status','cluster_stats'):
        if old['outputs'][name]!=new['outputs'][name]:raise ValueError('full output bitwise mismatch '+name)
    result={'status':'complete_same_library_B1_Rv_pair','task_ticks':{a:r['task_ticks'] for a,r in reports.items()},
        'B1_over_Rv':old['task_ticks']/new['task_ticks'],'net_positive':new['task_ticks']<old['task_ticks'],
        'all_partial_lse_status_counters_bitwise':True,'independent_oracle':'passed_both_arms','profiles_each':{'task':1,'AIC':20,'AIV':40},
        'scope':'Q336/recent1/C65 singlehistoryKV CV kernel; offline merged oracle, device merge/copy not timed',
        'additional_workspace_UB_L1_bytes':0,'NPU':'not_run','graph':'not_run','32K_64K_native_gt30pct':'not_established'}
    save(a.output/'comparison.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='mode',required=True)
    c=s.add_parser('cpu');c.add_argument('--case',choices=('numeric','qr'),required=True)
    for n in ('output','fixture','executable'):c.add_argument('--'+n,type=Path,required=True)
    c=s.add_parser('camodel')
    for n in ('output','fixture','prepared','library','env','cpu-numeric','cpu-qr'):c.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    for key,value in vars(a).items():
        if isinstance(value,Path):setattr(a,key,value.resolve())
    if os.environ.get('LD_PRELOAD'):raise ValueError('supervisor must not preload simulator')
    if a.mode=='cpu':run_cpu(a)
    else:run_camodel(a)
