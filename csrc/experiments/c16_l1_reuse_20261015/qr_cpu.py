"""Archive #148/#154: one late-member QR fault, complete two-arm CPU parity."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.c4_composed_long.qualify import bound,save,digest


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);reports={};hashes={};controls={}
    for arm in ('baseline','candidate'):
        controls[arm]=bound(ROOT,a.output,arm,120,[str(a.executable),str(a.fixture),str(a.output/arm),arm])
        reports[arm]=json.loads((a.output/arm/'report.json').read_text())
        if reports[arm]['status']!='passed' or reports[arm]['fault_statuses']!=1:
            raise ValueError('exact one-member/one-lane fault not established')
        hashes[arm]={name:digest(a.output/arm/(name+'.bin')) for name in ('partial','lse','status','cluster_stats','tasks')}
    if hashes['baseline']!=hashes['candidate']:raise ValueError('QR isolation bitwise mismatch')
    result={'status':'complete_late_member_QR_fault_bitwise_pair','reports':reports,'hashes':hashes['baseline'],
        'controls':{arm:str(path) for arm,path in controls.items()},
        'injection':'query315/head0/dim0 QR NaN, member15 lane0 error2 only',
        'failed_closed_history_rows':64,'unaffected_original_space_oracle_rows':1952,
        'shared_KV_mutated':False,'other_members_error_status':'zero','NPU':'not_run','performance':'not_measured'}
    save(a.output/'comparison.json',result);print(json.dumps({'status':result['status']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('output','fixture','executable'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    for key,value in vars(a).items():setattr(a,key,value.resolve())
    run(a)
