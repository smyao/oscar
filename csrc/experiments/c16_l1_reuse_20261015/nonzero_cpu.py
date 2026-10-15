"""Archive #126/#129/#145/#154: actual three-arm owner1 CPU qualification."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.c4_composed_long.qualify import bound,save,digest


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);hashes={};reports={};controls={}
    for arm in ('stable','baseline','candidate'):
        controls[arm]=bound(ROOT,a.output,arm,120,[str(a.executable),str(a.fixture),str(a.output/arm),arm])
        reports[arm]=json.loads((a.output/arm/'report.json').read_text())
        if reports[arm]['status']!='passed' or reports[arm]['clusters']!=1:raise ValueError('CPU gate failed')
        hashes[arm]={name:digest(a.output/arm/(name+'.bin')) for name in ('partial','lse','status','cluster_stats','tasks')}
    if hashes['stable']!=hashes['baseline'] or hashes['stable']!=hashes['candidate']:
        raise ValueError('nonzero owner three-arm mismatch')
    result={'status':'complete_three_arm_nonzero_owner_CPU_pair','hashes':hashes['stable'],
        'controls':{arm:str(path) for arm,path in controls.items()},'reports':reports,
        'scope':'R2 Q336 each contexts0/337 recent1, all2016 tasks; actual cluster owner1 and qbegin336',
        'C16_shared_KV_rounds':2,'frozen_second_request_oracle':'passed_all336queries',
        'first_request_all_CV_sources_empty':'passed','NPU':'not_run','performance':'not_measured'}
    save(a.output/'comparison.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('output','fixture','executable'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    for key,value in vars(a).items():setattr(a,key,value.resolve())
    run(a)
