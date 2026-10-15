"""Archive #126/#145/#154: new B1 metadata faults against qualified pipeline gold.

The prior pipeline valid bytes/source/control are checked, not rerun. CPU
elapsed is not a speed metric. QR isolation is a separate fresh two-arm test.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.c4_composed_long.qualify import bound,save,digest
from csrc.experiments.c16_member_pipeline import generate as old
from csrc.experiments.c16_l1_reuse_20261015 import generate


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);cases={}
    prior=json.loads((a.qualified/'comparison.json').read_text())
    if prior['manifest_sha256']!=digest(old.HERE/'manifest.json'):raise ValueError('qualified pipeline source drift')
    if generate.sha(old.generate(True,production=True)[0])!=generate.sha(generate.SOURCE.read_bytes()):
        raise ValueError('B1 denominator is not qualified pipeline')
    for case in ('bad80','bad96','bad336'):
        control=json.loads((a.qualified/(case+'-candidate-control.json')).read_text())
        if control['child_returncode']!=0 or control['timed_out'] or not control['cleanup_complete']:
            raise ValueError('old pipeline arm incomplete')
        label=case+'-candidate'
        receipt=bound(ROOT,a.output,label,120,[str(a.executable),str(a.fixtures/case),str(a.output/label),'candidate'])
        report=json.loads((a.output/label/'report.json').read_text())
        if report['status']!='passed' or report['oracle']!='invalid_metadata_status' or report['clusters']!=1:
            raise ValueError('B1 exact fault/lane failed')
        hashes={name:digest(a.output/label/(name+'.bin')) for name in ('partial','lse','status','cluster_stats','tasks')}
        if hashes!=prior['cases'][case]['equal_hashes']:raise ValueError('B1 differs from qualified pipeline fault bytes')
        cases[case]={'report':report,'hashes':hashes,'control':str(receipt),
            'denominator_control':str(a.qualified/(case+'-candidate-control.json'))}
    result={'status':'B1_all_three_metadata_fault_domains_bitwise_qualified','cases':cases,
        'source_sha256':generate.sha(generate.SOURCE.read_bytes()),'NPU':'not_run','performance':'not_measured'}
    save(a.output/'comparison.json',result);print(json.dumps({'status':result['status']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('output','qualified','executable','fixtures'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    for key,value in vars(a).items():setattr(a,key,value.resolve())
    run(a)
