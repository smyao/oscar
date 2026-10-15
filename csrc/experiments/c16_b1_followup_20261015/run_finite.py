"""Archive #94/#134/#140/#145/#148/#154: bounded actual Softmax CPU proof.

Keep the raw report/control intact; add separate exact-source qualification.
"""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.c4_composed_long.qualify import bound,save,digest
from csrc.experiments.c16_b1_followup_20261015.finite_probe import generate

EXPECTED={'normal_full':0,'finite_sum_overflow':2,'finite_cancel':0,
    'cancel_intermediate_overflow':2,'visible_NaN':2,'visible_positive_Inf':2,
    'visible_negative_Inf':2,'masked_NaN_still_prechecked':2,'empty_row_NaN_not_checked':0,
    'precise_masked_NaN':2,'padding_NaN_not_checked':0,'existing_metadata_error_preserved':3}


def run(a):
    a.output.mkdir(parents=True,exist_ok=False)
    for arm in ('baseline','candidate'):
        source,_=generate(arm)
        if (HERE/'finite_probe'/f'{arm}.cpp').read_text()!=source:raise ValueError('actual method extraction drift')
    source_hash=digest(HERE/'manifest.json');probe_hash=digest(HERE/'finite_probe/manifest.json')
    save(a.output/'plan.json',{'deadline_s':120,'expected_cases':EXPECTED,
        'source_manifest_sha256':source_hash,'finite_probe_manifest_sha256':probe_hash,
        'executable_sha256':digest(a.executable),'full_C320_pair':'separate mandatory gate'})
    control=bound(ROOT,a.output,'finite',120,[str(a.executable),str(a.output/'raw')])
    raw=json.loads((a.output/'raw/report.json').read_text())
    if raw['status']!='passed' or len(raw['cases'])!=len(EXPECTED):raise ValueError('complete special-value cases required')
    got={row['case']:row['error'] for row in raw['cases']}
    if got!=EXPECTED or any(row['bitwise'] is not True or row['guards'] is not True for row in raw['cases']):
        raise ValueError('finite/error/bitwise gate mismatch')
    result={**raw,'failures':[],'control':str(control),'backend':'official tikicpulib CPU-debug',
        'source_manifest_sha256':source_hash,'finite_probe_manifest_sha256':probe_hash,
        'executable_sha256':digest(a.executable),'scope':'isolated exact Softmax method, not full C320/Cube/NPU performance'}
    save(a.output/'qualified.json',result);print(json.dumps({'status':result['status'],'cases':len(raw['cases']),'full_CV':'not_implied'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--executable',type=Path,required=True);a=p.parse_args()
    a.output=a.output.resolve();a.executable=a.executable.resolve()
    if os.environ.get('LD_PRELOAD'):raise ValueError('CPU supervisor must not preload simulator')
    run(a)
