"""Bounded phase driver; no retries, budget increases or target access.

CPU small pairs use the original complete checker. Long shape uses actual
prepare and complete CAModel oracle, not an unrecorded CPU substitution.
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.decode_finite_20261015.generate import digest
from csrc.experiments.decode_finite_20261015.run import compare,save,source_check
from csrc.experiments.decode_mdl_load.qualify_q1 import bound
EXPECTED={'normal_full':0,'second_lane_full_or_empty':0,'finite_sum_overflow':2,'finite_cancel':0,
 'cancel_intermediate_overflow':2,'visible_NaN':2,'visible_positive_Inf':2,'visible_negative_Inf':2,
 'masked_NaN_still_prechecked':2,'empty_row_NaN_not_checked':0,'precise_masked_NaN':2,
 'padding_NaN_not_checked':0,'existing_metadata_error_preserved':3,
 'score_NaN_overwrites_incoming_metadata':2,'score_NaN_then_pending_metadata_commit':3}

def cpu(a):
    a.output.mkdir(parents=True,exist_ok=False);a.exports.mkdir(parents=True,exist_ok=True)
    library=ROOT/'build/device/lib/liboscar_decode_finite.so';binding=source_check(library)
    if json.loads((ROOT/'build/cpu/source_binding.json').read_text())!=binding:raise ValueError('CPU/device source mismatch')
    proof={'family':a.family,'source_manifest_sha256':digest(HERE/'manifest.json'),
       'finite_probe_manifest_sha256':digest(HERE/'finite_probe/manifest.json'),'cases':[],
       'library_sha256':digest(library),'CPU_debug_executable_sha256':digest(ROOT/'build/cpu/oscar_decode_finite_cpu')}
    control,_=bound(ROOT,a.output,'methods',120,[str(ROOT/f'build/methods/oscar_decode_finite_{a.family}'),str(a.output/'methods')])
    raw=json.loads((a.output/'methods/report.json').read_text())
    if raw['status']!='passed' or {r['case']:r['error'] for r in raw['cases']}!=EXPECTED or any(not r['bitwise'] or not r['guards'] for r in raw['cases']):raise ValueError('actual methods failed')
    proof['actual_methods']=raw;save(a.output/'progress.json',proof)
    for suffix in ('cpu','query_metadata','long'):
        case=a.family+'_finite_'+suffix;out=a.exports/case;out.mkdir(exist_ok=False)
        command=[str(ROOT/'build/cpu/oscar_decode_finite_cpu'),a.family,str(a.fixture/case),str(out)]
        if suffix=='long':command+=['prepare-only']
        control,receipt=bound(ROOT,a.output,suffix,120,command)
        rows=[json.loads(line) for line in (a.output/f'{suffix}-logs/stdout.log').read_text().splitlines() if line.startswith('{"backend":"ascendc_cpu_debug"')]
        if len(rows)!=1:raise ValueError('complete CPU result missing')
        result=rows[0]
        if suffix=='long':
            if result['status']!='prepared_only' or result['positions']!='passed' or not result['gm_guards'] or not result['all_tasks_exported']:raise ValueError('actual prepare failed')
        elif result['status']!='passed' or any(result[k] is not True for k in ('bitwise','gm_guards','input_immutable','fault_order' if suffix=='query_metadata' else 'oracle')):raise ValueError('full CPU pair failed')
        proof['cases'].append({'case':case,'result':result,'control':str(control),'elapsed_seconds':receipt['elapsed_seconds']})
        save(out/'report.json',result);save(a.output/'progress.json',proof)
    proof.update(status='passed',long_reference='actual prepare only, full CAModel bitwise and independent oracle required',NPU='not_run')
    save(a.output/'qualified.json',proof);print(json.dumps({'family':a.family,'CPU':'passed'}),flush=True)

def camodel(a):
    prior=json.loads(a.cpu.read_text())
    if prior['status']!='passed' or prior['family']!=a.family or prior['source_manifest_sha256']!=digest(HERE/'manifest.json') or prior['finite_probe_manifest_sha256']!=digest(HERE/'finite_probe/manifest.json'):raise ValueError('current CPU/source proof required')
    a.output.mkdir(parents=True,exist_ok=False);library=ROOT/'build/device/lib/liboscar_decode_finite.so'
    if prior['library_sha256']!=digest(library):raise ValueError('CPU/device library changed')
    env=json.loads(a.env.read_text());config=env['CAMODEL_CONFIG_PATH']
    controls={}
    for arm in ('baseline','candidate'):
        controls[arm],_=bound(ROOT,a.output,arm,600,[sys.executable,str(HERE/'run.py'),'run','--family',a.family,'--arm',arm,
            '--output',str(a.output/arm),'--fixture-root',str(a.fixture),'--export-root',str(a.exports),
            '--library',str(library),'--simulator-config-dir',config],a.env)
    compare(argparse.Namespace(baseline=a.output/'baseline',candidate=a.output/'candidate',
      baseline_control=controls['baseline'],candidate_control=controls['candidate'],output=a.output/'comparison.json'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('cpu','camodel'));p.add_argument('--family',choices=('q4','q1'),required=True)
    for n in ('output','fixture','exports'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--cpu',type=Path);p.add_argument('--env',type=Path);a=p.parse_args()
    for n,v in vars(a).items():
        if isinstance(v,Path):setattr(a,n,v.resolve())
    if os.environ.get('LD_PRELOAD'):raise ValueError('supervisor must not preload simulator')
    cpu(a) if a.phase=='cpu' else camodel(a)
