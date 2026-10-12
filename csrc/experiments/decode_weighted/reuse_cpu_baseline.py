"""Archive #126/#140/#145/#155: finish a CPU comparison from saved baseline.

Same complete fixture, original120s invocation budget, real official checker,
all output bytes and independent oracle. A failed pair remains failed; the
new candidate-only evidence explicitly records its baseline provenance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.window_batch_decode.run_cpu import _signature,_save_trace_summary,validation_lock
from tools.phase import atomic_json,run_phase

def run(args):
    cases=json.loads(args.cases.read_text())['cases']
    choices=[row for row in cases if row['name']==args.case]
    if len(choices)!=1:raise ValueError('unknown case')
    case=choices[0];fixture=(args.cases.parent/case['name']).resolve()
    reference_report=json.loads(args.reference_report.read_text())
    rows=[row for row in reference_report['cases'] if row['name']==case['name']]
    if reference_report.get('backend')!='official_ascendc_cpu_debug' or len(rows)!=1:
        raise ValueError('missing official CPU baseline producer')
    previous=rows[0]
    digest=_signature(fixture)
    if previous['golden_sha256']!=digest:raise ValueError('stale reference fixture')
    log=Path(previous['log']).read_text()
    marker='CPU_BASELINE_COMPLETE mode='+case['mode']
    if log.splitlines().count(marker)!=1:raise ValueError('baseline never completed exactly once')
    prefix=log.split(marker)[0]
    if '[ERROR]' in prefix or 'error happened!' in prefix:
        raise ValueError('baseline CPU checker reported errors')
    reference=(args.reference_dir/case['name']).resolve()
    hashes={name:hashlib.sha256((reference/name).read_bytes()).hexdigest()
            for name in ('tasks.bin','attrs.txt','partial.bin','lse.bin','status.bin')}
    export=(args.export_dir/case['name']).resolve()
    if export==reference:raise ValueError('cannot overwrite the witnessed baseline')
    export.mkdir(parents=True,exist_ok=False)
    report={'status':'running','backend':'official_ascendc_cpu_debug',
            'case':case['name'],'golden_sha256':digest,'baseline_execution':'reused_complete_output',
            'baseline_report':str(args.reference_report),'baseline_artifacts':str(reference),
            'baseline_sha256':hashes,'npu':'not_run','device_speed':'not_established'}
    atomic_json(args.output,report)
    with tempfile.TemporaryDirectory(prefix='oscar-weighted-reuse-',dir='/dev/shm') as work:
        try:
            result=run_phase('candidate-'+case['name'],[str(args.executable.resolve()),case['mode'],
                str(fixture),str(export),'candidate-from',str(reference)],cwd=Path(work),
                log_dir=args.log_dir,timeout=120,grace=5,heartbeat=30,
                env={**os.environ,'OSCAR_TERMINAL_LOG_MODE':'compact'})
        finally:
            report['sdk_trace_summary']=_save_trace_summary(Path(work),args.log_dir/'trace')
    text=Path(result.log).read_text()
    markers=[json.loads(line) for line in text.splitlines()
             if line.startswith('{"backend":"ascendc_cpu_debug"')]
    evidence=markers[0] if len(markers)==1 else {}
    passed=(result.returncode==0 and not result.timed_out and result.cleanup_complete
        and evidence.get('status')=='passed' and evidence.get('bitwise') is True
        and evidence.get('cores')==case['cores'] and evidence.get('input_immutable') is True
        and evidence.get('gm_guards') is True and evidence.get('fault_order') is case['expected_error']
        and evidence.get('oracle') is (not case['expected_error'])
        and text.splitlines().count('CPU_BASELINE_REUSED mode='+case['mode'])==1
        and '[ERROR]' not in text and 'error happened!' not in text
        and _signature(fixture)==digest
        and hashes=={name:hashlib.sha256((reference/name).read_bytes()).hexdigest() for name in hashes})
    report.update(status='passed' if passed else 'failed',returncode=result.returncode,
                  timed_out=result.timed_out,log=result.log,evidence=evidence)
    atomic_json(args.output,report)
    return 0 if passed else 1

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('executable','cases','reference-report','reference-dir','export-dir','output','log-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--case',required=True)
    args=p.parse_args()
    with validation_lock():raise SystemExit(run(args))
