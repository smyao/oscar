"""Matched complete current-striped and window/range CPU arms, 120s each.

Reference outputs are persisted before the candidate starts. The same complete
input/shape, grid and all-source tasks are used; no whole-service timing claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from tools.phase import atomic_json,run_phase
from csrc.experiments.window_batch_decode.run_cpu import _save_trace_summary

CASES=(('base','base_d256_q3q33'),('balanced','balanced_d256_q4_n128_s3'),
       ('cluster4','c4_d256_q168_c65_active'),('cluster16','c16_d256_q336_c65_active'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--fixtures',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--log-dir',type=Path,required=True)
    parser.add_argument('--reference-dir',type=Path,required=True)
    args=parser.parse_args()
    report={'status':'running','baseline':'current_production_striped','cases':[],'npu':'not_run'}
    atomic_json(args.output,report)
    for mode,name in CASES:
        fixture=args.fixtures/name;reference=args.reference_dir/name
        reference.mkdir(parents=True,exist_ok=True)
        digest=hashlib.sha256()
        for file in sorted(fixture.iterdir()):
            if file.is_file():digest.update(file.name.encode()+b'\0'+file.read_bytes())
        row={'name':name,'mode':mode,'input_tree_sha256':digest.hexdigest(),'arms':[]}
        for arm in ('reference','candidate'):
            phase=f'{mode}-{arm}'
            with tempfile.TemporaryDirectory(prefix='oscar-prefill-window-',dir='/dev/shm') as work:
                result=run_phase(phase,[str(args.executable.resolve()),mode,arm,
                    str(fixture.resolve()),str(reference.resolve())],cwd=Path(work),
                    log_dir=args.log_dir,timeout=120,grace=5,heartbeat=30,
                    env={**os.environ,'OSCAR_TERMINAL_LOG_MODE':'compact'})
                _save_trace_summary(Path(work),args.log_dir/(phase+'-trace'))
            log=Path(result.log).read_text()
            markers=[json.loads(line) for line in log.splitlines() if line.startswith('{"backend":"ascendc_cpu_debug"')]
            ok=(result.returncode==0 and not result.timed_out and result.cleanup_complete
                and len(markers)==1 and markers[0].get('status')=='passed'
                and markers[0].get('mode')==mode and markers[0].get('phase')==arm
                and '[ERROR]' not in log and 'error happened!' not in log)
            row['arms'].append({'arm':arm,'status':'passed' if ok else 'failed',
                               'returncode':result.returncode,'timed_out':result.timed_out,
                               'evidence':markers,'log':result.log})
            if not ok:
                report['cases'].append(row);report['status']='failed';atomic_json(args.output,report)
                return 1
        row['status']='passed';report['cases'].append(row);atomic_json(args.output,report)
    report['status']='passed';atomic_json(args.output,report);return 0


if __name__=='__main__':raise SystemExit(main())
