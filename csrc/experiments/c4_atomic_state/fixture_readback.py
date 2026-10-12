"""Archive #126/#145/#154, D.4: revision2 adds cross-core visibility evidence.

Same input bytes as v1, distinct source manifest. No compiler or kernel run.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.c4_atomic_state import fixture as base
from csrc.experiments.c4_atomic_state import generate_readback_v2 as revision

CASES=base.CASES


def check_source():
    original=base.check_source();revision.check()
    return {**original,'revision':2,'baseline_kernel_sha256':original['kernel_sha256'],
            'kernel_sha256':base.sha((revision.HERE/'kernel_readback.cpp').read_bytes())}


def export(output):
    source=check_source();report=base.export(output)
    report.update(revision=2,source=source,
                  scope='FP32 final-add and immediate AIV MTE2 visibility after PIPE_FIX notification')
    for row in report['cases']:
        row.update(revision=2,gm_pointer_count=8,
                   readback_acceptance='all128 rows bitwise equal to both atomic and reference; padding separately counted')
        (output/row['case']/'fixture.json').write_text(json.dumps(row,indent=2)+'\n')
    (output/'cases.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-source',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.check_source:print(json.dumps(check_source(),indent=2))
    elif args.output:print(json.dumps({'status':'revision2_inputs_exported_not_executed','cases':len(export(args.output)['cases'])}))
    else:parser.error('choose --check-source or --output NEW_DIRECTORY')
