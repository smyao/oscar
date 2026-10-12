"""Archive #126/#129/#145/#148/#153/#154-156 and startup D.4.

Package the qualified MDL q4 source by reversible symbol/include relocation;
all mathematical and synchronization methods stay byte-identical. Separate
registration keeps normal q4 and q1 available for paired device gates.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from tools.package_c4_member_pipeline import relocate

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'csrc/experiments/decode_mdl_load/attention_candidate.cpp'
SOURCE_SHA = 'f2047dde10aa464202c23c8a8d39a9806c712ac8eb090ff752d7346d1b723151'
BASELINE = 'csrc/kernels/attention_cv_decode_bundle.cpp'
BASELINE_SHA = 'd8171f0403cf176087e1c76c64e454e6fb1e74fd79252aab3b59e848cca19401'
TARGET = 'csrc/kernels/attention_cv_mdl_decode.cpp'
OPERATOR = 'attention_cv_mdl_decode_out'


def package(root=ROOT, *, check=False):
    source = (root/SOURCE).read_text()
    for path, expected in ((SOURCE, SOURCE_SHA), (BASELINE, BASELINE_SHA)):
        if hashlib.sha256((root/path).read_bytes()).hexdigest() != expected:
            raise ValueError('qualified MDL source drift: '+path)
    changes = [
        ('// GENERATED RESEARCH ONLY.', '// Qualified MDL q4 candidate; NPU/graph gate pending.', 1),
        *[(f'#include "../../kernels/{h}"', f'#include "{h}"', 1)
          for h in ('oscar_common.h', 'attention_striped_unpack_padded.h', 'attention_range_mask.h')],
        *[(f'#include "../../include/{h}"', f'#include "../include/{h}"', 1)
          for h in ('oscar_attention_launch.h', 'oscar_cv_schedule.h')],
        ('#include "../include/oscar_attention_launch.h"',
         '#include "../include/oscar_mdl_decode_launch.h"', 1),
        ('oscar_mdl_candidate_q4_kernel', 'oscar_attention_cv_mdl_decode_kernel', 2),
        ('mdl_candidate_q4_launch', 'attention_cv_mdl_decode_launch', 1),
        # Preserve the q1 body as a private unused symbol, never a registered
        # operator or production route: no unmeasured q1 promotion.
        ('oscar_mdl_candidate_q1_kernel', 'oscar_mdl_unregistered_q1_kernel', 2),
        ('mdl_candidate_q1_launch', 'mdl_unregistered_q1_launch', 1),
    ]
    result = relocate(source, changes)
    if check:
        if (root/TARGET).read_text() != result:
            raise ValueError('packaged MDL source differs')
    else:
        (root/TARGET).write_text(result)
    return dict(status='source_packaged_not_device_acceptance', source=SOURCE,
                source_sha256=SOURCE_SHA, baseline=BASELINE, baseline_sha256=BASELINE_SHA,
                target=TARGET, target_sha256=hashlib.sha256(result.encode()).hexdigest(),
                inverse_full_source_bytes=True, operator=OPERATOR, q1_production='unchanged',
                workspace_bytes_per_core_D256=1146880, explicit_UB_bytes_per_AIV_D256=158656,
                L1_bytes=395264, CANN_fresh='pending', NPU='pending', graph='pending',
                service_performance='pending', pushed=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/mdl_decode_packaging_20261012.json')
    args = parser.parse_args()
    report = package(check=args.check)
    if not args.check:
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status':report['status'], 'check':args.check, 'target':TARGET}))

if __name__ == '__main__':
    main()
