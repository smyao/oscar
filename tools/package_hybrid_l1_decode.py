"""Archive #126/#129/#145/#148/#153/#154-157 and startup D.4.

Relocate the complete qualified hybrid-L1 q4 source without editing arithmetic,
flags, task ownership or error order. Unqualified q1 entrypoints are excluded;
production q1 retains the separately qualified MDL operator. Rv is losslessly
packed once at initialization, never per request or from historical KV.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from tools.package_c4_member_pipeline import relocate

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'csrc/experiments/fp32_dataflow_20261013/attention_candidate.cpp'
SOURCE_SHA = '78a21cd466b1eff15991c1b6a1095402dfab2ed4c919b6a58536cad6ed2c56b1'
HELPER = 'csrc/experiments/decode_direct_nz/direct_nz_unpack.h'
HELPER_SHA = '46a5d8086f767e608cde3517826c0bfadadecceb1f6971cadc2d2dfda45a9508'
BASELINE = 'csrc/experiments/decode_mdl_load/attention_candidate.cpp'
BASELINE_SHA = 'f2047dde10aa464202c23c8a8d39a9806c712ac8eb090ff752d7346d1b723151'
TARGET = 'csrc/kernels/attention_cv_hybrid_l1_decode.cpp'
HELPER_TARGET = 'csrc/kernels/attention_hybrid_l1_unpack.h'
OPERATOR = 'attention_cv_hybrid_l1_decode_out'
EVIDENCE = 'reports/fp32_dataflow_20261013/completed/oscar-hybrid-l1-long-resume-20261013/comparison.json'


def package(root=ROOT, *, check=False):
    for path, expected in ((SOURCE, SOURCE_SHA), (HELPER, HELPER_SHA), (BASELINE, BASELINE_SHA)):
        if hashlib.sha256((root/path).read_bytes()).hexdigest() != expected:
            raise ValueError('qualified hybrid L1 source drift: '+path)
    source = (root/SOURCE).read_text()
    result = relocate(source, [
        ('// GENERATED ISOLATED HYBRID L1 B RESEARCH.',
         '// Qualified hybrid L1 q4 candidate; NPU/graph gates pending.', 1),
        *[(f'#include "../../kernels/{h}"', f'#include "{h}"', 1)
          for h in ('oscar_common.h', 'attention_range_mask.h')],
        ('#include "../decode_direct_nz/direct_nz_unpack.h"',
         '#include "attention_hybrid_l1_unpack.h"', 1),
        ('#include "../../include/oscar_attention_launch.h"',
         '#include "../include/oscar_hybrid_l1_decode_launch.h"', 1),
        ('#include "../../include/oscar_cv_schedule.h"',
         '#include "../include/oscar_cv_schedule.h"', 1),
        ('oscar_hybrid_l1_candidate_q4_kernel', 'oscar_attention_cv_hybrid_l1_decode_kernel', 2),
        ('hybrid_l1_candidate_q4_launch', 'attention_cv_hybrid_l1_decode_launch', 1),
        ('extern "C" __global__ __aicore__ void oscar_hybrid_l1_candidate_q1_kernel(\n'
         '    OSCAR_CV_ARGUMENTS) { OSCAR_CV_RUN(1); }',
         '// q1 entrypoint omitted: production retains the separately qualified MDL q1.', 1),
        ('OSCAR_STRIPED_LAUNCH(hybrid_l1_candidate_q1_launch,\n'
         '    oscar_hybrid_l1_candidate_q1_kernel)',
         '// q1 launch omitted: hybrid L1 qualification covers q4 only.', 1),
    ])
    helper = relocate((root/HELPER).read_text(), [
        ('// ISOLATED DIRECT-NZ SOURCE0 HELPER.', '// Qualified hybrid L1 history helper.', 1),
        ('#include "../../kernels/oscar_common.h"', '#include "oscar_common.h"', 1),
    ])
    for path, content in ((TARGET, result), (HELPER_TARGET, helper)):
        if check:
            if (root/path).read_text() != content:
                raise ValueError('packaged hybrid L1 source differs: '+path)
        else:
            (root/path).write_text(content)
    return dict(status='source_packaged_not_device_acceptance', source=SOURCE,
        source_sha256=SOURCE_SHA, helper=HELPER, helper_sha256=HELPER_SHA,
        baseline=BASELINE, baseline_sha256=BASELINE_SHA,
        target=TARGET, target_sha256=hashlib.sha256(result.encode()).hexdigest(),
        helper_target=HELPER_TARGET, helper_target_sha256=hashlib.sha256(helper.encode()).hexdigest(),
        inverse_full_source_bytes=True, mathematical_methods_flags_tasks_errors='unchanged',
        omitted_entrypoints='unqualified q1 kernel and launch; q4 template methods unchanged',
        operator=OPERATOR, workspace_bytes_per_core_D256=1146880,
        explicit_UB_bytes_per_AIV_D256=171680, external_B1_bytes=262144,
        SDK_L1_bytes=133120, total_L1_bytes=395264,
        Rv_NZ_extra_bytes_per_layer=262144,
        Rv_NZ_index='(column // 8) * 256 * 8 + row * 8 + column % 8',
        evidence=EVIDENCE, qualified_research_CANN='completed_same_library_pair',
        production_CANN_fresh='pending', NPU='pending', graph='pending',
        model_quality='not_established', service_performance='not_established', pushed=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--output', type=Path,
                        default=ROOT/'reports/hybrid_l1_decode_packaging_20261015.json')
    args = parser.parse_args()
    report = package(check=args.check)
    if not args.check:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status':report['status'], 'check':args.check, 'target':TARGET}))


if __name__ == '__main__':
    main()
