"""Package the measured H32/member pipeline without modifying its mathematics.

Archive #148/#150/#153/#154 and full startup D.4: exact source inheritance,
separate capability/ABI/workspace checks, no numerical or graph claims from
packaging. The old C4/canonical sources remain unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE='csrc/experiments/c4_member_pipeline/c4_candidate.cpp'
SOURCE_SHA='da59a923a44cdb7ac0916596cd89f802a24fc1ffef4426952402d90b5d2e7b31'
HELPER='csrc/experiments/c4_unpack32/unpack32.h'
HELPER_SHA='ea73ea49efaad4cfb43f9cdb91c8f1493fa95579e7e1750d33cac6c51c002bfc'
TARGET='csrc/kernels/attention_cv_member_pipeline_cluster4.cpp'
HELPER_TARGET='csrc/kernels/attention_striped_unpack32.h'


def digest(text):return hashlib.sha256(text.encode()).hexdigest()


def relocate(text,changes):
    result=text;edits=[]
    for before,after,count in changes:
        if result.count(before)!=count:raise ValueError(('qualified C4 source drift',before,result.count(before),count))
        result=result.replace(before,after);edits.append((before,after,count))
    restored=result
    for before,after,count in reversed(edits):
        if restored.count(after)!=count:raise ValueError('C4 inverse mapping drift')
        restored=restored.replace(after,before)
    if restored!=text:raise ValueError('C4 byte restoration failed')
    return result


def package(root=ROOT,*,check=False):
    source=(root/SOURCE).read_text();helper=(root/HELPER).read_text()
    if digest(source)!=SOURCE_SHA or digest(helper)!=HELPER_SHA:
        raise ValueError('measured C4/member/H32 mother changed; re-qualify before packaging')
    result=relocate(source,[
        ('// ISOLATED C4 MEMBER PIPELINE; not registered or enabled in production.',
         '// Explicit C4 member-pipeline candidate; source-qualified, NPU/graph gates required.',1),
        ('#include "../../kernels/attention_fast_unpack.h"','#include "attention_fast_unpack.h"',2),
        ('#include "../../kernels/attention_striped_unpack.h"','#include "attention_striped_unpack.h"',1),
        ('#include "../../kernels/oscar_common.h"','#include "oscar_common.h"',1),
        ('#include "../../kernels/attention_range_mask.h"','#include "attention_range_mask.h"',1),
        ('#include "../../include/oscar_attention_launch.h"','#include "../include/oscar_attention_launch.h"',1),
        ('#include "../../include/oscar_cv_schedule.h"','#include "../include/oscar_cv_schedule.h"',1),
        ('#include "../c4_unpack32/unpack32.h"','#include "attention_striped_unpack32.h"',1),
        ('#include "../include/oscar_attention_launch.h"',
         '#include "../include/oscar_attention_launch.h"\n#include "../include/oscar_member_pipeline_launch.h"',1),
        ('c4_member_pipeline_candidate_launch','attention_cv_member_pipeline_cluster4_launch',1),
        ('oscar_c4_member_pipeline_candidate_kernel','oscar_attention_cv_member_pipeline_cluster4_kernel',1),
    ])
    packaged_helper=relocate(helper,[
        ('// RESEARCH ONLY. Archive','// Candidate-only qualified H32 helper. Archive',1),
        ('#include "../../kernels/oscar_common.h"','#include "oscar_common.h"',1),
    ])
    for path,content in [(TARGET,result),(HELPER_TARGET,packaged_helper)]:
        if check:
            if (root/path).read_text()!=content:raise ValueError('C4 packaged source differs: '+path)
        else:(root/path).write_text(content)
    return {'status':'source_packaged_not_device_acceptance','source':SOURCE,'source_sha256':SOURCE_SHA,
        'target':TARGET,'target_sha256':digest(result),'helper':HELPER,'helper_sha256':HELPER_SHA,
        'helper_target':HELPER_TARGET,'helper_target_sha256':digest(packaged_helper),
        'inverse_full_source_bytes':True,'math_flags_task_error_order':'unchanged',
        'changes':['include relocation','distinct symbols','candidate banner','authoritative ABI header'],
        'operator':'attention_cv_member_pipeline_cluster4_out',
        'request_operator':'attention_request_window_member_pipeline_cluster4_out',
        'D256_workspace_bytes_per_core':2494464,'AIV_UB_bytes':157760,
        'evidence':'reports/c4_member_pipeline_20261010/reports/c4-pipeline-camodel337/comparison.json',
        'CPU_experiment':'qualified finite/fault cases, not new production build',
        'CANN_fresh':'pending','NPU':'pending','graph':'pending','service_performance':'pending','pushed':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'reports/c4_member_pipeline_packaging_20261011.json')
    a=p.parse_args();report=package(check=a.check)
    if not a.check:a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'check':a.check,'target':TARGET}))


if __name__=='__main__':main()
