"""Archive #126/#129/#145/#148/#154-157 and complete startup D.4.

Relocate qualified C16 B1 plus final-Rv residency bytes without SDK snapshots.
The research generator separately requires its matched official SDK snapshot.
This packaging step does not establish NPU, graph, or whole-model performance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from tools.package_c4_member_pipeline import relocate

ROOT=Path(__file__).resolve().parents[1]
SOURCE='csrc/experiments/c16_rv_reuse_20261015/c16_candidate.cpp'
SOURCE_SHA='9603817dddfba0e6357b42756a9956880137348ce8076c61aff58884acb6210b'
MOTHER='csrc/kernels/attention_cv_member_pipeline_cluster16.cpp'
MOTHER_SHA='e2be8877e0a2b877bb8a9532c3940a5c6f2aa6d134cd834922a50e56477580af'
STABLE='csrc/kernels/attention_cv_window_range_cluster16.cpp'
STABLE_SHA='708d83d87d436550bd82f545d64af19cd5c5af34acc92b47fbe6375a0496f335'
TARGET='csrc/kernels/attention_cv_l1_reuse_cluster16.cpp'
HEADER_TARGET='csrc/include/oscar_c16_l1_reuse_launch.h'
HEADER='''// Archive #126/#129/#145/#148/#154-157, D.4: fixed C16 single-B1 ABI.
// Sixteen score/P slots, double bounded KV, one256KiB AIC B1; no history-sized tensor.
#pragma once
#include "oscar_attention_launch.h"
namespace oscar_ascend {
constexpr int64_t attention_c16_l1_reuse_workspace_per_core(int64_t dim) {
  return attention_cluster16_workspace_per_core(dim)
      + (2*kAttentionKvRows*dim+15*kAttentionQueryRows*kAttentionKvRows)*4;
}
static_assert(attention_c16_l1_reuse_workspace_per_core(256)==7487488,
    "C16 B1 D256 workspace must include all16 score/P slots");
void attention_cv_l1_reuse_cluster16_launch(void* stream, void* query, void* query_rot,
    void* current_key, void* current_value, void* rotation_v, void* raw,
    void* block_table, void* window_key, void* window_value, void* window_tags,
    void* tasks, void* partial, void* lse, void* status, void* workspace,
    void* cluster_stats, int64_t tokens, int64_t query_heads, int64_t kv_heads,
    int64_t dim, int64_t requests, int64_t table_columns, int64_t task_count,
    int64_t block_tokens, int64_t physical_blocks, int64_t ssm_offset,
    int64_t page_stride, int64_t window_stride, int64_t tag_stride,
    int64_t sink, int64_t recent, int64_t speculative, int64_t splits,
    float scale, uint32_t cores);
}
'''


def package(root=ROOT,*,check=False):
    for path,expected in ((SOURCE,SOURCE_SHA),(MOTHER,MOTHER_SHA),(STABLE,STABLE_SHA)):
        if hashlib.sha256((root/path).read_bytes()).hexdigest()!=expected:
            raise ValueError('fixed C16 B1 source identity changed: '+path)
    result=relocate((root/SOURCE).read_text(),[
        *[(f'#include "../../kernels/{name}"',f'#include "{name}"',2 if name=='attention_fast_unpack.h' else 1)
          for name in ('attention_fast_unpack.h','attention_striped_unpack.h','attention_striped_unpack32.h',
                       'attention_range_mask.h','oscar_common.h')],
        ('"../../include/oscar_attention_launch.h"','"../include/oscar_attention_launch.h"',1),
        ('"../../include/oscar_cv_schedule.h"','"../include/oscar_cv_schedule.h"',1),
        ('"../../include/oscar_c16_member_pipeline_launch.h"','"../include/oscar_c16_l1_reuse_launch.h"',1),
        ('c16_rv_reuse_candidate_launch','attention_cv_l1_reuse_cluster16_launch',1),
        ('oscar_c16_rv_reuse_candidate_kernel','oscar_attention_cv_l1_reuse_cluster16_kernel',1),
    ])
    for path,content in ((TARGET,result),(HEADER_TARGET,HEADER)):
        if check:
            if (root/path).read_text()!=content:raise ValueError('packaged C16 B1 source differs: '+path)
        else:(root/path).write_text(content)
    return {'status':'source_packaged_not_device_acceptance','source':SOURCE,'source_sha256':SOURCE_SHA,
        'mother':MOTHER,'mother_sha256':MOTHER_SHA,'stable':STABLE,'stable_sha256':STABLE_SHA,
        'target':TARGET,'target_sha256':hashlib.sha256(result.encode()).hexdigest(),
        'header':HEADER_TARGET,'header_sha256':hashlib.sha256(HEADER.encode()).hexdigest(),
        'inverse_full_source_bytes':True,'operator':'attention_cv_l1_reuse_cluster16_out',
        'D256_workspace_per_core':7487488,'D256_extra_vs_stable_per_core':2490368,
        'AIV_UB_bytes':157760,'AIC_external_B1_bytes':262144,
        'math_flags_task_error_order':'research bytes preserved',
        'NPU':'not_run','graph':'not_run','whole_service':'not_established'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true')
    p.add_argument('--output',type=Path);a=p.parse_args();result=package(check=a.check)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
