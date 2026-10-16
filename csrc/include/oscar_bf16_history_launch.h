// SPDX-License-Identifier: Apache-2.0
// Archive G4/#81-83/#126/#145/#148/#154/#157; full startup D.4 reviewed.
// Historical ABI names now cover BF16 matrix operands for every source.
// User default selection is not numerical/device/graph acceptance.
// D.4: fixed-tile FIA, no 6.5s whole-history restore, separate typed arenas;
// bounded extra capacity and casts do not establish speed or NPU precision.
#pragma once
#include "oscar_attention_launch.h"
#include "oscar_member_pipeline_launch.h"

namespace oscar_ascend {
// Per Cube: original FP32 arena, followed by one BF16 element per original
// float index. All original offsets and all FP32 state/outputs are retained.
constexpr int64_t attention_bf16_history_workspace_per_core(int64_t dim) {
  return attention_workspace_per_core(dim) / 2 * 3;
}
constexpr int64_t attention_bf16_history_mdl_workspace_per_core(int64_t dim) {
  return attention_decode_bundle_workspace_per_core(dim) / 2 * 3;
}
constexpr int64_t attention_bf16_history_member_pipeline_workspace_per_core(int64_t dim) {
  return attention_member_pipeline_workspace_per_core(dim) / 2 * 3;
}
constexpr int64_t attention_bf16_history_cluster16_workspace_per_core(int64_t dim) {
  return attention_cluster16_workspace_per_core(dim) / 2 * 3;
}
static_assert(attention_bf16_history_workspace_per_core(256)==1376256);
static_assert(attention_bf16_history_mdl_workspace_per_core(256)==1720320);
static_assert(attention_bf16_history_member_pipeline_workspace_per_core(256)==3741696);
static_assert(attention_bf16_history_cluster16_workspace_per_core(256)==7495680);

#define OSCAR_BF16_HISTORY_DECLARE(name) \
void name(void* stream,void* query,void* query_rot,void* current_key, \
    void* current_value,void* rotation_v,void* raw,void* block_table, \
    void* window_key,void* window_value,void* window_tags,void* tasks, \
    void* partial,void* lse,void* status,void* workspace, \
    int64_t tokens,int64_t query_heads,int64_t kv_heads,int64_t dim, \
    int64_t requests,int64_t table_columns,int64_t task_count, \
    int64_t block_tokens,int64_t physical_blocks,int64_t ssm_offset, \
    int64_t page_stride,int64_t window_stride,int64_t tag_stride, \
    int64_t sink,int64_t recent,int64_t speculative,int64_t splits, \
    float scale,uint32_t cores)
OSCAR_BF16_HISTORY_DECLARE(attention_bf16_history_mdl_launch);
OSCAR_BF16_HISTORY_DECLARE(attention_bf16_history_mdl_q1_launch);
OSCAR_BF16_HISTORY_DECLARE(attention_bf16_history_window_range_launch);
OSCAR_BF16_HISTORY_DECLARE(attention_bf16_history_window_range_balanced_launch);
#undef OSCAR_BF16_HISTORY_DECLARE

#define OSCAR_BF16_HISTORY_DECLARE_CLUSTER(name) \
void name(void* stream,void* query,void* query_rot,void* current_key, \
    void* current_value,void* rotation_v,void* raw,void* block_table, \
    void* window_key,void* window_value,void* window_tags,void* tasks, \
    void* partial,void* lse,void* status,void* workspace,void* cluster_stats, \
    int64_t tokens,int64_t query_heads,int64_t kv_heads,int64_t dim, \
    int64_t requests,int64_t table_columns,int64_t task_count, \
    int64_t block_tokens,int64_t physical_blocks,int64_t ssm_offset, \
    int64_t page_stride,int64_t window_stride,int64_t tag_stride, \
    int64_t sink,int64_t recent,int64_t speculative,int64_t splits, \
    float scale,uint32_t cores)
OSCAR_BF16_HISTORY_DECLARE_CLUSTER(attention_bf16_history_member_pipeline_cluster4_launch);
OSCAR_BF16_HISTORY_DECLARE_CLUSTER(attention_bf16_history_window_range_cluster16_launch);
#undef OSCAR_BF16_HISTORY_DECLARE_CLUSTER
}  // namespace oscar_ascend
