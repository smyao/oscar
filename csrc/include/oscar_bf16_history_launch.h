// SPDX-License-Identifier: Apache-2.0
// Archive G4/#81-83/#126/#145/#148/#154/#157; full startup D.4 reviewed.
// Historical ABI names now cover BF16 matrix operands for every source.
// User default selection is not numerical/device/graph acceptance.
// D.4: (1) fixed-tile FIA scratch; (2) no 6.5s whole-history restore;
// (3) independent compact BF16 operands/FP32 score-state arenas preserve
// member/round ownership; (4) fewer reserved GM bytes, not measured speed.
#pragma once
#include "oscar_attention_launch.h"
#include "oscar_member_pipeline_launch.h"

namespace oscar_ascend {
// Element offsets are relative to their OWN typed arena. Q/K/V/P are BF16;
// score, PV, normalized inverse-rotation input, accumulators and stats FP32.
// Normalized input and PV/rotation each need one MxD tile: final member
// normalization/rotation is serialized by the existing Cube/Vector handshake.
// No live state aliases another region, including C4's two score/P slots.
struct Bf16AttentionWorkspace {
  int64_t dim, rows, members, kvBuffers, scoreSlots;
  constexpr int64_t Norm() const { return 0; }
  constexpr int64_t Score() const { return rows*dim; }
  constexpr int64_t ExtraScore() const { return Score()+rows*kAttentionKvRows; }
  constexpr int64_t Pv() const { return Score()+scoreSlots*rows*kAttentionKvRows; }
  constexpr int64_t Acc() const { return Pv()+rows*dim; }
  constexpr int64_t Stat() const { return Acc()+(members>1?members*rows*dim:0); }
  constexpr int64_t FloatElements() const { return Stat()+(members>1?members*2*rows:0); }
  constexpr int64_t Query() const { return 0; }
  constexpr int64_t Key() const { return members*rows*dim; }
  constexpr int64_t Value() const { return Key()+kAttentionKvRows*dim; }
  constexpr int64_t ExtraKv() const { return Key()+2*kAttentionKvRows*dim; }
  constexpr int64_t Probability() const { return Key()+kvBuffers*2*kAttentionKvRows*dim; }
  constexpr int64_t ExtraProbability() const { return Probability()+rows*kAttentionKvRows; }
  constexpr int64_t OperandElements() const { return Probability()+scoreSlots*rows*kAttentionKvRows; }
  constexpr int64_t Bytes() const { return FloatElements()*4+OperandElements()*2; }
};
constexpr int64_t attention_bf16_history_workspace_per_core(int64_t dim) {
  return Bf16AttentionWorkspace{dim,128,1,1,1}.Bytes();
}
constexpr int64_t attention_bf16_history_mdl_workspace_per_core(int64_t dim) {
  return Bf16AttentionWorkspace{dim,32,1,2,1}.Bytes();
}
constexpr int64_t attention_bf16_history_member_pipeline_workspace_per_core(int64_t dim) {
  return Bf16AttentionWorkspace{dim,128,4,2,2}.Bytes();
}
constexpr int64_t attention_bf16_history_cluster16_workspace_per_core(int64_t dim) {
  return Bf16AttentionWorkspace{dim,128,16,1,1}.Bytes();
}
static_assert(attention_bf16_history_workspace_per_core(256)==786432);
static_assert(attention_bf16_history_mdl_workspace_per_core(256)==655360);
static_assert(attention_bf16_history_member_pipeline_workspace_per_core(256)==1970176);
static_assert(attention_bf16_history_cluster16_workspace_per_core(256)==3883008);

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
