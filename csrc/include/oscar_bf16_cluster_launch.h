// Archive #126/#145/#148/#153/#154/#157; D.4: bounded BF16 cluster ABI.
// Natural BF16 Rv, genuine BF16 Q/K/V/P, FP32 score/state/partial/LSE.
// Each launch completes original-space output; no second Rv launch exists.
#pragma once
#include "oscar_attention_launch.h"
namespace oscar_ascend {
constexpr int64_t attention_bf16_cluster4_workspace_per_core(int64_t dim) {
  return attention_cluster4_workspace_per_core(dim)
      +(2*kAttentionKvRows*dim+kAttentionQueryRows*kAttentionKvRows)*4;
}
constexpr int64_t attention_bf16_cluster16_workspace_per_core(int64_t dim) {
  return attention_cluster16_workspace_per_core(dim)
      +(2*kAttentionKvRows*dim+15*kAttentionQueryRows*kAttentionKvRows)*4;
}
static_assert(attention_bf16_cluster4_workspace_per_core(256)==2494464,"C4 BF16 GM upper bound changed");
static_assert(attention_bf16_cluster16_workspace_per_core(256)==7487488,"C16 BF16 GM upper bound changed");
#define OSCAR_BF16_CLUSTER_DECL(name) \
void name(void* stream,void* query,void* query_rot,void* current_key,void* current_value, \
    void* rotation_v,void* raw,void* block_table,void* window_key,void* window_value, \
    void* window_tags,void* tasks,void* partial,void* lse,void* status,void* workspace, \
    void* cluster_stats,int64_t tokens,int64_t query_heads,int64_t kv_heads,int64_t dim, \
    int64_t requests,int64_t table_columns,int64_t task_count,int64_t block_tokens, \
    int64_t physical_blocks,int64_t ssm_offset,int64_t page_stride,int64_t window_stride, \
    int64_t tag_stride,int64_t sink,int64_t recent,int64_t speculative,int64_t splits, \
    float scale,uint32_t cores)
OSCAR_BF16_CLUSTER_DECL(attention_cv_bf16_cluster4_launch);
OSCAR_BF16_CLUSTER_DECL(attention_cv_bf16_cluster16_launch);
#undef OSCAR_BF16_CLUSTER_DECL
}
