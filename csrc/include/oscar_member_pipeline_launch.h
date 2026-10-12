// Archive #148/#150/#153/#154, D.4: qualified H32/member-pipeline ABI.
// Old C4 declarations/workspace remain unchanged. Both new readers require
// the actual two-KV/two-score arena; registration is not NPU/graph acceptance.
#pragma once
#include "oscar_attention_launch.h"
namespace oscar_ascend {
constexpr int64_t attention_member_pipeline_workspace_per_core(int64_t dim) {
  return attention_cluster4_workspace_per_core(dim)
      + (2*kAttentionKvRows*dim+kAttentionQueryRows*kAttentionKvRows)*4;
}
static_assert(attention_member_pipeline_workspace_per_core(256)==2494464,
    "C4 member pipeline D256 workspace must match the qualified kernel");
void attention_cv_member_pipeline_cluster4_launch(void* stream, void* query, void* query_rot,
    void* current_key, void* current_value, void* rotation_v, void* raw,
    void* block_table, void* window_key, void* window_value, void* window_tags,
    void* tasks, void* partial, void* lse, void* status, void* workspace,
    void* cluster_stats, int64_t tokens, int64_t query_heads, int64_t kv_heads,
    int64_t dim, int64_t requests, int64_t table_columns, int64_t task_count,
    int64_t block_tokens, int64_t physical_blocks, int64_t ssm_offset,
    int64_t page_stride, int64_t window_stride, int64_t tag_stride,
    int64_t sink, int64_t recent, int64_t speculative, int64_t splits,
    float scale, uint32_t cores);
void attention_request_window_member_pipeline_cluster4_launch(void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,float,int64_t,int64_t,uint32_t);
}
