// Archive #126/#129/#145/#148/#154-157, D.4: fixed C16 single-B1 ABI.
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
