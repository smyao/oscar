// Archive #111/#126/#145/#164; D.4 bounded144B format, BF16 compute.
#pragma once
#include "oscar_bf16_cluster_launch.h"
namespace oscar_ascend {
#define OSCAR_ALIGNED_CLUSTER_DECL(name) \
void name(void* stream,void* query,void* query_rot,void* current_key,void* current_value, \
    void* rotation_v,void* raw,void* block_table,void* window_key,void* window_value, \
    void* window_tags,void* tasks,void* partial,void* lse,void* status,void* workspace, \
    void* cluster_stats,int64_t tokens,int64_t query_heads,int64_t kv_heads,int64_t dim, \
    int64_t requests,int64_t table_columns,int64_t task_count,int64_t block_tokens, \
    int64_t physical_blocks,int64_t ssm_offset,int64_t page_stride,int64_t window_stride, \
    int64_t tag_stride,int64_t sink,int64_t recent,int64_t speculative,int64_t splits, \
    float scale,int32_t dequant_mode,uint32_t cores)
OSCAR_ALIGNED_CLUSTER_DECL(attention_cv_aligned_cluster4_launch);
OSCAR_ALIGNED_CLUSTER_DECL(attention_cv_aligned_cluster16_launch);
OSCAR_ALIGNED_CLUSTER_DECL(attention_cv_aligned_cluster16_control_launch);
#undef OSCAR_ALIGNED_CLUSTER_DECL
}
