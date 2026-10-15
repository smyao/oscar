// Archive #126/#129/#145/#148/#153/#154-157 and full startup D.4.
// Qualified q4 hybrid-L1 ABI. The rotation pointer carries lossless FP32 NZ,
// index=(column/8)*256*8+row*8+column%8, packed once at initialization.
// Bounded KV256 GM arena; no full-history restore or native source changes.
#pragma once
#include "oscar_attention_launch.h"
namespace oscar_ascend {
static_assert(attention_decode_bundle_workspace_per_core(256)==1146880,
    "hybrid L1 retains the qualified decode GM workspace");
void attention_cv_hybrid_l1_decode_launch(void* stream, void* query, void* query_rot,
    void* current_key, void* current_value, void* rotation_v_nz, void* raw,
    void* block_table, void* window_key, void* window_value, void* window_tags,
    void* tasks, void* partial, void* lse, void* status, void* workspace,
    int64_t tokens, int64_t query_heads, int64_t kv_heads, int64_t dim,
    int64_t requests, int64_t table_columns, int64_t task_count,
    int64_t block_tokens, int64_t physical_blocks, int64_t ssm_offset,
    int64_t page_stride, int64_t window_stride, int64_t tag_stride,
    int64_t sink, int64_t recent, int64_t speculative, int64_t splits,
    float scale, uint32_t cores);
}
