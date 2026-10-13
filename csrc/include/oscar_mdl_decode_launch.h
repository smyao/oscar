// Archive #126/#129/#145/#153/#154 and full D.4: qualified MDL q4 ABI.
// No change to query/cache/task mathematics, flags or the bounded GM arena.
#pragma once
#include "oscar_attention_launch.h"
namespace oscar_ascend {
static_assert(attention_decode_bundle_workspace_per_core(256)==1146880,
    "MDL retains the qualified decode workspace");
void attention_cv_mdl_decode_launch(void* stream, void* query, void* query_rot,
    void* current_key, void* current_value, void* rotation_v, void* raw,
    void* block_table, void* window_key, void* window_value, void* window_tags,
    void* tasks, void* partial, void* lse, void* status, void* workspace,
    int64_t tokens, int64_t query_heads, int64_t kv_heads, int64_t dim,
    int64_t requests, int64_t table_columns, int64_t task_count,
    int64_t block_tokens, int64_t physical_blocks, int64_t ssm_offset,
    int64_t page_stride, int64_t window_stride, int64_t tag_stride,
    int64_t sink, int64_t recent, int64_t speculative, int64_t splits,
    float scale, uint32_t cores);
void attention_cv_mdl_q1_launch(void* stream, void* query, void* query_rot,
    void* current_key, void* current_value, void* rotation_v, void* raw,
    void* block_table, void* window_key, void* window_value, void* window_tags,
    void* tasks, void* partial, void* lse, void* status, void* workspace,
    int64_t tokens, int64_t query_heads, int64_t kv_heads, int64_t dim,
    int64_t requests, int64_t table_columns, int64_t task_count,
    int64_t block_tokens, int64_t physical_blocks, int64_t ssm_offset,
    int64_t page_stride, int64_t window_stride, int64_t tag_stride,
    int64_t sink, int64_t recent, int64_t speculative, int64_t splits,
    float scale, uint32_t cores);
}
