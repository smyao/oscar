// striped_fp32_v2: QR and Rv are BF16; dequant_mode0=prefill FP32,1=decode BF16 steps.
// Archive #126/#145/#148/#153/#154/#157 and full D.4: BF16 decode CV ABI.
// QK/PV/final-Rv all use BF16 operands and FP32 accumulation in one launch.
// Rotation consumes natural BF16 Rv; no FP32 NZ8 pointer is accepted here.
#pragma once
#include "oscar_attention_launch.h"
namespace oscar_ascend {
static_assert(attention_decode_bundle_workspace_per_core(256)==1146880,
    "BF16 decode retains the bounded GM upper bound");
void attention_cv_aligned_decode_launch(void* stream, void* query, void* query_rot,
    void* current_key, void* current_value, void* rotation_v, void* raw,
    void* block_table, void* window_key, void* window_value, void* window_tags,
    void* tasks, void* partial, void* lse, void* status, void* workspace,
    int64_t tokens, int64_t query_heads, int64_t kv_heads, int64_t dim,
    int64_t requests, int64_t table_columns, int64_t task_count,
    int64_t block_tokens, int64_t physical_blocks, int64_t ssm_offset,
    int64_t page_stride, int64_t window_stride, int64_t tag_stride,
    int64_t sink, int64_t recent, int64_t speculative, int64_t splits,
    float scale, int32_t dequant_mode, uint32_t cores);
void attention_cv_aligned_decode_q1_launch(void* stream, void* query, void* query_rot,
    void* current_key, void* current_value, void* rotation_v, void* raw,
    void* block_table, void* window_key, void* window_value, void* window_tags,
    void* tasks, void* partial, void* lse, void* status, void* workspace,
    int64_t tokens, int64_t query_heads, int64_t kv_heads, int64_t dim,
    int64_t requests, int64_t table_columns, int64_t task_count,
    int64_t block_tokens, int64_t physical_blocks, int64_t ssm_offset,
    int64_t page_stride, int64_t window_stride, int64_t tag_stride,
    int64_t sink, int64_t recent, int64_t speculative, int64_t splits,
    float scale, int32_t dequant_mode, uint32_t cores);
}
