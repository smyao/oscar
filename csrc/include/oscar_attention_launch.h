// SPDX-License-Identifier: Apache-2.0
// Archive G5/G6/G12/G13/#53/#108: primitive-only, single authoritative ABI.
// D.4: prepare/history/window; fixed task/workspace shapes prevent host loops
// and full-history allocations. D.4's 725ms prepare / 6.5s restore are forbidden
// targets, not claimed measurements. See docs/cv_implementation.md four questions.
#pragma once
#include <cstdint>
namespace oscar_ascend {
constexpr int64_t kAttentionTaskColumns = 16;
constexpr int64_t attention_workspace_per_core(int64_t dim) {
  return (256 * dim + 4096) * 4;
}
void prepare_attention_tasks_launch(void* stream, void* qstarts, void* lengths,
    void* slots, void* tasks, void* positions, int64_t requests, int64_t tokens,
    int64_t query_heads, int64_t kv_heads, int64_t sink, int64_t recent,
    int64_t splits, bool slots_i64, void* block_table, int64_t table_columns,
    bool use_slot_context, uint32_t cores);
void attention_cv_launch(void* stream, void* query, void* query_rot,
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
