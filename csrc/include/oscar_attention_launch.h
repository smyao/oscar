// SPDX-License-Identifier: Apache-2.0
// Archive G5/G6/G12/G13/#53/#108: primitive-only, single authoritative ABI.
// D.4/#140/#144: fixed task/workspace shapes prevent host loops and full-history
// allocations. A 256-KV work unit reuses score as P and PV as final rotation;
// K/V each hold one bounded tile. The old 725ms prepare / 6.5s restore remain
// forbidden, not claimed measurements. See docs/cv_implementation.md.
#pragma once
#include <cstdint>
namespace oscar_ascend {
constexpr int64_t kAttentionTaskColumns = 16;
constexpr int64_t kAttentionQueryRows = 128;
constexpr int64_t kAttentionKvRows = 256;
// Archive #70-73/#129/#143/#150 and startup D.4: diagnostic-only CV timing.
// Per owner: 4 sources x 24 int64 = 768 bytes = twelve 64-byte cache lines.
constexpr int64_t kAttentionProfileEngines = 3; // AIC, AIV0, AIV1.
constexpr int64_t kAttentionProfileSources = 4; // history/window/current/total.
constexpr int64_t kAttentionProfileFields = 24;
constexpr int64_t kAttentionProfileCacheLineBytes = 64;
constexpr int64_t attention_workspace_per_core(int64_t dim) {
  // Q[M,D], natural K/V[B,D], score/P[M,B], PV/rotation[M,D].
  return ((2 * kAttentionQueryRows + 2 * kAttentionKvRows) * dim
          + kAttentionQueryRows * kAttentionKvRows) * 4;
}
// Two fixed history tiles for the explicit M32 decode bundle; D256=1146880B.
constexpr int64_t attention_decode_bundle_workspace_per_core(int64_t dim) {
  return ((2 * 32 + 4 * kAttentionKvRows) * dim + 32 * kAttentionKvRows) * 4;
}
// Archive #126/#129/#140-145 and startup D.4: experimental source0 C4 uses
// four bounded Q/FP32 online states around one KV256 tile. The production
// workspace function above and its fe0 attention_cv_out ABI stay unchanged.
constexpr int64_t attention_cluster4_workspace_per_core(int64_t dim) {
  // Q[4,M,D], K/V[B,D], score/P[M,B], PV[M,D], acc[4,M,D],
  // max/sum[4,2,M]. No whole-history materialization.
  return ((4 * kAttentionQueryRows + 2 * kAttentionKvRows
           + kAttentionQueryRows + 4 * kAttentionQueryRows) * dim
          + kAttentionQueryRows * kAttentionKvRows
          + 4 * 2 * kAttentionQueryRows) * 4;
}
// Archive #148-151/D.4: same bounded KV256, sixteen independent query states.
// The 2026-09-29 mixed CV measurement motivates reuse; no history-sized allocation.
constexpr int64_t attention_cluster16_workspace_per_core(int64_t dim) {
  return ((16 * kAttentionQueryRows + 2 * kAttentionKvRows
           + kAttentionQueryRows + 16 * kAttentionQueryRows) * dim
          + kAttentionQueryRows * kAttentionKvRows
          + 16 * 2 * kAttentionQueryRows) * 4;
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
// Archive #126/#129/#140-151/D.4: experimental exact INT2 unpack variants.
// The original three launch functions and their workspace formulas are kept.
void attention_cv_fast_launch(void* stream, void* query, void* query_rot,
    void* current_key, void* current_value, void* rotation_v, void* raw,
    void* block_table, void* window_key, void* window_value, void* window_tags,
    void* tasks, void* partial, void* lse, void* status, void* workspace,
    int64_t tokens, int64_t query_heads, int64_t kv_heads, int64_t dim,
    int64_t requests, int64_t table_columns, int64_t task_count,
    int64_t block_tokens, int64_t physical_blocks, int64_t ssm_offset,
    int64_t page_stride, int64_t window_stride, int64_t tag_stride,
    int64_t sink, int64_t recent, int64_t speculative, int64_t splits,
    float scale, uint32_t cores);
void attention_cv_fast_q1_launch(void* stream, void* query, void* query_rot,
    void* current_key, void* current_value, void* rotation_v, void* raw,
    void* block_table, void* window_key, void* window_value, void* window_tags,
    void* tasks, void* partial, void* lse, void* status, void* workspace,
    int64_t tokens, int64_t query_heads, int64_t kv_heads, int64_t dim,
    int64_t requests, int64_t table_columns, int64_t task_count,
    int64_t block_tokens, int64_t physical_blocks, int64_t ssm_offset,
    int64_t page_stride, int64_t window_stride, int64_t tag_stride,
    int64_t sink, int64_t recent, int64_t speculative, int64_t splits,
    float scale, uint32_t cores);
// Archive #151/D.4 mixed history and q4 ownership candidates, separate exact ABIs.
void attention_cv_fast_balanced_launch(void* stream, void* query, void* query_rot,
    void* current_key, void* current_value, void* rotation_v, void* raw,
    void* block_table, void* window_key, void* window_value, void* window_tags,
    void* tasks, void* partial, void* lse, void* status, void* workspace,
    int64_t tokens, int64_t query_heads, int64_t kv_heads, int64_t dim,
    int64_t requests, int64_t table_columns, int64_t task_count,
    int64_t block_tokens, int64_t physical_blocks, int64_t ssm_offset,
    int64_t page_stride, int64_t window_stride, int64_t tag_stride,
    int64_t sink, int64_t recent, int64_t speculative, int64_t splits,
    float scale, uint32_t cores);
void attention_cv_fast_cluster16_launch(void* stream, void* query, void* query_rot,
    void* current_key, void* current_value, void* rotation_v, void* raw,
    void* block_table, void* window_key, void* window_value, void* window_tags,
    void* tasks, void* partial, void* lse, void* status, void* workspace,
    void* cluster_stats, int64_t tokens, int64_t query_heads, int64_t kv_heads,
    int64_t dim, int64_t requests, int64_t table_columns, int64_t task_count,
    int64_t block_tokens, int64_t physical_blocks, int64_t ssm_offset,
    int64_t page_stride, int64_t window_stride, int64_t tag_stride,
    int64_t sink, int64_t recent, int64_t speculative, int64_t splits,
    float scale, uint32_t cores);
void attention_cv_fast_cluster4_launch(void* stream, void* query, void* query_rot,
    void* current_key, void* current_value, void* rotation_v, void* raw,
    void* block_table, void* window_key, void* window_value, void* window_tags,
    void* tasks, void* partial, void* lse, void* status, void* workspace,
    void* cluster_stats, int64_t tokens, int64_t query_heads, int64_t kv_heads,
    int64_t dim, int64_t requests, int64_t table_columns, int64_t task_count,
    int64_t block_tokens, int64_t physical_blocks, int64_t ssm_offset,
    int64_t page_stride, int64_t window_stride, int64_t tag_stride,
    int64_t sink, int64_t recent, int64_t speculative, int64_t splits,
    float scale, uint32_t cores);
// Archive #129/#150 and D.4: independent q1 schedule with the exact fe0
// tensor ABI and bounded fe0 workspace; runtime selects it only for proven q1.
void attention_cv_q1_launch(void* stream, void* query, void* query_rot,
    void* current_key, void* current_value, void* rotation_v, void* raw,
    void* block_table, void* window_key, void* window_value, void* window_tags,
    void* tasks, void* partial, void* lse, void* status, void* workspace,
    int64_t tokens, int64_t query_heads, int64_t kv_heads, int64_t dim,
    int64_t requests, int64_t table_columns, int64_t task_count,
    int64_t block_tokens, int64_t physical_blocks, int64_t ssm_offset,
    int64_t page_stride, int64_t window_stride, int64_t tag_stride,
    int64_t sink, int64_t recent, int64_t speculative, int64_t splits,
    float scale, uint32_t cores);
void attention_cv_profile_launch(void* stream, void* query, void* query_rot,
    void* current_key, void* current_value, void* rotation_v, void* raw,
    void* block_table, void* window_key, void* window_value, void* window_tags,
    void* tasks, void* partial, void* lse, void* status, void* workspace,
    void* profile, int64_t tokens, int64_t query_heads, int64_t kv_heads,
    int64_t dim, int64_t requests, int64_t table_columns, int64_t task_count,
    int64_t block_tokens, int64_t physical_blocks, int64_t ssm_offset,
    int64_t page_stride, int64_t window_stride, int64_t tag_stride,
    int64_t sink, int64_t recent, int64_t speculative, int64_t splits,
    float scale, uint32_t cores);
void attention_cv_profile_launch(void* stream, void* query, void* query_rot,
    void* current_key, void* current_value, void* rotation_v, void* raw,
    void* block_table, void* window_key, void* window_value, void* window_tags,
    void* tasks, void* partial, void* lse, void* status, void* workspace,
    void* profile, int64_t tokens, int64_t query_heads, int64_t kv_heads,
    int64_t dim, int64_t requests, int64_t table_columns, int64_t task_count,
    int64_t block_tokens, int64_t physical_blocks, int64_t ssm_offset,
    int64_t page_stride, int64_t window_stride, int64_t tag_stride,
    int64_t sink, int64_t recent, int64_t speculative, int64_t splits,
    float scale, uint32_t cores);
void attention_cv_cluster4_launch(void* stream, void* query, void* query_rot,
    void* current_key, void* current_value, void* rotation_v, void* raw,
    void* block_table, void* window_key, void* window_value, void* window_tags,
    void* tasks, void* partial, void* lse, void* status, void* workspace,
    void* cluster_stats, int64_t tokens, int64_t query_heads, int64_t kv_heads,
    int64_t dim, int64_t requests, int64_t table_columns, int64_t task_count,
    int64_t block_tokens, int64_t physical_blocks, int64_t ssm_offset,
    int64_t page_stride, int64_t window_stride, int64_t tag_stride,
    int64_t sink, int64_t recent, int64_t speculative, int64_t splits,
    float scale, uint32_t cores);

// Archive #154: paired versioned INT2 layout; old launch ABI is unchanged.
void attention_cv_striped_launch(void* stream, void* query, void* query_rot,
    void* current_key, void* current_value, void* rotation_v, void* raw,
    void* block_table, void* window_key, void* window_value, void* window_tags,
    void* tasks, void* partial, void* lse, void* status, void* workspace,
    int64_t tokens, int64_t query_heads, int64_t kv_heads, int64_t dim,
    int64_t requests, int64_t table_columns, int64_t task_count,
    int64_t block_tokens, int64_t physical_blocks, int64_t ssm_offset,
    int64_t page_stride, int64_t window_stride, int64_t tag_stride,
    int64_t sink, int64_t recent, int64_t speculative, int64_t splits,
    float scale, uint32_t cores);
void attention_cv_striped_q1_launch(void* stream, void* query, void* query_rot,
    void* current_key, void* current_value, void* rotation_v, void* raw,
    void* block_table, void* window_key, void* window_value, void* window_tags,
    void* tasks, void* partial, void* lse, void* status, void* workspace,
    int64_t tokens, int64_t query_heads, int64_t kv_heads, int64_t dim,
    int64_t requests, int64_t table_columns, int64_t task_count,
    int64_t block_tokens, int64_t physical_blocks, int64_t ssm_offset,
    int64_t page_stride, int64_t window_stride, int64_t tag_stride,
    int64_t sink, int64_t recent, int64_t speculative, int64_t splits,
    float scale, uint32_t cores);
void attention_cv_striped_decode_launch(void* stream, void* query, void* query_rot,
    void* current_key, void* current_value, void* rotation_v, void* raw,
    void* block_table, void* window_key, void* window_value, void* window_tags,
    void* tasks, void* partial, void* lse, void* status, void* workspace,
    int64_t tokens, int64_t query_heads, int64_t kv_heads, int64_t dim,
    int64_t requests, int64_t table_columns, int64_t task_count,
    int64_t block_tokens, int64_t physical_blocks, int64_t ssm_offset,
    int64_t page_stride, int64_t window_stride, int64_t tag_stride,
    int64_t sink, int64_t recent, int64_t speculative, int64_t splits,
    float scale, uint32_t cores);
void attention_cv_striped_balanced_launch(void* stream, void* query, void* query_rot,
    void* current_key, void* current_value, void* rotation_v, void* raw,
    void* block_table, void* window_key, void* window_value, void* window_tags,
    void* tasks, void* partial, void* lse, void* status, void* workspace,
    int64_t tokens, int64_t query_heads, int64_t kv_heads, int64_t dim,
    int64_t requests, int64_t table_columns, int64_t task_count,
    int64_t block_tokens, int64_t physical_blocks, int64_t ssm_offset,
    int64_t page_stride, int64_t window_stride, int64_t tag_stride,
    int64_t sink, int64_t recent, int64_t speculative, int64_t splits,
    float scale, uint32_t cores);
void attention_cv_striped_cluster4_launch(void* stream, void* query, void* query_rot,
    void* current_key, void* current_value, void* rotation_v, void* raw,
    void* block_table, void* window_key, void* window_value, void* window_tags,
    void* tasks, void* partial, void* lse, void* status, void* workspace,
    void* cluster_stats, int64_t tokens, int64_t query_heads, int64_t kv_heads,
    int64_t dim, int64_t requests, int64_t table_columns, int64_t task_count,
    int64_t block_tokens, int64_t physical_blocks, int64_t ssm_offset,
    int64_t page_stride, int64_t window_stride, int64_t tag_stride,
    int64_t sink, int64_t recent, int64_t speculative, int64_t splits,
    float scale, uint32_t cores);
void attention_cv_striped_cluster16_launch(void* stream, void* query, void* query_rot,
    void* current_key, void* current_value, void* rotation_v, void* raw,
    void* block_table, void* window_key, void* window_value, void* window_tags,
    void* tasks, void* partial, void* lse, void* status, void* workspace,
    void* cluster_stats, int64_t tokens, int64_t query_heads, int64_t kv_heads,
    int64_t dim, int64_t requests, int64_t table_columns, int64_t task_count,
    int64_t block_tokens, int64_t physical_blocks, int64_t ssm_offset,
    int64_t page_stride, int64_t window_stride, int64_t tag_stride,
    int64_t sink, int64_t recent, int64_t speculative, int64_t splits,
    float scale, uint32_t cores);

// #154: SIMD metadata guards retain the same full FP32 tensor contract.
void attention_cv_striped_decode_simd_launch(void* stream, void* query, void* query_rot,
    void* current_key, void* current_value, void* rotation_v, void* raw,
    void* block_table, void* window_key, void* window_value, void* window_tags,
    void* tasks, void* partial, void* lse, void* status, void* workspace,
    int64_t tokens, int64_t query_heads, int64_t kv_heads, int64_t dim,
    int64_t requests, int64_t table_columns, int64_t task_count,
    int64_t block_tokens, int64_t physical_blocks, int64_t ssm_offset,
    int64_t page_stride, int64_t window_stride, int64_t tag_stride,
    int64_t sink, int64_t recent, int64_t speculative, int64_t splits,
    float scale, uint32_t cores);
void attention_cv_striped_q1_simd_launch(void* stream, void* query, void* query_rot,
    void* current_key, void* current_value, void* rotation_v, void* raw,
    void* block_table, void* window_key, void* window_value, void* window_tags,
    void* tasks, void* partial, void* lse, void* status, void* workspace,
    int64_t tokens, int64_t query_heads, int64_t kv_heads, int64_t dim,
    int64_t requests, int64_t table_columns, int64_t task_count,
    int64_t block_tokens, int64_t physical_blocks, int64_t ssm_offset,
    int64_t page_stride, int64_t window_stride, int64_t tag_stride,
    int64_t sink, int64_t recent, int64_t speculative, int64_t splits,
    float scale, uint32_t cores);
// Explicit, separately gated candidate entrypoints. Old readers remain callable.
void attention_cv_decode_bundle_launch(void* stream, void* query, void* query_rot,
    void* current_key, void* current_value, void* rotation_v, void* raw,
    void* block_table, void* window_key, void* window_value, void* window_tags,
    void* tasks, void* partial, void* lse, void* status, void* workspace,
    int64_t tokens, int64_t query_heads, int64_t kv_heads, int64_t dim,
    int64_t requests, int64_t table_columns, int64_t task_count,
    int64_t block_tokens, int64_t physical_blocks, int64_t ssm_offset,
    int64_t page_stride, int64_t window_stride, int64_t tag_stride,
    int64_t sink, int64_t recent, int64_t speculative, int64_t splits,
    float scale, uint32_t cores);
void attention_cv_decode_bundle_q1_launch(void* stream, void* query, void* query_rot,
    void* current_key, void* current_value, void* rotation_v, void* raw,
    void* block_table, void* window_key, void* window_value, void* window_tags,
    void* tasks, void* partial, void* lse, void* status, void* workspace,
    int64_t tokens, int64_t query_heads, int64_t kv_heads, int64_t dim,
    int64_t requests, int64_t table_columns, int64_t task_count,
    int64_t block_tokens, int64_t physical_blocks, int64_t ssm_offset,
    int64_t page_stride, int64_t window_stride, int64_t tag_stride,
    int64_t sink, int64_t recent, int64_t speculative, int64_t splits,
    float scale, uint32_t cores);
void attention_cv_window_range_launch(void* stream, void* query, void* query_rot,
    void* current_key, void* current_value, void* rotation_v, void* raw,
    void* block_table, void* window_key, void* window_value, void* window_tags,
    void* tasks, void* partial, void* lse, void* status, void* workspace,
    int64_t tokens, int64_t query_heads, int64_t kv_heads, int64_t dim,
    int64_t requests, int64_t table_columns, int64_t task_count,
    int64_t block_tokens, int64_t physical_blocks, int64_t ssm_offset,
    int64_t page_stride, int64_t window_stride, int64_t tag_stride,
    int64_t sink, int64_t recent, int64_t speculative, int64_t splits,
    float scale, uint32_t cores);
void attention_cv_window_range_balanced_launch(void* stream, void* query, void* query_rot,
    void* current_key, void* current_value, void* rotation_v, void* raw,
    void* block_table, void* window_key, void* window_value, void* window_tags,
    void* tasks, void* partial, void* lse, void* status, void* workspace,
    int64_t tokens, int64_t query_heads, int64_t kv_heads, int64_t dim,
    int64_t requests, int64_t table_columns, int64_t task_count,
    int64_t block_tokens, int64_t physical_blocks, int64_t ssm_offset,
    int64_t page_stride, int64_t window_stride, int64_t tag_stride,
    int64_t sink, int64_t recent, int64_t speculative, int64_t splits,
    float scale, uint32_t cores);
void attention_cv_window_range_cluster4_launch(void* stream, void* query, void* query_rot,
    void* current_key, void* current_value, void* rotation_v, void* raw,
    void* block_table, void* window_key, void* window_value, void* window_tags,
    void* tasks, void* partial, void* lse, void* status, void* workspace,
    void* cluster_stats, int64_t tokens, int64_t query_heads, int64_t kv_heads,
    int64_t dim, int64_t requests, int64_t table_columns, int64_t task_count,
    int64_t block_tokens, int64_t physical_blocks, int64_t ssm_offset,
    int64_t page_stride, int64_t window_stride, int64_t tag_stride,
    int64_t sink, int64_t recent, int64_t speculative, int64_t splits,
    float scale, uint32_t cores);
void attention_cv_window_range_cluster16_launch(void* stream, void* query, void* query_rot,
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
