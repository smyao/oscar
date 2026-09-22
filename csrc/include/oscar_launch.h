// SPDX-License-Identifier: Apache-2.0
// Archive G5/G6/G12/G13/#87: one authoritative primitive-only launch ABI.
#pragma once
#include <cstdint>

// Single direct-launch ABI. No AscendC-specific types cross this header.
namespace oscar_ascend {
void store_int2_launch(void* stream, void* key, void* value, void* slots,
    void* raw, void* status, int64_t tokens, int64_t heads, int64_t dim,
    int64_t block_tokens, int64_t blocks, int64_t ssm_offset,
    int64_t page_stride, bool slots_i64, uint32_t cores);
void merge_lse_launch(void* stream, void* partial, void* partial_lse,
    void* output, void* lse, void* status, int64_t rows, int64_t splits,
    int64_t dim, uint32_t cores);
}  // namespace oscar_ascend
