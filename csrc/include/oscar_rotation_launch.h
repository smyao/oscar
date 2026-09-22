// SPDX-License-Identifier: Apache-2.0
// Archive G5/G6/G12/G13/#87/#108: a single primitive-only direct-launch ABI.
#pragma once
#include <cstdint>

namespace oscar_ascend {
// dtype: 0=float32, 1=float16, 2=bfloat16. Rotation is contiguous FP32 R^T.
void rotate_launch(void* stream, void* input, void* rotation_transposed,
    void* output, void* status, int64_t rows, int64_t dim, int32_t dtype,
    bool hadamard, void* slots, int64_t heads, uint32_t cores);

// Raw K/V strides are BF16 ELEMENTS; tag stride is int64 ELEMENTS. Packed
// storage offsets/strides are BYTES. recent_capacity includes the draft slack.
// slots and positions are int64[N]; status is int32[N,H]. Slots in this
// batch must contain at most one contiguous writable interval per page.
void rotate_clip_store_launch(void* stream, void* key, void* value,
    void* rk_transposed, void* rv_transposed, void* slots, void* positions,
    void* packed, void* raw_key, void* raw_value, void* raw_tags, void* status,
    int64_t tokens, int64_t heads, int64_t dim, int32_t dtype,
    int64_t block_tokens, int64_t blocks, int64_t ssm_offset,
    int64_t page_stride, int64_t raw_key_page_stride,
    int64_t raw_value_page_stride, int64_t tag_page_stride,
    int64_t sink_tokens, int64_t recent_capacity, float k_clip,
    float v_clip, bool hadamard, uint32_t cores);
}
