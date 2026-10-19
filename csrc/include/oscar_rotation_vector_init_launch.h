// SPDX-License-Identifier: Apache-2.0
// Archive G5/G6/#87/#108/#153/#154: additive optional VectorInit direct-launch ABI.
// Separate from the frozen oscar_rotation_launch.h; parameter order, dtypes,
// checked out-only semantics and selected NPU stream match rotate_launch.
#pragma once
#include <cstdint>

namespace oscar_ascend {
// BF16 [N,H,D+16]: D operands + 32B trailer containing pre-round FP32 sum.
void rotate_bf16_query_launch(void* stream, void* input, void* rotation_transposed,
    void* output, void* status, int64_t rows, int64_t dim, bool hadamard,
    void* slots, int64_t heads, uint32_t cores);
// dtype: 0=float32, 1=float16, 2=bfloat16. Rotation is contiguous FP32 R^T.
void rotate_vector_init_launch(void* stream, void* input, void* rotation_transposed,
    void* output, void* status, int64_t rows, int64_t dim, int32_t dtype,
    bool hadamard, void* slots, int64_t heads, uint32_t cores);
}
