// SPDX-License-Identifier: Apache-2.0
// Archive G5/G6/#91/#125/#145 and startup D.4: optional out-only current
// suffix launch ABI. The validated experiment source supplies its math.
#pragma once
#include <cstdint>

namespace oscar_ascend {
void copy_validate_current_launch(void* stream, void* current_out,
    void* current_lse, void* output, void* output_lse, void* row_status,
    int64_t rows, int64_t dim, uint32_t vector_cores);
}
