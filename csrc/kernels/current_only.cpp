// SPDX-License-Identifier: Apache-2.0
// Archive G30-G34/#91/#126/#145 and startup D.4: compile the exact validated
// isolated BF16-current/FP32-LSE kernel, without editing its mathematical body
// or enabling a production route. D.4 old full-history restore never appears.
#include "../experiments/current_only/current_only.cpp"
#include "../include/oscar_current_only_launch.h"

#ifndef ASCENDC_CPU_DEBUG
namespace oscar_ascend {
void copy_validate_current_launch(void* stream, void* current_out,
    void* current_lse, void* output, void* output_lse, void* row_status,
    int64_t rows, int64_t dim, uint32_t vector_cores) {
  oscar_current_only_kernel<<<vector_cores, nullptr, stream>>>(
      static_cast<uint8_t*>(current_out),
      static_cast<uint8_t*>(current_lse),
      static_cast<uint8_t*>(output),
      static_cast<uint8_t*>(output_lse),
      static_cast<uint8_t*>(row_status), rows, dim);
}
}
#endif
