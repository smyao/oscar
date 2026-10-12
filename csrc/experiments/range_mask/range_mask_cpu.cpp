// SPDX-License-Identifier: Apache-2.0
// Archive #126/#129/#145; startup D.4: official CPU-debug runs the isolated
// AscendC mask helper, checks exact FP32 bits including NaN and signed zero.
// It supplies no NPU cycle or whole-model performance evidence.
#include "tikicpulib.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>

extern "C" void oscar_range_mask_probe_kernel(uint8_t*, int32_t);

int main() {
  try {
    auto* result = static_cast<uint8_t*>(AscendC::GmAlloc(3 * sizeof(int64_t)));
    if (!result) throw std::runtime_error("GmAlloc failed");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    for (int32_t mode = 0; mode <= 1; ++mode) {
      std::memset(result, 0, 3 * sizeof(int64_t));
      ICPU_RUN_KF(oscar_range_mask_probe_kernel, 1, result, mode);
      int64_t words[3];
      std::memcpy(words, result, sizeof(words));
      const int64_t expected = mode == 0 ? 33153LL * 256 : 9LL * 256;
      if (words[0] != expected || words[1] != 0 || words[2] != -1)
        throw std::runtime_error("mask bitwise mismatch or incomplete coverage");
      std::cout << "RANGE_MASK_CPU mode=" << mode << " checked=" << words[0]
                << " bitwise_mismatch=" << words[1] << '\n';
    }
    AscendC::GmFree(result);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "RANGE_MASK_CPU_ERROR " << error.what() << '\n';
    return 1;
  }
}
