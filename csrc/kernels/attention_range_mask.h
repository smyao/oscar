// SPDX-License-Identifier: Apache-2.0
// Archive #126/#129/#145; startup D.4 four questions:
// (1) This is only the FP32 score-mask portion of fused FIA, after the old
//     unmasked ReduceSum finite check and before unchanged online softmax.
// (2) D.4 measured full-history dequant at 6.5s versus FIA at 18.7ms;
//     this experiment never restores history or changes KV ownership.
// (3) At most three complement intervals of two visible intervals are
//     written in the existing 256-score UB row, with exact -inf bits.
// (4) The instruction reduction is a hypothesis, not a speed claim; old
//     scalar mask and new mask must first be bitwise identical on CPU-debug.
#pragma once

#include "oscar_common.h"

namespace oscar_range_mask {

constexpr int32_t kColumns = 256;
constexpr int32_t kFloatPerBlock = 8;  // 32-byte A2 vector block.

struct AlignedRange {
  int32_t begin;
  int32_t end;
};

// Caller has completed V->S before SetValue. All scalar boundary writes are
// performed first; one S->V handoff precedes disjoint Duplicate interiors.
// Caller must complete V writes (PIPE_V barrier for the next vector consumer,
// or V->S when a scalar read follows). No GM access or cross-core flag occurs.
__aicore__ inline void MaskComplement(
    AscendC::LocalTensor<float> scores,
    int32_t lo0, int32_t hi0, int32_t lo1, int32_t hi1) {
  constexpr float kNegativeInfinity = -__builtin_inff();
  const bool second = hi1 > lo1;
  const int32_t maskedBegin[3] = {0, hi0, second ? hi1 : kColumns};
  const int32_t maskedEnd[3] = {lo0, second ? lo1 : kColumns, kColumns};
  AlignedRange vectorRanges[3];
  int32_t vectorCount = 0;
  for (int32_t interval = 0; interval < 3; ++interval) {
    const int32_t begin = maskedBegin[interval];
    const int32_t end = maskedEnd[interval];
    if (end <= begin) continue;
    const int32_t alignedBegin =
        (begin + kFloatPerBlock - 1) & -kFloatPerBlock;
    const int32_t prefixEnd = end < alignedBegin ? end : alignedBegin;
    for (int32_t j = begin; j < prefixEnd; ++j)
      scores.SetValue(j, kNegativeInfinity);
    const int32_t interiorEnd = end & -kFloatPerBlock;
    if (interiorEnd > prefixEnd) {
      vectorRanges[vectorCount++] = {prefixEnd, interiorEnd};
    }
    const int32_t suffixBegin = prefixEnd > interiorEnd ? prefixEnd : interiorEnd;
    for (int32_t j = suffixBegin; j < end; ++j)
      scores.SetValue(j, kNegativeInfinity);
  }
  oscar_ascend_device::Fence<AscendC::HardEvent::S_V>();
  for (int32_t i = 0; i < vectorCount; ++i) {
    const AlignedRange span = vectorRanges[i];
    AscendC::Duplicate(scores[span.begin], kNegativeInfinity,
                       span.end - span.begin);
  }
}

}  // namespace oscar_range_mask
