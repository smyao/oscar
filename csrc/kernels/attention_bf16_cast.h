// Archive #164/#126/#145; complete startup D.4 previously read.
// D.4: only bounded fused-FIA operand conversion, never full-history restore.
// The old restore cost6499.8--6655.1ms is a forbidden structure, not a target.
// Use explicit normal-mask single repeats for FP32->BF16 RNE: no large
// counter-mode VCONV. Physical payload, FP32 accumulators and math stay fixed.
// Each repeat consumes <=256B and produces <=128B. Caller owns DMA fences.
// This repairs the newly introduced conversion form; target PC attribution
// and NPU completion remain separate from source/local functional evidence.
#pragma once
#include "oscar_common.h"

namespace oscar_bf16 {
__aicore__ inline void CastRne(AscendC::LocalTensor<bfloat16_t> destination,
                             AscendC::LocalTensor<float> source,
                             int32_t count) {
  using namespace AscendC;
  static_assert(sizeof(float)==4 && sizeof(bfloat16_t)==2,
      "BF16 operand conversion requires the audited element widths");
  // FP32 has64 elements per256B repeat; BF16 output occupies four32B blocks.
  const UnaryRepeatParams strides{1,1,4,8};
  SetMaskNorm();
  for(int32_t offset=0;offset<count;offset+=64) {
    const uint64_t active=static_cast<uint64_t>(count-offset<64?count-offset:64);
    Cast(destination[offset],source[offset],RoundMode::CAST_RINT,active,
         static_cast<uint8_t>(1),strides);
  }
  ResetMask();
}
}
