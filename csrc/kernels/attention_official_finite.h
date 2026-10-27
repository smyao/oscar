// SPDX-License-Identifier: Apache-2.0
// Archive #126/#134/#145/#148/#164; complete startup D.4 reviewed.
// D.4(1): fused FIA score finite validation, before the unchanged softmax.
// D.4(2): no full-history restoration (old 6499.8--6655.1 ms/card).
// D.4(3): a sufficient bound certifies the ORIGINAL row-sum predicate. If
// uncertified, execute that same count-mode vcadd/ACC_VAL predicate, including
// finite overflow/cancellation. Scores, masks, math, errors and DMA are intact.
// D.4(4): reuse dead KV dequant UB, <=8192 floats, no new allocation. This
// removes row-by-row scalar waits on certified blocks; device speed unmeasured.
#pragma once
#include "attention_c16_softmax_helpers.h"

namespace oscar_official_finite {
// The certificate is deliberately far from FP32 overflow. Abs followed by a
// nonnegative sum cannot hide an input larger than 2^110. Hence every term of
// every ORIGINAL 256-column row is bounded by 2^110; all intermediates of its
// addition tree are below 2^119 even allowing rounding, far below 2^128.
// NaN/Inf and extreme finite cancellation are NOT decided by this bound:
// they retain the original per-row reduction and exact finite/error result.
// dst: original finite-result scratch; work: dead, disjoint >=rows*256 floats.
__aicore__ inline bool FiniteRows256(
    const AscendC::LocalTensor<float>& dst,
    const AscendC::LocalTensor<float>& scores,
    const AscendC::LocalTensor<float>& work, const int32_t rows) {
  if ASCEND_IS_AIV {
    // Tiny blocks retain their original work. Complete C16 blocks have32 rows.
    if(rows>=8) {
      const int32_t count=rows*256;
      AscendC::Abs(work,scores,count);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskCount();
      AscendC::SetVectorMask<float,AscendC::MaskMode::COUNTER>(count);
      vcadd(reinterpret_cast<__ubuf__ float*>(dst.GetPhyAddr()),
          reinterpret_cast<__ubuf__ float*>(work.GetPhyAddr()),1,1,1,8,1);
      oscar_ascend_device::Fence<AscendC::HardEvent::V_S>();
      int64_t accVal=get_acc_val();
      const float bound=*reinterpret_cast<float*>(&accVal);
      AscendC::SetMaskNorm();
      AscendC::ResetMask();
      if(oscar_ascend_device::Finite(bound) && bound<=0x1p110F)return true;
    }
  }
  return oscar_c16_softmax::FiniteRows256(dst,scores,rows);
}
} // namespace oscar_official_finite
