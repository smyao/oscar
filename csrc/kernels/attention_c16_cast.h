// Archive #164: keep normal-mask VCONV, explicit repeat strides and RNE.
// Archive #126/#145: caller-owned DMA fences and buffer lifetimes are unchanged.
// Timer evidence: C16 BF16 conversion occupies 7.476--7.628 ms per representative
// AIV; this helper targets its repeated scalar instruction issue, not INT2 math.
// D.4 four questions:
// 1. Fused FIA operand publication, inside the existing bounded C16 tile.
// 2. The failed implementation restored all history in 6499.8--6655.1 ms/card.
// 3. This only batches independent 64-element conversion repeats. It allocates
//    no UB/GM buffer and creates no history arena or host/device transfer.
// 4. Production calls cover <=8192 elements; one normal-mask instruction can
//    cover 128 repeats. Complete C16 and device speed gains remain unmeasured.
// SDK: CANN 9.1.0 dav_c220/kernel_operator_vec_vconv_impl.h, CastIntrinsicsImpl
// for bfloat16_t/float forwards explicit repeat/strides to vconv_f322bf16r;
// unlike the four-argument count overload, it never enables mask-count mode.
// Native precedent: dispatch_ffn_combine/op_kernel/utils/
// block_epilogue_pertoken_v2.hpp:182 uses FP32->BF16 RINT with {1,1,4,8}.
#pragma once
#include "oscar_common.h"

namespace oscar_c16_cast {
__aicore__ inline void CastRne(AscendC::LocalTensor<bfloat16_t> destination,
                             AscendC::LocalTensor<float> source,
                             int32_t count) {
  using namespace AscendC;
  static_assert(sizeof(float) == 4 && sizeof(bfloat16_t) == 2,
      "C16 BF16 conversion requires the audited element widths");
  const UnaryRepeatParams strides{1, 1, 4, 8};
  SetMaskNorm();
  int32_t offset = 0;
  int32_t fullRepeats = count / 64;
  while (fullRepeats > 0) {
    const uint8_t repeats =
        static_cast<uint8_t>(fullRepeats > 255 ? 255 : fullRepeats);
    // Repeat r reads [offset+64*r, offset+64*r+64) FP32 elements
    // and writes the identical element interval in BF16. The 8/4 repeat
    // strides are 256/128 bytes, so repeats neither overlap nor skip data.
    Cast(destination[offset], source[offset], RoundMode::CAST_RINT,
         static_cast<uint64_t>(64), repeats, strides);
    const int32_t elements = static_cast<int32_t>(repeats) * 64;
    offset += elements;
    fullRepeats -= static_cast<int32_t>(repeats);
  }
  if (offset < count) {
    // Only the final <=63 lanes are active. Both bases still have the
    // original 32-byte alignment; no element at index >=count is accessed.
    Cast(destination[offset], source[offset], RoundMode::CAST_RINT,
         static_cast<uint64_t>(count - offset), static_cast<uint8_t>(1),
         strides);
  }
  ResetMask();
}
}  // namespace oscar_c16_cast
