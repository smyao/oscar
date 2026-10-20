// SPDX-License-Identifier: Apache-2.0
// Archive #126/#134/#145/#148/#164: keep FP32 finite/error semantics, the
// original reduction instruction, bounded UB, and score/P ownership fences.
// D.4 four questions (complete original timing log reread):
// 1. This changes the finite-check part of fused history FIA/softmax only.
// 2. Historical whole-history dequant6499.8-6655.1ms/95.4% and stores209-215ms
//    forbid solving this by restoring history or changing the write path.
// 3. The same KV256 score rows, same count-mode vcadd/ACC_VAL and same Finite
//    predicate remain. Only unused scalar-to-UB result publication is removed;
//    every V->S completion required to read ACC_VAL remains. No new buffer,
//    history materialization, reduced precision, mask change or cross-core flag.
// 4. Target timer ranks softmax_math first (23.4-23.8ms per representative
//    AIV); its finite subshare and this candidate's net speed are UNMEASURED.
//
// Source contract: official CANN9.1 A2 (__NPU_ARCH__==2201),
// impl/basic_api/dav_c220/kernel_operator_vec_reduce_impl.h:153-178.
// ReduceSum<float>(...,count) uses vcadd(...,1), reads get_acc_val after
// V_S, stores that scalar to dst, then publishes the store with S_V/S_MTE3.
// Our caller only asks Finite(scalar); nothing consumes the scalar UB store.
// WholeReduceSum uses vcadd(...,0) and is intentionally NOT substituted.
#pragma once
#include "oscar_common.h"

#if defined(__NPU_ARCH__) && __NPU_ARCH__ != 2201
#error "C16 ACC_VAL finite check is qualified only for the Ascend A2 ISA"
#endif

namespace oscar_c16_softmax {

// scores has exactly 256 FP32 elements per row. dst preserves the original
// 32-byte result-slot spacing and is dead unpack UB, not score/online state.
// All rows must execute even after failure: error ownership stays unchanged.
__aicore__ inline bool FiniteRows256(
    const AscendC::LocalTensor<float>& dst,
    const AscendC::LocalTensor<float>& scores, const int32_t rows) {
  bool finite = true;
  if ASCEND_IS_AIV {
    AscendC::SetMaskCount();
    AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(256);
    for (int32_t r = 0; r < rows; ++r) {
      AscendC::PipeBarrier<PIPE_V>();
      // Exactly the SDK count-mode ReduceSum instruction and row order.
      vcadd(reinterpret_cast<__ubuf__ float*>(dst[r * 8].GetPhyAddr()),
          reinterpret_cast<__ubuf__ float*>(scores[r * 256].GetPhyAddr()),
          1, 1, 1, 8, 1);
      oscar_ascend_device::Fence<AscendC::HardEvent::V_S>();
      int64_t accVal = get_acc_val();
      const float value = *reinterpret_cast<float*>(&accVal);
      if (!oscar_ascend_device::Finite(value)) finite = false;
    }
    // Restore the exact normal-mask state the original ReduceSum leaves.
    AscendC::SetMaskNorm();
    AscendC::ResetMask();
  }
  return finite;
}

}  // namespace oscar_c16_softmax
