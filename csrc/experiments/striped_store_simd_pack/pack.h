// SPDX-License-Identifier: Apache-2.0
// Research only. Archive G25-G34/#13-22/#126/#144-145/#151/#154-156:
// retain quantizer arithmetic, FP16-rounded scale/zero, final status ordering,
// and the current striped code permutation. #21 forbids unsigned/float casts;
// this helper converts already-clamped int32 codes to int16 with CAST_NONE.
// #145/#154 require explicit UB ownership; CPU equality cannot prove NPU races.
// #155 requires raw-byte equality, including metadata, NaN payloads and zeros.
// D.4 four questions (complete original timing log personally reread):
// 1. This replaces only UB packing inside phase1_stores, after quantization.
// 2. Historical phase1_stores cost 208.6-216.0 ms/16K, while full-history
//    dequant cost 6499.8-6655.1 ms. Neither is a measurement of this helper.
// 3. Bounded SIMD integer shifts/OR replace per-code Scalar accesses; only
//    new-token codes are processed. No history read, materialization, extra
//    launch, changed reduction tree, quantizer, clipping, metadata or status.
// 4. D256 removes 256 GetValue + 64 byte SetValue calls per side. Added work
//    is one exact integer Cast, seven shifts and seven ORs. Actual latency,
//    NPU bitwise/graph safety and service benefit remain unmeasured.
// API evidence (installed CANN 9.1, impl/dav_c220):
// kernel_operator_vec_vconv_impl.h:640-653: vconv_s322s16, CAST_NONE only;
// kernel_operator_vec_binary_scalar_impl.h:241-250: uint16 ShiftLeft/vshl;
// kernel_operator_vec_binary_impl.h:390-397: uint16 Or/vor. Do not use the
// 32-bit count-Or compatibility path, which reinterprets its operands as 16b.
// PR semantic reference: references/oscar-vllm-pr46774/vllm/v1/attention/ops/
// triton_oscar_store.py:47-77. Current striped physical bytes are the accepted
// invertible permutation, not the PR's contiguous-four-code byte layout.
#pragma once

#include "../../kernels/oscar_common.h"

namespace oscar_striped_store_simd_pack {
using namespace AscendC;
using oscar_ascend_device::Fence;

// All plane origins and dst are 32-byte aligned. Only width words are live.
// Each source code is 0..3; shift 14 therefore fits exactly in uint16_t.
__aicore__ inline void ShiftAndFold(LocalTensor<uint16_t> planes,
                                    int32_t stride, int32_t width,
                                    LocalTensor<uint16_t> dst) {
  for (int32_t plane = 1; plane < 8; ++plane) {
    ShiftLeft(planes[plane * stride], planes[plane * stride],
              static_cast<uint16_t>(plane * 2), width);
  }
  PipeBarrier<PIPE_V>();
  for (int32_t plane = 0; plane < 8; plane += 2) {
    Or(planes[plane * stride], planes[plane * stride],
       planes[(plane + 1) * stride], width);
  }
  PipeBarrier<PIPE_V>();
  Or(planes, planes, planes[2 * stride], width);
  Or(planes[4 * stride], planes[4 * stride], planes[6 * stride], width);
  PipeBarrier<PIPE_V>();
  Or(dst, planes, planes[4 * stride], width);
}

// Caller contract:
// - dim is 64, 128 or 256, as checked by the writer's binding;
// - codes contains the original CAST_TRUNC output, already clamped to 0..3;
// - deadWork is disjoint from codes/packed and has at least 1024 bytes;
// - packedOffset is 8-byte aligned for D64, 32-byte aligned otherwise;
// - packed has exactly dim/4 writable bytes at this offset. No padded write
//   may touch the adjacent V codes, metadata, another row, or a guard.
// The writer's row*160 + side*(dim/4) offsets satisfy these alignments.
__aicore__ inline void PackCodes(LocalTensor<int32_t> codes,
                                 LocalTensor<float> deadWork,
                                 LocalTensor<uint8_t> packed,
                                 int32_t dim, int32_t packedOffset) {
  // Original float->int32 Cast reads deadWork. Retire that read and produce
  // codes before reusing the same work allocation as the int16 destination.
  PipeBarrier<PIPE_V>();
  auto narrow = deadWork.ReinterpretCast<int16_t>();
  Cast(narrow, codes, RoundMode::CAST_NONE, dim);
  PipeBarrier<PIPE_V>();
  auto planes = narrow.ReinterpretCast<uint16_t>();

  if (dim == 64) {
    // Compact input: bytes [0,128). Eight padded planes: [128,384), each
    // 32-byte aligned and only its first 16 bytes live. Scalar relocation
    // is needed because the compact plane stride is just 16 bytes.
    Fence<HardEvent::V_S>();
    auto words = deadWork.ReinterpretCast<uint64_t>();
    for (int32_t plane = 0; plane < 8; ++plane) {
      const uint64_t low = words.GetValue(2 * plane);
      const uint64_t high = words.GetValue(2 * plane + 1);
      words.SetValue(16 + 4 * plane, low);
      words.SetValue(16 + 4 * plane + 1, high);
    }
    Fence<HardEvent::S_V>();
    auto padded = planes[64];
    ShiftAndFold(padded, 16, 8, padded);
    Fence<HardEvent::V_S>();
    // D64 V begins at byte16, so a 32-byte vector destination is illegal.
    // Exactly 16 bytes are copied, retaining the target's little-endian
    // uint16 representation and never widening a store over adjacent data.
    auto output = packed.ReinterpretCast<uint64_t>();
    output.SetValue(packedOffset / 8, words.GetValue(16));
    output.SetValue(packedOffset / 8 + 1, words.GetValue(17));
    // Retire Scalar reads of work before the next quantizer reuses it.
    Fence<HardEvent::S_V>();
  } else {
    // D128: 8x16 words; D256: 8x32 words. Every plane and the destination
    // are aligned. Final OR writes precisely D/4 bytes, without byte loops.
    const int32_t width = dim / 8;
    auto output = packed.ReinterpretCast<uint16_t>();
    ShiftAndFold(planes, width, width, output[packedOffset / 2]);
    Fence<HardEvent::V_S>();
  }
  // The caller now executes its unchanged Scalar metadata writes and later
  // S_MTE3 publication. Both branches have completed the Vector packed write
  // before those operations; existing writer MTE3_S reuse fences stay intact.
}
}  // namespace oscar_striped_store_simd_pack
