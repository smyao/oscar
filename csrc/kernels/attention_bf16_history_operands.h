// SPDX-License-Identifier: Apache-2.0
// RESEARCH ONLY; the independent CPU numerical gate is FAILED.
// Archive G4/#81-83/#126/#145/#148/#154/#157; startup D.4 reread in full.
// D.4: (1) bounded source0 operand publication and QK/PV inside fused FIA;
// (2) no repetition of the 6499.8-6655.1ms full-history restore;
// (3) one KV256 tile, separate FP32 state/BF16 operands, explicit local
// ownership fences, and one initialized Cube Matmul resource set at a time;
// (4) 16KiB extra AIV scratch, finite tile traffic and dtype-switch overhead.
// No target precision, latency, CANN-build or resource-reinit claim follows.
// Native A2 BF16 operand/float C precedent: sparse_flash_attention/op_kernel/
// arch22/sparse_flash_attention_service_cube_mla.h; BF16 CAST_RINT precedent:
// sparse_flash_attention_service_vector_mla.h:1264-1268 and this project's
// rotate_clip_store_striped.cpp:399. P at native :686 instead uses CAST_ROUND;
// this candidate deliberately matches stage1 RNE, not native FIA bit identity.
// Native TPipe Reset use: hamming_dist_top_k_parallel.h:70-79,205-212.
// That proves the public reset API, not MatmulImpl's repeated Init contract:
// CANN CPU-debug/NPU must validate the latter before any promotion.
#pragma once
#include "oscar_common.h"
#include "lib/matmul_intf.h"

namespace oscar_bf16_history {
using namespace AscendC;
using oscar_ascend_device::Fence;
constexpr int32_t kOperandScratchElements = 8192;
constexpr int32_t kOperandScratchBytes = kOperandScratchElements * 2;

// Only the source0 publisher calls this. All published lengths are D256 or
// KV256 multiples, hence each BF16 DataCopy is a multiple of 32 bytes.
// Each chunk finishes its MTE3 read before either Vector or MTE2 can reuse
// the scratch. Existing caller fences continue to own the source FP32 UB.
__aicore__ inline void Publish(GlobalTensor<bfloat16_t> destination,
    LocalTensor<float> source, TBuf<TPosition::VECCALC>& scratch,
    int32_t elements) {
  auto rounded = scratch.Get<bfloat16_t>();
  for (int32_t begin = 0; begin < elements; begin += kOperandScratchElements) {
    const int32_t count = elements - begin < kOperandScratchElements
        ? elements - begin : kOperandScratchElements;
    Fence<HardEvent::MTE3_V>();
    Cast(rounded, source[begin], RoundMode::CAST_RINT, count);
    Fence<HardEvent::V_MTE3>();
    DataCopy(destination[begin], rounded, count);
    Fence<HardEvent::MTE3_V>();
  }
}

// The old MDL float Matmul alone reserves 395264B L1 and all double-buffered
// L0A/L0B. Initializing both dtypes against one live TPipe would overallocate.
// At a dtype transition, the prior IterateAll has End'ed; drain all local
// pipelines, reset the allocator/events, then initialize only the new type.
// No AIV TPipe is reset, and no cross-core protocol flag is changed here.
// source1/2 and final Rv always request false, including after source0.
// This conservative resource lifetime is intentionally a device-test gate:
// the current host has no SDK to prove MatmulImpl re-Init implementation.
template<class FloatMatmul, class Bf16Matmul>
__aicore__ inline void SelectMatmul(bool history, bool& historyActive,
    TPipe& pipe, FloatMatmul& fp32, Bf16Matmul& bf16) {
  if (history == historyActive) return;
  if (historyActive) bf16.End(); else fp32.End();
  PipeBarrier<PIPE_ALL>();
  pipe.Reset();
  if (history) bf16.Init(static_cast<const TCubeTiling*>(nullptr), &pipe);
  else fp32.Init(static_cast<const TCubeTiling*>(nullptr), &pipe);
  historyActive = history;
  SetHF32Mode(false);
}
}  // namespace oscar_bf16_history
