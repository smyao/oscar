// SPDX-License-Identifier: Apache-2.0
// Residual-compensated BF16 precision contract (#173); default selection is not target acceptance.
// Archive G4/#81-83/#126/#145/#148/#154/#157; startup D.4 reread in full.
// D.4: (1) bounded all-source operand publication and QK/PV inside fused FIA;
// (2) no repetition of the 6499.8-6655.1ms full-history restore;
// (3) one KV256 tile, separate FP32 state/BF16 operands, explicit local
// ownership fences, and one initialized Cube Matmul resource set at a time;
// (4) 16KiB extra AIV scratch, finite tile traffic and dtype-switch overhead;
// BF16-first initialization removes the unused initial FP32 allocation/reset.
// No target precision, latency, CANN-build or resource-reinit claim follows.
// Native A2 BF16 operand/float C precedent: sparse_flash_attention/op_kernel/
// arch22/sparse_flash_attention_service_cube_mla.h; BF16 CAST_RINT precedent:
// sparse_flash_attention_service_vector_mla.h:1264-1268 and this project's
// rotate_clip_store_striped.cpp:399. P at native :686 instead uses CAST_ROUND;
// this implementation specifies RNE, not native dense FIA bit identity.
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

// Archive #173: preserve the BF16 rounding residual without modifying source
// (softmax still needs its original FP32 probabilities for the denominator).
// D.4: two bounded BF16 planes, no full history restoration; 4 Cube products
// per history multiplication are additional work, not a performance claim.
__aicore__ inline void Publish(GlobalTensor<bfloat16_t> destination,
    GlobalTensor<bfloat16_t> residualDestination, LocalTensor<float> source,
    TBuf<TPosition::VECCALC>& scratch, int32_t elements) {
  constexpr int32_t chunk=2048;
  static_assert(chunk*8<=kOperandScratchBytes);
  auto high=scratch.Get<bfloat16_t>();
  auto low=high[chunk];
  auto reconstructed=scratch.Get<float>()[chunk];
  for(int32_t begin=0;begin<elements;begin+=chunk) {
    const int32_t count=elements-begin<chunk?elements-begin:chunk;
    Fence<HardEvent::MTE3_V>();
    Cast(high,source[begin],RoundMode::CAST_RINT,count);
    PipeBarrier<PIPE_V>();
    Cast(reconstructed,high,RoundMode::CAST_NONE,count);
    PipeBarrier<PIPE_V>();
    Sub(reconstructed,source[begin],reconstructed,count);
    PipeBarrier<PIPE_V>();
    Cast(low,reconstructed,RoundMode::CAST_RINT,count);
    Fence<HardEvent::V_MTE3>();
    DataCopy(destination[begin],high,count);
    DataCopy(residualDestination[begin],low,count);
    Fence<HardEvent::MTE3_V>();
  }
}

// Every nonempty reader starts with BF16 QK. Allocate only that initial type;
// do not allocate FP32 merely to End/Reset it before the first multiplication.
// A later inverse rotation uses the unchanged single-live-type transition.
template<class Bf16Matmul>
__aicore__ inline void InitMatmul(bool& historyActive, TPipe& pipe,
    Bf16Matmul& bf16) {
  bf16.Init(static_cast<const TCubeTiling*>(nullptr), &pipe);
  historyActive = true;
  SetHF32Mode(false);
}

// The old MDL float Matmul alone reserves 395264B L1 and all double-buffered
// L0A/L0B. Initializing both dtypes against one live TPipe would overallocate.
// At a dtype transition, the prior IterateAll has End'ed; drain all local
// pipelines, reset the allocator/events, then initialize only the new type.
// No AIV TPipe is reset, and no cross-core protocol flag is changed here.
// All QK/PV calls request true; only final inverse Rv requests false.
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
