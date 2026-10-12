// Shared candidate helper; target NPU/graph qualification pending. Archive G26-G34/#13-22/#70-73/#126/#154 and startup D.4.
// 1. Rotation metadata setup preceding new-KV quantization (phase1_stores).
// 2. D.4 writes cost 208.6-216ms/16K; never restore historical K/V here.
// 3. Replace 2*D*log2(D) scalar UB writes by bounded vector operations.
//    Exact indices and +/-1 FP32 signs only: rotation arithmetic is unchanged.
//    Reuse input/work/red before Load; no extra persistent memory or launch.
// 4. Instruction reduction is not a timing claim; whole-store A/B is needed.
// A2 precedents: native moe_custom_sort_one_core.h ArithProgression<int32_t>
// and in-place int32->float Cast; existing striped helper int16 And/Or path.
#pragma once
#include "oscar_common.h"

namespace oscar_hadamard_indices_trial {
using namespace AscendC;
using oscar_ascend_device::Fence;

__aicore__ inline void Build(LocalTensor<uint32_t> indices,
    LocalTensor<float> signs, LocalTensor<int32_t> base,
    LocalTensor<int32_t> mask, LocalTensor<int32_t> temporary, int32_t dim) {
  ArithProgression<int32_t>(base, 0, 4, dim);
  PipeBarrier<PIPE_V>();
  int32_t stage=0;
  for(int32_t distance=1;distance<dim;distance*=2,++stage) {
    const int32_t byteDistance=4*distance;
    Duplicate(mask,byteDistance,dim);
    PipeBarrier<PIPE_V>();
    auto a=base.ReinterpretCast<int16_t>();
    auto b=mask.ReinterpretCast<int16_t>();
    auto t=temporary.ReinterpretCast<int16_t>();
    auto out=indices[stage*dim].ReinterpretCast<int16_t>();
    auto sign=signs[stage*dim];
    // XOR=(a|b)&~(a&b), also valid for the zero high half of each uint32.
    And(t,a,b,2*dim);
    Or(out,a,b,2*dim);
    PipeBarrier<PIPE_V>();
    Cast(sign,temporary,RoundMode::CAST_NONE,dim);
    // The Cast must finish reading temporary before Not overwrites it.
    PipeBarrier<PIPE_V>();
    Not(t,t,2*dim);
    PipeBarrier<PIPE_V>();
    And(out,out,t,2*dim);
    // temporary is either zero or exactly byteDistance (a power of two).
    // These operations produce exact +1/-1, without touching any KV value.
    Muls(sign,sign,-2.0F/static_cast<float>(byteDistance),dim);
    PipeBarrier<PIPE_V>();
    Adds(sign,sign,1.0F,dim);
    PipeBarrier<PIPE_V>();
  }
}
} // namespace oscar_hadamard_indices_trial
