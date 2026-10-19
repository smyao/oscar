// SPDX-License-Identifier: Apache-2.0
// Archive #13/#126/#145/#148/#154/#155; startup section0 and full D.4 read.
// D.4 four questions: (1) exact sink/recent/current validation within fused
// attention; (2) no whole-history restore (old D.4 6499.8-6655.1ms/card);
// (3) four-row BF16->FP32 Cast in existing dead natural unpack scratch,
// retaining each original D-element ReduceSum, finite check and error order;
// (4) no extra UB/GM: 4KiB reused at D256, same converted elements and
// reductions, fewer Cast launches/PIPE_V barriers for contiguous live runs.
// This instruction-count bound is not a measured NPU latency improvement.
#pragma once
#include "oscar_common.h"

namespace oscar_bf16_precise_check {
using namespace AscendC;
using oscar_ascend_device::Fence;
using oscar_ascend_device::Finite;
constexpr int32_t kRows = 4;

// naturalBuf is dead between source0 unpack/online-alpha operations and
// precise K/V checks; it is not the BF16 operand or the reduction workspace.
// Caller retains MTE2_V before this call and all tag/MTE2 ownership fences.
// A completed CheckRow releases its scalar check read before the next tile.
template<int32_t D, int32_t NaturalRows>
__aicore__ inline LocalTensor<float> CastTile(LocalTensor<bfloat16_t> source,
    int32_t rows, TBuf<TPosition::VECCALC>& naturalBuf) {
  static_assert(D==256,"precise BF16 checks require the audited D256 route");
  static_assert(kRows*D*4<=NaturalRows*(D+16)*2,"natural scratch too small");
  auto converted=naturalBuf.Get<float>();
  // rows is min(remaining,kRows), strictly positive at both callers below.
  Cast(converted,source,RoundMode::CAST_NONE,rows*D);
  PipeBarrier<PIPE_V>();
  return converted;
}

template<int32_t D>
__aicore__ inline void CheckRow(LocalTensor<float> row,
    LocalTensor<float> check,int32_t& error) {
  // Do not substitute an exponent scan: a finite BF16 row can have an
  // overflowing FP32 sum. Preserve the original ReduceSum API and ordering.
  ReduceSum(check,row,check[16],D);Fence<HardEvent::V_S>();
  if(!Finite(check.GetValue(0)))error=2;
  Fence<HardEvent::S_V>();
}

template<int32_t D,int32_t NaturalRows>
__aicore__ inline void CheckRows(LocalTensor<bfloat16_t> source,int32_t rows,
    TBuf<TPosition::VECCALC>& naturalBuf,
    TBuf<TPosition::VECCALC>& scratchBuf,int32_t& error) {
  auto check=scratchBuf.Get<float>();
  for(int32_t begin=0;begin<rows;begin+=kRows) {
    const int32_t count=rows-begin<kRows?rows-begin:kRows;
    auto converted=CastTile<D,NaturalRows>(source[begin*D],count,naturalBuf);
    for(int32_t row=0;row<count;++row)CheckRow<D>(converted[row*D],check,error);
  }
}

template<int32_t D,int32_t NaturalRows>
__aicore__ inline void CheckTaggedRows(LocalTensor<bfloat16_t> source,
    LocalTensor<int64_t> tags,int64_t inpage,int32_t rows,
    TBuf<TPosition::VECCALC>& naturalBuf,
    TBuf<TPosition::VECCALC>& scratchBuf,int32_t& error) {
  auto check=scratchBuf.Get<float>();
  for(int32_t begin=0;begin<rows;begin+=kRows) {
    const int32_t count=rows-begin<kRows?rows-begin:kRows;
    auto converted=CastTile<D,NaturalRows>(source[begin*D],count,naturalBuf);
    // Invalid tags had no K/V DMA and retain the caller's initialized +0.
    // Publish errors only here, in precisely the old tag->finite row order.
    for(int32_t row=0;row<count;++row) {
      if(tags.GetValue(begin+row)!=inpage+begin+row) {error=4;continue;}
      CheckRow<D>(converted[row*D],check,error);
    }
  }
}
}  // namespace oscar_bf16_precise_check
