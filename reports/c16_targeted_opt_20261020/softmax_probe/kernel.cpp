// SPDX-License-Identifier: Apache-2.0
// Archive #126/#145/#148/#164; D.4 complete log reread.
// 1. Test only the measured fused-history softmax finite-check candidate.
// 2. Old full-history dequant6499.8-6655.1ms/95.4%, store209-215ms must not return.
// 3. Fixed32x256 score tile, old SDK ReduceSum and production helper; no history
//    materialization, changed reduction tree, lowered precision or skipped row.
// 4. Functional probe only: no timing, device/model performance claim.
#include "../../../csrc/kernels/attention_c16_softmax_helpers.h"

using namespace AscendC;
using oscar_ascend_device::Fence;
namespace {
constexpr int32_t kRows=32,kColumns=256,kScoreCount=kRows*kColumns;
constexpr int32_t kGuard=8,kStorage=kScoreCount+2*kGuard;
constexpr float kCanary=123.25F;
class Probe {
 public:
  __aicore__ void Run(GM_ADDR input,GM_ADDR result,GM_ADDR after,int32_t rows) {
    if(GetBlockIdx()!=0)return;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input));
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(result));
    copy.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(after));
    pipe.InitBuffer(scoreBuf,kStorage*sizeof(float));
    pipe.InitBuffer(resultBuf,(kRows*8+2*kGuard)*sizeof(float));
    pipe.InitBuffer(workBuf,128*sizeof(float));
    auto storage=scoreBuf.Get<float>();
    auto slots=resultBuf.Get<float>();
    auto scores=storage[kGuard];
    auto sums=slots[kGuard];
    auto tmp=workBuf.Get<float>();
    Duplicate(storage,kCanary,kStorage);
    Duplicate(slots,kCanary,kRows*8+2*kGuard);
    Fence<HardEvent::V_MTE2>();
    DataCopy(scores,src,kScoreCount);
    Fence<HardEvent::MTE2_V>();
    for(int32_t r=0;r<rows;++r) {
      PipeBarrier<PIPE_V>();
      ReduceSum(sums[r*8],scores[r*kColumns],tmp[16],kColumns);
    }
    Fence<HardEvent::V_S>();
    bool baseline=true;
    for(int32_t r=0;r<rows;++r)
      if(!oscar_ascend_device::Finite(sums.GetValue(r*8)))baseline=false;
    Fence<HardEvent::S_V>();
    const bool candidate=oscar_c16_softmax::FiniteRows256(sums,scores,rows);
    Fence<HardEvent::V_S>();
    int32_t guards=0;
    for(int32_t i=0;i<kGuard;++i) {
      if(storage.GetValue(i)!=kCanary || storage.GetValue(kGuard+kScoreCount+i)!=kCanary)++guards;
      if(slots.GetValue(i)!=kCanary || slots.GetValue(kGuard+kRows*8+i)!=kCanary)++guards;
    }
    Fence<HardEvent::S_MTE3>();
    Fence<HardEvent::V_MTE3>();
    DataCopy(copy,storage,kStorage);
    Fence<HardEvent::MTE3_S>();
    dst.SetValue(0,baseline?1:0);dst.SetValue(1,candidate?1:0);
    dst.SetValue(2,guards);dst.SetValue(3,rows);
    DataCacheCleanAndInvalid<int64_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(dst);
  }
 private:
  TPipe pipe;
  TBuf<TPosition::VECCALC> scoreBuf,resultBuf,workBuf;
  GlobalTensor<float> src,copy;
  GlobalTensor<int64_t> dst;
};
}  // namespace
extern "C" __global__ __aicore__ void oscar_c16_finite_probe_kernel(
    GM_ADDR input,GM_ADDR result,GM_ADDR after,int32_t rows) {
  Probe op;op.Run(input,result,after,rows);
}
