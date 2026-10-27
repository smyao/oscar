// SPDX-License-Identifier: Apache-2.0
// Archive #126/#145/#148/#164, startup D.4 original logs reviewed.
// D.4: fixed fused-attention finite check; no history materialization or stores.
// Compare original row vcadd with the sufficient certificate on unchanged scores.
// Bounded score/scratch guards, no new production UB, no timing or NPU claim.
#include "../../kernels/attention_official_finite.h"

using namespace AscendC;
using oscar_ascend_device::Fence;
namespace {
constexpr int32_t kRows=32,kColumns=256,kCount=kRows*kColumns,kGuard=8;
constexpr float kCanary=123.25F;
class Probe {
 public:
  __aicore__ void Run(GM_ADDR input,GM_ADDR result,GM_ADDR after,int32_t rows) {
    if(GetBlockIdx()!=0)return;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input));
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(result));
    copy.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(after));
    pipe.InitBuffer(scoreBuf,(kCount+2*kGuard)*sizeof(float));
    pipe.InitBuffer(resultBuf,(kRows*8+2*kGuard)*sizeof(float));
    pipe.InitBuffer(workBuf,(kCount+2*kGuard)*sizeof(float));
    auto scoreStorage=scoreBuf.Get<float>();
    auto resultStorage=resultBuf.Get<float>();
    auto workStorage=workBuf.Get<float>();
    auto scores=scoreStorage[kGuard];
    auto sums=resultStorage[kGuard];
    auto work=workStorage[kGuard];
    Duplicate(scoreStorage,kCanary,kCount+2*kGuard);
    Duplicate(resultStorage,kCanary,kRows*8+2*kGuard);
    Duplicate(workStorage,kCanary,kCount+2*kGuard);
    Fence<HardEvent::V_MTE2>();
    DataCopy(scores,src,kCount);
    Fence<HardEvent::MTE2_V>();
    const bool baseline=oscar_c16_softmax::FiniteRows256(sums,scores,rows);
    Fence<HardEvent::S_V>();
    const bool candidate=oscar_official_finite::FiniteRows256(sums,scores,work,rows);
    Fence<HardEvent::V_S>();
    int32_t guards=0;
    for(int32_t i=0;i<kGuard;++i) {
      if(scoreStorage.GetValue(i)!=kCanary || scoreStorage.GetValue(kGuard+kCount+i)!=kCanary)++guards;
      if(resultStorage.GetValue(i)!=kCanary || resultStorage.GetValue(kGuard+kRows*8+i)!=kCanary)++guards;
      if(workStorage.GetValue(i)!=kCanary || workStorage.GetValue(kGuard+kCount+i)!=kCanary)++guards;
    }
    const int32_t written=rows>=8?rows*kColumns:0;
    for(int32_t i=written;i<kCount;++i)if(work.GetValue(i)!=kCanary)++guards;
    Fence<HardEvent::S_MTE3>();
    Fence<HardEvent::V_MTE3>();
    DataCopy(copy,scoreStorage,kCount+2*kGuard);
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
}
extern "C" __global__ __aicore__ void oscar_official_finite_probe_kernel(
    GM_ADDR input,GM_ADDR result,GM_ADDR after,int32_t rows) {
  Probe op;op.Run(input,result,after,rows);
}
