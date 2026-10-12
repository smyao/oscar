// Independent revision2: append readback only; v1 kernel.cpp remains unchanged.
// SPDX-License-Identifier: Apache-2.0
// Archive #126/#140/#145/#154; startup D.4 four questions:
// (1) Isolate FP32 PV writeback plus alpha-scaled state addition inside FIA.
// (2) D.4's 6499.8-6655.1ms full-history restore is not introduced here.
// (3) Exact current C4 static Matmul/HF32-off; paired paths share P/V. Only
//     the last FP32 addition moves from Vector Add to complete-K Fixpipe atomic.
//     Different results falsify equivalence; they are never tolerance-hidden.
// (4) This is a legality probe, not an optimization or a speedup claim.
#include "../../kernels/oscar_common.h"
#include "lib/matmul_intf.h"
#include "lib/matmul/constant_tiling.h"
using namespace oscar_ascend_device;
namespace {
constexpr int32_t kRows=126,kPaddedRows=128,kCols=256,kK=256,kLaneRows=64,kReadRows=16;
constexpr uint16_t kInitialized=8,kNormalReady=9,kReferenceDone=10,kAtomicDone=11,kVectorsDone=12;
using Matrix=MatmulType<TPosition::GM,CubeFormat::ND,float,false>;
using TransposedMatrix=MatmulType<TPosition::GM,CubeFormat::ND,float,true>;
__aicore__ constexpr MatmulConfig CvConfig() {
  auto cfg=GetNormalConfig();
  cfg.basicM=64;cfg.basicN=64;cfg.basicK=128;
  cfg.singleCoreM=128;cfg.singleCoreN=256;cfg.singleCoreK=256;
  cfg.enableSetBias=false;
  return cfg;
}
constexpr auto kCubeConfig=GetMatmulApiTiling<Matrix,TransposedMatrix,Matrix,Matrix>(CvConfig());
using CubeMatmul=matmul::MatmulImpl<Matrix,TransposedMatrix,Matrix,Matrix,kCubeConfig>;

class Probe {
 public:
  __aicore__ void Init(GM_ADDR p,GM_ADDR v,GM_ADDR alpha,GM_ADDR pv,
                      GM_ADDR reference,GM_ADDR atomic,GM_ADDR status,GM_ADDR atomic_readback) {
    p_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(p));
    v_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(v));
    alpha_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(alpha));
    pv_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pv));
    reference_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(reference));
    atomic_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(atomic));
    readback_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(atomic_readback));
    status_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(status));
    if ASCEND_IS_AIC {
      mm_.Init(static_cast<const TCubeTiling*>(nullptr),&pipe_);
    } else {
      lane_=GetBlockIdx()%2;
      pipe_.InitBuffer(accBuf_,kLaneRows*kCols*4);
      pipe_.InitBuffer(pvBuf_,kReadRows*kCols*4);
      pipe_.InitBuffer(statusBuf_,64);
    }
  }
  __aicore__ void Run() {
    if ASCEND_IS_AIC {
      CrossCoreWaitFlag(kInitialized);
      Matmul(pv_,0);
      CrossCoreSetFlag<2,PIPE_FIX>(kNormalReady);
      CrossCoreWaitFlag(kReferenceDone);
      Matmul(atomic_,1);
      CrossCoreSetFlag<2,PIPE_FIX>(kAtomicDone);
      CrossCoreWaitFlag(kVectorsDone);
      for(int32_t field=0;field<16;++field)status_.SetValue(field,StatusValue(0,field));
      DataCacheCleanAndInvalid<int32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(status_);
    } else {
      auto acc=accBuf_.Get<float>();auto result=pvBuf_.Get<float>();
      const int32_t first=lane_*kLaneRows;
      DataCopy(acc,alpha_[first*kCols],kLaneRows*kCols);
      // Native A2 direct DMA handoff: no arithmetic modifies signed zero/NaN.
      Fence<HardEvent::MTE2_MTE3>();
      DataCopy(reference_[first*kCols],acc,kLaneRows*kCols);
      DataCopy(atomic_[first*kCols],acc,kLaneRows*kCols);
      Fence<HardEvent::MTE3_V>();
      CrossCoreSetFlag<2,PIPE_MTE3>(kInitialized);
      CrossCoreWaitFlag(kNormalReady);
      // Same DataCopy/MTE2_V/Add/PIPE_V/V_MTE2 sequence as C4 VectorCluster.
      for(int32_t row=first;row<first+kLaneRows && row<kRows;row+=kReadRows) {
        const int32_t count=(kRows-row<kReadRows)?kRows-row:kReadRows;
        DataCopy(result,pv_[row*kCols],count*kCols);
        Fence<HardEvent::MTE2_V>();
        auto target=acc[(row-first)*kCols];
        Add(target,target,result,count*kCols);PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_MTE2>();
      }
      Fence<HardEvent::V_MTE3>();
      DataCopy(reference_[first*kCols],acc,kLaneRows*kCols);
      Fence<HardEvent::MTE3_V>();
      CrossCoreSetFlag<2,PIPE_MTE3>(kReferenceDone);
      CrossCoreWaitFlag(kAtomicDone);
      // Revision2: verify immediate cross-core MTE2 visibility after PIPE_FIX.
      // acc's reference writeback completed before ReferenceDone above.
      DataCopy(acc,atomic_[first*kCols],kLaneRows*kCols);
      Fence<HardEvent::MTE2_MTE3>();
      DataCopy(readback_[first*kCols],acc,kLaneRows*kCols);
      auto words=statusBuf_.Get<int32_t>();
      for(int32_t field=0;field<16;++field)words.SetValue(field,StatusValue(lane_+1,field));
      Fence<HardEvent::S_MTE3>();
      DataCopy(status_[(lane_+1)*16],words,16);
      CrossCoreSetFlag<2,PIPE_MTE3>(kVectorsDone);
    }
  }
 private:
  __aicore__ int32_t StatusValue(int32_t actor,int32_t field) {
    if(field==0)return 1;
    if(field==1)return actor;
    if(field==2)return kRows;
    if(field==3)return kCols;
    if(field==4)return kK;
    return 0;
  }
  __aicore__ void Matmul(const GlobalTensor<float>& output,uint8_t atomic) {
    SetHF32Mode(false);
    mm_.SetOrgShape(kRows,kCols,kK);mm_.SetSingleShape(kRows,kCols,kK);
    mm_.SetTensorA(p_,false);mm_.SetTensorB(v_,false);
    // GetNormalConfig has no partial/intrablock output. SDK completes KLoop
    // before GetTensorC applies this atomic flag to final-C copy-out.
    mm_.template IterateAll<true>(output,atomic);
    mm_.End();
  }
  TPipe pipe_;CubeMatmul mm_;
  GlobalTensor<float> p_,v_,alpha_,pv_,reference_,atomic_,readback_;
  GlobalTensor<int32_t> status_;
  TBuf<TPosition::VECCALC> accBuf_,pvBuf_,statusBuf_;
  int32_t lane_=0;
};
}
extern "C" __global__ __aicore__ void oscar_c4_atomic_state_readback_kernel(
    GM_ADDR p,GM_ADDR v,GM_ADDR alpha,GM_ADDR pv,GM_ADDR reference,GM_ADDR atomic,GM_ADDR status,GM_ADDR atomic_readback) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  if(GetBlockNum()!=1)return;
  Probe probe;probe.Init(p,v,alpha,pv,reference,atomic,status,atomic_readback);probe.Run();
}
