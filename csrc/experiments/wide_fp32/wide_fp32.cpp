// SPDX-License-Identifier: Apache-2.0
// EXPERIMENT ONLY. Archive #126/#129/#145/#148-155 and startup D.4 four questions:
// (1) One fused FP32 history-attention query group M=126, D256, bounded B512/
//     B1024 consumes already-precise rotated K/V produced from real INT2.
// (2) D.4 old full-history dequant took 6499.8-6655.1ms/card vs FIA18.7ms;
//     this single bounded block never creates [history,D] storage or host D2H.
// (3) QK writes GM[M,B]; AIV checks raw scores in visible KV256 tiles before mask, scans 256
//     at a time for final per-row max, then rereads/reweights P into that same
//     GM plane. AIC performs one FP32 PV and one final FP32 Rv^T. Cross-core
//     flags and MTE3->MTE2/V buffer reuse handoffs are explicit (#126/#145).
// (4) One API QK/PV call does not reduce FP32 MMAD FLOPs or prove speed.
//     Frozen oracle/CPU and real NPU checks must decide whether this survives.
// Controlled-input prototype only: grid=1 MIX AIC1/AIV2; all B rows must be
// valid store-reachable finite K/V. No producer metadata errors, private task
// tail, window/current source, multi-block online state, or production route.
// Finite inputs alone do not establish equivalence to the old error domain.
#include "../../kernels/oscar_common.h"
#include "lib/matmul_intf.h"
#include "lib/matmul/constant_tiling.h"

using namespace oscar_ascend_device;

namespace {
constexpr int32_t kM=126,kD=256,kLaneRows=64,kSubtile=256;
constexpr float kNegInf=-__builtin_inff();
constexpr uint16_t kScoreReady=8,kPReady=9,kPvReady=10,kNormReady=11,
                   kRotReady=12,kDone=13;
using Matrix=MatmulType<TPosition::GM,CubeFormat::ND,float,false>;
using TransposedMatrix=MatmulType<TPosition::GM,CubeFormat::ND,float,true>;

template<int B> __aicore__ constexpr MatmulConfig Config() {
  auto cfg=GetNormalConfig();
  cfg.basicM=64;cfg.basicN=64;cfg.basicK=128;
  cfg.singleCoreM=128;cfg.singleCoreN=B;cfg.singleCoreK=B;
  cfg.enableSetBias=false;
  return cfg;
}
template<int B>
constexpr auto kTiling=GetMatmulApiTiling<Matrix,TransposedMatrix,Matrix,Matrix>(Config<B>());
template<int B>
using WideMatmul=matmul::MatmulImpl<Matrix,TransposedMatrix,Matrix,Matrix,kTiling<B>>;

template<int B> class Wide {
 public:
  __aicore__ void Init(GM_ADDR q,GM_ADDR k,GM_ADDR v,GM_ADDR rotation,
      GM_ADDR visibleEnd,GM_ADDR workspace,GM_ADDR output,GM_ADDR lse,
      GM_ADDR status,float scale) {
    q_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(q));
    k_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(k));
    v_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(v));
    rotation_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rotation));
    visible_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(visibleEnd));
    work_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
    lse_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lse));
    status_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(status));
    scale_=scale;
    if ASCEND_IS_AIC {
      mm_.Init(static_cast<const TCubeTiling*>(nullptr),&pipe_);
      SetHF32Mode(false);
    } else {
      lane_=GetBlockIdx()%2;
      pipe_.InitBuffer(scoreBuf_,kSubtile*4);
      pipe_.InitBuffer(reduceBuf_,kSubtile*4);
      pipe_.InitBuffer(scalarBuf_,32);
      pipe_.InitBuffer(alphaBuf_,32);
      pipe_.InitBuffer(logBuf_,32);
      pipe_.InitBuffer(pvBuf_,kD*4);
      pipe_.InitBuffer(visibleBuf_,kLaneRows*4);
      pipe_.InitBuffer(maxBuf_,kLaneRows*4);
      pipe_.InitBuffer(sumBuf_,kLaneRows*4);
      pipe_.InitBuffer(errorBuf_,kLaneRows*4);
      pipe_.InitBuffer(lseBuf_,kLaneRows*4);
    }
  }

  __aicore__ void Process() {
    // This isolated prototype owns exactly one Cube and its two AIV lanes.
    // All three branches make the same grid decision before any flag wait.
    if(GetBlockNum()!=1) {
      // No numerical result is defined for an unsupported launch. A single
      // owner publishes explicit status=1; all actors avoid cross-core waits.
      if ASCEND_IS_AIV {
        if(GetBlockIdx()==0)for(int32_t r=0;r<kM;++r)status_.SetValue(r,1);
      }
      return;
    }
    if ASCEND_IS_AIC {
      Matmul(q_,k_,work_[0],kM,B,kD,true);
      CrossCoreSetFlag<2,PIPE_FIX>(kScoreReady);
      CrossCoreWaitFlag(kPReady);
      Matmul(work_[0],v_,work_[kM*B],kM,kD,B,false);
      CrossCoreSetFlag<2,PIPE_FIX>(kPvReady);
      CrossCoreWaitFlag(kNormReady);
      Matmul(work_[kM*B],rotation_,output_,kM,kD,kD,true);
      CrossCoreSetFlag<2,PIPE_FIX>(kRotReady);
      CrossCoreWaitFlag(kDone);
      mm_.End();
    } else {
      auto scores=scoreBuf_.Get<float>();
      auto reduce=reduceBuf_.Get<float>();
      auto scalar=scalarBuf_.Get<float>();
      auto alpha=alphaBuf_.Get<float>();
      auto log=logBuf_.Get<float>();
      auto pv=pvBuf_.Get<float>();
      auto ends=visibleBuf_.Get<int32_t>();
      auto maxima=maxBuf_.Get<float>();
      auto totals=sumBuf_.Get<float>();
      auto errors=errorBuf_.Get<int32_t>();
      auto lses=lseBuf_.Get<float>();
      const int32_t first=lane_*kLaneRows;
      const int32_t rows=first+kLaneRows<kM?kLaneRows:kM-first;
      DataCopyExtParams boundsCopy{1,static_cast<uint32_t>(rows*4),0,0,0};
      DataCopyPadExtParams<int32_t> boundsPad{false,0,0,0};
      DataCopyPad(ends,visible_[first],boundsCopy,boundsPad);
      Fence<HardEvent::MTE2_S>();
      CrossCoreWaitFlag(kScoreReady);
      // First score read: check a KV256 tile before its elementwise causal
      // mask, but skip tiles that are wholly invisible to this query row.
      for(int32_t r=0;r<rows;++r) {
        const int32_t visible=ends.GetValue(r);
        int32_t error=visible<0||visible>B?1:0;
        // Preserve the original QR finite check independently of whether
        // this row has a visible KV tile.
        DataCopy(pv,q_[(first+r)*kD],kD);
        Fence<HardEvent::MTE2_V>();
        ReduceSum(scalar,pv,reduce,kD);Fence<HardEvent::V_S>();
        if(!Finite(scalar.GetValue(0)))error=2;
        Fence<HardEvent::V_MTE2>();
        float blockMax=-__FLT_MAX__;
        for(int32_t start=0;start<B;start+=kSubtile) {
          if(visible<=start || visible>B)continue;
          DataCopy(scores,work_[(first+r)*B+start],kSubtile);
          Fence<HardEvent::MTE2_V>();
          Muls(scores,scores,scale_,kSubtile);PipeBarrier<PIPE_V>();
          ReduceSum(scalar,scores,reduce,kSubtile);Fence<HardEvent::V_S>();
          if(!Finite(scalar.GetValue(0)))error=2;
          if(visible>start) {
            const int32_t last=visible-start<kSubtile?visible-start:kSubtile;
            if(last<kSubtile) {
              for(int32_t j=last;j<kSubtile;++j)scores.SetValue(j,kNegInf);
              Fence<HardEvent::S_V>();
            }
            ReduceMax(scalar,scores,reduce,kSubtile);Fence<HardEvent::V_S>();
            const float local=scalar.GetValue(0);
            if(local>blockMax)blockMax=local;
          }
          Fence<HardEvent::V_MTE2>();
        }
        maxima.SetValue(r,blockMax);
        errors.SetValue(r,error);
      }
      // Second score read: exact FP32 P for each 256 section is rescaled to
      // the one final max for the whole B. The old raw score GM is now dead.
      for(int32_t r=0;r<rows;++r) {
        const int32_t visible=ends.GetValue(r);
        const float blockMax=maxima.GetValue(r);
        float sum=0.0F;
        for(int32_t start=0;start<B;start+=kSubtile) {
          if(visible>start && errors.GetValue(r)!=1) {
            DataCopy(scores,work_[(first+r)*B+start],kSubtile);
            Fence<HardEvent::MTE2_V>();
            Muls(scores,scores,scale_,kSubtile);PipeBarrier<PIPE_V>();
            const int32_t last=visible-start<kSubtile?visible-start:kSubtile;
            if(last<kSubtile) {
              Fence<HardEvent::V_S>();
              for(int32_t j=last;j<kSubtile;++j)scores.SetValue(j,kNegInf);
              Fence<HardEvent::S_V>();
            }
            ReduceMax(scalar,scores,reduce,kSubtile);Fence<HardEvent::V_S>();
            const float localMax=scalar.GetValue(0);
            Adds(scores,scores,-localMax,kSubtile);PipeBarrier<PIPE_V>();
            Exp(scores,scores,kSubtile);PipeBarrier<PIPE_V>();
            alpha.SetValue(0,localMax-blockMax);
            Fence<HardEvent::S_V>();
            Exp(alpha,alpha,1);Fence<HardEvent::V_S>();
            Muls(scores,scores,alpha.GetValue(0),kSubtile);PipeBarrier<PIPE_V>();
            ReduceSum(scalar,scores,reduce,kSubtile);Fence<HardEvent::V_S>();
            sum+=scalar.GetValue(0);
          } else {
            Duplicate(scores,0.0F,kSubtile);PipeBarrier<PIPE_V>();
          }
          Fence<HardEvent::V_MTE3>();
          DataCopy(work_[(first+r)*B+start],scores,kSubtile);
          Fence<HardEvent::MTE3_MTE2>();
          Fence<HardEvent::MTE3_V>();
        }
        if((visible>0 && (!(sum>0.0F)||!Finite(sum))) &&
            errors.GetValue(r)==0)errors.SetValue(r,2);
        totals.SetValue(r,sum);
      }
      CrossCoreSetFlag<2,PIPE_MTE3>(kPReady);
      CrossCoreWaitFlag(kPvReady);
      for(int32_t r=0;r<rows;++r) {
        const int32_t visible=ends.GetValue(r);
        const float sum=totals.GetValue(r);
        DataCopy(pv,work_[kM*B+(first+r)*kD],kD);
        Fence<HardEvent::MTE2_V>();
        if(errors.GetValue(r)!=0) Duplicate(pv,__builtin_nanf(""),kD);
        else if(visible<=0) Duplicate(pv,0.0F,kD);
        else Muls(pv,pv,1.0F/sum,kD);
        Fence<HardEvent::V_MTE3>();
        DataCopy(work_[kM*B+(first+r)*kD],pv,kD);
        Fence<HardEvent::MTE3_MTE2>();
        Fence<HardEvent::MTE3_V>();
      }
      CrossCoreSetFlag<2,PIPE_MTE3>(kNormReady);
      CrossCoreWaitFlag(kRotReady);
      for(int32_t r=0;r<rows;++r) {
        const int32_t visible=ends.GetValue(r);
        const float sum=totals.GetValue(r);
        int32_t error=errors.GetValue(r);
        if(error==0 && visible>0) {
          DataCopy(pv,output_[(first+r)*kD],kD);
          Fence<HardEvent::MTE2_V>();
          ReduceSum(scalar,pv,reduce,kD);Fence<HardEvent::V_S>();
          if(!Finite(scalar.GetValue(0)))error=2;
          Fence<HardEvent::V_MTE2>();
        }
        float lse=kNegInf;
        if(error==0 && visible>0) {
          scalar.SetValue(0,sum);Fence<HardEvent::S_V>();
          Log(log,scalar,1);Fence<HardEvent::V_S>();
          lse=maxima.GetValue(r)+log.GetValue(0);
        }
        if(error!=0) {
          Duplicate(pv,__builtin_nanf(""),kD);
          Fence<HardEvent::V_MTE3>();
          DataCopy(output_[(first+r)*kD],pv,kD);
          Fence<HardEvent::MTE3_MTE2>();
          Fence<HardEvent::MTE3_V>();
          lse=__builtin_nanf("");
        }
        lses.SetValue(r,lse);
        errors.SetValue(r,error);
      }
      Fence<HardEvent::S_MTE3>();
      DataCopyExtParams scalarCopy{1,static_cast<uint32_t>(rows*4),0,0,0};
      DataCopyPad(lse_[first],lses,scalarCopy);
      DataCopyPad(status_[first],errors,scalarCopy);
      Fence<HardEvent::MTE3_S>();
      CrossCoreSetFlag<2,PIPE_MTE3>(kDone);
    }
  }
 private:
  __aicore__ void Matmul(GlobalTensor<float> a,GlobalTensor<float> b,
      GlobalTensor<float> c,int32_t m,int32_t n,int32_t k,bool transposeB) {
    SetHF32Mode(false);
    mm_.SetOrgShape(m,n,k);mm_.SetSingleShape(m,n,k);
    mm_.SetTensorA(a,false);mm_.SetTensorB(b,transposeB);
    mm_.template IterateAll<true>(c);mm_.End();
  }
  TPipe pipe_;
  WideMatmul<B> mm_;
  GlobalTensor<float> q_,k_,v_,rotation_,work_,output_,lse_;
  GlobalTensor<int32_t> visible_,status_;
  TBuf<TPosition::VECCALC> scoreBuf_,reduceBuf_,scalarBuf_,alphaBuf_,
      logBuf_,pvBuf_,visibleBuf_,maxBuf_,sumBuf_,errorBuf_,lseBuf_;
  int32_t lane_=0;
  float scale_;
};
}  // namespace

extern "C" __global__ __aicore__ void oscar_wide_fp32_b512_kernel(
    GM_ADDR q,GM_ADDR k,GM_ADDR v,GM_ADDR rotation,GM_ADDR visibleEnd,
    GM_ADDR workspace,GM_ADDR output,GM_ADDR lse,GM_ADDR status,float scale) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  Wide<512> op;
  op.Init(q,k,v,rotation,visibleEnd,workspace,output,lse,status,scale);
  op.Process();
}
extern "C" __global__ __aicore__ void oscar_wide_fp32_b1024_kernel(
    GM_ADDR q,GM_ADDR k,GM_ADDR v,GM_ADDR rotation,GM_ADDR visibleEnd,
    GM_ADDR workspace,GM_ADDR output,GM_ADDR lse,GM_ADDR status,float scale) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  Wide<1024> op;
  op.Init(q,k,v,rotation,visibleEnd,workspace,output,lse,status,scale);
  op.Process();
}
