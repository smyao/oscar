// SPDX-License-Identifier: Apache-2.0
// KV256 BASELINE, EXPERIMENT ONLY. Archive #126/#129/#140/#145/#148/#154/#155.
// D.4 four questions: (1) bounded history-CV QK/P/FP32-PV/final RvT;
// (2) old full-history dequant cost 6499.8-6655.1ms/card vs FIA18.5-18.9ms;
// (3) this reference math pipeline keeps original KV256 order, 32-row V2,
// UB online FP32 accumulator and Brcb alpha, then one final RvT. Decoded
// K/V input excludes the INT2 producer on BOTH sides of the A/B comparison.
// (4) its same-core comparison with wide tests bounded math cost only, never
// producer/task/source scheduling, model quality or whole-service speed.
// M126/grid1 MIX AIC1/AIV2 controlled-input experiment; all B K/V rows must
// be finite store-reachable decoded values. Production task tails, producer
// metadata/status, window/current and multi-block online state are absent.
// HF32 is disabled. No production CMake, routing, variants or GDN changes.
#include "../../kernels/oscar_common.h"
#include "lib/matmul_intf.h"
#include "lib/matmul/constant_tiling.h"
#include "adv_api/activation/softmaxflashv2.h"

using namespace oscar_ascend_device;

namespace {
constexpr int32_t kM=126,kD=256,kLaneRows=64,kSubtile=256;
constexpr uint32_t kSoftBytes=16384;
constexpr SoftmaxConfig kSoftConfig={false,0,0,SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};
constexpr float kNegInf=-__builtin_inff();
constexpr uint16_t kScoreReady=8,kPReady=9,kPvReady=10,kNormReady=11,
                   kRotReady=12,kDone=13,kAccReady=14;
using Matrix=MatmulType<TPosition::GM,CubeFormat::ND,float,false>;
using TransposedMatrix=MatmulType<TPosition::GM,CubeFormat::ND,float,true>;

template<int B> __aicore__ constexpr MatmulConfig Config() {
  auto cfg=GetNormalConfig();
  cfg.basicM=64;cfg.basicN=64;cfg.basicK=128;
  cfg.singleCoreM=128;cfg.singleCoreN=256;cfg.singleCoreK=256;
  cfg.enableSetBias=false;
  return cfg;
}
template<int B>
constexpr auto kTiling=GetMatmulApiTiling<Matrix,TransposedMatrix,Matrix,Matrix>(Config<B>());
template<int B>
using WideMatmul=matmul::MatmulImpl<Matrix,TransposedMatrix,Matrix,Matrix,kTiling<B>>;

template<int B> class Wide {
  static constexpr int32_t kBatchRows=32;
  static constexpr int32_t kScoreFloats=kM*kSubtile;
 public:
  static constexpr int32_t kExplicitUbBytes=32768+16384+65536+16384+1024+1024+32+128+128+128+32+1024+5*256;
  static_assert(kExplicitUbBytes<=188416,"A2 configured vector UB budget");
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
      pipe_.InitBuffer(scoreBuf_,kBatchRows*kSubtile*4);
      pipe_.InitBuffer(accBuf_,kLaneRows*kD*4);
      pipe_.InitBuffer(pvBlockBuf_,16*kD*4);
      pipe_.InitBuffer(alphaBlocksBuf_,32*8*4);
      pipe_.InitBuffer(softBuf_,kSoftBytes);
      pipe_.InitBuffer(inMaxBuf_,32*4);
      pipe_.InitBuffer(inSumBuf_,32*4);
      pipe_.InitBuffer(reduceBuf_,kSubtile*4);
      pipe_.InitBuffer(scalarBuf_,32);
      pipe_.InitBuffer(alphaBuf_,32*4);
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
      for(int32_t start=0;start<B;start+=kSubtile) {
        Matmul(q_,k_[start*kD],work_[0],kM,kSubtile,kD,true);
        CrossCoreSetFlag<2,PIPE_FIX>(kScoreReady);
        CrossCoreWaitFlag(kPReady);
        Matmul(work_[0],v_[start*kD],work_[kScoreFloats],kM,kD,kSubtile,false);
        CrossCoreSetFlag<2,PIPE_FIX>(kPvReady);
        CrossCoreWaitFlag(kAccReady);
      }
      CrossCoreWaitFlag(kNormReady);
      Matmul(work_[kScoreFloats],rotation_,output_,kM,kD,kD,true);
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
      // QR owns pv independently of score ownership. #154: no next DMA may
      // overwrite it until the vector finite reduction has completed.
      for(int32_t r=0;r<rows;++r) {
        const int32_t visible=ends.GetValue(r);
        int32_t error=visible<0||visible>B?1:0;
        DataCopy(pv,q_[(first+r)*kD],kD);
        Fence<HardEvent::MTE2_V>();
        ReduceSum(scalar,pv,reduce,kD);Fence<HardEvent::V_S>();
        if(!Finite(scalar.GetValue(0)))error=2;
        Fence<HardEvent::V_MTE2>();
        errors.SetValue(r,error);
      }
      auto softWork=softBuf_.Get<uint8_t>();
      auto inMax=inMaxBuf_.Get<float>();
      auto inSum=inSumBuf_.Get<float>();
      auto acc=accBuf_.Get<float>();
      auto pvBlock=pvBlockBuf_.Get<float>();
      auto alphaBlocks=alphaBlocksBuf_.Get<float>();
      Duplicate(acc,0.0F,kLaneRows*kD);
      Duplicate(maxima,-__FLT_MAX__,kLaneRows);
      Duplicate(totals,0.0F,kLaneRows);
      Fence<HardEvent::V_S>();
      for(int32_t start=0;start<B;start+=kSubtile) {
        CrossCoreWaitFlag(kScoreReady);
        for(int32_t base=0;base<rows;base+=kBatchRows) {
          const int32_t active=rows-base<kBatchRows?rows-base:kBatchRows;
          DataCopy(scores,work_[(first+base)*kSubtile],active*kSubtile);
          Fence<HardEvent::MTE2_V>();
          Muls(scores,scores,scale_,active*kSubtile);
          Fence<HardEvent::V_S>();
          for(int32_t r=0;r<active;++r) {
            const int32_t visible=ends.GetValue(base+r);
            int32_t error=errors.GetValue(base+r);
            if(visible>start && visible<=B) {
              Fence<HardEvent::S_V>();
              ReduceSum(scalar,scores[r*kSubtile],reduce,kSubtile);
              Fence<HardEvent::V_S>();
              if(!Finite(scalar.GetValue(0)))error=2;
            }
            errors.SetValue(base+r,error);
            Fence<HardEvent::S_V>();
            if(error!=0 || visible<=start)Duplicate(scores[r*kSubtile],kNegInf,kSubtile);
            else if(visible<start+kSubtile) {
              const int32_t localEnd=visible-start,aligned=(localEnd+7)/8*8;
              for(int32_t j=localEnd;j<aligned;++j)scores.SetValue(r*kSubtile+j,kNegInf);
              Fence<HardEvent::S_V>();
              if(aligned<kSubtile)Duplicate(scores[r*kSubtile+aligned],kNegInf,kSubtile-aligned);
            }
            Fence<HardEvent::V_S>();
          }
          Fence<HardEvent::S_V>();
          if(active<kBatchRows)Duplicate(scores[active*kSubtile],kNegInf,(kBatchRows-active)*kSubtile);
          Adds(inMax,maxima[base],0.0F,kBatchRows);
          Adds(inSum,totals[base],0.0F,kBatchRows);
          PipeBarrier<PIPE_V>();
          const SoftMaxShapeInfo shape{32,256,32,256};
          const auto tiling=SoftMaxFlashV2TilingFunc(shape,4,4,kSoftBytes,true,false);
          SoftmaxFlashV2<float,true,true,false,false,kSoftConfig>(
              scores,totals[base],maxima[base],scores,alpha,inSum,inMax,
              softWork,tiling,shape);
          Fence<HardEvent::V_S>();
          // Original attention_cv.cpp: retain FP32 UB online state and the
          // same Brcb/Mul order before adding this tile's FP32 PV.
          Brcb(alphaBlocks,alpha,static_cast<uint8_t>((active+7)/8),{1,8});
          PipeBarrier<PIPE_V>();
          const BinaryRepeatParams params{1,1,0,static_cast<uint8_t>(kD/8),
              static_cast<uint8_t>(kD/8),1};
          for(int32_t chunk=0;chunk<kD/64;++chunk)
            Mul(acc[base*kD+chunk*64],acc[base*kD+chunk*64],alphaBlocks,
                static_cast<uint64_t>(64),static_cast<uint8_t>(active),params);
          PipeBarrier<PIPE_V>();
          Fence<HardEvent::V_MTE3>();
          DataCopy(work_[(first+base)*kSubtile],scores,active*kSubtile);
          Fence<HardEvent::MTE3_MTE2>();
          Fence<HardEvent::MTE3_V>();
        }
        CrossCoreSetFlag<2,PIPE_MTE3>(kPReady);
        CrossCoreWaitFlag(kPvReady);
        for(int32_t base=0;base<rows;base+=16) {
          const int32_t active=rows-base<16?rows-base:16;
          DataCopy(pvBlock,work_[kScoreFloats+(first+base)*kD],active*kD);
          Fence<HardEvent::MTE2_V>();
          Add(acc[base*kD],acc[base*kD],pvBlock,active*kD);
          PipeBarrier<PIPE_V>();
          Fence<HardEvent::V_MTE2>();
        }
        CrossCoreSetFlag<2,PIPE_MTE2>(kAccReady);
      }
      Fence<HardEvent::V_S>();
      for(int32_t r=0;r<rows;++r) {
        const int32_t visible=ends.GetValue(r);
        const float sum=totals.GetValue(r);
        int32_t error=errors.GetValue(r);
        if(error==0 && visible>0 && (!(sum>0.0F)||!Finite(sum)))error=2;
        errors.SetValue(r,error);
        Fence<HardEvent::S_V>();
        if(error!=0)Duplicate(acc[r*kD],__builtin_nanf(""),kD);
        else if(visible<=0)Duplicate(acc[r*kD],0.0F,kD);
        else Muls(acc[r*kD],acc[r*kD],1.0F/sum,kD);
        Fence<HardEvent::V_S>();
      }
      Fence<HardEvent::V_MTE3>();
      DataCopy(work_[kScoreFloats+first*kD],acc,rows*kD);
      Fence<HardEvent::MTE3_MTE2>();
      Fence<HardEvent::MTE3_V>();
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
      logBuf_,pvBuf_,visibleBuf_,maxBuf_,sumBuf_,errorBuf_,lseBuf_,
      softBuf_,inMaxBuf_,inSumBuf_,accBuf_,pvBlockBuf_,alphaBlocksBuf_;
  int32_t lane_=0;
  float scale_;
};
}  // namespace

extern "C" __global__ __aicore__ void oscar_kv256_fp32_b512_kernel(
    GM_ADDR q,GM_ADDR k,GM_ADDR v,GM_ADDR rotation,GM_ADDR visibleEnd,
    GM_ADDR workspace,GM_ADDR output,GM_ADDR lse,GM_ADDR status,float scale) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  Wide<512> op;
  op.Init(q,k,v,rotation,visibleEnd,workspace,output,lse,status,scale);
  op.Process();
}
extern "C" __global__ __aicore__ void oscar_kv256_fp32_b1024_kernel(
    GM_ADDR q,GM_ADDR k,GM_ADDR v,GM_ADDR rotation,GM_ADDR visibleEnd,
    GM_ADDR workspace,GM_ADDR output,GM_ADDR lse,GM_ADDR status,float scale) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  Wide<1024> op;
  op.Init(q,k,v,rotation,visibleEnd,workspace,output,lse,status,scale);
  op.Process();
}
