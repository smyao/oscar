// SPDX-License-Identifier: Apache-2.0
// EXPERIMENT ONLY. Archive #126/#129/#140/#145/#148/#154/#155.
// D.4 four questions: (1) bounded history-CV QK/P/FP32-PV/final RvT;
// (2) old full-history dequant cost 6499.8-6655.1ms/card vs FIA18.5-18.9ms;
// (3) each M126,D256,B512/B1024 block has one QK and one PV; score/P share
// fixed GM, with one GM score read per row and batched FP32 SoftmaxFlashV2.
// Every active original KV256 subdomain is checked BEFORE causal masking;
// fully invisible subdomains are skipped; all-empty rows publish P=0.
// MTE2_V / V_MTE2 / MTE3_MTE2 / MTE3_V own each reused UB explicitly.
// (4) API-count/GM-byte reduction is a structural hypothesis, not device or
// whole-service speed evidence. No INT2 producer or full-history restore.
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
  static constexpr int32_t kBatchRows=B==512?32:16;
  static_assert(kBatchRows*B*4==65536,"fixed score UB budget");
 public:
  static constexpr int32_t kExplicitUbBytes=65536+16384+1024+32+128+128+128+32+1024+5*256;
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
#ifdef ASCENDC_CPU_DEBUG
      if(GetBlockIdx()==0) {
        const SoftMaxShapeInfo auditShape{static_cast<uint32_t>(kBatchRows),
            static_cast<uint32_t>(B),static_cast<uint32_t>(kBatchRows),static_cast<uint32_t>(B)};
        const auto audit=SoftMaxFlashV2TilingFunc(auditShape,4,4,kSoftBytes,true,false);
        printf("WIDE_FP32_SIMD_TILING B=%d rows=%d reduceM=%u rangeM=%u tailM=%u scratch=%u explicitUb=%d\n",
            B,kBatchRows,audit.reduceM,audit.rangeM,audit.tailM,kSoftBytes,kExplicitUbBytes);
      }
#endif
      pipe_.InitBuffer(scoreBuf_,kBatchRows*B*4);
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
      for(int32_t base=0;base<rows;base+=kBatchRows) {
        const int32_t active=rows-base<kBatchRows?rows-base:kBatchRows;
        // One read of each real score. Padded rows never read poisoned GM.
        DataCopy(scores,work_[(first+base)*B],active*B);
        Fence<HardEvent::MTE2_V>();
        Muls(scores,scores,scale_,active*B);
        Fence<HardEvent::V_S>();
        for(int32_t r=0;r<active;++r) {
          const int32_t visible=ends.GetValue(base+r);
          int32_t error=errors.GetValue(base+r);
          if(visible>0 && visible<=B) {
            // Preserve the original KV256 error domain. A full-B sum would
            // change overflow/cancellation behavior before the causal mask.
            for(int32_t start=0;start<visible;start+=kSubtile) {
              Fence<HardEvent::S_V>();
              ReduceSum(scalar,scores[r*B+start],reduce,kSubtile);
              Fence<HardEvent::V_S>();
              if(!Finite(scalar.GetValue(0)))error=2;
            }
          }
          errors.SetValue(base+r,error);
          Fence<HardEvent::S_V>();
          if(error!=0 || visible==0) Duplicate(scores[r*B],kNegInf,B);
          else if(visible<B) {
            // Vector Duplicate starts at a 32B boundary; at most seven
            // scalar elements handle an arbitrary causal endpoint.
            const int32_t aligned=(visible+7)/8*8;
            for(int32_t j=visible;j<aligned;++j)scores.SetValue(r*B+j,kNegInf);
            Fence<HardEvent::S_V>();
            if(aligned<B)Duplicate(scores[r*B+aligned],kNegInf,B-aligned);
          }
          Fence<HardEvent::V_S>();
        }
        Fence<HardEvent::S_V>();
        if(active<kBatchRows)Duplicate(scores[active*B],kNegInf,(kBatchRows-active)*B);
        Duplicate(inMax,-__FLT_MAX__,kBatchRows);
        Duplicate(inSum,0.0F,kBatchRows);
        PipeBarrier<PIPE_V>();
        const SoftMaxShapeInfo shape{static_cast<uint32_t>(kBatchRows),
            static_cast<uint32_t>(B),static_cast<uint32_t>(kBatchRows),
            static_cast<uint32_t>(B)};
        // SDK float/ND/nonbasic needs (8+64)*4 bytes per reduced row.
        // 16KiB covers the full 32/16-row batch, within 188416B total UB.
        const auto tiling=SoftMaxFlashV2TilingFunc(shape,4,4,kSoftBytes,true,false);
        SoftmaxFlashV2<float,true,true,false,false,kSoftConfig>(
            scores,totals[base],maxima[base],scores,alpha,inSum,inMax,
            softWork,tiling,shape);
        Fence<HardEvent::V_S>();
        for(int32_t r=0;r<active;++r) {
          const float sum=totals.GetValue(base+r);
          if(ends.GetValue(base+r)>0 && errors.GetValue(base+r)==0 &&
             (!(sum>0.0F)||!Finite(sum)))errors.SetValue(base+r,2);
        }
        Fence<HardEvent::V_MTE3>();
        DataCopy(work_[(first+base)*B],scores,active*B);
        // #145: scoreBuf will be refilled by MTE2 on the next batch.
        Fence<HardEvent::MTE3_MTE2>();
        Fence<HardEvent::MTE3_V>();
        // #154: the next batch can reuse Vector scratch only after V is done.
        Fence<HardEvent::V_MTE2>();
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
      logBuf_,pvBuf_,visibleBuf_,maxBuf_,sumBuf_,errorBuf_,lseBuf_,
      softBuf_,inMaxBuf_,inSumBuf_;
  int32_t lane_=0;
  float scale_;
};
}  // namespace

extern "C" __global__ __aicore__ void oscar_wide_fp32_simd_b512_kernel(
    GM_ADDR q,GM_ADDR k,GM_ADDR v,GM_ADDR rotation,GM_ADDR visibleEnd,
    GM_ADDR workspace,GM_ADDR output,GM_ADDR lse,GM_ADDR status,float scale) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  Wide<512> op;
  op.Init(q,k,v,rotation,visibleEnd,workspace,output,lse,status,scale);
  op.Process();
}
extern "C" __global__ __aicore__ void oscar_wide_fp32_simd_b1024_kernel(
    GM_ADDR q,GM_ADDR k,GM_ADDR v,GM_ADDR rotation,GM_ADDR visibleEnd,
    GM_ADDR workspace,GM_ADDR output,GM_ADDR lse,GM_ADDR status,float scale) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  Wide<1024> op;
  op.Init(q,k,v,rotation,visibleEnd,workspace,output,lse,status,scale);
  op.Process();
}
