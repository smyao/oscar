// SPDX-License-Identifier: Apache-2.0
// Archive G3-G13/#3/#21/#81-83/#87-92: native A2 interfaces, bounded UB,
// primitive launch ABI, signed conversions and explicit event dependencies.
// Archive G26-G34/#4-20/#53-69: FP32 QK/PV/softmax accumulation, exact INT2
// layout, per-query causal masks, all output/LSE rows written, checked tags.
// Archive #126: last valid Q row must finish MTE3 reads before V zeroes the
// reused UB for padding. MTE3_MTE2 alone cannot order a V-only padding path.
// D.4 four questions: this replaces dequant+FIA and precise-window FIA.
// Prior full-history restore cost 6499.8-6655.1ms vs FIA 18.5-18.9ms at 32K.
// Here Vector unpacks one 32-token tile with SIMD Gather/Shift/And/Cast;
// Cube computes QK, PV and final Rv^T. Up to 64 GQA/query rows reuse each
// compressed read, including all q_len=4 when GQA<=16. No full-history tensor
// or history inverse rotation. A2 uses bounded per-Cube GM tile communication
// (not an on-chip-only claim): capacity (256*D+4096)*4 B/core, total traffic
// remains linear in history. UB is statically <192KiB/AIV for D<=256.
// Target: short step <=0.6-1.1ms and 32K same-order as native FIA; NO target
// precision/latency is claimed before fixed-tolerance NPU/profiler validation.
// #126 adds one local MTE3->V dependency before padding writes; tile storage,
// arithmetic and history traffic stay unchanged (no CPU/global sync or retry).
// Native precedents: hc_pre_m_k_split_core.h mode-2 Cube/Vector flags;
// hc_pre_cube_compute.h FP32 Mmad; moe_grouped_matmul.h MatmulImpl;
// add_rms_norm_bias_multi_n.h Gather; PR triton_oscar_decode.py INT2 and LSE.
#include "oscar_common.h"
#include "../include/oscar_attention_launch.h"
#include "lib/matmul_intf.h"
#include "lib/matmul/constant_tiling.h"
using namespace oscar_ascend_device;
namespace {
constexpr int32_t kQueryRows=64, kHalfRows=32, kKvRows=32, kHalfKv=16;
constexpr uint16_t kVectorReady=8, kCubeReady=9;
constexpr float kNegativeInfinity=-__builtin_inff();
__aicore__ inline int64_t Min64(int64_t a,int64_t b){return a<b?a:b;}
__aicore__ inline int64_t Max64(int64_t a,int64_t b){return a>b?a:b;}
using Matrix=MatmulType<TPosition::GM,CubeFormat::ND,float,false>;
using TransposedMatrix=MatmulType<TPosition::GM,CubeFormat::ND,float,true>;
__aicore__ constexpr MatmulConfig CvConfig() {
  auto cfg=GetNormalConfig();
  cfg.basicM=16;cfg.basicN=32;cfg.basicK=32;
  cfg.singleCoreM=64;cfg.singleCoreN=256;cfg.singleCoreK=256;
  cfg.enableSetBias=false;
  return cfg;
}
constexpr auto kCubeConfig=GetMatmulApiTiling<Matrix,TransposedMatrix,Matrix,Matrix>(CvConfig());
using CubeMatmul=matmul::MatmulImpl<Matrix,TransposedMatrix,Matrix,Matrix,kCubeConfig>;

struct Geometry {
  int64_t tokens,hq,hk,requests,columns,taskCount,blockTokens,blocks,ssmOffset;
  int64_t pageStride,windowStride,tagStride,sink,recent,speculative,splits;
  float scale;
};

template<int32_t D> class AttentionCv {
  static constexpr int32_t kSlotBytes=D/2+8;
  static constexpr int32_t kPackedStride=(kSlotBytes+31)/32*32;
  static constexpr int32_t kElements=kHalfKv*D;
  static constexpr int32_t kWords=kElements/8;
 public:
  __aicore__ void Init(GM_ADDR query,GM_ADDR queryRot,GM_ADDR key,GM_ADDR value,
      GM_ADDR rotation,GM_ADDR raw,GM_ADDR table,GM_ADDR windowKey,
      GM_ADDR windowValue,GM_ADDR tags,GM_ADDR tasks,GM_ADDR partial,
      GM_ADDR lse,GM_ADDR status,GM_ADDR workspace,const Geometry& geometry) {
    g=geometry;
    q.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(query));
    qr.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(queryRot));
    ck.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(key));
    cv.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(value));
    rv.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rotation));
    bytes.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(raw));
    bt.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(table));
    wk.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(windowKey));
    wv.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(windowValue));
    wt.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(tags));
    taskGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(tasks));
    out.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(partial));
    outLse.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lse));
    outStatus.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(status));
    int64_t core=GetBlockIdx();
    if ASCEND_IS_AIV {lane=core%2;core/=2;}
    coreIndex=core;
    const int64_t floatsPerCore=256*D+4096;
    work.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace)+core*floatsPerCore);
    // Q(64D), K(32D), V^T(32D), score(64*32), P(64*32), PV(64D), rotation(64D).
    qOffset=0;kOffset=64*D;vOffset=96*D;scoreOffset=128*D;
    pOffset=scoreOffset+2048;pvOffset=pOffset+2048;rotOffset=pvOffset+64*D;
    if ASCEND_IS_AIC {
      // Static tiling is compiler-derived by the real SDK. SetOrg/SingleShape
      // below specify each QK/PV/rotation matrix, never hand-written tiling POD.
      mm.Init(static_cast<const TCubeTiling*>(nullptr),&pipe);
      SetHF32Mode(false);
    } else {
      pipe.InitBuffer(taskBuf,128);pipe.InitBuffer(packedBuf,kHalfKv*kPackedStride);
      pipe.InitBuffer(planeBuf,kElements*2);pipe.InitBuffer(naturalBuf,kElements*2);
      pipe.InitBuffer(wordBuf,kWords*2);pipe.InitBuffer(maskBuf,kWords*2);
      pipe.InitBuffer(wordIndexBuf,kWords*4);pipe.InitBuffer(laneIndexBuf,kElements*4);
      pipe.InitBuffer(transposeIndexBuf,kElements*4);
      pipe.InitBuffer(dequantBuf,kElements*4);pipe.InitBuffer(transposeBuf,kElements*4);
      pipe.InitBuffer(scoreBuf,kHalfRows*kKvRows*4);
      pipe.InitBuffer(accBuf,kHalfRows*D*4);pipe.InitBuffer(pvBuf,kHalfRows*D*4);
      pipe.InitBuffer(qFloatBuf,D*4);pipe.InitBuffer(qBf16Buf,D*2);
      pipe.InitBuffer(statsBuf,kHalfRows*4*4);pipe.InitBuffer(scratchBuf,512);
      pipe.InitBuffer(addressBuf,128);
      InitIndices();
    }
  }
  __aicore__ void Process() {
    // Scalar Cube metadata reads must invalidate stale cache on changed-input
    // graph replay. The Vector side exclusively DMA-loads mutable metadata.
    if ASCEND_IS_AIC {
      DataCacheCleanAndInvalid<int64_t,CacheLine::ENTIRE_DATA_CACHE>(taskGm);
    }
    for(int64_t id=coreIndex;id<g.taskCount;id+=GetBlockNum()) {
      LoadTask(id);
      if(qcount==0) {if ASCEND_IS_AIV {PublishStatus(id,0);} continue;}
      if(qcount<0 || taskError || !TaskValid()) {
        if ASCEND_IS_AIV {PublishEmpty(id,taskError?taskError:(!TaskValid()?1:0));}
        continue;
      }
      if ASCEND_IS_AIC {CubeTask();}
      else {VectorTask(id);}
    }
    if ASCEND_IS_AIC {mm.End();}
  }
 private:
  __aicore__ void LoadTask(int64_t id) {
    int64_t data[11];
    if ASCEND_IS_AIC {for(int32_t i=0;i<11;++i)data[i]=taskGm.GetValue(id*16+i);}
    else {
      auto row=taskBuf.Get<int64_t>();DataCopy(row,taskGm[id*16],16);
      Fence<HardEvent::MTE2_S>();
      for(int32_t i=0;i<11;++i)data[i]=row.GetValue(i);
    }
    qbegin=data[0];qcount=data[1];kvhead=data[2];kvbegin=data[3];kvend=data[4];
    request=data[5];split=data[6];kind=data[7];context=data[8];requestBegin=data[9];
    taskError=static_cast<int32_t>(data[10]);
  }
  __aicore__ bool TaskValid() {
    return qbegin>=0 && qbegin<g.tokens && qcount<=64/(g.hq/g.hk) &&
        (qcount<0 || qbegin+qcount<=g.tokens) && kvhead>=0 && kvhead<g.hk &&
        kvbegin>=0 && kvend>=kvbegin && request>=0 && request<g.requests &&
        split>=0 && split<3*g.splits && kind>=0 && kind<3 && context>=0 &&
        requestBegin>=0 && requestBegin<=qbegin &&
        (kind!=0 || kvend<=context) &&
        (kind!=2 || (kvbegin>=context && requestBegin+kvend-context<=g.tokens));
  }
  __aicore__ void InitIndices() {
    auto wi=wordIndexBuf.Get<uint32_t>();auto li=laneIndexBuf.Get<uint32_t>();
    auto ti=transposeIndexBuf.Get<uint32_t>();
    for(int32_t i=0;i<kWords;++i)
      wi.SetValue(i,static_cast<uint32_t>((i/(D/8))*kPackedStride+(i%(D/8))*2));
    for(int32_t i=0;i<kElements;++i) {
      const int32_t r=i/D,d=i%D;
      li.SetValue(i,static_cast<uint32_t>(((d%8)*kWords+r*(D/8)+d/8)*2));
      // Output [D,16] gathers from input [16,D].
      ti.SetValue(i,static_cast<uint32_t>(((i%kHalfKv)*D+i/kHalfKv)*4));
    }
    Fence<HardEvent::S_V>();
    Duplicate(maskBuf.Get<uint16_t>(),static_cast<uint16_t>(3),kWords);
    PipeBarrier<PIPE_V>();
  }
  __aicore__ void Matmul(int64_t a,int64_t b,int64_t c,int32_t m,int32_t n,int32_t k,
      bool rotation=false) {
    SetHF32Mode(false); // no implicit HF32/TF32 relaxation of the frozen oracle.
    mm.SetOrgShape(m,n,k);mm.SetSingleShape(m,n,k);
    mm.SetTensorA(work[a],false);
    if(rotation) mm.SetTensorB(rv,true);else mm.SetTensorB(work[b],true);
    mm.template IterateAll<true>(work[c]);
    // End closes the iteration state; buffers remain initialized for reuse.
    mm.End();
  }
  __aicore__ void CubeTask() {
    const int32_t rows=static_cast<int32_t>(qcount*(g.hq/g.hk));
    for(int64_t start=kvbegin;start<kvend;start+=kKvRows) {
      CrossCoreWaitFlag(kVectorReady);
      Matmul(qOffset,kOffset,scoreOffset,rows,kKvRows,D);
      CrossCoreSetFlag<2,PIPE_FIX>(kCubeReady);
      CrossCoreWaitFlag(kVectorReady);
      Matmul(pOffset,vOffset,pvOffset,rows,D,kKvRows);
      CrossCoreSetFlag<2,PIPE_FIX>(kCubeReady);
      // Vector consumes PV before the next tile may overwrite shared buffers.
      CrossCoreWaitFlag(kVectorReady);
    }
    if(kind==0 && kvbegin<kvend) {
      CrossCoreWaitFlag(kVectorReady);
      Matmul(qOffset,0,rotOffset,rows,D,D,true);
      CrossCoreSetFlag<2,PIPE_FIX>(kCubeReady);
      CrossCoreWaitFlag(kVectorReady);
    }
  }
  __aicore__ void LoadQueries() {
    auto dst=qFloatBuf.Get<float>();auto src=qBf16Buf.Get<bfloat16_t>();
    const int64_t group=g.hq/g.hk;
    const int64_t rows=qcount*group;
    for(int32_t row=lane*kHalfRows;row<(lane+1)*kHalfRows;++row) {
      if(row<rows) {
        const int64_t off=((qbegin+row/group)*g.hq+kvhead*group+row%group)*D;
        if(kind==0) {DataCopy(dst,qr[off],D);Fence<HardEvent::MTE2_V>();}
        else {DataCopy(src,q[off],D);Fence<HardEvent::MTE2_V>();
          Cast(dst,src,RoundMode::CAST_NONE,D);PipeBarrier<PIPE_V>();}
        auto check=scratchBuf.Get<float>();
        ReduceSum(check,dst,check[16],D);Fence<HardEvent::V_S>();
        if(!Finite(check.GetValue(0)))error=2;
        Fence<HardEvent::V_MTE3>();
      } else {
        // The preceding row may still be reading dst on MTE3. Valid rows
        // chain MTE3_MTE2 -> MTE2_V, but padding does no MTE2 load: without
        // this edge Duplicate can zero the last valid query before its DMA.
        Fence<HardEvent::MTE3_V>();
        Duplicate(dst,0.0F,D);Fence<HardEvent::V_MTE3>();
      }
      DataCopy(work[qOffset+row*D],dst,D);Fence<HardEvent::MTE3_MTE2>();
    }
  }
  __aicore__ int64_t LogicalPosition(int64_t candidate) {
    if(kind!=1) return candidate;
    const int64_t sinkCount=Min64(g.sink,context);
    const int64_t firstPosition=context+qbegin-requestBegin;
    const int64_t tailBegin=Min64(context,Max64(g.sink,firstPosition+1-g.recent));
    return candidate<sinkCount?candidate:tailBegin+candidate-sinkCount;
  }
  __aicore__ bool Physical(int64_t position,int64_t& block,int64_t& inpage) {
    if(position<0 || position/128>=g.columns) {error=1;return false;}
    auto local=addressBuf.Get<int32_t>();
    DataCopyExtParams copy{1,4,0,0,0};DataCopyPadExtParams<int32_t> pad{false,0,0,0};
    DataCopyPad(local,bt[request*g.columns+position/128],copy,pad);
    Fence<HardEvent::MTE2_S>();
    const int64_t virtualBlock=local.GetValue(0);
    const int64_t ratio=g.blockTokens/128;
    if(virtualBlock<0 || virtualBlock>=g.blocks*ratio) {error=1;return false;}
    block=virtualBlock/ratio;inpage=(virtualBlock%ratio)*128+position%128;
    return true;
  }
  __aicore__ void LoadPacked(int64_t start) {
    liveKvRows=static_cast<int32_t>(Max64(0,Min64(kHalfKv,kvend-start-lane*kHalfKv)));
    auto packed=packedBuf.Get<uint8_t>();
    Duplicate(packed.ReinterpretCast<uint16_t>(),static_cast<uint16_t>(0),
        kHalfKv*kPackedStride/2);Fence<HardEvent::V_MTE2>();
    for(int32_t j=0;j<kHalfKv;++j) {
      const int64_t candidate=start+lane*kHalfKv+j;
      if(candidate>=kvend)continue;
      int64_t block=0,inpage=0;
      if(!Physical(candidate,block,inpage))continue;
      const int64_t off=g.ssmOffset+block*g.pageStride+
          (inpage*g.hk+kvhead)*kSlotBytes;
      DataCopyExtParams copy{1,kSlotBytes,0,0,0};
      DataCopyPadExtParams<uint8_t> pad{false,0,0,0};
      DataCopyPad(packed[j*kPackedStride],bytes[off],copy,pad);
    }
    Fence<HardEvent::MTE2_V>();
  }
  __aicore__ void Unpack(bool value) {
    auto packed=packedBuf.Get<uint16_t>();auto words=wordBuf.Get<uint16_t>();
    auto planes=planeBuf.Get<uint16_t>();auto natural=naturalBuf.Get<int16_t>();
    auto mask=maskBuf.Get<uint16_t>();auto values=dequantBuf.Get<float>();
    const uint32_t byteBase=value?D/4+4:0;
    Gather(words,packed,wordIndexBuf.Get<uint32_t>(),byteBase,kWords);
    PipeBarrier<PIPE_V>();
    for(int32_t bit=0;bit<8;++bit) {
      ShiftRight(planes[bit*kWords],words,static_cast<uint16_t>(bit*2),kWords);
      PipeBarrier<PIPE_V>();And(planes[bit*kWords],planes[bit*kWords],mask,kWords);
      PipeBarrier<PIPE_V>();
    }
    Gather(natural,planes.ReinterpretCast<int16_t>(),laneIndexBuf.Get<uint32_t>(),0,kElements);
    PipeBarrier<PIPE_V>();Cast(values,natural,RoundMode::CAST_NONE,kElements);
    Fence<HardEvent::V_S>();
    auto halves=packedBuf.Get<half>();
    for(int32_t row=0;row<kHalfKv;++row) {
      const int32_t meta=(row*kPackedStride+byteBase+D/4)/2;
      const float scale=static_cast<float>(halves.GetValue(meta));
      const float zero=static_cast<float>(halves.GetValue(meta+1));
      if(row<liveKvRows && (!Finite(scale)||!Finite(zero)||scale<=0.0F))error=3;
      Muls(values[row*D],values[row*D],scale,D);PipeBarrier<PIPE_V>();
      Adds(values[row*D],values[row*D],zero,D);PipeBarrier<PIPE_V>();
    }
  }
  __aicore__ void LoadPrecise(int64_t start,bool value) {
    auto dst=dequantBuf.Get<float>();auto src=qBf16Buf.Get<bfloat16_t>();
    Duplicate(dst,0.0F,kElements);PipeBarrier<PIPE_V>();
    for(int32_t j=0;j<kHalfKv;++j) {
      const int64_t candidate=start+lane*kHalfKv+j;
      if(candidate>=kvend)continue;
      const int64_t position=LogicalPosition(candidate);
      if(kind==2) {
        const int64_t token=requestBegin+position-context;
        if(token<0 || token>=g.tokens) {error=1;continue;}
        if(value)DataCopy(src,cv[(token*g.hk+kvhead)*D],D);
        else DataCopy(src,ck[(token*g.hk+kvhead)*D],D);
      } else {
        int64_t block=0,inpage=0;
        if(!Physical(position,block,inpage))continue;
        const int64_t row=position<g.sink?position:g.sink+inpage%(g.recent+g.speculative);
        auto tag=addressBuf.Get<int64_t>()[4];
        DataCopyExtParams copy{1,8,0,0,0};DataCopyPadExtParams<int64_t> pad{false,0,0,0};
        DataCopyPad(tag,wt[block*g.tagStride+row],copy,pad);
        Fence<HardEvent::MTE2_S>();
        if(tag.GetValue(0)!=inpage) {error=4;continue;}
        const int64_t off=block*g.windowStride+(row*g.hk+kvhead)*D;
        if(value)DataCopy(src,wv[off],D);else DataCopy(src,wk[off],D);
      }
      Fence<HardEvent::MTE2_V>();Cast(dst[j*D],src,RoundMode::CAST_NONE,D);
      PipeBarrier<PIPE_V>();
      auto check=scratchBuf.Get<float>();ReduceSum(check,dst[j*D],check[16],D);
      Fence<HardEvent::V_S>();if(!Finite(check.GetValue(0)))error=2;
      Fence<HardEvent::V_MTE2>();
    }
  }
  __aicore__ void PublishKv(bool value) {
    auto src=dequantBuf.Get<float>();
    if(!value) {Fence<HardEvent::V_MTE3>();
      DataCopy(work[kOffset+lane*kElements],src,kElements);Fence<HardEvent::MTE3_V>();}
    else {
      auto transposed=transposeBuf.Get<float>();
      Gather(transposed,src,transposeIndexBuf.Get<uint32_t>(),0,kElements);
      Fence<HardEvent::V_MTE3>();
      DataCopyExtParams copy{D,kHalfKv*4,0,kHalfKv*4,0};
      DataCopyPad(work[vOffset+lane*kHalfKv],transposed,copy);
      Fence<HardEvent::MTE3_V>();
    }
  }
  __aicore__ bool Visible(int64_t queryPosition,int64_t candidate) {
    if(candidate>=kvend)return false;
    const int64_t pos=LogicalPosition(candidate);
    if(pos>queryPosition)return false;
    const int64_t cut=Max64(g.sink,queryPosition+1-g.recent);
    if(kind==0)return pos>=g.sink && pos<cut && pos<context;
    if(kind==1)return pos<context && (pos<g.sink || pos>=cut);
    return pos>=context;
  }
  __aicore__ void Softmax(int64_t start) {
    auto scores=scoreBuf.Get<float>();auto acc=accBuf.Get<float>();
    auto stats=statsBuf.Get<float>();auto tmp=scratchBuf.Get<float>();
    DataCopy(scores,work[scoreOffset+lane*kHalfRows*kKvRows],kHalfRows*kKvRows);
    Fence<HardEvent::MTE2_V>();Muls(scores,scores,g.scale,kHalfRows*kKvRows);
    Fence<HardEvent::V_S>();
    const int64_t group=g.hq/g.hk;
    for(int32_t r=0;r<kHalfRows;++r) {
      const int64_t row=lane*kHalfRows+r;
      const bool validRow=row<qcount*group;
      const int64_t position=context+qbegin+row/group-requestBegin;
      bool any=false;
      for(int32_t j=0;j<kKvRows;++j) {
        if(!validRow || !Visible(position,start+j))scores.SetValue(r*kKvRows+j,kNegativeInfinity);
        else {any=true;if(!Finite(scores.GetValue(r*kKvRows+j)))error=2;}
      }
      const float oldMax=stats.GetValue(r*4),oldSum=stats.GetValue(r*4+1);
      if(!any) {Fence<HardEvent::S_V>();Duplicate(scores[r*kKvRows],0.0F,kKvRows);
        Fence<HardEvent::V_S>();continue;}
      Fence<HardEvent::S_V>();ReduceMax(tmp,scores[r*kKvRows],tmp[16],kKvRows);
      Fence<HardEvent::V_S>();
      const float tileMax=tmp.GetValue(0),newMax=oldMax>tileMax?oldMax:tileMax;
      tmp.SetValue(0,oldSum>0.0F?oldMax-newMax:kNegativeInfinity);
      Fence<HardEvent::S_V>();Exp(tmp[8],tmp,1);Fence<HardEvent::V_S>();
      const float alpha=tmp.GetValue(8);
      Adds(scores[r*kKvRows],scores[r*kKvRows],-newMax,kKvRows);PipeBarrier<PIPE_V>();
      Exp(scores[r*kKvRows],scores[r*kKvRows],kKvRows);PipeBarrier<PIPE_V>();
      ReduceSum(tmp,scores[r*kKvRows],tmp[16],kKvRows);Fence<HardEvent::V_S>();
      const float sum=oldSum*alpha+tmp.GetValue(0);
      stats.SetValue(r*4,newMax);stats.SetValue(r*4+1,sum);
      Muls(acc[r*D],acc[r*D],alpha,D);PipeBarrier<PIPE_V>();
    }
    Fence<HardEvent::V_MTE3>();
    DataCopy(work[pOffset+lane*kHalfRows*kKvRows],scores,kHalfRows*kKvRows);
    Fence<HardEvent::MTE3_V>();
  }
  __aicore__ void VectorTask(int64_t id) {
    error=0;
    auto acc=accBuf.Get<float>();auto stats=statsBuf.Get<float>();
    Duplicate(acc,0.0F,kHalfRows*D);Fence<HardEvent::V_S>();
    for(int32_t r=0;r<kHalfRows;++r) {stats.SetValue(r*4,kNegativeInfinity);
      stats.SetValue(r*4+1,0.0F);}
    LoadQueries();
    for(int64_t start=kvbegin;start<kvend;start+=kKvRows) {
      if(kind==0) {LoadPacked(start);Unpack(false);}else LoadPrecise(start,false);
      PublishKv(false);
      if(kind==0)Unpack(true);else LoadPrecise(start,true);
      PublishKv(true);
      CrossCoreSetFlag<2,PIPE_MTE3>(kVectorReady);
      CrossCoreWaitFlag(kCubeReady);
      Softmax(start);
      CrossCoreSetFlag<2,PIPE_MTE3>(kVectorReady);
      CrossCoreWaitFlag(kCubeReady);
      auto result=pvBuf.Get<float>();
      DataCopy(result,work[pvOffset+lane*kHalfRows*D],kHalfRows*D);
      Fence<HardEvent::MTE2_V>();Add(acc,acc,result,kHalfRows*D);PipeBarrier<PIPE_V>();
      CrossCoreSetFlag<2,PIPE_MTE2>(kVectorReady);
    }
    Fence<HardEvent::V_S>();
    for(int32_t r=0;r<kHalfRows;++r) {
      const float sum=stats.GetValue(r*4+1);
      if(sum>0.0F && Finite(sum))Muls(acc[r*D],acc[r*D],1.0F/sum,D);
      else {if(sum!=0.0F)error=2;Duplicate(acc[r*D],0.0F,D);}
      PipeBarrier<PIPE_V>();
    }
    if(kind==0 && kvbegin<kvend) {
      Fence<HardEvent::V_MTE3>();DataCopy(work[qOffset+lane*kHalfRows*D],acc,kHalfRows*D);
      CrossCoreSetFlag<2,PIPE_MTE3>(kVectorReady);
      CrossCoreWaitFlag(kCubeReady);
      DataCopy(acc,work[rotOffset+lane*kHalfRows*D],kHalfRows*D);
      Fence<HardEvent::MTE2_V>();CrossCoreSetFlag<2,PIPE_MTE2>(kVectorReady);
    }
    PublishRows(id);
  }
  __aicore__ void PublishRows(int64_t id) {
    auto acc=accBuf.Get<float>();auto stats=statsBuf.Get<float>();auto scalar=scratchBuf.Get<float>();
    const int64_t group=g.hq/g.hk,rows=qcount*group;
    Fence<HardEvent::V_S>();
    for(int32_t r=0;r<kHalfRows && lane*kHalfRows+r<rows;++r) {
      const int64_t row=lane*kHalfRows+r;
      const int64_t outRow=((qbegin+row/group)*g.hq+kvhead*group+row%group)*3*g.splits+split;
      const float sum=stats.GetValue(r*4+1),maximum=stats.GetValue(r*4);
      float lse=kNegativeInfinity;
      if(sum>0.0F && Finite(sum)) {
        scalar.SetValue(0,sum);Fence<HardEvent::S_V>();
        Log(scalar[8],scalar,1);Fence<HardEvent::V_S>();lse=maximum+scalar.GetValue(8);
      }
      if(error) {Duplicate(acc[r*D],__builtin_nanf(""),D);lse=__builtin_nanf("");}
      Fence<HardEvent::V_MTE3>();DataCopy(out[outRow*D],acc[r*D],D);
      scalar.SetValue(0,lse);Fence<HardEvent::S_MTE3>();
      DataCopyExtParams copy{1,4,0,0,0};DataCopyPad(outLse[outRow],scalar,copy);
      Fence<HardEvent::MTE3_S>();
    }
    PublishStatus(id,error);
  }
  __aicore__ void PublishEmpty(int64_t id,int32_t code) {
    // Padding has one output owner per token/head/segment, never leaves poison.
    if(qbegin>=0 && qbegin<g.tokens && kvhead>=0 && kvhead<g.hk &&
        split>=0 && split<3*g.splits) {
      qcount=1;error=code;
      auto stats=statsBuf.Get<float>();auto acc=accBuf.Get<float>();
      Duplicate(acc,code?__builtin_nanf(""):0.0F,kHalfRows*D);Fence<HardEvent::V_S>();
      for(int32_t r=0;r<kHalfRows;++r) {stats.SetValue(r*4,kNegativeInfinity);stats.SetValue(r*4+1,0.0F);}
      PublishRows(id);
    }else PublishStatus(id,code?code:1);
  }
  __aicore__ void PublishStatus(int64_t id,int32_t code) {
    auto word=addressBuf.Get<int32_t>();word.SetValue(0,code);
    Fence<HardEvent::S_MTE3>();DataCopyExtParams copy{1,4,0,0,0};
    DataCopyPad(outStatus[id*2+lane],word,copy);Fence<HardEvent::MTE3_S>();
  }
  TPipe pipe;CubeMatmul mm;
  Geometry g;
  GlobalTensor<bfloat16_t> q,ck,cv,wk,wv;
  GlobalTensor<float> qr,rv,out,outLse,work;
  GlobalTensor<uint8_t> bytes;GlobalTensor<int32_t> bt,outStatus;
  GlobalTensor<int64_t> wt,taskGm;
  TBuf<TPosition::VECCALC> taskBuf,packedBuf,planeBuf,naturalBuf,wordBuf,maskBuf,
      wordIndexBuf,laneIndexBuf,transposeIndexBuf,dequantBuf,transposeBuf,
      scoreBuf,accBuf,pvBuf,qFloatBuf,qBf16Buf,statsBuf,scratchBuf,addressBuf;
  int64_t coreIndex,qbegin,qcount,kvhead,kvbegin,kvend,request,split,kind,context,requestBegin;
  int64_t qOffset,kOffset,vOffset,scoreOffset,pOffset,pvOffset,rotOffset;
  int32_t lane=0,error=0,taskError=0,liveKvRows=0;
};
}
#define OSCAR_CV_ARGUMENTS GM_ADDR query, GM_ADDR queryRot, GM_ADDR key, GM_ADDR value, \
    GM_ADDR rotation, GM_ADDR raw, GM_ADDR table, GM_ADDR wk, GM_ADDR wv, GM_ADDR tags, \
    GM_ADDR tasks, GM_ADDR output, GM_ADDR lse, GM_ADDR status, GM_ADDR workspace, \
    int64_t tokens, int64_t hq, int64_t hk, int64_t dim, int64_t requests, \
    int64_t columns, int64_t taskCount, int64_t blockTokens, int64_t blocks, \
    int64_t ssmOffset, int64_t pageStride, int64_t windowStride, int64_t tagStride, \
    int64_t sink, int64_t recent, int64_t speculative, int64_t splits, float scale
#define OSCAR_CV_INIT op.Init(query,queryRot,key,value,rotation,raw,table,wk,wv,tags, \
    tasks,output,lse,status,workspace,g);op.Process()
extern "C" __global__ __aicore__ void oscar_attention_cv_kernel(OSCAR_CV_ARGUMENTS) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  const Geometry g{tokens,hq,hk,requests,columns,taskCount,blockTokens,blocks,ssmOffset,
      pageStride,windowStride,tagStride,sink,recent,speculative,splits,scale};
  if(dim==64) {AttentionCv<64> op;OSCAR_CV_INIT;}
  else if(dim==128) {AttentionCv<128> op;OSCAR_CV_INIT;}
  else {AttentionCv<256> op;OSCAR_CV_INIT;}
}
#ifndef ASCENDC_CPU_DEBUG
namespace oscar_ascend {
void attention_cv_launch(void* stream,void* query,void* queryRot,void* key,void* value,
    void* rotation,void* raw,void* table,void* wk,void* wv,void* tags,void* tasks,
    void* output,void* lse,void* status,void* workspace,int64_t tokens,int64_t hq,
    int64_t hk,int64_t dim,int64_t requests,int64_t columns,int64_t taskCount,
    int64_t blockTokens,int64_t blocks,int64_t ssmOffset,int64_t pageStride,
    int64_t windowStride,int64_t tagStride,int64_t sink,int64_t recent,
    int64_t speculative,int64_t splits,float scale,uint32_t cores) {
  oscar_attention_cv_kernel<<<cores,nullptr,stream>>>(
      static_cast<uint8_t*>(query),static_cast<uint8_t*>(queryRot),
      static_cast<uint8_t*>(key),static_cast<uint8_t*>(value),
      static_cast<uint8_t*>(rotation),static_cast<uint8_t*>(raw),
      static_cast<uint8_t*>(table),static_cast<uint8_t*>(wk),static_cast<uint8_t*>(wv),
      static_cast<uint8_t*>(tags),static_cast<uint8_t*>(tasks),static_cast<uint8_t*>(output),
      static_cast<uint8_t*>(lse),static_cast<uint8_t*>(status),static_cast<uint8_t*>(workspace),
      tokens,hq,hk,dim,requests,columns,taskCount,blockTokens,blocks,ssmOffset,
      pageStride,windowStride,tagStride,sink,recent,speculative,splits,scale);
}
}
#endif
