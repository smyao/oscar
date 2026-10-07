// ISOLATED COMBINED DECODE TRIAL. Archive #126/#129/#140/#145/#148/#151/#154/#155.
// D.4 four questions: complete source0/1/2 q4/q1 FIA; no full-history restore
// repeating6499.8-6655.1ms. Fixed history KV256 pipeline + padded natural
// INT16 scratch + batched exact-window reads + exact range masks. Only serial
// precise sources omit padded-zero Cube columns. No new score clearing in
// pipelined history, preserving its score/P and PV ownership proof.
// UB158656B/AIV, GM1146880B/core, independent of history. No production/NPU
// performance claim: combined numerical and complete cycle gates still apply.
// SPDX-License-Identifier: Apache-2.0
// ISOLATED EXPERIMENT. Source: current production SIMD decode kernel, SHA256
// c67219ce15c98e932acaa8b1c5515a3f31c99dd6e4963823623528ba2b3a3930
// Archive #126/#129/#140/#145/#148/#154/#155, startup D.4 four questions:
// 1. Fused INT2 history attention for q4/q1; window/current remain original.
// 2. D.4 full-history restore cost 6499.8-6655.1ms; no such tensor here.
// 3. Two fixed KV256 GM buffers, two packed HalfKv64 UB buffers. Next K
//    loads during current QK; next V during current PV. FP32 operations,
//    KV order and final RvT stay unchanged; future errors commit in old order.
// 4. Overlap is a hypothesis. No CPU/CANN/model/NPU speed claimed yet.
// D256 workspace 1146880 bytes/core, additional UB10240 bytes/lane;
// no memory grows with historical length. Source0 nonempty only.
#include "oscar_common.h"
#include "attention_striped_unpack_padded.h"
#include "attention_range_mask.h"
#include "../include/oscar_attention_launch.h"
#define OSCAR_SCHEDULE_FN __aicore__ inline
#include "../include/oscar_cv_schedule.h"
#undef OSCAR_SCHEDULE_FN
#include "lib/matmul_intf.h"
#include "lib/matmul/constant_tiling.h"
#include "adv_api/activation/softmaxflashv2.h"
using namespace oscar_ascend_device;
namespace {
constexpr int32_t kQueryRows=32;
constexpr int32_t kKvRows=oscar_ascend::kAttentionKvRows;
constexpr int32_t kHalfRows=16, kHalfKv=64;
constexpr int32_t kLaneRows=kQueryRows/2, kRowBlocks=kLaneRows/kHalfRows;
constexpr int32_t kKvSubtiles=kKvRows/(2*kHalfKv);
constexpr uint16_t kVectorReady=8, kCubeReady=9;
constexpr uint16_t kKvReady=8,kScoreReady=9,kPReady=10,kPvReady=11,
                   kNormReady=12,kRotReady=13,kTaskDone=14;
constexpr float kNegativeInfinity=-__builtin_inff();
constexpr float kEmptyMax=-__FLT_MAX__;
constexpr SoftmaxConfig kCvSoftmaxConfig={false,0,0,SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};
__aicore__ inline int64_t Min64(int64_t a,int64_t b){return a<b?a:b;}
__aicore__ inline int64_t Max64(int64_t a,int64_t b){return a>b?a:b;}
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

struct Geometry {
  int64_t tokens,hq,hk,requests,columns,taskCount,blockTokens,blocks,ssmOffset;
  int64_t pageStride,windowStride,tagStride,sink,recent,speculative,splits;
  float scale;
};

template<int32_t D,int32_t OwnerTile> class AttentionCv {
  static constexpr int32_t kSlotBytes=D/2+8;
  static constexpr int32_t kPackedStride=(kSlotBytes+31)/32*32;
  static constexpr int32_t kElements=kHalfKv*D;
  static constexpr int32_t kWords=kElements/8;
  static constexpr int32_t kPlaneStride=kWords+16;
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
    const int64_t floatsPerCore=((2*kQueryRows+4*kKvRows)*D+kQueryRows*kKvRows);
    work.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace)+core*floatsPerCore);
    // Q(128D), K(256D), natural V(256D), score/P(128*256),
    // PV/rotation(128D). Query accumulator stays in UB across KV units.
    // QK is complete before Vector overwrites score with P; PV is consumed
    // before the final Rv rotation reuses that same GM tile. No active aliases.
    qOffset=0;kOffset=kQueryRows*D;vOffset=(kQueryRows+kKvRows)*D;
    scoreOffset=(kQueryRows+4*kKvRows)*D;
    pOffset=scoreOffset;pvOffset=pOffset+kQueryRows*kKvRows;rotOffset=pvOffset;
    if ASCEND_IS_AIC {
      // Static tiling is compiler-derived by the real SDK. SetOrg/SingleShape
      // below specify each QK/PV/rotation matrix, never hand-written tiling POD.
      mm.Init(static_cast<const TCubeTiling*>(nullptr),&pipe);
      SetHF32Mode(false);
    } else {
      pipe.InitBuffer(taskBuf,128);pipe.InitBuffer(packedBuf,kHalfKv*kPackedStride);
      pipe.InitBuffer(packedNextBuf,kHalfKv*kPackedStride);
      pipe.InitBuffer(planeBuf,32);
      pipe.InitBuffer(naturalBuf,kHalfKv*(D+16)*2);
      pipe.InitBuffer(wordBuf,32);pipe.InitBuffer(maskBuf,D/8*2);
      pipe.InitBuffer(wordIndexBuf,32);pipe.InitBuffer(laneIndexBuf,32);
      pipe.InitBuffer(metadataIndexBuf,2*kHalfKv*4);pipe.InitBuffer(metadataHalfBuf,2*kHalfKv*2);
      pipe.InitBuffer(dequantBuf,kElements*4);
      pipe.InitBuffer(scoreBuf,kHalfRows*kKvRows*4);
      pipe.InitBuffer(accBuf,kLaneRows*D*4);
      pipe.InitBuffer(qFloatBuf,D*4);pipe.InitBuffer(qBf16Buf,D*2);
      pipe.InitBuffer(statsBuf,(2*kLaneRows+2*kHalfRows)*4);pipe.InitBuffer(scratchBuf,kHalfKv*8*4);
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
    const int64_t queryTile=OwnerTile;
    const int64_t perToken=g.hk*3*g.splits;
    const oscar_ascend_schedule::CvTaskSchedule schedule{g.tokens,queryTile,perToken};
    // A work item owns one global token tile and one source/head/split.
    // Scan every slot in it: request boundaries/padding holes may put the
    // active leader anywhere, so no task is discarded or counted twice.
    for(int64_t workId=coreIndex;workId<schedule.WorkItems();workId+=GetBlockNum()) {
      const int64_t tokenBegin=schedule.TokenBegin(workId);
      for(int64_t token=tokenBegin;token<Min64(tokenBegin+queryTile,g.tokens);++token) {
        const int64_t id=schedule.TaskId(workId,token);
        LoadTask(id);
        if(qcount==0) {if ASCEND_IS_AIV {PublishStatus(id,0);} continue;}
        if(qcount<0 || taskError || !TaskValid()) {
          if ASCEND_IS_AIV {PublishEmpty(id,taskError?taskError:(!TaskValid()?1:0));}
          continue;
        }
        if(kind==0 && kvbegin<kvend) {
          if ASCEND_IS_AIC {HistoryCubeTask();} else {HistoryVectorTask(id);}
        } else {
          if ASCEND_IS_AIC {CubeTask();} else {VectorTask(id);}
        }
      }
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
    return qbegin>=0 && qbegin<g.tokens && g.hk>0 && g.hq>0 &&
        g.hq%g.hk==0 && g.hq/g.hk<=8 && qcount<=OwnerTile &&
        qcount*(g.hq/g.hk)<=kQueryRows &&
        (qcount<0 || qbegin+qcount<=g.tokens) && kvhead>=0 && kvhead<g.hk &&
        kvbegin>=0 && kvend>=kvbegin && request>=0 && request<g.requests &&
        split>=0 && split<3*g.splits && kind>=0 && kind<3 && context>=0 &&
        requestBegin>=0 && requestBegin<=qbegin &&
        (kind!=0 || kvend<=context) &&
        (kind!=2 || (kvbegin>=context && requestBegin+kvend-context<=g.tokens));
  }
  __aicore__ void InitIndices() {
    oscar_ascend_padded_natural::InitIndices<D,kHalfKv>(wordIndexBuf,laneIndexBuf,
        metadataIndexBuf,maskBuf);
  }
  __aicore__ void Matmul(int64_t a,int64_t b,int64_t c,int32_t m,int32_t n,int32_t k,
      bool rotation=false,bool transposeB=true) {
    SetHF32Mode(false); // no implicit HF32/TF32 relaxation of the frozen oracle.
    // Physical Q/P/K/V strides remain the original bounded layout.
    mm.SetOrgShape(m,rotation?D:(transposeB?kKvRows:D),
        rotation?D:(transposeB?D:kKvRows));
    mm.SetSingleShape(m,n,k);
    mm.SetTensorA(work[a],false);
    if(rotation) mm.SetTensorB(rv,true);else mm.SetTensorB(work[b],transposeB);
    mm.template IterateAll<true>(work[c]);
    // End closes the iteration state; buffers remain initialized for reuse.
    mm.End();
  }
  __aicore__ int32_t TailColumns(int64_t start) {
    if(kind==0)return kKvRows; // history pipeline shared score/P plane remains unchanged
    return static_cast<int32_t>((Min64(kKvRows,kvend-start)+15)/16*16);
  }
  __aicore__ void ClearPaddedScores() {
    // QK writes only TailColumns columns. All omitted scores are the same
    // +0 produced by finite Q multiplied by the old padded zero K rows.
    // Invalid Q is still checked by unchanged LoadQueries and status logic.
    auto zeros=scoreBuf.Get<float>();
    Fence<HardEvent::MTE3_V>();
    Duplicate(zeros,0.0F,kHalfRows*kKvRows);Fence<HardEvent::V_MTE3>();
    for(int32_t block=0;block<kRowBlocks;++block)
      DataCopy(work[scoreOffset+(lane*kLaneRows+block*kHalfRows)*kKvRows],
          zeros,kHalfRows*kKvRows);
    Fence<HardEvent::MTE3_V>();
  }
  __aicore__ void CubeTask() {
    const int32_t rows=static_cast<int32_t>(qcount*(g.hq/g.hk));
    for(int64_t start=kvbegin;start<kvend;start+=kKvRows) {
      CrossCoreWaitFlag(kVectorReady);
      Matmul(qOffset,kOffset,scoreOffset,rows,TailColumns(start),D);
      CrossCoreSetFlag<2,PIPE_FIX>(kCubeReady);
      CrossCoreWaitFlag(kVectorReady);
      Matmul(pOffset,vOffset,pvOffset,rows,D,TailColumns(start),false,false);
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
    const int32_t laneBegin=lane*kLaneRows,laneEnd=(lane+1)*kLaneRows;
    const int32_t validEnd=static_cast<int32_t>(Min64(laneEnd,Max64(laneBegin,rows)));
    for(int32_t row=laneBegin;row<validEnd;++row) {
      const int64_t off=((qbegin+row/group)*g.hq+kvhead*group+row%group)*D;
      if(kind==0) {DataCopy(dst,qr[off],D);Fence<HardEvent::MTE2_V>();}
      else {DataCopy(src,q[off],D);Fence<HardEvent::MTE2_V>();
        Cast(dst,src,RoundMode::CAST_NONE,D);PipeBarrier<PIPE_V>();}
      auto check=scratchBuf.Get<float>();
      ReduceSum(check,dst,check[16],D);Fence<HardEvent::V_S>();
      if(!Finite(check.GetValue(0)))error=2;
      Fence<HardEvent::V_MTE3>();
      DataCopy(work[qOffset+row*D],dst,D);Fence<HardEvent::MTE3_MTE2>();
    }
    // Pad the inactive query rows in bounded contiguous pieces. This keeps
    // the full initialized Q tile required by the Cube contract without
    // adding one scalar/GM copy per padded row to q_len=1/4 graph replay.
    // The previous live-row MTE3 must finish before Vector reuses the scratch.
    auto zeros=dequantBuf.Get<float>();
    Fence<HardEvent::MTE3_V>();
    for(int32_t row=validEnd;row<laneEnd;row+=kHalfKv) {
      const int32_t count=static_cast<int32_t>(Min64(kHalfKv,laneEnd-row));
      Duplicate(zeros,0.0F,count*D);Fence<HardEvent::V_MTE3>();
      DataCopy(work[qOffset+row*D],zeros,count*D);Fence<HardEvent::MTE3_V>();
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
  __aicore__ void LoadPackedInto(int64_t start,TBuf<TPosition::VECCALC>& destination) {
    liveKvRows=static_cast<int32_t>(Max64(0,Min64(kHalfKv,kvend-start-lane*kHalfKv)));
    auto packed=destination.Get<uint8_t>();
    Duplicate(packed.ReinterpretCast<uint16_t>(),static_cast<uint16_t>(0),
        kHalfKv*kPackedStride/2);Fence<HardEvent::V_MTE2>();
    for(int32_t j=0;j<liveKvRows;) {
      const int64_t candidate=start+lane*kHalfKv+j;
      const int32_t rows=static_cast<int32_t>(Min64(liveKvRows-j,128-candidate%128));
      int64_t block=0,inpage=0;
      if(!Physical(candidate,block,inpage)) {j+=rows;continue;}
      const int64_t off=g.ssmOffset+block*g.pageStride+
          (inpage*g.hk+kvhead)*kSlotBytes;
      // GM rows are interleaved by KV head; UB rows are padded to 32 bytes.
      // Never assume adjacent virtual128 entries point to adjacent pages.
      DataCopyExtParams copy{static_cast<uint16_t>(rows),kSlotBytes,
          static_cast<uint32_t>((g.hk-1)*kSlotBytes),0,0};
      DataCopyPadExtParams<uint8_t> pad{false,0,0,0};
      DataCopyPad(packed[j*kPackedStride],bytes[off],copy,pad);
      j+=rows;
    }
    Fence<HardEvent::MTE2_V>();
  }
  __aicore__ void LoadPacked(int64_t start) {LoadPackedInto(start,packedBuf);}
  __aicore__ void Unpack(bool value) {
    oscar_ascend_padded_natural::Unpack<D,kHalfKv>(value,liveKvRows,error,packedBuf,wordBuf,
        planeBuf,naturalBuf,maskBuf,wordIndexBuf,laneIndexBuf,
        metadataIndexBuf,metadataHalfBuf,dequantBuf,scratchBuf);
  }
  __aicore__ void LoadWindowRuns(int64_t start,bool value) {
    auto dst=dequantBuf.Get<float>();
    auto batch=naturalBuf.Get<bfloat16_t>();
    // These buffers are dead after source0 unpack. Permanent unpack indices
    // (metadataIndexBuf/maskBuf/etc.) remain untouched for the next task.
    auto tags=packedBuf.Get<int64_t>();
    const int64_t first=start+lane*kHalfKv;
    const int32_t live=static_cast<int32_t>(Max64(0,Min64(kHalfKv,kvend-first)));
    const int64_t sinkCount=Min64(g.sink,context);
    const int64_t ring=g.recent+g.speculative;
    for(int32_t j=0;j<live;) {
      const int64_t candidate=first+j;
      const int64_t position=LogicalPosition(candidate);
      int32_t rows=static_cast<int32_t>(Min64(live-j,128-position%128));
      if(candidate<sinkCount)rows=static_cast<int32_t>(Min64(rows,sinkCount-candidate));
      int64_t block=0,inpage=0;
      if(!Physical(position,block,inpage)) {j+=rows;continue;}
      rows=static_cast<int32_t>(Min64(rows,g.blockTokens-inpage));
      int64_t windowRow;
      if(position<g.sink) {
        rows=static_cast<int32_t>(Min64(rows,g.sink-position));
        windowRow=position;
      } else {
        rows=static_cast<int32_t>(Min64(rows,ring-inpage%ring));
        windowRow=g.sink+inpage%ring;
      }
      // #126/#145: protect every old Vector/scalar owner before MTE2
      // overwrites the reused packed (tag) UB.
      Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();
      DataCopyExtParams tagCopy{1,static_cast<uint32_t>(rows*8),0,0,0};
      DataCopyPadExtParams<int64_t> tagPad{false,0,0,0};
      DataCopyPad(tags,wt[block*g.tagStride+windowRow],tagCopy,tagPad);
      Fence<HardEvent::MTE2_S>();
      // Invalid tags never cause a K/V read. All invalid and unused output
      // rows retain the exact initial +0, including poisoned ring slots.
      Duplicate(batch.ReinterpretCast<uint16_t>(),static_cast<uint16_t>(0),rows*D);
      Fence<HardEvent::V_MTE2>();
      for(int32_t begin=0;begin<rows;) {
        if(tags.GetValue(begin)!=inpage+begin) {++begin;continue;}
        int32_t end=begin+1;
        while(end<rows && tags.GetValue(end)==inpage+end)++end;
        const int64_t off=block*g.windowStride+((windowRow+begin)*g.hk+kvhead)*D;
        DataCopyExtParams copy{static_cast<uint16_t>(end-begin),D*2,
            static_cast<uint32_t>((g.hk-1)*D*2),0,0};
        DataCopyPadExtParams<bfloat16_t> pad{false,0,0,0};
        if(value)DataCopyPad(batch[begin*D],wv[off],copy,pad);
        else DataCopyPad(batch[begin*D],wk[off],copy,pad);
        begin=end;
      }
      Fence<HardEvent::MTE2_V>();
      Cast(dst[j*D],batch,RoundMode::CAST_NONE,rows*D);PipeBarrier<PIPE_V>();
      // Do NOT publish errors during the tag/DMA prepass. Original order is
      // physical->tag->finite for each row, K side first then V side. A later
      // bad tag must overwrite an earlier finite error, and vice versa.
      auto check=scratchBuf.Get<float>();
      for(int32_t row=0;row<rows;++row) {
        if(tags.GetValue(row)!=inpage+row) {error=4;continue;}
        ReduceSum(check,dst[(j+row)*D],check[16],D);Fence<HardEvent::V_S>();
        if(!Finite(check.GetValue(0)))error=2;
      }
      Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();
      j+=rows;
    }
    Fence<HardEvent::S_V>();
  }
  __aicore__ void LoadPrecise(int64_t start,bool value) {
    auto dst=dequantBuf.Get<float>();auto src=qBf16Buf.Get<bfloat16_t>();
    Duplicate(dst,0.0F,kElements);PipeBarrier<PIPE_V>();
    if(kind==2) {
      const int32_t rows=static_cast<int32_t>(Max64(0,Min64(kHalfKv,kvend-start-lane*kHalfKv)));
      if(rows==0)return;
      const int64_t token=requestBegin+start+lane*kHalfKv-context;
      if(token<0 || token+rows>g.tokens) {error=1;return;}
      // The natural unpack buffer is not used by the precise source. Reuse
      // it for contiguous BF16 rows; no extra UB or full-history buffer.
      auto batch=naturalBuf.Get<bfloat16_t>();
      Duplicate(batch.ReinterpretCast<uint16_t>(),static_cast<uint16_t>(0),kElements);
      Fence<HardEvent::V_MTE2>();
      DataCopyExtParams copy{static_cast<uint16_t>(rows),D*2,
          static_cast<uint32_t>((g.hk-1)*D*2),0,0};
      DataCopyPadExtParams<bfloat16_t> pad{false,0,0,0};
      const int64_t off=(token*g.hk+kvhead)*D;
      if(value)DataCopyPad(batch,cv[off],copy,pad);else DataCopyPad(batch,ck[off],copy,pad);
      Fence<HardEvent::MTE2_V>();Cast(dst,batch,RoundMode::CAST_NONE,kElements);
      PipeBarrier<PIPE_V>();
      auto check=scratchBuf.Get<float>();
      for(int32_t row=0;row<rows;++row) {
        ReduceSum(check,dst[row*D],check[16],D);Fence<HardEvent::V_S>();
        if(!Finite(check.GetValue(0)))error=2;
      }
      return;
    }
    if(kind==1) {LoadWindowRuns(start,value);return;}
    for(int32_t j=0;j<kHalfKv;++j) {
      const int64_t candidate=start+lane*kHalfKv+j;
      if(candidate>=kvend)continue;
      const int64_t position=LogicalPosition(candidate);
      {
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
  __aicore__ void PublishKv(bool value,int32_t subtile) {
    auto src=dequantBuf.Get<float>();
    const int64_t slot=subtile*(2*kHalfKv)+lane*kHalfKv;
    // Both K and V are natural [KV,D] FP32. QK requests B transpose=true;
    // PV requests false, as permitted by the same static B matmul type.
    Fence<HardEvent::V_MTE3>();
    DataCopy(work[(value?vOffset:kOffset)+slot*D],src,kElements);
    Fence<HardEvent::MTE3_V>();
  }
  __aicore__ void AddVisibleRange(int64_t begin,int64_t end,int64_t start,
      int32_t& lo0,int32_t& hi0,int32_t& lo1,int32_t& hi1) {
    if(end<=begin)return;
    const int32_t lo=static_cast<int32_t>(begin-start);
    const int32_t hi=static_cast<int32_t>(end-start);
    if(hi0==lo0) {lo0=lo;hi0=hi;}
    else if(lo<=hi0) hi0=static_cast<int32_t>(Max64(hi0,hi));
    else {lo1=lo;hi1=hi;}
  }
  __aicore__ void Softmax(int64_t start,int32_t rowBase) {
    auto scores=scoreBuf.Get<float>();
    auto acc=accBuf.Get<float>()[(rowBase-lane*kLaneRows)*D];
    auto stats=statsBuf.Get<float>();auto tmp=scratchBuf.Get<float>();
    const int64_t group=g.hq/g.hk;
    const int32_t activeRows=static_cast<int32_t>(Max64(0,
        Min64(kHalfRows,qcount*group-rowBase)));
    if(activeRows==0) {
      // Matmul SetSingleShape(M=real rows) excludes this full block from PV.
      // Keep the outer flag protocol, but publish no unused GM P rows.
      return;
    }
    DataCopy(scores,work[scoreOffset+rowBase*kKvRows],kHalfRows*kKvRows);
    Fence<HardEvent::MTE2_V>();Muls(scores,scores,g.scale,kHalfRows*kKvRows);
    Fence<HardEvent::V_S>();
    const int32_t softmaxRows=(activeRows+7)/8*8;
    if(activeRows<softmaxRows) {
      // -inf plus the finite empty-max sentinel gives P=0, sum=0 even when
      // this query has no prior visible tile. V2 never sees -inf - (-inf).
      Fence<HardEvent::S_V>();
      Duplicate(scores[activeRows*kKvRows],kNegativeInfinity,
          (softmaxRows-activeRows)*kKvRows);
      Fence<HardEvent::V_S>();
    }
    if(softmaxRows<kHalfRows) {
      Fence<HardEvent::S_V>();
      Duplicate(scores[softmaxRows*kKvRows],0.0F,
          (kHalfRows-softmaxRows)*kKvRows);
      Fence<HardEvent::V_S>();
    }
    // Archive #134/#140/D.4: every source has at most two visible candidate
    // intervals. Derive them once per query row instead of scanning all 128
    // candidate positions. The remaining mask writes and finite check are
    // unchanged, including padded tail columns and empty source rows.
    const int32_t jEnd=static_cast<int32_t>(Min64(kKvRows,kvend-start));
    const int64_t tileEnd=start+jEnd;
    const int64_t sinkCount=Min64(g.sink,context);
    const int64_t firstPosition=context+qbegin-requestBegin;
    const int64_t tailBegin=Min64(context,Max64(g.sink,firstPosition+1-g.recent));
    for(int32_t r=0;r<activeRows;++r) {
      const int64_t row=rowBase+r;
      const int64_t position=context+qbegin+row/group-requestBegin;
      int32_t lo0=0,hi0=0,lo1=0,hi1=0;
      const int64_t cut=Max64(g.sink,position+1-g.recent);
      if(kind==0) {
        AddVisibleRange(Max64(start,g.sink),
            Min64(tileEnd,Min64(context,Min64(cut,position+1))),start,
            lo0,hi0,lo1,hi1);
      } else if(kind==1) {
        // LogicalPosition maps [0,sinkCount) to the exact sink, and the
        // remaining candidate interval to [tailBegin,context). Every query
        // position is >=context, so only the recent cut affects the tail.
        AddVisibleRange(Max64(start,0),Min64(tileEnd,sinkCount),start,
            lo0,hi0,lo1,hi1);
        AddVisibleRange(Max64(start,sinkCount+Max64(0,cut-tailBegin)),
            Min64(tileEnd,sinkCount+context-tailBegin),start,
            lo0,hi0,lo1,hi1);
      } else {
        AddVisibleRange(Max64(start,context),Min64(tileEnd,position+1),start,
            lo0,hi0,lo1,hi1);
      }
      const bool any=hi0>lo0||hi1>lo1;
      if(!any) {Fence<HardEvent::S_V>();Duplicate(scores[r*kKvRows],kNegativeInfinity,kKvRows);
        Fence<HardEvent::V_S>();continue;}
      PipeBarrier<PIPE_V>();
      ReduceSum(tmp,scores[r*kKvRows],tmp[16],kKvRows);Fence<HardEvent::V_S>();
      if(!Finite(tmp.GetValue(0)))error=2;
      if(lo0>0 || hi0<kKvRows) {
        oscar_range_mask::MaskComplement(scores[r*kKvRows],lo0,hi0,lo1,hi1);
        Fence<HardEvent::V_S>();
      }
    }
    // The two 16-row max/sum arrays persist across KV work units. Shared
    // 16-row input arrays let CANN's FP32 WITHOUT_BRC V2 update one score
    // block at a time without spilling the running state to GM.
    const int32_t block=rowBase-lane*kLaneRows;
    auto outMax=stats[block],outSum=stats[kLaneRows+block];
    auto inMax=stats[2*kLaneRows],inSum=stats[2*kLaneRows+kHalfRows];
    Adds(inMax,outMax,0.0F,softmaxRows);
    Adds(inSum,outSum,0.0F,softmaxRows);
    PipeBarrier<PIPE_V>();
    // All K/V staging into GM finished before CubeReady. The final dequant
    // UB values are dead until the next outer KV tile, so its D64/128/256
    // capacity gives the SDK 4/8/16KiB of temporary space with no new UB.
    auto softWork=dequantBuf.Get<float>();
    const SoftMaxShapeInfo shape{static_cast<uint32_t>(softmaxRows),
        static_cast<uint32_t>(kKvRows),static_cast<uint32_t>(softmaxRows),
        static_cast<uint32_t>(kKvRows)};
    const auto tiling=SoftMaxFlashV2TilingFunc(shape,sizeof(float),sizeof(float),
        softWork.GetSize()*sizeof(float),true,false);
    SoftmaxFlashV2<float,true,true,false,false,kCvSoftmaxConfig>(
        scores,outSum,outMax,scores,tmp,inSum,inMax,
        softWork.ReinterpretCast<uint8_t>(),tiling,shape);
    Fence<HardEvent::V_S>();
    // Reuse the now-dead natural unpack UB for the same row-strided Brcb
    // pattern as the validated scale/zero path. This multiplies all active
    // accumulator rows by their FP32 online alpha without scalar readback.
    auto alphaBlocks=naturalBuf.Get<float>();
    Brcb(alphaBlocks,tmp,static_cast<uint8_t>((activeRows+7)/8),{1,8});
    PipeBarrier<PIPE_V>();
    const BinaryRepeatParams alphaParams{1,1,0,static_cast<uint8_t>(D/8),
        static_cast<uint8_t>(D/8),1};
    for(int32_t chunk=0;chunk<D/64;++chunk)
      Mul(acc[chunk*64],acc[chunk*64],alphaBlocks,static_cast<uint64_t>(64),
          static_cast<uint8_t>(activeRows),alphaParams);
    PipeBarrier<PIPE_V>();
    Fence<HardEvent::V_MTE3>();
    // V2 may use padded UB rows to satisfy its 8-row shape, but Cube PV has
    // M=qcount*group and consumes only the real P rows from this block.
    DataCopy(work[pOffset+rowBase*kKvRows],scores,activeRows*kKvRows);
    // The next Softmax block refills scoreBuf on MTE2, not Vector. MTE3_V
    // cannot prevent that DMA from overwriting P while MTE3 still reads it.
    // The next copy-in is followed by MTE2_V before any Vector score writes.
    Fence<HardEvent::MTE3_MTE2>();
  }
  __aicore__ void VectorTask(int64_t id) {
    error=0;
    auto acc=accBuf.Get<float>();auto stats=statsBuf.Get<float>();
    Duplicate(acc,0.0F,kLaneRows*D);Fence<HardEvent::V_S>();
    Duplicate(stats,kEmptyMax,kLaneRows);
    Duplicate(stats[kLaneRows],0.0F,kLaneRows);
    Fence<HardEvent::V_S>();
    LoadQueries();
    for(int64_t start=kvbegin;start<kvend;start+=kKvRows) {
      // Two 128-token subtiles fill one bounded 256-token K/V work unit.
      // Packed history is read once per subtile and reused for K and V;
      // all incomplete tail slots are published as zero before Cube QK.
      for(int32_t subtile=0;subtile<kKvSubtiles;++subtile) {
        const int64_t subStart=start+subtile*(2*kHalfKv);
        // A live 64-row producer chunk initializes its entire tail.
        // TailColumns rounds only to16, so skipped chunks are unread.
        if(subStart+lane*kHalfKv>=kvend)continue;
        if(kind==0) {LoadPacked(subStart);Unpack(false);}
        else LoadPrecise(subStart,false);
        PublishKv(false,subtile);
        if(kind==0)Unpack(true);else LoadPrecise(subStart,true);
        PublishKv(true,subtile);
      }
      if(TailColumns(start)<kKvRows)ClearPaddedScores();
      CrossCoreSetFlag<2,PIPE_MTE3>(kVectorReady);
      CrossCoreWaitFlag(kCubeReady);
      for(int32_t block=0;block<kRowBlocks;++block)
        Softmax(start,lane*kLaneRows+block*kHalfRows);
      CrossCoreSetFlag<2,PIPE_MTE3>(kVectorReady);
      CrossCoreWaitFlag(kCubeReady);
      // P is fully consumed by Cube before this point. Reuse the now-dead
      // dequant UB for bounded PV reads; the 16-row FP32 accumulator
      // stays in UB across every 256-KV unit in the same query task.
      auto result=dequantBuf.Get<float>();
      const int32_t activeRows=static_cast<int32_t>(qcount*(g.hq/g.hk));
      for(int32_t row=lane*kLaneRows;row<(lane+1)*kLaneRows && row<activeRows;
          row+=kHalfKv) {
        const int32_t count=static_cast<int32_t>(Min64(kHalfKv,
            Min64(activeRows,(lane+1)*kLaneRows)-row));
        DataCopy(result,work[pvOffset+row*D],count*D);
        Fence<HardEvent::MTE2_V>();
        auto target=acc[(row-lane*kLaneRows)*D];
        Add(target,target,result,count*D);PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_MTE2>();
      }
      CrossCoreSetFlag<2,PIPE_MTE2>(kVectorReady);
    }
    Fence<HardEvent::V_S>();
    for(int32_t r=0;r<kLaneRows;++r) {
      const float sum=stats.GetValue(kLaneRows+r);
      if(sum>0.0F && Finite(sum))Muls(acc[r*D],acc[r*D],1.0F/sum,D);
      else {if(sum!=0.0F)error=2;Duplicate(acc[r*D],0.0F,D);}
      PipeBarrier<PIPE_V>();
    }
    if(kind==0 && kvbegin<kvend) {
      Fence<HardEvent::V_MTE3>();DataCopy(work[qOffset+lane*kLaneRows*D],acc,kLaneRows*D);
      CrossCoreSetFlag<2,PIPE_MTE3>(kVectorReady);
      CrossCoreWaitFlag(kCubeReady);
      DataCopy(acc,work[rotOffset+lane*kLaneRows*D],kLaneRows*D);
      Fence<HardEvent::MTE2_V>();CrossCoreSetFlag<2,PIPE_MTE2>(kVectorReady);
    }
    PublishRows(id);
  }

  // source0 only: cache one next packed tile, split K/V dequant across the
  // current QK/PV windows. Each array is bounded by two HalfKv64 subtiles.
  __aicore__ void PipelineUnpack(TBuf<TPosition::VECCALC>& packedInput,bool value) {
    oscar_ascend_padded_natural::Unpack<D,kHalfKv>(value,liveKvRows,error,packedInput,wordBuf,
        planeBuf,naturalBuf,maskBuf,wordIndexBuf,laneIndexBuf,metadataIndexBuf,
        metadataHalfBuf,dequantBuf,scratchBuf);
  }
  __aicore__ void PipelinePublishKv(bool value,int32_t subtile,int32_t buffer) {
    const int64_t slot=subtile*(2*kHalfKv)+lane*kHalfKv;
    Fence<HardEvent::V_MTE3>();
    DataCopy(work[(value?vOffset:kOffset)+buffer*(2*kKvRows*D)+slot*D],
        dequantBuf.Get<float>(),kElements);
    Fence<HardEvent::MTE3_V>();
  }
  __aicore__ void PrefetchKeys(int64_t start,int32_t buffer) {
    const int32_t savedError=error;
    for(int32_t sub=0;sub<kKvSubtiles;++sub) {
      auto& packedInput=sub==0?packedBuf:packedNextBuf;
      error=0;
      LoadPackedInto(start+sub*(2*kHalfKv),packedInput);
      pendingLive[sub]=liveKvRows;
      PipelineUnpack(packedInput,false);
      pendingKeyError[sub]=error;
      PipelinePublishKv(false,sub,buffer);
    }
    error=savedError;
  }
  __aicore__ void PrefetchValues(int32_t buffer) {
    const int32_t savedError=error;
    for(int32_t sub=0;sub<kKvSubtiles;++sub) {
      auto& packedInput=sub==0?packedBuf:packedNextBuf;
      error=0;liveKvRows=pendingLive[sub];
      PipelineUnpack(packedInput,true);
      pendingValueError[sub]=error;
      PipelinePublishKv(true,sub,buffer);
    }
    error=savedError;
  }
  __aicore__ void CommitPrefetchErrors() {
    // Baseline order is Load/K0,V0,Load/K1,V1. Speculative loads may run
    // earlier, but their errors must not overwrite an older score failure
    // until the previous PV accumulation has completed.
    for(int32_t sub=0;sub<kKvSubtiles;++sub) {
      if(pendingKeyError[sub])error=pendingKeyError[sub];
      if(pendingValueError[sub])error=pendingValueError[sub];
    }
  }
  __aicore__ void PipelineConsumePv() {
    auto result=dequantBuf.Get<float>();auto acc=accBuf.Get<float>();
    const int32_t activeRows=static_cast<int32_t>(qcount*(g.hq/g.hk));
    for(int32_t row=lane*kLaneRows;row<(lane+1)*kLaneRows && row<activeRows;
        row+=kHalfKv) {
      const int32_t count=static_cast<int32_t>(Min64(kHalfKv,
          Min64(activeRows,(lane+1)*kLaneRows)-row));
      DataCopy(result,work[pvOffset+row*D],count*D);
      Fence<HardEvent::MTE2_V>();
      auto target=acc[(row-lane*kLaneRows)*D];
      Add(target,target,result,count*D);PipeBarrier<PIPE_V>();
      Fence<HardEvent::V_MTE2>();
    }
  }
  __aicore__ void HistoryCubeTask() {
    const int32_t rows=static_cast<int32_t>(qcount*(g.hq/g.hk));
    int32_t buffer=0;
    for(int64_t start=kvbegin;start<kvend;start+=kKvRows,buffer^=1) {
      CrossCoreWaitFlag(kKvReady);
      Matmul(qOffset,kOffset+buffer*(2*kKvRows*D),scoreOffset,rows,kKvRows,D);
      CrossCoreSetFlag<2,PIPE_FIX>(kScoreReady);
      CrossCoreWaitFlag(kPReady);
      Matmul(pOffset,vOffset+buffer*(2*kKvRows*D),pvOffset,rows,D,kKvRows,false,false);
      CrossCoreSetFlag<2,PIPE_FIX>(kPvReady);
      // Next QK may overwrite score/P after this PV has finished reading it.
      // The next PReady cannot be sent before Vector consumes this PV, so
      // the one PV result buffer cannot be overwritten early.
    }
    CrossCoreWaitFlag(kNormReady);
    Matmul(qOffset,0,rotOffset,rows,D,D,true);
    CrossCoreSetFlag<2,PIPE_FIX>(kRotReady);
    CrossCoreWaitFlag(kTaskDone);
  }
  __aicore__ void HistoryVectorTask(int64_t id) {
    error=0;
    auto acc=accBuf.Get<float>();auto stats=statsBuf.Get<float>();
    Duplicate(acc,0.0F,kLaneRows*D);Fence<HardEvent::V_S>();
    Duplicate(stats,kEmptyMax,kLaneRows);
    Duplicate(stats[kLaneRows],0.0F,kLaneRows);Fence<HardEvent::V_S>();
    LoadQueries();
    for(int32_t sub=0;sub<kKvSubtiles;++sub) {
      LoadPacked(kvbegin+sub*(2*kHalfKv));
      Unpack(false);PipelinePublishKv(false,sub,0);
      Unpack(true);PipelinePublishKv(true,sub,0);
    }
    CrossCoreSetFlag<2,PIPE_MTE3>(kKvReady);
    int32_t buffer=0;
    for(int64_t start=kvbegin;start<kvend;start+=kKvRows,buffer^=1) {
      const bool hasNext=start+kKvRows<kvend;
      // Packed buffers, metadata scratch and dequant UB are not used by
      // Cube. K is fully published before Softmax reuses dequant UB.
      if(hasNext)PrefetchKeys(start+kKvRows,buffer^1);
      CrossCoreWaitFlag(kScoreReady);
      for(int32_t block=0;block<kRowBlocks;++block)
        Softmax(start,lane*kLaneRows+block*kHalfRows);
      CrossCoreSetFlag<2,PIPE_MTE3>(kPReady);
      if(hasNext) {
        PrefetchValues(buffer^1);
        CrossCoreSetFlag<2,PIPE_MTE3>(kKvReady);
      }
      CrossCoreWaitFlag(kPvReady);
      PipelineConsumePv();
      if(hasNext)CommitPrefetchErrors();
    }
    Fence<HardEvent::V_S>();
    for(int32_t r=0;r<kLaneRows;++r) {
      const float sum=stats.GetValue(kLaneRows+r);
      if(sum>0.0F && Finite(sum))Muls(acc[r*D],acc[r*D],1.0F/sum,D);
      else {if(sum!=0.0F)error=2;Duplicate(acc[r*D],0.0F,D);}
      PipeBarrier<PIPE_V>();
    }
    Fence<HardEvent::V_MTE3>();DataCopy(work[qOffset+lane*kLaneRows*D],acc,kLaneRows*D);
    CrossCoreSetFlag<2,PIPE_MTE3>(kNormReady);
    CrossCoreWaitFlag(kRotReady);
    DataCopy(acc,work[rotOffset+lane*kLaneRows*D],kLaneRows*D);
    Fence<HardEvent::MTE2_V>();
    PublishRows(id);
    CrossCoreSetFlag<2,PIPE_MTE3>(kTaskDone);
  }

  __aicore__ void PublishRows(int64_t id) {
    auto acc=accBuf.Get<float>();auto stats=statsBuf.Get<float>();auto scalar=scratchBuf.Get<float>();
    const int64_t group=g.hq/g.hk,rows=qcount*group;
    Fence<HardEvent::V_S>();
    for(int32_t r=0;r<kLaneRows && lane*kLaneRows+r<rows;++r) {
      const int64_t row=lane*kLaneRows+r;
      const int64_t outRow=((qbegin+row/group)*g.hq+kvhead*group+row%group)*3*g.splits+split;
      const float sum=stats.GetValue(kLaneRows+r),maximum=stats.GetValue(r);
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
    if(qbegin>=0 && qbegin<g.tokens && g.hk>0 && g.hq>0 &&
        g.hq%g.hk==0 && g.hq/g.hk<=8 && kvhead>=0 && kvhead<g.hk &&
        split>=0 && split<3*g.splits) {
      qcount=1;error=code;
      auto stats=statsBuf.Get<float>();auto acc=accBuf.Get<float>();
      Duplicate(acc,code?__builtin_nanf(""):0.0F,kLaneRows*D);Fence<HardEvent::V_S>();
      Duplicate(stats,kEmptyMax,kLaneRows);
      Duplicate(stats[kLaneRows],0.0F,kLaneRows);
      Fence<HardEvent::V_S>();
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
  TBuf<TPosition::VECCALC> taskBuf,packedBuf,packedNextBuf,planeBuf,naturalBuf,wordBuf,maskBuf,
      wordIndexBuf,laneIndexBuf,metadataIndexBuf,metadataHalfBuf,dequantBuf,
      scoreBuf,accBuf,qFloatBuf,qBf16Buf,statsBuf,scratchBuf,addressBuf;
  int64_t coreIndex,qbegin,qcount,kvhead,kvbegin,kvend,request,split,kind,context,requestBegin;
  int64_t qOffset,kOffset,vOffset,scoreOffset,pOffset,pvOffset,rotOffset;
  int32_t lane=0,error=0,taskError=0,liveKvRows=0;
  int32_t pendingLive[kKvSubtiles],pendingKeyError[kKvSubtiles],pendingValueError[kKvSubtiles];
};
}
#define OSCAR_CV_ARGUMENTS GM_ADDR query, GM_ADDR queryRot, GM_ADDR key, GM_ADDR value, \
    GM_ADDR rotation, GM_ADDR raw, GM_ADDR table, GM_ADDR wk, GM_ADDR wv, GM_ADDR tags, \
    GM_ADDR tasks, GM_ADDR output, GM_ADDR lse, GM_ADDR status, GM_ADDR workspace, \
    int64_t tokens, int64_t hq, int64_t hk, int64_t dim, int64_t requests, \
    int64_t columns, int64_t taskCount, int64_t blockTokens, int64_t blocks, \
    int64_t ssmOffset, int64_t pageStride, int64_t windowStride, int64_t tagStride, \
    int64_t sink, int64_t recent, int64_t speculative, int64_t splits, float scale
#define OSCAR_CV_RUN(owner) \
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2); \
  if(dim!=256) return; \
  const Geometry g{tokens,hq,hk,requests,columns,taskCount,blockTokens,blocks,ssmOffset, \
      pageStride,windowStride,tagStride,sink,recent,speculative,splits,scale}; \
  AttentionCv<256,owner> op; \
  op.Init(query,queryRot,key,value,rotation,raw,table,wk,wv,tags, \
      tasks,output,lse,status,workspace,g); \
  op.Process()
extern "C" __global__ __aicore__ void oscar_attention_cv_decode_bundle_kernel(
    OSCAR_CV_ARGUMENTS) { OSCAR_CV_RUN(4); }
extern "C" __global__ __aicore__ void oscar_attention_cv_decode_bundle_q1_kernel(
    OSCAR_CV_ARGUMENTS) { OSCAR_CV_RUN(1); }
#undef OSCAR_CV_RUN

#ifndef ASCENDC_CPU_DEBUG
namespace oscar_ascend {
#define OSCAR_STRIPED_LAUNCH(name,kernel) \
void name(void* stream,void* query,void* queryRot,void* key,void* value, \
    void* rotation,void* raw,void* table,void* wk,void* wv,void* tags,void* tasks, \
    void* output,void* lse,void* status,void* workspace,int64_t tokens,int64_t hq, \
    int64_t hk,int64_t dim,int64_t requests,int64_t columns,int64_t taskCount, \
    int64_t blockTokens,int64_t blocks,int64_t ssmOffset,int64_t pageStride, \
    int64_t windowStride,int64_t tagStride,int64_t sink,int64_t recent, \
    int64_t speculative,int64_t splits,float scale,uint32_t cores) { \
  kernel<<<cores,nullptr,stream>>>( \
      static_cast<uint8_t*>(query),static_cast<uint8_t*>(queryRot), \
      static_cast<uint8_t*>(key),static_cast<uint8_t*>(value), \
      static_cast<uint8_t*>(rotation),static_cast<uint8_t*>(raw), \
      static_cast<uint8_t*>(table),static_cast<uint8_t*>(wk),static_cast<uint8_t*>(wv), \
      static_cast<uint8_t*>(tags),static_cast<uint8_t*>(tasks),static_cast<uint8_t*>(output), \
      static_cast<uint8_t*>(lse),static_cast<uint8_t*>(status),static_cast<uint8_t*>(workspace), \
      tokens,hq,hk,dim,requests,columns,taskCount,blockTokens,blocks,ssmOffset, \
      pageStride,windowStride,tagStride,sink,recent,speculative,splits,scale); \
}
OSCAR_STRIPED_LAUNCH(attention_cv_decode_bundle_launch,
    oscar_attention_cv_decode_bundle_kernel)
OSCAR_STRIPED_LAUNCH(attention_cv_decode_bundle_q1_launch,
    oscar_attention_cv_decode_bundle_q1_kernel)
#undef OSCAR_STRIPED_LAUNCH
}
#endif
#undef OSCAR_CV_ARGUMENTS
