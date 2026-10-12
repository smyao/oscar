// SPDX-License-Identifier: Apache-2.0
// ISOLATED EXPERIMENT, source0 only. Archive #126/#129/#145/#148-155.
// D.4 four questions: (1) historical INT2 expansion + fused FIA; (2) former
// full-history restore spent 6499.8-6655.1ms/card, query groups repeat work;
// (3) share one <=8192 rotated FP32 KV block, preserve original KV256 order,
// QK/PV FP32, softmax recurrence, per-lane errors and final RvT once. Full
// tiles read shared GM directly; task-local tails copy/zero privately. Raw
// FP32 acc/max/sum spill only at 8K boundaries. No BF16/inverse restore;
// (4) removes repeated unpack/publication, but adds bounded state/Q traffic.
// No target latency or whole-run gain is claimed by compilation/CPU parity.
// Math methods retained from this project's fe0 attention_cv.cpp. The
// production implementation and route remain unchanged.
#include "../../kernels/oscar_common.h"
#include "../../include/oscar_attention_launch.h"
#define OSCAR_SCHEDULE_FN __aicore__ inline
#include "../../include/oscar_cv_schedule.h"
#undef OSCAR_SCHEDULE_FN
#include "lib/matmul_intf.h"
#include "lib/matmul/constant_tiling.h"
#include "adv_api/activation/softmaxflashv2.h"
using namespace oscar_ascend_device;
namespace {
constexpr int32_t kQueryRows=oscar_ascend::kAttentionQueryRows;
constexpr int32_t kKvRows=oscar_ascend::kAttentionKvRows;
constexpr int32_t kHalfRows=32, kHalfKv=16;
constexpr int32_t kLaneRows=kQueryRows/2, kRowBlocks=kLaneRows/kHalfRows;
constexpr int32_t kKvSubtiles=kKvRows/(2*kHalfKv);
constexpr uint16_t kVectorReady=8, kCubeReady=9;
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

template<int32_t D> class SharedHistoryCv {
  static constexpr int32_t kSlotBytes=D/2+8;
  static constexpr int32_t kPackedStride=(kSlotBytes+31)/32*32;
  static constexpr int32_t kElements=kHalfKv*D;
  static constexpr int32_t kWords=kElements/8;
 public:
  __aicore__ void Init(GM_ADDR query,GM_ADDR queryRot,GM_ADDR key,GM_ADDR value,
      GM_ADDR rotation,GM_ADDR raw,GM_ADDR table,GM_ADDR windowKey,
      GM_ADDR windowValue,GM_ADDR tags,GM_ADDR tasks,GM_ADDR partial,
      GM_ADDR lse,GM_ADDR status,GM_ADDR workspace,GM_ADDR sharedKey,GM_ADDR sharedValue,
      GM_ADDR sharedErrors,GM_ADDR state, int64_t tileStart,int64_t tileEnd,const Geometry& geometry) {
    g=geometry;blockStart=tileStart;blockEnd=tileEnd;
    sharedK.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sharedKey));
    sharedV.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sharedValue));
    rowErrors.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sharedErrors));
    saved.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(state));
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
    const int64_t floatsPerCore=oscar_ascend::attention_workspace_per_core(D)/4;
    work.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace)+core*floatsPerCore);
    // Q(128D), K(256D), natural V(256D), score/P(128*256),
    // PV/rotation(128D). Query accumulator stays in UB across KV units.
    // QK is complete before Vector overwrites score with P; PV is consumed
    // before the final Rv rotation reuses that same GM tile. No active aliases.
    qOffset=0;kOffset=kQueryRows*D;vOffset=(kQueryRows+kKvRows)*D;
    scoreOffset=(kQueryRows+2*kKvRows)*D;
    pOffset=scoreOffset;pvOffset=pOffset+kQueryRows*kKvRows;rotOffset=pvOffset;
    if ASCEND_IS_AIC {
      // Static tiling is compiler-derived by the real SDK. SetOrg/SingleShape
      // below specify each QK/PV/rotation matrix, never hand-written tiling POD.
      mm.Init(static_cast<const TCubeTiling*>(nullptr),&pipe);
      SetHF32Mode(false);
    } else {
      pipe.InitBuffer(taskBuf,128);pipe.InitBuffer(naturalBuf,kElements*2);
      pipe.InitBuffer(dequantBuf,kElements*4);
      pipe.InitBuffer(scoreBuf,kHalfRows*kKvRows*4);
      pipe.InitBuffer(accBuf,kLaneRows*D*4);
      pipe.InitBuffer(qFloatBuf,D*4);pipe.InitBuffer(qBf16Buf,D*2);
      pipe.InitBuffer(statsBuf,(2*kLaneRows+2*kHalfRows)*4);pipe.InitBuffer(scratchBuf,512);
      pipe.InitBuffer(addressBuf,128);
    }
  }
  __aicore__ void Process() {
    // Scalar Cube metadata reads must invalidate stale cache on changed-input
    // graph replay. The Vector side exclusively DMA-loads mutable metadata.
    if ASCEND_IS_AIC {
      DataCacheCleanAndInvalid<int64_t,CacheLine::ENTIRE_DATA_CACHE>(taskGm);
    }
    const int64_t queryTile=kQueryRows/(g.hq/g.hk);
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
        if(kind!=0)continue;
        if(qcount>0 && kvend<=blockStart && kvbegin<kvend)continue;
        tileBegin=Max64(kvbegin,blockStart);tileEnd=Min64(kvend,blockEnd);
        finish=kvend<=blockEnd;
        if(qcount>0 && (request!=0 || split!=0 || kvhead!=0 ||
            kvbegin!=g.sink || qbegin%queryTile ||
            (!finish && blockEnd-blockStart!=8192) ||
            (tileBegin-kvbegin)%kKvRows)) {trap();return;}
        if(qcount==0) {if ASCEND_IS_AIV {PublishStatus(id,0);} continue;}
        if(qcount<0 || taskError || !TaskValid()) {
          if ASCEND_IS_AIV {PublishEmpty(id,taskError?taskError:(!TaskValid()?1:0));}
          continue;
        }
        if ASCEND_IS_AIC {CubeTask();}
        else {VectorTask(id);}
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
    return qbegin>=0 && qbegin<g.tokens && qcount<=kQueryRows/(g.hq/g.hk) &&
        (qcount<0 || qbegin+qcount<=g.tokens) && kvhead>=0 && kvhead<g.hk &&
        kvbegin>=0 && kvend>=kvbegin && request>=0 && request<g.requests &&
        split>=0 && split<3*g.splits && kind>=0 && kind<3 && context>=0 &&
        requestBegin>=0 && requestBegin<=qbegin &&
        (kind!=0 || kvend<=context) &&
        (kind!=2 || (kvbegin>=context && requestBegin+kvend-context<=g.tokens));
  }
  __aicore__ void Matmul(int64_t a,int64_t b,int64_t c,int32_t m,int32_t n,int32_t k,
      bool rotation=false,bool transposeB=true) {
    SetHF32Mode(false); // no implicit HF32/TF32 relaxation of the frozen oracle.
    mm.SetOrgShape(m,n,k);mm.SetSingleShape(m,n,k);
    mm.SetTensorA(work[a],false);
    if(rotation) mm.SetTensorB(rv,true);else mm.SetTensorB(work[b],transposeB);
    mm.template IterateAll<true>(work[c]);
    // End closes the iteration state; buffers remain initialized for reuse.
    mm.End();
  }
  __aicore__ void CubeTask() {
    const int32_t rows=static_cast<int32_t>(qcount*(g.hq/g.hk));
    for(int64_t start=tileBegin;start<tileEnd;start+=kKvRows) {
      CrossCoreWaitFlag(kVectorReady);
      if(start+kKvRows<=kvend)SharedMatmul(start,false,rows);
      else Matmul(qOffset,kOffset,scoreOffset,rows,kKvRows,D);
      CrossCoreSetFlag<2,PIPE_FIX>(kCubeReady);
      CrossCoreWaitFlag(kVectorReady);
      if(start+kKvRows<=kvend)SharedMatmul(start,true,rows);
      else Matmul(pOffset,vOffset,pvOffset,rows,D,kKvRows,false,false);
      CrossCoreSetFlag<2,PIPE_FIX>(kCubeReady);
      // Vector consumes PV before the next tile may overwrite shared buffers.
      CrossCoreWaitFlag(kVectorReady);
    }
    if(finish && kind==0 && kvbegin<kvend) {
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
      // fe0 checks Q once before the first historical KV256. Repeating the
      // check after restoring state can overwrite a later metadata error3.
      if(blockStart<=kvbegin) {
        auto check=scratchBuf.Get<float>();
        ReduceSum(check,dst,check[16],D);Fence<HardEvent::V_S>();
        if(!Finite(check.GetValue(0)))error=2;
      }
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
  __aicore__ void SharedMatmul(int64_t start,bool value,int32_t rows) {
    SetHF32Mode(false);
    if(value) {
      mm.SetOrgShape(rows,D,kKvRows);mm.SetSingleShape(rows,D,kKvRows);
      mm.SetTensorA(work[pOffset],false);
      mm.SetTensorB(sharedV[(start-blockStart)*D],false);
      mm.template IterateAll<true>(work[pvOffset]);
    }else {
      mm.SetOrgShape(rows,kKvRows,D);mm.SetSingleShape(rows,kKvRows,D);
      mm.SetTensorA(work[qOffset],false);
      mm.SetTensorB(sharedK[(start-blockStart)*D],true);
      mm.template IterateAll<true>(work[scoreOffset]);
    }
    mm.End();
  }
  __aicore__ void CheckRows(int64_t start) {
    const int64_t first=start+lane*kHalfKv;
    liveKvRows=static_cast<int32_t>(Max64(0,Min64(kHalfKv,kvend-first)));
    if(liveKvRows<=0)return;
    auto codes=addressBuf.Get<int32_t>();
    DataCopyExtParams copy{1,static_cast<uint32_t>(liveKvRows*4),0,0,0};
    DataCopyPadExtParams<int32_t> pad{false,0,0,0};
    DataCopyPad(codes,rowErrors[first-blockStart],copy,pad);
    Fence<HardEvent::MTE2_S>();
    for(int32_t j=0;j<liveKvRows;++j)if(codes.GetValue(j))error=codes.GetValue(j);
    Fence<HardEvent::S_MTE2>();
  }
  __aicore__ void LoadShared(int64_t start,bool value) {
    auto dst=dequantBuf.Get<float>();
    Fence<HardEvent::MTE3_V>();
    Duplicate(dst,0.0F,kElements);Fence<HardEvent::V_MTE2>();
    if(liveKvRows>0) {
      const int64_t offset=(start+lane*kHalfKv-blockStart)*D;
      if(value)DataCopy(dst,sharedV[offset],liveKvRows*D);
      else DataCopy(dst,sharedK[offset],liveKvRows*D);
    }
    Fence<HardEvent::MTE2_V>();
  }
  __aicore__ int64_t StateOffset() {
    // Prototype admits one request, Hq6/Hkv1, S1 and original qtile21.
    return (qbegin/(kQueryRows/(g.hq/g.hk))*2+lane)*(kLaneRows*D+2*kLaneRows+8);
  }
  __aicore__ void LoadState() {
    const int64_t off=StateOffset();
    auto acc=accBuf.Get<float>();auto stats=statsBuf.Get<float>();
    DataCopy(acc,saved[off],kLaneRows*D);
    DataCopy(stats,saved[off+kLaneRows*D],2*kLaneRows);
    DataCopy(scratchBuf.Get<float>(),saved[off+kLaneRows*D+2*kLaneRows],8);
    Fence<HardEvent::MTE2_S>();
    error=scratchBuf.Get<int32_t>().GetValue(0);
    Fence<HardEvent::MTE2_V>();
  }
  __aicore__ void SaveState() {
    const int64_t off=StateOffset();
    Fence<HardEvent::V_MTE3>();
    DataCopy(saved[off],accBuf.Get<float>(),kLaneRows*D);
    DataCopy(saved[off+kLaneRows*D],statsBuf.Get<float>(),2*kLaneRows);
    Fence<HardEvent::V_S>();
    scratchBuf.Get<int32_t>().SetValue(0,error);Fence<HardEvent::S_MTE3>();
    DataCopy(saved[off+kLaneRows*D+2*kLaneRows],scratchBuf.Get<float>(),8);
    Fence<HardEvent::MTE3_MTE2>();Fence<HardEvent::MTE3_V>();
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
        Fence<HardEvent::S_V>();
        for(int32_t j=0;j<kKvRows;++j) {
          const bool visible=(j>=lo0&&j<hi0)||(j>=lo1&&j<hi1);
          if(!visible)scores.SetValue(r*kKvRows+j,kNegativeInfinity);
        }
        Fence<HardEvent::S_V>();
      }
    }
    // The two 64-row max/sum arrays persist across KV work units. Two shared
    // 32-row input arrays let CANN's FP32 WITHOUT_BRC V2 update one score
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
    if(blockStart>kvbegin)LoadState();
    LoadQueries();
    for(int64_t start=tileBegin;start<tileEnd;start+=kKvRows) {
      // Full KV256 reads are shared directly by Cube. Only an original
      // task's partial tail makes a private copy, zero beyond original kvend.
      for(int32_t subtile=0;subtile<kKvSubtiles;++subtile) {
        const int64_t subStart=start+subtile*(2*kHalfKv);
        CheckRows(subStart);
        if(start+kKvRows>kvend) {
          LoadShared(subStart,false);PublishKv(false,subtile);
          LoadShared(subStart,true);PublishKv(true,subtile);
        }
      }
      CrossCoreSetFlag<2,PIPE_MTE3>(kVectorReady);
      CrossCoreWaitFlag(kCubeReady);
      for(int32_t block=0;block<kRowBlocks;++block)
        Softmax(start,lane*kLaneRows+block*kHalfRows);
      CrossCoreSetFlag<2,PIPE_MTE3>(kVectorReady);
      CrossCoreWaitFlag(kCubeReady);
      // P is fully consumed by Cube before this point. Reuse the now-dead
      // dequant UB for bounded 16-row PV reads; the 64-row FP32 accumulator
      // stays in UB across every 256-KV unit in the same query task.
      auto result=dequantBuf.Get<float>();
      const int32_t activeRows=static_cast<int32_t>(qcount*(g.hq/g.hk));
      for(int32_t row=lane*kLaneRows;row<(lane+1)*kLaneRows && row<activeRows;
          row+=kHalfKv) {
        const int32_t count=static_cast<int32_t>(Min64(kHalfKv,activeRows-row));
        DataCopy(result,work[pvOffset+row*D],count*D);
        Fence<HardEvent::MTE2_V>();
        auto target=acc[(row-lane*kLaneRows)*D];
        Add(target,target,result,count*D);PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_MTE2>();
      }
      CrossCoreSetFlag<2,PIPE_MTE2>(kVectorReady);
    }
    if(!finish) {SaveState();return;}
    Fence<HardEvent::V_S>();
    for(int32_t r=0;r<kLaneRows;++r) {
      const float sum=stats.GetValue(kLaneRows+r);
      if(sum>0.0F && Finite(sum))Muls(acc[r*D],acc[r*D],1.0F/sum,D);
      else {if(sum!=0.0F)error=2;Duplicate(acc[r*D],0.0F,D);}
      PipeBarrier<PIPE_V>();
    }
    if(finish && kind==0 && kvbegin<kvend) {
      Fence<HardEvent::V_MTE3>();DataCopy(work[qOffset+lane*kLaneRows*D],acc,kLaneRows*D);
      CrossCoreSetFlag<2,PIPE_MTE3>(kVectorReady);
      CrossCoreWaitFlag(kCubeReady);
      DataCopy(acc,work[rotOffset+lane*kLaneRows*D],kLaneRows*D);
      Fence<HardEvent::MTE2_V>();CrossCoreSetFlag<2,PIPE_MTE2>(kVectorReady);
    }
    PublishRows(id);
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
    if(qbegin>=0 && qbegin<g.tokens && kvhead>=0 && kvhead<g.hk &&
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
  GlobalTensor<float> qr,rv,out,outLse,work,sharedK,sharedV,saved;
  GlobalTensor<int32_t> rowErrors;
  int64_t blockStart,blockEnd,tileBegin,tileEnd;bool finish=false;
  GlobalTensor<uint8_t> bytes;GlobalTensor<int32_t> bt,outStatus;
  GlobalTensor<int64_t> wt,taskGm;
  TBuf<TPosition::VECCALC> taskBuf,naturalBuf,dequantBuf,
      scoreBuf,accBuf,qFloatBuf,qBf16Buf,statsBuf,scratchBuf,addressBuf;
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
    int64_t sink, int64_t recent, int64_t speculative, int64_t splits, float scale, \
    GM_ADDR sharedKey, GM_ADDR sharedValue, GM_ADDR sharedErrors, GM_ADDR state, \
    int64_t tileStart, int64_t tileEnd
#define OSCAR_CV_INIT op.Init(query,queryRot,key,value,rotation,raw,table,wk,wv,tags, \
    tasks,output,lse,status,workspace,sharedKey,sharedValue,sharedErrors,state,tileStart,tileEnd,g);op.Process()
extern "C" __global__ __aicore__ void oscar_shared_fp32_history_consumer_kernel(OSCAR_CV_ARGUMENTS) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  const Geometry g{tokens,hq,hk,requests,columns,taskCount,blockTokens,blocks,ssmOffset,
      pageStride,windowStride,tagStride,sink,recent,speculative,splits,scale};
  if(dim!=256 || hq!=6 || hk!=1 || requests!=1 || splits!=1 ||
      tileStart<sink || tileEnd<tileStart || tileEnd-tileStart>8192 ||
      (tileStart-sink)%8192) {trap();return;}
  SharedHistoryCv<256> op;OSCAR_CV_INIT;
}
