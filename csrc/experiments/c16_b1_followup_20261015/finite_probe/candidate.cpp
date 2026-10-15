// GENERATED actual Softmax CPU isolation. Archive #134/#140/#145/#148/#154.
// D.4: bounded FIA score validation, no history restoration. Original per-row
// ReduceSum and full V2/mask/alpha body are tested; helper is not full CV,
// hardware/graph or speed evidence. Zero production resource change.
#include "../../../kernels/oscar_common.h"
#include "../../../kernels/attention_range_mask.h"
#include "adv_api/activation/softmaxflashv2.h"
using namespace oscar_ascend_device;
namespace {
constexpr int32_t D=256,kKvRows=256,kHalfRows=32,kLaneRows=64,kElements=4096;
constexpr float kNegativeInfinity=-__builtin_inff(),kEmptyMax=-__FLT_MAX__;
constexpr SoftmaxConfig kCvSoftmaxConfig={false,0,0,SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};
__aicore__ inline int64_t Min64(int64_t a,int64_t b){return a<b?a:b;}
__aicore__ inline int64_t Max64(int64_t a,int64_t b){return a>b?a:b;}
class Probe {
 public:
  __aicore__ void Run(GM_ADDR w,GM_ADDR ia,GM_ADDR is,GM_ADDR oa,GM_ADDR os,GM_ADDR oe,
      int64_t source,int64_t ctx,int64_t queries,int64_t group,int64_t start,int64_t end,
      int64_t sink,int64_t recent,int64_t rowbase,int64_t initialError) {
    g={group,1,sink,recent,1.0F};kind=source;context=ctx;qcount=queries;
    qbegin=requestBegin=0;kvend=end;lane=static_cast<int32_t>(rowbase/64);error=initialError;
    scoreOffset=pOffset=0;
    work.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(w));
    GlobalTensor<float> inAcc,inStats,outAcc,outStats;GlobalTensor<int32_t> outError;
    inAcc.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ia));inStats.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(is));
    outAcc.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(oa));outStats.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(os));
    outError.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(oe));
    pipe.InitBuffer(naturalBuf,17408);pipe.InitBuffer(dequantBuf,32768);
    pipe.InitBuffer(scoreBuf,32768);pipe.InitBuffer(accBuf,65536);
    pipe.InitBuffer(statsBuf,768);pipe.InitBuffer(scratchBuf,1024);pipe.InitBuffer(addressBuf,128);
    DataCopy(accBuf.Get<float>(),inAcc,64*D);DataCopy(statsBuf.Get<float>(),inStats,192);
    Fence<HardEvent::MTE2_V>();
    Softmax(start,static_cast<int32_t>(rowbase));
    Fence<HardEvent::V_MTE3>();DataCopy(outAcc,accBuf.Get<float>(),64*D);
    DataCopy(outStats,statsBuf.Get<float>(),192);Fence<HardEvent::MTE3_V>();
    auto result=addressBuf.Get<int32_t>();result.SetValue(0,error);
    Fence<HardEvent::S_MTE3>();DataCopyExtParams copy{1,4,0,0,0};DataCopyPad(outError,result,copy);
    Fence<HardEvent::MTE3_S>();
  }
 private:
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
    auto stats=statsBuf.Get<float>();auto tmp=scratchBuf.Get<float>(128);
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
    // Complete history tile: the earliest live query must see every
    // column. Later live rows have monotone query position/cut, so their
    // original any=true and mask-complement branch is empty as well.
    // jEnd==256 proves tileEnd is bounded by this task's kvend.
    const int64_t earliestPosition=context+qbegin+rowBase/group-requestBegin;
    const int64_t earliestCut=Max64(g.sink,earliestPosition+1-g.recent);
    const bool fullyVisibleHistory=kind==0 && jEnd==kKvRows && start>=g.sink &&
        tileEnd<=Min64(context,Min64(earliestCut,earliestPosition+1));
    if(fullyVisibleHistory) {
      // Natural unpack is dead until later alpha Brcb. Each scalar result
      // has its own32B-aligned slot; the original tmp[16] reduction work
      // area stays separate. PIPE_V keeps its reuse in original row order.
      auto finiteSums=naturalBuf.Get<float>(kHalfRows*8);
      for(int32_t r=0;r<activeRows;++r) {
        PipeBarrier<PIPE_V>();
        ReduceSum(finiteSums[r*8],scores[r*kKvRows],tmp[16],kKvRows);
      }
      Fence<HardEvent::V_S>();
      for(int32_t r=0;r<activeRows;++r)
        if(!Finite(finiteSums.GetValue(r*8)))error=2;
      // Protect scalar reads before any later Vector reuse of naturalBuf.
      Fence<HardEvent::S_V>();
    } else {
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
    auto softWork=dequantBuf.Get<float>(kElements);
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
  TPipe pipe;
  struct Geometry {int64_t hq,hk,sink,recent;float scale;} g;
  TBuf<TPosition::VECCALC> naturalBuf,dequantBuf,scoreBuf,accBuf,statsBuf,scratchBuf,addressBuf;
  GlobalTensor<float> work;
  int64_t qcount,qbegin,requestBegin,context,kvend,kind,scoreOffset,pOffset;
  int32_t lane,error;
};
}
extern "C" __global__ __aicore__ void finite_probe_candidate(
    GM_ADDR w,GM_ADDR ia,GM_ADDR is,GM_ADDR oa,GM_ADDR os,GM_ADDR oe,
    int64_t source,int64_t ctx,int64_t queries,int64_t group,int64_t start,int64_t end,
    int64_t sink,int64_t recent,int64_t rowbase,int64_t initialError) {
  Probe op;op.Run(w,ia,is,oa,os,oe,source,ctx,queries,group,start,end,sink,recent,rowbase,initialError);
}
