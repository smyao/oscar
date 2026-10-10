// SPDX-License-Identifier: Apache-2.0
// Archive #34/#36/#37-49/#67/#92/#156: fixed capture capacities, physical ownership,
// explicit causal positions, exact metadata bounds and every padding row reset.
// D.4 four questions: prepare (725ms historical host stall); the failed route
// rebuilt per-request metadata on host. This single device launch reads native
// arrays and emits all tasks, no online CPU values or Python request loops.
// Work O(N log R + N*Hkv*splits), UB <=50KiB, no history reads/restore. Budget
// should be below short FIA 0.6-1.1ms; only a device profiler can establish it.
// Archive #129/D.4: the current source stops at this query tile's causal end,
// not the whole prefill chunk. No future-only QK/PV tiles are submitted.
// Archive #144/D.4: (1) task preparation for the shared query tile; (2) a
// smaller leader group repeats long-history reads across adjacent queries;
// (3) use the single M128 geometry constant for leader spacing and bounded
// slot-group DMA, preserving holes, request boundaries and every task field;
// (4) fewer leaders are expected, while device speed remains unverified.
// #156/D.4 checked variant (fe0 stays frozen):
// (1) prepare phase, before CV/current-only and either cache writer;
// (2) replaces the per-token random table DMA with one cached entry per
//     request/virtual128 column per core; later MTP reuses its existing scan;
// (3) O(N log R + N*Hkv*splits) metadata only, no KV restore/readback,
//     same launch count and bounded UB as fe0; S->MTE2 protects every reuse;
// (4) correctness checks are required; no claim of NPU latency improvement.
// This proves exact slot/table consistency, NOT ref-count or immutability.
#include "oscar_common.h"
#include "../include/oscar_attention_launch.h"
using namespace oscar_ascend_device;
namespace {
constexpr int32_t kMetadataRequests = 4096;
template<class Slot> class CheckedAttentionTasks {
 public:
  __aicore__ void Run(GM_ADDR qstarts, GM_ADDR lengths, GM_ADDR slots,
      GM_ADDR tasks, GM_ADDR positions, int64_t requests, int64_t tokens, int64_t hq,
      int64_t hk, int64_t sink, int64_t recent, int64_t splits,
      GM_ADDR blockTable, int64_t tableColumns, bool slotContext) {
    GlobalTensor<int32_t> startsGm, lengthsGm, tableGm;
    GlobalTensor<Slot> slotsGm;
    GlobalTensor<int64_t> tasksGm, positionsGm;
    startsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(qstarts));
    if(tableColumns>0)tableGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(blockTable));
    lengthsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(lengths));
    slotsGm.SetGlobalBuffer(reinterpret_cast<__gm__ Slot*>(slots));
    tasksGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(tasks));
    positionsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(positions));
    pipe.InitBuffer(metadata, (2*kMetadataRequests+16)*4);
    pipe.InitBuffer(taskBuf, 128);
    pipe.InitBuffer(slotBuf, 32);
    pipe.InitBuffer(slotGroupBuf,
        static_cast<int32_t>(oscar_ascend::kAttentionQueryRows*sizeof(Slot)));
    pipe.InitBuffer(tableBuf,512*4);
    auto starts=metadata.Get<int32_t>();
    auto lens=starts[kMetadataRequests+8];
    DataCopyExtParams startCopy{1,static_cast<uint32_t>((requests+1)*4),0,0,0};
    DataCopyExtParams lensCopy{1,static_cast<uint32_t>(requests*4),0,0,0};
    DataCopyPadExtParams<int32_t> pad{false,0,0,0};
    DataCopyPad(starts,startsGm,startCopy,pad);
    DataCopyPad(lens,lengthsGm,lensCopy,pad);
    Fence<HardEvent::MTE2_S>();
    auto row=taskBuf.Get<int64_t>();
    auto slot=slotBuf.Get<Slot>();
    const int64_t qtile=oscar_ascend::kAttentionQueryRows/(hq/hk);
    const int64_t perToken=hk*3*splits;
    bool malformed=starts.GetValue(0)!=0;
    int64_t cachedRequest=-1,cachedColumn=-1,cachedBlock=-1;
    for(int64_t r=0;r<requests;++r) {
      const int32_t a=starts.GetValue(r), b=starts.GetValue(r+1);
      if(a<0 || b<a || b>tokens) malformed=true;
    }
    for(int64_t token=GetBlockIdx();token<tokens;token+=GetBlockNum()) {
      DataCopyExtParams slotCopy{1,sizeof(Slot),0,0,0};
      DataCopyPadExtParams<Slot> slotPad{false,0,0,0};
      Fence<HardEvent::S_MTE2>();
      DataCopyPad(slot,slotsGm[token],slotCopy,slotPad);
      Fence<HardEvent::MTE2_S>();
      int64_t r=0, stop=requests;
      while(r<stop) {const int64_t mid=(r+stop)/2;
        if(starts.GetValue(mid+1)<=token) r=mid+1; else stop=mid;}
      const int64_t physicalSlot=slot.GetValue(0);
      const bool padding=physicalSlot<0;
      bool invalid=malformed || (!padding && r>=requests);
      int64_t begin=0, context=0, count=-1;
      int64_t metadataError=invalid?1:0;
      if(!padding && !invalid) {
        begin=starts.GetValue(r);
        const int64_t qlength=starts.GetValue(r+1)-begin;
        context=lens.GetValue(r)-qlength;
        invalid=context<0;
        if(slotContext && !invalid) {
          // Native subsequent MTP draft seq_lens retain rejected tokens.
          // Resolve the actual logical position from the physical slot and
          // this request's virtual128 table; never use an mRoPE coordinate.
          const int64_t validColumns=Min(tableColumns,(static_cast<int64_t>(lens.GetValue(r))+127)/128);
          int64_t matched=-1,matches=0;
          auto tableLocal=tableBuf.Get<int32_t>();
          if(qlength!=1 || tableColumns<=0) {invalid=true;metadataError=5;}
          else {
            for(int64_t column=0;column<validColumns;column+=512) {
              const int64_t countColumns=Min(512,validColumns-column);
              DataCopyExtParams tableCopy{1,static_cast<uint32_t>(countColumns*4),0,0,0};
              Fence<HardEvent::S_MTE2>();
              DataCopyPad(tableLocal,tableGm[r*tableColumns+column],tableCopy,pad);
              Fence<HardEvent::MTE2_S>();
              for(int64_t j=0;j<countColumns;++j) {
                if(static_cast<int64_t>(tableLocal.GetValue(j))==physicalSlot/128) {
                  matched=column+j;++matches;
                }
              }
            }
            const int64_t position=matched*128+physicalSlot%128;
            if(matches!=1 || position<0 || position>=lens.GetValue(r)) {
              invalid=true;metadataError=5;
            }else context=position;
          }
        }
        if(invalid && metadataError==0)metadataError=1;
        bool leader=(token-begin)%qtile==0;
        if(!leader) {
          Fence<HardEvent::S_MTE2>();
          DataCopyPad(slot,slotsGm[token-1],slotCopy,slotPad);
          Fence<HardEvent::MTE2_S>();
          leader=slot.GetValue(0)<0;
        }
        count=0;
        if(leader && !invalid) {
          const int64_t groupEnd=Min(begin+qlength,
              begin+((token-begin)/qtile+1)*qtile);
          const int64_t capacity=groupEnd-token;
          auto groupSlots=slotGroupBuf.Get<Slot>();
          DataCopyExtParams groupCopy{1,static_cast<uint32_t>(capacity*sizeof(Slot)),0,0,0};
          Fence<HardEvent::S_MTE2>();
          DataCopyPad(groupSlots,slotsGm[token],groupCopy,slotPad);
          Fence<HardEvent::MTE2_S>();
          while(count<capacity && groupSlots.GetValue(count)>=0)++count;
        }
      }
      const int64_t logicalPosition=context+token-begin;
      if(!padding && !invalid && tableColumns>0 && !slotContext) {
        const int64_t column=logicalPosition/128;
        if(logicalPosition<0 || column>=tableColumns) {
          invalid=true;metadataError=1;
        } else {
          if(cachedRequest!=r || cachedColumn!=column) {
            auto tableLocal=tableBuf.Get<int32_t>();
            DataCopyExtParams tableCopy{1,4,0,0,0};
            Fence<HardEvent::S_MTE2>();
            DataCopyPad(tableLocal,tableGm[r*tableColumns+column],tableCopy,pad);
            Fence<HardEvent::MTE2_S>();
            cachedBlock=tableLocal.GetValue(0);
            cachedRequest=r;cachedColumn=column;
          }
          if(cachedBlock<0 || cachedBlock*128+logicalPosition%128!=physicalSlot) {
            invalid=true;metadataError=1;
          }
        }
      }
      if(invalid) count=-1;
      row.SetValue(0,padding || invalid ? -1 : logicalPosition);
      Fence<HardEvent::S_MTE3>();
      DataCopyExtParams posCopy{1,8,0,0,0};
      DataCopyPad(positionsGm[token],row,posCopy);
      Fence<HardEvent::MTE3_S>();
      for(int64_t h=0;h<hk;++h) for(int64_t kind=0;kind<3;++kind)
      for(int64_t split=0;split<splits;++split) {
        for(int32_t col=0;col<16;++col) row.SetValue(col,0);
        int64_t first=0,last=0;
        const int64_t firstPos=context+token-begin;
        if(count>0) {
          if(kind==0) {first=Min(sink,context);
            last=Min(context,Max(sink,firstPos+count-recent));}
          if(kind==1) {first=0; last=Min(sink,context)+
              context-Min(context,Max(sink,firstPos+1-recent));}
          if(kind==2) {first=context;last=firstPos+count;}
        }
        const int64_t width=(last-first+splits-1)/splits;
        const int64_t splitBegin=Min(last,first+split*width);
        const int64_t splitEnd=Min(last,splitBegin+width);
        row.SetValue(0,token); row.SetValue(1,count); row.SetValue(2,h);
        row.SetValue(3,splitBegin);row.SetValue(4,splitEnd);
        row.SetValue(5,r<requests?r:0);row.SetValue(6,kind*splits+split);
        row.SetValue(7,kind);row.SetValue(8,context);row.SetValue(9,begin);
        row.SetValue(10,metadataError);
        Fence<HardEvent::S_MTE3>();
        DataCopy(tasksGm[(token*perToken+(h*3+kind)*splits+split)*16],row,16);
        Fence<HardEvent::MTE3_S>();
      }
    }
  }
 private:
  __aicore__ int64_t Min(int64_t a,int64_t b){return a<b?a:b;}
  __aicore__ int64_t Max(int64_t a,int64_t b){return a>b?a:b;}
  TPipe pipe;
  TBuf<TPosition::VECCALC> metadata,taskBuf,slotBuf,slotGroupBuf,tableBuf;
};
}
extern "C" __global__ __aicore__ void oscar_prepare_attention_tasks_checked_kernel(
    GM_ADDR qstarts, GM_ADDR lengths, GM_ADDR slots, GM_ADDR tasks, GM_ADDR positions,
    int64_t requests,int64_t tokens,int64_t hq,int64_t hk,int64_t sink,
    int64_t recent,int64_t splits,bool slots64,GM_ADDR blockTable,
    int64_t tableColumns,bool slotContext) {
  if(slots64) {CheckedAttentionTasks<int64_t> op;op.Run(qstarts,lengths,slots,tasks,positions,
      requests,tokens,hq,hk,sink,recent,splits,blockTable,tableColumns,slotContext);}
  else {CheckedAttentionTasks<int32_t> op;op.Run(qstarts,lengths,slots,tasks,positions,
      requests,tokens,hq,hk,sink,recent,splits,blockTable,tableColumns,slotContext);}
}
#ifndef ASCENDC_CPU_DEBUG
namespace oscar_ascend {
void prepare_attention_tasks_checked_launch(void* stream,void* qstarts,void* lengths,
    void* slots,void* tasks,void* positions,int64_t requests,int64_t tokens,int64_t hq,
    int64_t hk,int64_t sink,int64_t recent,int64_t splits,bool slots64,
    void* blockTable,int64_t tableColumns,bool slotContext,uint32_t cores) {
  oscar_prepare_attention_tasks_checked_kernel<<<cores,nullptr,stream>>>(
      static_cast<uint8_t*>(qstarts),static_cast<uint8_t*>(lengths),
      static_cast<uint8_t*>(slots),static_cast<uint8_t*>(tasks),static_cast<uint8_t*>(positions),requests,tokens,
      hq,hk,sink,recent,splits,slots64,static_cast<uint8_t*>(blockTable),tableColumns,slotContext);
}
}
#endif
