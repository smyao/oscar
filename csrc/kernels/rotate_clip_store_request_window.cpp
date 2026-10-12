// SPDX-License-Identifier: Apache-2.0
// Archive G27/#13-22/#37-49/#69/#126-128/#145/#154/#155; D.4 four questions:
// 1. phase1_stores exact BF16 publication, after the accepted packed writer.
// 2. D.4 stores208.6-216.0ms/16K; dequant6499.8-6655.1ms and prepare725ms
//    forbid historical restore, per-token host work, and quadratic scans.
// 3. One owner per request scans new metadata once in64-row DMA tiles into
//    bounded UB last-writer indices. At most323 exact rows/request copied.
//    No history reads, no rotated raw window, no per-token later-token scan.
//    All heads must have packed status0; tags publish only after both K/V.
// 4. O(N*H + requests^2 + requests*323*H*D) work (requests<=128), bounded UB.
//    Duplicate-lease validation scans cached request metadata, never tokens.
//    Extra launch and
//    request-parallel shape require NPU timings; no CPU performance claim.
// Duplicate live lease slots explicitly fail. Epoch prevents stale reads;
// caller must keep both launches and all lifecycle updates on one stream.
#include "oscar_common.h"
using namespace oscar_ascend_device;
namespace {
constexpr int32_t kScanRows=64,kMaxHeads=16,kMaxWindow=1024,kMaxRequests=128;
template<typename T> class RequestWindowStore {
 public:
  __aicore__ inline void Run(GM_ADDR key,GM_ADDR value,GM_ADDR slots,GM_ADDR positions,
      GM_ADDR qstarts,GM_ADDR mapping,GM_ADDR packedStatus,GM_ADDR rawKey,
      GM_ADDR rawValue,GM_ADDR tags,GM_ADDR requestStatus,int64_t tokens,
      int64_t requests,int64_t capacity,int64_t heads,int32_t dim,int64_t sink,
      int64_t recent,int64_t positionBits,int64_t kStride,int64_t vStride,int64_t tagStride) {
    k_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(key));v_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(value));
    slots_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(slots));positions_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(positions));
    starts_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(qstarts));map_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(mapping));
    status_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(packedStatus));
    rawK_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(rawKey));rawV_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(rawValue));
    tags_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(tags));outStatus_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(requestStatus));
    pipe_.InitBuffer(metaBuf_,32);pipe_.InitBuffer(tagBuf_,32);pipe_.InitBuffer(outBuf_,32);
    pipe_.InitBuffer(slotBuf_,kScanRows*8);pipe_.InitBuffer(posBuf_,kScanRows*8);
    pipe_.InitBuffer(statusBuf_,kScanRows*kMaxHeads*4);pipe_.InitBuffer(chosenBuf_,kMaxWindow*4);
    pipe_.InitBuffer(mapBuf_,kMaxRequests*16);pipe_.InitBuffer(startBuf_,(kMaxRequests+8)*4);
    pipe_.InitBuffer(inputBuf_,kMaxDim*4);pipe_.InitBuffer(floatBuf_,kMaxDim*4);pipe_.InitBuffer(rawBuf_,kMaxDim*2);
    for(int64_t req=GetBlockIdx();req<capacity;req+=GetBlockNum()) {
      if(req>=requests){WriteStatus(req,0);continue;}
      int32_t error=0;
      const int64_t begin=Read(starts_,req),end=Read(starts_,req+1);
      if(requests>capacity||requests>kMaxRequests||heads<1||heads>kMaxHeads||sink<0||recent<=0||sink+recent>kMaxWindow||positionBits<1||positionBits>31) error=5;
      else if(begin<0||end<begin||end>tokens) error=1;
      if(!error&&Read(starts_,0)!=0)error=6;
      // Only req0 scans the graph-padding tail, once. Other requests can
      // publish their valid windows concurrently; status_guard must reject
      // the whole batch before any reader when this global flag is nonzero.
      if(!error&&req==0) {
        const int64_t tail=Read(starts_,requests);
        if(tail<0||tail>tokens)error=1;
        else for(int64_t base=tail;base<tokens;base+=kScanRows) {
          const int32_t count=static_cast<int32_t>(tokens-base<kScanRows?tokens-base:kScanRows);
          auto local=slotBuf_.Get<int64_t>();DataCopyExtParams copy{1,static_cast<uint32_t>(count*8),0,0,0};
          DataCopyPadExtParams<int64_t> pad{false,0,0,0};DataCopyPad(local,slots_[base],copy,pad);
          Fence<HardEvent::MTE2_S>();for(int32_t i=0;i<count;++i)if(local.GetValue(i)>=0){error=7;break;}
          Fence<HardEvent::S_MTE2>();if(error)break;
        }
      }
      if(error){WriteStatus(req,error);continue;}
      if(begin==end) {WriteStatus(req,0);continue;}
      auto chosen=chosenBuf_.Get<int32_t>();
      Fence<HardEvent::S_V>();Duplicate(chosen,int32_t(-1),kMaxWindow);Fence<HardEvent::V_S>();
      const int64_t limit=int64_t(1)<<positionBits;
      int64_t previous=-1;bool hasLive=false;
      for(int64_t base=begin;base<end;base+=kScanRows) {
        const int32_t count=static_cast<int32_t>(end-base<kScanRows?end-base:kScanRows);
        auto ls=slotBuf_.Get<int64_t>(),lp=posBuf_.Get<int64_t>();auto st=statusBuf_.Get<int32_t>();
        DataCopyExtParams copy64{1,static_cast<uint32_t>(count*8),0,0,0};
        DataCopyPadExtParams<int64_t> pad64{false,0,0,0};
        DataCopyPad(ls,slots_[base],copy64,pad64);DataCopyPad(lp,positions_[base],copy64,pad64);
        DataCopyExtParams copy32{1,static_cast<uint32_t>(count*heads*4),0,0,0};
        DataCopyPadExtParams<int32_t> pad32{false,0,0,0};DataCopyPad(st,status_[base*heads],copy32,pad32);
        Fence<HardEvent::MTE2_S>();
        for(int32_t row=0;row<count;++row) {
          if(ls.GetValue(row)<0)continue;
          bool good=true;for(int64_t h=0;h<heads;++h)good=good&&st.GetValue(row*heads+h)==0;
          if(!good)continue;
          const int64_t p=lp.GetValue(row);
          if(p<0||p>=limit||p<=previous){error=4;break;}
          previous=p;
          hasLive=true;
          const int64_t ring=p<sink?p:sink+p%recent;
          chosen.SetValue(ring,static_cast<int32_t>(base+row));
        }
        Fence<HardEvent::S_MTE2>();if(error)break;
      }
      // Invalid position/order rejects all sidecar publication for this request.
      if(error){WriteStatus(req,error);continue;}
      // Native FULL capture has nonempty qstarts but every slot is padding.
      // Such a row needs no published lease and must not inspect map(-1,0).
      if(!hasLive){WriteStatus(req,0);continue;}
      const int64_t lease=Read(map_,req*2),epoch=Read(map_,req*2+1);
      if(lease<0||lease>=capacity||epoch<=0||epoch>((int64_t(1)<<(63-positionBits))-1)) error=2;
      if(!error) {
        auto allMap=mapBuf_.Get<int64_t>();auto allStarts=startBuf_.Get<int32_t>();
        DataCopyExtParams mapCopy{1,static_cast<uint32_t>(requests*16),0,0,0};
        DataCopyExtParams startsCopy{1,static_cast<uint32_t>((requests+1)*4),0,0,0};
        DataCopyPadExtParams<int64_t> mapPad{false,0,0,0};DataCopyPadExtParams<int32_t> startsPad{false,0,0,0};
        DataCopyPad(allMap,map_,mapCopy,mapPad);DataCopyPad(allStarts,starts_,startsCopy,startsPad);
        Fence<HardEvent::MTE2_S>();
        for(int64_t other=0;other<requests;++other) {
          if(other==req)continue;
          // Provider marks unscheduled/padding map rows with slot -1.
          if(allStarts.GetValue(other)<allStarts.GetValue(other+1)&&allMap.GetValue(other*2)==lease) {error=3;break;}
        }
        Fence<HardEvent::S_MTE2>();
      }
      if(error){WriteStatus(req,error);continue;}
      for(int64_t ring=0;ring<sink+recent;++ring) {
        const int64_t token=chosen.GetValue(ring);if(token<0)continue;
        const int64_t p=Read(positions_,token);
        for(int64_t h=0;h<heads;++h) WriteRaw(token,h,lease,ring,heads,dim,kStride,vStride);
        auto tag=tagBuf_.Get<int64_t>();tag.SetValue(0,(epoch<<positionBits)|p);
        Fence<HardEvent::S_MTE3>();DataCopyExtParams copy{1,8,0,0,0};
        DataCopyPad(tags_[lease*tagStride+ring],tag,copy);Fence<HardEvent::MTE3_S>();
      }
      WriteStatus(req,0);
    }
  }
 private:
  template<typename U> __aicore__ inline int64_t Read(const GlobalTensor<U>& source,int64_t index) {
    auto local=metaBuf_.Get<U>();DataCopyExtParams copy{1,sizeof(U),0,0,0};DataCopyPadExtParams<U> pad{false,0,0,0};
    DataCopyPad(local,source[index],copy,pad);Fence<HardEvent::MTE2_S>();
    const int64_t value=local.GetValue(0);Fence<HardEvent::S_MTE2>();return value;
  }
  __aicore__ inline void WriteStatus(int64_t req,int32_t value) {
    auto local=outBuf_.Get<int32_t>();local.SetValue(0,value);Fence<HardEvent::S_MTE3>();
    DataCopyExtParams copy{1,4,0,0,0};DataCopyPad(outStatus_[req],local,copy);Fence<HardEvent::MTE3_S>();
  }
  __aicore__ inline void WriteRaw(int64_t token,int64_t head,int64_t lease,int64_t ring,
      int64_t heads,int32_t dim,int64_t ks,int64_t vs) {
    auto original=inputBuf_.Get<T>();auto converted=rawBuf_.Get<bfloat16_t>();
    for(int32_t side=0;side<2;++side) {
      DataCopy(original,(side==0?k_:v_)[(token*heads+head)*dim],dim);Fence<HardEvent::MTE2_V>();
      if constexpr(IsSameType<T,bfloat16_t>::value)DataCopy(converted,original,dim);
      else if constexpr(IsSameType<T,float>::value)Cast(converted,original,RoundMode::CAST_RINT,dim);
      else {auto tmp=floatBuf_.Get<float>();Cast(tmp,original,RoundMode::CAST_NONE,dim);PipeBarrier<PIPE_V>();Cast(converted,tmp,RoundMode::CAST_RINT,dim);}
      Fence<HardEvent::V_MTE3>();DataCopyExtParams copy{1,static_cast<uint32_t>(dim*2),0,0,0};
      DataCopyPad((side==0?rawK_:rawV_)[lease*(side==0?ks:vs)+(ring*heads+head)*dim],converted,copy);
      Fence<HardEvent::MTE3_V>();Fence<HardEvent::V_MTE2>();
    }
  }
  TPipe pipe_;
  TBuf<TPosition::VECCALC> metaBuf_,tagBuf_,outBuf_,slotBuf_,posBuf_,statusBuf_,chosenBuf_,inputBuf_,floatBuf_,rawBuf_,mapBuf_,startBuf_;
  GlobalTensor<T> k_,v_;GlobalTensor<int64_t> slots_,positions_,map_,tags_;
  GlobalTensor<int32_t> starts_,status_,outStatus_;GlobalTensor<bfloat16_t> rawK_,rawV_;
};
}
extern "C" __global__ __aicore__ void oscar_request_window_store_kernel(
    GM_ADDR key,GM_ADDR value,GM_ADDR slots,GM_ADDR positions,GM_ADDR qstarts,
    GM_ADDR mapping,GM_ADDR packedStatus,GM_ADDR rawKey,GM_ADDR rawValue,
    GM_ADDR tags,GM_ADDR requestStatus,int64_t tokens,int64_t requests,int64_t capacity,
    int64_t heads,int64_t dim,int32_t dtype,int64_t sink,int64_t recent,
    int64_t positionBits,int64_t kStride,int64_t vStride,int64_t tagStride) {
#define RUN(TYPE) RequestWindowStore<TYPE> op;op.Run(key,value,slots,positions,qstarts,mapping,packedStatus,rawKey,rawValue,tags,requestStatus,tokens,requests,capacity,heads,dim,sink,recent,positionBits,kStride,vStride,tagStride)
  if(dtype==0){RUN(float);}else if(dtype==1){RUN(half);}else{RUN(bfloat16_t);}
#undef RUN
}
#ifndef ASCENDC_CPU_DEBUG
#include "../include/oscar_request_window_store_launch.h"
namespace oscar_ascend {
void rotate_clip_store_request_window_launch(void* stream,uint32_t grid,
    void* key,void* value,void* rk,void* rv,void* slots,void* positions,
    void* qstarts,void* windowMap,void* packed,void* wk,void* wv,void* tags,
    void* packedStatus,void* windowStatus,int64_t tokens,int64_t requests,
    int64_t heads,int64_t dim,int32_t dtype,int64_t blockTokens,int64_t blocks,
    int64_t offset,int64_t pageStride,int64_t kStride,int64_t vStride,int64_t tagStride,
    int64_t sink,int64_t recentWithSpec,float kClip,float vClip,bool hadamard,
    int64_t windowCapacity,int64_t positionBits) {
  request_packed_store_launch(stream,grid,key,value,rk,rv,slots,positions,packed,
      packedStatus,tokens,heads,dim,dtype,blockTokens,blocks,offset,pageStride,kClip,vClip,hadamard);
  oscar_request_window_store_kernel<<<grid,nullptr,stream>>>(
      (GM_ADDR)key,(GM_ADDR)value,(GM_ADDR)slots,(GM_ADDR)positions,(GM_ADDR)qstarts,
      (GM_ADDR)windowMap,(GM_ADDR)packedStatus,(GM_ADDR)wk,(GM_ADDR)wv,(GM_ADDR)tags,
      (GM_ADDR)windowStatus,tokens,requests,windowCapacity,heads,dim,dtype,sink,
      recentWithSpec,positionBits,kStride,vStride,tagStride);
}
}
#endif
