// Research only. Archive #13-22/#126/#144-145/#148/#151/#154-156; startup D.4.
// 1. Bounded source0 INT2 reconstruction. 2. No old 6.5s full-history restore.
// 3. Same 32 logical rows, packed bits, FP16 metadata and FP32 Mul then Add;
//    old runs two HalfKv16 blocks; candidate one padded HalfKv32 block.
// 4. Numeric/status/guard validation precedes cycles. Not a full-C4 benchmark.
#include "../../kernels/attention_striped_unpack.h"
#include "unpack32.h"
using namespace AscendC;
using oscar_ascend_device::Fence;
template<bool Candidate> class Helper {
  static constexpr int32_t H=Candidate?32:16,D=256;
 public:
  __aicore__ void Run(GM_ADDR source,GM_ADDR liveInput,GM_ADDR output,GM_ADDR status,int32_t cases) {
    pipe.InitBuffer(packed,H*160);pipe.InitBuffer(plane,Candidate?32:8448);
    pipe.InitBuffer(natural,Candidate?H*272*2:H*D*2);
    pipe.InitBuffer(word,Candidate?32:1024);pipe.InitBuffer(mask,Candidate?64:1024);
    pipe.InitBuffer(wordIndex,Candidate?32:2048);pipe.InitBuffer(laneIndex,Candidate?32:16384);
    pipe.InitBuffer(metadataIndex,2*H*4);pipe.InitBuffer(metadataHalf,2*H*2);
    pipe.InitBuffer(dequant,H*D*4);pipe.InitBuffer(scratch,H*8*4);pipe.InitBuffer(control,32);
    if constexpr(Candidate)oscar_ascend_unpack32::InitIndices<D,H>(wordIndex,laneIndex,metadataIndex,mask);
    else oscar_ascend_striped::InitIndices<D,H>(wordIndex,laneIndex,metadataIndex,mask);
    GlobalTensor<uint16_t> input;GlobalTensor<float> out;GlobalTensor<int32_t> live,stat;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(source));out.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
    live.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(liveInput));stat.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(status));
    auto ctrl=control.Get<int32_t>();
    for(int32_t c=GetBlockIdx();c<cases;c+=GetBlockNum()) {
      DataCopyExtParams one{1,4,0,0,0};DataCopyPadExtParams<int32_t> pad{false,0,0,0};
      DataCopyPad(ctrl,live[c],one,pad);Fence<HardEvent::MTE2_S>();
      const int32_t liveRows=ctrl.GetValue(0);if(liveRows<0||liveRows>32){trap();return;}
      int32_t error=0;
      for(int32_t first=0;first<32;first+=H) {
        const int32_t active=liveRows<=first?0:(liveRows-first<H?liveRows-first:H);
        DataCopy(packed.Get<uint16_t>(),input[c*32*80+first*80],H*80);Fence<HardEvent::MTE2_V>();
        for(int32_t side=0;side<2;++side) {
          if constexpr(Candidate)oscar_ascend_unpack32::Unpack<D,H>(side!=0,active,error,packed,word,plane,natural,mask,
              wordIndex,laneIndex,metadataIndex,metadataHalf,dequant,scratch);
          else oscar_ascend_striped::Unpack<D,H>(side!=0,active,error,packed,word,plane,natural,mask,
              wordIndex,laneIndex,metadataIndex,metadataHalf,dequant,scratch);
          Fence<HardEvent::V_MTE3>();DataCopy(out[c*2*32*D+side*32*D+first*D],dequant.Get<float>(),H*D);
          Fence<HardEvent::MTE3_V>();
        }
        Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();
      }
      ctrl.SetValue(0,error);Fence<HardEvent::S_MTE3>();DataCopyPad(stat[c],ctrl,one);Fence<HardEvent::MTE3_S>();
    }
  }
 private:
  TPipe pipe;TBuf<TPosition::VECCALC> packed,plane,natural,word,mask,wordIndex,laneIndex,
      metadataIndex,metadataHalf,dequant,scratch,control;
};
extern "C" __global__ __aicore__ void oscar_unpack16_helper(GM_ADDR i,GM_ADDR l,GM_ADDR o,GM_ADDR s,int32_t n) {Helper<false> h;h.Run(i,l,o,s,n);}
extern "C" __global__ __aicore__ void oscar_unpack32_helper(GM_ADDR i,GM_ADDR l,GM_ADDR o,GM_ADDR s,int32_t n) {Helper<true> h;h.Run(i,l,o,s,n);}
