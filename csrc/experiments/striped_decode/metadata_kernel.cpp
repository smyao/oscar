// Isolated A2 SIMD metadata predicate experiment. Archive #126/#145/#154;
// startup D.4: this checks fixed 64-row INT2 metadata only, never history-wide.
#include "../../kernels/oscar_common.h"
using namespace AscendC;
using oscar_ascend_device::Fence;

extern "C" __global__ __aicore__ void oscar_striped_metadata_guard_kernel(
    GM_ADDR input,GM_ADDR output,int32_t batches) {
  if ASCEND_IS_AIC {return;}
  TPipe pipe;
  TBuf<TPosition::VECCALC> halfBuf,floatBuf,m0Buf,m1Buf,m2Buf,m3Buf,outBuf;
  pipe.InitBuffer(halfBuf,256);pipe.InitBuffer(floatBuf,512);
  pipe.InitBuffer(m0Buf,32);pipe.InitBuffer(m1Buf,32);
  pipe.InitBuffer(m2Buf,32);pipe.InitBuffer(m3Buf,32);
  pipe.InitBuffer(outBuf,32);
  GlobalTensor<int16_t> source;
  GlobalTensor<uint64_t> destination;
  source.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t*>(input));
  destination.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(output));
  auto half=halfBuf.Get<int16_t>();auto raw=floatBuf.Get<float>();
  auto gt0=m0Buf.Get<uint8_t>();auto ltMax=m1Buf.Get<uint8_t>();
  auto ltNegInf=m2Buf.Get<uint8_t>();auto ge0=m3Buf.Get<uint8_t>();
  auto result=outBuf.Get<uint64_t>();
  for(int32_t batch=0;batch<batches;++batch) {
    DataCopy(half,source[batch*128],128);Fence<HardEvent::MTE2_V>();
    Cast(raw,half,RoundMode::CAST_NONE,128);PipeBarrier<PIPE_V>();
    Duplicate(gt0.ReinterpretCast<uint16_t>(),static_cast<uint16_t>(0),16);
    Duplicate(ltMax.ReinterpretCast<uint16_t>(),static_cast<uint16_t>(0),16);
    Duplicate(ltNegInf.ReinterpretCast<uint16_t>(),static_cast<uint16_t>(0),16);
    Duplicate(ge0.ReinterpretCast<uint16_t>(),static_cast<uint16_t>(0),16);
    PipeBarrier<PIPE_V>();
    Compares<float,uint8_t>(gt0,raw,0.0F,CMPMODE::GT,128);
    Compares<float,uint8_t>(ltMax,raw,31744.0F,CMPMODE::LT,128);
    Compares<float,uint8_t>(ltNegInf,raw,-1024.0F,CMPMODE::LT,128);
    Compares<float,uint8_t>(ge0,raw,0.0F,CMPMODE::GE,128);
    Fence<HardEvent::V_S>();
    const uint64_t scaleValid=gt0.ReinterpretCast<uint64_t>().GetValue(0) &
        ltMax.ReinterpretCast<uint64_t>().GetValue(0);
    const uint64_t zeroValid=ltNegInf.ReinterpretCast<uint64_t>().GetValue(1) |
        (ge0.ReinterpretCast<uint64_t>().GetValue(1) &
         ltMax.ReinterpretCast<uint64_t>().GetValue(1));
    uint64_t invalid=~(scaleValid&zeroValid);
    if(batch==2048)invalid&=((1ULL<<17)-1ULL); // invalid dead tail ignored.
    result.SetValue(0,invalid);
    Fence<HardEvent::S_MTE3>();DataCopy(destination[batch*4],result,4);
    Fence<HardEvent::MTE3_MTE2>();Fence<HardEvent::MTE3_S>();
  }
}
