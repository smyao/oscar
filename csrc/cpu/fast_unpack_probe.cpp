// SPDX-License-Identifier: Apache-2.0
// Archive G26-G34/#13-20/#126/#140-151 and startup D.4: official CPU-debug
// executes the actual fast AscendC helper over every uint16 code word.
// D.4 four questions: source0 bounded INT2 dequant+FIA; old whole-history
// restore took 6499.8-6655.1ms; this writes only a fixed test output tile,
// never production history; count-based precision proof is not NPU speed.
#include "../kernels/attention_fast_unpack.h"
using namespace AscendC;

namespace {
constexpr int32_t kDim=256;
using G=oscar_ascend_fast::Geometry<kDim>;
class FastWords {
 public:
  __aicore__ void Run(GM_ADDR output) {
    out.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
    pipe.InitBuffer(packedBuf,G::halfKv*G::packedStride);
    pipe.InitBuffer(wordBuf,G::words*2);
    pipe.InitBuffer(planeBuf,G::planeBytes);
    pipe.InitBuffer(naturalBuf,G::elements*2);
    pipe.InitBuffer(maskBuf,G::words*2);
    pipe.InitBuffer(wordIndexBuf,G::words*4);
    pipe.InitBuffer(laneIndexBuf,G::elements*4);
    pipe.InitBuffer(metadataIndexBuf,128);
    pipe.InitBuffer(metadataHalfBuf,64);
    pipe.InitBuffer(dequantBuf,G::elements*4);
    pipe.InitBuffer(scratchBuf,512);
    oscar_ascend_fast::InitIndices<kDim>(wordIndexBuf,laneIndexBuf,
        metadataIndexBuf,maskBuf);
    auto packed=packedBuf.Get<uint16_t>();
    for(int64_t batch=GetBlockIdx();batch<128;batch+=GetBlockNum()) {
      Duplicate(packed,static_cast<uint16_t>(0),G::halfKv*G::packedStride/2);
      oscar_ascend_device::Fence<HardEvent::V_S>();
      for(int32_t i=0;i<G::words;++i) {
        const int32_t row=i/(kDim/8),word=i%(kDim/8);
        packed.SetValue(row*(G::packedStride/2)+word,
            static_cast<uint16_t>(batch*G::words+i));
      }
      for(int32_t row=0;row<G::halfKv;++row) {
        const int32_t base=row*(G::packedStride/2)+kDim/8;
        packed.SetValue(base,static_cast<uint16_t>(0x3c00)); // FP16 scale=1.
        packed.SetValue(base+1,static_cast<uint16_t>(0)); // zero=+0.
      }
      oscar_ascend_device::Fence<HardEvent::S_V>();
      int32_t error=0;
      oscar_ascend_fast::Unpack<kDim>(false,G::halfKv,error,packedBuf,
          wordBuf,planeBuf,naturalBuf,maskBuf,wordIndexBuf,laneIndexBuf,
          metadataIndexBuf,metadataHalfBuf,dequantBuf,scratchBuf);
      auto values=dequantBuf.Get<float>();
      if(error)Duplicate(values,__builtin_nanf(""),G::elements);
      oscar_ascend_device::Fence<HardEvent::V_MTE3>();
      DataCopy(out[batch*G::elements],values,G::elements);
      oscar_ascend_device::Fence<HardEvent::MTE3_S>();
      oscar_ascend_device::Fence<HardEvent::V_S>();
    }
  }
 private:
  TPipe pipe;
  GlobalTensor<float> out;
  TBuf<TPosition::VECCALC> packedBuf,wordBuf,planeBuf,naturalBuf,maskBuf,
      wordIndexBuf,laneIndexBuf,metadataIndexBuf,metadataHalfBuf,dequantBuf,
      scratchBuf;
};
}
extern "C" __global__ __aicore__ void oscar_fast_unpack_words_kernel(GM_ADDR output) {
  FastWords op;op.Run(output);
}
