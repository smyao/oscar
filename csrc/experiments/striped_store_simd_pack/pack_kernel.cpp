// Archive #13-22/#126/#145/#154/#155; D.4: isolate exact integer packing.
// No quantization/rotation math changed; no full-history buffer. Precision
// only; 65536 patterns at every word position must match independent bytes.
#include "pack.h"
using namespace AscendC;
using oscar_ascend_device::Fence;
extern "C" __global__ __aicore__ void oscar_simd_pack_words_kernel(
    GM_ADDR codes,GM_ADDR output,int32_t dim,int64_t rows) {
  if(dim!=64&&dim!=128&&dim!=256)return;
  TPipe pipe;TBuf<TPosition::VECCALC> inputBuf,workBuf,packBuf;
  pipe.InitBuffer(inputBuf,1024);pipe.InitBuffer(workBuf,1024);pipe.InitBuffer(packBuf,160);
  GlobalTensor<int32_t> source;source.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(codes));
  GlobalTensor<uint8_t> dst;dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(output));
  auto input=inputBuf.Get<int32_t>();auto bytes=packBuf.Get<uint8_t>();
  for(int64_t row=GetBlockIdx();row<rows;row+=GetBlockNum()) {
    Duplicate(bytes.ReinterpretCast<uint16_t>(),static_cast<uint16_t>(0xadad),80);PipeBarrier<PIPE_V>();
    for(int32_t side=0;side<2;++side) {
      DataCopy(input,source[(row*2+side)*dim],dim);Fence<HardEvent::MTE2_V>();
      oscar_striped_store_simd_pack::PackCodes(input,workBuf.Get<float>(),bytes,dim,side*(dim/4));
      Fence<HardEvent::V_MTE2>();
    }
    Fence<HardEvent::V_MTE3>();Fence<HardEvent::S_MTE3>();
    DataCopy(dst[row*160],bytes,160);
    Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_S>();
  }
}
