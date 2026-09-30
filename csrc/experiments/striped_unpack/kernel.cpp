// Archive #126/#129/#145/#148-154 and startup D.4 four questions:
// (1) this is bounded D256 INT2 source0 dequant inside fused FIA;
// (2) no full-history FP32/BF16 restoration like D.4's 6.5s/card;
// (3) fixed 64-row packed UB uses exact striped code permutation, original
// fp16 metadata and old FP32 Mul/Add, with no changed mask or layout at output;
// (4) CPU word-exhaustive parity plus real CANN compile prove correctness of
// this helper only; NPU timing and full attention remain unverified.
#include "../../kernels/attention_striped_unpack.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void oscar_striped_unpack_kernel(
    GM_ADDR packedRows,GM_ADDR outputK,GM_ADDR outputV,GM_ADDR outputStatus,
    int32_t batches,int32_t liveRows) {
  constexpr int32_t D=256,HalfKv=64,rowStride=160,elements=HalfKv*D;
  TPipe pipe;
  TBuf<TPosition::VECCALC> packedBuf,naturalBuf,maskBuf,metadataIndexBuf,
      metadataHalfBuf,dequantBuf,scratchBuf,dummyBuf,statusBuf;
  pipe.InitBuffer(packedBuf,HalfKv*rowStride);
  pipe.InitBuffer(naturalBuf,elements*2);
  pipe.InitBuffer(maskBuf,D/8*2);
  pipe.InitBuffer(metadataIndexBuf,2*HalfKv*4);
  pipe.InitBuffer(metadataHalfBuf,2*HalfKv*2);
  pipe.InitBuffer(dequantBuf,elements*4);
  pipe.InitBuffer(scratchBuf,HalfKv*8*4);
  pipe.InitBuffer(dummyBuf,32);
  pipe.InitBuffer(statusBuf,32);
  oscar_ascend_striped::InitIndices<D,HalfKv>(
      dummyBuf,dummyBuf,metadataIndexBuf,maskBuf);
  GlobalTensor<uint16_t> source;
  GlobalTensor<float> kOut,vOut;
  GlobalTensor<int32_t> statusOut;
  source.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(packedRows));
  kOut.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(outputK));
  vOut.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(outputV));
  statusOut.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(outputStatus));
  for(int32_t batch=0;batch<batches;++batch) {
    auto packed=packedBuf.Get<uint16_t>();
    DataCopy(packed,source[batch*HalfKv*rowStride/2],HalfKv*rowStride/2);
    oscar_ascend_device::Fence<HardEvent::MTE2_V>();
    int32_t error=0;
    oscar_ascend_striped::Unpack<D,HalfKv>(false,liveRows,error,
        packedBuf,dummyBuf,dummyBuf,naturalBuf,maskBuf,dummyBuf,dummyBuf,
        metadataIndexBuf,metadataHalfBuf,dequantBuf,scratchBuf);
    oscar_ascend_device::Fence<HardEvent::V_MTE3>();
    DataCopy(kOut[batch*elements],dequantBuf.Get<float>(),elements);
    oscar_ascend_device::Fence<HardEvent::MTE3_V>();
    oscar_ascend_striped::Unpack<D,HalfKv>(true,liveRows,error,
        packedBuf,dummyBuf,dummyBuf,naturalBuf,maskBuf,dummyBuf,dummyBuf,
        metadataIndexBuf,metadataHalfBuf,dequantBuf,scratchBuf);
    oscar_ascend_device::Fence<HardEvent::V_MTE3>();
    DataCopy(vOut[batch*elements],dequantBuf.Get<float>(),elements);
    oscar_ascend_device::Fence<HardEvent::MTE3_S>();
    auto status=statusBuf.Get<int32_t>();
    status.SetValue(0,error);
    oscar_ascend_device::Fence<HardEvent::S_MTE3>();
    DataCopyExtParams one{1,4,0,0,0};
    DataCopyPad(statusOut[batch],status,one);
    oscar_ascend_device::Fence<HardEvent::MTE3_MTE2>();
  }
}
