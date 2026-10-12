// SPDX-License-Identifier: Apache-2.0
// ISOLATED EXPERIMENT: bounded shared rotated-FP32 history producer.
// Archive #126/#129/#145/#148-155 and startup D.4 four questions:
// (1) source0 historical INT2 expansion for fused FIA; (2) old whole-history
// restoration cost 6.5s/card; (3) only one ≤8192-row block is expanded at a
// time, retaining original FP16 metadata, FP32 Mul then Add and virtual128
// page boundaries; (4) repeated query-group unpacks may fall manyfold, but
// this producer alone proves neither consumer math nor target performance.
// The production cache/reader/route remain unchanged.
#include "../../kernels/attention_striped_unpack.h"
using namespace AscendC;
using oscar_ascend_device::Fence;
using oscar_ascend_device::Finite;
__aicore__ inline int64_t Min64(int64_t a,int64_t b) {return a<b?a:b;}

extern "C" __global__ __aicore__ void oscar_shared_fp32_history_producer_kernel(
    GM_ADDR raw,GM_ADDR virtualTable,GM_ADDR sharedK,GM_ADDR sharedV,
    GM_ADDR rowErrors,int64_t context,int64_t blockStart,int64_t blockEnd,
    int64_t tableColumns,int64_t blockTokens,int64_t physicalBlocks,
    int64_t ssmOffset,int64_t pageStride) {
  constexpr int32_t D=256,Half=64,packedStride=160,slotBytes=136;
  constexpr int32_t elements=Half*D,historyTile=8192;
  if(blockStart<0 || blockEnd<blockStart || blockEnd>context ||
      blockEnd-blockStart>historyTile || blockTokens<128 || blockTokens%128 ||
      physicalBlocks<=0 || tableColumns<=0) {trap();return;}
  TPipe pipe;
  TBuf<TPosition::VECCALC> packedBuf,naturalBuf,maskBuf,metadataIndexBuf,
      metadataHalfBuf,dequantBuf,scratchBuf,dummyBuf,statusBuf,tableBuf;
  pipe.InitBuffer(packedBuf,Half*packedStride);
  pipe.InitBuffer(naturalBuf,elements*2);
  pipe.InitBuffer(maskBuf,D/8*2);
  pipe.InitBuffer(metadataIndexBuf,2*Half*4);
  pipe.InitBuffer(metadataHalfBuf,2*Half*2);
  pipe.InitBuffer(dequantBuf,elements*4);
  pipe.InitBuffer(scratchBuf,Half*8*4);
  pipe.InitBuffer(dummyBuf,32);
  pipe.InitBuffer(statusBuf,Half*4);
  pipe.InitBuffer(tableBuf,32);
  oscar_ascend_striped::InitIndices<D,Half>(
      dummyBuf,dummyBuf,metadataIndexBuf,maskBuf);
  GlobalTensor<uint8_t> bytes;
  GlobalTensor<int32_t> table,errors;
  GlobalTensor<float> kOut,vOut;
  bytes.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(raw));
  table.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(virtualTable));
  errors.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rowErrors));
  kOut.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sharedK));
  vOut.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sharedV));
  const int64_t batches=(blockEnd-blockStart+Half-1)/Half;
  for(int64_t batch=GetBlockIdx();batch<batches;batch+=GetBlockNum()) {
    const int64_t start=blockStart+batch*Half;
    const int32_t live=static_cast<int32_t>(Min64(Half,blockEnd-start));
    auto packed=packedBuf.Get<uint8_t>();
    auto perRow=statusBuf.Get<int32_t>();
    Duplicate(packed.ReinterpretCast<uint16_t>(),static_cast<uint16_t>(0),Half*packedStride/2);
    Duplicate(perRow,0,Half);
    Fence<HardEvent::V_MTE2>();
    for(int32_t j=0;j<live;) {
      const int64_t position=start+j;
      const int32_t rows=static_cast<int32_t>(Min64(live-j,128-position%128));
      if(position/128>=tableColumns) {j+=rows;continue;}
      auto blockBuf=tableBuf.Get<int32_t>();
      DataCopyExtParams one{1,4,0,0,0};
      DataCopyPadExtParams<int32_t> onePad{false,0,0,0};
      DataCopyPad(blockBuf,table[position/128],one,onePad);
      Fence<HardEvent::MTE2_S>();
      const int64_t virtualBlock=blockBuf.GetValue(0);
      const int64_t ratio=blockTokens/128;
      if(virtualBlock<0 || virtualBlock>=physicalBlocks*ratio) {j+=rows;continue;}
      const int64_t physical=virtualBlock/ratio;
      const int64_t inpage=(virtualBlock%ratio)*128+position%128;
      const int64_t offset=ssmOffset+physical*pageStride+inpage*slotBytes;
      DataCopyExtParams copy{static_cast<uint16_t>(rows),slotBytes,0,0,0};
      DataCopyPadExtParams<uint8_t> pad{false,0,0,0};
      DataCopyPad(packed[j*packedStride],bytes[offset],copy,pad);
      j+=rows;
    }
    Fence<HardEvent::MTE2_V>();
    int32_t aggregate=0;
    oscar_ascend_striped::Unpack<D,Half>(false,live,aggregate,packedBuf,
        dummyBuf,dummyBuf,naturalBuf,maskBuf,dummyBuf,dummyBuf,
        metadataIndexBuf,metadataHalfBuf,dequantBuf,scratchBuf);
    Fence<HardEvent::V_S>();
    auto metadata=naturalBuf.Get<float>();
    for(int32_t row=0;row<live;++row) {
      const float scale=metadata.GetValue(row),zero=metadata.GetValue(Half+row);
      if(!Finite(scale) || !Finite(zero) || scale<=0.0F)perRow.SetValue(row,3);
    }
    Fence<HardEvent::V_MTE3>();
    DataCopy(kOut[batch*elements],dequantBuf.Get<float>(),elements);
    Fence<HardEvent::MTE3_V>();
    oscar_ascend_striped::Unpack<D,Half>(true,live,aggregate,packedBuf,
        dummyBuf,dummyBuf,naturalBuf,maskBuf,dummyBuf,dummyBuf,
        metadataIndexBuf,metadataHalfBuf,dequantBuf,scratchBuf);
    Fence<HardEvent::V_S>();
    metadata=naturalBuf.Get<float>();
    for(int32_t row=0;row<live;++row) {
      const float scale=metadata.GetValue(row),zero=metadata.GetValue(Half+row);
      if(!Finite(scale) || !Finite(zero) || scale<=0.0F)perRow.SetValue(row,3);
    }
    Fence<HardEvent::V_MTE3>();
    DataCopy(vOut[batch*elements],dequantBuf.Get<float>(),elements);
    Fence<HardEvent::S_MTE3>();
    DataCopy(errors[batch*Half],perRow,Half);
    Fence<HardEvent::MTE3_MTE2>();
    // #126/#145: the next batch clears packed/status on Vector; MTE2
    // ownership alone cannot keep that clear behind the preceding MTE3.
    Fence<HardEvent::MTE3_V>();
  }
}
