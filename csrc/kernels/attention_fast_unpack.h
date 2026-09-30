// SPDX-License-Identifier: Apache-2.0
// Experimental exact INT2 unpack shared by fast fe0/q1/C4 clones only.
// Archive G26-G34/#13-20/#126/#129/#140-151; startup D.4 four questions:
// (1) This is source0's bounded INT2 dequant inside fused CV/FIA.
// (2) D.4's failed full-history restore took 6499.8-6655.1ms/device while
// native FIA was 18.5-18.9ms; this helper must not restore history in HBM.
// (3) The eight 2-bit planes gain 16 uint16 padding words each, and the
// Gather index changes by the same stride. FP16 scale/zero are gathered and
// cast to FP32 as a 32-element vector, then each live row retains the exact
// finite/positive check and original FP32 Mul followed by Add. The compressed
// slot, causal mask, page order, output/status and all producer flags stay.
// (4) D256 adds only 448B explicit UB, no history-sized tensor. Bank relief
// and speed are hypotheses; frozen bitwise oracle, actual NPU and graph/quality
// gates decide them. No HF32, FMA or arithmetic-order relaxation is allowed.
#pragma once
#include "oscar_common.h"

namespace oscar_ascend_fast {
using namespace AscendC;
using oscar_ascend_device::Fence;
using oscar_ascend_device::Finite;

template<int32_t D> struct Geometry {
  static constexpr int32_t halfKv=16;
  static constexpr int32_t elements=halfKv*D;
  static constexpr int32_t words=elements/8;
  static constexpr int32_t planeStride=words+16;
  static constexpr int32_t packedStride=((D/2+8+31)/32)*32;
  static constexpr int32_t planeBytes=8*planeStride*2;
};

template<int32_t D>
__aicore__ inline void InitIndices(
    TBuf<TPosition::VECCALC>& wordIndexBuf,
    TBuf<TPosition::VECCALC>& laneIndexBuf,
    TBuf<TPosition::VECCALC>& metadataIndexBuf,
    TBuf<TPosition::VECCALC>& maskBuf) {
  constexpr int32_t words=Geometry<D>::words;
  constexpr int32_t elements=Geometry<D>::elements;
  constexpr int32_t planeStride=Geometry<D>::planeStride;
  constexpr int32_t packedStride=Geometry<D>::packedStride;
  auto wi=wordIndexBuf.Get<uint32_t>();
  auto li=laneIndexBuf.Get<uint32_t>();
  auto mi=metadataIndexBuf.Get<uint32_t>();
  for(int32_t i=0;i<words;++i)
    wi.SetValue(i,static_cast<uint32_t>((i/(D/8))*packedStride+(i%(D/8))*2));
  for(int32_t i=0;i<elements;++i) {
    const int32_t row=i/D,column=i%D;
    li.SetValue(i,static_cast<uint32_t>(
        ((column%8)*planeStride+row*(D/8)+column/8)*2));
  }
  for(int32_t row=0;row<Geometry<D>::halfKv;++row) {
    const uint32_t scale=static_cast<uint32_t>(row*packedStride+D/4);
    mi.SetValue(row,scale);
    mi.SetValue(Geometry<D>::halfKv+row,scale+2);
  }
  Fence<HardEvent::S_V>();
  Duplicate(maskBuf.Get<uint16_t>(),static_cast<uint16_t>(3),words);
  PipeBarrier<PIPE_V>();
}

template<int32_t D>
__aicore__ inline void InitIndicesV2(
    TBuf<TPosition::VECCALC>& laneIndexBuf,
    TBuf<TPosition::VECCALC>& maskBuf) {
  constexpr int32_t words=Geometry<D>::words;
  constexpr int32_t elements=Geometry<D>::elements;
  constexpr int32_t planeStride=Geometry<D>::planeStride;
  auto li=laneIndexBuf.Get<uint32_t>();
  for(int32_t i=0;i<elements;++i) {
    const int32_t row=i/D,column=i%D;
    li.SetValue(i,static_cast<uint32_t>(
        ((column%8)*planeStride+(column/8)*Geometry<D>::halfKv+row)*2));
  }
  Fence<HardEvent::S_V>();
  Duplicate(maskBuf.Get<uint16_t>(),static_cast<uint16_t>(3),words);
  PipeBarrier<PIPE_V>();
}

template<int32_t D>
__aicore__ inline void UnpackV2(int32_t liveKvRows,int32_t& error,
    TBuf<TPosition::VECCALC>& wordBuf,
    TBuf<TPosition::VECCALC>& planeBuf,
    TBuf<TPosition::VECCALC>& naturalBuf,
    TBuf<TPosition::VECCALC>& maskBuf,
    TBuf<TPosition::VECCALC>& laneIndexBuf,
    TBuf<TPosition::VECCALC>& metadataHalfBuf,
    TBuf<TPosition::VECCALC>& dequantBuf,
    TBuf<TPosition::VECCALC>& scratchBuf) {
  constexpr int32_t words=Geometry<D>::words;
  constexpr int32_t elements=Geometry<D>::elements;
  constexpr int32_t planeStride=Geometry<D>::planeStride;
  constexpr int32_t halfKv=Geometry<D>::halfKv;
  auto codeWords=wordBuf.Get<uint16_t>();
  auto planes=planeBuf.Get<uint16_t>();
  auto natural=naturalBuf.Get<int16_t>();
  auto mask=maskBuf.Get<uint16_t>();
  auto values=dequantBuf.Get<float>();
  for(int32_t bit=0;bit<8;++bit)
    ShiftRight(planes[bit*planeStride],codeWords,static_cast<uint16_t>(bit*2),words);
  PipeBarrier<PIPE_V>();
  for(int32_t bit=0;bit<8;++bit)
    And(planes[bit*planeStride],planes[bit*planeStride],mask,words);
  PipeBarrier<PIPE_V>();
  Gather(natural,planes.ReinterpretCast<int16_t>(),laneIndexBuf.Get<uint32_t>(),0,elements);
  PipeBarrier<PIPE_V>();
  Cast(values,natural,RoundMode::CAST_NONE,elements);
  PipeBarrier<PIPE_V>();
  auto metadata=naturalBuf.Get<float>();
  Cast(metadata,metadataHalfBuf.Get<half>(),RoundMode::CAST_NONE,2*halfKv);
  Fence<HardEvent::V_S>();
  for(int32_t row=0;row<halfKv;++row) {
    const float scale=metadata.GetValue(row),zero=metadata.GetValue(halfKv+row);
    if(row<liveKvRows && (!Finite(scale)||!Finite(zero)||scale<=0.0F))error=3;
  }
  auto rowBlocks=scratchBuf.Get<float>();
  const BinaryRepeatParams rowParams{1,1,0,static_cast<uint8_t>(D/8),
      static_cast<uint8_t>(D/8),1};
  Fence<HardEvent::S_V>();
  Brcb(rowBlocks,metadata,2,{1,8});PipeBarrier<PIPE_V>();
  for(int32_t chunk=0;chunk<D/64;++chunk)
    Mul(values[chunk*64],values[chunk*64],rowBlocks,static_cast<uint64_t>(64),
        static_cast<uint8_t>(halfKv),rowParams);
  PipeBarrier<PIPE_V>();
  Brcb(rowBlocks,metadata[halfKv],2,{1,8});PipeBarrier<PIPE_V>();
  for(int32_t chunk=0;chunk<D/64;++chunk)
    Add(values[chunk*64],values[chunk*64],rowBlocks,static_cast<uint64_t>(64),
        static_cast<uint8_t>(halfKv),rowParams);
  PipeBarrier<PIPE_V>();
}

template<int32_t D>
__aicore__ inline void Unpack(bool value,int32_t liveKvRows,int32_t& error,
    TBuf<TPosition::VECCALC>& packedBuf,
    TBuf<TPosition::VECCALC>& wordBuf,
    TBuf<TPosition::VECCALC>& planeBuf,
    TBuf<TPosition::VECCALC>& naturalBuf,
    TBuf<TPosition::VECCALC>& maskBuf,
    TBuf<TPosition::VECCALC>& wordIndexBuf,
    TBuf<TPosition::VECCALC>& laneIndexBuf,
    TBuf<TPosition::VECCALC>& metadataIndexBuf,
    TBuf<TPosition::VECCALC>& metadataHalfBuf,
    TBuf<TPosition::VECCALC>& dequantBuf,
    TBuf<TPosition::VECCALC>& scratchBuf) {
  constexpr int32_t words=Geometry<D>::words;
  constexpr int32_t elements=Geometry<D>::elements;
  constexpr int32_t planeStride=Geometry<D>::planeStride;
  constexpr int32_t halfKv=Geometry<D>::halfKv;
  auto packed=packedBuf.Get<uint16_t>();
  auto codeWords=wordBuf.Get<uint16_t>();
  auto planes=planeBuf.Get<uint16_t>();
  auto natural=naturalBuf.Get<int16_t>();
  auto mask=maskBuf.Get<uint16_t>();
  auto values=dequantBuf.Get<float>();
  const uint32_t byteBase=value?D/4+4:0;
  Gather(codeWords,packed,wordIndexBuf.Get<uint32_t>(),byteBase,words);
  PipeBarrier<PIPE_V>();
  for(int32_t bit=0;bit<8;++bit)
    ShiftRight(planes[bit*planeStride],codeWords,
        static_cast<uint16_t>(bit*2),words);
  PipeBarrier<PIPE_V>();
  for(int32_t bit=0;bit<8;++bit)
    And(planes[bit*planeStride],planes[bit*planeStride],mask,words);
  PipeBarrier<PIPE_V>();
  Gather(natural,planes.ReinterpretCast<int16_t>(),
      laneIndexBuf.Get<uint32_t>(),0,elements);
  PipeBarrier<PIPE_V>();
  Cast(values,natural,RoundMode::CAST_NONE,elements);
  Fence<HardEvent::V_S>();

  // Gather offsets are bytes from the first packed slot. byteBase selects
  // K or V metadata while the 32 indexes select scale[16], then zero[16].
  // naturalBuf is dead after Cast(values,natural) and can hold the 32 FP32
  // metadata values; the half scratch is disjoint from both source buffers.
  auto halfMetadata=metadataHalfBuf.Get<half>();
  Gather(halfMetadata,packedBuf.Get<half>(),
      metadataIndexBuf.Get<uint32_t>(),byteBase,2*halfKv);
  PipeBarrier<PIPE_V>();
  auto metadata=naturalBuf.Get<float>();
  Cast(metadata,halfMetadata,RoundMode::CAST_NONE,2*halfKv);
  Fence<HardEvent::V_S>();
  for(int32_t row=0;row<halfKv;++row) {
    const float scale=metadata.GetValue(row);
    const float zero=metadata.GetValue(halfKv+row);
    if(row<liveKvRows && (!Finite(scale)||!Finite(zero)||scale<=0.0F))error=3;
  }
  auto rowBlocks=scratchBuf.Get<float>();
  const BinaryRepeatParams rowParams{1,1,0,static_cast<uint8_t>(D/8),
      static_cast<uint8_t>(D/8),1};
  Fence<HardEvent::S_V>();
  Brcb(rowBlocks,metadata,2,{1,8});PipeBarrier<PIPE_V>();
  for(int32_t chunk=0;chunk<D/64;++chunk)
    Mul(values[chunk*64],values[chunk*64],rowBlocks,static_cast<uint64_t>(64),
        static_cast<uint8_t>(halfKv),rowParams);
  PipeBarrier<PIPE_V>();
  Brcb(rowBlocks,metadata[halfKv],2,{1,8});PipeBarrier<PIPE_V>();
  for(int32_t chunk=0;chunk<D/64;++chunk)
    Add(values[chunk*64],values[chunk*64],rowBlocks,static_cast<uint64_t>(64),
        static_cast<uint8_t>(halfKv),rowParams);
  PipeBarrier<PIPE_V>();
}
} // namespace oscar_ascend_fast
