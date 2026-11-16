// SPDX-License-Identifier: Apache-2.0
// Archive #126/#145/#150/#154/#155/#164/#172; full startup D.4 reviewed.
// D.4(1): bounded source0 history INT2 reconstruction inside q4/q1 FIA.
// D.4(2): old full-history dequant6499.8--6655.1ms/store209--215ms are
// rejected structures, not measurements of this helper. No history cache.
// D.4(3): retain the144B codec, FP32 metadata checks, Sub then Mul, and
// both mode1 intermediate BF16 rounds. Compute four BF16 levels per row,
// then gather their raw uint16 bits. No FMA, new finite check, or tolerance.
// D.4(4): no added UB/GM allocation; H64 scratch2048B is partitioned into
// metadata512B/FP32 levels1024B/BF16 levels512B. Index construction and
// Gather are additional work. #164 single-repeat CastRne stays unchanged;
// #172 forbids treating fewer API calls as proof of target speed. Host,
// CANN, NPU/graph, model quality and performance remain separate gates.
// Existing API evidence: aligned_unpack Cast<float,int16_t>; this project's
// hybrid_l1_unpack uint16 Gather and Cast<float,int32_t>; native
// rms_norm_dynamic_quant_helper.h in-place Cast<int32_t,float>. Gather's
// 2048-element chunks below are a bounded choice, not a claimed SDK limit.
#pragma once
#include "oscar_common.h"
#include "attention_bf16_cast.h"

namespace oscar_ascend_codebook {
using namespace AscendC;
using oscar_ascend_device::Fence;
using oscar_ascend_device::Finite;

// Identical arguments to aligned::Unpack, but the result is BF16 in
// naturalBuf. dequantBuf contains integer byte offsets on return, NOT FP32
// reconstructed KV. Only source0's explicitly paired publisher may call it.
// The caller retains its MTE2/V and V/MTE3/MTE3/V ownership fences and all
// cross-core flags; this helper neither loads nor publishes any GM data.
template<int32_t D,int32_t HalfKv=64>
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
    TBuf<TPosition::VECCALC>& scratchBuf,int32_t dequantMode) {
  static_assert(D==256 && HalfKv==64,
      "codebook reconstruction is qualified only for decode D256/H64");
  constexpr int32_t rowStride=((D/2+16+31)/32)*32;
  constexpr int32_t wordCount=D/8;
  constexpr int32_t elements=HalfKv*D;
  constexpr int32_t naturalStride=D+16;
  constexpr int32_t metadataCount=2*HalfKv;
  constexpr int32_t levelCount=4*HalfKv;
  constexpr int32_t levelFloatOffset=metadataCount;
  constexpr int32_t levelHalfOffset=(metadataCount+levelCount)*2;
  static_assert(sizeof(float)==4 && sizeof(int32_t)==4 &&
      sizeof(uint16_t)==2 && sizeof(bfloat16_t)==2);
  static_assert(rowStride==160 && wordCount==32);
  static_assert(metadataCount*4+levelCount*4+levelCount*2==HalfKv*8*4,
      "metadata and codebook must fit the existing scratch allocation");
  static_assert(HalfKv*naturalStride*2>=elements*2 &&
      HalfKv*naturalStride*2>=HalfKv*8*4,
      "natural buffer must fit each nonoverlapping lifetime separately");

  auto packed=packedBuf.Get<int16_t>();
  auto natural=naturalBuf.Get<int16_t>();
  auto mask=maskBuf.Get<int16_t>();
  auto values=dequantBuf.Get<float>();
  const int32_t wordOffset=value?wordCount:0;
  // Keep the proven striped extraction and I16->FP32 conversion exactly.
  const UnaryRepeatParams shiftStride{static_cast<uint16_t>(naturalStride/16),
      static_cast<uint16_t>(rowStride/32),static_cast<uint8_t>(8*naturalStride/16),
      static_cast<uint8_t>(8*rowStride/32)};
  for(int32_t bit=0;bit<8;++bit)
    for(int32_t wordBlock=0;wordBlock<2;++wordBlock)
      ShiftRight(natural[bit*wordCount+wordBlock*16],
          packed[wordOffset+wordBlock*16],static_cast<int16_t>(bit*2),
          static_cast<uint64_t>(128),static_cast<uint8_t>(HalfKv/8),shiftStride);
  PipeBarrier<PIPE_V>();
  const BinaryRepeatParams andStride{static_cast<uint8_t>(naturalStride/16),
      static_cast<uint8_t>(naturalStride/16),0,
      static_cast<uint8_t>(8*naturalStride/16),
      static_cast<uint8_t>(8*naturalStride/16),0};
  for(int32_t bit=0;bit<8;++bit)
    for(int32_t wordBlock=0;wordBlock<2;++wordBlock)
      And(natural[bit*wordCount+wordBlock*16],
          natural[bit*wordCount+wordBlock*16],mask,
          static_cast<uint64_t>(128),static_cast<uint8_t>(HalfKv/8),andStride);
  PipeBarrier<PIPE_V>();
  const UnaryRepeatParams castStride{1,1,static_cast<uint8_t>(D/8),
      static_cast<uint8_t>(naturalStride/16)};
  for(int32_t chunk=0;chunk<D/64;++chunk)
    Cast(values[chunk*64],natural[chunk*64],RoundMode::CAST_NONE,
        static_cast<uint64_t>(64),static_cast<uint8_t>(HalfKv),castStride);
  Fence<HardEvent::V_S>();

  auto metadata=scratchBuf.Get<float>();
  auto levels=metadata[levelFloatOffset];
  auto table=scratchBuf.Get<bfloat16_t>()[levelHalfOffset];
  Gather(metadata,packedBuf.Get<float>(),metadataIndexBuf.Get<uint32_t>(),
      value?8:0,metadataCount);
  Fence<HardEvent::V_S>();
  for(int32_t row=0;row<liveKvRows;++row) {
    const float scale=metadata.GetValue(row),zero=metadata.GetValue(HalfKv+row);
    if(!(scale>0.0F) || !Finite(scale) || !Finite(zero))error=3;
  }
  Fence<HardEvent::S_V>();
  if(dequantMode==1) {
    auto roundedMetadata=metadataHalfBuf.Get<bfloat16_t>();
    oscar_bf16::CastRne(roundedMetadata,metadata,metadataCount);
    PipeBarrier<PIPE_V>();
    Cast(metadata,roundedMetadata,RoundMode::CAST_NONE,metadataCount);
    Fence<HardEvent::V_S>();
    for(int32_t row=0;row<liveKvRows;++row) {
      const float scale=metadata.GetValue(row),zero=metadata.GetValue(HalfKv+row);
      if(!(scale>0.0F) || !Finite(scale) || !Finite(zero))error=3;
    }
    Fence<HardEvent::S_V>();
  } else if(dequantMode!=0)error=1;

  // Planar table[q][row] gives aligned contiguous metadata operands. Keep
  // each subtraction/multiplication separate, including invalid metadata.
  // Never validate unused levels: one may overflow although all actually
  // selected codes in this row reconstruct to finite values.
  for(int32_t code=0;code<4;++code)
    Duplicate(levels[code*HalfKv],static_cast<float>(code),HalfKv);
  PipeBarrier<PIPE_V>();
  for(int32_t code=0;code<4;++code)
    Sub(levels[code*HalfKv],levels[code*HalfKv],metadata[HalfKv],HalfKv);
  PipeBarrier<PIPE_V>();
  if(dequantMode==1) {
    oscar_bf16::CastRne(table,levels,levelCount);
    PipeBarrier<PIPE_V>();
    Cast(levels,table,RoundMode::CAST_NONE,levelCount);
    PipeBarrier<PIPE_V>();
  }
  for(int32_t code=0;code<4;++code)
    Mul(levels[code*HalfKv],levels[code*HalfKv],metadata,HalfKv);
  PipeBarrier<PIPE_V>();
  oscar_bf16::CastRne(table,levels,levelCount);
  PipeBarrier<PIPE_V>();

  // FP32 levels and metadata are now dead; the512B BF16 table is disjoint.
  // Natural INT16 codes are also dead after the initial Cast. Reuse its
  // first2048B for eight identical FP32 row-byte-offsets per row. This
  // eliminates a persistent/new index buffer and scalar per-row writes.
  auto rowIntegers=scratchBuf.Get<int32_t>()[levelFloatOffset];
  ArithProgression<int32_t>(rowIntegers,0,2,HalfKv);
  PipeBarrier<PIPE_V>();
  Cast(metadata,rowIntegers,RoundMode::CAST_NONE,HalfKv);
  PipeBarrier<PIPE_V>();
  auto rowBlocks=naturalBuf.Get<float>();
  Brcb(rowBlocks,metadata,static_cast<uint8_t>(HalfKv/8),{1,8});
  PipeBarrier<PIPE_V>();
  const UnaryRepeatParams indexStride{1,1,static_cast<uint8_t>(D/8),
      static_cast<uint8_t>(D/8)};
  for(int32_t chunk=0;chunk<D/64;++chunk)
    Muls(values[chunk*64],values[chunk*64],static_cast<float>(2*HalfKv),
        static_cast<uint64_t>(64),static_cast<uint8_t>(HalfKv),indexStride);
  PipeBarrier<PIPE_V>();
  const BinaryRepeatParams rowParams{1,1,0,static_cast<uint8_t>(D/8),
      static_cast<uint8_t>(D/8),1};
  for(int32_t chunk=0;chunk<D/64;++chunk)
    Add(values[chunk*64],values[chunk*64],rowBlocks,static_cast<uint64_t>(64),
        static_cast<uint8_t>(HalfKv),rowParams);
  PipeBarrier<PIPE_V>();
  auto integerIndices=dequantBuf.Get<int32_t>();
  for(int32_t chunk=0;chunk<D/64;++chunk)
    Cast(integerIndices[chunk*64],values[chunk*64],RoundMode::CAST_RINT,
        static_cast<uint64_t>(64),static_cast<uint8_t>(HalfKv),indexStride);
  PipeBarrier<PIPE_V>();

  // Every index is an exact even integer in[0,510]. The previous barrier
  // completes every read of natural's rowBlocks before Gather overwrites
  // that same storage. Raw uint16 moves preserve BF16 NaNs and signed zero.
  auto output=naturalBuf.Get<uint16_t>();
  auto rawTable=table.ReinterpretCast<uint16_t>();
  auto indices=dequantBuf.Get<uint32_t>();
  constexpr int32_t gatherChunk=2048;
  static_assert(elements%gatherChunk==0);
  for(int32_t offset=0;offset<elements;offset+=gatherChunk)
    Gather(output[offset],rawTable,indices[offset],static_cast<uint32_t>(0),gatherChunk);
  PipeBarrier<PIPE_V>();
}
} // namespace oscar_ascend_codebook
