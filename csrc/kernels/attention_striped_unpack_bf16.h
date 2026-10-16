// SPDX-License-Identifier: Apache-2.0
// Archive G4/#13-22/#81-83/#126/#145/#148/#151/#154/#155/#157/#164; startup section0
// and the complete D.4 original timing log were reread for this helper.
// D.4 four questions: (1) bounded INT2 reconstruction inside fused history
// attention; (2) prior whole-history restoration cost6499.8-6655.1ms/card;
// (3) only four FP32 Mul->Add values/token, one RNE to BF16, then indexed
// 16-bit copies: no FP32 K/V tile, no history-sized restored allocation;
// (4) existing H16/32/64 tile, 16KiB bounded index scratch, 32H-byte small
// scratch, and BF16 publication. These work bounds are not measured speed.
// Native API evidence (references/vllm-ascend/csrc): scatter_nd_update_v2/
// op_kernel/scatter_nd_update_linear_index.h:202,229-250 (signed integer
// Adds, float/int32 Cast); moe_gating_top_k_generalized.h:520-528 (int32
// Muls and byte-offset Gather); sparse_flash_attention/op_kernel/arch22/
// sparse_flash_attention_service_vector_mla.h:1264-1268 (BF16 CAST_RINT).
// Gather<half> is the existing attention_striped_unpack32.h metadata path;
// archive#151 records its A2 vgather evidence. BF16 bits are reinterpreted
// as half only for Gather; no half arithmetic/conversion touches them.
// No A2 int16->int32 Cast or BF16 Gather API is assumed. Float index scratch
// holds only exact integer codes0..3; it never holds reconstructed K/V.
// CANN compilation, CPU-debug, true NPU/graph and net speed remain unverified
// on this SDK-less host. Do not equate the CPU arithmetic model with them.
#pragma once
#include "oscar_common.h"

namespace oscar_ascend_striped_bf16 {
using namespace AscendC;
using oscar_ascend_device::Fence;

template<int32_t D,int32_t HalfKv=16> struct Geometry {
  static_assert(D==256,"BF16 striped decode supports only the audited D256 layout");
  static_assert(HalfKv==16 || HalfKv==32 || HalfKv==64,"BF16 striped H must be16/32/64");
  static constexpr int32_t halfKv=HalfKv;
  static constexpr int32_t elements=HalfKv*D;
  static constexpr int32_t words=elements/8;
  static constexpr int32_t packedStride=((D/2+8+31)/32)*32;
  static constexpr int32_t naturalStride=D+16;
  static constexpr int32_t naturalBytes=HalfKv*naturalStride*2;
  static constexpr int32_t outputBytes=elements*2;
  static constexpr int32_t scratchBytes=HalfKv*8*4;
  static constexpr int32_t indexRows=8;
  static constexpr int32_t indexElements=indexRows*D;
  static constexpr int32_t indexScratchBytes=2*indexElements*4;
  static constexpr int32_t planeStride=words+16;
  static constexpr int32_t planeBytes=8*planeStride*2;
  static_assert(indexScratchBytes==16384 && scratchBytes<=2048);
  static_assert(outputBytes%32==0 && naturalBytes%32==0 && scratchBytes%32==0);
  // scratch: metadata/raw-bit compare at [0,4H) floats, table[4H,8H).
  // H16 compares64 raw-bit lanes, overlapping only the not-yet-live table.
  static_assert((2*HalfKv+(HalfKv<=32?64:128))*4<=scratchBytes);
  static_assert(4*HalfKv*2<=4*HalfKv*4,"BF16 table must not overlap its FP32 source");
};

template<int32_t D,int32_t HalfKv=16>
__aicore__ inline void InitIndices(
    TBuf<TPosition::VECCALC>& wordIndexBuf,
    TBuf<TPosition::VECCALC>& laneIndexBuf,
    TBuf<TPosition::VECCALC>& metadataIndexBuf,
    TBuf<TPosition::VECCALC>& maskBuf) {
  constexpr int32_t rowStride=Geometry<D,HalfKv>::packedStride;
  auto index=metadataIndexBuf.Get<uint32_t>();
  for(int32_t row=0;row<HalfKv;++row) {
    index.SetValue(row,static_cast<uint32_t>(row*rowStride+D/2));
    index.SetValue(HalfKv+row,static_cast<uint32_t>(row*rowStride+D/2+2));
  }
  Fence<HardEvent::S_V>();
  Duplicate(maskBuf.Get<int16_t>(),static_cast<int16_t>(3),D/8);
  PipeBarrier<PIPE_V>();
}

// Buffer minima: packed160H B, four guard buffers32B each, natural544H B,
// mask64B, metadataIndex8H B, metadataHalf4H B, output512H B, scratch32H B,
// indexScratch16384B. The buffers must not alias. Their existing larger
// allocations may be retained for softmax/PV use outside this call.
// Caller owns packed MTE2->V, prior output MTE3->V, and output V->MTE3.
// Output is contiguous [HalfKv,D] BF16 in dequantBuf, including padded rows.
// LoadPacked initializes dead rows to code0/scale0/zero0; this helper must
// not sanitize invalid live metadata or change its later error precedence.
template<int32_t D,int32_t HalfKv=16>
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
    TBuf<TPosition::VECCALC>& scratchBuf,
    TBuf<TPosition::VECCALC>& indexScratchBuf) {
  using G=Geometry<D,HalfKv>;
  constexpr int32_t rowStride=G::packedStride;
  constexpr int32_t wordCount=D/8;
  constexpr int32_t naturalStride=G::naturalStride;
  auto packed=packedBuf.Get<int16_t>();
  auto natural=naturalBuf.Get<int16_t>();
  auto mask=maskBuf.Get<int16_t>();
  auto values=dequantBuf.Get<bfloat16_t>();
  const int32_t wordOffset=value?wordCount:0;
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

  // This is the current helper's exact raw-half guard, including the H32
  // mask extraction and error=3 override only for an invalid live row.
  auto halfMetadata=metadataHalfBuf.Get<half>();
  Gather(halfMetadata,packedBuf.Get<half>(),metadataIndexBuf.Get<uint32_t>(),
      value?4:0,2*HalfKv);
  PipeBarrier<PIPE_V>();
  auto metadata=scratchBuf.Get<float>();
  auto rawBits=metadata[2*HalfKv];
  constexpr int32_t compareRows=HalfKv<=32?64:128;
  if constexpr (HalfKv==16) {
    Duplicate(rawBits,0.0F,compareRows);
    PipeBarrier<PIPE_V>();
  }
  Cast(rawBits,halfMetadata.ReinterpretCast<int16_t>(),RoundMode::CAST_NONE,2*HalfKv);
  PipeBarrier<PIPE_V>();
  auto gt0=wordBuf.Get<uint8_t>();
  auto ltMax=planeBuf.Get<uint8_t>();
  auto ltNegInf=wordIndexBuf.Get<uint8_t>();
  auto ge0=laneIndexBuf.Get<uint8_t>();
  Duplicate(gt0.ReinterpretCast<uint16_t>(),static_cast<uint16_t>(0),16);
  Duplicate(ltMax.ReinterpretCast<uint16_t>(),static_cast<uint16_t>(0),16);
  Duplicate(ltNegInf.ReinterpretCast<uint16_t>(),static_cast<uint16_t>(0),16);
  Duplicate(ge0.ReinterpretCast<uint16_t>(),static_cast<uint16_t>(0),16);
  PipeBarrier<PIPE_V>();
  Compares<float,uint8_t>(gt0,rawBits,0.0F,CMPMODE::GT,compareRows);
  Compares<float,uint8_t>(ltMax,rawBits,31744.0F,CMPMODE::LT,compareRows);
  Compares<float,uint8_t>(ltNegInf,rawBits,-1024.0F,CMPMODE::LT,compareRows);
  Compares<float,uint8_t>(ge0,rawBits,0.0F,CMPMODE::GE,compareRows);
  Fence<HardEvent::V_S>();
  uint64_t gtScale=gt0.ReinterpretCast<uint64_t>().GetValue(0);
  uint64_t ltScale=ltMax.ReinterpretCast<uint64_t>().GetValue(0);
  uint64_t ltZero,negZero,geZero;
  if constexpr (HalfKv==16) {
    gtScale&=0xffffULL;ltScale&=0xffffULL;
    ltZero=(ltMax.ReinterpretCast<uint64_t>().GetValue(0)>>16)&0xffffULL;
    negZero=(ltNegInf.ReinterpretCast<uint64_t>().GetValue(0)>>16)&0xffffULL;
    geZero=(ge0.ReinterpretCast<uint64_t>().GetValue(0)>>16)&0xffffULL;
  } else if constexpr (HalfKv==32) {
    gtScale&=0xffffffffULL;ltScale&=0xffffffffULL;
    ltZero=(ltMax.ReinterpretCast<uint64_t>().GetValue(0)>>32)&0xffffffffULL;
    negZero=(ltNegInf.ReinterpretCast<uint64_t>().GetValue(0)>>32)&0xffffffffULL;
    geZero=(ge0.ReinterpretCast<uint64_t>().GetValue(0)>>32)&0xffffffffULL;
  } else {
    ltZero=ltMax.ReinterpretCast<uint64_t>().GetValue(1);
    negZero=ltNegInf.ReinterpretCast<uint64_t>().GetValue(1);
    geZero=ge0.ReinterpretCast<uint64_t>().GetValue(1);
  }
  const uint64_t validScale=gtScale & ltScale;
  const uint64_t validZero=negZero | (geZero & ltZero);
  const uint64_t liveMask=liveKvRows==64?~0ULL:((1ULL<<liveKvRows)-1ULL);
  if((~(validScale&validZero)&liveMask)!=0)error=3;
  // Retire scalar mask readers before Vector can reuse these guard buffers
  // on a later call; invalid dead rows remain excluded by the exact mask.
  Fence<HardEvent::S_V>();
  Cast(metadata,halfMetadata,RoundMode::CAST_NONE,2*HalfKv);
  PipeBarrier<PIPE_V>();

  // Four code-major planes [code,H]. Keep even code0's multiply: positive
  // scale*+0 followed by +(-0) produces+0, unlike copying zero's raw bits.
  auto table=metadata[4*HalfKv];
  for(int32_t code=0;code<4;++code) {
    Duplicate(table[code*HalfKv],static_cast<float>(code),HalfKv);
    PipeBarrier<PIPE_V>();
    Mul(table[code*HalfKv],table[code*HalfKv],metadata,HalfKv);
    PipeBarrier<PIPE_V>();
    Add(table[code*HalfKv],table[code*HalfKv],metadata[HalfKv],HalfKv);
    PipeBarrier<PIPE_V>();
  }
  auto tableBf16=scratchBuf.Get<bfloat16_t>();
  Cast(tableBf16,table,RoundMode::CAST_RINT,4*HalfKv);
  PipeBarrier<PIPE_V>();

  // Eight rows at a time: float codes0..3 -> exact int32 byte offsets.
  // Neither half of this scratch ever contains dequantized floating K/V.
  auto codeFloat=indexScratchBuf.Get<float>();
  auto offsets=indexScratchBuf.Get<int32_t>()[G::indexElements];
  const UnaryRepeatParams castStride{1,1,static_cast<uint8_t>(D/8),
      static_cast<uint8_t>(naturalStride/16)};
  for(int32_t begin=0;begin<HalfKv;begin+=G::indexRows) {
    for(int32_t chunk=0;chunk<D/64;++chunk)
      Cast(codeFloat[chunk*64],natural[begin*naturalStride+chunk*64],
          RoundMode::CAST_NONE,static_cast<uint64_t>(64),
          static_cast<uint8_t>(G::indexRows),castStride);
    PipeBarrier<PIPE_V>();
    Cast(offsets,codeFloat,RoundMode::CAST_ROUND,G::indexElements);
    PipeBarrier<PIPE_V>();
    Muls(offsets,offsets,static_cast<int32_t>(2*HalfKv),G::indexElements);
    PipeBarrier<PIPE_V>();
    for(int32_t row=0;row<G::indexRows;++row)
      Adds(offsets[row*D],offsets[row*D],static_cast<int32_t>(2*(begin+row)),D);
    PipeBarrier<PIPE_V>();
    Gather(values.ReinterpretCast<half>()[begin*D],tableBf16.ReinterpretCast<half>(),
        offsets.template ReinterpretCast<uint32_t>(),0,G::indexElements);
    // Gather must finish reading offsets/table before the next chunk, and
    // before the caller reuses indexScratch as the Q/P Cast publisher.
    PipeBarrier<PIPE_V>();
  }
}
} // namespace oscar_ascend_striped_bf16
