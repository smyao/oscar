// ISOLATED EXPERIMENT. Archive #126/#145/#148/#151/#154/#155; startup D.4.
// 1. Existing SIMD INT2 source0 dequant inside full q4/q1 fused attention.
// 2. D.4 restore cost 6499.8-6655.1ms/card; no history-sized tensor here.
// 3. Natural INT16 rows change512->544B, removing power-of-two row stride
//    from the eight-row bit operations. Four exact Cast slices restore the
//    original FP32 layout before unchanged metadata guards and Mul/Add.
//    Extra UB2048B (D256 total148416B), unchanged GM/workspace/task/flags.
// 4. Bank-conflict relief is a hypothesis, not an 8-minute or NPU claim.
// SPDX-License-Identifier: Apache-2.0
// Archive #126/#129/#145/#148-154 and startup D.4 four questions:
// (1) this replaces only the bounded source0 INT2 unpack inside fused FIA;
// (2) the failed full-history restoration took 6.5s/card and cannot recur;
// (3) a versioned, lossless code permutation lets AIV ShiftRight/And write
// natural [KV,D] INT2 levels directly. All fp16 metadata conversion, live
// finite/scale checks, FP32 Mul then Add, masks, softmax and status survive;
// (4) two large Gather, eight plane buffers and their index arrays vanish,
// and HalfKv64 may reduce KV256 subtiles from eight to two. CPU bit parity,
// target NPU numerical completion and Event time remain separate gates.
// Physical slot for D256: [Kcode64,Vcode64,Kscale2,Kzero2,Vscale2,Vzero2].
// Each code word i holds levels at dimensions i+b*(D/8), b=0..7, LSB first.
#pragma once
#include "oscar_common.h"

namespace oscar_ascend_padded_natural {
using namespace AscendC;
using oscar_ascend_device::Fence;
using oscar_ascend_device::Finite;

template<int32_t D,int32_t HalfKv=16> struct Geometry {
  static_assert(D==256,"striped helper initially supports only D256; other D must fail closed");
  static_assert(HalfKv==16 || HalfKv==64,"striped HalfKv must be 16 or 64");
  static constexpr int32_t halfKv=HalfKv;
  static constexpr int32_t elements=HalfKv*D;
  static constexpr int32_t words=elements/8;
  static constexpr int32_t packedStride=((D/2+8+31)/32)*32;
  // Old prefill clones allocate this legacy buffer, but the striped helper
  // never reads/writes it. Keep its compile-time geometry ABI unchanged.
  static constexpr int32_t planeStride=words+16;
  static constexpr int32_t planeBytes=8*planeStride*2;
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
    TBuf<TPosition::VECCALC>& scratchBuf) {
  constexpr int32_t rowStride=Geometry<D,HalfKv>::packedStride;
  constexpr int32_t wordCount=D/8;
  constexpr int32_t elements=Geometry<D,HalfKv>::elements;
  constexpr int32_t naturalStride=D+16;
  static_assert(rowStride%32==0 && D%16==0 && wordCount==32);
  // Cast<float,int16_t> is the supported A2 vector path. Arithmetic right
  // shift of signed raw words is safe here: the following &3 retains only
  // the selected two low bits, independent of sign extension.
  auto packed=packedBuf.Get<int16_t>();
  auto natural=naturalBuf.Get<int16_t>();
  auto mask=maskBuf.Get<int16_t>();
  auto values=dequantBuf.Get<float>();
  // K/V code arrays start at byte 0 and D/4 (64B), both 32B aligned.
  const int32_t wordOffset=value?wordCount:0;
  // A full 128-element vector repeat spans eight rows x sixteen words.
  // Intrarepeat block strides jump between padded packed rows and natural
  // D256 rows; interrepeat strides advance eight rows. Compared with one
  // 32-element repeat per row, this uses all 8 blocks of each A2 vector op.
  const UnaryRepeatParams shiftStride{static_cast<uint16_t>(naturalStride/16),
      static_cast<uint16_t>(rowStride/32),static_cast<uint8_t>(8*naturalStride/16),
      static_cast<uint8_t>(8*rowStride/32)};
  for(int32_t bit=0;bit<8;++bit)
    for(int32_t wordBlock=0;wordBlock<2;++wordBlock)
      ShiftRight(natural[bit*wordCount+wordBlock*16],
          packed[wordOffset+wordBlock*16],static_cast<int16_t>(bit*2),
          static_cast<uint64_t>(128),static_cast<uint8_t>(HalfKv/8),shiftStride);
  PipeBarrier<PIPE_V>();
  // One all-3 block is broadcast across eight row blocks and all repeats.
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

  // Code bytes are permuted, metadata is not quantized or rounded again.
  // Metadata follows both packed K and V arrays: K at D/2, V at D/2+4.
  auto halfMetadata=metadataHalfBuf.Get<half>();
  Gather(halfMetadata,packedBuf.Get<half>(),metadataIndexBuf.Get<uint32_t>(),
      value?4:0,2*HalfKv);
  PipeBarrier<PIPE_V>();
  // Exact raw-bit FP16 metadata guard. A2 relational packed Compares does not
  // accept uint16/int16, so signed raw int16 converts exactly to FP32 first.
  // Comparing 64 or 128 FP32 lanes satisfies its 256-byte count requirement.
  // wordBuf/planeBuf/wordIndexBuf/laneIndexBuf are unused by striped unpack;
  // their 32-byte aligned allocations hold four disjoint packed bit masks.
  auto metadata=naturalBuf.Get<float>(); // natural codes are dead after Cast.
  auto rawBits=metadata[2*HalfKv];
  constexpr int32_t compareRows=HalfKv==16?64:128;
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
  } else {
    ltZero=ltMax.ReinterpretCast<uint64_t>().GetValue(1);
    negZero=ltNegInf.ReinterpretCast<uint64_t>().GetValue(1);
    geZero=ge0.ReinterpretCast<uint64_t>().GetValue(1);
  }
  const uint64_t validScale=gtScale & ltScale;
  const uint64_t validZero=negZero | (geZero & ltZero);
  const uint64_t liveMask=liveKvRows==64?~0ULL:((1ULL<<liveKvRows)-1ULL);
  if((~(validScale&validZero)&liveMask)!=0)error=3;
  // Only the guard is vectorized. Dequant uses the original half->float Cast
  // and the same FP32 Brcb/Mul then Add order as the baseline helper.
  Cast(metadata,halfMetadata,RoundMode::CAST_NONE,2*HalfKv);
  Fence<HardEvent::V_S>();
  // Preserve old FP32 reconstruction exactly: q*scale first, then +zero.
  auto rowBlocks=scratchBuf.Get<float>();
  const BinaryRepeatParams rowParams{1,1,0,static_cast<uint8_t>(D/8),
      static_cast<uint8_t>(D/8),1};
  Fence<HardEvent::S_V>();
  Brcb(rowBlocks,metadata,static_cast<uint8_t>(HalfKv/8),{1,8});
  PipeBarrier<PIPE_V>();
  for(int32_t chunk=0;chunk<D/64;++chunk)
    Mul(values[chunk*64],values[chunk*64],rowBlocks,static_cast<uint64_t>(64),
        static_cast<uint8_t>(HalfKv),rowParams);
  PipeBarrier<PIPE_V>();
  Brcb(rowBlocks,metadata[HalfKv],static_cast<uint8_t>(HalfKv/8),{1,8});
  PipeBarrier<PIPE_V>();
  for(int32_t chunk=0;chunk<D/64;++chunk)
    Add(values[chunk*64],values[chunk*64],rowBlocks,static_cast<uint64_t>(64),
        static_cast<uint8_t>(HalfKv),rowParams);
  PipeBarrier<PIPE_V>();
}
} // namespace oscar_ascend_padded_natural
