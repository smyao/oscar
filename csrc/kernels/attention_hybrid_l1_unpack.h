// Qualified hybrid L1 history helper. Archive #13-22/#126/#145/#148/#154-156.
// D.4 four questions: (1) bounded history INT2 dequant inside FIA;
// (2) full-history6.5s recovery forbidden; (3) one compressed-word uint16
// Gather precedes Shift/And and FP32 Cast/Mul/Add, producing[KV/8,D,8]
// directly, without64KiB post-FP32 Gather or cross-lane exchange;
// (4) added4.25KiB Gather, index Init, extra Cast/Mul/Add calls and exact-
// window conversion MUST count in the eventual complete MDL operator.
// H64 only: wordIndexBuf8736B (first32B original guard), planeBuf4352B,
// naturalBuf34816B unchanged; whole decode explicit UB158656->171680B.
// CPU/CANN/NPU/graph/performance remain independent and initially unrun.
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

namespace oscar_direct_nz {
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
    TBuf<TPosition::VECCALC>& maskBuf,
    TBuf<TPosition::VECCALC>& naturalBuf) {
  constexpr int32_t rowStride=Geometry<D,HalfKv>::packedStride;
  auto index=metadataIndexBuf.Get<uint32_t>();
  for(int32_t row=0;row<HalfKv;++row) {
    index.SetValue(row,static_cast<uint32_t>(row*rowStride+D/2));
    index.SetValue(HalfKv+row,static_cast<uint32_t>(row*rowStride+D/2+2));
  }
  Fence<HardEvent::S_V>();
  Duplicate(maskBuf.Get<int16_t>(),static_cast<int16_t>(3),D/8);
  PipeBarrier<PIPE_V>();

  // Init-only vector scratch; no query/KV data is live yet. Keep the first
  // 32B of wordIndexBuf for the original metadata guard; indices start at32.
  auto base=naturalBuf.Get<int32_t>();
  auto low=base[256];auto high=base[512];
  auto temporary=base[768].ReinterpretCast<float>();
  ArithProgression<int32_t>(base,0,1,256);
  Duplicate(low,static_cast<int32_t>(7),256);PipeBarrier<PIPE_V>();
  And(low.ReinterpretCast<int16_t>(),base.ReinterpretCast<int16_t>(),
      low.ReinterpretCast<int16_t>(),512);
  ShiftRight(high.ReinterpretCast<int16_t>(),base.ReinterpretCast<int16_t>(),
      static_cast<int16_t>(3),512);
  PipeBarrier<PIPE_V>();
  auto lowFloat=low.ReinterpretCast<float>();auto highFloat=high.ReinterpretCast<float>();
  Cast(lowFloat,low,RoundMode::CAST_NONE,256);
  Cast(highFloat,high,RoundMode::CAST_NONE,256);PipeBarrier<PIPE_V>();
  Muls(lowFloat,lowFloat,160.0F,256);Muls(highFloat,highFloat,2.0F,256);
  PipeBarrier<PIPE_V>();Add(lowFloat,lowFloat,highFloat,256);PipeBarrier<PIPE_V>();
  auto output=wordIndexBuf.Get<int32_t>()[8];
  for(int32_t group=0;group<8;++group) {
    Adds(temporary,lowFloat,static_cast<float>(group*1280),256);PipeBarrier<PIPE_V>();
    Cast(output[group*272],temporary,RoundMode::CAST_TRUNC,256);PipeBarrier<PIPE_V>();
    Duplicate(output[group*272+256],static_cast<int32_t>(0),16);PipeBarrier<PIPE_V>();
  }
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

  static_assert(D==256 && HalfKv==64,"research direct-NZ helper is D256/H64 only");
  constexpr int32_t rowStride=Geometry<D,HalfKv>::packedStride;
  constexpr int32_t elements=Geometry<D,HalfKv>::elements;
  static_assert(rowStride==160 && elements==16384);
  auto gathered=planeBuf.Get<uint16_t>();
  auto natural=naturalBuf.Get<int16_t>();
  auto mask=maskBuf.Get<int16_t>();
  auto values=dequantBuf.Get<float>();
  // Gather raw code WORDS as uint16, never as floating half. Every one of
  // 65536 bit patterns (including half NaN encodings) is a payload word.
  Gather(gathered,packedBuf.Get<uint16_t>(),wordIndexBuf.Get<uint32_t>()[8],
      static_cast<uint32_t>(value?64:0),2176);
  PipeBarrier<PIPE_V>();
  // Each padded group contains8 token rows x32 encoded words. Shift planes
  // land directly in[KVgroup8,dimension,tokenWithin8], padded16words/group.
  const UnaryRepeatParams shiftStride{1,1,129,17};
  for(int32_t bit=0;bit<8;++bit)
    for(int32_t chunk=0;chunk<2;++chunk)
      ShiftRight(natural[bit*256+chunk*128],planeBuf.Get<int16_t>()[chunk*128],
          static_cast<int16_t>(bit*2),static_cast<uint64_t>(128),
          static_cast<uint8_t>(8),shiftStride);
  PipeBarrier<PIPE_V>();
  const BinaryRepeatParams andStride{1,1,0,129,129,0};
  for(int32_t bit=0;bit<8;++bit)
    for(int32_t chunk=0;chunk<2;++chunk)
      And(natural[bit*256+chunk*128],natural[bit*256+chunk*128],mask,
          static_cast<uint64_t>(128),static_cast<uint8_t>(8),andStride);
  PipeBarrier<PIPE_V>();
  // Eight fixed Casts strip only the unused16-word gaps. Every Cast is
  // still exact int16->FP32 of0..3; no FP32 layout-repair Gather follows.
  for(int32_t group=0;group<8;++group)
    Cast(values[group*2048],natural[group*2064],RoundMode::CAST_NONE,2048);
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
  // Code Gather has completed; planeBuf may reuse its first32B. The
  // first32B of wordIndexBuf is guard scratch; its permanent table starts32B
  // later. All four guard masks retain their exact old bytes and operations.
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
  // Same per-element FP32 q*scale then+zero, now token8 is the innermost
  // dimension. A32B block contains the8 original tokens' metadata. Repeat
  // that block acrossD without Brcb or changing metadata/error row order.
  const BinaryRepeatParams tokenEightParams{1,1,0,8,8,0};
  Fence<HardEvent::S_V>();
  for(int32_t group=0;group<8;++group)
    Mul(values[group*2048],values[group*2048],metadata[group*8],
        static_cast<uint64_t>(64),static_cast<uint8_t>(32),tokenEightParams);
  PipeBarrier<PIPE_V>();
  for(int32_t group=0;group<8;++group)
    Add(values[group*2048],values[group*2048],metadata[HalfKv+group*8],
        static_cast<uint64_t>(64),static_cast<uint8_t>(32),tokenEightParams);
  PipeBarrier<PIPE_V>();
}
} // namespace oscar_direct_nz
