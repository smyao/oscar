// SPDX-License-Identifier: Apache-2.0
// D.4: phase1_stores; historical failure 209-216 ms/16K tokens.
// Structure: one launch, vector quantizer, UB packing, one exact-size scatter
// per (token,head); no intermediate HBM quantized tensor or host token loop.
// Complexity: O(N*H*D), fixed < 8 KiB UB/core, independent of history length.
// Target: materially below 215 ms/16K. NOT compiled/timed on CANN/NPU here.
// This is a quant/pack component, not fused rotation/percentile clipping.
// Archive: G2-G13, #3/#21, G25-G34/#4-22, #79-83/#85, #87-93/#111.
// Native API evidence (references/vllm-ascend/csrc/):
// moe/moe_gating_top_k/op_kernel/moe_gating_top_k_generalized.h:185,197 ReduceMax/Sum;
// moe/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant.h:652-661 Mins/Maxs;
// moe/moe_gating_top_k/op_kernel/moe_gating_top_k_without_group.h:248 Div;
// moe/add_rms_norm_bias/op_kernel/rms_norm_base.h:257-264 S/MTE3 events;
// kernels/bgmv_expand.cpp:173 DataCopyPad; get_masked_input_and_mask_kernel.cpp:364 launch.
// G30/#12: never use GlobalTensor SetValue for adjacent status rows: scalar
// stores write back whole 64B cache lines and can overwrite another core.
#include "oscar_common.h"
#include "../include/oscar_launch.h"

using namespace oscar_ascend_device;

template <typename Slot>
class OscarStore {
 public:
  __aicore__ inline void Init(GM_ADDR key, GM_ADDR value, GM_ADDR slots,
      GM_ADDR raw, GM_ADDR status, int64_t tokens, int64_t heads, int64_t dim,
      int64_t blockTokens, int64_t blocks, int64_t ssmOffset, int64_t pageStride) {
    k_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(key));
    v_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(value));
    slots_.SetGlobalBuffer(reinterpret_cast<__gm__ Slot*>(slots));
    raw_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(raw));
    status_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(status));
    rows_=tokens*heads; heads_=heads; dim_=dim; blockTokens_=blockTokens;
    blocks_=blocks; ssmOffset_=ssmOffset; pageStride_=pageStride;
    pipe_.InitBuffer(xBuf_, kMaxDim*4); pipe_.InitBuffer(workBuf_, kMaxDim*4);
    pipe_.InitBuffer(divisorBuf_, kMaxDim*4); pipe_.InitBuffer(qBuf_, kMaxDim*4);
    pipe_.InitBuffer(reduceBuf_, kMaxDim*4); pipe_.InitBuffer(scalarBuf_, 64);
    pipe_.InitBuffer(metaBuf_, 32); pipe_.InitBuffer(packBuf_, 160);
    pipe_.InitBuffer(slotBuf_,32); pipe_.InitBuffer(statusBuf_,32);
  }
  __aicore__ inline void Process() {
    for (int64_t row=GetBlockIdx(); row<rows_; row+=GetBlockNum()) {
      auto slotLocal=slotBuf_.Get<Slot>();
      DataCopyExtParams slotCopy{1,static_cast<uint32_t>(sizeof(Slot)),0,0,0};
      DataCopyPadExtParams<Slot> slotPad{false,0,0,0};
      DataCopyPad(slotLocal,slots_[row/heads_],slotCopy,slotPad);
      Fence<HardEvent::MTE2_S>();
      const int64_t slot=static_cast<int64_t>(slotLocal.GetValue(0));
      Fence<HardEvent::S_MTE2>();
      if (slot<0) { PublishStatus(row,0); continue; }
      if (slot>=blocks_*blockTokens_) { PublishStatus(row,1); continue; }
      auto packed=packBuf_.Get<uint8_t>();
      const int32_t keyBytes=dim_/4+4;
      const int32_t kStatus=Quantize(k_, row*dim_, 0);
      const int32_t vStatus=Quantize(v_, row*dim_, keyBytes);
      if (kStatus || vStatus) {
        PublishStatus(row,kStatus ? kStatus : vStatus);
        continue;  // Invalid quantizer never publishes a partial cache row.
      }
      const int64_t dst=ssmOffset_+(slot/blockTokens_)*pageStride_
          +(slot%blockTokens_)*heads_*(2*keyBytes)+(row%heads_)*(2*keyBytes);
      Fence<HardEvent::S_MTE3>();
      DataCopyExtParams copy{1, static_cast<uint32_t>(2*keyBytes), 0, 0, 0};
      DataCopyPad(raw_[dst], packed, copy);
      Fence<HardEvent::MTE3_S>();
      PublishStatus(row,0);
    }
  }
 private:
  __aicore__ inline void PublishStatus(int64_t row,int32_t value) {
    auto local=statusBuf_.Get<int32_t>(); local.SetValue(0,value);
    Fence<HardEvent::S_MTE3>();
    DataCopyExtParams copy{1,4,0,0,0};
    DataCopyPad(status_[row],local,copy);
    Fence<HardEvent::MTE3_S>();
  }
  __aicore__ inline int32_t Quantize(const GlobalTensor<float>& source,
      int64_t offset, int32_t region) {
    auto x=xBuf_.Get<float>(); auto work=workBuf_.Get<float>();
    auto divisor=divisorBuf_.Get<float>(); auto q=qBuf_.Get<int32_t>();
    auto red=reduceBuf_.Get<float>(); auto scalar=scalarBuf_.Get<float>();
    auto meta=metaBuf_.Get<half>(); auto packed=packBuf_.Get<uint8_t>();
    DataCopy(x, source[offset], dim_);
    Fence<HardEvent::MTE2_V>();
    // A sum propagates non-finite source values which max/min may otherwise
    // suppress. Finite representable INT2 input-domain sums cannot overflow.
    ReduceSum(scalar, x, red, dim_);
    Fence<HardEvent::V_S>();
    if (!Finite(scalar.GetValue(0))) return 2;
    ReduceMax(scalar, x, red, dim_);
    Fence<HardEvent::V_S>();
    const float xmax=scalar.GetValue(0);
    Muls(work, x, -1.0F, dim_); PipeBarrier<PIPE_V>();
    ReduceMax(scalar, work, red, dim_);
    Fence<HardEvent::V_S>();
    const float xmin=-scalar.GetValue(0);
    if (!Finite(xmin) || !Finite(xmax)) return 2;
    float scale=(xmax-xmin)/3.0F;
    if (scale<1.0e-8F) scale=1.0e-8F;
    // The source remains FP32. Only scale/zero are rounded before division.
    Duplicate(scalar, 0.0F, 16); Fence<HardEvent::V_S>();
    scalar.SetValue(0, scale); scalar.SetValue(1, xmin);
    Fence<HardEvent::S_V>();
    Cast(meta, scalar, RoundMode::CAST_RINT, 16); PipeBarrier<PIPE_V>();
    Cast(scalar, meta, RoundMode::CAST_NONE, 16);
    Fence<HardEvent::V_S>();
    scale=scalar.GetValue(0); const float zero=scalar.GetValue(1);
    // PR's 1e-8 floor underflows in FP16. Do not silently substitute a larger
    // scale or invent a device-dependent NaN/Inf-to-int conversion result.
    if (!(scale>0.0F) || !Finite(scale) || !Finite(zero)) return 3;
    Duplicate(divisor, scale, dim_);
    Adds(work, x, -zero, dim_); PipeBarrier<PIPE_V>();
    Div(work, work, divisor, dim_); PipeBarrier<PIPE_V>();
    Adds(work, work, 0.5F, dim_); PipeBarrier<PIPE_V>();
    // Clamp before Cast is equivalent for finite inputs and avoids overflow.
    Maxs(work, work, 0.0F, dim_); PipeBarrier<PIPE_V>();
    Mins(work, work, 3.0F, dim_); PipeBarrier<PIPE_V>();
    Cast(q, work, RoundMode::CAST_TRUNC, dim_);
    Fence<HardEvent::V_S>();
    // Bounded scalar UB packing (16/32/64 bytes), not scalar attention dots.
    // This component needs target profiling before its latency can be signed.
    for (int32_t b=0; b<dim_/4; ++b) {
      const int32_t bits=q.GetValue(4*b)|(q.GetValue(4*b+1)<<2)
          |(q.GetValue(4*b+2)<<4)|(q.GetValue(4*b+3)<<6);
      packed.SetValue(region+b, static_cast<uint8_t>(bits));
    }
    auto words=meta.ReinterpretCast<uint16_t>();
    const uint16_t sc=words.GetValue(0), zr=words.GetValue(1);
    packed.SetValue(region+dim_/4, static_cast<uint8_t>(sc&255));
    packed.SetValue(region+dim_/4+1, static_cast<uint8_t>(sc>>8));
    packed.SetValue(region+dim_/4+2, static_cast<uint8_t>(zr&255));
    packed.SetValue(region+dim_/4+3, static_cast<uint8_t>(zr>>8));
    return 0;
  }
  TPipe pipe_;
  TBuf<TPosition::VECCALC> xBuf_,workBuf_,divisorBuf_,qBuf_,reduceBuf_,scalarBuf_,metaBuf_,packBuf_,slotBuf_,statusBuf_;
  GlobalTensor<float> k_,v_; GlobalTensor<Slot> slots_;
  GlobalTensor<uint8_t> raw_; GlobalTensor<int32_t> status_;
  int64_t rows_,heads_,dim_,blockTokens_,blocks_,ssmOffset_,pageStride_;
};

extern "C" __global__ __aicore__ void oscar_store_int2_kernel(GM_ADDR key,
    GM_ADDR value, GM_ADDR slots, GM_ADDR raw, GM_ADDR status, int64_t tokens,
    int64_t heads, int64_t dim, int64_t blockTokens, int64_t blocks,
    int64_t ssmOffset, int64_t pageStride, bool slotsI64) {
  if (slotsI64) {
    OscarStore<int64_t> op; op.Init(key,value,slots,raw,status,tokens,heads,dim,
        blockTokens,blocks,ssmOffset,pageStride); op.Process();
  } else {
    OscarStore<int32_t> op; op.Init(key,value,slots,raw,status,tokens,heads,dim,
        blockTokens,blocks,ssmOffset,pageStride); op.Process();
  }
}
namespace oscar_ascend {
void store_int2_launch(void* stream, void* key, void* value, void* slots,
    void* raw, void* status, int64_t tokens, int64_t heads, int64_t dim,
    int64_t blockTokens, int64_t blocks, int64_t ssmOffset, int64_t pageStride,
    bool slotsI64, uint32_t cores) {
  oscar_store_int2_kernel<<<cores,nullptr,stream>>>(static_cast<uint8_t*>(key),
      static_cast<uint8_t*>(value),static_cast<uint8_t*>(slots),
      static_cast<uint8_t*>(raw),static_cast<uint8_t*>(status),tokens,heads,dim,
      blockTokens,blocks,ssmOffset,pageStride,slotsI64);
}
}
