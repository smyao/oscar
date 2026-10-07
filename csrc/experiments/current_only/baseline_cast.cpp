// SPDX-License-Identifier: Apache-2.0
// Archive G30-G34/#91/#145; startup D.4: CPU-debug baseline adapters only.
// The oracle is the real merge_lse_kernel with splits=1, bracketed by the
// same BF16<->FP32 CANN Cast operations as the existing plugin path.
#include "../../kernels/oscar_common.h"

using namespace oscar_ascend_device;

class BaselineCast {
 public:
  __aicore__ void Run(GM_ADDR source, GM_ADDR destination,
                      int64_t rows, int64_t dim, bool toFloat) {
    source_ = source;
    destination_ = destination;
    dim_ = dim;
    toFloat_ = toFloat;
    pipe_.InitBuffer(inputBuf_, kMaxDim * 4);
    pipe_.InitBuffer(outputBuf_, kMaxDim * 4);
    for (int64_t row=GetBlockIdx(); row<rows; row+=GetBlockNum()) {
      if (toFloat_) {
        GlobalTensor<bfloat16_t> src;
        GlobalTensor<float> dst;
        src.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(source_));
        dst.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(destination_));
        auto input=inputBuf_.Get<bfloat16_t>();
        auto output=outputBuf_.Get<float>();
        DataCopy(input,src[row*dim_],dim_);
        Fence<HardEvent::MTE2_V>();
        Cast(output,input,RoundMode::CAST_NONE,dim_);
        Fence<HardEvent::V_MTE3>();
        DataCopy(dst[row*dim_],output,dim_);
      } else {
        GlobalTensor<float> src;
        GlobalTensor<bfloat16_t> dst;
        src.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(source_));
        dst.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(destination_));
        auto input=inputBuf_.Get<float>();
        auto output=outputBuf_.Get<bfloat16_t>();
        DataCopy(input,src[row*dim_],dim_);
        Fence<HardEvent::MTE2_V>();
        Cast(output,input,RoundMode::CAST_RINT,dim_);
        Fence<HardEvent::V_MTE3>();
        DataCopy(dst[row*dim_],output,dim_);
      }
      Fence<HardEvent::MTE3_V>();
    }
  }
 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> inputBuf_,outputBuf_;
  GM_ADDR source_;
  GM_ADDR destination_;
  int64_t dim_;
  bool toFloat_;
};

extern "C" __global__ __aicore__ void oscar_current_baseline_cast_kernel(
    GM_ADDR source, GM_ADDR destination, int64_t rows, int64_t dim,
    bool toFloat) {
  BaselineCast op;
  op.Run(source,destination,rows,dim,toFloat);
}
