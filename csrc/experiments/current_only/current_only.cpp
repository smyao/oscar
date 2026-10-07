// SPDX-License-Identifier: Apache-2.0
// Archive G30-G34/#4-16/#91/#126/#145; startup D.4 four questions:
// (1) Fresh context=0 has only native current FIA, so this isolated primitive
//     validates its BF16 output and FP32 LSE instead of merging empty history.
// (2) D.4's failed 6.5s full-history restore and 725ms host preparation do
//     not occur: at most 16 current rows are staged in local UB per core.
// (3) Mirror merge_lse(splits=1): +0 accumulator, weight-one Muls/Add,
//     same ReduceSum finite test and LSE + Log(1), then BF16 output cast.
// (4) This is correctness/compile evidence only; no NPU or model speed claim.
//     Source and output buffers have explicit MTE2/V/MTE3 reuse handoffs.
#include "../../kernels/oscar_common.h"

using namespace oscar_ascend_device;

namespace {
constexpr int32_t kBatch = 16;
constexpr float kNegInf = -__builtin_inff();

class CurrentOnly {
 public:
  __aicore__ void Init(GM_ADDR currentOut, GM_ADDR currentLse,
      GM_ADDR output, GM_ADDR outputLse, GM_ADDR rowStatus,
      int64_t rows, int64_t dim) {
    in_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(currentOut));
    inLse_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(currentLse));
    out_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(output));
    outLse_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(outputLse));
    status_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rowStatus));
    rows_ = rows;
    dim_ = dim;
    pipe_.InitBuffer(inputBuf_, kBatch * kMaxDim * 2);
    pipe_.InitBuffer(valueBuf_, kBatch * kMaxDim * 4);
    pipe_.InitBuffer(accBuf_, kBatch * kMaxDim * 4);
    pipe_.InitBuffer(outputBuf_, kBatch * kMaxDim * 2);
    pipe_.InitBuffer(inputLseBuf_, kBatch * 4);
    pipe_.InitBuffer(outputLseBuf_, kBatch * 4);
    pipe_.InitBuffer(statusBuf_, kBatch * 4);
    pipe_.InitBuffer(scalarBuf_, 32);
    pipe_.InitBuffer(logBuf_, 32);
    pipe_.InitBuffer(reduceBuf_, kMaxDim * 4);
  }

  __aicore__ void Process() {
    auto input = inputBuf_.Get<bfloat16_t>();
    auto values = valueBuf_.Get<float>();
    auto accumulator = accBuf_.Get<float>();
    auto output = outputBuf_.Get<bfloat16_t>();
    auto inputLse = inputLseBuf_.Get<float>();
    auto outputLse = outputLseBuf_.Get<float>();
    auto statuses = statusBuf_.Get<int32_t>();
    auto scalar = scalarBuf_.Get<float>();
    auto logarithm = logBuf_.Get<float>();
    auto reduce = reduceBuf_.Get<float>();
    for (int64_t start = GetBlockIdx() * kBatch;
         start < rows_; start += GetBlockNum() * kBatch) {
      const int32_t count = static_cast<int32_t>(
          rows_ - start < kBatch ? rows_ - start : kBatch);
      Fence<HardEvent::S_MTE2>();
      DataCopy(input, in_[start * dim_], count * dim_);
      DataCopyExtParams lseCopy{1, static_cast<uint32_t>(count * 4), 0, 0, 0};
      DataCopyPadExtParams<float> lsePad{false, 0, 0, 0};
      DataCopyPad(inputLse, inLse_[start], lseCopy, lsePad);
      Fence<HardEvent::MTE2_V>();
      Fence<HardEvent::MTE2_S>();
      for (int32_t row = 0; row < count; ++row) {
        const float x = inputLse.GetValue(row);
        const bool empty = x == kNegInf;
        const bool invalid = !empty && !Finite(x);
        auto accRow = accumulator[row * dim_];
        auto valueRow = values[row * dim_];
        auto outputRow = output[row * dim_];
        Duplicate(accRow, 0.0F, dim_);
        if (!empty && !invalid) {
          Cast(valueRow, input[row * dim_], RoundMode::CAST_NONE, dim_);
          PipeBarrier<PIPE_V>();
          Muls(valueRow, valueRow, 1.0F, dim_);
          PipeBarrier<PIPE_V>();
          Add(accRow, accRow, valueRow, dim_);
          PipeBarrier<PIPE_V>();
          ReduceSum(scalar, accRow, reduce, dim_);
          Fence<HardEvent::V_S>();
          statuses.SetValue(row, Finite(scalar.GetValue(0)) ? 0 : 2);
          // merge_lse(splits=1) computes maximum + Log(exp(0)); retaining
          // that operation normalizes -0 LSE to +0 without changing x!=0.
          scalar.SetValue(0, 1.0F);
          Fence<HardEvent::S_V>();
          Log(logarithm, scalar, 1);
          Fence<HardEvent::V_S>();
          outputLse.SetValue(row, x + logarithm.GetValue(0));
        } else {
          statuses.SetValue(row, invalid ? 2 : 0);
          outputLse.SetValue(row, kNegInf);
        }
        PipeBarrier<PIPE_V>();
        Cast(outputRow, accRow, RoundMode::CAST_RINT, dim_);
        PipeBarrier<PIPE_V>();
      }
      Fence<HardEvent::V_MTE3>();
      Fence<HardEvent::S_MTE3>();
      DataCopy(out_[start * dim_], output, count * dim_);
      DataCopyExtParams scalarCopy{1, static_cast<uint32_t>(count * 4), 0, 0, 0};
      DataCopyPad(outLse_[start], outputLse, scalarCopy);
      DataCopyPad(status_[start], statuses, scalarCopy);
      // #145: next batch reuses the same output UB; complete both MTE3
      // consumers before Vector/scalar overwrites it.
      Fence<HardEvent::MTE3_V>();
      Fence<HardEvent::MTE3_S>();
    }
  }

 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> inputBuf_, valueBuf_, accBuf_, outputBuf_,
      inputLseBuf_, outputLseBuf_, statusBuf_, scalarBuf_, logBuf_, reduceBuf_;
  GlobalTensor<bfloat16_t> in_, out_;
  GlobalTensor<float> inLse_, outLse_;
  GlobalTensor<int32_t> status_;
  int64_t rows_, dim_;
};
}  // namespace

extern "C" __global__ __aicore__ void oscar_current_only_kernel(
    GM_ADDR currentOut, GM_ADDR currentLse, GM_ADDR output,
    GM_ADDR outputLse, GM_ADDR rowStatus, int64_t rows, int64_t dim) {
  CurrentOnly op;
  op.Init(currentOut, currentLse, output, outputLse, rowStatus, rows, dim);
  op.Process();
}
