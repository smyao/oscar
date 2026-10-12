// SPDX-License-Identifier: Apache-2.0
// Archive G25-G34/#13-22/#126/#129/#145; startup D.4 four questions:
// (1) Isolate only the C16 FP32 pre-softmax range mask; no attention math.
// (2) D.4's 6.5s full-history restore is forbidden and absent here.
// (3) Both arms use one 256-float UB row; old per-column SetValue is the
//     current production rule, new arm uses bounded complement intervals.
// (4) CPU-debug proves bitwise score rows; raw SYS_CNT is device-only evidence
//     if this independent microkernel is later run on a real A2 NPU.
#include "range_mask.h"

using namespace AscendC;
using oscar_ascend_device::Fence;

namespace {
constexpr int32_t kColumns = oscar_range_mask::kColumns;
constexpr float kNegativeInfinity = -__builtin_inff();
struct Visible {
  int32_t lo0, hi0, lo1, hi1;
};

__aicore__ inline void Initialize(LocalTensor<float> dst) {
  Fence<HardEvent::S_V>();
  Duplicate(dst, 0.125F, kColumns);
  Fence<HardEvent::V_S>();
  dst.SetValue(0, -0.0F);
  dst.SetValue(1, +0.0F);
  dst.SetValue(7, __builtin_nanf("0x123"));
  dst.SetValue(8, +__builtin_inff());
  dst.SetValue(9, -__builtin_inff());
  dst.SetValue(127, -7.5F);
  dst.SetValue(128, __builtin_nanf("0x321"));
  dst.SetValue(255, 3.125F);
}

__aicore__ inline bool VisibleColumn(int32_t column, Visible range) {
  return (column >= range.lo0 && column < range.hi0) ||
         (column >= range.lo1 && column < range.hi1);
}

class Probe {
 public:
  __aicore__ void Run(GM_ADDR output, int32_t mode) {
    if (GetBlockIdx() != 0) return;
    out.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(output));
    pipe.InitBuffer(oldBuf, kColumns * sizeof(float));
    pipe.InitBuffer(newBuf, kColumns * sizeof(float));
    int64_t checked = 0;
    int64_t failures = 0;
    int64_t firstBad = -1;
    if (mode == 0) {
      for (int32_t begin = 0; begin <= kColumns; ++begin) {
        for (int32_t end = begin; end <= kColumns; ++end)
          Check({begin, end, 0, 0}, checked, failures, firstBad);
      }
    } else if (mode == 1) {
      const Visible examples[] = {
          {0, 64, 192, 256}, {1, 7, 9, 15}, {7, 129, 129, 255},
          {64, 65, 129, 130}, {0, 1, 255, 256}, {1, 128, 129, 256},
          {63, 64, 191, 192}, {0, 0, 0, 0}, {0, 256, 0, 0},
      };
      for (int32_t i = 0; i < 9; ++i)
        Check(examples[i], checked, failures, firstBad);
    } else {
      failures = -1;
    }
    out.SetValue(0, checked);
    out.SetValue(1, failures);
    out.SetValue(2, firstBad);
    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE,
        DcciDst::CACHELINE_OUT>(out);
  }

 private:
  __aicore__ void Check(Visible range, int64_t& checked,
                        int64_t& failures, int64_t& firstBad) {
    auto oldScores = oldBuf.Get<float>();
    auto newScores = newBuf.Get<float>();
    Initialize(oldScores);
    Initialize(newScores);
    // Prior vector initialization has completed before scalar SetValue.
    for (int32_t j = 0; j < kColumns; ++j) {
      if (!VisibleColumn(j, range)) oldScores.SetValue(j, kNegativeInfinity);
    }
    oscar_range_mask::MaskComplement(newScores,
        range.lo0, range.hi0, range.lo1, range.hi1);
    Fence<HardEvent::V_S>();
    auto oldBits = oldScores.ReinterpretCast<uint32_t>();
    auto newBits = newScores.ReinterpretCast<uint32_t>();
    for (int32_t j = 0; j < kColumns; ++j) {
      ++checked;
      if (oldBits.GetValue(j) != newBits.GetValue(j)) {
        ++failures;
        if (firstBad < 0) firstBad = checked - 1;
      }
    }
  }

  TPipe pipe;
  TBuf<TPosition::VECCALC> oldBuf, newBuf;
  GlobalTensor<int64_t> out;
};
}  // namespace

extern "C" __global__ __aicore__ void oscar_range_mask_probe_kernel(
    GM_ADDR output, int32_t mode) {
  Probe op;
  op.Run(output, mode);
}
