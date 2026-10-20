// Archive #164/#126/#145: CPU-debug functional comparison, never a timing test.
// D.4: helper-only UB tiles; no full-history restore or production entrypoint.
// Read D.4 in full before writing this probe. It checks unchanged conversion
// bits and guards; target NPU completion/performance remain separate.
#include "attention_bf16_cast.h"
#include "attention_c16_cast.h"
using namespace AscendC;
using namespace oscar_ascend_device;

extern "C" __global__ __aicore__ void oscar_c16_cast_pair_kernel(
    GM_ADDR source, GM_ADDR oldResult, GM_ADDR newResult, GM_ADDR controls,
    int32_t count) {
  // This independent probe intentionally covers the 255-repeat boundary,
  // beyond production C16's largest 8192-element call. 161024 total UB bytes.
  constexpr int32_t kElements = 20128;
  constexpr int32_t kGuard = 64;
  TPipe pipe;
  TBuf<TPosition::VECCALC> inputBuf, oldBuf, newBuf;
  pipe.InitBuffer(inputBuf, kElements * sizeof(float));
  pipe.InitBuffer(oldBuf, kElements * sizeof(bfloat16_t));
  pipe.InitBuffer(newBuf, kElements * sizeof(bfloat16_t));
  auto input = inputBuf.Get<float>();
  auto oldOutput = oldBuf.Get<bfloat16_t>();
  auto newOutput = newBuf.Get<bfloat16_t>();
  GlobalTensor<float> inputGm;
  GlobalTensor<bfloat16_t> oldGm, newGm;
  GlobalTensor<int64_t> ctrlGm;
  inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(source), kElements);
  oldGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(oldResult), kElements);
  newGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(newResult), kElements);
  ctrlGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(controls), 4);
  DataCopy(input, inputGm, kElements);
  DataCopy(oldOutput, oldGm, kElements);
  DataCopy(newOutput, newGm, kElements);
  Fence<HardEvent::MTE2_V>();
  // SDK dav_c220/kernel_operator_common_impl.h: FLOAT overflow uses CTRL[48];
  // 0 means saturation, 1 means infinity. Neither helper may change this.
  ctrlGm.SetValue(0, GetCtrlSpr<48, 48>());
  oscar_bf16::CastRne(oldOutput[kGuard], input[kGuard], count);
  ctrlGm.SetValue(1, GetCtrlSpr<48, 48>());
  oscar_c16_cast::CastRne(newOutput[kGuard], input[kGuard], count);
  ctrlGm.SetValue(2, GetCtrlSpr<48, 48>());
  Fence<HardEvent::V_MTE3>();
  DataCopy(oldGm, oldOutput, kElements);
  DataCopy(newGm, newOutput, kElements);
  PipeBarrier<PIPE_ALL>();
}
