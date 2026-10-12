// SPDX-License-Identifier: Apache-2.0
// RESEARCH ONLY. Archive G3-G13/#21/#81-85/#126/#140/#145/#148/#154-156;
// personally reread complete startup D.4, four questions:
// 1. Isolate one QK/PV Cube Matmul, not a complete attention/serving path.
// 2. D.4's 6.5s full-history reconstruction cannot appear: fixed 256x256 B.
// 3. Same source values are exactly representable in BF16 and FP32; integer
//    dot golden proves all intermediate sums exact in FP32. C stays FP32.
//    Same basic64/64/128, true M24/M126, grid1 and one Matmul per kernel.
// 4. Measure complete local Matmul R only; no claim that arbitrary inputs
//    can be cast, or that multi-component BF16/whole decode gains follow.
// A2 precedent: references/vllm-ascend/csrc/attention/sparse_flash_attention/
// op_kernel/arch22/sparse_flash_attention_service_cube_mla.h:118-122,220,808,832
// uses BF16 operands with float L0C and float GM intermediates. No GDN edits.
#include "../../kernels/oscar_common.h"
#include "lib/matmul_intf.h"
#include "lib/matmul/constant_tiling.h"
using namespace oscar_ascend_device;
namespace {
__aicore__ constexpr MatmulConfig Config() {
  auto cfg=GetNormalConfig();
  cfg.basicM=64;cfg.basicN=64;cfg.basicK=128;
  cfg.singleCoreM=128;cfg.singleCoreN=256;cfg.singleCoreK=256;
  cfg.enableSetBias=false;
  return cfg;
}
template<typename T,int Arm> class Probe {
  using A=MatmulType<TPosition::GM,CubeFormat::ND,T,false>;
  using B=MatmulType<TPosition::GM,CubeFormat::ND,T,true>;
  using C=MatmulType<TPosition::GM,CubeFormat::ND,float,false>;
  static constexpr auto tiling=GetMatmulApiTiling<A,B,C,C>(Config());
  using MM=matmul::MatmulImpl<A,B,C,C,tiling>;
 public:
  __aicore__ void Run(GM_ADDR a,GM_ADDR b,GM_ADDR c,GM_ADDR status,int64_t rows,int64_t layout) {
    GlobalTensor<T> ag,bg;GlobalTensor<float> cg;GlobalTensor<int32_t> sg;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(a));bg.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(b));
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));sg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(status));
    if(GetBlockNum()!=1 || !((rows==24&&layout==0)||(rows==126&&layout==1))) {
      sg.SetValue(0,-1);
      DataCacheCleanAndInvalid<int32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(sg);return;
    }
    mm.Init(static_cast<const TCubeTiling*>(nullptr),&pipe);
    SetHF32Mode(false);
    mm.SetOrgShape(rows,256,256);mm.SetSingleShape(rows,256,256);
    mm.SetTensorA(ag,false);mm.SetTensorB(bg,layout==0);
    mm.template IterateAll<true>(cg);mm.End();
    const int32_t fields[16]={1,Arm,static_cast<int32_t>(rows),static_cast<int32_t>(layout),
      sizeof(T),64,64,128,128,256,256,tiling.shareL1Size,tiling.shareL0CSize,
      tiling.dbL0A,tiling.dbL0B,0};
    for(int32_t i=0;i<16;++i)sg.SetValue(i,fields[i]);
    DataCacheCleanAndInvalid<int32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(sg);
  }
 private:
  TPipe pipe;MM mm;
};
}
extern "C" __global__ __aicore__ void oscar_mmad_ratio_fp32_kernel(
    GM_ADDR a,GM_ADDR b,GM_ADDR c,GM_ADDR status,int64_t rows,int64_t layout) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
  Probe<float,0> op;op.Run(a,b,c,status,rows,layout);
}
extern "C" __global__ __aicore__ void oscar_mmad_ratio_bf16_kernel(
    GM_ADDR a,GM_ADDR b,GM_ADDR c,GM_ADDR status,int64_t rows,int64_t layout) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
  Probe<bfloat16_t,1> op;op.Run(a,b,c,status,rows,layout);
}
