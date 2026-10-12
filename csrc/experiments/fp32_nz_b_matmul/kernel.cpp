// SPDX-License-Identifier: Apache-2.0
// RESEARCH ONLY. Archive G3-G13/#21/#81-85/#126/#140/#145/#148/#154-156;
// startup D.4 four questions (complete log reread):
// 1. Test the bounded FIA Cube B transfer; A/C and all math remain FP32 ND.
// 2. Historical full-cache restore took6.5s; no history-sized tensor here.
// 3. Only B's lossless GM layout changes ND->NZ; both kernels retain
//    basic64/64/128, trueM24/M126, K128 accumulation and HF32-off.
//    Host prepack cost is excluded: this is an optimistic ceiling only.
// 4. Require same output bits and real MMAD K128 sequence before timing;
//    conversion/publisher/Rv costs and complete decode gain remain unknown.
// SDK9.1 mx_matmul_utils.h:102-105 + matmul_config_utils.h:41 give F32 C0=8.
// copy_tile_to_cube_common.h:484-489 selects direct CopyNZ2NZ, whose GM
// index is (col/8)*align16(rows)*8 + row*8 + col%8 (wrapper_utils:292-334).
// Do not host-transpose B: runtime B flag stays true QK / false PV.
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
template<CubeFormat Format,int Arm> class Probe {
  using A=MatmulType<TPosition::GM,CubeFormat::ND,float,false>;
  using B=MatmulType<TPosition::GM,Format,float,true>;
  using C=MatmulType<TPosition::GM,CubeFormat::ND,float,false>;
  static constexpr auto tiling=GetMatmulApiTiling<A,B,C,C>(Config());
  using MM=matmul::MatmulImpl<A,B,C,C,tiling>;
 public:
  __aicore__ void Run(GM_ADDR a,GM_ADDR b,GM_ADDR c,GM_ADDR status,int64_t rows,int64_t layout) {
    GlobalTensor<float> ag,bg;GlobalTensor<float> cg;GlobalTensor<int32_t> sg;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
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
    const int32_t fields[32]={1,Arm,static_cast<int32_t>(rows),static_cast<int32_t>(layout),
      4,64,64,128,128,256,256,tiling.shareL1Size,tiling.shareL0CSize,
      tiling.dbL0A,tiling.dbL0B,0,tiling.depthA1,tiling.depthB1,
      tiling.stepM,tiling.stepN,tiling.stepKa,tiling.stepKb,
      static_cast<int32_t>(tiling.iterateOrder),tiling.dbL0C,static_cast<int32_t>(Format),8};
    for(int32_t i=0;i<32;++i)sg.SetValue(i,fields[i]);
    DataCacheCleanAndInvalid<int32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(sg);
    DataCacheCleanAndInvalid<int32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(sg[16]);
  }
 private:
  TPipe pipe;MM mm;
};
}
extern "C" __global__ __aicore__ void oscar_fp32_nd_b_matmul_kernel(
    GM_ADDR a,GM_ADDR b,GM_ADDR c,GM_ADDR status,int64_t rows,int64_t layout) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
  Probe<CubeFormat::ND,0> op;op.Run(a,b,c,status,rows,layout);
}
extern "C" __global__ __aicore__ void oscar_fp32_nz_b_matmul_kernel(
    GM_ADDR a,GM_ADDR b,GM_ADDR c,GM_ADDR status,int64_t rows,int64_t layout) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
  Probe<CubeFormat::NZ,1> op;op.Run(a,b,c,status,rows,layout);
}
