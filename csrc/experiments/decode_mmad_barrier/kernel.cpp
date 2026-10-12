// SPDX-License-Identifier: Apache-2.0
// RESEARCH ONLY. Archive G3-G13/#21/#81-85/#126/#140/#145/#154; full D.4:
// 1. Isolate the existing GM FP32 QK/PV/Rv Matmul inside the FIA phase.
// 2. D.4 failed on 6499.8-6655.1ms full-history restore, not its 18.5-18.9ms
//    FIA. This fixed 32x256/256x256 primitive never restores any history.
// 3. One Matmul and one TPipe per kernel; no overlapping L0 pools. A retains
//    the production config; B shrinks only M resource capacity; C deliberately
//    tests K64/M32/N128 with zero input rows. No HF32 or quantizer change.
// 4. C may change reduction results or become slower. Any live bit difference
//    rejects this prototype; primitive cycles never establish decode speedup.
// SDK9.1 asc/impl/adv_api/detail/matmul/stage/compute/mmad_compute.h:79-85
// inserts PIPE_M if (actualM/16)*(actualN/16)<10. We do not edit/delete it.
// Native AIC-only task precedent: batch_matmul_transpose_kernel.cpp:661.
#include "../../kernels/oscar_common.h"
#include "lib/matmul_intf.h"
#include "lib/matmul/constant_tiling.h"
using namespace oscar_ascend_device;
namespace {
using Matrix=MatmulType<TPosition::GM,CubeFormat::ND,float,false>;
using TransposedMatrix=MatmulType<TPosition::GM,CubeFormat::ND,float,true>;
template<int Arm> __aicore__ constexpr MatmulConfig Config() {
  auto cfg=GetNormalConfig();
  cfg.basicM=Arm==0?64:32;cfg.basicN=Arm==2?128:64;cfg.basicK=Arm==2?64:128;
  cfg.singleCoreM=Arm==0?128:32;cfg.singleCoreN=256;cfg.singleCoreK=256;
  cfg.enableSetBias=false;
  return cfg;
}
template<int Arm> class Probe {
  static constexpr auto tiling=GetMatmulApiTiling<Matrix,TransposedMatrix,Matrix,Matrix>(Config<Arm>());
  using MM=matmul::MatmulImpl<Matrix,TransposedMatrix,Matrix,Matrix,tiling>;
  static_assert(tiling.shareL1Size<=512*1024,"single-instance L1 exceeds A2 budget");
 public:
  __aicore__ void Run(GM_ADDR a,GM_ADDR b,GM_ADDR c,GM_ADDR status,int64_t live,int64_t layout) {
    GlobalTensor<float> ag,bg,cg;GlobalTensor<int32_t> sg;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    sg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(status));
    if(GetBlockNum()!=1 || (live!=18 && live!=24) || layout<0 || layout>2) {
      sg.SetValue(0,-1);
      DataCacheCleanAndInvalid<int32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(sg);
      return;
    }
    const int32_t rows=Arm==2?32:static_cast<int32_t>(live);
    mm.Init(static_cast<const TCubeTiling*>(nullptr),&pipe);
    SetHF32Mode(false);
    Marker(0);
    mm.SetOrgShape(rows,256,256);mm.SetSingleShape(rows,256,256);
    mm.SetTensorA(ag,false);mm.SetTensorB(bg,layout!=1);
    mm.template IterateAll<true>(cg);mm.End();
    Marker(1);
    // Scalar markers delimit SDK submission/End, not a fabricated device
    // clock. Actual MMAD/barrier completion comes from the simulator trace.
    const int32_t values[16]={1,Arm,static_cast<int32_t>(live),rows,static_cast<int32_t>(layout),
      Config<Arm>().basicM,Config<Arm>().basicN,Config<Arm>().basicK,Config<Arm>().singleCoreM,
      tiling.shareL1Size,tiling.shareL0CSize,tiling.depthA1,tiling.depthB1,tiling.dbL0A,tiling.dbL0B,0};
    for(int32_t i=0;i<16;++i)sg.SetValue(i,values[i]);
    DataCacheCleanAndInvalid<int32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(sg);
  }
 private:
  __aicore__ inline void Marker(uint64_t end) {
#ifdef OSCAR_MMAD_MARKERS
    const uint64_t tag=0x4d4d426100000000ULL | (static_cast<uint64_t>(Arm)<<8) | end;
    uint64_t output;
    __asm__ __volatile__("MOV %0, %1\n" : "=l"(output) : "l"(tag) : "memory");
#endif
  }
  TPipe pipe;MM mm;
};
}
#define ENTRY(NAME,ARM) \
extern "C" __global__ __aicore__ void NAME(GM_ADDR a,GM_ADDR b,GM_ADDR c,GM_ADDR status,int64_t live,int64_t layout) { \
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY); \
  Probe<ARM> op;op.Run(a,b,c,status,live,layout); \
}
ENTRY(oscar_decode_mmad_a_kernel,0)
ENTRY(oscar_decode_mmad_b_kernel,1)
ENTRY(oscar_decode_mmad_c_kernel,2)
#undef ENTRY
