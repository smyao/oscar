// Research metadata-only qualification. Archive #126/#145/#154; startup D.4.
// Setup for quantization rotation; bounded 8*D output, not history restore.
// Exact old/new tables required before any store integration or speed claim.
#include "vector_indices.h"
using namespace AscendC;
using oscar_ascend_device::Fence;
template<bool Vector> __aicore__ inline void Run(GM_ADDR indexOut, GM_ADDR signOut,int32_t dim) {
  TPipe pipe;
  TBuf<TPosition::VECCALC> indexBuf,signBuf,baseBuf,maskBuf,tmpBuf;
  pipe.InitBuffer(indexBuf,8*256*4);pipe.InitBuffer(signBuf,8*256*4);
  pipe.InitBuffer(baseBuf,256*4);pipe.InitBuffer(maskBuf,256*4);pipe.InitBuffer(tmpBuf,256*4);
  auto indices=indexBuf.Get<uint32_t>();auto signs=signBuf.Get<float>();
  Duplicate(indices,0xa5a5a5a5U,8*256);
  Duplicate(signs.ReinterpretCast<uint32_t>(),0xa5a5a5a5U,8*256);
  Fence<HardEvent::V_S>();
  if constexpr(Vector) {
    oscar_hadamard_indices_trial::Build(indices,signs,baseBuf.Get<int32_t>(),
        maskBuf.Get<int32_t>(),tmpBuf.Get<int32_t>(),dim);
  } else {
    int32_t stage=0;
    for(int32_t distance=1;distance<dim;distance*=2,++stage)
      for(int32_t i=0;i<dim;++i) {
        indices.SetValue(stage*dim+i,static_cast<uint32_t>((i^distance)*4));
        signs.SetValue(stage*dim+i,(i&distance)?-1.0F:1.0F);
      }
    Fence<HardEvent::S_V>();
  }
  GlobalTensor<uint32_t> indicesGm;GlobalTensor<float> signsGm;
  indicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(indexOut));
  signsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(signOut));
  Fence<HardEvent::V_MTE3>();
  DataCopy(indicesGm,indices,8*256);DataCopy(signsGm,signs,8*256);
  Fence<HardEvent::MTE3_S>();
}
extern "C" __global__ __aicore__ void oscar_hadamard_indices_old(GM_ADDR i,GM_ADDR s,int32_t dim) {Run<false>(i,s,dim);}
extern "C" __global__ __aicore__ void oscar_hadamard_indices_vector(GM_ADDR i,GM_ADDR s,int32_t dim) {Run<true>(i,s,dim);}
