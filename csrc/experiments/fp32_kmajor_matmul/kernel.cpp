// SPDX-License-Identifier: Apache-2.0
// RESEARCH ONLY. Archive G3-G13/#13-22/#81-85/#126/#140/#145/#148/#154-156.
// Startup D.4 full original log reread; four questions:
// 1. Bounded QK/PV Cube movement/compute inside fused FIA, not a service path.
// 2. D.4 full-history dequant cost 6499.8-6655.1ms; never restore full history.
// 3. Only fixed A[32/128,256],B[256,256] NZ buffers. Retain FP32/HF32-off,
//    M24 or separate M64/M62, N64,K128; every output does K0 then K128.
//    Independent C slabs preserve old NZ stride. Two proven Load3D calls per
//    K build separate A slabs; never assume a large ZA tensor's subview.
// 4. Actual A transfer bytes may fall4x, prefill B2x. This is not wall speedup:
//    full-GM prepacking/producer cost excluded, all numerical/timing gates pending.
// Native A2 hc_pre: float Load3D/MMAD event API (its HF32 setting NOT copied).
// SDK snapshot cann_sdk_cube_public_20261010: MmadParams range4095; F32/F32 mad;
// LoadData3DParamsV2, dav_c220 LoadData3DV2L12L0{A,B}Cal; Fixpipe float NoQuant.
// SDK matmul load_to_l0b_loadInstr: full QK NZ B has4 Load2D calls per N64 tile.
// SDK mandatory small-M PIPE_M is retained after every M24 MMAD.
// Local CPU finding 2026-10-11: A2 TPipe::Init owns/preseeds M_MTE1
// events0/1/2. Reserve private live protocol events with AllocEventID;
// FetchEventID alone does not reserve distinct slots. Drain then release.
// Local CPU finding: public Load3D's automatic reset writes FMATRIX_LEFT
// even when fMatrixCtrl selects RIGHT. Initialize RIGHT explicitly for PV,
// matching the SDK's B Prepare path; disable that automatic LEFT reset.
#include "../../kernels/oscar_common.h"
#include "lib/matmul_intf.h"
#include "lib/matmul/constant_tiling.h"
using namespace oscar_ascend_device;
namespace {
__aicore__ constexpr MatmulConfig Config() {
  auto cfg=GetNormalConfig();cfg.basicM=64;cfg.basicN=64;cfg.basicK=128;
  cfg.singleCoreM=128;cfg.singleCoreN=256;cfg.singleCoreK=256;cfg.enableSetBias=false;return cfg;
}
__aicore__ bool Valid(int64_t m,int64_t layout) {
  return GetBlockNum()==1 && ((m==24&&layout==0)||(m==126&&layout==1));
}
__aicore__ void Status(GM_ADDR address,int arm,int m,int layout,int l1,int c0,int aLoads) {
  GlobalTensor<int32_t> s;s.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(address));
  const int pad=m==24?32:128;
  const int32_t fields[32]={1,arm,m,layout,4,64,64,128,128,256,256,l1,c0,
    1,2,0,aLoads,8,(m+63)/64,8*((m+63)/64),4*((m+63)/64),arm,0,0,1,8,1,pad,pad,m,0,0};
  for(int i=0;i<32;++i)s.SetValue(i,fields[i]);
  DataCacheCleanAndInvalid<int32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(s);
  DataCacheCleanAndInvalid<int32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(s[16]);
}
class Reference {
  using A=MatmulType<TPosition::GM,CubeFormat::NZ,float,false>;
  using B=MatmulType<TPosition::GM,CubeFormat::NZ,float,true>;
  using C=MatmulType<TPosition::GM,CubeFormat::ND,float,false>;
  static constexpr auto tiling=GetMatmulApiTiling<A,B,C,C>(Config());
  using MM=matmul::MatmulImpl<A,B,C,C,tiling>;
 public:
  __aicore__ void Run(GM_ADDR a,GM_ADDR b,GM_ADDR c,GM_ADDR status,int64_t m,int64_t layout) {
    if(!Valid(m,layout))return;
    GlobalTensor<float> ag,bg,cg;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    mm.Init(static_cast<const TCubeTiling*>(nullptr),&pipe);SetHF32Mode(false);
    mm.SetOrgShape(m==24?32:128,256,256);mm.SetSingleShape(m,256,256);
    mm.SetTensorA(ag,false);mm.SetTensorB(bg,layout==0);mm.template IterateAll<true>(cg);mm.End();
    Status(status,0,m,layout,tiling.shareL1Size,tiling.shareL0CSize,8*((m+63)/64));
  }
 private: TPipe pipe;MM mm;
};
class KMajor {
 public:
  __aicore__ void Run(GM_ADDR a,GM_ADDR b,GM_ADDR c,GM_ADDR status,int64_t rows,int64_t layout) {
    if(!Valid(rows,layout))return;
    m=static_cast<int>(rows);pad=m==24?32:128;blocks=(m+63)/64;
    GlobalTensor<float> ag,bg,cg;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    pipe.InitBuffer(a1buf,pad*256*4);pipe.InitBuffer(b1buf,256*256*4);
    pipe.InitBuffer(a0buf,64*1024);pipe.InitBuffer(b0buf,64*1024);pipe.InitBuffer(c0buf,128*1024);
    a1=a1buf.Get<float>();b1=b1buf.Get<float>();a0=a0buf.Get<float>();b0=b0buf.Get<float>();c0=c0buf.Get<float>();
    SetHF32Mode(false);
    // Both inputs are already full NZ. Descriptor lengths are32-byte blocks.
    DataCopy(a1,ag,{1,static_cast<uint16_t>(pad*256*4/32),0,0});
    DataCopy(b1,bg,{1,8192,0,0});
    Fence<HardEvent::MTE2_MTE1>();
    TEventID freeIds[3],readyIds[2];
    for(int i=0;i<3;++i) {
      freeIds[i]=pipe.AllocEventID<HardEvent::M_MTE1>();
      SetFlag<HardEvent::M_MTE1>(freeIds[i]);
    }
    for(int i=0;i<2;++i)readyIds[i]=pipe.AllocEventID<HardEvent::MTE1_M>();
    for(int k=0;k<2;++k) {
      WaitFlag<HardEvent::M_MTE1>(freeIds[2]);
      LoadA(k*128);
      WaitFlag<HardEvent::M_MTE1>(freeIds[0]);LoadB(0,k*128,0,layout);SetFlag<HardEvent::MTE1_M>(readyIds[0]);
      for(int n=0;n<4;++n) {
        const int slot=n&1;
        // Queue next independent B before current MMAD. Its ready event is
        // already published before any later MTE1 wait can block that pipe.
        if(n<3) {
          const int next=(n+1)&1;
          WaitFlag<HardEvent::M_MTE1>(freeIds[next]);LoadB(next,k*128,(n+1)*64,layout);
          SetFlag<HardEvent::MTE1_M>(readyIds[next]);
        }
        WaitFlag<HardEvent::MTE1_M>(readyIds[slot]);
        for(int block=0;block<blocks;++block) {
          const int live=block==0?(m<64?m:64):m-64;
          MmadParams q;q.m=live;q.n=64;q.k=128;q.unitFlag=0;
          q.cmatrixSource=false;q.cmatrixInitVal=k==0;q.isBias=false;
          q.fmOffset=0;q.enSsparse=false;q.enWinogradA=false;q.enWinogradB=false;q.kDirectionAlign=false;
          Mmad(c0[(n*2+block)*4096],a0[block*8192],b0[slot*8192],q);
          if((live/16)*(64/16)<10)PipeBarrier<PIPE_M>();
        }
        // Both M consumers must finish before B is overwritten.
        SetFlag<HardEvent::M_MTE1>(freeIds[slot]);
      }
      // All N/M consumers must finish before either A slab is overwritten.
      SetFlag<HardEvent::M_MTE1>(freeIds[2]);
    }
    for(int i=0;i<3;++i) {
      WaitFlag<HardEvent::M_MTE1>(freeIds[i]);
      pipe.ReleaseEventID<HardEvent::M_MTE1>(freeIds[i]);
    }
    for(int i=0;i<2;++i)pipe.ReleaseEventID<HardEvent::MTE1_M>(readyIds[i]);
    Fence<HardEvent::M_FIX>();
    for(int n=0;n<4;++n)for(int block=0;block<blocks;++block) {
      const int live=block==0?(m<64?m:64):m-64;
      FixpipeParamsV220 f;f.nSize=64;f.mSize=live;f.srcStride=(live+15)/16*16;f.dstStride=256;
      f.quantPre=QuantMode_t::NoQuant;f.deqScalar=0;f.ndNum=1;f.srcNdStride=0;f.dstNdStride=0;
      f.reluEn=false;f.unitFlag=0;f.isChannelSplit=false;
      Fixpipe<float,float,CFG_ROW_MAJOR>(cg[block*64*256+n*64],c0[(n*2+block)*4096],f);
    }
    // Native hamming_dist_top_k_parallel.h:399-400: explicit FIX completion.
    Fence<HardEvent::FIX_MTE2>();
    // Drain the completion wait before scalar status is published.
    PipeBarrier<PIPE_ALL>();Status(status,1,m,layout,(pad+256)*256*4,128*1024,2*blocks);
  }
 private:
  __aicore__ LoadData3DParamsV2<float> LoadParams(int width,int channels,int mExt,int kExt,int mStart,int kStart,bool b) {
    LoadData3DParamsV2<float> p;
    for(int i=0;i<4;++i)p.padList[i]=0;
    p.l1H=1;p.l1W=width;p.channelSize=channels;p.mExtension=mExt;p.kExtension=kExt;
    p.mStartPt=mStart;p.kStartPt=kStart;p.strideW=1;p.strideH=1;p.filterW=1;p.filterH=1;
    p.dilationFilterW=1;p.dilationFilterH=1;p.enTranspose=false;p.enSmallK=false;p.padValue=0.0F;
    p.filterSizeW=false;p.filterSizeH=false;p.fMatrixCtrl=b;return p;
  }
  __aicore__ void LoadA(int k) {
    static constexpr IsResetLoad3dConfig reset={true,true};
    for(int block=0;block<blocks;++block) {
      const int live=block==0?(m<64?m:64):m-64;
      auto p=LoadParams(pad,256,(live+15)/16*16,128,block*64,k,false);
      LoadData<float,reset>(a0[block*8192],a1,p);
    }
  }
  __aicore__ void LoadB(int slot,int k,int n,int layout) {
    if(layout==0) {
      // Full bL1N=256: SDK TransLoadDataToL0 N-loop, C0=8,N0=16.
      LoadData2DParams p;p.startIndex=0;p.repeatTimes=16;p.srcStride=16;p.sid=0;
      p.dstGap=3;p.ifTranspose=false;p.addrMode=0;
      for(int j=0;j<4;++j)LoadData(b0[slot*8192+j*128],b1[n*8+k*256+j*128],p);
    } else {
      static constexpr IsResetLoad3dConfig reset={false,true};
      auto p=LoadParams(256,256,128,64,k,n,true);
      SetFmatrix(1,256,p.padList,FmatrixMode::FMATRIX_RIGHT);
      LoadData<float,reset>(b0[slot*8192],b1,p);
    }
  }
  int m,pad,blocks;TPipe pipe;
  TBuf<TPosition::A1> a1buf;TBuf<TPosition::B1> b1buf;
  TBuf<TPosition::A2> a0buf;TBuf<TPosition::B2> b0buf;TBuf<TPosition::CO1> c0buf;
  LocalTensor<float> a1,b1,a0,b0,c0;
};
}
extern "C" __global__ __aicore__ void oscar_fp32_sdk_kmajor_reference_kernel(
    GM_ADDR a,GM_ADDR b,GM_ADDR c,GM_ADDR status,int64_t rows,int64_t layout) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);Reference op;op.Run(a,b,c,status,rows,layout);
}
extern "C" __global__ __aicore__ void oscar_fp32_kmajor_matmul_kernel(
    GM_ADDR a,GM_ADDR b,GM_ADDR c,GM_ADDR status,int64_t rows,int64_t layout) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);KMajor op;op.Run(a,b,c,status,rows,layout);
}
