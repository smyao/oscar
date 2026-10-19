// Archive #145/#164; D.4 complete original log read: bounded C16 diagnosis
// only, never historical restoration. Original arithmetic/flags stay intact.
// Extra local completion fences deliberately perturb the diagnostic clone.
// D.4 four questions: (1) phase=fia/history/window; (2) current bounded fused
// C16 tiles remain, never a full-history BF16 arena; (3) new storage is only
// 512B/actor, outputs remain independently owned and paired; (4) no speed claim
// or latency ceiling: report original Event time plus diagnostic perturbation.
#pragma once
#include "kernel_operator.h"
#include "../../kernels/oscar_common.h"
namespace oscar_c16_timer {
using namespace AscendC;
using oscar_ascend_device::Fence;
constexpr int kPhases=24,kHeaderRows=8,kRows=32,kWords=64,kMaxDepth=16;
enum Phase {TaskMetadata,QueryLoad,HistoryLoad,PreciseLoad,UnpackK,UnpackV,
 PublishKv,Qk,SoftmaxLoad,SoftmaxMath,SoftmaxPublish,Pv,StateRestore,StateSpill,
 FinalRv,Normalize,OutputPublish,CrossWait,B1Load,Bf16Cast,TaskOther,
 ClusterOther,InitPhase,ProcessOther};
enum Drain {None=0,Vector=1,Load=2,Store=4,Fix=8};
struct Recorder {
 GlobalTensor<int64_t> gm;
 uint64_t cycles[kPhases],counts[kPhases],starts[kMaxDepth],children[kMaxDepth];
 int phases[kMaxDepth],depth=0,maximum=0,flags=0,dropped=0;
 int actor=0,core=0,engine=0,lane=-1;
 uint64_t begin=0;
 __aicore__ void Start(GM_ADDR pointer) {
  gm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(pointer));
  core=GetBlockIdx();
  if ASCEND_IS_AIV {lane=core%2;core/=2;engine=lane+1;}
  actor=core*3+engine;
  begin=GetSystemCycle();
  for(int p=0;p<kPhases;++p){cycles[p]=0;counts[p]=0;}
 }
 __aicore__ bool Enter(int phase) {
  if(depth>=kMaxDepth){flags|=1;++dropped;return false;}
  phases[depth]=phase;children[depth]=0;starts[depth]=GetSystemCycle();
  ++depth;if(depth>maximum)maximum=depth;return true;
 }
 __aicore__ void Leave(int drain) {
  if(drain&Vector){PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();}
  if(drain&Load)Fence<HardEvent::MTE2_S>();
  if(drain&Store)Fence<HardEvent::MTE3_S>();
  if(drain&Fix)Fence<HardEvent::FIX_S>();
  const uint64_t end=GetSystemCycle();
  --depth;const uint64_t start=starts[depth];
  if(end<start){flags|=2;return;}
  const uint64_t elapsed=end-start;
  if(elapsed<children[depth]){flags|=4;return;}
  const int p=phases[depth];const uint64_t exclusive=elapsed-children[depth];
  if(cycles[p]+exclusive<cycles[p] || cycles[p]+exclusive>0x7fffffffffffffffULL){flags|=8;return;}
  cycles[p]+=exclusive;++counts[p];
  if(depth)children[depth-1]+=elapsed;
 }
 __aicore__ void Finish() {
  const uint64_t end=GetSystemCycle();
  uint64_t sum=0;for(int p=0;p<kPhases;++p)sum+=cycles[p];
  if(depth || end<begin || sum>end-begin)flags|=16;
  const int64_t base=actor*kWords;
  const int64_t header[16]={0x4f534331,1,actor,engine,core,lane,
    static_cast<int64_t>(end-begin),static_cast<int64_t>(sum),flags,1,
    maximum,dropped,static_cast<int64_t>(begin),static_cast<int64_t>(end),kPhases,kWords};
  for(int i=0;i<16;++i)gm.SetValue(base+i,header[i]);
  for(int p=0;p<kPhases;++p){gm.SetValue(base+16+2*p,cycles[p]);gm.SetValue(base+17+2*p,counts[p]);}
  // Every actor owns eight complete64B lines, matching the existing profiler.
  for(int line=0;line<kWords/8;++line)
   DataCacheCleanAndInvalid<int64_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(gm[base+line*8]);
 }
};
struct Scope {
 Recorder& recorder;int drain;bool entered;
 __aicore__ Scope(Recorder& r,int p,int d=None):recorder(r),drain(d),entered(r.Enter(p)){}
 __aicore__ ~Scope(){if(entered)recorder.Leave(drain);}
};
}
