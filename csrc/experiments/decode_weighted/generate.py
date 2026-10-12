"""Archive #129/#140/#145/#155/D.4: isolated full-decode history ownership.

Only the current workspace's same-FP32 bundle is read. Every task, including
padding/faults, still executes exactly once. Math methods stay byte-identical.
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'csrc/kernels/attention_cv_decode_bundle.cpp'
PIN = 'd8171f0403cf176087e1c76c64e454e6fb1e74fd79252aab3b59e848cca19401'

PLANNER = r'''
  // Startup scratch lifetime: dequant UB is not live until the first task.
  // All workers compute the same bucketed ownership from one strided metadata
  // DMA, but publish only their own list. No inter-block wait/atomic queue.
  __aicore__ bool UseWeightedHistory() {
    return D==256 && g.hk==1 && g.tokens>=64 && g.tokens<=128 && g.splits<=3 &&
        GetBlockNum()<=24;
  }
  __aicore__ int64_t ReadPlan(int64_t index) {
    if ASCEND_IS_AIC {return planGm.GetValue(index);}
    else {
      auto tmp=addressBuf.Get<int64_t>();
      const DataCopyExtParams copy{1,8,0,0,0};
      const DataCopyPadExtParams<int64_t> pad{false,0,0,0};
      DataCopyPad(tmp,planGm[index],copy,pad);Fence<HardEvent::MTE2_S>();
      return tmp.GetValue(0);
    }
  }
  __aicore__ void BuildHistoryPlan(const oscar_ascend_schedule::CvTaskSchedule& schedule) {
    if ASCEND_IS_AIV {
      if(lane==0) {
        auto rows=dequantBuf.Get<int64_t>();
        const DataCopyExtParams copy{static_cast<uint16_t>(g.tokens),
            static_cast<uint32_t>(g.splits*128),
            static_cast<uint32_t>(2*g.splits*128),0,0};
        const DataCopyPadExtParams<int64_t> pad{false,0,0,0};
        DataCopyPad(rows,taskGm,copy,pad);Fence<HardEvent::MTE2_S>();
        int64_t load[24]={0},original[24]={0};
        const int64_t historyItems=schedule.QueryTiles()*g.splits;
        // Small list lives in the dead packed-next UB; GM plan has its own
        // fixed 4096B region, never aliases Q/K/V/PV/softmax working memory.
        auto indices=packedNextBuf.Get<int64_t>();
        auto buckets=naturalBuf.Get<int64_t>();
        auto next=buckets[128];auto weights=buckets[512];
        int64_t count=0;
        int64_t totalCost=0,maxCost=0;
        for(int64_t wid=0;wid<historyItems;++wid) {
          const int64_t split=schedule.Segment(wid);
          const int64_t begin=schedule.TokenBegin(wid);
          int64_t cost=0;
          for(int64_t token=begin;token<Min64(begin+OwnerTile,g.tokens);++token) {
            const int64_t base=(token*g.splits+split)*16;
            const int64_t queries=rows.GetValue(base+1);
            if(queries>0) {
              ++cost;
              const int64_t first=rows.GetValue(base+3),last=rows.GetValue(base+4);
              if(queries>0 && first>=0 && last>=first) {
                const int64_t width=last-first;
                cost+=Min64(width/kKvRows+(width%kKvRows!=0),1LL<<24);
              }
            }
          }
          original[wid%GetBlockNum()]+=cost;
          weights.SetValue(wid,cost);
          totalCost+=cost;maxCost=Max64(maxCost,cost);
        }
        int64_t oldMax=0,newMax=0;
        for(int64_t core=0;core<GetBlockNum();++core)oldMax=Max64(oldMax,original[core]);
        const int64_t lowerBound=Max64(maxCost,(totalCost+GetBlockNum()-1)/GetBlockNum());
        if(oldMax*100<=lowerBound*105) {
          newMax=oldMax;
        } else {
        for(int64_t bucket=0;bucket<128;++bucket)buckets.SetValue(bucket,-1);
        for(int64_t wid=0;wid<historyItems;++wid) {
          const int64_t bucket=Min64(weights.GetValue(wid),127);
          next.SetValue(wid,buckets.GetValue(bucket));buckets.SetValue(bucket,wid);
        }
        // Long work first avoids placing the last large history on an
        // already busy core.128 bounded buckets replace a quadratic sort.
        // Costs above127 share a bucket but retain their full load weight.
        for(int64_t bucket=127;bucket>=0;--bucket) {
          for(int64_t wid=buckets.GetValue(bucket);wid>=0;wid=next.GetValue(wid)) {
            const int64_t cost=weights.GetValue(wid);
            int64_t owner=wid%GetBlockNum();
            if(cost) {
              owner=0;
              for(int64_t core=1;core<GetBlockNum();++core)
                if(load[core]<load[owner])owner=core;
              load[owner]+=cost;
            }
            if(owner==coreIndex)indices.SetValue(++count,wid);
          }
        }
        for(int64_t core=0;core<GetBlockNum();++core) {
          newMax=Max64(newMax,load[core]);
        }
        }
        // No cost-fit constants: require >=5% fewer estimated KV units on
        // the busiest worker before changing ownership. The whole-kernel
        // CAModel comparison includes this planner even when it declines.
        if(oldMax*100<=newMax*105) {
          count=0;
          for(int64_t wid=coreIndex;wid<historyItems;wid+=GetBlockNum())
            indices.SetValue(++count,wid);
        }
        indices.SetValue(0,count);
        // Initialize DMA padding too; at most 384 IDs plus count (<4096B).
        const int64_t padded=(count+4)/4*4;
        for(int64_t i=count+1;i<padded;++i)indices.SetValue(i,0);
        Fence<HardEvent::S_MTE3>();DataCopy(planGm,indices,padded);
      }
      CrossCoreSetFlag<2,PIPE_MTE3>(kVectorReady);
      CrossCoreWaitFlag(kCubeReady);
    } else {
      CrossCoreWaitFlag(kVectorReady);
      DataCacheCleanAndInvalid<int64_t,CacheLine::ENTIRE_DATA_CACHE>(planGm);
      CrossCoreSetFlag<2,PIPE_FIX>(kCubeReady);
    }
  }
'''

def main():
    source=SOURCE.read_text()
    if hashlib.sha256(source.encode()).hexdigest()!=PIN:
        raise RuntimeError('Re-audit the pinned bundle before generating the schedule trial')
    start=source.index('    for(int64_t workId=coreIndex;workId<schedule.WorkItems();workId+=GetBlockNum()) {')
    end=source.index('    if ASCEND_IS_AIC {mm.End();}',start)
    loop=source[start:end]
    body=loop[loop.index('      const int64_t tokenBegin='):]
    # Remove only the outer loop close, preserving the complete task body.
    body=body[:-6].replace('tokenBegin+queryTile','tokenBegin+OwnerTile')
    process_work='  __aicore__ void ProcessWork(const oscar_ascend_schedule::CvTaskSchedule& schedule,int64_t workId) {\n'+body+'  }\n'
    replacement='''    if(UseWeightedHistory()) {
      BuildHistoryPlan(schedule);
      const int64_t count=ReadPlan(0);
      for(int64_t i=1;i<=count;++i)ProcessWork(schedule,ReadPlan(i));
      const int64_t historyItems=schedule.QueryTiles()*g.splits;
      const int64_t first=historyItems+(coreIndex+GetBlockNum()-historyItems%GetBlockNum())%GetBlockNum();
      for(int64_t workId=first;workId<schedule.WorkItems();workId+=GetBlockNum())
        ProcessWork(schedule,workId);
    } else {
      for(int64_t workId=coreIndex;workId<schedule.WorkItems();workId+=GetBlockNum())
        ProcessWork(schedule,workId);
    }
'''
    source=source[:start]+replacement+source[end:]
    source=source.replace(' private:\n',' private:\n'+PLANNER+process_work,1)
    old='work.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace)+core*floatsPerCore);'
    new='''planGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(workspace)+core*(floatsPerCore/2+512));
    work.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace)+core*(floatsPerCore+1024)+1024);'''
    assert source.count(old)==1
    source=source.replace(old,new)
    # AIC and AIV own an immutable startup list after a fully drained8/9 pair.
    position=source.index('  Geometry g;')
    source=source[:position]+'  GlobalTensor<int64_t> planGm;\n'+source[position:]
    for old,new in (
        ('oscar_attention_cv_decode_bundle_kernel','oscar_decode_weighted_q4_kernel'),
        ('oscar_attention_cv_decode_bundle_q1_kernel','oscar_decode_weighted_q1_kernel'),
        ('attention_cv_decode_bundle_launch','attention_decode_weighted_q4_launch'),
        ('attention_cv_decode_bundle_q1_launch','attention_decode_weighted_q1_launch')):
        source=source.replace(old,new)
    source=source.replace('#include "oscar_common.h"','#include "../../kernels/oscar_common.h"')
    source=source.replace('#include "attention_', '#include "../../kernels/attention_')
    source=source.replace('#include "../include/', '#include "../../include/')
    header='''// ISOLATED weighted-history ownership; archive #129/#140/#145/#155, D.4.
// (1) Complete q4/q1 source0 FIA, keeping every task/source and original math.
// (2) D.4 full-history restore remains forbidden. Current R32 complete-CAModel
//     workers span10.01M..16.23M ticks; round-robin tile ownership correlates
//     with 20/23/27/30K histories. A plain rotation was rejected for harmingq1.
// (3) One <=49KiB metadata DMA into startup-dead UB; bounded bucket ordering
//     and greedy assignment by KV256 counts, no global atomic/barrier or host copy.
//     Only a fixed4096B/core plan is added; no historical KV tensor/cache.
//     N<64 keeps original ownership without any planning. Padding/fault tasks
//     retain original owners and receive no fictitious KV weight. A lower
//     bound avoids sorting when the old mapping is already within5% of optimal.
// (4) Estimated worker balance is not speed. Complete CPU/CAModel/numeric
//     gates include planning overhead; no NPU or whole-run prediction yet.
'''
    (Path(__file__).parent/'attention_weighted.cpp').write_text(header+source)

if __name__=='__main__': main()
