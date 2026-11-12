// Archive #145/#150/#154/#164/#172 and startup D.4: host control-flow proof.
// Python injects real protected/derived methods. No floating-point kernel or
// asynchronous AscendC behavior is simulated by this executable.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include "oscar_cv_schedule.h"

#define __aicore__
#define ASCEND_IS_AIC (Aic)
#define ASCEND_IS_AIV (!Aic)
using Task = std::array<int64_t, 16>;
using Action = std::array<int64_t, 16>;
void require(bool ok, const char* reason) { if (!ok) throw std::runtime_error(reason); }
int64_t Min64(int64_t a, int64_t b) { return std::min(a, b); }

struct Counters {
  int64_t gm_scalars=0, ub_scalars=0, dma_rows=0, fences=0, invalidations=0, ends=0;
  std::vector<std::array<int64_t,2>> gm_reads, ub_reads;
  std::vector<Action> actions;
};
thread_local Counters* activeCounters=nullptr;
enum class HardEvent { MTE2_S };
enum class CacheLine { ENTIRE_DATA_CACHE };
template<HardEvent> void Fence() { ++activeCounters->fences; }

struct GmSource { const std::vector<Task>* tasks; int64_t offset; Counters* counters; };
struct Global {
  const std::vector<Task>* tasks;
  Counters* counters;
  int64_t GetValue(int64_t offset) {
    require(counters->invalidations>0,"AIC scalar read preceded cache invalidation");
    ++counters->gm_scalars;
    counters->gm_reads.push_back({offset/16,offset%16});
    return tasks->at(offset/16).at(offset%16);
  }
  GmSource operator[](int64_t offset) { return {tasks,offset,counters}; }
};
template<typename T> struct Local {
  Task* data;
  int64_t* task_id;
  Counters* counters;
  T GetValue(int64_t offset) {
    require(counters->fences==counters->dma_rows,"AIV scalar read preceded DMA completion fence");
    ++counters->ub_scalars;
    counters->ub_reads.push_back({*task_id,offset});
    return data->at(offset);
  }
};
struct Buffer {
  Task data{};
  int64_t task_id=-1;
  Counters* counters;
  template<typename T> Local<T> Get() {
    static_assert(std::is_same_v<T,int64_t>);
    return {&data,&task_id,counters};
  }
};
void DataCopy(Local<int64_t> destination, GmSource source, int64_t count) {
  require(source.offset%16==0 && count==16,"AIV DMA contract changed");
  *destination.data=source.tasks->at(source.offset/16);
  *destination.task_id=source.offset/16;
  ++source.counters->dma_rows;
}
template<typename T, CacheLine> void DataCacheCleanAndInvalid(Global& gm) {
  ++gm.counters->invalidations;
}
struct Matmul { Counters* counters; void End() { ++counters->ends; } };
struct Geometry {
  int64_t tokens,hq=6,hk=1,requests,columns=512,splits;
  int32_t dequantMode=0;
};

template<int OwnerTile, bool Aic, bool Active> struct Kernel {
  Counters counters;
  Geometry g;
  Global taskGm;
  Buffer taskBuf;
  Matmul mm;
  int64_t coreIndex=0, cores=1, lane=0;
  int64_t qbegin=0,qcount=0,kvhead=0,kvbegin=0,kvend=0,request=0,split=0,
          kind=0,context=0,requestBegin=0;
  int32_t taskError=0;
  static constexpr int32_t kQueryRows=32;
  Kernel(std::vector<Task>& tasks, Geometry geometry):g(geometry),taskGm{&tasks,&counters},
      taskBuf{{},-1,&counters},mm{&counters} {}
  int64_t GetBlockNum() const { return cores; }
  void Run() { activeCounters=&counters; Process(); activeCounters=nullptr; }
  void LoadTask(int64_t id) {
    if constexpr(Active) LoadTaskActive(id); else LoadTaskBaseline(id);
  }
  void record(int64_t action, int64_t id, int64_t code, bool fields) {
    Action value{}; value[0]=action;value[1]=id;value[2]=code;value[3]=lane;
    if(fields) {
      const int64_t field[]={qbegin,qcount,kvhead,kvbegin,kvend,request,split,
                             kind,context,requestBegin,taskError};
      std::copy(field,field+11,value.begin()+4);
    }
    counters.actions.push_back(value);
  }
  void HistoryCubeTask() { record(1,-1,0,true); }
  void HistoryVectorTask(int64_t id) { record(2,id,0,true); }
  void CubeTask() { record(3,-1,0,true); }
  void VectorTask(int64_t id) { record(4,id,0,true); }
  void PublishStatus(int64_t id,int32_t code) { record(5,id,code,false); }
  void PublishEmpty(int64_t id,int32_t code) {
    record(6,id,code,true);
    // Preserve the actual PublishEmpty output-owner qualification and status.
    if(qbegin>=0 && qbegin<g.tokens && g.hk>0 && g.hq>0 && g.hq%g.hk==0 &&
       g.hq/g.hk<=8 && kvhead>=0 && kvhead<g.hk && split>=0 && split<3*g.splits) {
      qcount=1;PublishStatus(id,code);
    } else PublishStatus(id,code?code:1);
  }
@@BASELINE_LOAD@@
@@ACTIVE_LOAD@@
@@PROCESS@@
@@TASK_VALID@@
};

Task active_task(int64_t token,int64_t q,int64_t source,int64_t split,int64_t s) {
  Task t{};
  t[0]=token;t[1]=q;t[2]=0;t[3]=source==2?1024:64;t[4]=source==2?1024+q:128;
  t[5]=token/q;t[6]=source*s+split;t[7]=source;t[8]=1024;t[9]=token;
  return t;
}
Task poison_zero(int64_t seed) {
  Task t{};
  for(int j=0;j<16;++j)t[j]=(j%2==0?std::numeric_limits<int64_t>::min():
                           std::numeric_limits<int64_t>::max())+seed;
  t[1]=0;
  return t;
}
Task native_negative_slot(int64_t token,int64_t source,int64_t split,int64_t s) {
  // Real prepare: count starts at -1 (attention_official_tasks.cpp:83),
  // and negative slots do not enter !padding, where followers become zero.
  // Empty ranges/context and fallback request0 are written at lines143-161.
  Task t{};t[0]=token;t[1]=-1;t[6]=source*s+split;t[7]=source;return t;
}
std::vector<Task> workload(int64_t batch,int64_t q,int64_t s) {
  const auto n=4*batch, factor=3*s;
  std::vector<Task> tasks(n*factor);
  for(int64_t token=0;token<n;++token)for(int64_t source=0;source<3;++source)
    for(int64_t split=0;split<s;++split) {
      auto id=token*factor+source*s+split;
      const bool live=q==4?token%4==0:token<batch;
      tasks[id]=live?active_task(token,q,source,split,s):
          (q==1?native_negative_slot(token,source,split,s):poison_zero(0));
    }
  return tasks;
}

template<int Owner, bool Aic>
void compare(std::vector<Task>& tasks,Geometry geometry,int64_t core,int64_t cores,
             int64_t lane,int64_t& baseline_reads,int64_t& active_reads) {
  Kernel<Owner,Aic,false> baseline(tasks,geometry);
  Kernel<Owner,Aic,true> candidate(tasks,geometry);
  baseline.coreIndex=candidate.coreIndex=core;baseline.cores=candidate.cores=cores;
  baseline.lane=candidate.lane=lane;
  baseline.Run();candidate.Run();
  const auto& a=baseline.counters;const auto& b=candidate.counters;
  require(a.actions==b.actions,"observable dispatch/fields/status changed");
  require(a.dma_rows==b.dma_rows && a.fences==b.fences,"AIV DMA/fence changed");
  require(a.invalidations==b.invalidations && a.ends==b.ends,"AIC lifecycle changed");
  require(a.invalidations==(Aic?1:0),"cache invalidation missing");
  const auto& reads=Aic?b.gm_reads:b.ub_reads;
  std::vector<std::array<int64_t,16>> seen(tasks.size());
  for(const auto& read:reads)++seen.at(read[0]).at(read[1]);
  int64_t visited=0,zero=0;
  for(size_t id=0;id<seen.size();++id)if(seen[id][1]) {
    ++visited;
    require(seen[id][1]==1,"qcount was read twice");
    const bool empty=tasks[id][1]==0;zero+=empty;
    for(int j=0;j<11;++j)require(seen[id][j]==(empty?(j==1):1),"field read set changed");
    for(int j=11;j<16;++j)require(seen[id][j]==0,"unused descriptor tail read");
  }
  const auto old=Aic?a.gm_scalars:a.ub_scalars,newer=Aic?b.gm_scalars:b.ub_scalars;
  require(old==visited*11 && newer==old-zero*10,"scalar count mismatch");
  require(a.dma_rows==(Aic?0:visited),"full AIV task DMA missing");
  require(a.fences==a.dma_rows,"DMA completion fence missing");
  baseline_reads+=old;active_reads+=newer;
}

template<int Owner> void check_workload(int64_t batch,int64_t s) {
  auto tasks=workload(batch,Owner,s);Geometry g{4*batch,6,1,batch,512,s,0};
  int64_t old_aic=0,new_aic=0,old_aiv=0,new_aiv=0;
  for(int core=0;core<20;++core) {
    compare<Owner,true>(tasks,g,core,20,0,old_aic,new_aic);
    for(int lane=0;lane<2;++lane)compare<Owner,false>(tasks,g,core,20,lane,old_aiv,new_aiv);
  }
  const auto zeros=Owner==4?3*batch*3*s:0;
  require(old_aic-new_aic==zeros*10 && old_aiv-new_aiv==zeros*20,"workload saving differs");
  std::cout<<"{\"batch\":"<<batch<<",\"q\":"<<Owner<<",\"splits\":"<<s
           <<",\"zero_tasks\":"<<zeros<<",\"aic_reads_before\":"<<old_aic
           <<",\"aic_reads_after\":"<<new_aic<<",\"aiv_reads_before\":"<<old_aiv
           <<",\"aiv_reads_after\":"<<new_aiv<<"}\n";
}

template<int Owner,bool Aic> void check_adversarial() {
  auto tasks=workload(4,Owner,1);Geometry g{16,6,1,16,512,1,0};
  // Exercise every field used by TaskValid and both original error priorities.
  std::vector<Task> cases;
  for(int field=0;field<11;++field)if(field!=1) {
    auto t=active_task(4,Owner,0,0,1);t[field]=-1;cases.push_back(t);
  }
  auto negative=active_task(4,Owner,0,0,1);negative[1]=-1;cases.push_back(negative);
  auto too_many=active_task(4,Owner,0,0,1);too_many[1]=Owner+1;cases.push_back(too_many);
  auto error=negative;error[10]=7;cases.push_back(error);
  auto history=active_task(4,Owner,0,0,1);history[4]=1025;cases.push_back(history);
  auto current=active_task(4,Owner,2,0,1);current[3]=1023;cases.push_back(current);
  for(const auto& t:cases) {
    std::fill(tasks.begin(),tasks.end(),poison_zero(0));
    tasks[0]=t;tasks[2]=active_task(8,Owner,1,0,1); // live after poison zero.
    int64_t old=0,newer=0;compare<Owner,Aic>(tasks,g,0,1,0,old,newer);
  }
  // Invalid mode remains an error for every live task; padding still status0.
  g.dequantMode=3;int64_t old=0,newer=0;compare<Owner,Aic>(tasks,g,0,1,0,old,newer);
}

template<int Owner,bool Aic> void check_same_address_refresh() {
  auto tasks=workload(4,Owner,1);Geometry g{16,6,1,16,512,1,0};
  const auto* address=tasks.data();
  Kernel<Owner,Aic,false> baseline(tasks,g);Kernel<Owner,Aic,true> candidate(tasks,g);
  for(int step=0;step<5;++step) {
    for(size_t i=0;i<tasks.size();++i)tasks[i]=poison_zero(0);
    if(step!=2) {
      tasks[step]=active_task(step,1,step%3,0,1);
      tasks[step][10]=step==3?9:0;
    }
    baseline.Run();candidate.Run();
    require(tasks.data()==address,"fixture changed task addresses");
    require(baseline.counters.actions==candidate.counters.actions,"same-address refresh used stale fields");
    require(baseline.counters.dma_rows==candidate.counters.dma_rows,"refresh DMA changed");
    require(candidate.counters.invalidations==(Aic?step+1:0),"refresh invalidation changed");
  }
}

int main() {
  try {
    check_workload<4>(16,5);check_workload<4>(28,4);
    check_workload<1>(16,5);check_workload<1>(28,4);
    check_adversarial<4,true>();check_adversarial<4,false>();
    check_adversarial<1,true>();check_adversarial<1,false>();
    check_same_address_refresh<4,true>();check_same_address_refresh<4,false>();
    check_same_address_refresh<1,true>();check_same_address_refresh<1,false>();
    std::cout<<"{\"adversarial\":true,\"same_address_refresh\":true,\"host_control_only\":true}\n";
  } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
