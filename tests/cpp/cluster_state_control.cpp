// Archive #145/#150/#151/#154/#164/#172 and startup D.4: host control proof.
// Real VectorCluster/SpillState/RestoreState/qualification methods are injected.
// Bounded synchronous DMA and deterministic math stubs check data ownership,
// addresses, publication, errors and call traces, NOT CANN/NPU asynchronous timing.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#define __aicore__
constexpr int PIPE_V=1,PIPE_MTE3=2;
void require(bool condition,const char* why){if(!condition)throw std::runtime_error(why);}
float bit_float(uint32_t value){float result;std::memcpy(&result,&value,4);return result;}
uint32_t float_bit(float value){uint32_t result;std::memcpy(&result,&value,4);return result;}
enum class HardEvent {V_S,V_MTE3,MTE3_MTE2,MTE3_V,MTE2_V};
struct Counts {
  int64_t bytes=0,calls=0,bursts=0,initialization=0;
  std::array<int64_t,5> fences{};
  std::vector<std::array<int64_t,2>> flags;
};
thread_local Counts* counters=nullptr;
template<HardEvent event> void Fence(){++counters->fences[static_cast<size_t>(event)];}
template<int> void PipeBarrier(){}
bool Finite(float value){return std::isfinite(value);}
struct Memory {
  static constexpr int guard=8;
  std::vector<float> cells;
  std::vector<uint8_t> initialized;
  size_t count;
  bool gm;
  Memory(size_t length,bool global=false):cells(length+2*guard,bit_float(0x7fc0babe)),
      initialized(length,0),count(length),gm(global){}
  float read(size_t index)const{
    require(index<count,"memory read out of bounds");
    require(initialized[index],"read of uninitialized online state");
    return cells[guard+index];
  }
  void write(size_t index,float value){
    require(index<count,"memory write out of bounds");
    cells[guard+index]=value;initialized[index]=1;
  }
  void check()const{
    for(int i=0;i<guard;++i)require(float_bit(cells[i])==0x7fc0babe &&
        float_bit(cells[guard+count+i])==0x7fc0babe,"memory guard modified");
  }
};
struct View {
  Memory* memory;size_t offset,capacity;
  View operator[](size_t index)const{
    require(index<capacity,"view offset out of bounds");return {memory,offset+index,capacity-index};
  }
  float GetValue(size_t index)const{
    require(index<capacity,"scalar read outside tensor");return memory->read(offset+index);
  }
  void set(size_t index,float value)const{
    require(index<capacity,"tensor write out of bounds");memory->write(offset+index,value);
  }
};
struct Buffer {
  Memory memory;
  explicit Buffer(size_t count):memory(count){}
  template<typename T> View Get(){static_assert(std::is_same_v<T,float>);return {&memory,0,memory.count};}
};
struct Global {
  Memory memory;
  explicit Global(size_t count):memory(count,true){}
  View operator[](size_t offset){require(offset<memory.count,"GM offset out of bounds");return {&memory,offset,memory.count-offset};}
};
struct DataCopyParams {uint16_t blockCount,blockLen,srcStride,dstStride;};
void burst(View dst,View src,size_t count){
  require(count<=dst.capacity && count<=src.capacity,"DMA burst outside tensor");
  for(size_t i=0;i<count;++i)dst.set(i,src.GetValue(i));
  ++counters->bursts;counters->bytes+=count*4;
}
void DataCopy(View dst,View src,int count){
  require(dst.memory->gm!=src.memory->gm,"state DMA must cross UB/GM");
  ++counters->calls;burst(dst,src,count);
}
void DataCopy(View dst,View src,DataCopyParams params){
  require(dst.memory->gm!=src.memory->gm,"state DMA must cross UB/GM");
  require(params.blockCount==2 && params.blockLen==8,"max/sum DMA shape changed");
  ++counters->calls;
  for(int block=0;block<params.blockCount;++block)
    burst(dst[block*(params.blockLen+params.dstStride)*8],
          src[block*(params.blockLen+params.srcStride)*8],params.blockLen*8);
}
void Duplicate(View dst,float value,int count){
  for(int i=0;i<count;++i)dst.set(i,value);
  ++counters->initialization;
}
void Muls(View dst,View src,float scale,int count){
  for(int i=0;i<count;++i)dst.set(i,src.GetValue(i)*scale);
}
struct TaskState {
  int64_t id=0,qbegin=0,qcount=0,kvhead=0,kvbegin=0,kvend=0,request=0,split=0,
      kind=0,context=0,requestBegin=0;
  int32_t taskError=0;
};
struct Geometry {int64_t tokens=0,hq=6,hk=1,requests=2,splits=1;};

template<int Members,bool Active> struct Kernel {
  static constexpr int D=256,kQueryRows=128,kLaneRows=64,kHalfRows=32,kRowBlocks=2,kKvRows=256;
  static constexpr int kVectorReady=8,kCubeReady=9,kClusterKvReady=8,kClusterScore0=9,
      kClusterP0=Members==4?11:0,kClusterRoundDone=13;
  static constexpr float kEmptyMax=-__FLT_MAX__;
  Counts counts;
  Buffer accBuf{kLaneRows*D},statsBuf{2*kLaneRows+2*kHalfRows};
  static constexpr int64_t accStateOffset=64,statStateOffset=accStateOffset+Members*kQueryRows*D;
  Global work{statStateOffset+Members*2*kQueryRows+64};
  Geometry g;
  int64_t lane,qOffset=0,qbegin=0,qcount=0,kvhead=0,kvbegin=0,kvend=0,request=0,
      split=0,kind=0,context=0,requestBegin=0;
  int32_t taskError=0,error=0;
  int tiles,round=0,scenario;
  std::array<TaskState,Members> members;
  std::vector<TaskState> descriptors;
  std::array<bool,Members> ready{};
  std::vector<std::vector<uint32_t>> state_publications,outputs;
  std::vector<std::array<int64_t,3>> operations;
  Kernel(int t,int l,int group,int fault):lane(l),tiles(t),scenario(fault){
    g.hq=group;g.tokens=5+Members*(kQueryRows/group)+3;
    descriptors.resize(g.tokens*3);
    for(int i=0;i<Members;++i){
      auto& member=members[i];member.qbegin=5+i*(kQueryRows/group);
      member.id=member.qbegin*3;member.qcount=kQueryRows/group;
      member.kvbegin=64;member.kvend=t?64+t*kKvRows-17:64;
      member.context=member.kvend;member.request=1;member.requestBegin=5;
      descriptors[member.id]=member;
    }
  }
  int member()const{return static_cast<int>(qOffset/(kQueryRows*D));}
  void LoadTask(int64_t id){UseTask(descriptors.at(id));}
@@CURRENT_TASK@@
@@USE_TASK@@
@@TASK_VALID@@
@@QUALIFY4@@
@@QUALIFY16@@
  bool Qualifies(){UseTask(members[0]);return Members==4?
      IsClusterMember4(members[0].id,nullptr):IsClusterMember16(members[0].id,nullptr);}
  void SelectClusterSlots(int buffer,int slot){operations.push_back({0,buffer,slot});}
  void LoadQueries(){
    operations.push_back({1,member(),lane});
    if(scenario && member()%3==1)error=2;
  }
  void LoadClusterKv(int64_t start){
    const int tile=static_cast<int>((start-members[0].kvbegin)/kKvRows);
    operations.push_back({2,tile,lane});
    if(scenario && tile%3==1)error=(lane?4:3);
  }
  void Softmax(int64_t start,int rowBase){
    const int tile=static_cast<int>((start-members[0].kvbegin)/kKvRows);
    operations.push_back({3,member(),rowBase});
    auto acc=accBuf.Get<float>(),stats=statsBuf.Get<float>();
    for(int r=0;r<kHalfRows;++r){
      const int local=rowBase-lane*kLaneRows+r;
      if(rowBase+r>=qcount*g.hq/g.hk)continue;
      const float alpha=(member()+tile+local)%3==0?1.0F:0.5F;
      stats.set(local,std::max(stats.GetValue(local),float(member()+tile+local)));
      stats.set(kLaneRows+local,stats.GetValue(kLaneRows+local)*alpha+0.25F);
      Muls(acc[local*D],acc[local*D],alpha,D);
    }
    if(scenario && (tile+member()+lane)%7==3)error=2;
    if(scenario==2 && member()==Members-1 && tile==tiles-1){
      // Include NaN, infinity and signed-zero state in final publication/error
      // handling; the real math method is byte-identical outside this harness.
      stats.set(kLaneRows,bit_float(0x7fc00123));
      acc.set(0,bit_float(0x80000000));
    }
  }
  std::vector<uint32_t> ub_snapshot(int id){
    std::vector<uint32_t> snapshot{uint32_t(id),uint32_t(error)};
    for(int i=0;i<kLaneRows*D;++i)snapshot.push_back(float_bit(accBuf.Get<float>().GetValue(i)));
    for(int i=0;i<2*kLaneRows;++i)snapshot.push_back(float_bit(statsBuf.Get<float>().GetValue(i)));
    return snapshot;
  }
  void PublishRows(int64_t id){outputs.push_back(ub_snapshot(static_cast<int>(id)));}
  template<int, int pipe> void CrossCoreSetFlag(int flag){
    counts.flags.push_back({1,flag});
    if(flag>=kClusterP0 && flag<kClusterP0+(Members==4?2:4)){
      require(pipe==PIPE_MTE3,"PReady producer pipe changed");
      require(!ready[member()],"member publishes P twice in one tile");
      ready[member()]=true;
      auto snapshot=ub_snapshot(member());
      const int64_t accBase=accStateOffset+member()*kQueryRows*D+lane*kLaneRows*D;
      const int64_t statBase=statStateOffset+member()*2*kQueryRows+lane*kLaneRows;
      for(int i=0;i<kLaneRows*D;++i)require(float_bit(work.memory.read(accBase+i))==snapshot[2+i],
          "Cube observes state before alpha publication");
      for(int i=0;i<kLaneRows;++i){
        require(float_bit(work.memory.read(statBase+i))==snapshot[2+kLaneRows*D+i],"max state publication differs");
        require(float_bit(work.memory.read(statBase+kQueryRows+i))==snapshot[2+kLaneRows*D+kLaneRows+i],"sum state publication differs");
      }
      state_publications.push_back(std::move(snapshot));
    }
  }
  void CrossCoreWaitFlag(int flag){
    counts.flags.push_back({0,flag});
    if(flag==kClusterRoundDone){
      for(int m=0;m<Members;++m){
        require(ready[m],"Cube round completed before P/state was published");ready[m]=false;
        const int64_t base=accStateOffset+m*kQueryRows*D+lane*kLaneRows*D;
        const int active=std::max(0,std::min(kLaneRows,int(members[m].qcount*g.hq/g.hk)-int(lane)*kLaneRows));
        for(int row=0;row<active;++row)for(int column=0;column<D;++column){
          const int64_t at=base+row*D+column;
          const float pv=float((m+row+column+round)%19-9)*0.03125F;
          work.memory.write(at,work.memory.read(at)+pv);
        }
      }
      ++round;
    }
  }
  void SpillState(int32_t member){if constexpr(Active)SpillStateActive(member);else SpillStateBaseline(member);}
  void RestoreState(int32_t member){if constexpr(Active)RestoreStateActive(member);else RestoreStateBaseline(member);}
@@BASELINE_SPILL@@
@@ACTIVE_SPILL@@
@@BASELINE_RESTORE@@
@@ACTIVE_RESTORE@@
@@BASELINE_CLUSTER4@@
@@ACTIVE_CLUSTER4@@
@@BASELINE_CLUSTER16@@
@@ACTIVE_CLUSTER16@@
  void Run(){
    require(Qualifies(),"invalid cluster was executed");counters=&counts;
    if constexpr(Members==4){if constexpr(Active)VectorCluster4Active(members.data());else VectorCluster4Baseline(members.data());}
    else {if constexpr(Active)VectorCluster16Active(members.data());else VectorCluster16Baseline(members.data());}
    counters=nullptr;
    require(round==tiles,"missing Cube completion");
    accBuf.memory.check();statsBuf.memory.check();work.memory.check();
    for(int m=0;m<Members;++m){
      const int64_t other=accStateOffset+m*kQueryRows*D+(1-lane)*kLaneRows*D;
      for(int i=0;i<kLaneRows*D;++i)require(!work.memory.initialized[other+i],"state DMA crossed into other AIV accumulator");
      const int64_t otherStats=statStateOffset+m*2*kQueryRows+(1-lane)*kLaneRows;
      for(int i=0;i<kLaneRows;++i)require(!work.memory.initialized[otherStats+i] &&
          !work.memory.initialized[otherStats+kQueryRows+i],"stats DMA crossed into other AIV lane");
    }
  }
};

int cases=0;
template<int Members> void compare(int tiles,int lane,int group,int scenario){
  Kernel<Members,false> old(tiles,lane,group,scenario);
  Kernel<Members,true> active(tiles,lane,group,scenario);
  old.Run();active.Run();
  require(old.operations==active.operations,"query/KV/Softmax/control ordering changed");
  require(old.counts.flags==active.counts.flags,"cross-core token sequence changed");
  require(old.state_publications==active.state_publications,"pre-atomic state or error bits changed");
  require(old.outputs==active.outputs,"normalized published state/error bits changed");
  require(old.counts.initialization==active.counts.initialization,"initialization arithmetic changed");
  const int64_t stateBytes=(64*256+2*64)*4;
  require(old.counts.bytes==2*Members*(tiles+1)*stateBytes,"baseline state-byte ledger differs");
  require(active.counts.bytes==2*Members*tiles*stateBytes,"candidate state-byte ledger differs");
  require(old.counts.calls==6*Members*(tiles+1),"baseline DMA call count differs");
  require(active.counts.calls==4*Members*tiles,"candidate DMA call count differs");
  require(old.counts.bursts-active.counts.bursts==6*Members,"strided merge changed transferred blocks");
  require(old.counts.fences[0]==active.counts.fences[0],"V_S initialization/normalization order changed");
  for(int event=1;event<5;++event)require(old.counts.fences[event]-active.counts.fences[event]==Members,
      "live transfer completion fence changed");
  ++cases;
}
template<int Members> void qualification(){
  Kernel<Members,true> empty(0,0,6,0);require(!empty.Qualifies(),"empty range accepted");
  Kernel<Members,true> valid(1,0,6,0);require(valid.Qualifies(),"valid tail range rejected");
  auto& bad=valid.descriptors[valid.members[Members-1].id];
  ++bad.kvbegin;require(!valid.Qualifies(),"mixed member range accepted");
  --bad.kvbegin;bad.taskError=3;require(!valid.Qualifies(),"member task error accepted");
  bad.taskError=0;bad.kind=1;require(!valid.Qualifies(),"nonhistory member accepted");
  ++cases;
}
int main(){
  try{
    qualification<4>();qualification<16>();
    for(int tiles:{1,2,3,129})for(int lane:{0,1})for(int group:{1,6,8})for(int fault:{0,1,2}){
      // Long lifetime coverage needs both families/lanes, not every short-case
      // error permutation again; those are fully exercised for T1/T2/T3.
      if(tiles==129 && (group!=6 || fault!=0))continue;
      compare<4>(tiles,lane,group,fault);compare<16>(tiles,lane,group,fault);
    }
    std::cout<<"{\"host_control_only\":true,\"cases\":"<<cases
      <<",\"uninitialized_GM_reads\":0,\"state_and_error_bits\":\"identical\",\"cross_flags\":\"identical\","
      <<"\"guards\":\"passed\",\"CANN\":\"not_run\",\"NPU\":\"not_run\",\"performance\":\"not_run\"}\n";
    return 0;
  }catch(const std::exception& error){std::cerr<<"CLUSTER_STATE_CONTROL_FAILED cases="<<cases<<" "<<error.what()<<"\n";return 1;}
}
