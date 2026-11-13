// Archive #145/#150/#151/#154/#164/#172; complete startup D.4 reviewed.
// Actual CubeCluster/VectorCluster/SelectClusterSlots methods are injected.
// They generate the three actors' actions; every legal interleaving is checked
// for short rounds, randomized/adversarial schedules for long rounds. DMA/math
// completion is abstracted at the existing publication fences, NOT simulated
// as NPU timing. No CANN/precision/performance claim follows from this model.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#define __aicore__
constexpr int PIPE_MTE3=2,PIPE_FIX=3,PIPE_V=4;
enum class HardEvent {V_S};
template<HardEvent> void Fence(){}
template<int> void PipeBarrier(){}
void require(bool pass,const char* why){if(!pass)throw std::runtime_error(why);}
struct FakeView {
  FakeView operator[](int)const{return {};}
  float GetValue(int)const{return 1.0F;}
};
struct FakeBuffer {template<class T> FakeView Get(){return {};}};
void Duplicate(FakeView,float,int){}
void Muls(FakeView,FakeView,float,int){}
bool Finite(float x){return std::isfinite(x);}
struct TaskState {
  int64_t id=0,qbegin=0,qcount=21,kvhead=0,kvbegin=0,kvend=0,request=0,
      split=0,kind=0,context=0,requestBegin=0;
  int32_t taskError=0;
};
enum Kind {Wait,Set,Write,Read,State,PV,Publish,KV};
struct Action {
  Kind kind;int flag=-1,token=-1,block=-1;int64_t address=0,bytes=0;int error=0;
};
struct Geometry {int hq=6,hk=1;};

template<bool Ring> struct TraceKernel {
  static constexpr int D=256,kQueryRows=128,kKvRows=256,kLaneRows=64,kHalfRows=32,kRowBlocks=2;
  static constexpr int kVectorReady=8,kCubeReady=9,kClusterKvReady=8,kClusterScore0=9,
      kClusterP0=0,kClusterRoundDone=13;
  static constexpr float kEmptyMax=-__FLT_MAX__;
  // Exact word offsets from the production Init; unchanged allocated arena.
  static constexpr int64_t baseKOffset=16*128*256,baseScoreOffset=baseKOffset+2*256*256;
  static constexpr int64_t accStateOffset=baseScoreOffset+128*256+128*256;
  static constexpr int64_t statStateOffset=accStateOffset+16*128*256;
  static constexpr int64_t extraKvOffset=statStateOffset+16*2*128;
  static constexpr int64_t extraScoreOffset=extraKvOffset+2*256*256;
  static constexpr int64_t arena=extraScoreOffset+15*128*256;
  Geometry g;FakeBuffer accBuf,statsBuf;
  int lane,round=-1,error=0,scenario;
  int64_t qOffset=0,kOffset=0,vOffset=0,scoreOffset=0,qbegin=0,qcount=21,
      kvbegin=0,kvend=0,requestBegin=0,context=0,kind=0;
  std::array<TaskState,16> members;
  std::vector<Action> actions;
  TraceKernel(int tiles,int actor,int fault):lane(actor-1),scenario(fault){
    for(int i=0;i<16;++i){members[i].id=i;members[i].qbegin=5+i*21;
      members[i].requestBegin=5;members[i].kvbegin=64;
      members[i].kvend=64+tiles*256-17;members[i].context=members[i].kvend;}
  }
  int member()const{return int(qOffset/(kQueryRows*D));}
  void UseTask(const TaskState& t){qbegin=t.qbegin;qcount=t.qcount;kvbegin=t.kvbegin;
    kvend=t.kvend;context=t.context;requestBegin=t.requestBegin;kind=t.kind;}
@@BASE_SELECT@@
@@RING_SELECT@@
  void SelectClusterSlots(int32_t buffer,int32_t slot){
    if constexpr(Ring)SelectRing(buffer,slot);else SelectBase(buffer,slot);
  }
@@PROBABILITY@@
  void LoadQueries(){if(scenario && member()%3==1)error=2;}
  void LoadClusterKv(int64_t start){round=int((start-members[0].kvbegin)/256);
    if(scenario && round%3==1)error=lane?4:3;
    actions.push_back({KV,-1,round,-1,kOffset,0,error});}
  void Softmax(int64_t start,int rowBase){
    const int tile=int((start-members[0].kvbegin)/256);
    actions.push_back({Read,-1,tile*16+member(),rowBase/32,
        scoreOffset+rowBase*kKvRows,kHalfRows*kKvRows*4,error});
    if(scenario && (tile+member()+lane)%7==3)error=2;
  }
  void SpillState(int32_t m){actions.push_back({State,-1,round*16+m,-1,0,0,error});}
  void RestoreState(int32_t){}
  void PublishRows(int64_t id){actions.push_back({Publish,-1,int(id),-1,0,0,error});}
  void LoadResidentB(int64_t b){if(b==kOffset)++round;}
  void MatmulResidentB(int64_t a,int64_t c,int32_t m,int32_t n,int32_t,
      bool=true,uint8_t atomic=0){
    if(atomic){const int member=int((a-kQueryRows*D/2)/(kQueryRows*D));
      actions.push_back({PV,-1,round*16+member,-1,c,int64_t(m)*n*4});}
    else {const int member=int(a/(kQueryRows*D));
      actions.push_back({Write,-1,round*16+member,-1,c,int64_t(m)*n*4});}
  }
  void CrossCoreWaitFlag(int flag){actions.push_back({Wait,flag});}
  template<int,int> void CrossCoreSetFlag(int flag){actions.push_back({Set,flag});}
@@CUBE_CLUSTER@@
@@VECTOR_CLUSTER@@
  void Run(int actor){if(actor==0)CubeCluster(members.data());else VectorCluster(members.data());}
};

using PC=std::array<int,3>;
using Flags=std::array<int,16>;
// Every score DMA is a whole number of D=256-element rows. A row's 256
// synthetic words deliberately share one tag, so a row-address shadow is an
// exact compression of that payload, not a token-derived read expectation.
using Shadow=std::vector<uint64_t>;
uint64_t payload(int epoch,int token,int row){
  return uint64_t(epoch)*1000000007+uint64_t(token)*1009+uint64_t(row);
}
struct Model {
  std::array<std::vector<Action>,3> trace;
  std::array<std::vector<Flags>,3> sets,waits;
  std::vector<int> written,nextWrite;
  std::array<std::vector<int>,2> finalRead,spilled;
  int slots,tiles;bool oneLane=false,staleWrites=false;
  std::set<int64_t> touched;
  int64_t scoreBytes=0;
  template<bool Ring> static Model create(int t,int scenario){
    Model model;model.slots=Ring?4:16;model.tiles=t;
    for(int a=0;a<3;++a){TraceKernel<Ring> kernel(t,a,scenario);kernel.Run(a);
      model.trace[a]=std::move(kernel.actions);}
    model.index();return model;
  }
  void index(){
    written.assign(tiles*16,-1);nextWrite.assign(tiles*16,-1);
    for(int l=0;l<2;++l){finalRead[l].assign(tiles*16,-1);spilled[l].assign(tiles*16,-1);}
    std::array<int,16> previous;previous.fill(-1);
    for(int a=0;a<3;++a){
      sets[a].resize(trace[a].size()+1);waits[a].resize(trace[a].size()+1);
      for(size_t p=0;p<trace[a].size();++p){const auto& e=trace[a][p];
        sets[a][p+1]=sets[a][p];waits[a][p+1]=waits[a][p];
        if(e.kind==Set)++sets[a][p+1][e.flag];
        if(e.kind==Wait)++waits[a][p+1][e.flag];
        if(e.kind==Write){
          require(a==0 && written.at(e.token)<0,"duplicate score producer");written[e.token]=int(p);
          const int slot=e.token%16%slots;
          if(previous[slot]>=0)nextWrite[previous[slot]]=int(p);
          previous[slot]=e.token;
        }
        if(e.kind==Read){require(a>0,"Cube reads Vector score block");finalRead[a-1].at(e.token)=int(p);}
        if(e.kind==State){require(a>0,"Cube writes Vector alpha state");spilled[a-1].at(e.token)=int(p);}
        if(e.kind==Read || e.kind==Write){
          scoreBytes+=e.bytes;
          require(e.address>=0 && (e.address*4+e.bytes)<=TraceKernel<true>::arena*4,
              "score access outside original arena");
          for(int64_t address=e.address*4;address<e.address*4+e.bytes;address+=32)touched.insert(address);
        }
      }
    }
    require(scoreBytes==int64_t(tiles)*16*(126+128)*256*4,"score DMA workload changed");
    require(touched.size()*32==size_t(slots)*128*256*4,"score address set does not match slot count");
  }
  bool done(const PC& p)const{
    for(int a=0;a<3;++a)if(p[a]!=int(trace[a].size()))return false;return true;
  }
  bool ready(const PC& p,int a)const{
    if(p[a]==int(trace[a].size()))return false;
    const auto& e=trace[a][p[a]];if(e.kind!=Wait)return true;
    if(a==0){const int consumed=waits[0][p[0]][e.flag];
      const bool first=sets[1][p[1]][e.flag]>consumed,second=sets[2][p[2]][e.flag]>consumed;
      return oneLane?(first||second):(first&&second);}
    return sets[0][p[0]][e.flag]>waits[a][p[a]][e.flag];
  }
  void step(PC& p,int a)const{
    require(ready(p,a),"scheduled a blocked actor");const auto& e=trace[a][p[a]];
    if(e.kind==Set){
      if(a==0){for(int l=1;l<3;++l)require(sets[0][p[0]][e.flag]==waits[l][p[l]][e.flag],
          "Cube flag overwritten before both AIVs consumed it");}
      else require(sets[a][p[a]][e.flag]==waits[0][p[0]][e.flag],
          "AIV flag overwritten before Cube consumed it");
    }
    if(e.kind==Write){
      const int predecessor=e.token-slots;
      // Ring numbering is 0..15 each round, so this is the last same-slot write.
      if(predecessor>=0)for(int l=0;l<2;++l)
        require(p[l+1]>finalRead[l][predecessor],"score overwritten before both AIVs consumed it");
    }
    if(e.kind==Read){
      require(p[0]>written[e.token],"read before Cube score publication");
      require(nextWrite[e.token]<0 || p[0]<=nextWrite[e.token],"read observed a newer score version");
    }
    if(e.kind==PV)for(int l=0;l<2;++l)
      require(p[l+1]>spilled[l][e.token],"PV before both P/alpha state publications");
    ++p[a];
  }
  void drained(const PC& p)const{
    require(done(p),"incomplete cluster round drain");
    for(int flag=0;flag<16;++flag)for(int l=1;l<3;++l){
      require(sets[0][p[0]][flag]==waits[l][p[l]][flag],"undrained Cube token");
      require(sets[l][p[l]][flag]==waits[0][p[0]][flag],"undrained Vector token");}
  }
  uint64_t exhaustive()const{
    std::vector<PC> pending{{0,0,0}};std::unordered_set<uint64_t> seen;
    while(!pending.empty()){
      PC p=pending.back();pending.pop_back();
      const uint64_t key=uint64_t(p[0])|(uint64_t(p[1])<<21)|(uint64_t(p[2])<<42);
      if(!seen.insert(key).second)continue;
      if(done(p)){drained(p);continue;}
      int available=0;
      for(int a=0;a<3;++a)if(ready(p,a)){++available;PC next=p;step(next,a);pending.push_back(next);}
      require(available>0,"deadlocked cluster action graph");
    }
    return seen.size();
  }
  void randomized(uint64_t seed,int mode,int epoch,Shadow& memory)const{
    PC p{};std::mt19937_64 random(seed);std::array<uint64_t,2> digest{};
    require(memory.size()==size_t(TraceKernel<true>::arena/256),"shadow arena shape drift");
    // The caller retains this exact memory across invocation epochs. Cube
    // writes through the extracted GM address; Vector reads that same address
    // and checks epoch/member/row ownership. Inactive rows remain poison/old
    // values, as in the real score DMA, and are excluded by the unchanged
    // Softmax padding path; they are never counted as valid query rows here.
    while(!done(p)){
      std::vector<int> possible;for(int a=0;a<3;++a)if(ready(p,a))possible.push_back(a);
      require(!possible.empty(),"random schedule deadlock");
      int a=possible[random()%possible.size()];
      if(mode>=0){for(int candidate:{mode,(mode+1)%3,(mode+2)%3})
        if(std::find(possible.begin(),possible.end(),candidate)!=possible.end()){a=candidate;break;}}
      const auto& e=trace[a][p[a]];
      step(p,a);
      if(e.kind==Write){
        require(e.address%256==0 && e.bytes==126*256*4,"Cube score payload shape changed");
        if(!staleWrites)for(int row=0;row<126;++row)
          memory.at(e.address/256+row)=payload(epoch,e.token,row);
      }
      if(e.kind==Read){
        require(e.address%256==0 && e.bytes==32*256*4,"Vector score DMA shape changed");
        for(int row=0;row<32;++row){
          const uint64_t observed=memory.at(e.address/256+row);
          const int logical=e.block*32+row;
          if(logical>=126)continue;
          require(observed==payload(epoch,e.token,logical),"score shadow address read wrong epoch/member/row");
          digest[a-1]+=observed;
        }
      }
    }
    drained(p);
    for(int l=0;l<2;++l){uint64_t expected=0;
      for(int token=0;token<tiles*16;++token)for(int row=l*64;row<std::min(126,l*64+64);++row)
        expected+=payload(epoch,token,row);
      require(digest[l]==expected,"same-address changed-input symbolic payload mismatch");}
  }
};

void same_trace(const Model& base,const Model& ring){
  require(base.scoreBytes==ring.scoreBytes,"score transfer byte count changed");
  for(int a=0;a<3;++a){require(base.trace[a].size()==ring.trace[a].size(),"actor event count changed");
    for(size_t p=0;p<base.trace[a].size();++p){const auto& x=base.trace[a][p];const auto& y=ring.trace[a][p];
      require(x.kind==y.kind && x.flag==y.flag && x.token==y.token && x.block==y.block &&
          x.bytes==y.bytes && x.error==y.error,"non-address protocol/error sequence changed");
      if(x.kind!=Read && x.kind!=Write)require(x.address==y.address,"non-score address changed");}
  }
}
int main(){try{
  uint64_t states=0;int schedules=0;
  for(int tiles:{1,2,3,129})for(int scenario:{0,1}){
    Model base=Model::create<false>(tiles,scenario),ring=Model::create<true>(tiles,scenario);
    same_trace(base,ring);
    if(tiles<=3 && scenario==0){states+=base.exhaustive();states+=ring.exhaustive();}
    for(int seed=0;seed<32;++seed){
      const int mode=seed<3?seed:-1;
      Shadow baseMemory(TraceKernel<true>::arena/256,UINT64_MAX),ringMemory=baseMemory;
      for(int epoch:{1,2}){
        base.randomized(seed,mode,epoch,baseMemory);ring.randomized(seed,mode,epoch,ringMemory);schedules+=2;
      }
    }
  }
  // A negative control must reject the tempting but invalid one-AIV wait.
  Model unsafe=Model::create<true>(1,0);unsafe.oneLane=true;
  bool rejected=false;try{unsafe.exhaustive();}catch(const std::runtime_error&){rejected=true;}
  require(rejected,"one-AIV acknowledgement mutation escaped the ownership model");
  Model stale=Model::create<true>(1,0);
  Shadow retained(TraceKernel<true>::arena/256,UINT64_MAX);
  stale.randomized(7,-1,1,retained);stale.staleWrites=true;
  bool staleRejected=false;
  try{stale.randomized(7,-1,2,retained);}catch(const std::runtime_error& e){
    staleRejected=std::string(e.what())=="score shadow address read wrong epoch/member/row";
  }
  require(staleRejected,"stale previous-invocation payload escaped the address shadow");
  std::cout<<"{\"host_control_only\":true,\"exhaustive_states\":"<<states
      <<",\"random_or_adversarial_schedules\":"<<schedules
      <<",\"double_AIV_required\":true,\"changed_input_epochs\":2,"
      <<"\"persistent_address_shadow\":true,\"stale_payload_rejected\":true,"
      <<"\"old_score_address_bytes\":2097152,\"new_score_address_bytes\":524288,"
      <<"\"score_DMA_bytes_per_tile\":4161536,\"CANN\":\"not_run\","
      <<"\"NPU\":\"not_run\",\"performance\":\"not_run\"}\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
