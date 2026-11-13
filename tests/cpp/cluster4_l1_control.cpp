// Archive #145/#150/#151/#154/#164/#172, full D.4: bounded host protocol test.
// Inject real Cube/Vector/transport methods. Three host actors and explicit
// BF16 ND/NZ buffers check ownership and matrix operand bits. Vector/GEMM math
// stubs are deterministic controls, NOT CANN, NPU, graph or performance evidence.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>
#define __aicore__
using bfloat16_t=uint16_t;
constexpr int PIPE_FIX=1,PIPE_MTE3=2,PIPE_V=3;
constexpr int D=256,kQueryRows=128,kLaneRows=64,kHalfRows=32,kRowBlocks=2,kKvRows=256;
constexpr int kVectorReady=8,kCubeReady=9,kClusterKvReady=8,kClusterScore0=9,kClusterP0=11,kClusterRoundDone=13;
constexpr float kEmptyMax=-__FLT_MAX__;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
bool Finite(float x){return std::isfinite(x);}
uint32_t bits(float x){uint32_t b;std::memcpy(&b,&x,4);return b;}
enum class HardEvent{V_S,V_MTE3,MTE3_MTE2,MTE3_V,MTE2_V,MTE1_MTE2,MTE2_MTE1};
struct Shared;
thread_local Shared* shared=nullptr;
thread_local int actor=-1;
thread_local bool hf32=true;
struct MatrixCall{
  int m,n,k,orgM,orgN,orgK;bool transpose,atomic,hf32;
  std::vector<uint16_t>a,b;
  bool operator==(const MatrixCall& other)const{
    return std::tie(m,n,k,orgM,orgN,orgK,transpose,atomic,hf32,a,b)==
      std::tie(other.m,other.n,other.k,other.orgM,other.orgN,other.orgK,
               other.transpose,other.atomic,other.hf32,other.a,other.b);
  }
};
struct Score{int tile=-1,member=-1;std::array<bool,2> consumed{true,true};};
struct Shared{
  static constexpr int qSlot=kQueryRows*D;
  static constexpr int baseK=4*qSlot,baseV=baseK+kKvRows*D;
  static constexpr int baseScore=baseV+kKvRows*D,pv=baseScore+kQueryRows*kKvRows;
  static constexpr int acc=pv+kQueryRows*D,stats=acc+4*qSlot;
  static constexpr int extraKv=stats+4*2*kQueryRows+64;
  static constexpr int extraScore=extraKv+2*kKvRows*D;
  static constexpr int extent=extraScore+kQueryRows*kKvRows+64;
  std::mutex mutex;std::condition_variable changed;
  bool stop=false,pendingMte1=false,pendingMte2=false,l1Ready=false;
  std::exception_ptr failure;
  std::array<std::array<int,16>,2> toVector{},toCube{};
  std::array<Score,2> score;
  std::array<std::array<std::array<int,2>,4>,2> pReady{};
  std::array<std::array<int,2>,2> kvTile{};
  std::vector<uint16_t> operands=std::vector<uint16_t>(2*extent,0x7fc1);
  std::vector<float> work=std::vector<float>(extent,0);
  std::vector<bool> initialized=std::vector<bool>(extent,false);
  std::map<std::tuple<int,int,int>,MatrixCall> calls;
  std::array<std::vector<uint32_t>,2> published;
  std::array<std::vector<std::array<int,2>>,3> flags;
  int generation=0,tiles=0,group=6,query=21,kind=0,currentAicTile=0;
  bool cluster=true;
  int bLoads=0,ends=0,atomicCalls=0,aicC1Waits=0;
  Shared(){reset(0,1,6,21,true,0);}
  void reset(int gen,int count,int g,int q,bool clustered,int source){
    generation=gen;tiles=count;group=g;query=q;cluster=clustered;kind=source;
    stop=false;failure=nullptr;pendingMte1=pendingMte2=l1Ready=false;
    toVector={};toCube={};score={};pReady={};kvTile={};
    for(auto& b:kvTile)b={-1,-1};
    for(auto& b:pReady)for(auto& m:b)m={-1,-1};
    calls.clear();for(auto& p:published)p.clear();for(auto& f:flags)f.clear();
    std::fill(initialized.begin(),initialized.end(),false);
    bLoads=ends=atomicCalls=currentAicTile=aicC1Waits=0;
  }
  void fail(){std::lock_guard lock(mutex);if(!failure)failure=std::current_exception();stop=true;changed.notify_all();}
  void set(int flag){
    std::lock_guard lock(mutex);flags[actor].push_back({1,flag});
    if(actor==0){for(auto& lane:toVector)++lane[flag];if(cluster&&flag==kClusterRoundDone)++currentAicTile;}
    else ++toCube[actor-1][flag];
    changed.notify_all();
  }
  void wait(int flag){
    std::unique_lock lock(mutex);flags[actor].push_back({0,flag});
    const bool ready=changed.wait_for(lock,std::chrono::seconds(3),[&]{
      return stop||(actor==0?(toCube[0][flag]>0&&toCube[1][flag]>0):toVector[actor-1][flag]>0);
    });
    require(ready&&!stop,"three-actor protocol deadlocked or peer failed");
    if(actor==0){
      --toCube[0][flag];--toCube[1][flag];
      if(!cluster&&flag==kVectorReady&&++aicC1Waits%3==0)++currentAicTile;
    }
    else --toVector[actor-1][flag];
  }
};
template<HardEvent E> void Fence(){
  if(actor!=0)return;
  if constexpr(E==HardEvent::MTE1_MTE2)shared->pendingMte1=false;
  if constexpr(E==HardEvent::MTE2_MTE1){require(shared->pendingMte2,"L1 consume fence before DMA");shared->pendingMte2=false;shared->l1Ready=true;}
}
template<int> void PipeBarrier(){}
void SetHF32Mode(bool enabled){hf32=enabled;}
template<int,int> void CrossCoreSetFlag(int flag){shared->set(flag);}
void CrossCoreWaitFlag(int flag){shared->wait(flag);}
template<typename T> struct Local{
  std::vector<T>* data;size_t offset=0;
  Local operator[](size_t at)const{require(offset+at<data->size(),"UB offset overflow");return {data,offset+at};}
  T GetValue(size_t at)const{require(offset+at<data->size(),"UB scalar overflow");return data->at(offset+at);}
};
template<typename T> struct Buffer{
  std::vector<T>data;
  explicit Buffer(size_t count):data(count){}
  template<typename U> Local<U>Get(){static_assert(std::is_same_v<T,U>);return {&data,0};}
};
template<typename T> struct Gm{int64_t offset;};
template<typename T> struct Global{Gm<T>operator[](int64_t at){return {at};}operator Gm<T>()const{return {0};}};
struct DataCopyParams{uint16_t blockCount,blockLen,srcStride,dstStride;};
void DataCopy(Gm<float>dst,Local<float>src,int count){
  std::lock_guard lock(shared->mutex);
  require(dst.offset>=0&&dst.offset+count<=Shared::extent,"GM store overflow");
  for(int i=0;i<count;++i){shared->work[dst.offset+i]=src.GetValue(i);shared->initialized[dst.offset+i]=true;}
}
void DataCopy(Local<float>dst,Gm<float>src,int count){
  std::lock_guard lock(shared->mutex);
  require(src.offset>=0&&src.offset+count<=Shared::extent,"GM load overflow");
  for(int i=0;i<count;++i){require(shared->initialized[src.offset+i],"uninitialized FP32 state read");dst.data->at(dst.offset+i)=shared->work[src.offset+i];}
}
template<typename A,typename B> void DataCopy(A dst,B src,DataCopyParams p){
  for(int block=0;block<p.blockCount;++block){
    auto d=dst;auto s=src;
    if constexpr(std::is_same_v<A,Gm<float>>)d.offset+=block*(p.blockLen+p.dstStride)*8;
    else d=d[block*(p.blockLen+p.dstStride)*8];
    if constexpr(std::is_same_v<B,Gm<float>>)s.offset+=block*(p.blockLen+p.srcStride)*8;
    else s=s[block*(p.blockLen+p.srcStride)*8];
    DataCopy(d,s,p.blockLen*8);
  }
}
struct Nd2NzParams{int ndNum,nValue,dValue,srcNdMatrixStride,srcDValue,dstNzC0Stride,dstNzNStride,dstNzMatrixStride;};
size_t nz_index(int row,int col){return (col/16)*256*16+row*16+col%16;}
void DataCopy(Local<uint16_t>dst,Gm<uint16_t>src,Nd2NzParams p){
  require(actor==0&&!shared->pendingMte1,"MTE2 overwrites live B1 MTE1 input");
  require(p.ndNum==1&&p.nValue==256&&p.dValue==256&&p.srcDValue==256&&
      p.dstNzC0Stride==256&&p.dstNzNStride==1&&p.srcNdMatrixStride==0&&p.dstNzMatrixStride==0,
      "protected C16 ND2NZ geometry changed");
  std::lock_guard lock(shared->mutex);
  for(int r=0;r<256;++r)for(int c=0;c<256;++c)
    dst.data->at(dst.offset+nz_index(r,c))=shared->operands.at(src.offset+r*256+c);
  ++shared->bLoads;shared->pendingMte2=true;shared->l1Ready=false;
}
void Duplicate(Local<float>dst,float value,int count){for(int i=0;i<count;++i)dst.data->at(dst.offset+i)=value;}
void Muls(Local<float>dst,Local<float>src,float value,int count){for(int i=0;i<count;++i)dst.data->at(dst.offset+i)=src.GetValue(i)*value;}
struct Matmul{
  int m=0,n=0,k=0,om=0,on=0,ok=0;Gm<uint16_t>a{0},b{0};
  Local<uint16_t> localB{nullptr,0};bool local=false,transpose=false;
  void SetOrgShape(int x,int y,int z){om=x;on=y;ok=z;}
  void SetSingleShape(int x,int y,int z){m=x;n=y;k=z;}
  void SetTensorA(Gm<uint16_t>value,bool trans){require(!trans,"A transpose changed");a=value;}
  void SetTensorB(Gm<uint16_t>value,bool trans){b=value;local=false;transpose=trans;}
  void SetTensorB(Global<uint16_t>,bool trans){SetTensorB(Gm<uint16_t>{0},trans);}
  void SetTensorB(Local<uint16_t>value,bool trans){
    require(shared->l1Ready&&!shared->pendingMte2,"B1 consumed before DMA completion");
    localB=value;local=true;transpose=trans;shared->pendingMte1=true;
  }
  template<bool>void IterateAll(Gm<float>out,uint8_t atomic=0){
    require(n==256&&k==256&&om==m&&on==n&&ok==k&&!hf32,"GEMM shape/FP32 accumulation contract changed");
    std::lock_guard lock(shared->mutex);
    const bool pv=!transpose;const int member=int(a.offset/(2*Shared::qSlot));
    const int tile=shared->currentAicTile;
    MatrixCall call{m,n,k,om,on,ok,transpose,bool(atomic),hf32,{}, {}};
    call.a.assign(shared->operands.begin()+a.offset,shared->operands.begin()+a.offset+m*k);
    call.b.resize(256*256);
    for(int r=0;r<256;++r)for(int c=0;c<256;++c)
      call.b[r*256+c]=local?localB.data->at(localB.offset+nz_index(r,c)):
          shared->operands.at(b.offset+r*256+c);
    require(shared->calls.emplace(std::tuple{tile,member,int(pv)},std::move(call)).second,"duplicate GEMM dispatch");
    if(!pv){
      const int slot=out.offset==Shared::baseScore?0:1;
      require(out.offset==Shared::baseScore||out.offset==Shared::extraScore,"new score storage introduced");
      require(shared->score[slot].consumed[0]&&shared->score[slot].consumed[1],"score overwritten before BOTH AIV reads finished");
      shared->score[slot]={tile,member,{false,false}};
    }else{
      require(atomic==(shared->cluster?1:0),"PV atomic mode changed");
      const int buffer=tile%2;
      require(shared->pReady[buffer][member][0]==tile&&shared->pReady[buffer][member][1]==tile,
          "PV consumes stale/unpublished P");
      if(atomic)++shared->atomicCalls;
      for(int row=0;row<m;++row)for(int col=0;col<256;++col){
        const int64_t at=out.offset+row*256+col;
        if(atomic)require(shared->initialized.at(at),"atomic PV consumes uninitialized alpha state");
        const float value=float((member+tile+row+col)%13-6)*0.03125F;
        shared->work.at(at)=(atomic?shared->work.at(at):0.0F)+value;
        shared->initialized.at(at)=true;
      }
    }
  }
  void End(){++shared->ends;}
};
struct TaskState{int64_t id,qbegin,qcount,kvhead,kvbegin,kvend,request,split,kind,context,requestBegin;int32_t taskError;};
struct Geometry{int hq=6,hk=1;};
template<bool Active>struct Kernel{
  ::Matmul mm;Buffer<uint16_t>residentB{256*256};
  Buffer<float>accBuf{64*256},statsBuf{192};
  Global<uint16_t>operands,rv;Global<float>work;
  Geometry g;int lane=0,error=0;
  int64_t qOffset=0,kOffset=Shared::baseK,vOffset=Shared::baseV,scoreOffset=Shared::baseScore,
      pOffset=kQueryRows*D/2,pvOffset=Shared::pv,accStateOffset=Shared::acc,
      statStateOffset=Shared::stats,baseKOffset=Shared::baseK,baseScoreOffset=Shared::baseScore,
      extraKvOffset=Shared::extraKv,extraScoreOffset=Shared::extraScore;
  int64_t qbegin=0,qcount=0,kvbegin=0,kvend=0,kind=0;
  int first=64;
  void UseTask(const TaskState&t){qbegin=t.qbegin;qcount=t.qcount;kvbegin=t.kvbegin;kvend=t.kvend;kind=t.kind;}
@@SELECT@@
@@PROBABILITY@@
@@RESIDENT_LOAD@@
@@RESIDENT_MATMUL@@
@@BASELINE_MATMUL@@
@@ACTIVE_MATMUL@@
@@BASELINE_ATOMIC@@
  void Matmul(int64_t a,int64_t b,int64_t c,int32_t m,int32_t n,int32_t k,bool rotation=false,bool transposeB=true){
    if constexpr(Active)MatmulActive(a,b,c,m,n,k,rotation,transposeB);else MatmulBaseline(a,b,c,m,n,k,rotation,transposeB);
  }
  void MatmulAtomic(int64_t a,int64_t b,int64_t c,int32_t m,int32_t n,int32_t k){MatmulAtomicBaseline(a,b,c,m,n,k);}
@@BASELINE_CUBE@@
@@ACTIVE_CUBE@@
@@CUBE_TASK@@
@@VECTOR_CLUSTER@@
@@SPILL@@
@@RESTORE@@
  void LoadQueries(){
    std::lock_guard lock(shared->mutex);const int member=int(qOffset/Shared::qSlot);
    for(int row=lane*64;row<(lane+1)*64;++row)for(int c=0;c<256;++c)
      shared->operands.at(2*qOffset+row*256+c)=uint16_t(shared->generation*101+member*157+row*256+c);
    if(shared->generation&&member==1&&lane==1)error=2;
  }
  void LoadClusterKv(int64_t start){
    const int tile=int((start-first)/256),buffer=(kOffset==baseKOffset?0:1);
    std::lock_guard lock(shared->mutex);
    for(int row=lane*128;row<(lane+1)*128;++row)for(int c=0;c<256;++c){
      shared->operands.at(2*kOffset+row*256+c)=uint16_t(shared->generation*131+tile*193+row*256+c);
      shared->operands.at(2*vOffset+row*256+c)=uint16_t(shared->generation*179+tile*223+row*256+c+7);
    }
    shared->kvTile[buffer][lane]=tile;
    if(shared->generation&&tile==1)error=3;
  }
  void Softmax(int64_t start,int rowBase){
    const int tile=int((start-first)/256),member=int(qOffset/Shared::qSlot),slot=member%2;
    std::lock_guard lock(shared->mutex);
    require(shared->score[slot].tile==tile&&shared->score[slot].member==member,"AIV reads stale score generation");
    for(int r=0;r<32;++r){
      const int row=rowBase+r,local=row-lane*64;
      if(row>=qcount*g.hq/g.hk)continue;
      statsBuf.data[local]=float(member+tile+row);
      statsBuf.data[64+local]=statsBuf.data[64+local]*0.5F+0.25F;
      for(int c=0;c<256;++c){
        accBuf.data[local*256+c]*=0.5F;
        shared->operands.at(2*ProbabilityOffset(qOffset)+row*256+c)=uint16_t(shared->generation*233+tile*269+member*281+row*256+c);
      }
    }
    if(rowBase==lane*64+32){shared->score[slot].consumed[lane]=true;shared->pReady[tile%2][member][lane]=tile;}
    if(shared->generation&&member==2&&lane==0)error=2;
    if((tile+member+lane)%2)std::this_thread::yield();
  }
  void PublishRows(int64_t id){
    std::lock_guard lock(shared->mutex);auto& result=shared->published[lane];
    result.push_back(uint32_t(id));result.push_back(uint32_t(error));
    for(float x:accBuf.data)result.push_back(bits(x));
    for(int i=0;i<128;++i)result.push_back(bits(statsBuf.data[i]));
  }
  void Cube(const TaskState*members){
    if constexpr(Active)CubeClusterActive(members);else CubeClusterBaseline(members);
  }
  void C1Vector(){
    qOffset=0;LoadQueries();
    for(int tile=0;tile<shared->tiles;++tile){
      LoadClusterKv(first+tile*256);CrossCoreSetFlag<2,PIPE_MTE3>(kVectorReady);
      CrossCoreWaitFlag(kCubeReady);
      // The real C1 numerical method is unchanged; model only its score/P
      // ownership here. No FP32 accumulator needs to be shared with Cube.
      for(int block=0;block<2;++block)Softmax(first+tile*256,lane*64+block*32);
      CrossCoreSetFlag<2,PIPE_MTE3>(kVectorReady);CrossCoreWaitFlag(kCubeReady);
      CrossCoreSetFlag<2,PIPE_MTE3>(kVectorReady);
    }
    if(kind==0)CrossCoreSetFlag<2,PIPE_MTE3>(kVectorReady);
  }
};

struct Result{
  std::map<std::tuple<int,int,int>,MatrixCall>calls;
  std::array<std::vector<uint32_t>,2>published;
  std::array<std::vector<std::array<int,2>>,3>flags;
  int loads,ends,atomics;
};
template<bool Active>Result run(Shared&model){
  std::array<TaskState,4>members{};
  for(int m=0;m<4;++m)members[m]={m,m*model.query,model.query,0,64,64+model.tiles*256-17,0,0,model.kind,64+model.tiles*256-17,0,0};
  std::array<std::thread,3>threads;
  for(int id=0;id<3;++id)threads[id]=std::thread([&,id]{
    shared=&model;actor=id;
    try{
      Kernel<Active>kernel;kernel.g.hq=model.group;kernel.lane=id-1;kernel.UseTask(members[0]);
      if(id==0){if(model.cluster)kernel.Cube(members.data());else kernel.CubeTask();}
      else if(model.cluster)kernel.VectorCluster(members.data());else kernel.C1Vector();
    }catch(...){model.fail();}
  });
  for(auto&thread:threads)thread.join();if(model.failure)std::rethrow_exception(model.failure);
  for(const auto&lane:model.toVector)for(int token:lane)require(token==0,"undrained AIC to AIV flag");
  for(const auto&lane:model.toCube)for(int token:lane)require(token==0,"undrained AIV to AIC flag");
  return {model.calls,model.published,model.flags,model.bLoads,model.ends,model.atomicCalls};
}
int main(){
  try{
    int cases=0;
    Shared old,candidate;
    const auto*address=candidate.operands.data();
    for(int generation:{0,1})for(int tiles:{1,2,3})for(int group:{1,6,8}){
      old.reset(generation,tiles,group,128/group,true,0);candidate.reset(generation,tiles,group,128/group,true,0);
      auto a=run<false>(old),b=run<true>(candidate);
      require(a.calls==b.calls&&a.published==b.published,"cluster inputs or published state/error bits changed");
      require(a.flags[1]==b.flags[1]&&a.flags[2]==b.flags[2],"unchanged Vector actor token sequence changed");
      require(b.loads==2*tiles&&a.ends==b.ends&&b.ends==8*tiles&&b.atomics==4*tiles,"cluster L1/matmul accounting wrong");
      require(candidate.operands.data()==address,"same-address generation storage changed");++cases;
    }
    for(int generation:{0,1})for(int source:{0,1,2})for(int query:{1,4,21}){
      old.reset(generation,2,6,query,false,source);candidate.reset(generation,2,6,query,false,source);
      auto a=run<false>(old),b=run<true>(candidate);
      require(a.calls==b.calls&&a.flags==b.flags,"C1 shape/input/transpose/flag order changed");
      require(b.loads==4&&a.ends==b.ends&&b.atomics==0,"C1 added repeated B load/atomic work");++cases;
    }
    std::cout<<"{\"host_control_only\":true,\"actors\":3,\"cases\":"<<cases
      <<",\"matrix_input_bits\":\"identical\",\"state_error_bits\":\"identical\",\"B1_bytes\":131072,"
      <<"\"cluster_B_loads_per_tile\":2,\"C1_B_loads_per_tile\":2,\"CANN\":\"not_run\",\"NPU\":\"not_run\",\"performance\":\"not_run\"}\n";
    return 0;
  }catch(const std::exception&e){std::cerr<<"C4_L1_CONTROL_FAILED "<<e.what()<<"\n";return 1;}
}
