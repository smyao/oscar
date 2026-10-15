// Archive #34/#36/#129/#148/#152/#157 and D.4: host traversal only.
// Actual Process/eligibility methods are injected byte-for-byte by generate.py.
// Numerical kernel methods are event recorders, NOT CPU-debug/NPU substitutes.
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "../../include/oscar_cv_schedule.h"
#define __aicore__
#define ASCEND_IS_AIC (aic)
#define ASCEND_IS_AIV (!aic)
constexpr int32_t kQueryRows=128,kKvRows=256;
int64_t Min64(int64_t a,int64_t b){return std::min(a,b);}
enum class CacheLine {ENTIRE_DATA_CACHE,SINGLE_CACHE_LINE};
enum class DcciDst {CACHELINE_OUT};
template<class T,CacheLine C,DcciDst D=DcciDst::CACHELINE_OUT,class U>
void DataCacheCleanAndInvalid(U&&) {}
using Row=std::array<int64_t,16>;
using Event=std::array<int64_t,14>;
struct Geometry {int64_t tokens,hq,hk,requests,splits;};
struct TaskState {int64_t id,qbegin,qcount,kvhead,kvbegin,kvend,request,split,kind,context,requestBegin;int32_t taskError;};
struct Stats {std::vector<int64_t> data;void SetValue(int64_t i,int64_t v){if(i>=int64_t(data.size()))data.resize(i+1);data[i]=v;}int64_t& operator[](int64_t i){return data[i];}};
struct Native {Geometry g;int64_t sink=64,recent=1;std::vector<int64_t> starts,lens,slots;bool slotContext=false;std::vector<std::vector<int64_t>> tables;};

// Direct transcription of the pinned attention_tasks.cpp native array rules.
// Each real saved CPU prepare table is compared against every generated word.
std::vector<Row> Prepare(const Native& n) {
  const auto g=n.g;const int64_t qt=128/(g.hq/g.hk),pt=g.hk*3*g.splits;
  bool malformed=n.starts[0]!=0;
  for(int64_t r=0;r<g.requests;++r)if(n.starts[r]<0 || n.starts[r+1]<n.starts[r] || n.starts[r+1]>g.tokens)malformed=true;
  std::vector<Row> rows(g.tokens*pt);
  for(int64_t t=0;t<g.tokens;++t) {
    int64_t r=0;while(r<g.requests && n.starts[r+1]<=t)++r;
    const bool padding=n.slots[t]<0;bool invalid=malformed || (!padding&&r>=g.requests);
    int64_t begin=0,context=0,count=-1,error=invalid?1:0;
    if(!padding&&!invalid) {
      begin=n.starts[r];const int64_t qlen=n.starts[r+1]-begin;
      context=n.lens[r]-qlen;invalid=context<0;
      if(n.slotContext&&!invalid) {
        const int64_t columns=n.tables.empty()?0:n.tables[r].size();
        if(qlen!=1 || columns<=0){invalid=true;error=5;}
        else {int64_t matched=-1,matches=0;const int64_t valid=std::min(columns,(n.lens[r]+127)/128);
          for(int64_t col=0;col<valid;++col)if(n.tables[r][col]==n.slots[t]/128){matched=col;++matches;}
          const int64_t position=matched*128+n.slots[t]%128;
          if(matches!=1 || position<0 || position>=n.lens[r]){invalid=true;error=5;}else context=position;
        }
      }
      if(invalid&&error==0)error=1;
      bool leader=(t-begin)%qt==0;if(!leader)leader=n.slots[t-1]<0;
      count=0;
      if(leader&&!invalid){const int64_t end=std::min(n.starts[r+1],begin+((t-begin)/qt+1)*qt);
        while(t+count<end && n.slots[t+count]>=0)++count;}
    }
    if(invalid)count=-1;
    for(int64_t h=0;h<g.hk;++h)for(int64_t kind=0;kind<3;++kind)for(int64_t s=0;s<g.splits;++s){
      int64_t first=0,last=0;const int64_t pos=context+t-begin;
      if(count>0){
        if(kind==0){first=std::min(n.sink,context);last=std::min(context,std::max(n.sink,pos+count-n.recent));}
        if(kind==1){first=0;last=std::min(n.sink,context)+context-std::min(context,std::max(n.sink,pos+1-n.recent));}
        if(kind==2){first=context;last=pos+count;}
      }
      const int64_t width=(last-first+g.splits-1)/g.splits;
      const int64_t a=std::min(last,first+s*width);int64_t b=std::min(last,a+width);
      if(kind==2 && error==0 && b>=a)b=a; // existing legal current-source suppression
      rows[t*pt+(h*3+kind)*g.splits+s]={t,count,h,a,b,r<g.requests?r:0,kind*g.splits+s,kind,context,begin,error,0,0,0,0,0};
    }
  }
  return rows;
}

class Engine {
 public:
  Geometry g;const std::vector<Row>& rows;const Native* native;
  int64_t coreIndex,cores,qOffset=0;bool aic;int taskGm=0;Stats clusterStats;struct MatmulStub {void End(){}} mm;
  std::array<int64_t,8> counters{};std::vector<Event> events;
  int64_t loads=0,clusters=0;
  int64_t qbegin=0,qcount=0,kvhead=0,kvbegin=0,kvend=0,request=0,split=0,kind=0,context=0,requestBegin=0;int32_t taskError=0;
  Engine(Geometry geometry,const std::vector<Row>& input,const Native* original,int64_t core,int64_t count,bool cube):g(geometry),rows(input),native(original),coreIndex(core),cores(count),aic(cube){}
  int64_t GetBlockNum()const{return cores;}
  void LoadTask(int64_t id){
    if(id<0 || id>=int64_t(rows.size()))throw std::runtime_error("task access out of range");
    ++loads;const auto& d=rows[id];qbegin=d[0];qcount=d[1];kvhead=d[2];kvbegin=d[3];kvend=d[4];request=d[5];split=d[6];kind=d[7];context=d[8];requestBegin=d[9];taskError=d[10];
  }
  void Record(int64_t action,int64_t id,int64_t code){events.push_back({action,id,code,qbegin,qcount,kvhead,kvbegin,kvend,request,split,kind,context,requestBegin,taskError});}
  int64_t Id()const{return qbegin*(g.hk*3*g.splits)+(kvhead*3)*g.splits+split;}
  void PublishStatus(int64_t id,int32_t code){Record(3,id,code);}
  void PublishEmpty(int64_t id,int32_t code){Record(4,id,code);}
  void CubeTask(){Record(1,Id(),0);}
  void VectorTask(int64_t id){Record(1,id,0);}
  void RecordCluster(const TaskState* members,bool vector){
    ++clusters;const auto saved=CurrentTask(members[0].id);
    if(native){
      const int64_t first=members[0].qbegin,width=16*(128/(g.hq/g.hk));
      if(first<native->starts[members[0].request] || first+width>native->starts[members[0].request+1])throw std::runtime_error("qualified cluster crossed native request boundary");
      for(int64_t t=first;t<first+width;++t)if(native->slots[t]<0)throw std::runtime_error("qualified cluster covered native slot hole");
    }
    for(int i=0;i<16;++i){UseTask(members[i]);Record(2,members[i].id,0);}
    // Actual AIC keeps its original fields; actual AIV exits on last member.
    if(!vector)UseTask(saved);
  }
  void CubeCluster(const TaskState* members){RecordCluster(members,false);}
  void VectorCluster(const TaskState* members){RecordCluster(members,true);}
__aicore__ TaskState CurrentTask(int64_t id) const {
    return TaskState{id,qbegin,qcount,kvhead,kvbegin,kvend,request,split,kind,
        context,requestBegin,taskError};
  }
__aicore__ void UseTask(const TaskState& t) {
    qbegin=t.qbegin;qcount=t.qcount;kvhead=t.kvhead;kvbegin=t.kvbegin;
    kvend=t.kvend;request=t.request;split=t.split;kind=t.kind;
    context=t.context;requestBegin=t.requestBegin;taskError=t.taskError;
  }
__aicore__ bool TaskValid() {
    return qbegin>=0 && qbegin<g.tokens && qcount<=kQueryRows/(g.hq/g.hk) &&
        (qcount<0 || qbegin+qcount<=g.tokens) && kvhead>=0 && kvhead<g.hk &&
        kvbegin>=0 && kvend>=kvbegin && request>=0 && request<g.requests &&
        split>=0 && split<3*g.splits && kind>=0 && kind<3 && context>=0 &&
        requestBegin>=0 && requestBegin<=qbegin &&
        (kind!=0 || kvend<=context) &&
        (kind!=2 || (kvbegin>=context && requestBegin+kvend-context<=g.tokens));
  }
__aicore__ bool IsClusterMember(int64_t id,TaskState* members) {
    const int64_t queryTile=kQueryRows/(g.hq/g.hk);
    const int64_t width=16*queryTile;
    if(g.tokens<width || kind!=0 || qcount!=queryTile || taskError!=0 ||
        kvbegin>=kvend || kvend!=context || qbegin<requestBegin ||
        (qbegin-requestBegin)%queryTile!=0 || !TaskValid())return false;
    const TaskState original=CurrentTask(id);
    const int64_t anchor=requestBegin+
        ((qbegin-requestBegin)/width)*width;
    if(anchor<0 || anchor+width>g.tokens) return false;
    const int64_t perToken=g.hk*3*g.splits;
    const int64_t segment=(kvhead*3)*g.splits+split;
    bool valid=true;
    TaskState found[16];
    for(int32_t j=0;j<16;++j) {
      const int64_t candidate=anchor+j*queryTile;
      const int64_t candidateId=candidate*perToken+segment;
      LoadTask(candidateId);
      found[j]=CurrentTask(candidateId);
      if(qbegin!=candidate || qcount!=queryTile || kind!=0 || taskError!=0 ||
          request!=original.request || kvhead!=original.kvhead ||
          split!=original.split || context!=original.context ||
          requestBegin!=original.requestBegin ||
          kvbegin!=original.kvbegin || kvend!=original.kvend ||
          kvend!=context || !TaskValid())valid=false;
    }
    UseTask(original);
    if(valid && members!=nullptr)for(int32_t j=0;j<16;++j)members[j]=found[j];
    return valid;
  }
__aicore__ void ProcessBaseline() {
    // Scalar Cube metadata reads must invalidate stale cache on changed-input
    // graph replay. The Vector side exclusively DMA-loads mutable metadata.
    if ASCEND_IS_AIC {
      DataCacheCleanAndInvalid<int64_t,CacheLine::ENTIRE_DATA_CACHE>(taskGm);
    }
    const int64_t queryTile=kQueryRows/(g.hq/g.hk);
    const int64_t perToken=g.hk*3*g.splits;
    const int64_t scheduleTile=4; // ownership only; mathematical query groups stay unchanged
    const oscar_ascend_schedule::CvTaskSchedule schedule{g.tokens,scheduleTile,perToken};
    // Pass one balances ownership of every follower, invalid leader and
    // noneligible leader. Exact C16 members alone are deferred to pass two.
    // The predicate is read-only and identical on AIC/AIV; no global barrier.
    for(int64_t workId=coreIndex;workId<schedule.WorkItems();workId+=GetBlockNum()) {
      const int64_t tokenBegin=schedule.TokenBegin(workId);
      for(int64_t token=tokenBegin;token<Min64(tokenBegin+scheduleTile,g.tokens);++token) {
        const int64_t id=schedule.TaskId(workId,token);
        LoadTask(id);
        if(qcount==0) {if ASCEND_IS_AIV {PublishStatus(id,0);} continue;}
        if(qcount<0 || taskError || !TaskValid()) {
          if ASCEND_IS_AIV {PublishEmpty(id,taskError?taskError:(!TaskValid()?1:0));}
          continue;
        }
        if(IsClusterMember(id,nullptr)) {
          if ASCEND_IS_AIC {++counters[6];}
          continue;
        }
        qOffset=0;
        if ASCEND_IS_AIC {
          if(kind==0) {
            ++counters[2];
            counters[5]+=(kvend-kvbegin+kKvRows-1)/kKvRows;
          }
          CubeTask();
        } else {VectorTask(id);}
      }
    }
    // Pass two maps contiguous 336-token buckets across ALL Cubes, avoiding
    // the original #129 stride/common-divisor alias of anchor%4. Each anchor
    // belongs to exactly one global bucket; its sixteen request-relative members
    // have already been excluded in pass one by the same predicate.
    if(g.tokens>=16*queryTile) {
      const int64_t bucketWidth=16*queryTile;
      const int64_t buckets=(g.tokens+bucketWidth-1)/bucketWidth;
      const int64_t segments=g.hk*g.splits;
      for(int64_t workId=coreIndex;workId<buckets*segments;workId+=GetBlockNum()) {
        const int64_t bucket=workId%buckets;
        const int64_t segment=workId/buckets;
        const int64_t kvHead=segment/g.splits;
        const int64_t splitIndex=segment%g.splits;
        const int64_t taskSegment=(kvHead*3)*g.splits+splitIndex;
        const int64_t begin=bucket*bucketWidth;
        const int64_t end=Min64(begin+bucketWidth,g.tokens);
        for(int64_t token=begin;token<end;++token) {
          const int64_t id=token*perToken+taskSegment;
          LoadTask(id);
          if(qcount!=queryTile || qbegin!=token || kind!=0 ||
              qbegin<requestBegin ||
              (qbegin-requestBegin)%bucketWidth!=0)continue;
          TaskState members[16];
          if(!IsClusterMember(id,members))continue;
          if ASCEND_IS_AIC {
            ++counters[0];counters[1]+=16;++counters[7];
            const int64_t tiles=(members[0].kvend-members[0].kvbegin+kKvRows-1)/kKvRows;
            counters[3]+=tiles;counters[4]+=15*tiles;
            CubeCluster(members);
          } else {VectorCluster(members);}
        }
      }
    }
    if ASCEND_IS_AIC {
      mm.End();
      for(int32_t i=0;i<8;++i)clusterStats.SetValue(coreIndex*8+i,counters[i]);
      // Each AIC owns exactly one 64-byte line; scalar GM SetValue needs an
      // explicit A2 DCache writeback before the host reads the diagnostics.
      DataCacheCleanAndInvalid<int64_t,CacheLine::SINGLE_CACHE_LINE,
          DcciDst::CACHELINE_OUT>(clusterStats[coreIndex*8]);
    }
  }
__aicore__ void ProcessCandidate() {
    // Scalar Cube metadata reads must invalidate stale cache on changed-input
    // graph replay. The Vector side exclusively DMA-loads mutable metadata.
    if ASCEND_IS_AIC {
      DataCacheCleanAndInvalid<int64_t,CacheLine::ENTIRE_DATA_CACHE>(taskGm);
    }
    const int64_t queryTile=kQueryRows/(g.hq/g.hk);
    const int64_t perToken=g.hk*3*g.splits;
    const int64_t scheduleTile=4; // ownership only; mathematical query groups stay unchanged
    const oscar_ascend_schedule::CvTaskSchedule schedule{g.tokens,scheduleTile,perToken};
    // Pass one balances ownership of every follower, invalid leader and
    // noneligible leader. Exact C16 members alone are deferred to pass two.
    // The predicate is read-only and identical on AIC/AIV; no global barrier.
    for(int64_t workId=coreIndex;workId<schedule.WorkItems();workId+=GetBlockNum()) {
      const int64_t tokenBegin=schedule.TokenBegin(workId);
      for(int64_t token=tokenBegin;token<Min64(tokenBegin+scheduleTile,g.tokens);++token) {
        const int64_t id=schedule.TaskId(workId,token);
        LoadTask(id);
        if(qcount==0) {if ASCEND_IS_AIV {PublishStatus(id,0);} continue;}
        if(qcount<0 || taskError || !TaskValid()) {
          if ASCEND_IS_AIV {PublishEmpty(id,taskError?taskError:(!TaskValid()?1:0));}
          continue;
        }
        if(IsClusterMember(id,nullptr)) {
          if ASCEND_IS_AIC {++counters[6];}
          continue;
        }
        qOffset=0;
        if ASCEND_IS_AIC {
          if(kind==0) {
            ++counters[2];
            counters[5]+=(kvend-kvbegin+kKvRows-1)/kKvRows;
          }
          CubeTask();
        } else {VectorTask(id);}
      }
    }
    // Pass two maps contiguous 336-token buckets across ALL Cubes, avoiding
    // the original #129 stride/common-divisor alias of anchor%4. Each anchor
    // belongs to exactly one global bucket; its sixteen request-relative members
    // have already been excluded in pass one by the same predicate.
    if(g.tokens>=16*queryTile) {
      const int64_t bucketWidth=16*queryTile;
      const int64_t buckets=(g.tokens+bucketWidth-1)/bucketWidth;
      const int64_t segments=g.hk*g.splits;
      for(int64_t workId=coreIndex;workId<buckets*segments;workId+=GetBlockNum()) {
        const int64_t bucket=workId%buckets;
        const int64_t segment=workId/buckets;
        const int64_t kvHead=segment/g.splits;
        const int64_t splitIndex=segment%g.splits;
        const int64_t taskSegment=(kvHead*3)*g.splits+splitIndex;
        const int64_t begin=bucket*bucketWidth;
        const int64_t end=Min64(begin+bucketWidth,g.tokens);
        for(int64_t token=begin;token<end;++token) {
          const int64_t id=token*perToken+taskSegment;
          LoadTask(id);
          if(qcount!=queryTile || qbegin!=token || kind!=0 ||
              qbegin<requestBegin ||
              (qbegin-requestBegin)%bucketWidth!=0)continue;
          TaskState members[16];
          if(!IsClusterMember(id,members))continue;
          if ASCEND_IS_AIC {
            ++counters[0];counters[1]+=16;++counters[7];
            const int64_t tiles=(members[0].kvend-members[0].kvbegin+kKvRows-1)/kKvRows;
            counters[3]+=tiles;counters[4]+=15*tiles;
            CubeCluster(members);
          } else {VectorCluster(members);}
          // The native prepare ABI gives sixteen adjacent full groups in
          // one request. Their union covers [token, token+bucketWidth),
          // so no additional request-relative cluster anchor is inside it.
          // First-pass errors/statuses have already retained every row.
          // VectorCluster mutates qbegin: use the unchanged loop token.
          // Clamp to this global bucket and avoid token+width overflow.
          token += Min64(bucketWidth,end-token)-1;
        }
      }
    }
    if ASCEND_IS_AIC {
      mm.End();
      for(int32_t i=0;i<8;++i)clusterStats.SetValue(coreIndex*8+i,counters[i]);
      // Each AIC owns exactly one 64-byte line; scalar GM SetValue needs an
      // explicit A2 DCache writeback before the host reads the diagnostics.
      DataCacheCleanAndInvalid<int64_t,CacheLine::SINGLE_CACHE_LINE,
          DcciDst::CACHELINE_OUT>(clusterStats[coreIndex*8]);
    }
  }
};

struct Totals {int64_t cases=0,actors=0,events=0,baseLoads=0,newLoads=0,clusters=0;};
Totals total;
void Replay(const Geometry& g,const std::vector<Row>& rows,const Native* n,int cores,const std::string& label,bool requireEqual=true){
  bool difference=false;
  for(int core=0;core<cores;++core)for(bool aic:{true,false}){
    Engine old(g,rows,n,core,cores,aic),candidate(g,rows,n,core,cores,aic);
    old.ProcessBaseline();candidate.ProcessCandidate();
    bool same=old.events==candidate.events && old.counters==candidate.counters && old.clusterStats.data==candidate.clusterStats.data;
    if(!same)difference=true;
    if(requireEqual&&!same)throw std::runtime_error("event/counter mismatch "+label+" core="+std::to_string(core)+" aic="+std::to_string(aic));
    if(requireEqual && candidate.loads>old.loads)throw std::runtime_error("metadata work increased");
    ++total.actors;total.events+=old.events.size();total.baseLoads+=old.loads;total.newLoads+=candidate.loads;total.clusters+=old.clusters;
  }
  if(!requireEqual&&!difference)throw std::runtime_error("unreachable overlapping-task counterexample did not distinguish scope");
  ++total.cases;
}
Native Make(std::vector<int64_t> lengths,int ratio=6,int hk=1,int splits=1,int64_t ctx=337,int recent=1,int padding=0){
  Native n;n.g={0,ratio*hk,hk,int64_t(lengths.size()),splits};n.recent=recent;n.starts={0};
  for(auto length:lengths){n.g.tokens+=length;n.starts.push_back(n.g.tokens);n.lens.push_back(length+ctx);}
  const int64_t actual=n.g.tokens;n.g.tokens=std::max<int64_t>(1,n.g.tokens+padding);n.slots.resize(n.g.tokens);
  for(int64_t t=0;t<n.g.tokens;++t)n.slots[t]=t<actual?t+1024:-1;
  return n;
}
void Case(const Native& n,const std::string& name,int cores=20){Replay(n.g,Prepare(n),&n,cores,name);}

template<class T>std::vector<T> Read(const std::string& path){std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("missing "+path);f.seekg(0,std::ios::end);auto size=f.tellg();f.seekg(0);if(size%sizeof(T))throw std::runtime_error("unaligned file");std::vector<T> out(size/sizeof(T));f.read(reinterpret_cast<char*>(out.data()),size);return out;}
void Actual(int argc,char**argv){
  if(argc!=11)throw std::runtime_error("actual expects fixture prepared N Hq Hkv R S sink recent");
  const std::string f=argv[2],p=argv[3];Native n;n.g={std::stoll(argv[4]),std::stoll(argv[5]),std::stoll(argv[6]),std::stoll(argv[7]),std::stoll(argv[8])};n.sink=std::stoll(argv[9]);n.recent=std::stoll(argv[10]);
  auto starts=Read<int32_t>(f+"/starts.bin"),lens=Read<int32_t>(f+"/lens.bin");n.starts.assign(starts.begin(),starts.end());n.lens.assign(lens.begin(),lens.end());n.slots=Read<int64_t>(f+"/slots.bin");
  auto words=Read<int64_t>(p+"/tasks.bin");auto generated=Prepare(n);
  if(words.size()!=generated.size()*16)throw std::runtime_error("real task capacity mismatch");
  for(size_t i=0;i<words.size();++i)if(words[i]!=generated[i/16][i%16])throw std::runtime_error("native transcription differs from actual AscendC prepare word "+std::to_string(i));
  for(int grid:{1,3,20,32})Replay(n.g,generated,&n,grid,"actual");
  std::cout<<"{\"actual_prepare_words_equal\":"<<words.size()<<",\"cases\":"<<total.cases<<",\"actors\":"<<total.actors<<",\"base_LoadTask\":"<<total.baseLoads<<",\"candidate_LoadTask\":"<<total.newLoads<<"}\n";
}
void Enumerate(){
  // Every mask of every 8-token group in one mature GQA16 C16 interval.
  for(int group=0;group<16;++group)for(int mask=0;mask<256;++mask){auto n=Make({128},16);for(int bit=0;bit<8;++bit)if(mask&(1<<bit))n.slots[group*8+bit]=-1;Case(n,"all_8bit_slot_masks",(mask%2)?3:20);}
  const int w=336;
  // Every request-start offset in a global336-token bucket, including empty
  // native rows. Requests have different lengths; no global alignment assumed.
  for(int prefix=0;prefix<w;++prefix){auto n=Make({prefix,2*w+5,3},6,1,3,337,256,4);Case(n,"request_offset",20);}
  // Every single and adjacent-pair hole, including exact query/cluster ends.
  for(int hole=0;hole<2*w+21;++hole){auto n=Make({2*w+21},6,1,1,337,1,3);n.slots[hole]=-1;Case(n,"single_hole",3);if(hole+1<2*w+21){n.slots[hole+1]=-1;Case(n,"adjacent_holes",20);}}
  for(int ratio=1;ratio<=16;++ratio)for(int hk:{1,2})for(int splits:{1,2,3,7})for(int recent:{1,256}){
    const int width=16*(128/ratio);auto n=Make({3,width*2+1,5},ratio,hk,splits,65536,recent,5);Case(n,"all_ratio_head_split",20);
  }
  for(int64_t ctx:{int64_t(0),int64_t(1),int64_t(64),int64_t(65),int64_t(337),int64_t(65536)})for(int length:{1,20,21,127,128,335,336,337,671,672,673})Case(Make({length},6,1,1,ctx,256,4),"context_tail",1);
  for(int grid:{1,2,3,20,32})Case(Make({16384},6,1,1,49152,256),"long16384",grid);
  // Malformed native starts and context errors retain original failure paths.
  auto bad=Make({336,336});bad.starts[0]=1;Case(bad,"bad_start",3);
  bad=Make({336,336});bad.starts[2]=335;Case(bad,"backward_start",20);
  bad=Make({336});bad.lens[0]=335;Case(bad,"negative_context",20);
  bad=Make({336});std::fill(bad.slots.begin(),bad.slots.end(),-1);Case(bad,"all_explicit_padding",20);
  bad=Make({336});bad.starts[1]=335;Case(bad,"live_row_outside_requests",20);
  // Later-draft valid q1 and bad table rows are not discarded or clustered.
  for(int fault=0;fault<3;++fault){auto n=Make({1,1},6,1,3,128,256,2);n.slotContext=true;n.slots[0]=128;n.slots[1]=128;n.tables={{0,1},{0,1}};if(fault==1)n.tables[0]={0,9};if(fault==2)n.tables[0]={1,1};Case(n,"later_draft_table",20);}
  bad=Make({336});bad.slotContext=true;bad.tables={{0,1,2,3,4,5}};Case(bad,"invalid_later_long_query",20);
  const Totals nativeTotals=total;
  // Boundary of the proof: forged overlapping request task domains cannot be
  // emitted by native starts/slots. The generic raw-row ABI alone is weaker.
  auto impossible=Make({337},6);auto rows=Prepare(impossible);impossible.g.requests=2;
  for(int j=0;j<16;++j){int t=1+j*21;rows[t*3]={t,21,0,64,337,1,0,0,337,1,0,0,0,0,0,0};}
  Replay(impossible.g,rows,nullptr,20,"unreachable_overlap",false);
  std::cout<<"{\"status\":\"native_reachable_traversal_events_equal\",\"cases\":"<<nativeTotals.cases<<",\"actors\":"<<nativeTotals.actors<<",\"events\":"<<nativeTotals.events<<",\"base_LoadTask\":"<<nativeTotals.baseLoads<<",\"candidate_LoadTask\":"<<nativeTotals.newLoads<<",\"arbitrary_overlapping_table_not_covered\":true,\"floating_point_kernel_executed\":false}\n";
}
int main(int argc,char**argv){try{if(argc>1&&std::string(argv[1])=="--actual")Actual(argc,argv);else Enumerate();return 0;}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 2;}}
