// SPDX-License-Identifier: Apache-2.0
// Archive G26-G34/#12/#13-20/#34/#36/#53-69/#92: execute the actual AscendC
// task/CV/merge kernel bodies under official tikicpulib with independent gold.
// D.4: this checks numerical correctness only; CPU-debug cannot establish NPU
// device completion, graph capture/replay, timing or the 32K performance gate.
// Archive #129: large task-table causal bounds are checked without a dense 16K oracle.
#include "tikicpulib.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "../include/oscar_cv_schedule.h"
extern "C" void oscar_prepare_attention_tasks_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,bool,uint8_t*,int64_t,bool);
extern "C" void oscar_attention_cv_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,float);
extern "C" void oscar_merge_lse_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    int64_t,int64_t,int64_t);
namespace {
std::vector<uint8_t> Read(const std::string& path,size_t n) {
  std::vector<uint8_t> data(n);std::ifstream f(path,std::ios::binary);
  if(!f.read(reinterpret_cast<char*>(data.data()),n)||f.peek()!=EOF)
    throw std::runtime_error("incorrect golden byte count: "+path);
  return data;
}
struct Gm {
  uint8_t* ptr;size_t size;
  explicit Gm(size_t n,uint8_t fill=0x85):ptr(static_cast<uint8_t*>(AscendC::GmAlloc(n))),size(n) {
    if(!ptr)throw std::runtime_error("GmAlloc failed");std::memset(ptr,fill,n);
  }
  Gm(const std::string& path,size_t n):Gm(n) {auto v=Read(path,n);std::memcpy(ptr,v.data(),n);}
  ~Gm(){AscendC::GmFree(ptr);}Gm(const Gm&)=delete;
};
void StatusZero(const Gm& status) {
  for(size_t i=0;i<status.size/4;++i) {int32_t s;std::memcpy(&s,status.ptr+i*4,4);
    if(s)throw std::runtime_error("device status="+std::to_string(s)+" index="+std::to_string(i));}
}
void CheckLargeCausalTasks() {
  constexpr int64_t n=16384,hq=6,hk=1,splits=3;
  for(int64_t cores:{2,20,24,32}) {
    const oscar_ascend_schedule::CvTaskSchedule schedule{n,64/hq,3};
    std::vector<int> seen(n*3,0),live(cores,0);
    for(int64_t core=0;core<cores;++core)
    for(int64_t item=core;item<schedule.WorkItems();item+=cores)
    for(int64_t token=schedule.TokenBegin(item);token<std::min(n,schedule.TokenBegin(item)+schedule.queryTile);++token) {
      auto id=schedule.TaskId(item,token);
      if(id<0 || id>=n*3 || ++seen[id]!=1)throw std::runtime_error("task scheduled outside capacity or twice");
      if(token%schedule.queryTile==0 && schedule.Segment(item)==2)++live[core];
    }
    for(auto count:seen)if(count!=1)throw std::runtime_error("task omitted from schedule");
    auto range=std::minmax_element(live.begin(),live.end());
    if(*range.first<=0 || *range.second-*range.first>1)throw std::runtime_error("prefill leaders stranded on a subset of Cubes");
    std::cout<<"scheduled_cores="<<cores<<" min_live="<<*range.first<<" max_live="<<*range.second<<std::endl;
  }
  Gm starts(12),lens(8),slots(n*8),tasks(n*3*splits*16*8),positions(n*8);
  const int32_t s[3]={0,7,n},l[2]={20,31+n-7};
  std::memcpy(starts.ptr,s,sizeof(s));std::memcpy(lens.ptr,l,sizeof(l));
  for(int64_t i=0;i<n;++i)reinterpret_cast<int64_t*>(slots.ptr)[i]=(i==43?-1:i);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  // Match the production task launch (32 AIVs) for the full 16K table.
  ICPU_RUN_KF(oscar_prepare_attention_tasks_kernel,32,starts.ptr,lens.ptr,slots.ptr,
      tasks.ptr,positions.ptr,int64_t{2},n,hq,hk,int64_t{4},int64_t{32},splits,true,
      static_cast<uint8_t*>(nullptr),int64_t{0},false);
  int64_t leaders=0,tiles=0,oldTiles=0;
  for(int64_t token=0;token<n;++token) {
    const auto* row=reinterpret_cast<const int64_t*>(tasks.ptr)+(token*3*splits+2*splits)*16;
    if(row[1]<=0)continue;
    ++leaders;
    const int64_t request=token<7?0:1,begin=s[request],context=request==0?13:31;
    const int64_t expectedEnd=context+token-begin+row[1];
    if(row[3]!=context)throw std::runtime_error("current source changed its first token");
    int64_t end=context;
    for(int64_t split=0;split<splits;++split) {
      const auto* part=row+split*16;
      if(part[3]!=end || part[4]<part[3] || part[4]>expectedEnd)
        throw std::runtime_error("current source scans future tokens or has a split gap");
      tiles+=(part[4]-part[3]+31)/32;end=part[4];
    }
    if(end!=expectedEnd)throw std::runtime_error("causal current source misses visible tokens");
    oldTiles+=(s[request+1]-begin+31)/32;
  }
  std::cout<<"causal_task_leaders="<<leaders<<" current_tiles="<<tiles
           <<" previous_unsplit_tiles="<<oldTiles<<std::endl;
  std::cout<<"{\"backend\":\"ascendc_cpu_debug\",\"op\":\"tasks_causal\",\"status\":\"passed\"}"<<std::endl;
}
void CheckPaddedMetadata() {
  constexpr int64_t n=8,heads=1;
  Gm starts(12),lens(8),slots(n*8),tasks(n*3*16*8),positions(n*8);
  const int32_t startValues[3]={0,4,8},lengthValues[2]={100,0};
  const int64_t slotValues[8]={96,97,98,99,-1,-1,-1,-1};
  std::memcpy(starts.ptr,startValues,12);std::memcpy(lens.ptr,lengthValues,8);
  std::memcpy(slots.ptr,slotValues,n*8);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(oscar_prepare_attention_tasks_kernel,4,starts.ptr,lens.ptr,slots.ptr,
      tasks.ptr,positions.ptr,int64_t{2},n,int64_t{6},heads,int64_t{4},int64_t{32},int64_t{1},true,static_cast<uint8_t*>(nullptr),int64_t{0},false);
  for(int64_t token=0;token<n;++token) {
    int64_t position;std::memcpy(&position,positions.ptr+token*8,8);
    if(position!=(token<4?96+token:-1))throw std::runtime_error("padded request position mismatch");
    for(int64_t kind=0;kind<3;++kind) {
      const auto* task=reinterpret_cast<const int64_t*>(tasks.ptr)+(token*3+kind)*16;
      const int64_t expectedCount=token==0?4:(token<4?0:-1);
      if(task[1]!=expectedCount || task[10]!=0)
        throw std::runtime_error("dummy request invalidated live request metadata");
    }
  }
  // A padding hole splits a live group; both sides still have unique owners.
  const int64_t holeSlots[8]={96,-1,98,99,-1,-1,-1,-1};
  std::memcpy(slots.ptr,holeSlots,n*8);
  ICPU_RUN_KF(oscar_prepare_attention_tasks_kernel,4,starts.ptr,lens.ptr,slots.ptr,
      tasks.ptr,positions.ptr,int64_t{2},n,int64_t{6},heads,int64_t{4},int64_t{32},int64_t{1},true,static_cast<uint8_t*>(nullptr),int64_t{0},false);
  const int64_t expectedCounts[8]={1,-1,2,0,-1,-1,-1,-1};
  for(int64_t token=0;token<n;++token) {
    const auto* task=reinterpret_cast<const int64_t*>(tasks.ptr)+token*3*16;
    if(task[1]!=expectedCounts[token] || task[10]!=0)
      throw std::runtime_error("padding hole has overlapping or missing task owners");
  }
  std::cout<<"metadata padded request and changed-slot cases passed"<<std::endl;
}
void CheckSlotContext() {
  Gm starts(12),lens(8),slots(16),tasks(2*3*16*8),positions(16),table(2*8*4);
  const int32_t s[3]={0,1,2},l[2]={134,260};
  const int64_t slot[2]={17*128+2,5*128+1};
  int32_t bt[16]={9,17,0,0,0,0,0,0,7,8,5,0,0,0,0,0};
  std::memcpy(starts.ptr,s,12);std::memcpy(lens.ptr,l,8);
  std::memcpy(slots.ptr,slot,16);std::memcpy(table.ptr,bt,sizeof(bt));
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(oscar_prepare_attention_tasks_kernel,4,starts.ptr,lens.ptr,slots.ptr,
      tasks.ptr,positions.ptr,int64_t{2},int64_t{2},int64_t{6},int64_t{1},
      int64_t{4},int64_t{32},int64_t{1},true,table.ptr,int64_t{8},true);
  const int64_t expected[2]={130,257};
  for(int64_t r=0;r<2;++r) {
    int64_t pos;std::memcpy(&pos,positions.ptr+r*8,8);
    if(pos!=expected[r])throw std::runtime_error("MTP kept rejected tokens in true context");
    for(int64_t kind=0;kind<3;++kind) {
      auto* task=reinterpret_cast<const int64_t*>(tasks.ptr)+(r*3+kind)*16;
      if(task[8]!=expected[r] || task[10]!=0 || task[1]!=1)
        throw std::runtime_error("MTP task context was not derived from unique physical slot");
    }
  }
  for(int scenario=0;scenario<2;++scenario) {
    bt[8]=scenario==0?5:7;bt[10]=scenario==0?5:6;
    std::memcpy(table.ptr,bt,sizeof(bt));
    ICPU_RUN_KF(oscar_prepare_attention_tasks_kernel,4,starts.ptr,lens.ptr,slots.ptr,
        tasks.ptr,positions.ptr,int64_t{2},int64_t{2},int64_t{6},int64_t{1},
        int64_t{4},int64_t{32},int64_t{1},true,table.ptr,int64_t{8},true);
    auto* secondTask=reinterpret_cast<const int64_t*>(tasks.ptr)+3*16;
    int64_t pos;std::memcpy(&pos,positions.ptr+8,8);
    if(secondTask[10]!=5 || secondTask[1]!=-1 || pos!=-1)
      throw std::runtime_error("ambiguous/missing MTP slot did not fail closed");
  }
  std::cout<<"MTP slot-derived context, duplicate and missing page cases passed"<<std::endl;
}
void Close(const Gm& output,const std::string& path) {
  auto expected=Read(path,output.size);float maxError=0;
  for(size_t i=0;i<output.size/4;++i) {
    float a,b;std::memcpy(&a,output.ptr+4*i,4);std::memcpy(&b,expected.data()+4*i,4);
    if(a==b)continue;
    maxError=std::max(maxError,std::abs(a-b));
    if(!std::isfinite(a)||!std::isfinite(b)||std::abs(a-b)>0.005F+0.005F*std::abs(b))
      throw std::runtime_error("numeric mismatch: "+path+" index="+std::to_string(i)+
          " expected="+std::to_string(b)+" actual="+std::to_string(a));
  }
  std::cout<<"max_abs="<<maxError<<" file="<<path<<std::endl;
}
}
int main(int argc,char** argv) {
 try {
  if(argc!=3)throw std::runtime_error("usage: oscar_cv_cpu attention_cv case_directory");
  if(std::string(argv[1])=="tasks_causal") {CheckLargeCausalTasks();return 0;}
  CheckPaddedMetadata();
  CheckSlotContext();
  const std::string dir=argv[2];
  if(dir=="metadata") {std::cout<<"{\"backend\":\"ascendc_cpu_debug\",\"op\":\"attention_metadata\",\"status\":\"passed\"}"<<std::endl;return 0;}
  std::ifstream shape(dir+"/shape.txt");
  int64_t n,hq,hk,d,context,sink,recent,spec,b,nb,prefix,stride,cores;
  if(!(shape>>n>>hq>>hk>>d>>context>>sink>>recent>>spec>>b>>nb>>prefix>>stride>>cores))
    throw std::runtime_error("invalid shape file");
  int64_t requests=1,splits=1;shape>>requests;shape>>splits;
  if(splits<1 || splits>32)throw std::runtime_error("invalid split count");
  const int64_t segments=3*splits;
  const int64_t windowRows=sink+recent+spec,tasksCount=n*hk*segments;
  Gm q(dir+"/q.bin",n*hq*d*2),qr(dir+"/qr.bin",n*hq*d*4);
  Gm k(dir+"/ck.bin",n*hk*d*2),v(dir+"/cv.bin",n*hk*d*2),rv(dir+"/rv.bin",d*d*4);
  Gm raw(dir+"/raw.bin",prefix+nb*stride),table(dir+"/table.bin",requests*8*4);
  Gm wk(dir+"/wk.bin",nb*windowRows*hk*d*2),wv(dir+"/wv.bin",nb*windowRows*hk*d*2);
  Gm tags(dir+"/tags.bin",nb*windowRows*8);
  Gm starts(dir+"/starts.bin",(requests+1)*4),lens(dir+"/lens.bin",requests*4),slots(dir+"/slots.bin",n*8);
  Gm tasks(tasksCount*16*8),positions(n*8),partial(n*hq*segments*d*4),partLse(n*hq*segments*4);
  Gm status(tasksCount*2*4),workspace(cores*(256*d+4096)*4);
  Gm output(n*hq*d*4),lse(n*hq*4),mergeStatus(n*hq*4);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(oscar_prepare_attention_tasks_kernel,4,starts.ptr,lens.ptr,slots.ptr,
      tasks.ptr,positions.ptr,requests,n,hq,hk,sink,recent,splits,true,static_cast<uint8_t*>(nullptr),int64_t{0},false);
  for(int64_t t=0;t<n;++t) {
    int64_t pos;std::memcpy(&pos,positions.ptr+t*8,8);
    int64_t expected=context+t;
    if(requests>1) {auto expectedBytes=Read(dir+"/expected_positions.bin",n*8);
      std::memcpy(&expected,expectedBytes.data()+t*8,8);}
    if(pos!=expected)throw std::runtime_error("position mismatch");
  }
  // Each KV position belongs to exactly one split. All splits retain the
  // same query tile, so MTP queries share its one compressed-history read.
  for(int64_t token=0;token<n;++token)for(int64_t head=0;head<hk;++head)
  for(int64_t kind=0;kind<3;++kind) {
    const auto* first=reinterpret_cast<const int64_t*>(tasks.ptr)+
        ((token*hk+head)*3+kind)*splits*16;
    if(first[1]<=0)continue;
    int64_t end=first[3];
    for(int64_t split=0;split<splits;++split) {
      const auto* task=first+split*16;
      if(task[0]!=first[0] || task[1]!=first[1] || task[2]!=head || task[7]!=kind ||
          task[3]!=end || task[4]<task[3] || task[6]!=kind*splits+split)
        throw std::runtime_error("split ownership overlaps, has a gap or changes query tile");
      end=task[4];
    }
  }
  if(std::string(argv[1])=="bad_tag")std::memset(tags.ptr,0xff,tags.size);
  if(std::string(argv[1])=="bad_meta") {
    const int64_t offset=prefix+stride+sink*(d/2+8)+d/4;
    std::memset(raw.ptr+offset,0,2);
  }
  if(std::string(argv[1])=="nan_value") {
    const uint16_t nanBf16=0x7fc0;std::memcpy(v.ptr,&nanBf16,2);
  }
  if(std::string(argv[1])=="nan_query") {
    const uint16_t nanBf16=0x7fc0;const float nanFloat=std::nanf("");
    std::memcpy(q.ptr,&nanBf16,2);std::memcpy(qr.ptr,&nanFloat,4);
  }
  AscendC::SetKernelMode(KernelMode::MIX_MODE);
  ICPU_RUN_KF(oscar_attention_cv_kernel,cores,q.ptr,qr.ptr,k.ptr,v.ptr,rv.ptr,raw.ptr,
      table.ptr,wk.ptr,wv.ptr,tags.ptr,tasks.ptr,partial.ptr,partLse.ptr,status.ptr,
      workspace.ptr,n,hq,hk,d,requests,int64_t{8},tasksCount,b,nb,prefix,stride,
      windowRows*hk*d,windowRows,sink,recent,spec,splits,1.0F/std::sqrt(float(d)));
  if(std::string(argv[1])=="bad_tag" || std::string(argv[1])=="nan_query" ||
      std::string(argv[1])=="bad_meta" || std::string(argv[1])=="nan_value") {
    const int32_t wanted=std::string(argv[1])=="bad_tag"?4:(std::string(argv[1])=="bad_meta"?3:2);
    bool seen=false;
    for(int64_t i=0;i<tasksCount*2;++i) {int32_t code;std::memcpy(&code,status.ptr+i*4,4);
      if(code==wanted)seen=true;
      if(code!=0 && code!=wanted)throw std::runtime_error("unexpected error code");}
    if(!seen)throw std::runtime_error("invalid input did not surface in device status");
    std::cout<<"{\"backend\":\"ascendc_cpu_debug\",\"op\":\""<<argv[1]
             <<"\",\"status\":\"passed\"}"<<std::endl;
    return 0;
  }
  StatusZero(status);
  for(int64_t i=0;i<n*hq*segments*d;++i) {
    float x;std::memcpy(&x,partial.ptr+i*4,4);
    if(!std::isfinite(x))throw std::runtime_error("unwritten/nonfinite partial at "+std::to_string(i));
  }
  int64_t emptySegments=0;
  for(int64_t taskId=0;taskId<tasksCount;++taskId) {
    const auto* task=reinterpret_cast<const int64_t*>(tasks.ptr)+taskId*16;
    if(task[1]<=0 || task[3]!=task[4])continue;
    ++emptySegments;
    for(int64_t queryIndex=0;queryIndex<task[1];++queryIndex)
    for(int64_t gh=0;gh<hq/hk;++gh) {
      const int64_t row=((task[0]+queryIndex)*hq+task[2]*(hq/hk)+gh)*segments+task[6];
      float part_lse;std::memcpy(&part_lse,partLse.ptr+row*4,4);
      if(part_lse!=-INFINITY)throw std::runtime_error("empty split did not write -inf LSE");
      for(int64_t col=0;col<d;++col) {
        float value;std::memcpy(&value,partial.ptr+(row*d+col)*4,4);
        if(value!=0.0F)throw std::runtime_error("empty split did not zero partial");
      }
    }
  }
  std::cout<<"splits="<<splits<<" empty_segments="<<emptySegments<<std::endl;
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(oscar_merge_lse_kernel,4,partial.ptr,partLse.ptr,output.ptr,lse.ptr,
      mergeStatus.ptr,n*hq,segments,d);
  StatusZero(mergeStatus);Close(output,dir+"/expected_output.bin");Close(lse,dir+"/expected_lse.bin");
  std::cout<<"{\"backend\":\"ascendc_cpu_debug\",\"op\":\"attention_cv\",\"status\":\"passed\"}"<<std::endl;
  return 0;
 }catch(const std::exception& e){std::cerr<<"CPU_DEBUG_FAILED: "<<e.what()<<std::endl;return 1;}
}
