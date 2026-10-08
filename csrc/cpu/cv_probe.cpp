// SPDX-License-Identifier: Apache-2.0
// Archive G26-G34/#12/#13-20/#34/#36/#53-69/#92: execute the actual AscendC
// task/CV/merge kernel bodies under official tikicpulib with independent gold.
// D.4/#143: NaN-poisoned bounded workspace proves live rows do not depend on
// unwritten padded score/P rows; CPU-debug cannot establish target NPU timing.
// This checks numerical correctness only; CPU-debug cannot establish NPU
// device completion, graph capture/replay, timing or the 32K performance gate.
// Archive #129/#144: large task-table causal bounds use the production M/B
// geometry without a dense 16K oracle; shared workspace sizing stays exact.
// Archive #126/#129/#140-150 and D.4: cluster4/q1 modes execute both the fe0
// kernel and the separate experimental C4 kernel on identical bytes, compares
// partial/LSE/status exactly, then checks the independent frozen dense oracle.
#include "tikicpulib.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "../include/oscar_cv_schedule.h"
#include "../include/oscar_attention_launch.h"
extern "C" void oscar_prepare_attention_tasks_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,bool,uint8_t*,int64_t,bool);
extern "C" void oscar_attention_cv_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,float);
extern "C" void oscar_attention_cv_q1_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,float);
extern "C" void oscar_attention_cv_fast_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,float);
extern "C" void oscar_attention_cv_fast_q1_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,float);
extern "C" void oscar_attention_cv_profile_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    uint8_t*,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,float);
extern "C" void oscar_fast_unpack_words_kernel(uint8_t*);
extern "C" void oscar_attention_cv_cluster4_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    uint8_t*,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,float);
extern "C" void oscar_attention_cv_unified_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    uint8_t*,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,
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
    const oscar_ascend_schedule::CvTaskSchedule schedule{
        n,oscar_ascend::kAttentionQueryRows/(hq/hk),hk*3};
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
      tiles+=(part[4]-part[3]+oscar_ascend::kAttentionKvRows-1)/
          oscar_ascend::kAttentionKvRows;end=part[4];
    }
    if(end!=expectedEnd)throw std::runtime_error("causal current source misses visible tokens");
    oldTiles+=(s[request+1]-begin+oscar_ascend::kAttentionKvRows-1)/
        oscar_ascend::kAttentionKvRows;
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
  if(argc!=3)throw std::runtime_error("usage: oscar_cv_cpu mode case_directory");
  const std::string mode=argv[1];
  if(mode=="fast_words") {
    // #17-20/#151: execute the real AscendC unpack helper on all 65536
    // possible uint16 words and compare all 8 LSB-first 2-bit lanes exactly.
    constexpr int64_t words=65536,values=words*8;
    Gm expanded(values*4);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(oscar_fast_unpack_words_kernel,16,expanded.ptr);
    for(int64_t batch=0;batch<128;++batch)
    for(int32_t row=0;row<16;++row)
    for(int32_t col=0;col<256;++col) {
      const int64_t wordIndex=row*32+col/8;
      const uint16_t word=static_cast<uint16_t>(batch*512+wordIndex);
      const float expected=static_cast<float>((word>>(2*(col%8)))&3);
      const int64_t outputIndex=batch*4096+row*256+col;
      float actual;std::memcpy(&actual,expanded.ptr+outputIndex*4,4);
      if(actual!=expected)
        throw std::runtime_error("fast INT2 word mismatch word="+
            std::to_string(word)+" lane="+std::to_string(col%8));
    }
    std::cout<<"{\"backend\":\"ascendc_cpu_debug\",\"op\":\"fast_words\",\"status\":\"passed\"}"<<std::endl;
    return 0;
  }
  const bool clusterMode=mode=="cluster4" || mode=="cluster4_required" ||
      mode=="cluster4_bad_meta" || mode=="cluster4_nan_query" ||
      mode=="fast_cluster4" || mode=="fast_cluster4_poison" ||
      mode=="fast_cluster4_error";
  const bool fastCluster=mode=="fast_cluster4" || mode=="fast_cluster4_poison" ||
      mode=="fast_cluster4_error";
  const bool q1Mode=mode=="q1_schedule" || mode=="q1_schedule_bad_meta" ||
      mode=="fast_q1";
  const bool profileMode=mode=="profile_fe0";
  const bool fastFe0=mode=="fast_fe0" || mode=="fast_fe0_dead" ||
      mode=="fast_fe0_error";
  if(mode=="tasks_causal") {CheckLargeCausalTasks();return 0;}
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
  Gm status(tasksCount*2*4),workspace(cores*oscar_ascend::attention_workspace_per_core(d));
  Gm output(n*hq*d*4),lse(n*hq*4),mergeStatus(n*hq*4);
  if(mode=="poison_workspace") {
    // The full per-Cube GM arena starts as NaN. Every consumed live Q/K/V/P
    // value must be published in this invocation, including M tails; padding
    // rows are intentionally left poisoned by the short-query P optimization.
    const uint32_t poison=0x7fc00000U;
    for(size_t offset=0;offset<workspace.size;offset+=sizeof(poison))
      std::memcpy(workspace.ptr+offset,&poison,sizeof(poison));
  }
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
  if(mode=="cluster4_bad_meta") {
    if(n<168 || context<=100 || hk!=1)
      throw std::runtime_error("cluster4_bad_meta requires the mature q168/hk1 fixture");
    const int64_t offset=prefix+stride+100*(d/2+8)+d/4;
    std::memset(raw.ptr+offset,0,2); // live K scale in shared history.
  }
  if(mode=="cluster4_nan_query") {
    if(n<168 || hk!=1)
      throw std::runtime_error("cluster4_nan_query requires the mature q168/hk1 fixture");
    const float nanFloat=std::nanf("");
    const int64_t offset=(105*hq)*d*4; // grouped leader 1, head 0 only.
    std::memcpy(qr.ptr+offset,&nanFloat,4);
  }
  if(mode=="q1_schedule_bad_meta") {
    std::ifstream source(dir+"/bad_meta_offset.txt");int64_t offset=-1;
    if(!(source>>offset) || offset<0 || offset+2>static_cast<int64_t>(raw.size))
      throw std::runtime_error("invalid q1 bad_meta_offset.txt");
    std::memset(raw.ptr+offset,0,2);
  }
  AscendC::SetKernelMode(KernelMode::MIX_MODE);
  if(!fastCluster) {
    ICPU_RUN_KF(oscar_attention_cv_kernel,cores,q.ptr,qr.ptr,k.ptr,v.ptr,rv.ptr,raw.ptr,
        table.ptr,wk.ptr,wv.ptr,tags.ptr,tasks.ptr,partial.ptr,partLse.ptr,status.ptr,
        workspace.ptr,n,hq,hk,d,requests,int64_t{8},tasksCount,b,nb,prefix,stride,
        windowRows*hk*d,windowRows,sink,recent,spec,splits,1.0F/std::sqrt(float(d)));
  }
  if(fastFe0) {
    Gm fastPartial(partial.size),fastLse(partLse.size),
       fastStatus(status.size),fastWorkspace(workspace.size);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_RUN_KF(oscar_attention_cv_fast_kernel,cores,q.ptr,qr.ptr,k.ptr,v.ptr,rv.ptr,raw.ptr,
        table.ptr,wk.ptr,wv.ptr,tags.ptr,tasks.ptr,fastPartial.ptr,fastLse.ptr,
        fastStatus.ptr,fastWorkspace.ptr,n,hq,hk,d,requests,int64_t{8},
        tasksCount,b,nb,prefix,stride,windowRows*hk*d,windowRows,sink,recent,
        spec,splits,1.0F/std::sqrt(float(d)));
    if(fastStatus.size!=status.size ||
        std::memcmp(fastStatus.ptr,status.ptr,status.size)!=0)
      throw std::runtime_error("fast fe0 status differs bytewise");
    if(mode=="fast_fe0_error") {
      bool sawError=false;
      for(int64_t i=0;i<tasksCount*2;++i) {
        int32_t code;std::memcpy(&code,status.ptr+i*4,4);
        if(code!=0)sawError=true;
      }
      if(!sawError)throw std::runtime_error("fast malformed live metadata was ignored");
      for(const auto* pair:{&fastPartial,&fastLse}) {
        const Gm& baseline=pair==&fastPartial?partial:partLse;
        for(size_t i=0;i<pair->size/4;++i) {
          float a,b;std::memcpy(&a,pair->ptr+i*4,4);
          std::memcpy(&b,baseline.ptr+i*4,4);
          if((std::isnan(a)!=std::isnan(b)) ||
              (std::isfinite(a)!=std::isfinite(b)) ||
              (std::isfinite(a) && a!=b))
            throw std::runtime_error("fast invalid output classification differs from fe0");
        }
      }
      std::cout<<"{\"backend\":\"ascendc_cpu_debug\",\"op\":\"fast_fe0_error\",\"status\":\"passed\"}"<<std::endl;
      return 0;
    }
    for(const auto* pair:{&fastPartial,&fastLse}) {
      const Gm& baseline=pair==&fastPartial?partial:partLse;
      if(pair->size!=baseline.size || std::memcmp(pair->ptr,baseline.ptr,pair->size)!=0)
        throw std::runtime_error("fast fe0 valid partial/LSE differs bytewise");
    }
  }
  if(profileMode) {
    constexpr int64_t engines=oscar_ascend::kAttentionProfileEngines;
    constexpr int64_t sources=oscar_ascend::kAttentionProfileSources;
    constexpr int64_t fields=oscar_ascend::kAttentionProfileFields;
    Gm measuredPartial(partial.size),measuredLse(partLse.size),
       measuredStatus(status.size),measuredWorkspace(workspace.size),
       ticks(cores*engines*sources*fields*sizeof(int64_t));
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_RUN_KF(oscar_attention_cv_profile_kernel,cores,q.ptr,qr.ptr,k.ptr,v.ptr,rv.ptr,raw.ptr,
        table.ptr,wk.ptr,wv.ptr,tags.ptr,tasks.ptr,measuredPartial.ptr,measuredLse.ptr,
        measuredStatus.ptr,measuredWorkspace.ptr,ticks.ptr,n,hq,hk,d,requests,
        int64_t{8},tasksCount,b,nb,prefix,stride,windowRows*hk*d,windowRows,
        sink,recent,spec,splits,1.0F/std::sqrt(float(d)));
    for(const auto* pair:{&measuredPartial,&measuredLse,&measuredStatus}) {
      const Gm& baseline=pair==&measuredPartial?partial:(pair==&measuredLse?partLse:status);
      if(pair->size!=baseline.size || std::memcmp(pair->ptr,baseline.ptr,pair->size)!=0) {
        size_t first=0;while(first<pair->size && pair->ptr[first]==baseline.ptr[first])++first;
        throw std::runtime_error("profile differs from fe0 bytewise at output byte "+
            std::to_string(first));
      }
    }
    int64_t allTasks=0,historyUnits=0;
    const auto readTick=[&](int64_t core,int64_t engine,int64_t source,int64_t field){
      int64_t x;const int64_t offset=(((core*engines+engine)*sources+source)*fields+field)*8;
      std::memcpy(&x,ticks.ptr+offset,8);return x;
    };
    for(int64_t core=0;core<cores;++core)for(int64_t engine=0;engine<engines;++engine) {
      if(readTick(core,engine,3,23)<=0)
        throw std::runtime_error("profile actor compute span was not published");
      for(int64_t field=0;field<23;++field) {
        int64_t sum=0;
        for(int64_t source=0;source<3;++source) {
          const int64_t x=readTick(core,engine,source,field);
          if(x<0)throw std::runtime_error("negative profile source counter");
          sum+=x;
        }
        if(sum!=readTick(core,engine,3,field))
          throw std::runtime_error("profile source totals are inconsistent");
      }
      for(int64_t source=0;source<3;++source)
        if(readTick(core,engine,source,23)!=0)
          throw std::runtime_error("actor span must be stored only in total source");
      if(engine==0) {
        allTasks+=readTick(core,engine,3,0);
        historyUnits+=readTick(core,engine,0,1);
      }
    }
    if(allTasks==0 || historyUnits==0)
      throw std::runtime_error("profile fixture did not exercise history CV");
    std::cout<<"profile_tasks="<<allTasks<<" history_units="<<historyUnits<<std::endl;
  }
  if(q1Mode) {
    // The producer's qcount/rows and all numerical inputs are unchanged.
    // One-token work items must cover the identical task table exactly once.
    const int64_t perToken=hk*segments;
    int64_t oldLeaders=0,newLeaders=0;
    std::vector<int64_t> oldCore(cores),newCore(cores);
    for(int32_t shape=0;shape<2;++shape) {
      const int64_t tile=shape==0?oscar_ascend::kAttentionQueryRows/(hq/hk):1;
      const oscar_ascend_schedule::CvTaskSchedule schedule{n,tile,perToken};
      for(int64_t workId=0;workId<schedule.WorkItems();++workId) {
        const int64_t segment=schedule.Segment(workId);
        if(segment%(3*splits)>=splits)continue; // source0 only
        const int64_t begin=schedule.TokenBegin(workId);
        for(int64_t token=begin;token<std::min(n,begin+tile);++token) {
          const int64_t id=schedule.TaskId(workId,token);
          const auto* row=reinterpret_cast<const int64_t*>(tasks.ptr)+id*16;
          if(row[1]>0) {
            if(shape==0) {++oldLeaders;++oldCore[workId%cores];}
            else {++newLeaders;++newCore[workId%cores];}
          }
        }
      }
    }
    if(oldLeaders!=newLeaders)
      throw std::runtime_error("q1 schedule changed source0 leader count");
    const int64_t oldActive=std::count_if(oldCore.begin(),oldCore.end(),[](int64_t x){return x>0;});
    const int64_t newActive=std::count_if(newCore.begin(),newCore.end(),[](int64_t x){return x>0;});
    if(n==128 && requests==32 && splits==3 && hk==1 && cores==20 &&
        (oldActive!=6 || newActive!=20))
      throw std::runtime_error("q1 padded128/S3 owner spread differs from the device-work proof");
    std::cout<<"q1_source0_leaders="<<newLeaders<<" old_active_cores="<<oldActive
             <<" new_active_cores="<<newActive<<std::endl;
    Gm candidatePartial(partial.size),candidateLse(partLse.size),
       candidateStatus(status.size),candidateWorkspace(workspace.size);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_RUN_KF(oscar_attention_cv_q1_kernel,cores,q.ptr,qr.ptr,k.ptr,v.ptr,rv.ptr,raw.ptr,
        table.ptr,wk.ptr,wv.ptr,tags.ptr,tasks.ptr,candidatePartial.ptr,candidateLse.ptr,
        candidateStatus.ptr,candidateWorkspace.ptr,n,hq,hk,d,requests,int64_t{8},
        tasksCount,b,nb,prefix,stride,windowRows*hk*d,windowRows,sink,recent,
        spec,splits,1.0F/std::sqrt(float(d)));
    for(const auto* pair:{&candidatePartial,&candidateLse,&candidateStatus}) {
      const Gm& baseline=pair==&candidatePartial?partial:(pair==&candidateLse?partLse:status);
      if(pair->size!=baseline.size || std::memcmp(pair->ptr,baseline.ptr,pair->size)!=0) {
        size_t first=0;while(first<pair->size && pair->ptr[first]==baseline.ptr[first])++first;
        throw std::runtime_error("q1 schedule differs from fe0 bytewise at output byte "+
            std::to_string(first));
      }
    }
    if(mode=="fast_q1") {
      Gm fastPartial(partial.size),fastLse(partLse.size),
         fastStatus(status.size),fastWorkspace(workspace.size);
      AscendC::SetKernelMode(KernelMode::MIX_MODE);
      ICPU_RUN_KF(oscar_attention_cv_fast_q1_kernel,cores,q.ptr,qr.ptr,k.ptr,v.ptr,rv.ptr,
          raw.ptr,table.ptr,wk.ptr,wv.ptr,tags.ptr,tasks.ptr,fastPartial.ptr,
          fastLse.ptr,fastStatus.ptr,fastWorkspace.ptr,n,hq,hk,d,requests,
          int64_t{8},tasksCount,b,nb,prefix,stride,windowRows*hk*d,windowRows,
          sink,recent,spec,splits,1.0F/std::sqrt(float(d)));
      for(const auto* pair:{&fastPartial,&fastLse,&fastStatus}) {
        const Gm& baseline=pair==&fastPartial?candidatePartial:
            (pair==&fastLse?candidateLse:candidateStatus);
        if(pair->size!=baseline.size || std::memcmp(pair->ptr,baseline.ptr,pair->size)!=0)
          throw std::runtime_error("fast q1 differs bytewise from old q1");
      }
    }
    if(mode=="q1_schedule_bad_meta") {
      bool sawError=false;
      for(int64_t id=0;id<tasksCount;++id)for(int32_t lane=0;lane<2;++lane) {
        int32_t code;std::memcpy(&code,status.ptr+(id*2+lane)*4,4);
        if(code==2 || code==3)sawError=true;
      }
      if(!sawError)throw std::runtime_error("q1 invalid live metadata did not surface");
      std::cout<<"{\"backend\":\"ascendc_cpu_debug\",\"op\":\"q1_schedule_bad_meta\",\"status\":\"passed\"}"<<std::endl;
      return 0;
    }
    StatusZero(status);
    for(int64_t id=0;id<tasksCount;++id) {
      const auto* row=reinterpret_cast<const int64_t*>(tasks.ptr)+id*16;
      if(row[1]<=0 || row[7]!=0 || row[3]!=row[4])continue;
      for(int64_t gh=0;gh<hq/hk;++gh) {
        const int64_t outputRow=((row[0]*hq+row[2]*(hq/hk)+gh)*segments+row[6]);
        float value;std::memcpy(&value,partLse.ptr+outputRow*4,4);
        if(value!=-INFINITY)throw std::runtime_error("q1 empty history lost -inf LSE");
        for(int64_t col=0;col<d;++col) {
          std::memcpy(&value,partial.ptr+(outputRow*d+col)*4,4);
          if(value!=0.0F)throw std::runtime_error("q1 empty history lost zero partial");
        }
      }
    }
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(oscar_merge_lse_kernel,4,partial.ptr,partLse.ptr,output.ptr,lse.ptr,
        mergeStatus.ptr,n*hq,segments,d);
    const auto expectedOutput=Read(dir+"/expected_output.bin",output.size);
    const auto expectedLse=Read(dir+"/expected_lse.bin",lse.size);
    int64_t live=0;
    for(int64_t token=0;token<n;++token) {
      int64_t slot;std::memcpy(&slot,slots.ptr+token*8,8);
      if(slot<0)continue;
      ++live;
      for(int64_t head=0;head<hq;++head) {
        const int64_t row=token*hq+head;
        int32_t mergeCode;std::memcpy(&mergeCode,mergeStatus.ptr+row*4,4);
        if(mergeCode)throw std::runtime_error("q1 live merge status is nonzero");
        float actualLse,referenceLse;
        std::memcpy(&actualLse,lse.ptr+row*4,4);
        std::memcpy(&referenceLse,expectedLse.data()+row*4,4);
        if(!std::isfinite(actualLse) ||
            std::abs(actualLse-referenceLse)>0.005F+0.005F*std::abs(referenceLse))
          throw std::runtime_error("q1 live LSE differs from frozen oracle");
        for(int64_t col=0;col<d;++col) {
          float actual,reference;const int64_t index=row*d+col;
          std::memcpy(&actual,output.ptr+index*4,4);
          std::memcpy(&reference,expectedOutput.data()+index*4,4);
          if(!std::isfinite(actual) ||
              std::abs(actual-reference)>0.005F+0.005F*std::abs(reference))
            throw std::runtime_error("q1 live output differs from frozen oracle");
        }
      }
    }
    if(live==0)throw std::runtime_error("q1 fixture has no live token");
    std::cout<<"q1_live_tokens="<<live<<std::endl;
    std::cout<<"{\"backend\":\"ascendc_cpu_debug\",\"op\":\""
             <<(mode=="fast_q1"?"fast_q1":"q1_schedule")
             <<"\",\"status\":\"passed\"}"<<std::endl;
    return 0;
  }
  if(clusterMode) {
    Gm candidatePartial(partial.size),candidateLse(partLse.size),
       candidateStatus(status.size),candidateWorkspace(
           cores*oscar_ascend::attention_cluster4_workspace_per_core(d)),
       stats(cores*8*sizeof(int64_t));
    if(mode=="fast_cluster4_poison") {
      const uint32_t poison=0x7fc00000U;
      for(size_t offset=0;offset<candidateWorkspace.size;offset+=4)
        std::memcpy(candidateWorkspace.ptr+offset,&poison,4);
    }
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_RUN_KF(oscar_attention_cv_cluster4_kernel,cores,q.ptr,qr.ptr,k.ptr,v.ptr,rv.ptr,raw.ptr,
        table.ptr,wk.ptr,wv.ptr,tags.ptr,tasks.ptr,candidatePartial.ptr,candidateLse.ptr,
        candidateStatus.ptr,candidateWorkspace.ptr,stats.ptr,n,hq,hk,d,requests,
        int64_t{8},tasksCount,b,nb,prefix,stride,windowRows*hk*d,windowRows,
        sink,recent,spec,splits,1.0F/std::sqrt(float(d)));
    if(!fastCluster) {
      for(const auto* pair: {&candidatePartial,&candidateLse,&candidateStatus}) {
        const Gm& baseline=pair==&candidatePartial?partial:(pair==&candidateLse?partLse:status);
        if(pair->size!=baseline.size || std::memcmp(pair->ptr,baseline.ptr,pair->size)!=0) {
          size_t first=0;while(first<pair->size && pair->ptr[first]==baseline.ptr[first])++first;
          throw std::runtime_error("cluster4 differs from fe0 bytewise at output byte "+
              std::to_string(first));
        }
      }
    }
    int64_t totals[8]={};
    for(int64_t core=0;core<cores;++core)for(int32_t field=0;field<8;++field) {
      int64_t x;std::memcpy(&x,stats.ptr+(core*8+field)*8,8);
      if(x<0)throw std::runtime_error("negative cluster4 diagnostic counter");
      totals[field]+=x;
    }
    if(totals[1]!=4*totals[0] || totals[4]!=3*totals[3] ||
        totals[6]!=totals[1] || totals[7]!=totals[0])
      throw std::runtime_error("cluster4 counters violate ownership/reuse identities");
    if(mode!="cluster4" && mode!="fast_cluster4" &&
        (totals[0]==0 || totals[3]==0))
      throw std::runtime_error("cluster4 fixture did not execute a shared history tile");
    std::cout<<"cluster4_clusters="<<totals[0]<<" grouped="<<totals[1]
             <<" solo="<<totals[2]<<" shared_kv_tiles="<<totals[3]
             <<" avoided_kv_loads="<<totals[4]<<std::endl;
    if(fastCluster) {
      Gm fastPartial(partial.size),fastLse(partLse.size),
         fastStatus(status.size),fastWorkspace(candidateWorkspace.size),
         fastStats(stats.size);
      if(mode=="fast_cluster4_poison") {
        const uint32_t poison=0x7fc00000U;
        for(size_t offset=0;offset<fastWorkspace.size;offset+=4)
          std::memcpy(fastWorkspace.ptr+offset,&poison,4);
      }
      AscendC::SetKernelMode(KernelMode::MIX_MODE);
      ICPU_RUN_KF(oscar_attention_cv_unified_kernel,cores,q.ptr,qr.ptr,k.ptr,v.ptr,
          rv.ptr,raw.ptr,table.ptr,wk.ptr,wv.ptr,tags.ptr,tasks.ptr,fastPartial.ptr,
          fastLse.ptr,fastStatus.ptr,fastWorkspace.ptr,fastStats.ptr,n,hq,hk,d,
          requests,int64_t{8},tasksCount,b,nb,prefix,stride,windowRows*hk*d,
          windowRows,sink,recent,spec,splits,1.0F/std::sqrt(float(d)));
      if(std::memcmp(fastStatus.ptr,candidateStatus.ptr,status.size)!=0 ||
          std::memcmp(fastStats.ptr,stats.ptr,stats.size)!=0)
        throw std::runtime_error("fast C4 status/counters differ from old C4");
      if(mode=="fast_cluster4_error") {
        bool sawError=false;
        for(int64_t id=0;id<tasksCount*2;++id) {
          int32_t code;std::memcpy(&code,candidateStatus.ptr+id*4,4);
          if(code)sawError=true;
        }
        if(!sawError)throw std::runtime_error("fast C4 invalid metadata was ignored");
        std::cout<<"{\"backend\":\"ascendc_cpu_debug\",\"op\":\"fast_cluster4_error\",\"status\":\"passed\"}"<<std::endl;
        return 0;
      }
      for(const auto* pair:{&fastPartial,&fastLse}) {
        const Gm& baseline=pair==&fastPartial?candidatePartial:candidateLse;
        if(std::memcmp(pair->ptr,baseline.ptr,pair->size)!=0)
          throw std::runtime_error("fast C4 valid partial/LSE differs bytewise");
      }
      std::memcpy(partial.ptr,candidatePartial.ptr,partial.size);
      std::memcpy(partLse.ptr,candidateLse.ptr,partLse.size);
      std::memcpy(status.ptr,candidateStatus.ptr,status.size);
    }
  }
  if(mode=="cluster4_bad_meta" || mode=="cluster4_nan_query") {
    const int32_t wanted=mode=="cluster4_bad_meta"?3:2;
    const int64_t taskId=105*hk*segments; // mature grouped leader, source0.
    int32_t left=0,right=0;
    std::memcpy(&left,status.ptr+(taskId*2)*4,4);
    std::memcpy(&right,status.ptr+(taskId*2+1)*4,4);
    if(left!=wanted && right!=wanted)
      throw std::runtime_error("cluster4 error fixture did not reach grouped leader");
    std::cout<<"{\"backend\":\"ascendc_cpu_debug\",\"op\":\""
             <<mode<<"\",\"status\":\"passed\"}"<<std::endl;
    return 0;
  }
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
  std::cout<<"{\"backend\":\"ascendc_cpu_debug\",\"op\":\""
           <<((fastFe0 || fastCluster)?mode:
              (mode=="poison_workspace"?"poison_workspace":
               (profileMode?"profile_fe0":"attention_cv")))
           <<"\",\"status\":\"passed\"}"<<std::endl;
  return 0;
 }catch(const std::exception& e){std::cerr<<"CPU_DEBUG_FAILED: "<<e.what()<<std::endl;return 1;}
}
