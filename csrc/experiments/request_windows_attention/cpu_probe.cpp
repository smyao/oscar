// Archive #126-129/#145/#154/#155 and D.4: isolated address/lifetime gate.
// All mathematical kernels are current decode-bundle clones. Official CPU
// comparison is not NPU/graph/performance acceptance; no full-history tensor.
#include "tikicpulib.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#define DIM_ARGS int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t, \
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,float
#define BASE_ARGS uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*, \
    uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*
extern "C" void oscar_attention_cv_decode_bundle_kernel(BASE_ARGS,DIM_ARGS);
extern "C" void oscar_attention_cv_decode_bundle_q1_kernel(BASE_ARGS,DIM_ARGS);
extern "C" void oscar_attention_request_window_decode_kernel(BASE_ARGS,uint8_t*,DIM_ARGS,int64_t,int64_t);
extern "C" void oscar_attention_request_window_decode_q1_kernel(BASE_ARGS,uint8_t*,DIM_ARGS,int64_t,int64_t);
#ifdef OSCAR_REQUEST_WINDOW_CLUSTER16
extern "C" void oscar_attention_window_range_cluster16_kernel(BASE_ARGS,uint8_t*,DIM_ARGS);
extern "C" void oscar_attention_request_window_cluster16_kernel(BASE_ARGS,uint8_t*,uint8_t*,DIM_ARGS,int64_t,int64_t);
#endif
extern "C" void oscar_prepare_attention_tasks_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,bool,uint8_t*,int64_t,bool);
extern "C" void oscar_merge_lse_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,int64_t,int64_t,int64_t);
#undef BASE_ARGS
#undef DIM_ARGS
namespace {
std::vector<uint8_t> Read(const std::string& path,size_t n) {
  std::vector<uint8_t> bytes(n);std::ifstream file(path,std::ios::binary);
  if(!file.read(reinterpret_cast<char*>(bytes.data()),n) || file.peek()!=EOF)
    throw std::runtime_error("incorrect golden byte count: "+path);
  return bytes;
}
void RequireMatchingTasks(const std::string& path,const uint8_t* actual,size_t bytes) {
  const auto saved=Read(path,bytes);
  if(std::memcmp(saved.data(),actual,bytes))
    throw std::runtime_error("reused baseline task domain differs from real preparation");
}
void Write(const std::string& path,const uint8_t* data,size_t n) {
  std::ofstream file(path,std::ios::binary|std::ios::trunc);
  if(!file.write(reinterpret_cast<const char*>(data),n) || !file.flush())
    throw std::runtime_error("cannot write export: "+path);
}
struct Gm {
  static constexpr size_t guard=64;
  uint8_t *base,*ptr;size_t size;
  explicit Gm(size_t n):base(static_cast<uint8_t*>(AscendC::GmAlloc(n+2*guard))),ptr(base?base+guard:nullptr),size(n) {
    if(!base)throw std::runtime_error("GmAlloc failed");
    std::memset(base,0xD7,n+2*guard);std::memset(ptr,0x85,n);
  }
  Gm(const std::string& path,size_t n):Gm(n){const auto bytes=Read(path,n);std::memcpy(ptr,bytes.data(),n);}
  ~Gm(){AscendC::GmFree(base);}
  Gm(const Gm&)=delete;Gm& operator=(const Gm&)=delete;
  void CheckGuard(const std::string& label) const {
    for(size_t i=0;i<guard;++i)if(base[i]!=0xD7 || ptr[size+i]!=0xD7)
      throw std::runtime_error("GM boundary guard changed: "+label);
  }
};
void CheckEqual(const Gm& old,const Gm& fresh,const char* label) {
  if(old.size!=fresh.size)throw std::runtime_error("A/B size mismatch");
  for(size_t i=0;i<old.size;++i)if(old.ptr[i]!=fresh.ptr[i])
    throw std::runtime_error(std::string(label)+" byte mismatch at "+std::to_string(i));
}
template<class T>T At(const uint8_t* ptr,int64_t index) {
  T value;std::memcpy(&value,ptr+index*sizeof(T),sizeof(T));return value;
}
void CheckOracle(const Gm& partial,const Gm& partLse,const std::string& dir,
    const Gm& slots,int64_t n,int64_t hq,int64_t d,int64_t segments) {
  float atol,rtol;std::ifstream tolerance(dir+"/tolerance.txt");
  if(!(tolerance>>atol>>rtol) || atol!=0.005F || rtol!=0.005F)
    throw std::runtime_error("frozen oracle tolerance missing");
  Gm output(n*hq*d*4),lse(n*hq*4),status(n*hq*4);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(oscar_merge_lse_kernel,4,partial.ptr,partLse.ptr,output.ptr,lse.ptr,status.ptr,n*hq,segments,d);
  output.CheckGuard("merged output");lse.CheckGuard("merged LSE");status.CheckGuard("merge status");
  const auto expectedOutput=Read(dir+"/expected_output.bin",output.size);
  const auto expectedLse=Read(dir+"/expected_lse.bin",lse.size);
  int64_t live=0;
  for(int64_t token=0;token<n;++token) {
    const bool active=At<int64_t>(slots.ptr,token)>=0;
    if(active)++live;
    for(int64_t head=0;head<hq;++head) {
      const int64_t row=token*hq+head;
      if(At<int32_t>(status.ptr,row))throw std::runtime_error("valid merge status is nonzero");
      const float actualLse=At<float>(lse.ptr,row),expectedLog=At<float>(expectedLse.data(),row);
      if(active && (!std::isfinite(actualLse) || std::abs(actualLse-expectedLog)>atol+rtol*std::abs(expectedLog)))
        throw std::runtime_error("LSE exceeds frozen PR oracle tolerance");
      if(!active && actualLse!=-INFINITY)throw std::runtime_error("padding LSE was not -inf");
      for(int64_t dim=0;dim<d;++dim) {
        const int64_t index=row*d+dim;
        const float actual=At<float>(output.ptr,index),expected=At<float>(expectedOutput.data(),index);
        if(!std::isfinite(actual) || std::abs(actual-expected)>atol+rtol*std::abs(expected))
          throw std::runtime_error("output exceeds frozen PR oracle tolerance");
        if(!active && actual!=0.0F)throw std::runtime_error("padding output was not zero");
      }
    }
  }
  if(!live)throw std::runtime_error("fixture has no live queries");
}
}

int main(int argc,char** argv) {
  try {
    bool reuseBaseline=false,baselineOnly=false,baselineOracleChecked=false;
#ifdef OSCAR_REQUEST_WINDOW_CLUSTER16
    reuseBaseline=argc==6 && std::string(argv[4])=="candidate-from";
    baselineOnly=argc==5 && std::string(argv[4])=="baseline-only";
    if((argc!=3 && argc!=4 && !reuseBaseline && !baselineOnly) || std::string(argv[1])!="c16")
      throw std::runtime_error("usage: c16 FIXTURE_DIR [EXPORT_DIR [baseline-only | candidate-from BASELINE_OUTPUT_DIR]]");
#else
    if((argc!=3 && argc!=4) || (std::string(argv[1])!="q4" && std::string(argv[1])!="q1"))
      throw std::runtime_error("usage: oscar_request_window_attention_cpu q4|q1 FIXTURE_DIR [EXPORT_DIR]");
#endif
    const std::string mode=argv[1],dir=argv[2];
    int64_t n,hq,hk,d,requests,columns,oldB,oldBlocks,newB,newBlocks,prefix,stride,capacity,bits;
    int64_t sink,recent,spec,splits,cores;
    std::ifstream shape(dir+"/shape.txt");
    if(!(shape>>n>>hq>>hk>>d>>requests>>columns>>oldB>>oldBlocks>>newB>>newBlocks>>prefix>>stride
         >>capacity>>bits>>sink>>recent>>spec>>splits>>cores) || n<=0 || d!=256 || hk!=1 || hq!=6 ||
         oldB<=0 || oldB%128 || newB!=2*oldB || oldBlocks<=0 || newBlocks<=0 || stride<oldB*hk*136 ||
         capacity<requests || bits<1 || bits>62 || splits!=1 || cores<1 || cores>32)
      throw std::runtime_error("invalid independent address fixture shape");
    int fault,mirrored,expectedCode;std::ifstream flags(dir+"/flags.txt");
    if(!(flags>>fault>>mirrored>>expectedCode) || (fault!=0 && fault!=1) || (mirrored!=0 && mirrored!=1) ||
         (fault?(expectedCode!=1 && expectedCode!=4):(expectedCode!=0 || mirrored)))
      throw std::runtime_error("invalid fault declaration");
#ifdef OSCAR_REQUEST_WINDOW_CLUSTER16
    if(fault || mirrored || expectedCode)throw std::runtime_error("C16 checkpoint requires the declared valid oracle domain");
#endif
    const int64_t segments=3*splits,taskCount=n*hk*segments,windowRows=sink+recent+spec;
    Gm q(dir+"/q.bin",n*hq*d*2),qr(dir+"/qr.bin",n*hq*d*4);
    Gm k(dir+"/ck.bin",n*hk*d*2),v(dir+"/cv.bin",n*hk*d*2),rv(dir+"/rv.bin",d*d*4);
    Gm starts(dir+"/starts.bin",(requests+1)*4),lens(dir+"/lens.bin",requests*4);
    Gm oldRaw(dir+"/old_raw.bin",prefix+oldBlocks*stride),newRaw(dir+"/new_raw.bin",prefix+2*newBlocks*stride);
    Gm oldTable(dir+"/old_table.bin",requests*columns*4),newTable(dir+"/new_table.bin",requests*columns*4);
    Gm oldSlots(dir+"/old_slots.bin",n*8),newSlots(dir+"/new_slots.bin",n*8);
    Gm oldK(dir+"/old_wk.bin",oldBlocks*windowRows*hk*d*2),oldV(dir+"/old_wv.bin",oldK.size);
    Gm newK(dir+"/new_wk.bin",capacity*windowRows*hk*d*2),newV(dir+"/new_wv.bin",newK.size);
    Gm oldTags(dir+"/old_tags.bin",oldBlocks*windowRows*8),newTags(dir+"/new_tags.bin",capacity*windowRows*8);
    Gm map(dir+"/request_map.bin",capacity*2*8);
    Gm oldTasks(taskCount*16*8),newTasks(oldTasks.size),oldPositions(n*8),newPositions(n*8);
    Gm oldPartial(n*hq*segments*d*4),newPartial(oldPartial.size),oldLse(n*hq*segments*4),newLse(oldLse.size);
#ifdef OSCAR_REQUEST_WINDOW_CLUSTER16
    const int64_t workspacePerCore=4997120;
    Gm oldStats(cores*8*8),newStats(oldStats.size);
#else
    const int64_t workspacePerCore=1146880;
#endif
    Gm oldStatus(taskCount*2*4),newStatus(oldStatus.size),oldWork(cores*workspacePerCore),newWork(oldWork.size);
    std::vector<std::pair<const char*,const Gm*>> buffers={
      {"q",&q},{"qr",&qr},{"k",&k},{"v",&v},{"rv",&rv},{"starts",&starts},{"lens",&lens},
      {"oldRaw",&oldRaw},{"newRaw",&newRaw},{"oldTable",&oldTable},{"newTable",&newTable},
      {"oldSlots",&oldSlots},{"newSlots",&newSlots},{"oldK",&oldK},{"oldV",&oldV},
      {"newK",&newK},{"newV",&newV},{"oldTags",&oldTags},{"newTags",&newTags},{"map",&map},
      {"oldTasks",&oldTasks},{"newTasks",&newTasks},{"oldPositions",&oldPositions},{"newPositions",&newPositions},
      {"oldPartial",&oldPartial},{"newPartial",&newPartial},{"oldLse",&oldLse},{"newLse",&newLse},
      {"oldStatus",&oldStatus},{"newStatus",&newStatus},{"oldWork",&oldWork},{"newWork",&newWork}};
#ifdef OSCAR_REQUEST_WINDOW_CLUSTER16
    buffers.push_back({"oldStats",&oldStats});buffers.push_back({"newStats",&newStats});
#endif
    auto guards=[&](){for(const auto& item:buffers)item.second->CheckGuard(item.first);};
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    const bool slotContext=mode=="q1";
#define PREPARE(table,slots,tasks,positions) \
    ICPU_RUN_KF(oscar_prepare_attention_tasks_kernel,4,starts.ptr,lens.ptr,slots.ptr,tasks.ptr,positions.ptr, \
        requests,n,hq,hk,sink,recent,splits,true,slotContext?table.ptr:static_cast<uint8_t*>(nullptr), \
        slotContext?columns:int64_t{0},slotContext)
    PREPARE(oldTable,oldSlots,oldTasks,oldPositions);
    PREPARE(newTable,newSlots,newTasks,newPositions);
#undef PREPARE
    CheckEqual(oldTasks,newTasks,"prepared metadata");CheckEqual(oldPositions,newPositions,"prepared positions");
    const auto positions=Read(dir+"/expected_positions.bin",oldPositions.size);
    if(std::memcmp(positions.data(),oldPositions.ptr,oldPositions.size))throw std::runtime_error("position oracle mismatch");
#ifdef OSCAR_REQUEST_WINDOW_CLUSTER16
    int suppress=0;std::ifstream suppression(dir+"/suppress_current.txt");
    if(!(suppression>>suppress) || suppress!=1 || n!=336 || recent!=1 || sink!=64 || requests!=1 || At<int32_t>(lens.ptr,0)-n!=65)
      throw std::runtime_error("C16 CPU gate requires explicit Q336/C65/R1 source0+1 scope");
    // Exact production _empty_current_source_ranges contract: only a healthy
    // source2 end becomes begin. Do not filter tasks or change qcount/status.
    for(auto* tasks:{&oldTasks,&newTasks})for(int64_t id=0;id<taskCount;++id) {
      const int64_t begin=At<int64_t>(tasks->ptr,id*16+3),end=At<int64_t>(tasks->ptr,id*16+4);
      if(At<int64_t>(tasks->ptr,id*16+7)==2 && At<int64_t>(tasks->ptr,id*16+10)==0 && end>=begin)
        std::memcpy(tasks->ptr+(id*16+4)*8,&begin,8);
    }
    CheckEqual(oldTasks,newTasks,"suppressed metadata");
#endif
    guards();
    // Include both independently prepared task tables in readonly ownership.
    const size_t inputCount=24;
    std::vector<std::vector<uint8_t>> copies;
    for(size_t i=0;i<inputCount;++i)copies.emplace_back(buffers[i].second->ptr,buffers[i].second->ptr+buffers[i].second->size);
    auto unchanged=[&](){for(size_t i=0;i<inputCount;++i)
      if(std::memcmp(copies[i].data(),buffers[i].second->ptr,copies[i].size()))
        throw std::runtime_error(std::string("readonly input modified: ")+buffers[i].first);};
    auto save=[&](const char* name,const Gm& part,const Gm& lse,const Gm& status){
      guards();unchanged();
      if(argc>=4) {
        const std::string base=std::string(argv[3])+"/"+name;
        Write(base+"_partial.bin",part.ptr,part.size);Write(base+"_lse.bin",lse.ptr,lse.size);
        Write(base+"_status.bin",status.ptr,status.size);
        Write(base+"_tasks.bin",oldTasks.ptr,oldTasks.size);
#ifdef OSCAR_REQUEST_WINDOW_CLUSTER16
        const auto& stats=std::string(name)=="old"?oldStats:newStats;
        Write(base+"_cluster_stats.bin",stats.ptr,stats.size);
#endif
      }
      std::cout<<"REQUEST_WINDOW_CPU_ARM_COMPLETE "<<name<<std::endl;
    };
    const float scale=1.0F/std::sqrt(float(d));
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
#define OLD_CALL(kernel) \
    ICPU_RUN_KF(kernel,cores,q.ptr,qr.ptr,k.ptr,v.ptr,rv.ptr,oldRaw.ptr,oldTable.ptr,oldK.ptr,oldV.ptr,oldTags.ptr, \
        oldTasks.ptr,oldPartial.ptr,oldLse.ptr,oldStatus.ptr,oldWork.ptr,n,hq,hk,d,requests,columns,taskCount, \
        oldB,oldBlocks,prefix,stride,windowRows*hk*d,windowRows,sink,recent,spec,splits,scale)
#define NEW_CALL(kernel) \
    ICPU_RUN_KF(kernel,cores,q.ptr,qr.ptr,k.ptr,v.ptr,rv.ptr,newRaw.ptr,newTable.ptr,newK.ptr,newV.ptr,newTags.ptr,map.ptr, \
        newTasks.ptr,newPartial.ptr,newLse.ptr,newStatus.ptr,newWork.ptr,n,hq,hk,d,requests,columns,taskCount, \
        newB,newBlocks,prefix,stride,windowRows*hk*d,windowRows,sink,recent,spec,splits,scale,capacity,bits)
#ifdef OSCAR_REQUEST_WINDOW_CLUSTER16
    if(reuseBaseline) {
      const std::string reference=argv[5];
      // A saved output is not proof of the original binary or of its oracle
      // gate. Match the actual task domain and revalidate it in this run.
      RequireMatchingTasks(reference+"/old_tasks.bin",oldTasks.ptr,oldTasks.size);
      for(auto item:{std::pair<Gm*,const char*>(&oldPartial,"old_partial.bin"),
                     std::pair<Gm*,const char*>(&oldLse,"old_lse.bin"),
                     std::pair<Gm*,const char*>(&oldStatus,"old_status.bin"),
                     std::pair<Gm*,const char*>(&oldStats,"old_cluster_stats.bin")}) {
        const auto saved=Read(reference+"/"+item.second,item.first->size);
        std::memcpy(item.first->ptr,saved.data(),saved.size());
      }
    }else {
      ICPU_RUN_KF(oscar_attention_window_range_cluster16_kernel,cores,q.ptr,qr.ptr,k.ptr,v.ptr,rv.ptr,oldRaw.ptr,oldTable.ptr,oldK.ptr,oldV.ptr,oldTags.ptr,
          oldTasks.ptr,oldPartial.ptr,oldLse.ptr,oldStatus.ptr,oldWork.ptr,oldStats.ptr,n,hq,hk,d,requests,columns,taskCount,
          oldB,oldBlocks,prefix,stride,windowRows*hk*d,windowRows,sink,recent,spec,splits,scale);
    }
    for(size_t word=0;word<oldStatus.size/4;++word)
      if(At<int32_t>(oldStatus.ptr,word))throw std::runtime_error("completed baseline status is nonzero");
    int64_t oldClusters=0,oldShared=0;
    for(int64_t core=0;core<cores;++core){oldClusters+=At<int64_t>(oldStats.ptr,core*8);oldShared+=At<int64_t>(oldStats.ptr,core*8+3);}
    if(oldClusters!=1 || oldShared!=1)throw std::runtime_error("completed baseline did not execute the declared C16 cluster");
    CheckOracle(oldPartial,oldLse,dir,oldSlots,n,hq,d,segments);
    baselineOracleChecked=true;guards();unchanged();
    if(reuseBaseline)std::cout<<"REQUEST_WINDOW_CPU_BASELINE_REVALIDATED source=completed_old_files original_binary_provenance=controller_report"<<std::endl;
    else save("old",oldPartial,oldLse,oldStatus);
    if(baselineOnly) {
      std::cout<<"{\"backend\":\"official_cpu_debug\",\"status\":\"baseline_checkpoint_passed\",\"mode\":\"c16\","
        "\"all_metadata_equal\":true,\"baseline_oracle_checked_this_run\":true,\"baseline_status_checked\":true,"
        "\"baseline_cluster_checked\":true,\"candidate_executed\":false,\"baseline_reused\":false,"
        "\"oracle_scope\":\"source0_and_source1_only\",\"input_immutable\":true,\"gm_guards\":true,\"npu\":\"not_run\"}"<<std::endl;
      return 0;
    }
    // The candidate may not mutate the witnessed baseline buffers either.
    std::vector<std::pair<const Gm*,std::vector<uint8_t>>> baselineWitness;
    for(const auto* buffer:{&oldPartial,&oldLse,&oldStatus,&oldStats})
      baselineWitness.emplace_back(buffer,std::vector<uint8_t>(buffer->ptr,buffer->ptr+buffer->size));
    AscendC::SetKernelMode(KernelMode::MIX_MODE); // baseline oracle used AIV merge.
    ICPU_RUN_KF(oscar_attention_request_window_cluster16_kernel,cores,q.ptr,qr.ptr,k.ptr,v.ptr,rv.ptr,newRaw.ptr,newTable.ptr,newK.ptr,newV.ptr,newTags.ptr,map.ptr,
        newTasks.ptr,newPartial.ptr,newLse.ptr,newStatus.ptr,newWork.ptr,newStats.ptr,n,hq,hk,d,requests,columns,taskCount,
        newB,newBlocks,prefix,stride,windowRows*hk*d,windowRows,sink,recent,spec,splits,scale,capacity,bits);
    save("new",newPartial,newLse,newStatus);
    for(const auto& witness:baselineWitness)
      if(std::memcmp(witness.first->ptr,witness.second.data(),witness.first->size))
        throw std::runtime_error("candidate modified the witnessed baseline outputs");
    CheckEqual(oldStats,newStats,"all cluster counters");
    int64_t clusters=0,shared=0;for(int64_t core=0;core<cores;++core){clusters+=At<int64_t>(newStats.ptr,core*8);shared+=At<int64_t>(newStats.ptr,core*8+3);}
    if(clusters!=1 || shared!=1)throw std::runtime_error("C16 fixture did not execute exactly one shared history tile");
#else
    if(mode=="q4") {OLD_CALL(oscar_attention_cv_decode_bundle_kernel);}
    else {OLD_CALL(oscar_attention_cv_decode_bundle_q1_kernel);}
    save("old",oldPartial,oldLse,oldStatus);
    if(mode=="q4") {NEW_CALL(oscar_attention_request_window_decode_kernel);}
    else {NEW_CALL(oscar_attention_request_window_decode_q1_kernel);}
    save("new",newPartial,newLse,newStatus);
#endif
#undef OLD_CALL
#undef NEW_CALL
    if(!fault || mirrored) {
      CheckEqual(oldPartial,newPartial,"all partial");CheckEqual(oldLse,newLse,"all LSE");
      CheckEqual(oldStatus,newStatus,"all status");
    }
    bool seenExpected=false;
    for(int64_t id=0;id<taskCount;++id) {
      const int64_t count=At<int64_t>(newTasks.ptr,id*16+1),kind=At<int64_t>(newTasks.ptr,id*16+7);
      for(int lane=0;lane<2;++lane) {
        const int32_t code=At<int32_t>(newStatus.ptr,id*2+lane);
        if(code && (!fault || kind!=1 || count<=0 || code!=expectedCode))
          throw std::runtime_error("unexpected candidate error domain/code");
        seenExpected|=code==expectedCode && code!=0;
        if((!fault || !mirrored) && At<int32_t>(oldStatus.ptr,id*2+lane))
          throw std::runtime_error("valid old baseline status nonzero");
        if(!mirrored && (kind!=1 || count<=0) && code!=At<int32_t>(oldStatus.ptr,id*2+lane))
          throw std::runtime_error("candidate fault escaped source1");
      }
    }
    if(bool(fault)!=seenExpected)throw std::runtime_error("fault status was not observed");
    if(fault && !mirrored) {
      // Map/range faults have no OLD map ABI. Prove source0/2 and padding
      // remain byte-identical, and every affected leader row is poisoned.
      for(int64_t id=0;id<taskCount;++id) {
        const int64_t begin=At<int64_t>(newTasks.ptr,id*16),count=At<int64_t>(newTasks.ptr,id*16+1);
        if(At<int64_t>(newTasks.ptr,id*16+7)!=1 || count<=0)continue;
        for(int64_t token=begin;token<begin+count;++token)for(int64_t head=0;head<hq;++head) {
          const int lane=static_cast<int>(((token-begin)*hq+head)/16);
          if(At<int32_t>(newStatus.ptr,id*2+lane)!=expectedCode)throw std::runtime_error("missing map/tag fault on live output owner");
          const int64_t row=(token*hq+head)*segments+1;
          if(!std::isnan(At<float>(newLse.ptr,row)))throw std::runtime_error("fault LSE is not NaN");
          for(int64_t dim=0;dim<d;++dim)if(!std::isnan(At<float>(newPartial.ptr,row*d+dim)))
            throw std::runtime_error("fault partial is not NaN");
        }
      }
      for(int64_t token=0;token<n;++token)for(int64_t head=0;head<hq;++head)for(int64_t source=0;source<3;++source) {
        if(source==1 && At<int64_t>(newSlots.ptr,token)>=0)continue;
        const int64_t row=(token*hq+head)*segments+source;
        if(std::memcmp(oldPartial.ptr+row*d*4,newPartial.ptr+row*d*4,d*4) ||
           std::memcmp(oldLse.ptr+row*4,newLse.ptr+row*4,4))
          throw std::runtime_error("unaffected source/padding changed on map/tag fault");
      }
    }
    if(!mirrored && !baselineOracleChecked)CheckOracle(oldPartial,oldLse,dir,oldSlots,n,hq,d,segments);
    if(!fault)CheckOracle(newPartial,newLse,dir,newSlots,n,hq,d,segments);
    guards();unchanged();
    std::cout<<"{\"backend\":\"official_cpu_debug\",\"status\":\"passed\",\"mode\":\""<<mode
      <<"\",\"all_metadata_equal\":true,\"all_ab_bytes_equal\":"<<((!fault||mirrored)?"true":"false")
      <<",\"candidate_oracle\":"<<(!fault?"true":"false")<<",\"fault_code\":"<<expectedCode
      <<",\"oracle_scope\":\""<<(mode=="c16"?"source0_and_source1_only":"complete_three_sources")<<"\""
      <<",\"cluster_stats_bitwise\":"<<(mode=="c16"?"true":"false")
      <<",\"baseline_reused\":"<<(reuseBaseline?"true":"false")
      <<",\"baseline_oracle_checked_this_run\":"<<(baselineOracleChecked?"true":"false")
      <<",\"candidate_executed\":true"
      <<",\"input_immutable\":true,\"gm_guards\":true,\"npu\":\"not_run\"}"<<std::endl;
    return 0;
  }catch(const std::exception& e){std::cerr<<"REQUEST_WINDOW_CPU_FAILED: "<<e.what()<<std::endl;return 1;}
}
