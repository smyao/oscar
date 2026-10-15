// Archive #126/#129/#140/#145/#148/#155: current production SIMD A/B.
// D.4: window source within FIA; full-history restore formerly cost6.5s.
// Both real kernels consume the same bounded task/input buffers and cores;
// every partial/LSE/status byte is compared before the independent PR oracle.
// Official CPU-debug and export for CAModel are not NPU/model performance.
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
#include "../../include/oscar_attention_launch.h"

#ifndef OSCAR_CANDIDATE_Q4_KERNEL
#define OSCAR_CANDIDATE_Q4_KERNEL oscar_attention_cv_window_batch_decode_kernel
#endif
#ifndef OSCAR_CANDIDATE_Q1_KERNEL
#define OSCAR_CANDIDATE_Q1_KERNEL oscar_attention_cv_window_batch_q1_kernel
#endif
#ifndef OSCAR_CANDIDATE_WORKSPACE_BYTES_PER_CORE
#define OSCAR_CANDIDATE_WORKSPACE_BYTES_PER_CORE 0
#endif
#ifndef OSCAR_CANDIDATE_LABEL
#define OSCAR_CANDIDATE_LABEL "window_batch"
#endif

#define CV_ARGS uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*, \
    uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*, \
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t, \
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,float
extern "C" void oscar_decode_finite_q4_baseline_kernel(CV_ARGS);
extern "C" void oscar_decode_finite_q1_baseline_q1_kernel(CV_ARGS);
extern "C" void OSCAR_CANDIDATE_Q4_KERNEL(CV_ARGS);
extern "C" void OSCAR_CANDIDATE_Q1_KERNEL(CV_ARGS);
#undef CV_ARGS
extern "C" void oscar_prepare_attention_tasks_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,bool,uint8_t*,int64_t,bool);
extern "C" void oscar_merge_lse_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    int64_t,int64_t,int64_t);

namespace {
std::vector<uint8_t> Read(const std::string& path,size_t n) {
  std::vector<uint8_t> bytes(n);std::ifstream file(path,std::ios::binary);
  if(!file.read(reinterpret_cast<char*>(bytes.data()),n) || file.peek()!=EOF)
    throw std::runtime_error("incorrect golden byte count: "+path);
  return bytes;
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
void CheckStatuses(const Gm& status,const std::string& dir,bool invalid) {
  bool nonzero=false;
  for(size_t i=0;i<status.size/4;++i)nonzero|=At<int32_t>(status.ptr,i)!=0;
  if(nonzero!=invalid)throw std::runtime_error("declared valid/fault status domain differs");
  std::ifstream checks(dir+"/required_status_checks.bin",std::ios::binary|std::ios::ate);
  const std::streamoff bytes=checks.tellg();
  if(bytes<0 || bytes%24 || (invalid && bytes==0))throw std::runtime_error("fault status-order checks missing");
  const auto data=Read(dir+"/required_status_checks.bin",static_cast<size_t>(bytes));
  for(int64_t offset=0;offset<bytes/8;offset+=3) {
    const int64_t task=At<int64_t>(data.data(),offset),lane=At<int64_t>(data.data(),offset+1);
    const int64_t expected=At<int64_t>(data.data(),offset+2);
    if(task<0 || lane<0 || lane>1 || (task*2+lane)*4>=static_cast<int64_t>(status.size))
      throw std::runtime_error("required status index is invalid");
    if(At<int32_t>(status.ptr,task*2+lane)!=expected)
      throw std::runtime_error("native window error order changed at task/lane "+std::to_string(task)+"/"+std::to_string(lane));
  }
}
}

int main(int argc,char** argv) {
  try {
    const bool reuseBaseline=argc==6 && std::string(argv[4])=="candidate-from";
    if((argc!=3 && argc!=4 && argc!=5 && !reuseBaseline) ||
        (argc==5 && std::string(argv[4])!="prepare-only") ||
        (std::string(argv[1])!="q4" && std::string(argv[1])!="q1"))
      throw std::runtime_error("usage: oscar_window_batch_decode_cpu q4|q1 GOLDEN_DIR [EXPORT_DIR [prepare-only | candidate-from REFERENCE_DIR]]");
    const std::string mode=argv[1],dir=argv[2];
    int64_t n,hq,hk,d,context,sink,recent,spec,b,blocks,prefix,stride,cores,requests,splits,columns;
    std::ifstream shape(dir+"/shape.txt");
    if(!(shape>>n>>hq>>hk>>d>>context>>sink>>recent>>spec>>b>>blocks>>prefix>>stride>>cores>>requests>>splits>>columns)
        || n<=0 || d!=256 || hk<=0 || hq%hk || hq/hk>8 || splits<1 || cores<1 || columns<1)
      throw std::runtime_error("invalid frozen window fixture geometry");
    int fault;std::ifstream flags(dir+"/case_flags.txt");
    if(!(flags>>fault) || (fault!=0 && fault!=1))throw std::runtime_error("invalid fault marker");
    const int64_t segments=3*splits,tasksCount=n*hk*segments,windowRows=sink+recent+spec;
    Gm q(dir+"/q.bin",n*hq*d*2),qr(dir+"/qr.bin",n*hq*d*4);
    Gm k(dir+"/ck.bin",n*hk*d*2),v(dir+"/cv.bin",n*hk*d*2),rv(dir+(mode=="q4"?"/rv_nz.bin":"/rv.bin"),d*d*4);
    Gm raw(dir+"/raw_striped.bin",prefix+blocks*stride),table(dir+"/table.bin",requests*columns*4);
    Gm wk(dir+"/wk.bin",blocks*windowRows*hk*d*2),wv(dir+"/wv.bin",blocks*windowRows*hk*d*2);
    Gm tags(dir+"/tags.bin",blocks*windowRows*8),starts(dir+"/starts.bin",(requests+1)*4);
    Gm lens(dir+"/lens.bin",requests*4),slots(dir+"/slots.bin",n*8);
    Gm tasks(tasksCount*16*8),positions(n*8);
    Gm oldPartial(n*hq*segments*d*4),oldLse(n*hq*segments*4),oldStatus(tasksCount*2*4);
    Gm newPartial(oldPartial.size),newLse(oldLse.size),newStatus(oldStatus.size);
    const size_t oldPerCore=size_t{1146880};
    const size_t newPerCore=OSCAR_CANDIDATE_WORKSPACE_BYTES_PER_CORE ?
        OSCAR_CANDIDATE_WORKSPACE_BYTES_PER_CORE : oldPerCore;
    Gm oldWork(cores*oldPerCore),newWork(cores*newPerCore);
    const std::vector<std::pair<const char*,const Gm*>> allBuffers={
      {"q",&q},{"qr",&qr},{"k",&k},{"v",&v},{"rv",&rv},{"raw",&raw},{"table",&table},
      {"wk",&wk},{"wv",&wv},{"tags",&tags},{"starts",&starts},{"lens",&lens},{"slots",&slots},
      {"tasks",&tasks},{"positions",&positions},{"oldPartial",&oldPartial},{"oldLse",&oldLse},
      {"oldStatus",&oldStatus},{"newPartial",&newPartial},{"newLse",&newLse},{"newStatus",&newStatus},
      {"oldWorkspace",&oldWork},{"newWorkspace",&newWork}};
    auto checkGuards=[&](){for(const auto& named:allBuffers)named.second->CheckGuard(named.first);};
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    const bool slotContext=mode=="q1";
    ICPU_RUN_KF(oscar_prepare_attention_tasks_kernel,4,starts.ptr,lens.ptr,slots.ptr,tasks.ptr,positions.ptr,
        requests,n,hq,hk,sink,recent,splits,true,slotContext?table.ptr:static_cast<uint8_t*>(nullptr),
        slotContext?columns:int64_t{0},slotContext);
    const auto expectedPositions=Read(dir+"/expected_positions.bin",positions.size);
    if(std::memcmp(positions.ptr,expectedPositions.data(),positions.size))
      throw std::runtime_error("real metadata prepare changed expected positions");
    checkGuards();
    const std::vector<const Gm*> inputs={&q,&qr,&k,&v,&rv,&raw,&table,&wk,&wv,&tags,&starts,&lens,&slots,&tasks,&positions};
    std::vector<std::vector<uint8_t>> snapshots;
    for(const auto* input:inputs)snapshots.emplace_back(input->ptr,input->ptr+input->size);
    auto checkInputs=[&](){for(size_t i=0;i<inputs.size();++i)
      if(std::memcmp(inputs[i]->ptr,snapshots[i].data(),inputs[i]->size))
        throw std::runtime_error("CV modified readonly input index "+std::to_string(i));};
    for(int64_t id=0;id<tasksCount;++id) {
      const int64_t count=At<int64_t>(tasks.ptr,id*16+1);
      if(count>((mode=="q1")?1:4))throw std::runtime_error("task exceeds short-query owner contract");
    }
    const float scale=1.0F/std::sqrt(float(d));
    // Reuse only a completed, separately witnessed baseline. The Python
    // controller checks the frozen fixture digest and baseline-complete log;
    // here the newly prepared complete task table must also match byte-for-byte.
    // This removes repeated baseline execution, never extends the120s budget.
    if(reuseBaseline) {
      const std::string reference=argv[5];
      const auto oldTasks=Read(reference+"/tasks.bin",tasks.size);
      if(std::memcmp(oldTasks.data(),tasks.ptr,tasks.size))
        throw std::runtime_error("reused baseline task domain differs");
      for(auto item : {std::pair<Gm*,const char*>(&oldPartial,"partial.bin"),
                       std::pair<Gm*,const char*>(&oldLse,"lse.bin"),
                       std::pair<Gm*,const char*>(&oldStatus,"status.bin")}) {
        const auto bytes=Read(reference+"/"+item.second,item.first->size);
        std::memcpy(item.first->ptr,bytes.data(),bytes.size());
      }
      checkGuards();checkInputs();
      // Diagnostic receipt precedes expected-status assertion. Preserve the
      // original baseline, every status and the prepared task/input buffers.
      if(argc>=4)Write(std::string(argv[3])+"/baseline-status-diagnostic.bin",oldStatus.ptr,oldStatus.size);
      if(fault)for(size_t index=0;index<oldStatus.size/4;++index) {
        const int32_t value=At<int32_t>(oldStatus.ptr,index);
        if(value)std::cout<<"BASELINE_STATUS task="<<index/2<<" lane="<<index%2<<" code="<<value<<std::endl;
      }
      CheckStatuses(oldStatus,dir,fault!=0);
      std::cout<<"CPU_BASELINE_REUSED mode="<<mode<<std::endl;
    }
    // Persist the real device-task ABI before a potentially expensive CPU CV
    // run. prepare-only is never numerical CV acceptance; it only enables a
    // complete CAModel A/B against the independent golden when SDK CPU-debug
    // is too slow on long shapes. No attention tasks are filtered or changed.
    if(argc>=4) {
      const std::string exported=argv[3];
      Write(exported+"/tasks.bin",tasks.ptr,tasks.size);
      std::ofstream attrs(exported+"/attrs.txt",std::ios::trunc);
      if(!(attrs<<n<<' '<<hq<<' '<<hk<<' '<<d<<' '<<requests<<' '<<columns<<' '<<tasksCount<<' '<<b<<' '
          <<blocks<<' '<<prefix<<' '<<stride<<' '<<windowRows*hk*d<<' '<<windowRows<<' '<<sink<<' '<<recent<<' '
          <<spec<<' '<<splits<<' '<<scale<<' '<<cores<<'\n') || !attrs.flush())
        throw std::runtime_error("failed to export identical CAModel attrs");
    }
    if(argc==5) {
      checkGuards();checkInputs();
      std::cout<<"{\"backend\":\"ascendc_cpu_debug\",\"status\":\"prepared_only\","
        "\"attention_numerics\":\"not_run\",\"positions\":\"passed\","
        "\"gm_guards\":true,\"all_tasks_exported\":true}"<<std::endl;
      return 0;
    }
    auto saveBaseline=[&](){
      checkGuards();checkInputs();
      // Diagnostic receipt precedes expected-status assertion. Preserve the
      // original baseline, every status and the prepared task/input buffers.
      if(argc>=4)Write(std::string(argv[3])+"/baseline-status-diagnostic.bin",oldStatus.ptr,oldStatus.size);
      if(fault)for(size_t index=0;index<oldStatus.size/4;++index) {
        const int32_t value=At<int32_t>(oldStatus.ptr,index);
        if(value)std::cout<<"BASELINE_STATUS task="<<index/2<<" lane="<<index%2<<" code="<<value<<std::endl;
      }
      CheckStatuses(oldStatus,dir,fault!=0);
      if(argc>=4) {
        const std::string exported=argv[3];
        Write(exported+"/partial.bin",oldPartial.ptr,oldPartial.size);
        Write(exported+"/lse.bin",oldLse.ptr,oldLse.size);
        Write(exported+"/status.bin",oldStatus.ptr,oldStatus.size);
      }
      std::cout<<"CPU_BASELINE_COMPLETE mode="<<mode<<std::endl;
    };
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
#define CV_CALL(kernel,partArg,lseArg,statusArg,workArg) \
    ICPU_RUN_KF(kernel,cores,q.ptr,qr.ptr,k.ptr,v.ptr,rv.ptr,raw.ptr,table.ptr,wk.ptr,wv.ptr,tags.ptr, \
        tasks.ptr,partArg,lseArg,statusArg,workArg,n,hq,hk,d,requests,columns,tasksCount,b,blocks,prefix,stride, \
        windowRows*hk*d,windowRows,sink,recent,spec,splits,scale)
    if(mode=="q4") {
      if(!reuseBaseline) {
        CV_CALL(oscar_decode_finite_q4_baseline_kernel,oldPartial.ptr,oldLse.ptr,oldStatus.ptr,oldWork.ptr);
        saveBaseline();
      }
      CV_CALL(OSCAR_CANDIDATE_Q4_KERNEL,newPartial.ptr,newLse.ptr,newStatus.ptr,newWork.ptr);
    }else {
      if(!reuseBaseline) {
        CV_CALL(oscar_decode_finite_q1_baseline_q1_kernel,oldPartial.ptr,oldLse.ptr,oldStatus.ptr,oldWork.ptr);
        saveBaseline();
      }
      CV_CALL(OSCAR_CANDIDATE_Q1_KERNEL,newPartial.ptr,newLse.ptr,newStatus.ptr,newWork.ptr);
    }
#undef CV_CALL
    checkGuards();checkInputs();
    CheckEqual(oldPartial,newPartial,"partial");CheckEqual(oldLse,newLse,"LSE");CheckEqual(oldStatus,newStatus,"status");
    CheckStatuses(oldStatus,dir,fault!=0);CheckStatuses(newStatus,dir,fault!=0);
    if(!fault)CheckOracle(newPartial,newLse,dir,slots,n,hq,d,segments);
    checkGuards();checkInputs();
    std::cout<<"{\"backend\":\"ascendc_cpu_debug\",\"op\":\""<<OSCAR_CANDIDATE_LABEL<<"_"<<mode
      <<"\",\"status\":\"passed\",\"baseline\":\"current_production_hybrid_q4_or_MDL_q1\",\"bitwise\":true,"
      <<"\"oracle\":"<<(fault?"false":"true")<<",\"fault_order\":"<<(fault?"true":"false")
      <<",\"cores\":"<<cores<<",\"input_immutable\":true,\"gm_guards\":true,"
      <<"\"baseline_workspace_bytes_per_core\":"<<oldPerCore
      <<",\"candidate_workspace_bytes_per_core\":"<<newPerCore<<"}"<<std::endl;
    return 0;
  }catch(const std::exception& e){std::cerr<<"WINDOW_BATCH_CPU_FAILED: "<<e.what()<<std::endl;return 1;}
}
