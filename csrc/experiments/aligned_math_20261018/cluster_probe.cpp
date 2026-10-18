// Archive #126/#145/#148/#154/#157: full BF16 decode source protocol check.
// D.4: fixed KV256 fused FIA including BF16 Rv, never history restore.
// Actual prepare -> BF16 CV/QK/PV/Rv -> merge. New independent task oracle
// is prepared before execution; old FP32 expected files are never overwritten.
// CPU-debug, CANN build, target completion and model quality remain separate.
#include "tikicpulib.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define CV_ARGS uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*, \
    uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*, \
    uint8_t*,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t, \
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,float,int32_t
extern "C" void oscar_attention_cv_aligned_cluster4_kernel(CV_ARGS);
extern "C" void oscar_attention_cv_aligned_cluster16_kernel(CV_ARGS);
#undef CV_ARGS
extern "C" void oscar_prepare_attention_tasks_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,bool,uint8_t*,int64_t,bool);
extern "C" void oscar_merge_lse_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,int64_t,int64_t,int64_t);

namespace {
std::vector<uint8_t> Read(const std::string& path,size_t n) {
  std::vector<uint8_t> result(n);std::ifstream file(path,std::ios::binary);
  if(!file.read(reinterpret_cast<char*>(result.data()),n) || file.peek()!=EOF)
    throw std::runtime_error("incorrect input byte count: "+path);
  return result;
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
  Gm(const std::string& path,size_t n):Gm(n){const auto value=Read(path,n);std::memcpy(ptr,value.data(),n);}
  ~Gm(){AscendC::GmFree(base);}
  Gm(const Gm&)=delete;Gm& operator=(const Gm&)=delete;
  void Guard() const {
    for(size_t i=0;i<guard;++i)if(base[i]!=0xD7 || ptr[size+i]!=0xD7)
      throw std::runtime_error("GM allocation boundary changed");
  }
};
template<class T>T At(const uint8_t* ptr,int64_t index) {
  T value;std::memcpy(&value,ptr+index*sizeof(T),sizeof(T));return value;
}
float CheckFloat(const Gm& actual,const std::string& path) {
  const auto expected=Read(path,actual.size);float maximum=0.0F;
  for(size_t i=0;i<actual.size/4;++i) {
    const float a=At<float>(actual.ptr,i),e=At<float>(expected.data(),i);
    if(e==-INFINITY) {if(a!=-INFINITY)throw std::runtime_error("empty LSE differs: "+path);continue;}
    if(!std::isfinite(a) || !std::isfinite(e) || std::abs(a-e)>0.005F+0.005F*std::abs(e))
      throw std::runtime_error("BF16 task oracle exceeds frozen tolerance: "+path+" index="+std::to_string(i)+
          " actual="+std::to_string(a)+" expected="+std::to_string(e));
    maximum=std::max(maximum,std::abs(a-e));
  }
  return maximum;
}
void CheckStatuses(const Gm& status,const std::string& dir,bool invalid) {
  bool nonzero=false;for(size_t i=0;i<status.size/4;++i)nonzero|=At<int32_t>(status.ptr,i)!=0;
  if(nonzero!=invalid)throw std::runtime_error("valid/fault status domain differs");
  std::ifstream input(dir+"/required_status_checks.bin",std::ios::binary|std::ios::ate);
  const std::streamoff bytes=input.tellg();
  if(bytes<0 || bytes%24 || (invalid && bytes==0))throw std::runtime_error("fault status checks missing");
  const auto checks=Read(dir+"/required_status_checks.bin",static_cast<size_t>(bytes));
  for(int64_t i=0;i<bytes/8;i+=3) {
    const int64_t task=At<int64_t>(checks.data(),i),lane=At<int64_t>(checks.data(),i+1),code=At<int64_t>(checks.data(),i+2);
    if(task<0 || lane<0 || lane>1 || (task*2+lane)*4>=static_cast<int64_t>(status.size) ||
        At<int32_t>(status.ptr,task*2+lane)!=code)throw std::runtime_error("fault status order differs");
  }
}
}
int main(int argc,char** argv) {
  try {
    if((argc!=5 && argc!=6) || (argc==6 && std::string(argv[5])!="prepare-only") ||
        (std::string(argv[1])!="c4" && std::string(argv[1])!="c16"))
      throw std::runtime_error("usage: oscar_bf16_clusters_cpu c4|c16 FIXTURE_DIR OUTPUT_AND_ORACLE_DIR DEQUANT_MODE [prepare-only]");
    const std::string mode=argv[1],dir=argv[2],output=argv[3];
    const int32_t dequantMode=std::stoi(argv[4]);if(dequantMode<0 || dequantMode>1)throw std::runtime_error("invalid dequant mode");
    int64_t n,hq,hk,d,context,sink,recent,spec,b,blocks,prefix,stride,cores,requests,splits,columns;
    std::ifstream shape(dir+"/shape.txt");
    if(!(shape>>n>>hq>>hk>>d>>context>>sink>>recent>>spec>>b>>blocks>>prefix>>stride>>cores>>requests>>splits>>columns)
        || n<=0 || d!=256 || hk<=0 || hq%hk || hq/hk>16 || splits<1 || cores<1 || columns<1)
      throw std::runtime_error("invalid frozen fixture geometry");
    int fault;std::ifstream flags(dir+"/case_flags.txt");
    if(!(flags>>fault) || (fault!=0 && fault!=1))throw std::runtime_error("fault marker missing");
    int64_t suppressCurrent,expectedClusters,badMetadata,badQr;
    std::ifstream clusterFlags(dir+"/cluster_flags.txt");
    if(!(clusterFlags>>suppressCurrent>>expectedClusters>>badMetadata>>badQr))throw std::runtime_error("cluster flags missing");
    if(mode=="c4")expectedClusters*=4;
    const int64_t segments=3*splits,taskCount=n*hk*segments,windowRows=sink+recent+spec;
    Gm q(dir+"/q.bin",n*hq*d*2),qr(dir+"/qr.bin",n*hq*d*2),k(dir+"/ck.bin",n*hk*d*2),v(dir+"/cv.bin",n*hk*d*2);
    Gm rv(d*d*2),raw(dir+"/raw_striped.bin",prefix+blocks*stride),table(dir+"/table.bin",requests*columns*4);
    if(argc!=6){const auto rotation=Read(output+"/rv_bf16.bin",rv.size);std::memcpy(rv.ptr,rotation.data(),rv.size);}
    Gm wk(dir+"/wk.bin",blocks*windowRows*hk*d*2),wv(dir+"/wv.bin",blocks*windowRows*hk*d*2),tags(dir+"/tags.bin",blocks*windowRows*8);
    Gm starts(dir+"/starts.bin",(requests+1)*4),lens(dir+"/lens.bin",requests*4),slots(dir+"/slots.bin",n*8);
    Gm tasks(taskCount*128),positions(n*8),partial(n*hq*segments*d*4),lse(n*hq*segments*4),status(taskCount*8);
    Gm stats(cores*64);
    Gm work(cores*(mode=="c16"?7487488:2494464)),merged(n*hq*d*4),mergedLse(n*hq*4),mergedStatus(n*hq*4);
    std::vector<const Gm*> inputs={&q,&qr,&k,&v,&rv,&raw,&table,&wk,&wv,&tags,&starts,&lens,&slots};
    std::vector<std::vector<uint8_t>> snapshots;for(const auto* input:inputs)snapshots.emplace_back(input->ptr,input->ptr+input->size);
    auto guards=[&](){for(const auto* gm:inputs)gm->Guard();for(const auto* gm:{&tasks,&positions,&partial,&lse,&status,&work,&stats,&merged,&mergedLse,&mergedStatus})gm->Guard();};
    auto immutable=[&](){for(size_t i=0;i<inputs.size();++i)if(std::memcmp(inputs[i]->ptr,snapshots[i].data(),inputs[i]->size))throw std::runtime_error("readonly input changed");};
    AscendC::SetKernelMode(KernelMode::AIV_MODE);const bool slotContext=false;
    ICPU_RUN_KF(oscar_prepare_attention_tasks_kernel,4,starts.ptr,lens.ptr,slots.ptr,tasks.ptr,positions.ptr,
        requests,n,hq,hk,sink,recent,splits,true,slotContext?table.ptr:static_cast<uint8_t*>(nullptr),slotContext?columns:int64_t{0},slotContext);
    const auto expectedPositions=Read(dir+"/expected_positions.bin",positions.size);
    if(std::memcmp(positions.ptr,expectedPositions.data(),positions.size))throw std::runtime_error("prepare positions differ");
    for(int64_t id=0;id<taskCount;++id) {
      auto* row=reinterpret_cast<int64_t*>(tasks.ptr)+id*16;
      if(row[1]>128/(hq/hk))throw std::runtime_error("task exceeds long query group");
      if(suppressCurrent && row[7]==2 && row[10]==0 && row[4]>=row[3])row[4]=row[3];
    }
    if(argc==6) {
      Write(output+"/tasks.bin",tasks.ptr,tasks.size);guards();immutable();
      std::cout<<"{\"status\":\"prepared_only\",\"attention_numerics\":\"not_run\",\"tasks\":"<<taskCount<<"}"<<std::endl;return 0;
    }
    const auto oldTasks=Read(output+"/tasks.bin",tasks.size);
    if(std::memcmp(tasks.ptr,oldTasks.data(),tasks.size))throw std::runtime_error("oracle task domain drift");
    const float scale=1.0F/std::sqrt(float(d));
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
#define CV_CALL(kernel) ICPU_RUN_KF(kernel,cores,q.ptr,qr.ptr,k.ptr,v.ptr,rv.ptr,raw.ptr,table.ptr,wk.ptr,wv.ptr,tags.ptr, \
    tasks.ptr,partial.ptr,lse.ptr,status.ptr,work.ptr,stats.ptr,n,hq,hk,d,requests,columns,taskCount,b,blocks,prefix,stride, \
    windowRows*hk*d,windowRows,sink,recent,spec,splits,scale,dequantMode)
    if(mode=="c4"){CV_CALL(oscar_attention_cv_aligned_cluster4_kernel);}else{CV_CALL(oscar_attention_cv_aligned_cluster16_kernel);}
#undef CV_CALL
    Write(output+"/partial.bin",partial.ptr,partial.size);Write(output+"/lse.bin",lse.ptr,lse.size);Write(output+"/status.bin",status.ptr,status.size);
    guards();immutable();CheckStatuses(status,dir,fault!=0);
    int64_t clusters=0,members=0,skips=0,independent=0;
    for(int64_t core=0;core<cores;++core){clusters+=At<int64_t>(stats.ptr,core*8);members+=At<int64_t>(stats.ptr,core*8+1);skips+=At<int64_t>(stats.ptr,core*8+6);independent+=At<int64_t>(stats.ptr,core*8+2);}
    const int clusterSize=mode=="c16"?16:4;
    if(independent!=1)throw std::runtime_error("tail C1 leader missing");
    if(clusters!=expectedClusters || members!=expectedClusters*clusterSize || skips!=members)throw std::runtime_error("actual mature cluster ownership differs");
    for(int64_t id=0;id<taskCount;++id)for(int lane=0;lane<2;++lane) {
      auto* row=reinterpret_cast<int64_t*>(tasks.ptr)+id*16;int expected=0;
      if(row[1]>0 && row[7]==0) {
        if(badMetadata>=row[3] && badMetadata<row[4] && ((badMetadata-row[3])%32)/16==lane)expected=3;
        if(badQr>=row[0] && badQr<row[0]+row[1] && ((badQr-row[0])*(hq/hk))/64==lane)expected=2;
      }
      if(At<int32_t>(status.ptr,id*2+lane)!=expected)throw std::runtime_error("complete status/lane oracle at "+std::to_string(id)+":"+std::to_string(lane));
    }
    Write(output+"/cluster_stats.bin",stats.ptr,stats.size);
    float partialError=0,lseError=0,outputError=0;
    if(!fault) {
      partialError=CheckFloat(partial,output+"/bf16_expected_partial.bin");lseError=CheckFloat(lse,output+"/bf16_expected_lse.bin");
      AscendC::SetKernelMode(KernelMode::AIV_MODE);
      ICPU_RUN_KF(oscar_merge_lse_kernel,4,partial.ptr,lse.ptr,merged.ptr,mergedLse.ptr,mergedStatus.ptr,n*hq,segments,d);
      for(int64_t i=0;i<n*hq;++i)if(At<int32_t>(mergedStatus.ptr,i))throw std::runtime_error("healthy merge has status error");
      outputError=CheckFloat(merged,output+"/bf16_expected_output.bin");CheckFloat(mergedLse,output+"/bf16_expected_output_lse.bin");
      Write(output+"/output.bin",merged.ptr,merged.size);Write(output+"/output_lse.bin",mergedLse.ptr,mergedLse.size);
    }
    guards();immutable();
    if(std::memcmp(tasks.ptr,oldTasks.data(),tasks.size))throw std::runtime_error("CV/Rv modified tasks");
    std::cout<<"{\"backend\":\"ascendc_cpu_debug\",\"status\":\"passed\",\"mode\":\""<<mode<<"\",\"oracle\":"<<(fault?"false":"true")
        <<",\"includes_bf16_rv\":true,\"partial_max_abs\":"<<partialError<<",\"lse_max_abs\":"<<lseError<<",\"output_max_abs\":"<<outputError
        <<",\"eligible_clusters\":"<<clusters<<",\"independent_tail_leaders\":"<<independent<<",\"dequant_mode\":"<<dequantMode<<",\"members\":"<<members<<",\"gm_guards\":true,\"inputs_immutable\":true,\"task_count\":"<<taskCount<<"}"<<std::endl;
    return 0;
  }catch(const std::exception& error){std::cerr<<"ALIGNED_CLUSTER_CPU_FAILED: "<<error.what()<<std::endl;return 1;}
}
