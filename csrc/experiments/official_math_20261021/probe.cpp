// Archive #13-20/#126/#145/#148/#164: actual prepare/writer/read/merge with rotated FP32 partials.
// D.4 fixed KV256 buffers; no device timing or full-history restoration.
// Actual writer/query validation -> fixed-forward prepare -> rotated CV ->
// FP32 merge. Independent v4 expected files are prepared before execution;
// final inverse is a separately validated runtime step, not executed here.
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
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t, \
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,float,int32_t
extern "C" void oscar_attention_cv_official_decode_kernel(CV_ARGS);
extern "C" void oscar_attention_cv_official_decode_q1_kernel(CV_ARGS);
#undef CV_ARGS
extern "C" void oscar_prepare_official_attention_tasks_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,bool,uint8_t*,int64_t,bool);
extern "C" void oscar_store_aligned_int2_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,int64_t,int64_t,int64_t,int64_t,int64_t,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,float,float);
extern "C" void oscar_copy_validate_bf16_query_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,int64_t,int64_t,int64_t);
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
        (std::string(argv[1])!="q4" && std::string(argv[1])!="q1"))
      throw std::runtime_error("usage: oscar_bf16_decode_cpu q4|q1 FIXTURE_DIR OUTPUT_AND_ORACLE_DIR DEQUANT_MODE [prepare-only]");
    const std::string mode=argv[1],dir=argv[2],output=argv[3];
    const int32_t dequantMode=std::stoi(argv[4]);if(dequantMode<0 || dequantMode>1)throw std::runtime_error("invalid dequant mode");
    int64_t n,hq,hk,d,context,sink,recent,spec,b,blocks,prefix,stride,cores,requests,splits,columns;
    std::ifstream shape(dir+"/shape.txt");
    if(!(shape>>n>>hq>>hk>>d>>context>>sink>>recent>>spec>>b>>blocks>>prefix>>stride>>cores>>requests>>splits>>columns)
        || n<=0 || d!=256 || hk<=0 || hq%hk || hq/hk>8 || splits<1 || cores<1 || columns<1)
      throw std::runtime_error("invalid frozen fixture geometry");
    int fault;std::ifstream flags(dir+"/case_flags.txt");
    if(!(flags>>fault) || (fault!=0 && fault!=1))throw std::runtime_error("fault marker missing");
    const int64_t segments=3*splits,taskCount=n*hk*segments,windowRows=sink+recent+spec;
    Gm q(dir+"/q.bin",n*hq*d*2),qr(dir+"/qr.bin",n*hq*d*2),k(dir+"/ck.bin",n*hk*d*2),v(dir+"/cv.bin",n*hk*d*2);
    Gm rv(d*d*2),raw(dir+"/raw_striped.bin",prefix+blocks*stride),table(dir+"/table.bin",requests*columns*4);
    if(argc!=6){const auto rotation=Read(output+"/rv_bf16.bin",rv.size);std::memcpy(rv.ptr,rotation.data(),rv.size);}
    Gm wk(dir+"/wk.bin",blocks*windowRows*hk*d*2),wv(dir+"/wv.bin",blocks*windowRows*hk*d*2),tags(dir+"/tags.bin",blocks*windowRows*8);
    Gm starts(dir+"/starts.bin",(requests+1)*4),lens(dir+"/lens.bin",requests*4),slots(dir+"/slots.bin",n*8);
    Gm tasks(taskCount*128),positions(n*8),partial(n*hq*segments*d*4),lse(n*hq*segments*4),status(taskCount*8);
    Gm work(cores*1146880),merged(n*hq*d*4),mergedLse(n*hq*4),mergedStatus(n*hq*4);
    // Independent writer input/expected bytes; candidate outputs never seed the oracle.
    int64_t storeN;std::ifstream storeShape(dir+"/store_tokens.txt");storeShape>>storeN;
    Gm rkInput(dir+"/store_rk.bin",storeN*hk*d*2),rvInput(dir+"/store_rv.bin",storeN*hk*d*2);
    Gm originalK(dir+"/store_k.bin",storeN*hk*d*2),originalV(dir+"/store_v.bin",storeN*hk*d*2);
    Gm storeSlots(dir+"/store_slots.bin",storeN*8),storePositions(dir+"/store_positions.bin",storeN*8),storeStatus(storeN*hk*4);
    Gm qBf16(dir+"/qr_bf16.bin",n*hq*d*2),qStatus(n*hq*4);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(oscar_store_aligned_int2_kernel,1,rkInput.ptr,rvInput.ptr,originalK.ptr,originalV.ptr,
        storeSlots.ptr,storePositions.ptr,raw.ptr,wk.ptr,wv.ptr,tags.ptr,storeStatus.ptr,
        storeN,hk,d,b,blocks,prefix,stride,windowRows*hk*d,windowRows*hk*d,windowRows,
        sink,recent+spec,0.96F,0.92F);
    for(int64_t i=0;i<storeN*hk;++i)if(At<int32_t>(storeStatus.ptr,i))throw std::runtime_error("store status differs");
    for(const auto& pair:std::vector<std::pair<const Gm*,std::string>>{{&raw,"raw"},{&wk,"wk"},{&wv,"wv"},{&tags,"tags"}}) {
      const auto expected=Read(dir+"/store_expected_"+pair.second+".bin",pair.first->size);
      if(std::memcmp(pair.first->ptr,expected.data(),expected.size())) {
        Write(output+"/writer_actual_"+pair.second+".bin",pair.first->ptr,pair.first->size);
        size_t first=0;while(first<expected.size() && pair.first->ptr[first]==expected[first])++first;
        throw std::runtime_error("writer bytes differ: "+pair.second+" offset="+std::to_string(first));
      }
    }
    ICPU_RUN_KF(oscar_copy_validate_bf16_query_kernel,1,qBf16.ptr,slots.ptr,qr.ptr,qStatus.ptr,n,hq,d);
    const auto expectedQr=Read(dir+"/qr.bin",qr.size);
    if(std::memcmp(qr.ptr,expectedQr.data(),qr.size))throw std::runtime_error("query copy differs");
    for(int64_t i=0;i<n*hq;++i)if(At<int32_t>(qStatus.ptr,i))throw std::runtime_error("query status differs");
    for(const auto* gm:{&rkInput,&rvInput,&originalK,&originalV,&storeSlots,&storePositions,&storeStatus,&qBf16,&qStatus})gm->Guard();
    // Real error-domain checks use distinct outputs and preserve the healthy fixture.
    Gm badKey(hk*d*2),badStoreStatus(hk*4),badQr(n*hq*d*2),badQStatus(n*hq*4);
    std::memcpy(badKey.ptr,rkInput.ptr,badKey.size);
    const uint16_t bfNan=0x7fc0;std::memcpy(badKey.ptr,&bfNan,2);
    const std::vector<uint8_t> beforeRaw(raw.ptr,raw.ptr+raw.size),beforeWk(wk.ptr,wk.ptr+wk.size),beforeTags(tags.ptr,tags.ptr+tags.size);
    ICPU_RUN_KF(oscar_store_aligned_int2_kernel,1,badKey.ptr,rvInput.ptr,originalK.ptr,originalV.ptr,
        storeSlots.ptr,storePositions.ptr,raw.ptr,wk.ptr,wv.ptr,tags.ptr,badStoreStatus.ptr,
        int64_t{1},hk,d,b,blocks,prefix,stride,windowRows*hk*d,windowRows*hk*d,windowRows,
        sink,recent+spec,0.96F,0.92F);
    if(At<int32_t>(badStoreStatus.ptr,0)!=2 || std::memcmp(raw.ptr,beforeRaw.data(),raw.size) ||
       std::memcmp(wk.ptr,beforeWk.data(),wk.size) || std::memcmp(tags.ptr,beforeTags.data(),tags.size))
      throw std::runtime_error("writer live nonfinite error/publish policy differs");
    const uint16_t oldQ=At<uint16_t>(qBf16.ptr,0);std::memcpy(qBf16.ptr,&bfNan,2);
    ICPU_RUN_KF(oscar_copy_validate_bf16_query_kernel,1,qBf16.ptr,slots.ptr,badQr.ptr,badQStatus.ptr,n,hq,d);
    if(At<int32_t>(badQStatus.ptr,0)!=2)throw std::runtime_error("query live nonfinite status differs");
    for(int64_t row=0;row<n*hq;++row) {
      if(At<int64_t>(slots.ptr,row/hq)>=0)continue;
      if(At<int32_t>(badQStatus.ptr,row)!=0)throw std::runtime_error("query padding status differs");
      for(int64_t col=0;col<d;++col)if(At<uint16_t>(badQr.ptr,row*d+col)!=0)throw std::runtime_error("query padding poison escaped");
    }
    std::memcpy(qBf16.ptr,&oldQ,2);
    for(const auto* gm:{&badKey,&badStoreStatus,&badQr,&badQStatus})gm->Guard();
    if(fault) {
      const uint32_t nanBits=0x7fc00000;
      std::memcpy(raw.ptr+prefix+64*hk*(d/2+16)+d/2,&nanBits,4);
    }
    std::vector<const Gm*> inputs={&q,&qr,&k,&v,&rv,&raw,&table,&wk,&wv,&tags,&starts,&lens,&slots};
    std::vector<std::vector<uint8_t>> snapshots;for(const auto* input:inputs)snapshots.emplace_back(input->ptr,input->ptr+input->size);
    auto guards=[&](){for(const auto* gm:inputs)gm->Guard();for(const auto* gm:{&tasks,&positions,&partial,&lse,&status,&work,&merged,&mergedLse,&mergedStatus})gm->Guard();};
    auto immutable=[&](){for(size_t i=0;i<inputs.size();++i)if(std::memcmp(inputs[i]->ptr,snapshots[i].data(),inputs[i]->size))throw std::runtime_error("readonly input changed");};
    AscendC::SetKernelMode(KernelMode::AIV_MODE);const bool slotContext=mode=="q1";
    ICPU_RUN_KF(oscar_prepare_official_attention_tasks_kernel,4,starts.ptr,lens.ptr,slots.ptr,tasks.ptr,positions.ptr,
        requests,n,hq,hk,sink,recent,splits,true,slotContext?table.ptr:static_cast<uint8_t*>(nullptr),slotContext?columns:int64_t{0},slotContext);
    const auto expectedPositions=Read(dir+"/expected_positions.bin",positions.size);
    if(std::memcmp(positions.ptr,expectedPositions.data(),positions.size))throw std::runtime_error("prepare positions differ");
    for(int64_t id=0;id<taskCount;++id)if(At<int64_t>(tasks.ptr,id*16+1)>(mode=="q1"?1:4))throw std::runtime_error("task owner exceeds decode family");
    if(argc==6) {
      Write(output+"/tasks.bin",tasks.ptr,tasks.size);guards();immutable();
      std::cout<<"{\"status\":\"prepared_only\",\"attention_numerics\":\"not_run\",\"tasks\":"<<taskCount<<"}"<<std::endl;return 0;
    }
    const auto oldTasks=Read(output+"/tasks.bin",tasks.size);
    if(std::memcmp(tasks.ptr,oldTasks.data(),tasks.size))throw std::runtime_error("oracle task domain drift");
    const float scale=1.0F/std::sqrt(float(d));
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
#define CV_CALL(kernel) ICPU_RUN_KF(kernel,cores,q.ptr,qr.ptr,k.ptr,v.ptr,rv.ptr,raw.ptr,table.ptr,wk.ptr,wv.ptr,tags.ptr, \
    tasks.ptr,partial.ptr,lse.ptr,status.ptr,work.ptr,n,hq,hk,d,requests,columns,taskCount,b,blocks,prefix,stride, \
    windowRows*hk*d,windowRows,sink,recent,spec,splits,scale,dequantMode)
    if(mode=="q4"){CV_CALL(oscar_attention_cv_official_decode_kernel);}else{CV_CALL(oscar_attention_cv_official_decode_q1_kernel);}
#undef CV_CALL
    Write(output+"/partial.bin",partial.ptr,partial.size);Write(output+"/lse.bin",lse.ptr,lse.size);Write(output+"/status.bin",status.ptr,status.size);
    guards();immutable();CheckStatuses(status,dir,fault!=0);
    float partialError=0,lseError=0,outputError=0;
    if(!fault) {
      partialError=CheckFloat(partial,output+"/official_expected_partial.bin");lseError=CheckFloat(lse,output+"/official_expected_lse.bin");
      AscendC::SetKernelMode(KernelMode::AIV_MODE);
      ICPU_RUN_KF(oscar_merge_lse_kernel,4,partial.ptr,lse.ptr,merged.ptr,mergedLse.ptr,mergedStatus.ptr,n*hq,segments,d);
      for(int64_t i=0;i<n*hq;++i)if(At<int32_t>(mergedStatus.ptr,i))throw std::runtime_error("healthy merge has status error");
      outputError=CheckFloat(merged,output+"/official_expected_output.bin");CheckFloat(mergedLse,output+"/official_expected_output_lse.bin");
      Write(output+"/output.bin",merged.ptr,merged.size);Write(output+"/output_lse.bin",mergedLse.ptr,mergedLse.size);
    }
    guards();immutable();
    if(std::memcmp(tasks.ptr,oldTasks.data(),tasks.size))throw std::runtime_error("CV/Rv modified tasks");
    std::cout<<"{\"backend\":\"ascendc_cpu_debug\",\"status\":\"passed\",\"mode\":\""<<mode<<"\",\"oracle\":"<<(fault?"false":"true")
        <<",\"includes_bf16_rv\":false,\"partial_max_abs\":"<<partialError<<",\"lse_max_abs\":"<<lseError<<",\"output_max_abs\":"<<outputError
        <<",\"writer_bytes_exact\":true,\"query_copy_exact\":true,\"dequant_mode\":"<<dequantMode<<",\"gm_guards\":true,\"inputs_immutable\":true,\"task_count\":"<<taskCount<<"}"<<std::endl;
    return 0;
  }catch(const std::exception& error){std::cerr<<"OFFICIAL_CPU_FAILED: "<<error.what()<<std::endl;return 1;}
}
