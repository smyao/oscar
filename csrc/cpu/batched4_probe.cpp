// Archive #126/#129/#145/#148-153 and startup D.4: official CPU-debug runs
// the actual fast C4 and independent M512 batched4 AscendC kernels on the same
// frozen golden bytes. It checks partial/LSE/status bitwise and an independent
// merge oracle; CPU-debug never certifies target NPU precision or speed.
#include "tikicpulib.h"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "../include/oscar_attention_launch.h"
#include "../include/oscar_batched4_experimental.h"

extern "C" void oscar_prepare_attention_tasks_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,bool,uint8_t*,int64_t,bool);
extern "C" void oscar_attention_cv_fast_cluster4_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    uint8_t*,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,float);
extern "C" void oscar_attention_cv_batched4_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    uint8_t*,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,float);
extern "C" void oscar_merge_lse_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    int64_t,int64_t,int64_t);

namespace {
std::vector<uint8_t> Read(const std::string& path,size_t size) {
  std::vector<uint8_t> bytes(size);std::ifstream stream(path,std::ios::binary);
  if(!stream.read(reinterpret_cast<char*>(bytes.data()),size) || stream.peek()!=EOF)
    throw std::runtime_error("incorrect golden byte count: "+path);
  return bytes;
}
void Write(const std::string& path,const uint8_t* bytes,size_t size) {
  std::ofstream stream(path,std::ios::binary|std::ios::trunc);
  if(!stream.write(reinterpret_cast<const char*>(bytes),size) || !stream.flush())
    throw std::runtime_error("failed to save checked reference bytes: "+path);
}
struct Gm {
  uint8_t* ptr;size_t size;
  explicit Gm(size_t n):ptr(static_cast<uint8_t*>(AscendC::GmAlloc(n))),size(n) {
    if(!ptr)throw std::runtime_error("GmAlloc failed");std::memset(ptr,0x85,n);
  }
  Gm(const std::string& path,size_t n):Gm(n) {
    const auto bytes=Read(path,n);std::memcpy(ptr,bytes.data(),n);
  }
  ~Gm(){AscendC::GmFree(ptr);}Gm(const Gm&)=delete;
};
void EqualBytes(const Gm& left,const Gm& right,const std::string& label) {
  if(left.size!=right.size || std::memcmp(left.ptr,right.ptr,left.size)!=0) {
    size_t at=0;while(at<left.size && left.ptr[at]==right.ptr[at])++at;
    throw std::runtime_error(label+" differs at byte "+std::to_string(at));
  }
}
void StatsParity(const Gm& oldStats,const Gm& newStats,int64_t cores) {
  if(oldStats.size!=newStats.size || oldStats.size!=static_cast<size_t>(cores*8*8))
    throw std::runtime_error("cluster stats shape changed");
  int64_t oldTotals[8]={},newTotals[8]={};
  for(int64_t core=0;core<cores;++core) {
    int64_t oldRow[8],newRow[8];
    std::memcpy(oldRow,oldStats.ptr+core*8*8,sizeof(oldRow));
    std::memcpy(newRow,newStats.ptr+core*8*8,sizeof(newRow));
    for(const auto* row:{oldRow,newRow}) {
      for(int field=0;field<8;++field)
        if(row[field]<0)throw std::runtime_error("negative C4 counter");
      if(row[1]!=4*row[0] || row[4]!=3*row[3] || row[7]!=row[0])
        throw std::runtime_error("C4 per-core grouping/reuse counter mismatch");
    }
    for(int field=0;field<8;++field) {
      oldTotals[field]+=oldRow[field];newTotals[field]+=newRow[field];
    }
  }
  for(int field=0;field<8;++field)
    if(oldTotals[field]!=newTotals[field])
      throw std::runtime_error("C4 global counter differs at field "+std::to_string(field));
  if(newTotals[6]!=newTotals[1])
    throw std::runtime_error("C4 global skipped members differ from grouped leaders");
}
void StatusZero(const Gm& data) {
  for(size_t i=0;i<data.size/sizeof(int32_t);++i) {
    int32_t value;std::memcpy(&value,data.ptr+i*sizeof(value),sizeof(value));
    if(value)throw std::runtime_error("nonzero status at word "+std::to_string(i));
  }
}
void Close(const Gm& observed,const std::string& path) {
  const auto bytes=Read(path,observed.size);
  for(size_t i=0;i<observed.size/4;++i) {
    float actual,expected;
    std::memcpy(&actual,observed.ptr+i*4,4);
    std::memcpy(&expected,bytes.data()+i*4,4);
    if(actual==expected)continue;
    if(!std::isfinite(actual) || !std::isfinite(expected) ||
       std::abs(actual-expected)>0.005F+0.005F*std::abs(expected))
      throw std::runtime_error("frozen oracle mismatch "+path+" index="+std::to_string(i));
  }
}
void Poison(Gm& gm) {
  constexpr uint32_t bits=0x7fc00000U;
  if(gm.size%4)throw std::runtime_error("workspace is not FP32 aligned");
  for(size_t offset=0;offset<gm.size;offset+=4)std::memcpy(gm.ptr+offset,&bits,4);
}
void SameClass(const Gm& left,const Gm& right,const std::string& label) {
  if(left.size!=right.size)throw std::runtime_error(label+" shape changed");
  for(size_t offset=0;offset<left.size;offset+=4) {
    float a,b;std::memcpy(&a,left.ptr+offset,4);std::memcpy(&b,right.ptr+offset,4);
    if(std::isnan(a)!=std::isnan(b) || std::isfinite(a)!=std::isfinite(b) ||
       (std::isfinite(a) && a!=b))
      throw std::runtime_error(label+" invalid-row classification differs");
  }
}
}

int main(int argc,char** argv) {
  try {
    if(argc<3 || argc>4)throw std::runtime_error("usage: oscar_batched4_cpu mode case_directory [fresh_artifact_directory]");
    const std::string mode=argv[1],dir=argv[2],artifacts=argc==4?argv[3]:"";
    const bool separatedReference=mode=="old_only_reference";
    const bool separatedCandidate=mode=="old_only_candidate";
    if((separatedReference || separatedCandidate)!=(argc==4))
      throw std::runtime_error("only separated old-only modes accept an artifact directory");
    if(mode!="normal" && mode!="poison" && mode!="bad_meta" &&
       mode!="nan_query" && mode!="c1" && mode!="old_only" &&
       mode!="poison_old_only" && !separatedReference && !separatedCandidate)
      throw std::runtime_error("unknown batched4 mode");
    const bool oldOnly=mode=="old_only" || mode=="poison_old_only" ||
        separatedReference || separatedCandidate;
    const auto began=std::chrono::steady_clock::now();
    const auto mark=[&](const char* stage) {
      const double seconds=std::chrono::duration<double>(
          std::chrono::steady_clock::now()-began).count();
      std::cout<<"BATCHED4_STAGE mode="<<mode<<" stage="<<stage
               <<" elapsed_s="<<seconds<<std::endl;
    };
    std::ifstream shape(dir+"/shape.txt");
    int64_t n,hq,hk,d,context,sink,recent,spec,blockTokens,blocks,prefix,stride,cores;
    if(!(shape>>n>>hq>>hk>>d>>context>>sink>>recent>>spec>>blockTokens>>blocks>>prefix>>stride>>cores))
      throw std::runtime_error("invalid golden shape");
    int64_t requests=1,splits=1;shape>>requests;shape>>splits;
    if(n<=0 || hq<=0 || hk<=0 || d<=0 || requests<=0 || splits<=0)
      throw std::runtime_error("nonpositive golden shape");
    const int64_t segments=3*splits,tasksCount=n*hk*segments;
    const int64_t windowRows=sink+recent+spec;
    Gm q(dir+"/q.bin",n*hq*d*2),qr(dir+"/qr.bin",n*hq*d*4);
    Gm ck(dir+"/ck.bin",n*hk*d*2),cv(dir+"/cv.bin",n*hk*d*2);
    Gm rv(dir+"/rv.bin",d*d*4),raw(dir+"/raw.bin",prefix+blocks*stride);
    Gm table(dir+"/table.bin",requests*8*4);
    Gm wk(dir+"/wk.bin",blocks*windowRows*hk*d*2);
    Gm wv(dir+"/wv.bin",blocks*windowRows*hk*d*2);
    Gm tags(dir+"/tags.bin",blocks*windowRows*8);
    Gm starts(dir+"/starts.bin",(requests+1)*4),lens(dir+"/lens.bin",requests*4);
    Gm slots(dir+"/slots.bin",n*8),tasks(tasksCount*16*8),positions(n*8);
    const int64_t partialBytes=n*hq*segments*d*4,lseBytes=n*hq*segments*4;
    Gm oldPartial(partialBytes),oldLse(lseBytes),oldStatus(tasksCount*2*4);
    Gm newPartial(partialBytes),newLse(lseBytes),newStatus(tasksCount*2*4);
    Gm oldWorkspace(cores*oscar_ascend::attention_cluster4_workspace_per_core(d));
    Gm newWorkspace(cores*oscar_ascend_experiment::batched4_workspace_per_core(d));
    Gm oldStats(cores*8*8),newStats(cores*8*8);
    Gm output(n*hq*d*4),lse(n*hq*4),mergeStatus(n*hq*4);
    if(mode=="poison" || mode=="poison_old_only") {
      Poison(oldWorkspace);Poison(newWorkspace);
    }
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    mark("PREPARE_START");
    ICPU_RUN_KF(oscar_prepare_attention_tasks_kernel,4,starts.ptr,lens.ptr,slots.ptr,
        tasks.ptr,positions.ptr,requests,n,hq,hk,sink,recent,splits,true,
        static_cast<uint8_t*>(nullptr),int64_t{0},false);
    mark("PREPARE_DONE");
    const bool hasExpectedPositions=requests>1;
    const auto expectedPositions=hasExpectedPositions?
        Read(dir+"/expected_positions.bin",n*8):std::vector<uint8_t>{};
    for(int64_t token=0;token<n;++token) {
      int64_t actual,expected=context+token;
      std::memcpy(&actual,positions.ptr+token*8,8);
      if(hasExpectedPositions)std::memcpy(&expected,expectedPositions.data()+token*8,8);
      if(actual!=expected)throw std::runtime_error("prepared position differs from golden");
    }
    if(oldOnly) {
      // Mirror production main eager prefill's source2 suppression. Both
      // kernels see identical tasks; the golden's independent oracle contains
      // exact sink/window + compressed history only.
      for(int64_t id=0;id<tasksCount;++id) {
        auto* row=reinterpret_cast<int64_t*>(tasks.ptr)+id*16;
        if(row[7]==2 && row[10]==0 && row[4]>=row[3])row[4]=row[3];
      }
    }
    if(mode=="bad_meta") {
      if(n<168 || context<=100 || d!=64 || hk!=1)
        throw std::runtime_error("bad_meta requires mature D64 fixture");
      // Exact physical page and scale byte used by the existing C4 CPU gate.
      const int64_t offset=prefix+stride+100*(d/2+8)+d/4;
      if(offset+2>static_cast<int64_t>(raw.size))throw std::runtime_error("bad scale offset");
      std::memset(raw.ptr+offset,0,2);
    }
    if(mode=="nan_query") {
      if(n<168 || d!=64)throw std::runtime_error("nan_query requires mature D64 fixture");
      const float value=std::nanf("");
      std::memcpy(qr.ptr+(105*hq)*d*4,&value,4);
    }
    if(separatedCandidate) {
      const auto a=Read(artifacts+"/partial.bin",oldPartial.size);
      const auto b=Read(artifacts+"/lse.bin",oldLse.size);
      const auto c=Read(artifacts+"/status.bin",oldStatus.size);
      const auto s=Read(artifacts+"/stats.bin",oldStats.size);
      std::memcpy(oldPartial.ptr,a.data(),a.size());
      std::memcpy(oldLse.ptr,b.data(),b.size());
      std::memcpy(oldStatus.ptr,c.data(),c.size());
      std::memcpy(oldStats.ptr,s.data(),s.size());
    }
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    if(!separatedCandidate)mark("REFERENCE_START");
    if(!separatedCandidate)ICPU_RUN_KF(oscar_attention_cv_fast_cluster4_kernel,cores,q.ptr,qr.ptr,ck.ptr,cv.ptr,
        rv.ptr,raw.ptr,table.ptr,wk.ptr,wv.ptr,tags.ptr,tasks.ptr,oldPartial.ptr,
        oldLse.ptr,oldStatus.ptr,oldWorkspace.ptr,oldStats.ptr,n,hq,hk,d,
        requests,int64_t{8},tasksCount,blockTokens,blocks,prefix,stride,
        windowRows*hk*d,windowRows,sink,recent,spec,splits,
        1.0F/std::sqrt(float(d)));
    if(!separatedCandidate)mark("REFERENCE_DONE");
    if(!separatedReference)mark("BATCHED4_START");
    if(!separatedReference)ICPU_RUN_KF(oscar_attention_cv_batched4_kernel,cores,q.ptr,qr.ptr,ck.ptr,cv.ptr,
        rv.ptr,raw.ptr,table.ptr,wk.ptr,wv.ptr,tags.ptr,tasks.ptr,newPartial.ptr,
        newLse.ptr,newStatus.ptr,newWorkspace.ptr,newStats.ptr,n,hq,hk,d,
        requests,int64_t{8},tasksCount,blockTokens,blocks,prefix,stride,
        windowRows*hk*d,windowRows,sink,recent,spec,splits,
        1.0F/std::sqrt(float(d)));
    if(!separatedReference)mark("BATCHED4_DONE");
    if(!separatedReference) {
      EqualBytes(oldStatus,newStatus,"status");StatsParity(oldStats,newStats,cores);
    }
    int64_t clusters=0;
    for(int64_t core=0;core<cores;++core) {
      int64_t count;std::memcpy(&count,oldStats.ptr+core*8*8,8);clusters+=count;
    }
    if((mode=="c1")!=(clusters==0))throw std::runtime_error("C4 branch coverage differs");
    if(mode=="bad_meta" || mode=="nan_query") {
      bool error=false;
      for(size_t offset=0;offset<oldStatus.size;offset+=4) {
        int32_t code;std::memcpy(&code,oldStatus.ptr+offset,4);
        if(code)error=true;
      }
      if(!error)throw std::runtime_error("invalid input did not reach status");
      SameClass(oldPartial,newPartial,"partial");SameClass(oldLse,newLse,"lse");
    } else {
      if(!separatedReference) {
        EqualBytes(oldPartial,newPartial,"partial");EqualBytes(oldLse,newLse,"lse");
      }
      StatusZero(oldStatus);
      AscendC::SetKernelMode(KernelMode::AIV_MODE);
      ICPU_RUN_KF(oscar_merge_lse_kernel,4,
          separatedReference?oldPartial.ptr:newPartial.ptr,
          separatedReference?oldLse.ptr:newLse.ptr,output.ptr,lse.ptr,
          mergeStatus.ptr,n*hq,segments,d);
      StatusZero(mergeStatus);
      Close(output,dir+"/expected_output.bin");Close(lse,dir+"/expected_lse.bin");
      mark("ORACLE_DONE");
      if(separatedReference) {
        // No bytes become trusted until fe0 status and independent oracle pass.
        Write(artifacts+"/partial.bin",oldPartial.ptr,oldPartial.size);
        Write(artifacts+"/lse.bin",oldLse.ptr,oldLse.size);
        Write(artifacts+"/status.bin",oldStatus.ptr,oldStatus.size);
        Write(artifacts+"/stats.bin",oldStats.ptr,oldStats.size);
      }
    }
    std::cout<<"batched4_clusters="<<clusters<<" mode="<<mode<<std::endl;
    std::cout<<"{\"backend\":\"ascendc_cpu_debug\",\"op\":\"batched4\",\"status\":\"passed\"}"<<std::endl;
    return 0;
  }catch(const std::exception& error) {
    std::cerr<<"BATCHED4_CPU_FAILED: "<<error.what()<<std::endl;return 1;
  }
}
