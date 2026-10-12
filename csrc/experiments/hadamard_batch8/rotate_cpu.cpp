// SPDX-License-Identifier: Apache-2.0
// Archive G26-G34/#13-22/#70-73/#126/#145/#148/#154-156; complete D.4.
// 1. Query rotation before bounded INT2 attention, not history recovery.
// 2. Historical 6.5s dequant and 215ms stores are not timing evidence here.
// 3. Run the actual unchanged/init-only/batch8 kernels, same inputs/grid;
//    persist outputs/guards before any checks; NaN payloads/zeros compare as
//    raw bytes BETWEEN arms. Independent oracle retains frozen2e-3/class gate.
// 4. Official CPU is a falsifier; no NPU, graph or speed claim. One failed
//    arm stops this case; no retry, substituted reference or wider tolerance.
#include "tikicpulib.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#define DECLARE(NAME) extern "C" void NAME(uint8_t*,uint8_t*,uint8_t*,uint8_t*,int64_t,int64_t,int32_t,bool,uint8_t*,int64_t)
DECLARE(oscar_rotate_kernel);DECLARE(oscar_rotate_batch8_init_kernel);DECLARE(oscar_rotate_batch8_kernel);
#undef DECLARE
namespace {
using Path=std::filesystem::path;
std::vector<uint8_t> Read(const Path& path,size_t n) {
  std::ifstream f(path,std::ios::binary);std::vector<uint8_t> bytes(n);
  if(!f.read(reinterpret_cast<char*>(bytes.data()),n)||f.peek()!=EOF)throw std::runtime_error("wrong byte count: "+path.string());
  return bytes;
}
struct Gm {
  uint8_t* base;uint8_t* ptr;size_t n;std::vector<uint8_t> initial;
  explicit Gm(size_t size):n(size),initial(size,0x85) {
    base=static_cast<uint8_t*>(AscendC::GmAlloc(n+128));if(!base)throw std::runtime_error("GmAlloc failed");
    ptr=base+64;std::memset(base,0xA5,n+128);std::memcpy(ptr,initial.data(),n);
  }
  ~Gm(){AscendC::GmFree(base);}Gm(const Gm&)=delete;
  void Load(const Path& p){initial=Read(p,n);std::memcpy(ptr,initial.data(),n);}
  bool Guards()const{for(size_t i=0;i<64;++i)if(base[i]!=0xA5||ptr[n+i]!=0xA5)return false;return true;}
  bool Unchanged()const{return !std::memcmp(ptr,initial.data(),n);}
  void Save(const Path& p,const std::string& name)const {
    std::ofstream f(p/(name+".bin"),std::ios::binary),g(p/(name+".guarded.bin"),std::ios::binary);
    f.write(reinterpret_cast<const char*>(ptr),n);g.write(reinterpret_cast<const char*>(base),n+128);
    if(!f||!g)throw std::runtime_error("cannot save "+name);
  }
};
size_t Differences(const uint8_t* left,const uint8_t* right,size_t n){size_t count=0;for(size_t i=0;i<n;++i)count+=left[i]!=right[i];return count;}
float Value(const uint8_t* data,size_t i){float v;std::memcpy(&v,data+i*4,4);return v;}
}
int main(int argc,char** argv){try {
  if(argc<3||argc>5)throw std::runtime_error("usage: rotate_cpu FIXTURE NEW_OUTPUT [ARM [BASELINE_DIR]]");
  const Path fixture=argv[1],out=argv[2];int selected=-1;
  if(argc>=4){selected=std::stoi(argv[3]);if(selected<0||selected>2)throw std::runtime_error("arm must be0/1/2");}
  if((selected>0&&argc!=5)||(selected<=0&&argc==5))throw std::runtime_error("single candidate arm requires exactly its saved arm0 directory");
  int64_t rows,dim,heads;int32_t dtype,grid;bool hadamard;
  std::ifstream shape(fixture/"shape.txt");shape>>rows>>dim>>dtype>>heads>>hadamard>>grid;
  if(!shape||rows<=0||heads<1||rows%heads||(dim!=64&&dim!=128&&dim!=256)||dtype<0||dtype>2||!hadamard||
      grid!=std::min<int64_t>(32,(rows+7)/8))throw std::runtime_error("invalid declared rotate shape/grid");
  if(std::filesystem::exists(out))throw std::runtime_error("refuse existing output");
  std::filesystem::create_directories(out);
  Gm source(rows*dim*(dtype==0?4:2)),rotation(dim*dim*4),slots(rows/heads*8);
  source.Load(fixture/"source.bin");rotation.Load(fixture/"rotation.bin");slots.Load(fixture/"slots.bin");
  auto expected=Read(fixture/"expected_output.bin",rows*dim*4),expectedStatus=Read(fixture/"expected_status.bin",rows*4);
  std::vector<uint8_t> baselineOutput,baselineStatus;
  if(selected>0) {
    const Path baseline=argv[4];int64_t br,bd,bh;int32_t bt,bg;bool bHad;
    std::string recorded;std::ifstream baselineReport(baseline/"arm.json");std::getline(baselineReport,recorded);
    if(!baselineReport||recorded.rfind("{\"arm\":0,\"status\":\"passed\",",0)!=0)
      throw std::runtime_error("baseline arm0 did not pass its saved gates");
    std::ifstream bs(baseline/"shape.txt");bs>>br>>bd>>bt>>bh>>bHad>>bg;
    if(!bs||br!=rows||bd!=dim||bt!=dtype||bh!=heads||bHad!=hadamard||bg!=grid||
        Read(baseline/"source_after.bin",source.n)!=source.initial||
        Read(baseline/"rotation_after.bin",rotation.n)!=rotation.initial||
        Read(baseline/"slots_after.bin",slots.n)!=slots.initial)throw std::runtime_error("baseline shape/inputs differ");
    baselineOutput=Read(baseline/"output.bin",rows*dim*4);baselineStatus=Read(baseline/"status.bin",rows*4);
  }
  AscendC::SetKernelMode(KernelMode::AIV_MODE);bool passed=true;std::string records="[";int count=0;
  for(int arm=0;arm<3;++arm) {
    if(selected>=0&&arm!=selected)continue;
    Path directory=selected<0?out/std::to_string(arm):out;
    if(selected<0)std::filesystem::create_directory(directory);
    Gm output(rows*dim*4),status(rows*4);
    if(arm==0){ICPU_RUN_KF(oscar_rotate_kernel,grid,source.ptr,rotation.ptr,output.ptr,status.ptr,rows,dim,dtype,hadamard,slots.ptr,heads);}
    else if(arm==1){ICPU_RUN_KF(oscar_rotate_batch8_init_kernel,grid,source.ptr,rotation.ptr,output.ptr,status.ptr,rows,dim,dtype,hadamard,slots.ptr,heads);}
    else {ICPU_RUN_KF(oscar_rotate_batch8_kernel,grid,source.ptr,rotation.ptr,output.ptr,status.ptr,rows,dim,dtype,hadamard,slots.ptr,heads);}
    // Persist every real payload including guards before comparison.
    source.Save(directory,"source_after");rotation.Save(directory,"rotation_after");slots.Save(directory,"slots_after");
    output.Save(directory,"output");status.Save(directory,"status");
    std::ofstream savedShape(directory/"shape.txt");savedShape<<rows<<' '<<dim<<' '<<dtype<<' '<<heads<<' '<<hadamard<<' '<<grid<<'\n';
    if(!savedShape)throw std::runtime_error("cannot save shape");
    const bool guards=source.Guards()&&rotation.Guards()&&slots.Guards()&&output.Guards()&&status.Guards();
    const bool inputs=source.Unchanged()&&rotation.Unchanged()&&slots.Unchanged();
    const size_t statusOracle=Differences(status.ptr,expectedStatus.data(),status.n);
    size_t finiteBad=0,classBad=0;
    for(size_t i=0;i<output.n/4;++i) {
      const float got=Value(output.ptr,i),want=Value(expected.data(),i);
      if(std::isfinite(want)) {
        if(!std::isfinite(got)||std::abs(got-want)>0.002F+0.002F*std::abs(want))++finiteBad;
      } else if(std::isnan(want)?!std::isnan(got):got!=want)++classBad;
    }
    size_t outputBits=0,statusBits=0;
    if(arm) {outputBits=Differences(output.ptr,baselineOutput.data(),output.n);statusBits=Differences(status.ptr,baselineStatus.data(),status.n);}
    else {baselineOutput.assign(output.ptr,output.ptr+output.n);baselineStatus.assign(status.ptr,status.ptr+status.n);}
    const bool ok=guards&&inputs&&!statusOracle&&!finiteBad&&!classBad&&!outputBits&&!statusBits;
    const std::string row="{\"arm\":"+std::to_string(arm)+",\"status\":\""+(ok?"passed":"failed")+
      "\",\"baseline_byte_comparison\":\""+(arm?"performed":"baseline_only")+
      "\",\"output_byte_differences\":"+std::to_string(outputBits)+",\"status_byte_differences\":"+std::to_string(statusBits)+
      ",\"oracle_status_byte_differences\":"+std::to_string(statusOracle)+",\"oracle_finite_failures\":"+std::to_string(finiteBad)+
      ",\"oracle_nonfinite_class_failures\":"+std::to_string(classBad)+",\"guards\":"+(guards?"true":"false")+
      ",\"inputs_unchanged\":"+(inputs?"true":"false")+",\"backend\":\"official_cpu_debug\",\"NPU\":\"not_run\"}";
    std::ofstream armReport(directory/"arm.json");armReport<<row<<'\n';if(!armReport)throw std::runtime_error("cannot save arm report");
    if(count++)records+=",";records+=row;std::cout<<"ROTATE_BATCH8_CPU_ARM "<<row<<std::endl;
    if(!ok){passed=false;break;}
  }
  const std::string report="{\"status\":\""+std::string(passed?"passed":"failed")+
    "\",\"selected_arm\":"+std::to_string(selected)+",\"rows\":"+std::to_string(rows)+
    ",\"grid\":"+std::to_string(grid)+",\"arms\":"+records+
    "],\"NPU\":\"not_run\",\"graph\":\"not_run\",\"performance\":\"not_measured\"}";
  std::ofstream reportFile(out/"report.json");reportFile<<report<<'\n';if(!reportFile)throw std::runtime_error("cannot save report");
  std::cout<<"ROTATE_BATCH8_CPU_RESULT "<<report<<std::endl;return passed?0:2;
}catch(const std::exception& e){std::cerr<<"ROTATE_BATCH8_CPU_ERROR "<<e.what()<<std::endl;return 2;}}
