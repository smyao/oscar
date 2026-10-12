// SPDX-License-Identifier: Apache-2.0
// Archive #21/#94/#125/#126/#140/#145/#154 and full D.4: CPU is a falsifier,
// not NPU acceptance. Each launch owns one Matmul. Save all actual bytes
// before checking; any live bit difference rejects B/C immediately. No
// timing claims, retries, tolerance expansion or dual-Matmul allocation.
#include "tikicpulib.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#define DECL(NAME) extern "C" void NAME(uint8_t*,uint8_t*,uint8_t*,uint8_t*,int64_t,int64_t)
DECL(oscar_decode_mmad_a_kernel);DECL(oscar_decode_mmad_b_kernel);DECL(oscar_decode_mmad_c_kernel);
#undef DECL
namespace {
constexpr size_t Rows=32,Cols=256,Guard=64;
std::vector<uint8_t> Read(const std::filesystem::path& p,size_t n) {
  std::vector<uint8_t> data(n);std::ifstream f(p,std::ios::binary);
  if(!f.read(reinterpret_cast<char*>(data.data()),n)||f.peek()!=EOF)throw std::runtime_error("input size: "+p.string());
  return data;
}
struct Gm {
  uint8_t* base;uint8_t* ptr;size_t n;std::vector<uint8_t> initial;
  explicit Gm(size_t size):n(size),initial(size,0xAD) {
    base=static_cast<uint8_t*>(AscendC::GmAlloc(n+2*Guard));
    if(!base)throw std::runtime_error("GmAlloc failed");ptr=base+Guard;
    std::memset(base,0xA5,n+2*Guard);std::memcpy(ptr,initial.data(),n);
  }
  ~Gm(){AscendC::GmFree(base);}Gm(const Gm&)=delete;
  void Load(const std::filesystem::path& p){initial=Read(p,n);std::memcpy(ptr,initial.data(),n);}
  bool Guards()const{for(size_t i=0;i<Guard;++i)if(base[i]!=0xA5||ptr[n+i]!=0xA5)return false;return true;}
  bool Unchanged()const{return !std::memcmp(ptr,initial.data(),n);}
  void Save(const std::filesystem::path& dir,const std::string& name)const {
    std::ofstream f(dir/(name+".bin"),std::ios::binary),g(dir/(name+".guarded.bin"),std::ios::binary);
    f.write(reinterpret_cast<const char*>(ptr),n);g.write(reinterpret_cast<const char*>(base),n+2*Guard);
    if(!f||!g)throw std::runtime_error("cannot save "+name);
  }
};
uint32_t Word(const uint8_t* p,size_t i){uint32_t x;std::memcpy(&x,p+4*i,4);return x;}
float Value(const uint8_t* p,size_t i){float x;std::memcpy(&x,p+4*i,4);return x;}
}
int main(int argc,char** argv) {try {
  if(argc!=3)throw std::runtime_error("usage: oscar_decode_mmad_cpu FIXTURE NEW_OUTPUT");
  std::filesystem::path dir=argv[1],out=argv[2];int64_t live,layout;
  std::ifstream shape(dir/"shape.txt");shape>>live>>layout;
  if(!shape||(live!=18&&live!=24)||layout<0||layout>2)throw std::runtime_error("shape must be M18/M24 and layout0/1/2");
  if(std::filesystem::exists(out))throw std::runtime_error("refuse existing output");
  std::filesystem::create_directories(out);
  Gm a(Rows*Cols*4),b(Cols*Cols*4);a.Load(dir/"a.bin");b.Load(dir/"b.bin");
  for(size_t i=live*Cols;i<Rows*Cols;++i)if(Word(a.ptr,i)!=0)throw std::runtime_error("A padding must be +0 bits");
  const auto expected=Read(dir/"expected.bin",Rows*Cols*4);
  std::vector<uint8_t> baseline;bool passed=true;std::string arms="[";
  AscendC::SetKernelMode(KernelMode::AIC_MODE);
  for(int arm=0;arm<3;++arm) {
    const auto armDir=out/std::to_string(arm);std::filesystem::create_directory(armDir);
    Gm c(Rows*Cols*4),status(64);
    if(arm==0){ICPU_RUN_KF(oscar_decode_mmad_a_kernel,1,a.ptr,b.ptr,c.ptr,status.ptr,live,layout);}
    else if(arm==1){ICPU_RUN_KF(oscar_decode_mmad_b_kernel,1,a.ptr,b.ptr,c.ptr,status.ptr,live,layout);}
    else {ICPU_RUN_KF(oscar_decode_mmad_c_kernel,1,a.ptr,b.ptr,c.ptr,status.ptr,live,layout);}
    a.Save(armDir,"a_after");b.Save(armDir,"b_after");c.Save(armDir,"c");status.Save(armDir,"status");
    bool guards=a.Guards()&&b.Guards()&&c.Guards()&&status.Guards(),inputs=a.Unchanged()&&b.Unchanged();
    const int fields[9]={1,arm,static_cast<int>(live),arm==2?32:static_cast<int>(live),static_cast<int>(layout),arm==0?64:32,arm==2?128:64,arm==2?64:128,arm==0?128:32};
    bool complete=true;for(int i=0;i<9;++i)complete=complete&&Word(status.ptr,i)==static_cast<uint32_t>(fields[i]);
    complete=complete&&Word(status.ptr,9)>0&&Word(status.ptr,9)<=512*1024&&Word(status.ptr,15)==0;
    size_t bits=0,oracleBad=0,paddingBad=0,first=Rows*Cols;
    if(arm==0)baseline.assign(c.ptr,c.ptr+c.n);
    for(size_t i=0;i<live*Cols;++i) {
      if(Word(c.ptr,i)!=Word(baseline.data(),i)){++bits;if(first==Rows*Cols)first=i;}
      const float got=Value(c.ptr,i),want=Value(expected.data(),i);
      // This independent oracle is diagnostic in addition to the stricter
      // old/new bit gate. Never turn a bit failure into a tolerance success.
      if(!std::isfinite(got)||!std::isfinite(want)||std::abs(got-want)>0.005F+0.005F*std::abs(want))++oracleBad;
    }
    for(size_t i=live*Cols;i<Rows*Cols;++i) {
      if(arm==2?Value(c.ptr,i)!=0.0F:Word(c.ptr,i)!=0xADADADADU)++paddingBad;
    }
    const bool ok=guards&&inputs&&complete&&!bits&&!oracleBad&&!paddingBad;
    const std::string row="{\"arm\":"+std::to_string(arm)+",\"status\":\""+(ok?"passed":"failed")+
      "\",\"live_bit_differences\":"+std::to_string(bits)+",\"oracle_failures\":"+std::to_string(oracleBad)+
      ",\"padding_failures\":"+std::to_string(paddingBad)+",\"guards\":"+(guards?"true":"false")+
      ",\"inputs_unchanged\":"+(inputs?"true":"false")+",\"completion\":"+(complete?"true":"false")+
      ",\"first_bit_difference\":"+(first==Rows*Cols?"null":std::to_string(first))+"}";
    std::ofstream armReport(armDir/"report.json");armReport<<row<<'\n';
    if(!armReport)throw std::runtime_error("cannot save arm report");
    if(arm)arms+=",";arms+=row;std::cout<<"MMAD_BARRIER_CPU_ARM "<<row<<std::endl;
    if(!ok)passed=false;
    // Preserve an independent-oracle failure while still allowing the first
    // old/new bit comparison. A bit/guard/completion failure stops at once.
    if(!guards||!inputs||!complete||bits||paddingBad)break;
  }
  const std::string report="{\"status\":\""+std::string(passed?"bitwise_passed":"failed")+
    "\",\"backend\":\"official_cpu_debug\",\"arms\":"+arms+
    "],\"NPU\":\"not_run\",\"graph\":\"not_run\",\"performance\":\"not_measured\"}";
  std::ofstream result(out/"report.json");result<<report<<'\n';if(!result)throw std::runtime_error("cannot save report");
  std::cout<<"MMAD_BARRIER_CPU_RESULT "<<report<<std::endl;return passed?0:2;
}catch(const std::exception& e){std::cerr<<"MMAD_BARRIER_CPU_ERROR "<<e.what()<<std::endl;return 2;}}
