// Archive #21/#94/#125/#126/#145/#154, full D.4: actual CPU-debug, fixed
// integer-exact operand subset. Save all bytes before exact oracle/pair gates.
// No altered production input, reduced accumulator precision or speed claim.
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
extern "C" void oscar_mmad_ratio_fp32_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,int64_t,int64_t);
extern "C" void oscar_mmad_ratio_bf16_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,int64_t,int64_t);
namespace {
using Path=std::filesystem::path;
std::vector<uint8_t> Read(const Path& path,size_t n) {
  std::vector<uint8_t> x(n);std::ifstream f(path,std::ios::binary);
  if(!f.read(reinterpret_cast<char*>(x.data()),n)||f.peek()!=EOF)throw std::runtime_error("input byte count: "+path.string());return x;
}
struct Gm {
  uint8_t* base;uint8_t* ptr;size_t n;std::vector<uint8_t> initial;
  explicit Gm(size_t size):n(size),initial(size,0x85) {
    base=static_cast<uint8_t*>(AscendC::GmAlloc(n+128));if(!base)throw std::runtime_error("GmAlloc failed");
    ptr=base+64;std::memset(base,0xa5,n+128);std::memcpy(ptr,initial.data(),n);
  }
  ~Gm(){AscendC::GmFree(base);}Gm(const Gm&)=delete;
  void Load(const Path& p){initial=Read(p,n);std::memcpy(ptr,initial.data(),n);}
  bool Guards()const{for(size_t i=0;i<64;++i)if(base[i]!=0xa5||ptr[n+i]!=0xa5)return false;return true;}
  bool Unchanged()const{return !std::memcmp(ptr,initial.data(),n);}
  void Save(const Path& dir,const std::string& name)const {
    std::ofstream f(dir/(name+".bin"),std::ios::binary),g(dir/(name+".guarded.bin"),std::ios::binary);
    f.write(reinterpret_cast<const char*>(ptr),n);g.write(reinterpret_cast<const char*>(base),n+128);
    if(!f||!g)throw std::runtime_error("cannot save "+name);
  }
};
uint32_t Word(const uint8_t* p,size_t i){uint32_t x;std::memcpy(&x,p+4*i,4);return x;}
void ExactEncoding(const std::vector<uint8_t>& f,const std::vector<uint8_t>& b) {
  if(f.size()!=2*b.size())throw std::runtime_error("encoding sizes differ");
  for(size_t i=0;i<f.size()/4;++i) {
    uint32_t bits=Word(f.data(),i);uint16_t half;std::memcpy(&half,b.data()+2*i,2);
    float value;std::memcpy(&value,&bits,4);
    if((bits&65535)||bits>>16!=half||!std::isfinite(value)||std::abs(value)>1.0F||
        value*64!=static_cast<int32_t>(value*64))throw std::runtime_error("not exact finite integer/64 operands");
  }
}
size_t Diff(const uint8_t* a,const uint8_t* b,size_t n){size_t count=0;for(size_t i=0;i<n;++i)count+=a[i]!=b[i];return count;}
}
int main(int argc,char** argv){try {
  if(argc!=3)throw std::runtime_error("usage: oscar_mmad_ratio_cpu FIXTURE NEW_OUTPUT");
  const Path fixture=argv[1],out=argv[2];int64_t m,pad,layout;
  std::ifstream shape(fixture/"shape.txt");shape>>m>>pad>>layout;
  if(!shape||!((m==24&&pad==32&&layout==0)||(m==126&&pad==128&&layout==1)))throw std::runtime_error("undeclared shape");
  if(std::filesystem::exists(out))throw std::runtime_error("refuse existing output");
  std::filesystem::create_directories(out);
  auto af=Read(fixture/"a_fp32.bin",pad*256*4),ab=Read(fixture/"a_bf16.bin",pad*256*2);
  auto bf=Read(fixture/"b_fp32.bin",256*256*4),bb=Read(fixture/"b_bf16.bin",256*256*2);
  ExactEncoding(af,ab);ExactEncoding(bf,bb);
  for(size_t i=m*256;i<pad*256;++i)if(Word(af.data(),i)!=0)throw std::runtime_error("input padding is not +0");
  auto expected=Read(fixture/"expected_output.bin",pad*256*4);std::vector<uint8_t> baseline;
  AscendC::SetKernelMode(KernelMode::AIC_MODE);bool pass=true;std::string arms="[";
  for(int arm=0;arm<2;++arm) {
    const std::string name=arm?"bf16":"fp32";const Path dir=out/name;std::filesystem::create_directory(dir);
    Gm a(pad*256*(arm?2:4)),b(256*256*(arm?2:4)),c(pad*256*4),status(64);
    a.Load(fixture/("a_"+name+".bin"));b.Load(fixture/("b_"+name+".bin"));
    if(arm){ICPU_RUN_KF(oscar_mmad_ratio_bf16_kernel,1,a.ptr,b.ptr,c.ptr,status.ptr,m,layout);}
    else {ICPU_RUN_KF(oscar_mmad_ratio_fp32_kernel,1,a.ptr,b.ptr,c.ptr,status.ptr,m,layout);}
    a.Save(dir,"a_after");b.Save(dir,"b_after");c.Save(dir,"c");status.Save(dir,"status");
    const bool guards=a.Guards()&&b.Guards()&&c.Guards()&&status.Guards(),inputs=a.Unchanged()&&b.Unchanged();
    const int fields[11]={1,arm,static_cast<int>(m),static_cast<int>(layout),arm?2:4,64,64,128,128,256,256};
    bool complete=true;for(int i=0;i<11;++i)complete=complete&&Word(status.ptr,i)==static_cast<uint32_t>(fields[i]);
    complete=complete&&Word(status.ptr,11)>0&&Word(status.ptr,11)<=512*1024&&Word(status.ptr,15)==0;
    if(!arm)baseline.assign(c.ptr,c.ptr+c.n);
    const size_t oracle=Diff(c.ptr,expected.data(),c.n),pair=Diff(c.ptr,baseline.data(),c.n);
    const bool ok=guards&&inputs&&complete&&!oracle&&!pair;
    const std::string row="{\"arm\":\""+name+"\",\"status\":\""+(ok?"passed":"failed")+
      "\",\"exact_oracle_byte_differences\":"+std::to_string(oracle)+",\"pair_byte_differences\":"+std::to_string(pair)+
      ",\"guards\":"+(guards?"true":"false")+",\"inputs_unchanged\":"+(inputs?"true":"false")+
      ",\"completion\":"+(complete?"true":"false")+"}";
    std::ofstream ar(dir/"report.json");ar<<row<<'\n';if(!ar)throw std::runtime_error("cannot save arm report");
    if(arm)arms+=",";arms+=row;std::cout<<"MMAD_RATIO_CPU_ARM "<<row<<std::endl;
    if(!ok){pass=false;break;}
  }
  const std::string report="{\"status\":\""+std::string(pass?"exact_passed":"failed")+
    "\",\"backend\":\"official_cpu_debug\",\"arms\":"+arms+
    "],\"NPU\":\"not_run\",\"graph\":\"not_run\",\"performance\":\"not_measured\"}";
  std::ofstream result(out/"report.json");result<<report<<'\n';if(!result)throw std::runtime_error("cannot save report");
  std::cout<<"MMAD_RATIO_CPU_RESULT "<<report<<std::endl;return pass?0:2;
}catch(const std::exception& e){std::cerr<<"MMAD_RATIO_CPU_ERROR "<<e.what()<<std::endl;return 2;}}
