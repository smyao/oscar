// Archive #21/#94/#125/#126/#145/#154, full D.4: actual CPU-debug, lossless bounded
// A/B layout change with fixed integer-exact operand values. Save all bytes before exact oracle/pair gates.
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
extern "C" void oscar_fp32_nd_matched_shapes_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,int64_t,int64_t);
extern "C" void oscar_fp32_ab_nz_matched_shapes_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,int64_t,int64_t);
extern "C" void oscar_fp32_b_nz_matched_shapes_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,int64_t,int64_t);
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
void ExactNz(const std::vector<uint8_t>& nd,const std::vector<uint8_t>& nz,size_t rows) {
  if(nd.size()!=rows*256*4||nz.size()!=nd.size())throw std::runtime_error("invalid bounded A/B bytes");
  for(size_t r=0;r<rows;++r)for(size_t c=0;c<256;++c) {
    const size_t packed=(c/8)*rows*8+r*8+c%8;
    if(Word(nd.data(),r*256+c)!=Word(nz.data(),packed))throw std::runtime_error("A/B NZ raw-bit inverse differs");
  }
}
size_t Diff(const uint8_t* a,const uint8_t* b,size_t n){size_t count=0;for(size_t i=0;i<n;++i)count+=a[i]!=b[i];return count;}
}
int main(int argc,char** argv){try {
  if(argc!=3)throw std::runtime_error("usage: oscar_fp32_nz_matched_shapes_cpu FIXTURE NEW_OUTPUT");
  const Path fixture=argv[1],out=argv[2];int64_t m,pad,layout;
  std::ifstream shape(fixture/"shape.txt");shape>>m>>pad>>layout;
  if(!shape||!((m==126&&pad==128&&layout==0)||(m==24&&pad==32&&layout==1)))throw std::runtime_error("undeclared shape");
  if(std::filesystem::exists(out))throw std::runtime_error("refuse existing output");
  std::filesystem::create_directories(out);
  auto af=Read(fixture/"a_nd.bin",pad*256*4),az=Read(fixture/"a_nz.bin",pad*256*4);
  ExactNz(af,az,pad);
  auto bn=Read(fixture/"b_nd.bin",256*256*4),bz=Read(fixture/"b_nz.bin",256*256*4);
  ExactNz(bn,bz,256);
  for(size_t i=m*256;i<pad*256;++i)if(Word(af.data(),i)!=0)throw std::runtime_error("input padding is not +0");
  auto expected=Read(fixture/"expected_output.bin",pad*256*4);std::vector<uint8_t> baseline;
  AscendC::SetKernelMode(KernelMode::AIC_MODE);bool pass=true;std::string arms="[";
  for(int arm=0;arm<3;++arm) {
    const std::string name=arm==0?"nd":arm==1?"b_nz":"ab_nz";const Path dir=out/name;std::filesystem::create_directory(dir);
    Gm a(pad*256*4),b(256*256*4),c(pad*256*4),status(128);
    a.Load(fixture/(arm==2?"a_nz.bin":"a_nd.bin"));b.Load(fixture/(arm==0?"b_nd.bin":"b_nz.bin"));
    if(arm==2){ICPU_RUN_KF(oscar_fp32_ab_nz_matched_shapes_kernel,1,a.ptr,b.ptr,c.ptr,status.ptr,m,layout);}
    else if(arm==1){ICPU_RUN_KF(oscar_fp32_b_nz_matched_shapes_kernel,1,a.ptr,b.ptr,c.ptr,status.ptr,m,layout);}
    else {ICPU_RUN_KF(oscar_fp32_nd_matched_shapes_kernel,1,a.ptr,b.ptr,c.ptr,status.ptr,m,layout);}
    a.Save(dir,"a_after");b.Save(dir,"b_after");c.Save(dir,"c");status.Save(dir,"status");
    const bool guards=a.Guards()&&b.Guards()&&c.Guards()&&status.Guards(),inputs=a.Unchanged()&&b.Unchanged();
    const int fields[11]={1,arm,static_cast<int>(m),static_cast<int>(layout),4,64,64,128,128,256,256};
    bool complete=true;for(int i=0;i<11;++i)complete=complete&&Word(status.ptr,i)==static_cast<uint32_t>(fields[i]);
    complete=complete&&Word(status.ptr,11)>0&&Word(status.ptr,11)<=512*1024&&Word(status.ptr,15)==0&&Word(status.ptr,25)==8&&Word(status.ptr,24)==static_cast<uint32_t>(arm!=0)&&Word(status.ptr,26)==static_cast<uint32_t>(arm==2)&&
      Word(status.ptr,27)==pad&&Word(status.ptr,28)==static_cast<uint32_t>(arm==2?pad:m)&&Word(status.ptr,29)==m;
    if(!arm)baseline.assign(c.ptr,c.ptr+c.n);
    const size_t oracle=Diff(c.ptr,expected.data(),c.n),pair=Diff(c.ptr,baseline.data(),c.n);
    const bool ok=guards&&inputs&&complete&&!oracle&&!pair;
    const std::string row="{\"arm\":\""+name+"\",\"status\":\""+(ok?"passed":"failed")+
      "\",\"exact_oracle_byte_differences\":"+std::to_string(oracle)+",\"pair_byte_differences\":"+std::to_string(pair)+
      ",\"guards\":"+(guards?"true":"false")+",\"inputs_unchanged\":"+(inputs?"true":"false")+
      ",\"completion\":"+(complete?"true":"false")+"}";
    std::ofstream ar(dir/"report.json");ar<<row<<'\n';if(!ar)throw std::runtime_error("cannot save arm report");
    if(arm)arms+=",";arms+=row;std::cout<<"NZ_MATCHED_CPU_ARM "<<row<<std::endl;
    if(!ok){pass=false;break;}
  }
  const std::string report="{\"status\":\""+std::string(pass?"exact_passed":"failed")+
    "\",\"backend\":\"official_cpu_debug\",\"arms\":"+arms+
    "],\"NPU\":\"not_run\",\"graph\":\"not_run\",\"performance\":\"not_measured\"}";
  std::ofstream result(out/"report.json");result<<report<<'\n';if(!result)throw std::runtime_error("cannot save report");
  std::cout<<"NZ_MATCHED_CPU_RESULT "<<report<<std::endl;return pass?0:2;
}catch(const std::exception& e){std::cerr<<"NZ_MATCHED_CPU_ERROR "<<e.what()<<std::endl;return 2;}}
