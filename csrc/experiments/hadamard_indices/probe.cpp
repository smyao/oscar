// Archive #13-22/#126/#154: independent index formula, bitwise signs and guards.
// CPU-debug qualifies metadata only, not hardware performance or full rotation.
#include "tikicpulib.h"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
extern "C" void oscar_hadamard_indices_old(uint8_t*,uint8_t*,int32_t);
extern "C" void oscar_hadamard_indices_vector(uint8_t*,uint8_t*,int32_t);
struct Gm {
  uint8_t* base;uint8_t* data;
  Gm(){base=static_cast<uint8_t*>(AscendC::GmAlloc(8192+128));if(!base)throw std::runtime_error("allocation");data=base+64;std::memset(base,0xa5,8192+128);}
  ~Gm(){AscendC::GmFree(base);}
  void save(const std::filesystem::path& p){std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<char*>(base),8192+128);if(!f)throw std::runtime_error("save");}
  uint32_t word(int j){uint32_t v;std::memcpy(&v,data+4*j,4);return v;}
  bool guarded(){for(int j=0;j<64;++j)if(base[j]!=0xa5||data[8192+j]!=0xa5)return false;return true;}
};
int main(int argc,char**argv){try{
  if(argc!=2)throw std::runtime_error("usage: CPU NEW_OUTPUT");
  std::filesystem::path out=argv[1];if(std::filesystem::exists(out))throw std::runtime_error("existing output");
  std::filesystem::create_directories(out);AscendC::SetKernelMode(KernelMode::AIV_MODE);
  for(int d:{64,128,256}){
    Gm oi,os,ni,ns;
    ICPU_RUN_KF(oscar_hadamard_indices_old,1,oi.data,os.data,d);
    ICPU_RUN_KF(oscar_hadamard_indices_vector,1,ni.data,ns.data,d);
    std::string prefix=std::to_string(d);
    oi.save(out/(prefix+"-old-indices.bin"));os.save(out/(prefix+"-old-signs.bin"));
    ni.save(out/(prefix+"-new-indices.bin"));ns.save(out/(prefix+"-new-signs.bin"));
    if(!oi.guarded()||!os.guarded()||!ni.guarded()||!ns.guarded()||std::memcmp(oi.data,ni.data,8192)||std::memcmp(os.data,ns.data,8192))throw std::runtime_error("bitwise/guard failure D"+prefix);
    int stage=0;
    for(int dist=1;dist<d;dist*=2,++stage)for(int i=0;i<d;++i)
      if(ni.word(stage*d+i)!=uint32_t((i^dist)*4)||ns.word(stage*d+i)!=uint32_t((i&dist)?0xbf800000:0x3f800000))throw std::runtime_error("independent formula");
    for(int i=stage*d;i<2048;++i)if(ni.word(i)!=0xa5a5a5a5U||ns.word(i)!=0xa5a5a5a5U)throw std::runtime_error("padding");
    std::cout<<"HADAMARD_INDICES D="<<d<<" bitwise=passed formula=passed guards=passed\n";
  }return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 2;}}
