// Archive #13-22/#126/#145/#154/#155; full D.4: exact helper before timing.
// Official CPU-debug only, all output bits A/B and independent finite oracle.
#include "tikicpulib.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
extern "C" void oscar_unpack16_helper(uint8_t*,uint8_t*,uint8_t*,uint8_t*,int32_t);
extern "C" void oscar_unpack32_helper(uint8_t*,uint8_t*,uint8_t*,uint8_t*,int32_t);
std::vector<uint8_t> Read(const std::filesystem::path& p,size_t n){
  std::vector<uint8_t> b(n);std::ifstream f(p,std::ios::binary);
  if(!f.read(reinterpret_cast<char*>(b.data()),n)||f.peek()!=EOF)throw std::runtime_error("input size "+p.string());return b;
}
struct Gm {
  uint8_t *base,*data;size_t n;
  explicit Gm(size_t size):base(static_cast<uint8_t*>(AscendC::GmAlloc(size+128))),data(base?base+64:nullptr),n(size){
    if(!base)throw std::runtime_error("GmAlloc");std::memset(base,0xa5,n+128);
  }
  Gm(const std::filesystem::path& p,size_t size):Gm(size){auto b=Read(p,n);std::memcpy(data,b.data(),n);}
  ~Gm(){AscendC::GmFree(base);}
  void check(){for(size_t i=0;i<64;++i)if(base[i]!=0xa5||data[n+i]!=0xa5)throw std::runtime_error("guard");}
  void save(const std::filesystem::path& p){std::ofstream f(p,std::ios::binary);if(!f.write(reinterpret_cast<char*>(data),n)||!f.flush())throw std::runtime_error("save");}
};
int main(int argc,char**argv){try {
  if(argc!=3)throw std::runtime_error("usage: CPU FIXTURE NEW_OUTPUT");
  std::filesystem::path fixture=argv[1],out=argv[2];if(std::filesystem::exists(out))throw std::runtime_error("output exists");
  int32_t cases=0;std::ifstream shape(fixture/"shape.txt");if(!(shape>>cases)||cases<1||cases>256)throw std::runtime_error("case bound");
  Gm packed(fixture/"packed.bin",cases*5120),live(fixture/"live.bin",cases*4),old(cases*65536),fresh(cases*65536),os(cases*4),ns(cases*4);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  std::filesystem::create_directories(out);
  ICPU_RUN_KF(oscar_unpack16_helper,1,packed.data,live.data,old.data,os.data,cases);
  old.save(out/"old.bin");os.save(out/"old_status.bin");
  ICPU_RUN_KF(oscar_unpack32_helper,1,packed.data,live.data,fresh.data,ns.data,cases);
  fresh.save(out/"new.bin");ns.save(out/"new_status.bin");
  for(Gm* g:{&packed,&live,&old,&fresh,&os,&ns})g->check();
  if(std::memcmp(old.data,fresh.data,old.n)||std::memcmp(os.data,ns.data,os.n))throw std::runtime_error("A/B byte mismatch");
  auto input=Read(fixture/"packed.bin",packed.n),lives=Read(fixture/"live.bin",live.n);
  if(std::memcmp(input.data(),packed.data,packed.n)||std::memcmp(lives.data(),live.data,live.n))throw std::runtime_error("input modified");
  auto expected=Read(fixture/"expected.bin",fresh.n),status=Read(fixture/"expected_status.bin",ns.n);
  if(std::memcmp(status.data(),ns.data,ns.n))throw std::runtime_error("independent status oracle");
  for(size_t offset=0;offset<fresh.n;offset+=4){
    float a,b;std::memcpy(&a,fresh.data+offset,4);std::memcpy(&b,expected.data()+offset,4);
    if(std::isnan(b)){if(!std::isnan(a))throw std::runtime_error("NaN class");}
    else if(std::memcmp(fresh.data+offset,expected.data()+offset,4))throw std::runtime_error("independent FP32 oracle at "+std::to_string(offset));
  }
  std::ofstream report(out/"report.json");report<<"{\"status\":\"passed\",\"cases\":"<<cases<<",\"A_B_bitwise\":true,\"independent_oracle\":true,\"guards\":true,\"input_immutable\":true,\"NPU\":\"not_run\"}\n";
  std::cout<<"UNPACK32_CPU cases="<<cases<<" bitwise=passed oracle=passed guards=passed\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
