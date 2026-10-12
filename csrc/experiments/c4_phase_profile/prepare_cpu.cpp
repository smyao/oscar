// Archive #34/#36/#126/#129/#145/#148/#154/#155; startup D.4.
// Execute only the real prepare primitive; do not fabricate/filter task rows.
// No attention speed or NPU claim. CPU metadata export has a separate 120s cap.
#include "tikicpulib.h"
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" void oscar_prepare_attention_tasks_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,bool,uint8_t*,int64_t,bool);
namespace {
std::vector<uint8_t> Read(const std::string& path,size_t n) {
  std::vector<uint8_t> bytes(n);std::ifstream file(path,std::ios::binary);
  if(!file.read(reinterpret_cast<char*>(bytes.data()),n) || file.peek()!=EOF)
    throw std::runtime_error("wrong fixture byte count: "+path);
  return bytes;
}
void Write(const std::string& path,const uint8_t* data,size_t n) {
  std::ofstream f(path,std::ios::binary);
  if(!f.write(reinterpret_cast<const char*>(data),n) || !f.flush())throw std::runtime_error("write failed: "+path);
}
struct Gm {
  uint8_t *base,*ptr;size_t size;
  explicit Gm(size_t n):base(static_cast<uint8_t*>(AscendC::GmAlloc(n+128))),ptr(base?base+64:nullptr),size(n) {
    if(!base)throw std::runtime_error("GmAlloc failed");std::memset(base,0xd7,n+128);std::memset(ptr,0x85,n);
  }
  Gm(const std::string& path,size_t n):Gm(n) {auto x=Read(path,n);std::memcpy(ptr,x.data(),n);}
  ~Gm(){AscendC::GmFree(base);}
  void Check() {for(size_t i=0;i<64;++i)if(base[i]!=0xd7 || ptr[size+i]!=0xd7)throw std::runtime_error("GM guard changed");}
  void Same(const std::string& path) {auto want=Read(path,size);if(std::memcmp(want.data(),ptr,size))throw std::runtime_error("input changed");}
};
}
int main(int argc,char** argv) {try {
  if(argc!=3)throw std::runtime_error("usage: prepare_cpu FIXTURE NEW_OUTPUT");
  const std::string dir=argv[1],out=argv[2];
  if(std::filesystem::exists(out))throw std::runtime_error("output must be new");
  int64_t a[17];std::ifstream shape(dir+"/shape.txt");for(auto& n:a)if(!(shape>>n))throw std::runtime_error("shape missing");
  const int64_t n=a[0];
  if((n!=336&&n!=1344)||a[1]!=6||a[2]!=1||a[3]!=256||a[4]!=1||a[6]!=3*n||a[13]!=64||a[14]!=256||a[15]!=3||a[16]!=1)
    throw std::runtime_error("not predeclared C4 shape");
  Gm starts(dir+"/starts.bin",8),lens(dir+"/lens.bin",4),slots(dir+"/slots.bin",n*8),table(dir+"/table.bin",a[5]*4);
  Gm tasks(3*n*16*8),positions(n*8);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(oscar_prepare_attention_tasks_kernel,20,starts.ptr,lens.ptr,slots.ptr,tasks.ptr,positions.ptr,
      int64_t{1},n,int64_t{6},int64_t{1},int64_t{64},int64_t{256},int64_t{1},true,table.ptr,a[5],false);
  for(auto* buffer:{&starts,&lens,&slots,&table,&tasks,&positions})buffer->Check();
  starts.Same(dir+"/starts.bin");lens.Same(dir+"/lens.bin");slots.Same(dir+"/slots.bin");table.Same(dir+"/table.bin");
  auto expected=Read(dir+"/expected_positions.bin",n*8);
  if(std::memcmp(expected.data(),positions.ptr,positions.size))throw std::runtime_error("logical positions differ");
  std::filesystem::create_directories(out);
  Write(out+"/tasks_before_suppression.bin",tasks.ptr,tasks.size);Write(out+"/positions.bin",positions.ptr,positions.size);
  const std::string receipt="{\"status\":\"real_prepare_kernel_completed\",\"cores\":20,\"all_positions_checked\":true,\"guards\":true}\n";
  Write(out+"/prepare_complete.json",reinterpret_cast<const uint8_t*>(receipt.data()),receipt.size());
  std::cout<<receipt;return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;} }
