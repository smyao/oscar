// Research A/B/C full writer byte/guard oracle; archive #126/#145/#155.
// SPDX-License-Identifier: Apache-2.0
// Archive #86/#97/#111/#126/#145/#155/D.4: bitwise old/active/identity store
// and independent oracle, including poison/NaN/signed zero. CPU != NPU proof.
#include "tikicpulib.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#define STORE_ARGS uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,int64_t,int64_t,int64_t,int32_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,float,float,bool
extern "C" void oscar_rotate_clip_store_striped_kernel(STORE_ARGS);
extern "C" void oscar_rotate_clip_store_striped_simd_pack_kernel(STORE_ARGS);
extern "C" void oscar_rotate_clip_store_striped_simd_pack_init_kernel(STORE_ARGS);
#undef STORE_ARGS
namespace {
constexpr size_t kGuard=64;
std::vector<uint8_t> Read(const std::string& path,size_t n) {
  std::ifstream file(path,std::ios::binary|std::ios::ate);
  if(!file || file.tellg()!=static_cast<std::streamoff>(n))throw std::runtime_error("size mismatch: "+path);
  file.seekg(0);std::vector<uint8_t> bytes(n);
  if(!file.read(reinterpret_cast<char*>(bytes.data()),n))throw std::runtime_error("read failed: "+path);
  return bytes;
}
void Equal(const uint8_t* got,const uint8_t* want,size_t n,const std::string& label) {
  for(size_t i=0;i<n;++i)if(got[i]!=want[i])throw std::runtime_error(label+" byte="+std::to_string(i)+
      " got="+std::to_string(got[i])+" expected="+std::to_string(want[i]));
}
struct Gm {
  uint8_t* base;uint8_t* ptr;size_t size;std::vector<uint8_t> initial;
  Gm(const std::string& path,size_t n):base(static_cast<uint8_t*>(AscendC::GmAlloc(n+2*kGuard))),ptr(base+kGuard),size(n),initial(Read(path,n)) {
    if(!base)throw std::runtime_error("allocation failed");std::memset(base,0xa5,kGuard);std::memcpy(ptr,initial.data(),n);std::memset(ptr+n,0xa5,kGuard);
  }
  ~Gm(){AscendC::GmFree(base);} Gm(const Gm&)=delete;
  void Reset(){std::memcpy(ptr,initial.data(),size);std::memset(ptr+size,0xa5,kGuard);}
  void Guard(const std::string& label)const {for(size_t j=0;j<kGuard;++j)if(base[j]!=0xa5||ptr[size+j]!=0xa5)throw std::runtime_error(label+" suffix guard");}
};
}
int main(int argc,char** argv) {
  try {
    if(argc!=3)throw std::runtime_error("usage: store_cpu CASE_DIRECTORY NEW_OUTPUT");
    const std::filesystem::path out=argv[2];
    if(std::filesystem::exists(out))throw std::runtime_error("refuse existing output");
    std::filesystem::create_directories(out);
    const std::string dir=argv[1];std::ifstream shape(dir+"/shape.txt");
    int64_t n,h,d,bt,blocks,offset,stride,ks,vs,ts,sink,recent;int32_t dtype;float kc,vc;bool hadamard;
    if(!(shape>>n>>h>>d>>dtype>>bt>>blocks>>offset>>stride>>ks>>vs>>ts>>sink>>recent>>kc>>vc>>hadamard)
        || n<=0 || h<=0 || (d!=64&&d!=128&&d!=256) || (dtype!=1&&dtype!=2))
      throw std::runtime_error("unsupported store fixture shape");
    Gm k(dir+"/key.bin",n*h*d*2),v(dir+"/value.bin",n*h*d*2),rk(dir+"/rk.bin",d*d*4),rv(dir+"/rv.bin",d*d*4),
       slots(dir+"/slots.bin",n*8),positions(dir+"/positions.bin",n*8),
       packed(dir+"/initial_packed.bin",offset+blocks*stride),rawK(dir+"/initial_raw_key.bin",blocks*ks*2),
       rawV(dir+"/initial_raw_value.bin",blocks*vs*2),tags(dir+"/initial_tags.bin",blocks*ts*8),
       status(dir+"/initial_status.bin",n*h*4);
    if(!hadamard)for(int64_t row=0;row<d;++row)for(int64_t col=0;col<d;++col) {
      float x,y;std::memcpy(&x,rk.ptr+(row*d+col)*4,4);std::memcpy(&y,rv.ptr+(row*d+col)*4,4);
      if(x!=(row==col?1.F:0.F)||y!=x)throw std::runtime_error("identity fixture matrix mismatch");
    }
    const std::vector<Gm*> outputs{&packed,&rawK,&rawV,&tags,&status};
    const std::vector<std::string> names{"packed","raw_key","raw_value","tags","status"};
    std::vector<std::vector<uint8_t>> baseline;
    const uint32_t grid=static_cast<uint32_t>(std::min<int64_t>(32,(n*h+7)/8));
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    for(int variant=0;variant<3;++variant) {
      for(auto* gm:outputs)gm->Reset();
      std::cout<<"STORE_CPU_START case="<<dir<<" variant="<<variant<<std::endl;
#define ARGS k.ptr,v.ptr,rk.ptr,rv.ptr,slots.ptr,positions.ptr,packed.ptr,rawK.ptr,rawV.ptr,tags.ptr,status.ptr,n,h,d,dtype,bt,blocks,offset,stride,ks,vs,ts,sink,recent,kc,vc,hadamard
      if(variant==0) {ICPU_RUN_KF(oscar_rotate_clip_store_striped_kernel,grid,ARGS);}
      else if(variant==1) {ICPU_RUN_KF(oscar_rotate_clip_store_striped_simd_pack_kernel,grid,ARGS);}
      else {ICPU_RUN_KF(oscar_rotate_clip_store_striped_simd_pack_init_kernel,grid,ARGS);}
#undef ARGS
      const auto arm=out/std::to_string(variant);std::filesystem::create_directory(arm);
      const std::vector<Gm*> all{&k,&v,&rk,&rv,&slots,&positions,&packed,&rawK,&rawV,&tags,&status};
      const std::vector<std::string> labels{"key","value","rk","rv","slots","positions","packed","raw_key","raw_value","tags","status"};
      for(size_t i=0;i<all.size();++i) {
        std::ofstream file(arm/(labels[i]+".guarded.bin"),std::ios::binary);
        file.write(reinterpret_cast<char*>(all[i]->base),all[i]->size+2*kGuard);
        if(!file)throw std::runtime_error("cannot save guarded output");
      }
      for(const Gm* gm:{&k,&v,&rk,&rv,&slots,&positions}) {
        gm->Guard("input");Equal(gm->ptr,gm->initial.data(),gm->size,"input bytes changed");
      }
      for(size_t i=0;i<outputs.size();++i) {
        outputs[i]->Guard(names[i]);const auto oracle=Read(dir+"/expected_"+names[i]+".bin",outputs[i]->size);
        Equal(outputs[i]->ptr,oracle.data(),outputs[i]->size,"oracle "+names[i]);
        if(variant==0)baseline.emplace_back(outputs[i]->ptr,outputs[i]->ptr+outputs[i]->size);
        else Equal(outputs[i]->ptr,baseline[i].data(),outputs[i]->size,"old bitwise "+names[i]);
      }
      std::cout<<"STORE_CPU_RESULT {\"status\":\"passed\",\"variant\":"<<variant
        <<",\"tokens\":"<<n<<",\"heads\":"<<h<<",\"dim\":"<<d<<",\"dtype\":"<<dtype
        <<",\"grid\":"<<grid<<",\"all_bytes\":\"identical\",\"NPU\":\"not_run\"}"<<std::endl;
    }
    return 0;
  } catch(const std::exception& e){std::cerr<<"STORE_CPU_FAILED "<<e.what()<<std::endl;return 1;}
}
