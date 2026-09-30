// Archive G26-G34/#13-22/#37-49/#126/#154, startup D.4: the accepted
// store must match the independent PR golden, and striped bytes must invert
// exactly to that accepted slot while all status/window/tag/guard bytes agree.
// CPU simulation is a precision contract, never NPU performance acceptance.
#include "tikicpulib.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void oscar_rotate_clip_store_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    int64_t,int64_t,int64_t,int32_t,int64_t,int64_t,int64_t,int64_t,
    int64_t,int64_t,int64_t,int64_t,int64_t,float,float,bool);
extern "C" void oscar_rotate_clip_store_striped_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    int64_t,int64_t,int64_t,int32_t,int64_t,int64_t,int64_t,int64_t,
    int64_t,int64_t,int64_t,int64_t,int64_t,float,float,bool);

namespace {
constexpr uint8_t kGuard=173;

std::vector<uint8_t> Read(const std::string& path,size_t n) {
  std::vector<uint8_t> data(n);
  std::ifstream file(path,std::ios::binary);
  if (!file.read(reinterpret_cast<char*>(data.data()),n) || file.peek()!=EOF)
    throw std::runtime_error("golden/input byte count mismatch: "+path);
  return data;
}

struct Gm {
  uint8_t* ptr; size_t size;
  Gm(size_t n,uint8_t fill):ptr(static_cast<uint8_t*>(AscendC::GmAlloc(n))),size(n) {
    if (!ptr) throw std::runtime_error("GmAlloc failed");
    std::memset(ptr,fill,n);
  }
  Gm(const std::string& path,size_t n):Gm(n,0) {
    auto bytes=Read(path,n);std::memcpy(ptr,bytes.data(),n);
  }
  ~Gm() {AscendC::GmFree(ptr);}
  Gm(const Gm&)=delete;
};

void Equal(const uint8_t* actual,const uint8_t* expected,size_t length,
           const std::string& label) {
  for(size_t i=0;i<length;++i) if(actual[i]!=expected[i])
    throw std::runtime_error(label+" byte="+std::to_string(i)+
        " expected="+std::to_string(expected[i])+" actual="+
        std::to_string(actual[i]));
}

void Golden(const Gm& actual,const std::string& dir,const std::string& file) {
  auto expected=Read(dir+"/expected_"+file+".bin",actual.size);
  Equal(actual.ptr,expected.data(),actual.size,file);
}

// Independent inverse of the new physical layout. The old independent PR
// golden remains the oracle; this inverse only maps physical bytes back.
std::vector<uint8_t> Canonicalize(const Gm& striped,const std::vector<uint8_t>& old,
                                  int64_t offset,int64_t blocks,int64_t stride,
                                  int64_t blockTokens,int64_t heads,int64_t dim) {
  const int64_t dataBytes=dim/4,slotBytes=dim/2+8,width=dim/8;
  std::vector<uint8_t> out(striped.ptr,striped.ptr+striped.size);
  for(int64_t page=0;page<blocks;++page) for(int64_t token=0;token<blockTokens;++token)
    for(int64_t head=0;head<heads;++head) {
      const int64_t base=offset+page*stride+(token*heads+head)*slotBytes;
      if(base<0 || base+slotBytes>static_cast<int64_t>(old.size()))
        throw std::runtime_error("slot address outside packed page");
      const bool untouched=std::all_of(old.begin()+base,old.begin()+base+slotBytes,
                                      [](uint8_t byte){return byte==kGuard;});
      if(untouched) continue;
      std::fill(out.begin()+base,out.begin()+base+slotBytes,0);
      for(int side=0;side<2;++side) {
        const int64_t src=base+side*dataBytes;
        const int64_t dst=base+side*(dataBytes+4);
        for(int64_t i=0;i<width;++i) {
          const uint16_t word=static_cast<uint16_t>(striped.ptr[src+2*i]) |
              (static_cast<uint16_t>(striped.ptr[src+2*i+1])<<8);
          for(int b=0;b<8;++b) {
            const int64_t index=b*width+i;
            out[dst+index/4] |= static_cast<uint8_t>(((word>>(2*b))&3)<<(2*(index%4)));
          }
        }
        for(int m=0;m<4;++m)
          out[dst+dataBytes+m]=striped.ptr[base+dim/2+side*4+m];
      }
    }
  return out;
}
}

int main(int argc,char**argv) {
  try {
    if(argc!=2) throw std::runtime_error("usage: oscar_striped_store_cpu golden_directory");
    const std::string dir=argv[1];std::ifstream shape(dir+"/shape.txt");
    int64_t n,h,d,b,nb,offset,stride,ks,vs,ts,sink,recent;
    int32_t dtype;float kc,vc;bool hadamard;
    if(!(shape>>n>>h>>d>>dtype>>b>>nb>>offset>>stride>>ks>>vs>>ts>>sink>>recent>>kc>>vc>>hadamard)
       || n<=0 || h<=0 || (d!=64 && d!=128 && d!=256) || b<=0 || nb<=0)
      throw std::runtime_error("invalid rotate_store shape");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    Gm key(dir+"/key.bin",n*h*d*(dtype==0?4:2));
    Gm value(dir+"/value.bin",n*h*d*(dtype==0?4:2));
    Gm rk(dir+"/rk.bin",d*d*4),rv(dir+"/rv.bin",d*d*4);
    Gm slots(dir+"/slots.bin",n*8),positions(dir+"/positions.bin",n*8);
    Gm oldPacked(offset+nb*stride,kGuard),newPacked(offset+nb*stride,kGuard);
    Gm oldRawK(nb*ks*2,kGuard),newRawK(nb*ks*2,kGuard);
    Gm oldRawV(nb*vs*2,kGuard),newRawV(nb*vs*2,kGuard);
    Gm oldTags(nb*ts*8,kGuard),newTags(nb*ts*8,kGuard);
    Gm oldStatus(n*h*4,0x85),newStatus(n*h*4,0x85);
#define OSCAR_RUN_STORE(KERNEL,PACKED,RAWK,RAWV,TAGS,STATUS) \
    ICPU_RUN_KF(KERNEL,4,key.ptr,value.ptr,rk.ptr,rv.ptr,slots.ptr,positions.ptr, \
      PACKED.ptr,RAWK.ptr,RAWV.ptr,TAGS.ptr,STATUS.ptr,n,h,d,dtype,b,nb,offset, \
      stride,ks,vs,ts,sink,recent,kc,vc,hadamard)
    OSCAR_RUN_STORE(oscar_rotate_clip_store_kernel,oldPacked,oldRawK,oldRawV,oldTags,oldStatus);
    OSCAR_RUN_STORE(oscar_rotate_clip_store_striped_kernel,newPacked,newRawK,newRawV,newTags,newStatus);
#undef OSCAR_RUN_STORE
    Golden(oldPacked,dir,"packed");Golden(oldRawK,dir,"raw_key");
    Golden(oldRawV,dir,"raw_value");Golden(oldTags,dir,"tags");Golden(oldStatus,dir,"status");
    Golden(newRawK,dir,"raw_key");Golden(newRawV,dir,"raw_value");
    Golden(newTags,dir,"tags");Golden(newStatus,dir,"status");
    auto oldBytes=Read(dir+"/expected_packed.bin",oldPacked.size);
    auto canonical=Canonicalize(newPacked,oldBytes,offset,nb,stride,b,h,d);
    Equal(canonical.data(),oldBytes.data(),oldBytes.size(),"striped canonicalized packed");
    std::cout<<"{\"backend\":\"ascendc_cpu_debug\",\"status\":\"passed\","
             <<"\"op\":\"rotate_clip_store_striped\",\"dim\":"<<d<<"}"<<std::endl;
    return 0;
  } catch(const std::exception& e) {std::cerr<<"CPU_DEBUG_FAILED: "<<e.what()<<std::endl;return 1;}
}
