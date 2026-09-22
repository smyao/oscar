// Archive G27/#13-22/#37-49/#111: run the real kernel, comparing packed bytes,
// untouched guard bytes, raw BF16 snapshots, page tags and every status row.
// CPU-debug execution is not NPU completion or graph/performance acceptance.
#include "tikicpulib.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void oscar_rotate_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    int64_t,int64_t,int32_t,bool,uint8_t*,int64_t);
extern "C" void oscar_rotate_clip_store_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    int64_t,int64_t,int64_t,int32_t,int64_t,int64_t,int64_t,int64_t,
    int64_t,int64_t,int64_t,int64_t,int64_t,float,float,bool);

std::vector<uint8_t> Read(const std::string& path,size_t n) {
  std::vector<uint8_t> data(n);std::ifstream file(path,std::ios::binary);
  if (!file.read(reinterpret_cast<char*>(data.data()),n) || file.peek()!=EOF)
    throw std::runtime_error("golden/input byte count mismatch: "+path);
  return data;
}
struct Gm {
  uint8_t* ptr;size_t size;
  Gm(size_t n,uint8_t fill):ptr(static_cast<uint8_t*>(AscendC::GmAlloc(n))),size(n) {
    if(!ptr) throw std::runtime_error("GmAlloc failed");std::memset(ptr,fill,n);
  }
  Gm(const std::string& path,size_t n):Gm(n,0) {auto v=Read(path,n);std::memcpy(ptr,v.data(),n);}
  ~Gm(){AscendC::GmFree(ptr);}Gm(const Gm&)=delete;
};
void Exact(const Gm& output,const std::string& path) {
  auto expected=Read(path,output.size);
  for(size_t i=0;i<output.size;i++) if(output.ptr[i]!=expected[i])
    throw std::runtime_error("exact byte mismatch "+path+" offset="+std::to_string(i)+
        " expected="+std::to_string(expected[i])+" actual="+std::to_string(output.ptr[i]));
}
void Close(const Gm& output,const std::string& path) {
  auto expected=Read(path,output.size);float maximum=0;
  for(size_t i=0;i<output.size/4;i++) {
    float a,b;std::memcpy(&a,output.ptr+i*4,4);std::memcpy(&b,expected.data()+i*4,4);
    if(a==b) continue;
    // Deliberately active-NaN cases require status 2 and matching NaNs;
    // finite goldens retain the unchanged frozen tolerance below.
    if(std::isnan(a) && std::isnan(b)) continue;
    maximum=std::max(maximum,std::abs(a-b));
    // Existing acceptance.json store/dequant gate, frozen before development.
    if(!std::isfinite(a)||!std::isfinite(b)||std::abs(a-b)>0.002F+0.002F*std::abs(b))
      throw std::runtime_error("rotation mismatch "+path+" index="+std::to_string(i)+
          " expected="+std::to_string(b)+" actual="+std::to_string(a));
  }
  std::cout<<"max_abs_error="<<maximum<<std::endl;
}
int main(int argc,char**argv) {
  try {
    if(argc!=3) throw std::runtime_error("usage: oscar_rotation_cpu rotate|rotate_store case_directory");
    std::string op=argv[1],dir=argv[2];std::ifstream meta(dir+"/shape.txt");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    if(op=="rotate") {
      int64_t rows,dim,heads;int32_t dtype;bool hadamard,masked;
      if(!(meta>>rows>>dim>>dtype>>hadamard>>heads>>masked) || heads<=0 || rows%heads)
        throw std::runtime_error("invalid rotate shape");
      Gm input(dir+"/input.bin",rows*dim*(dtype==0?4:2)),rotation(dir+"/rotation.bin",dim*dim*4);
      Gm output(rows*dim*4,0x85),status(rows*4,0x85);
      Gm slots((rows/heads)*8,0);
      if(masked) {auto bytes=Read(dir+"/slots.bin",slots.size);std::memcpy(slots.ptr,bytes.data(),slots.size);}
      ICPU_RUN_KF(oscar_rotate_kernel,4,input.ptr,rotation.ptr,output.ptr,status.ptr,rows,dim,dtype,hadamard,
                  masked ? slots.ptr : static_cast<uint8_t*>(nullptr),heads);
      Exact(status,dir+"/expected_status.bin");Close(output,dir+"/expected_output.bin");
    } else if(op=="rotate_store") {
      int64_t n,h,d,b,nb,offset,stride,ks,vs,ts,sink,recent;int32_t dtype;float kc,vc;bool hadamard;
      if(!(meta>>n>>h>>d>>dtype>>b>>nb>>offset>>stride>>ks>>vs>>ts>>sink>>recent>>kc>>vc>>hadamard))
        throw std::runtime_error("invalid rotate_store shape");
      Gm key(dir+"/key.bin",n*h*d*(dtype==0?4:2)),value(dir+"/value.bin",n*h*d*(dtype==0?4:2));
      Gm rk(dir+"/rk.bin",d*d*4),rv(dir+"/rv.bin",d*d*4),slots(dir+"/slots.bin",n*8),positions(dir+"/positions.bin",n*8);
      Gm packed(offset+nb*stride,173),rawKey(nb*ks*2,173),rawValue(nb*vs*2,173),tags(nb*ts*8,173),status(n*h*4,0x85);
      ICPU_RUN_KF(oscar_rotate_clip_store_kernel,4,key.ptr,value.ptr,rk.ptr,rv.ptr,slots.ptr,positions.ptr,
          packed.ptr,rawKey.ptr,rawValue.ptr,tags.ptr,status.ptr,n,h,d,dtype,b,nb,offset,stride,ks,vs,ts,sink,recent,kc,vc,hadamard);
      Exact(status,dir+"/expected_status.bin");Exact(packed,dir+"/expected_packed.bin");
      Exact(rawKey,dir+"/expected_raw_key.bin");Exact(rawValue,dir+"/expected_raw_value.bin");Exact(tags,dir+"/expected_tags.bin");
    } else throw std::runtime_error("unknown rotation op");
    std::cout<<"{\"backend\":\"ascendc_cpu_debug\",\"status\":\"passed\",\"op\":\""<<op<<"\"}"<<std::endl;
    return 0;
  } catch(const std::exception& e) {std::cerr<<"CPU_DEBUG_FAILED: "<<e.what()<<std::endl;return 1;}
}
