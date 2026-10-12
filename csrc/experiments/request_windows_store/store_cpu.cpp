// Archive G26-G34/#13-22/#37-49/#126/#145/#154/#155 and startup D.4:
// official CPU-debug compares unchanged math via equivalent baseline slots,
// independent PR packed goldens, exact BF16/tag bytes and all guard bytes.
// No device, graph or speed acceptance follows from this executable.
#include "tikicpulib.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" void oscar_rotate_clip_store_striped_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,int64_t,int64_t,int64_t,int32_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,float,float,bool);
extern "C" void oscar_request_packed_store_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,int64_t,int64_t,int64_t,int32_t,int64_t,int64_t,int64_t,int64_t,float,float,bool);
extern "C" void oscar_request_window_store_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,int64_t,int64_t,int64_t,int64_t,int64_t,int32_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);
namespace {
std::vector<uint8_t> Read(const std::string& path,size_t n) {
  std::ifstream file(path,std::ios::binary|std::ios::ate);
  if(!file||file.tellg()!=static_cast<std::streamoff>(n))throw std::runtime_error("byte size mismatch: "+path);
  file.seekg(0);std::vector<uint8_t> x(n);if(!file.read(reinterpret_cast<char*>(x.data()),n))throw std::runtime_error("read failed: "+path);return x;
}
void Equal(const uint8_t* got,const uint8_t* expected,size_t n,const std::string& label) {
  for(size_t i=0;i<n;++i)if(got[i]!=expected[i])throw std::runtime_error(label+" byte="+std::to_string(i)+" got="+std::to_string(got[i])+" expected="+std::to_string(expected[i]));
}
struct Gm {
  uint8_t* ptr;size_t size;std::vector<uint8_t> initial;
  explicit Gm(size_t n):ptr(static_cast<uint8_t*>(AscendC::GmAlloc(n+64))),size(n),initial(n,0xAD){if(!ptr)throw std::runtime_error("GmAlloc failed");std::memset(ptr,0xAD,n+64);}
  Gm(const std::string& path,size_t n):Gm(n){initial=Read(path,n);std::memcpy(ptr,initial.data(),n);}
  ~Gm(){AscendC::GmFree(ptr);}Gm(const Gm&)=delete;
  void Guard()const{for(size_t i=0;i<64;++i)if(ptr[size+i]!=0xAD)throw std::runtime_error("suffix guard changed");}
  void Golden(const std::string& dir,const std::string& name)const{auto e=Read(dir+"/expected_"+name+".bin",size);Equal(ptr,e.data(),size,name);Guard();}
};
}
int main(int argc,char**argv){try{
  if(argc!=2)throw std::runtime_error("usage: oscar_request_windows_store_cpu CASE_DIRECTORY");
  const std::string dir=argv[1];std::ifstream shape(dir+"/shape.txt");
  int64_t n,h,d,segment,blocks,offset,stride,requests,capacity,ks,vs,ts,sink,recent,bits;int32_t dtype;float kc,vc;bool hadamard;
  if(!(shape>>n>>h>>d>>dtype>>segment>>blocks>>offset>>stride>>requests>>capacity>>ks>>vs>>ts>>sink>>recent>>bits>>kc>>vc>>hadamard)
     ||n<=0||h<1||h>16||(d!=64&&d!=128&&d!=256)||segment<=0||blocks<=0||requests<1||requests>capacity
     ||dtype<0||dtype>2||sink<0||recent<1||sink+recent>1024||bits<1||bits>31
     ||stride<segment*h*(d/2+8)||ks<(sink+recent)*h*d||vs<(sink+recent)*h*d||ts<sink+recent)
    throw std::runtime_error("invalid fixture shape");
  Gm k(dir+"/key.bin",n*h*d*(dtype==0?4:2)),v(dir+"/value.bin",n*h*d*(dtype==0?4:2)),rk(dir+"/rk.bin",d*d*4),rv(dir+"/rv.bin",d*d*4);
  Gm slots(dir+"/slots.bin",n*8),oldSlots(dir+"/baseline_slots.bin",n*8),positions(dir+"/positions.bin",n*8),starts(dir+"/qstarts.bin",(requests+1)*4),map(dir+"/mapping.bin",requests*2*8);
  Gm packed(dir+"/initial_packed.bin",offset+2*blocks*stride),oldPacked(dir+"/initial_packed.bin",offset+2*blocks*stride);
  Gm status(dir+"/initial_status.bin",n*h*4),oldStatus(dir+"/initial_status.bin",n*h*4),windowStatus(dir+"/initial_window_status.bin",capacity*4);
  Gm wk(dir+"/initial_raw_key.bin",capacity*ks*2),wv(dir+"/initial_raw_value.bin",capacity*vs*2),tags(dir+"/initial_tags.bin",capacity*ts*8);
  const int64_t oldStride=(sink+recent)*h*d,oldTagStride=sink+recent;
  Gm oldK(2*blocks*oldStride*2),oldV(2*blocks*oldStride*2),oldTags(2*blocks*oldTagStride*8);
  const uint32_t packGrid=static_cast<uint32_t>(std::min<int64_t>(32,(n*h+7)/8));
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  std::cout<<"REQUEST_STORE_CPU_PHASE baseline"<<std::endl;
  ICPU_RUN_KF(oscar_rotate_clip_store_striped_kernel,packGrid,k.ptr,v.ptr,rk.ptr,rv.ptr,oldSlots.ptr,positions.ptr,oldPacked.ptr,oldK.ptr,oldV.ptr,oldTags.ptr,oldStatus.ptr,n,h,d,dtype,segment,2*blocks,offset,stride,oldStride,oldStride,oldTagStride,sink,recent,kc,vc,hadamard);
  oldPacked.Golden(dir,"packed");oldStatus.Golden(dir,"status");
  std::cout<<"REQUEST_STORE_CPU_PHASE candidate_packed"<<std::endl;
  ICPU_RUN_KF(oscar_request_packed_store_kernel,packGrid,k.ptr,v.ptr,rk.ptr,rv.ptr,slots.ptr,positions.ptr,packed.ptr,status.ptr,n,h,d,dtype,2*segment,blocks,offset,stride,kc,vc,hadamard);
  packed.Golden(dir,"packed");status.Golden(dir,"status");
  Equal(packed.ptr,oldPacked.ptr,packed.size,"baseline/candidate packed");Equal(status.ptr,oldStatus.ptr,status.size,"baseline/candidate status");
  std::cout<<"REQUEST_STORE_CPU_PHASE candidate_window"<<std::endl;
  ICPU_RUN_KF(oscar_request_window_store_kernel,static_cast<uint32_t>(std::min<int64_t>(32,requests)),k.ptr,v.ptr,slots.ptr,positions.ptr,starts.ptr,map.ptr,status.ptr,wk.ptr,wv.ptr,tags.ptr,windowStatus.ptr,n,requests,capacity,h,d,dtype,sink,recent,bits,ks,vs,ts);
  wk.Golden(dir,"raw_key");wv.Golden(dir,"raw_value");tags.Golden(dir,"tags");windowStatus.Golden(dir,"window_status");
  packed.Golden(dir,"packed");status.Golden(dir,"status");
  for(const Gm* input:{&k,&v,&rk,&rv,&slots,&oldSlots,&positions,&starts,&map}){input->Guard();Equal(input->ptr,input->initial.data(),input->size,"input changed");}
  for(const Gm* output:{&oldK,&oldV,&oldTags})output->Guard();
  std::cout<<"REQUEST_STORE_CPU_RESULT {\"status\":\"passed\",\"backend\":\"official_cpu_debug\",\"NPU\":\"not_run\",\"all_bytes\":\"identical\"}"<<std::endl;return 0;
}catch(const std::exception&e){std::cerr<<"REQUEST_STORE_CPU_FAILED "<<e.what()<<std::endl;return 1;}}
