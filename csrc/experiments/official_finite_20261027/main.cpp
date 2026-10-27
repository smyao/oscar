// Archive #126/#145/#148/#164: independent SDK equivalence and guarded buffers.
// D.4: test only bounded finite-check work, with no timing or device claims.
#include "tikicpulib.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void oscar_official_finite_probe_kernel(uint8_t*,uint8_t*,uint8_t*,int32_t);
namespace {
constexpr int kRows=32,kColumns=256,kCount=kRows*kColumns,kGuard=8;
struct Gm {
  uint8_t* base; uint8_t* ptr; size_t bytes;
  explicit Gm(size_t n):base(static_cast<uint8_t*>(AscendC::GmAlloc(n+128))),
      ptr(base?base+64:nullptr),bytes(n) {
    if(!base)throw std::runtime_error("GmAlloc failed");
    std::memset(base,0xD7,n+128);
  }
  ~Gm(){AscendC::GmFree(base);}
  void Check()const {
    for(int i=0;i<64;++i)
      if(base[i]!=0xD7 || ptr[bytes+i]!=0xD7)throw std::runtime_error("GM guard changed");
  }
};
float Bits(uint32_t bits){float x;std::memcpy(&x,&bits,4);return x;}
int64_t At(const Gm& gm,int i){int64_t x;std::memcpy(&x,gm.ptr+8*i,8);return x;}
std::vector<float> Base() {
  std::vector<float> values(kCount);
  for(int i=0;i<kCount;++i)values[i]=((i*31)%511-255)*0.03125F;
  return values;
}
void Check(const std::vector<float>& values,int rows,int expected,
           const std::string& label,int& cases) {
  Gm input(kCount*4),result(64),after((kCount+2*kGuard)*4);
  std::memcpy(input.ptr,values.data(),input.bytes);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(oscar_official_finite_probe_kernel,1,input.ptr,result.ptr,after.ptr,rows);
  input.Check();result.Check();after.Check();
  if(std::memcmp(input.ptr,values.data(),input.bytes) ||
     std::memcmp(after.ptr+kGuard*4,values.data(),input.bytes))
    throw std::runtime_error(label+": input/score bytes changed");
  const int64_t baseline=At(result,0),candidate=At(result,1);
  if(baseline!=candidate || (expected>=0 && baseline!=expected) ||
      At(result,2)!=0 || At(result,3)!=rows)
    throw std::runtime_error(label+": baseline="+std::to_string(baseline)+
        " candidate="+std::to_string(candidate)+" guards="+std::to_string(At(result,2)));
  ++cases;
}
}
int main() {
  int cases=0;
  try {
    for(int rows=0;rows<=32;++rows) {
      auto v=Base();
      if(rows<32)v[rows*kColumns]=Bits(0x7fc00123);
      Check(v,rows,1,"active_rows_"+std::to_string(rows),cases);
    }
    // Every row and each 64-element repeat, including its last element.
    for(int row=0;row<32;++row)for(int repeat=0;repeat<4;++repeat) {
      for(int edge:{0,63})for(uint32_t bad:{0x7fc00123U,0x7f800000U,0xff800000U,0x7f800001U}) {
        auto v=Base();const int offset=row*kColumns+repeat*64+edge;v[offset]=Bits(bad);
        Check(v,32,0,"nonfinite_offset_"+std::to_string(offset),cases);
      }
    }
    const float boundary=std::ldexp(1.0F,110);
    for(int rows:{7,8,9,16,31,32}) {
      for(float scale:{std::nextafter(boundary,0.0F),boundary,
                       std::nextafter(boundary,std::numeric_limits<float>::infinity())}) {
        for(int offset:{0,255,(rows*256)/2,rows*256-1}) {
          std::vector<float> v(kCount,0.0F);v[offset]=scale;
          Check(v,rows,1,"threshold_single_"+std::to_string(rows),cases);
          v[offset]=-scale;Check(v,rows,1,"threshold_negative_"+std::to_string(rows),cases);
        }
        std::vector<float> v(kCount,scale/(rows*256));
        Check(v,rows,1,"threshold_distributed_"+std::to_string(rows),cases);
      }
    }
    for(int row=0;row<32;++row) {
      auto v=Base();
      for(int c=0;c<kColumns;++c)v[row*kColumns+c]=std::numeric_limits<float>::max();
      Check(v,32,0,"finite_overflow_"+std::to_string(row),cases);
      for(int period:{2,4,8,16,128,256}) {
        for(int c=0;c<kColumns;++c)v[row*kColumns+c]=(c%period<period/2?1.0F:-1.0F)*std::numeric_limits<float>::max();
        Check(v,32,-1,"cancel_"+std::to_string(row)+"_"+std::to_string(period),cases);
      }
      for(int c=0;c<kColumns;++c)v[row*kColumns+c]=Bits(c%2?0x80000000:0);
      Check(v,32,1,"signed_zero_"+std::to_string(row),cases);
      for(int c=0;c<kColumns;++c)v[row*kColumns+c]=Bits(c%2?0x80000001:1);
      Check(v,32,1,"subnormal_"+std::to_string(row),cases);
    }
    std::cout<<"{\"status\":\"passed\",\"cases\":"<<cases
      <<",\"engine\":\"official_CANN_CPU_debug\",\"score_bytes\":\"unchanged\","
      <<"\"guards\":\"passed\",\"performance\":\"not_run\",\"NPU\":\"not_run\"}\n";
    return 0;
  }catch(const std::exception& e) {
    std::cerr<<"OFFICIAL_FINITE_FAILED cases="<<cases<<" "<<e.what()<<"\n";return 1;
  }
}
