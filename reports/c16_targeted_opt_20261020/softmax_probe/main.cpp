// Archive #126/#145/#148/#164: real SDK CPU comparison, errors at every row,
// finite overflow/cancellation, read-only score bytes and GM/UB guards.
#include "tikicpulib.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void oscar_c16_finite_probe_kernel(uint8_t*,uint8_t*,uint8_t*,int32_t);
namespace {
constexpr int kRows=32,kColumns=256,kCount=kRows*kColumns,kGuard=8;
struct Gm {
  uint8_t* base;
  uint8_t* ptr;
  size_t bytes;
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
  std::vector<float> result(kCount);
  for(int r=0;r<kRows;++r)for(int c=0;c<kColumns;++c)
    result[r*kColumns+c]=((r+1)*17+(c%13)-6)*0.125F;
  return result;
}
void Check(const std::vector<float>& values,int rows,int expected,
           const std::string& label,int& cases) {
  Gm input(kCount*4),result(64),after((kCount+2*kGuard)*4);
  std::memcpy(input.ptr,values.data(),input.bytes);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(oscar_c16_finite_probe_kernel,1,input.ptr,result.ptr,after.ptr,rows);
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
      auto values=Base();
      // An invalid inactive row must not be checked, including rows==0.
      if(rows<32)values[rows*kColumns]=Bits(0x7fc00123);
      Check(values,rows,1,"active_rows_"+std::to_string(rows),cases);
    }
    // Every error position separately proves that counter-mask hoisting did
    // not cause later rows to stop executing after the first vcadd.
    for(int row=0;row<32;++row) {
      for(uint32_t bad:{0x7fc00123U,0x7f800000U,0xff800000U,0x7f800001U}) {
        auto values=Base();values[row*kColumns+(row*17)%kColumns]=Bits(bad);
        Check(values,32,0,"nonfinite_row_"+std::to_string(row),cases);
      }
      auto values=Base();
      for(int c=0;c<kColumns;++c)values[row*kColumns+c]=std::numeric_limits<float>::max();
      Check(values,32,0,"finite_overflow_row_"+std::to_string(row),cases);
      for(int c=0;c<kColumns;++c)
        values[row*kColumns+c]=(c%4<2?1.0F:-1.0F)*std::numeric_limits<float>::max();
      // Exact SDK cancellation/overflow result determines the contract;
      // do not replace it with an assumed real-arithmetic finite answer.
      Check(values,32,-1,"overflow_cancel_row_"+std::to_string(row),cases);
      for(int c=0;c<kColumns;++c)values[row*kColumns+c]=Bits(c%2?0x80000000:0);
      Check(values,32,1,"signed_zero_row_"+std::to_string(row),cases);
      for(int c=0;c<kColumns;++c)values[row*kColumns+c]=Bits(c%2?0x80000001:1);
      Check(values,32,1,"subnormal_row_"+std::to_string(row),cases);
    }
    std::cout<<"{\"status\":\"passed\",\"cases\":"<<cases
      <<",\"engine\":\"official_CANN_CPU_debug\",\"score_bytes\":\"unchanged\","
      <<"\"guards\":\"passed\",\"performance\":\"not_run\",\"NPU\":\"not_run\"}\n";
    return 0;
  } catch(const std::exception& error) {
    std::cerr<<"C16_FINITE_FAILED cases="<<cases<<" "<<error.what()<<"\n";
    return 1;
  }
}
