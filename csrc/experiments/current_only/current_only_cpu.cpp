// SPDX-License-Identifier: Apache-2.0
// Archive G30-G34/#4-16/#91/#126/#145 and startup D.4: official CPU-debug
// compares this independent current-only kernel with real merge_lse(splits=1)
// bracketed by native CANN BF16/FP32 Cast. CPU parity is not NPU acceptance.
#include "tikicpulib.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

extern "C" void oscar_current_only_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    uint8_t*,int64_t,int64_t);
extern "C" void oscar_current_baseline_cast_kernel(uint8_t*,uint8_t*,int64_t,
    int64_t,bool);
extern "C" void oscar_merge_lse_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    uint8_t*,int64_t,int64_t,int64_t);

namespace {
struct Gm {
  uint8_t* ptr;
  size_t size;
  explicit Gm(size_t bytes):ptr(static_cast<uint8_t*>(AscendC::GmAlloc(bytes))),size(bytes) {
    if (!ptr) throw std::runtime_error("GmAlloc failed");
    std::memset(ptr,0x85,size);
  }
  ~Gm(){AscendC::GmFree(ptr);}
  Gm(const Gm&)=delete;
};

void Put16(Gm& gm,size_t index,uint16_t bits) {
  if ((index+1)*2>gm.size) throw std::runtime_error("BF16 input OOB");
  std::memcpy(gm.ptr+index*2,&bits,2);
}
void PutFloatBits(Gm& gm,size_t index,uint32_t bits) {
  if ((index+1)*4>gm.size) throw std::runtime_error("FP32 LSE input OOB");
  std::memcpy(gm.ptr+index*4,&bits,4);
}

uint16_t InputBits(int32_t row,int32_t column) {
  switch(row) {
    case 0: return column%2 ? 0xbf80 : 0x3f80; // alternating +/-1
    case 1: return 0x8000;                       // -0 normalization
    case 2: return 0x7fc1;                       // ignored empty-row NaN
    case 3: return 0x7fc1;                       // ignored bad-LSE NaN
    case 4: return 0x7f7f;                       // finite BF16 sum overflow
    case 5: return column%2 ? 0xff7f : 0x7f7f; // cancelling huge values
    case 6: return column==0 ? 0x7fc1 : 0x3f80; // live NaN payload
    case 7: return column%2 ? 0xff80 : 0x7f80; // live mixed infinities
    case 8: return 0x8000;                       // -0 output and -0 LSE
    case 9: return column%2 ? 0xbf80 : 0x3f80;
    case 10:return column==0 ? 0x8000 : 0x0000;
    case 11:return 0x3f80;
    case 12:return 0x3f80;
    case 13:return column%3 ? 0x3d00 : 0xbd00;
    case 14:return 0x3f80;                       // ignored -inf LSE
    case 15:return column%2 ? 0x8001 : 0x0001; // BF16 subnormals
    default:return column%2 ? 0x3e80 : 0xbe80;  // batch tail, row 16
  }
}

uint32_t LseBits(int32_t row) {
  switch(row) {
    case 1: case 14:return 0xff800000U; // -inf empty
    case 2:return 0x7fc12345U;         // NaN payload
    case 3:return 0x7f800000U;         // +inf invalid
    case 6:return 0x3fc00000U;         // +1.5
    case 8:return 0x80000000U;         // -0
    case 9:return 0x40000000U;         // +2
    case 11:return 0xff7fffffU;        // -FLT_MAX
    case 12:return 0x7f7fffffU;        // +FLT_MAX
    case 13:return 0xc2f70000U;        // -123.5
    default:return 0;
  }
}

void Equal(const Gm& expected,const Gm& actual,const std::string& name,
           size_t word) {
  if (expected.size!=actual.size) throw std::runtime_error(name+" shape changed");
  for (size_t offset=0;offset<expected.size;offset+=word) {
    if (std::memcmp(expected.ptr+offset,actual.ptr+offset,word)) {
      throw std::runtime_error(name+" differs at row-word "+
          std::to_string(offset/word));
    }
  }
}

void Case(int32_t d) {
  constexpr int64_t rows=17;
  const size_t inBytes=rows*d*2,outFloatBytes=rows*d*4,scalarBytes=rows*4;
  Gm input(inBytes),inputLse(scalarBytes),partial(outFloatBytes);
  Gm oldFloat(outFloatBytes),oldOutput(inBytes),oldLse(scalarBytes),oldStatus(scalarBytes);
  Gm newOutput(inBytes),newLse(scalarBytes),newStatus(scalarBytes);
  for (int32_t row=0;row<rows;++row) {
    PutFloatBits(inputLse,row,LseBits(row));
    for (int32_t j=0;j<d;++j)
      Put16(input,row*d+j,InputBits(row,j));
  }
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(oscar_current_baseline_cast_kernel,4,
      input.ptr,partial.ptr,rows,int64_t{d},true);
  ICPU_RUN_KF(oscar_merge_lse_kernel,4,partial.ptr,inputLse.ptr,oldFloat.ptr,
      oldLse.ptr,oldStatus.ptr,rows,int64_t{1},int64_t{d});
  ICPU_RUN_KF(oscar_current_baseline_cast_kernel,4,
      oldFloat.ptr,oldOutput.ptr,rows,int64_t{d},false);
  ICPU_RUN_KF(oscar_current_only_kernel,4,input.ptr,inputLse.ptr,newOutput.ptr,
      newLse.ptr,newStatus.ptr,rows,int64_t{d});
  Equal(oldOutput,newOutput,"BF16 output D"+std::to_string(d),2);
  Equal(oldLse,newLse,"FP32 LSE D"+std::to_string(d),4);
  Equal(oldStatus,newStatus,"int32 status D"+std::to_string(d),4);
  std::cout << "CURRENT_ONLY_CPU D=" << d << " rows=" << rows
            << " output_lse_status=bitwise_equal\n";
}
}  // namespace

int main() {
  try {
    for (int32_t d : {64,128,256}) Case(d);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "CURRENT_ONLY_CPU_ERROR " << error.what() << '\n';
    return 1;
  }
}
