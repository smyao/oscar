// Exhaust all 65536 FP16 metadata patterns separately as live scale and zero.
// Official CPU-debug tests packed mask lane order and live/dead row handling;
// it does not certify target NPU or whole-model performance. D.4/#126/#154.
#include "tikicpulib.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
extern "C" void oscar_striped_metadata_guard_kernel(uint8_t*,uint8_t*,int32_t);
int main() {
  constexpr int32_t batches=2049,rows=64;
  auto* source=static_cast<uint8_t*>(AscendC::GmAlloc(batches*128*2));
  auto* output=static_cast<uint8_t*>(AscendC::GmAlloc(batches*4*8));
  if(!source || !output)throw std::runtime_error("metadata GmAlloc failed");
  std::memset(source,0,batches*128*2);std::memset(output,0x85,batches*4*8);
  for(int32_t batch=0;batch<batches;++batch)
    for(int32_t row=0;row<rows;++row) {
      uint16_t scale=0x3c00,zero=0;
      if(batch<1024)scale=static_cast<uint16_t>(batch*64+row);
      else if(batch<2048)zero=static_cast<uint16_t>((batch-1024)*64+row);
      else if(row==32)scale=0x7c00;
      std::memcpy(source+(batch*128+row)*2,&scale,2);
      std::memcpy(source+(batch*128+64+row)*2,&zero,2);
    }
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(oscar_striped_metadata_guard_kernel,1,source,output,batches);
  for(int32_t batch=0;batch<batches;++batch) {
    uint64_t observed;std::memcpy(&observed,output+batch*32,8);
    uint64_t expected=0;
    for(int32_t row=0;row<rows;++row) {
      if(batch==2048 && row>=17)continue;
      uint16_t scaleBits,zeroBits;
      std::memcpy(&scaleBits,source+(batch*128+row)*2,2);
      std::memcpy(&zeroBits,source+(batch*128+64+row)*2,2);
      _Float16 scaleHalf,zeroHalf;
      std::memcpy(&scaleHalf,&scaleBits,2);std::memcpy(&zeroHalf,&zeroBits,2);
      const float scale=static_cast<float>(scaleHalf),zero=static_cast<float>(zeroHalf);
      if(!(std::isfinite(scale) && scale>0.0F && std::isfinite(zero)))
        expected|=1ULL<<row;
    }
    if(observed!=expected)
      throw std::runtime_error("metadata SIMD packed mask mismatch at batch "+
          std::to_string(batch));
  }
  AscendC::GmFree(source);AscendC::GmFree(output);
  std::cout<<"{\"backend\":\"ascendc_cpu_debug\",\"op\":\"striped_metadata_guard\","
           <<"\"fp16_patterns_each_role\":65536,\"dead_tail\":true,"
           <<"\"status\":\"passed\"}"<<std::endl;
  return 0;
}
