// Archive #126/#129/#145/#148-154: official AscendC CPU-debug validates every
// uint16 word pattern, natural D order, FP32 reconstruction, and live/tail
// metadata error domains. CPU-debug does not establish NPU speed or graph.
#include "tikicpulib.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
extern "C" void oscar_striped_unpack_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,int32_t,int32_t);

namespace {
constexpr int D=256,HalfKv=64,Stride=160,Words=D/8,MaxBatches=32;
void Put16(uint8_t* ptr,uint16_t value) {std::memcpy(ptr,&value,2);}
float Get32(const uint8_t* ptr) {float value;std::memcpy(&value,ptr,4);return value;}
void Fill(uint8_t* raw,int batches) {
  for(int batch=0;batch<batches;++batch)for(int row=0;row<HalfKv;++row) {
    uint8_t* slot=raw+(batch*HalfKv+row)*Stride;
    for(int word=0;word<Words;++word) {
      const uint16_t bits=static_cast<uint16_t>(batch*HalfKv*Words+row*Words+word);
      Put16(slot+word*2,bits);
      Put16(slot+D/4+word*2,static_cast<uint16_t>(~bits));
    }
    Put16(slot+D/2,0x3c00);Put16(slot+D/2+2,0x0000); // K: 1,0
    Put16(slot+D/2+4,0x3800);Put16(slot+D/2+6,0xb400); // V: 0.5,-0.25
  }
}
}
int main(int argc,char** argv) {
  try {
    const std::string mode=argc==2?argv[1]:"";
    if(mode!="exhaustive" && mode!="bad_live" && mode!="bad_dead")
      throw std::runtime_error("usage: oscar_striped_unpack_cpu exhaustive|bad_live|bad_dead");
    const int batches=mode=="exhaustive"?MaxBatches:1;
    const int liveRows=mode=="exhaustive"?HalfKv:18;
    auto* raw=static_cast<uint8_t*>(AscendC::GmAlloc(batches*HalfKv*Stride));
    auto* k=static_cast<uint8_t*>(AscendC::GmAlloc(batches*HalfKv*D*4));
    auto* v=static_cast<uint8_t*>(AscendC::GmAlloc(batches*HalfKv*D*4));
    auto* status=static_cast<uint8_t*>(AscendC::GmAlloc(batches*4));
    if(!raw || !k || !v || !status)throw std::runtime_error("GmAlloc failed");
    std::memset(raw,0,batches*HalfKv*Stride);
    std::memset(k,0x85,batches*HalfKv*D*4);
    std::memset(v,0x85,batches*HalfKv*D*4);
    std::memset(status,0x85,batches*4);
    Fill(raw,batches);
    if(mode!="exhaustive") {
      const int row=mode=="bad_live"?5:29;
      Put16(raw+row*Stride+D/2,0); // K scale invalid; only live rows count.
    }
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(oscar_striped_unpack_kernel,1,raw,k,v,status,batches,liveRows);
    for(int batch=0;batch<batches;++batch) {
      int32_t code;std::memcpy(&code,status+batch*4,4);
      if(code!=(mode=="bad_live"?3:0))
        throw std::runtime_error("striped live/tail metadata status mismatch");
      if(mode=="bad_live")continue;
      for(int row=0;row<liveRows;++row)for(int d=0;d<D;++d) {
        const int word=d%Words,bit=d/Words;
        const uint16_t bits=static_cast<uint16_t>(batch*HalfKv*Words+row*Words+word);
        const float expectedK=static_cast<float>((bits>>(2*bit))&3);
        const float expectedV=0.5F*static_cast<float>((static_cast<uint16_t>(~bits)>>(2*bit))&3)-0.25F;
        const size_t offset=((batch*HalfKv+row)*D+d)*4;
        if(Get32(k+offset)!=expectedK || Get32(v+offset)!=expectedV)
          throw std::runtime_error("striped natural-dimension FP32 value mismatch");
      }
    }
    AscendC::GmFree(raw);AscendC::GmFree(k);AscendC::GmFree(v);AscendC::GmFree(status);
    std::cout<<"striped_mode="<<mode<<" words="<<batches*HalfKv*Words
             <<" status=passed"<<std::endl;
    std::cout<<"{\"backend\":\"ascendc_cpu_debug\",\"op\":\"striped_unpack\",\"status\":\"passed\"}"<<std::endl;
    return 0;
  }catch(const std::exception& error) {
    std::cerr<<"STRIPED_CPU_FAILED: "<<error.what()<<std::endl;return 1;
  }
}
