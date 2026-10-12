// Official CPU-debug for one bounded striped->FP32 shared-history producer.
// Independent scalar canonical INT2 decode checks every valid FP32 K/V word.
// This is not full-history serving restoration, consumer parity, or NPU speed.
#include "../striped_prefill/probe_common.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
extern "C" void oscar_shared_fp32_history_producer_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    uint8_t*,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);
namespace {
float HalfAt(const uint8_t* ptr) {
  _Float16 value;std::memcpy(&value,ptr,2);return static_cast<float>(value);
}
float FloatAt(const uint8_t* ptr) {
  float value;std::memcpy(&value,ptr,4);return value;
}
}
int main(int argc,char** argv) {
  try {
    if(argc!=3 || (std::string(argv[1])!="valid" && std::string(argv[1])!="bad_scale"))
      throw std::runtime_error("usage: oscar_shared_fp32_producer_cpu valid|bad_scale GOLDEN_DIR");
    const bool bad=std::string(argv[1])=="bad_scale";
    const std::string dir=argv[2];
    std::ifstream shape(dir+"/shape.txt");
    int64_t tokens,hq,hk,d,context,sink,recent,spec,blockTokens,blocks,prefix,stride,cores;
    if(!(shape>>tokens>>hq>>hk>>d>>context>>sink>>recent>>spec>>blockTokens>>blocks>>prefix>>stride>>cores)
        || d!=256 || hk!=1 || context<=sink || context-sink>8192)
      throw std::runtime_error("invalid single-request D256 history golden");
    int64_t requests=1,splits=1,columns=8;shape>>requests;shape>>splits;shape>>columns;
    if(requests!=1 || columns<4)throw std::runtime_error("producer requires one request/table");
    Gm canonical(dir+"/raw.bin",prefix+blocks*stride),striped(canonical.size);
    Gm table(dir+"/table.bin",requests*columns*4);
    Gm k(8192*256*4),v(k.size),errors(8192*4);
    StripeRaw(canonical,striped,prefix,blocks,stride,blockTokens,hk);
    constexpr int64_t corrupted=300;
    if(bad) {
      const int32_t virtualBlock=reinterpret_cast<const int32_t*>(table.ptr)[corrupted/128];
      const int64_t ratio=blockTokens/128,physical=virtualBlock/ratio;
      const int64_t inpage=(virtualBlock%ratio)*128+corrupted%128;
      const int64_t offset=prefix+physical*stride+inpage*136;
      striped.ptr[offset+128]=0;striped.ptr[offset+129]=0;
    }
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(oscar_shared_fp32_history_producer_kernel,40,striped.ptr,table.ptr,
        k.ptr,v.ptr,errors.ptr,context,sink,context,columns,blockTokens,blocks,prefix,stride);
    for(int64_t position=sink;position<context;++position) {
      int32_t status;std::memcpy(&status,errors.ptr+(position-sink)*4,4);
      if(status!=((bad && position==corrupted)?3:0))
        throw std::runtime_error("per-row producer status differs at "+std::to_string(position));
      if(bad && position==corrupted)continue;
      const int32_t virtualBlock=reinterpret_cast<const int32_t*>(table.ptr)[position/128];
      const int64_t ratio=blockTokens/128,physical=virtualBlock/ratio;
      const int64_t inpage=(virtualBlock%ratio)*128+position%128;
      const uint8_t* slot=canonical.ptr+prefix+physical*stride+inpage*136;
      const float ks=HalfAt(slot+64),kz=HalfAt(slot+66);
      const float vs=HalfAt(slot+132),vz=HalfAt(slot+134);
      for(int64_t col=0;col<d;++col) {
        const int64_t byte=col/4,shift=(col%4)*2;
        const float expectedK=static_cast<float>((slot[byte]>>shift)&3)*ks+kz;
        const float expectedV=static_cast<float>((slot[68+byte]>>shift)&3)*vs+vz;
        const int64_t offset=((position-sink)*d+col)*4;
        const float actualK=FloatAt(k.ptr+offset),actualV=FloatAt(v.ptr+offset);
        if(std::memcmp(&actualK,&expectedK,4) || std::memcmp(&actualV,&expectedV,4))
          throw std::runtime_error("producer FP32 rotated KV differs at "+
              std::to_string(position)+":"+std::to_string(col));
      }
    }
    std::cout<<"{\"backend\":\"ascendc_cpu_debug\",\"op\":\"shared_fp32_producer\","
             <<"\"mode\":\""<<(bad?"bad_scale":"valid")<<"\",\"status\":\"passed\"}"<<std::endl;
    return 0;
  }catch(const std::exception& e) {
    std::cerr<<"SHARED_FP32_PRODUCER_FAILED: "<<e.what()<<std::endl;return 1;
  }
}
