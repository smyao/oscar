// Archive #13-22/#126/#145/#155; D.4: exhaustive integer byte/neighbor oracle.
// Official CPU-debug is not NPU correctness or timing evidence.
#include "tikicpulib.h"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
extern "C" void oscar_simd_pack_words_kernel(uint8_t*,uint8_t*,int32_t,int64_t);
uint16_t Pattern(int64_t row,int position,int side,int width,bool compact) {
  if(!compact)return static_cast<uint16_t>(row+position*(side?513:257)+(side?12345:0));
  const int64_t domainRows=65536/width;
  if(row<domainRows)
    return static_cast<uint16_t>(row*width+(position+row)%width+(side?12345:0));
  // Each physical word position sees zero, ffff and all sixteen single bits.
  // Adjacent positions differ; primary-domain K/V use different permutations.
  const int boundary=static_cast<int>((row-domainRows+position)%18);
  return boundary==0?0:boundary==1?65535:static_cast<uint16_t>(1U<<(boundary-2));
}
int main(int argc,char** argv) {
  try {
    if(argc!=3&&argc!=4)throw std::runtime_error("usage: pack_cpu DIM NEW_OUTPUT_FILE [full-per-position|compact-domain]");
    const std::string mode=argc==4?argv[3]:"full-per-position";
    if(mode!="full-per-position"&&mode!="compact-domain")throw std::runtime_error("unknown enumeration mode");
    const bool compact=mode=="compact-domain";
    const int32_t d=std::stoi(argv[1]);if(d!=64&&d!=128&&d!=256)throw std::runtime_error("unsupported dim");
    std::ifstream old(argv[2]);if(old.good())throw std::runtime_error("refuse existing output");
    const int32_t width=d/8;const int64_t rows=compact?65536/width+18:65536;
    const size_t bytes=rows*2*d*4,outBytes=rows*160;
    auto* inBase=static_cast<uint8_t*>(AscendC::GmAlloc(bytes+128));
    auto* outBase=static_cast<uint8_t*>(AscendC::GmAlloc(outBytes+128));
    if(!inBase||!outBase)throw std::runtime_error("GmAlloc failed");
    auto* input=inBase+64;auto* output=outBase+64;
    std::memset(inBase,0xa5,bytes+128);std::memset(outBase,0xa5,outBytes+128);
    for(int64_t r=0;r<rows;++r)for(int side=0;side<2;++side)for(int i=0;i<width;++i) {
      const uint16_t word=Pattern(r,i,side,width,compact);
      for(int b=0;b<8;++b) {
        const int32_t code=(word>>(2*b))&3;
        std::memcpy(input+((r*2+side)*d+b*width+i)*4,&code,4);
      }
    }
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(oscar_simd_pack_words_kernel,32,input,output,d,rows);
    std::ofstream saved(argv[2],std::ios::binary);saved.write(reinterpret_cast<char*>(outBase),outBytes+128);
    if(!saved)throw std::runtime_error("cannot save actual output+guards");
    for(int i=0;i<64;++i)if(inBase[i]!=0xa5||input[bytes+i]!=0xa5||outBase[i]!=0xa5||output[outBytes+i]!=0xa5)
      throw std::runtime_error("guard changed");
    for(int64_t r=0;r<rows;++r)for(int side=0;side<2;++side)for(int i=0;i<width;++i) {
      const uint16_t want=Pattern(r,i,side,width,compact);
      uint16_t got;std::memcpy(&got,output+r*160+side*(d/4)+i*2,2);
      if(got!=want)throw std::runtime_error("word/position mismatch row="+std::to_string(r));
      for(int b=0;b<8;++b) {int32_t code;std::memcpy(&code,input+((r*2+side)*d+b*width+i)*4,4);
        if(code!=((want>>(2*b))&3))throw std::runtime_error("input changed");}
    }
    for(int64_t r=0;r<rows;++r)for(int i=d/2;i<160;++i)
      if(output[r*160+i]!=0xad)throw std::runtime_error("metadata/neighbor poison overwritten");
    AscendC::GmFree(inBase);AscendC::GmFree(outBase);
    std::cout<<"PACK_EXHAUSTIVE {\"status\":\"passed\",\"dim\":"<<d
      <<",\"mode\":\""<<mode<<"\",\"total_words_per_side\":"<<rows*width;
    if(compact)std::cout<<",\"domain_words_per_side\":65536,\"boundary_words_per_side\":"<<18*width;
    else std::cout<<",\"words_per_position\":65536";
    std::cout<<",\"sides\":2,\"NPU\":\"not_run\"}"<<std::endl;return 0;
  }catch(const std::exception& error){std::cerr<<"PACK_EXHAUSTIVE_FAILED "<<error.what()<<std::endl;return 2;}
}
