// Experimental fixture helpers copied from the already CPU-validated striped
// decode probe. Archive #126/#129/#145/#154; not production code.
#pragma once
#include "tikicpulib.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
std::vector<uint8_t> Read(const std::string& path,size_t n) {
  std::vector<uint8_t> data(n);std::ifstream file(path,std::ios::binary);
  if(!file.read(reinterpret_cast<char*>(data.data()),n) || file.peek()!=EOF)
    throw std::runtime_error("incorrect golden byte count: "+path);
  return data;
}
void Write(const std::string& path,const uint8_t* data,size_t n) {
  std::ofstream file(path,std::ios::binary|std::ios::trunc);
  if(!file.write(reinterpret_cast<const char*>(data),n) || !file.flush())
    throw std::runtime_error("failed to write CAModel fixture: "+path);
}
struct Gm {
  uint8_t* ptr;size_t size;
  explicit Gm(size_t n):ptr(static_cast<uint8_t*>(AscendC::GmAlloc(n))),size(n) {
    if(!ptr)throw std::runtime_error("GmAlloc failed");std::memset(ptr,0x85,n);
  }
  Gm(const std::string& path,size_t n):Gm(n) {
    const auto data=Read(path,n);std::memcpy(ptr,data.data(),n);
  }
  ~Gm(){AscendC::GmFree(ptr);}
  Gm(const Gm&)=delete;Gm& operator=(const Gm&)=delete;
};
void CheckEqual(const Gm& baseline,const Gm& candidate,const char* label) {
  if(baseline.size!=candidate.size)throw std::runtime_error("size mismatch");
  for(size_t i=0;i<baseline.size;++i)
    if(baseline.ptr[i]!=candidate.ptr[i])
      throw std::runtime_error(std::string(label)+" byte mismatch at "+std::to_string(i));
}
void StripeSlot(const uint8_t* old,uint8_t* out) {
  // Old: [Kcode64,Ks2,Kz2,Vcode64,Vs2,Vz2]. New:
  // [Kcode64,Vcode64,Ks2,Kz2,Vs2,Vz2]. All FP16 metadata bytes unchanged.
  std::array<uint8_t,136> source{};std::memcpy(source.data(),old,source.size());
  std::memset(out,0,128);
  for(int side=0;side<2;++side) {
    const int oldOffset=side?68:0,newOffset=side?64:0;
    for(int dimension=0;dimension<256;++dimension) {
      const uint8_t code=(source[oldOffset+dimension/4]>>(2*(dimension%4)))&3;
      const int word=dimension%32,bit=dimension/32;
      const uint16_t prior=static_cast<uint16_t>(out[newOffset+2*word]) |
          static_cast<uint16_t>(out[newOffset+2*word+1])<<8;
      const uint16_t updated=prior | static_cast<uint16_t>(code)<<(2*bit);
      out[newOffset+2*word]=static_cast<uint8_t>(updated);
      out[newOffset+2*word+1]=static_cast<uint8_t>(updated>>8);
    }
  }
  std::memcpy(out+128,source.data()+64,4);
  std::memcpy(out+132,source.data()+132,4);
}
void StripeRaw(const Gm& old,Gm& striped,int64_t prefix,int64_t blocks,
    int64_t pageStride,int64_t blockTokens,int64_t kvHeads) {
  if(old.size!=striped.size || prefix<0 || blocks<0 || pageStride<blockTokens*kvHeads*136 ||
      prefix+blocks*pageStride>static_cast<int64_t>(old.size))
    throw std::runtime_error("invalid raw page geometry");
  std::memcpy(striped.ptr,old.ptr,old.size);
  for(int64_t block=0;block<blocks;++block)
    for(int64_t row=0;row<blockTokens*kvHeads;++row) {
      const int64_t offset=prefix+block*pageStride+row*136;
      StripeSlot(old.ptr+offset,striped.ptr+offset);
    }
}
}
