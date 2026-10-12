// SPDX-License-Identifier: Apache-2.0
// Archive #94/#125/#126/#145/#154; D.4 four questions:
// (1) Compare the last FP32 acc+PV update, not full attention.
// (2) No full-history dequant/host timing is substituted for device evidence.
// (3) Save every input/output/guard before checks; never hide any bit difference.
// (4) CPU-debug is a falsifier only; atomic NPU rounding/graph remain unproved.
#include "tikicpulib.h"
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void oscar_c4_atomic_state_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*);
namespace {
constexpr int M=126,MP=128,N=256,K=256;
struct Gm {
  uint8_t* base;uint8_t* ptr;size_t size;std::vector<uint8_t> initial;
  explicit Gm(size_t bytes):size(bytes) {
    base=static_cast<uint8_t*>(AscendC::GmAlloc(bytes+128));
    if(!base)throw std::runtime_error("GmAlloc failed");ptr=base+64;
    std::memset(base,0xa5,bytes+128);std::memset(ptr,0x85,bytes);
  }
  ~Gm(){AscendC::GmFree(base);}Gm(const Gm&)=delete;
  void Load(const std::filesystem::path& path) {
    initial.resize(size);std::ifstream in(path,std::ios::binary);
    if(!in.read(reinterpret_cast<char*>(initial.data()),size)||in.peek()!=EOF)
      throw std::runtime_error("wrong input byte count "+path.string());
    std::memcpy(ptr,initial.data(),size);
  }
  bool Unchanged()const{return initial.size()==size&&!std::memcmp(ptr,initial.data(),size);}
  bool Guards()const {
    for(size_t i=0;i<64;++i)if(base[i]!=0xa5||ptr[size+i]!=0xa5)return false;
    return true;
  }
  void Save(const std::filesystem::path& dir,const std::string& name)const {
    std::ofstream data(dir/(name+".bin"),std::ios::binary);
    data.write(reinterpret_cast<const char*>(ptr),size);
    std::ofstream all(dir/(name+".guarded.bin"),std::ios::binary);
    all.write(reinterpret_cast<const char*>(base),size+128);
    if(!data||!all)throw std::runtime_error("failed to save "+name);
  }
};
uint32_t Word(const Gm& gm,size_t index){uint32_t value;std::memcpy(&value,gm.ptr+index*4,4);return value;}
int Class(uint32_t bits) {
  if((bits&0x7fffffffU)>0x7f800000U)return 3; // NaN, payload is separately compared.
  if((bits&0x7fffffffU)==0x7f800000U)return bits>>31?2:1;
  return 0;
}
}
int main(int argc,char** argv) {
  try {
    if(argc!=3)throw std::runtime_error("usage: oscar_c4_atomic_state_cpu FIXTURE NEW_OUTPUT");
    const std::filesystem::path fixture=argv[1],output=argv[2];
    int m,mp,n,k;std::ifstream shape(fixture/"shape.txt");shape>>m>>mp>>n>>k;
    if(!shape||m!=M||mp!=MP||n!=N||k!=K)throw std::runtime_error("requires M126/Mpad128/N256/K256");
    if(std::filesystem::exists(output))throw std::runtime_error("refuse existing output");
    std::filesystem::create_directories(output);
    Gm p(MP*K*4),v(K*N*4),alpha(MP*N*4),pv(MP*N*4),reference(MP*N*4),atomic(MP*N*4),status(3*64);
    p.Load(fixture/"p.bin");v.Load(fixture/"v.bin");alpha.Load(fixture/"alpha.bin");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_RUN_KF(oscar_c4_atomic_state_kernel,1,p.ptr,v.ptr,alpha.ptr,pv.ptr,reference.ptr,atomic.ptr,status.ptr);
    // Persist actual bytes before comparison; a failed case remains inspectable.
    p.Save(output,"p_after");v.Save(output,"v_after");alpha.Save(output,"alpha_after");
    pv.Save(output,"pv");reference.Save(output,"reference");atomic.Save(output,"atomic");status.Save(output,"status");
    const bool guards=p.Guards()&&v.Guards()&&alpha.Guards()&&pv.Guards()&&reference.Guards()&&atomic.Guards()&&status.Guards();
    const bool unchanged=p.Unchanged()&&v.Unchanged()&&alpha.Unchanged();
    bool actors=true;
    for(int actor=0;actor<3;++actor)for(int field=0;field<16;++field) {
      const uint32_t expected=field==0?1:field==1?actor:field==2?M:field==3?N:field==4?K:0;
      actors=actors&&Word(status,actor*16+field)==expected;
    }
    size_t liveDiff=0,tailDiff=0,classDiff=0,nanPayloadDiff=0,nonfiniteReference=0;
    std::string first="[";size_t shown=0;
    bool anti=true;const bool isAnti=fixture.filename()=="anti_k128";
    for(size_t i=0;i<MP*N;++i) {
      const auto left=Word(reference,i),right=Word(atomic,i);
      if(i<M*N) {
        classDiff+=Class(left)!=Class(right);nonfiniteReference+=Class(left)!=0;
        nanPayloadDiff+=Class(left)==3&&Class(right)==3&&left!=right;
        if(isAnti)anti=anti&&left==0x3e800000U&&(Word(pv,i)&0x7fffffffU)==0;
      }
      if(left==right)continue;
      if(i<M*N)++liveDiff;else ++tailDiff;
      if(shown++<8) {
        if(shown>1)first+=",";
        first+="{\"index\":"+std::to_string(i)+",\"reference_bits\":"+std::to_string(left)+
          ",\"atomic_bits\":"+std::to_string(right)+"}";
      }
    }
    first+="]";
    const bool pass=guards&&unchanged&&actors&&anti&&!liveDiff&&!tailDiff;
    const std::string report="{\"status\":\""+std::string(pass?"bitwise_passed":"failed")+
      "\",\"backend\":\"official_cpu_debug\",\"guards\":"+(guards?"true":"false")+
      ",\"inputs_unchanged\":"+(unchanged?"true":"false")+",\"actor_completion\":"+(actors?"true":"false")+
      ",\"live_bit_differences\":"+std::to_string(liveDiff)+",\"padding_bit_differences\":"+std::to_string(tailDiff)+
      ",\"live_class_differences\":"+std::to_string(classDiff)+",\"live_nan_payload_differences\":"+std::to_string(nanPayloadDiff)+
      ",\"nonfinite_reference_elements\":"+std::to_string(nonfiniteReference)+
      ",\"anti_k128_checked\":"+(isAnti?"true":"false")+",\"anti_k128_expected\":"+(anti?"true":"false")+
      ",\"first_differences\":"+first+
      ",\"npu\":\"not_run\",\"graph\":\"not_run\",\"performance\":\"not_measured\",\"full_c4_error_contract\":\"not_run\"}";
    std::ofstream out(output/"report.json");out<<report<<'\n';
    if(!out)throw std::runtime_error("cannot save report");
    std::cout<<"ATOMIC_STATE_CPU_RESULT "<<report<<'\n';return pass?0:2;
  } catch(const std::exception& error) {
    std::cerr<<"ATOMIC_STATE_CPU_ERROR "<<error.what()<<'\n';return 2;
  }
}
