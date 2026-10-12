// SPDX-License-Identifier: Apache-2.0
// Archive #126/#145/#148/#154/#155 and startup D.4: official CPU-debug
// executes the real isolated kernel against independent store/original-space
// oracle fixtures. Poison output/workspace and check all rows and guard bytes.
// CPU checks cannot prove asynchronous NPU ownership, model quality or speed.
#include "tikicpulib.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define WIDE_ARGS uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,float
extern "C" void oscar_wide_fp32_b512_kernel(WIDE_ARGS);
extern "C" void oscar_wide_fp32_b1024_kernel(WIDE_ARGS);
#undef WIDE_ARGS

namespace {
constexpr int kRows=126,kDim=256;
constexpr size_t kGuard=64;
std::vector<uint8_t> Read(const std::string& path,size_t bytes) {
  std::ifstream file(path,std::ios::binary|std::ios::ate);
  if(!file || file.tellg()!=static_cast<std::streamoff>(bytes))
    throw std::runtime_error("wrong file size: "+path);
  file.seekg(0);
  std::vector<uint8_t> result(bytes);
  if(!file.read(reinterpret_cast<char*>(result.data()),bytes))
    throw std::runtime_error("read failed: "+path);
  return result;
}
struct Gm {
  uint8_t* ptr;
  size_t size;
  explicit Gm(size_t bytes):ptr(static_cast<uint8_t*>(AscendC::GmAlloc(bytes+kGuard))),size(bytes) {
    if(!ptr)throw std::runtime_error("GmAlloc failed");
    const uint32_t nan=0x7fc12345;
    for(size_t offset=0;offset<size;offset+=4)std::memcpy(ptr+offset,&nan,4);
    std::memset(ptr+size,0xa5,kGuard);
  }
  Gm(const std::string& path,size_t bytes):Gm(bytes) {
    const auto data=Read(path,bytes);std::memcpy(ptr,data.data(),bytes);
  }
  ~Gm(){AscendC::GmFree(ptr);}
  Gm(const Gm&)=delete;
  void CheckGuard()const {
    for(size_t i=0;i<kGuard;++i)if(ptr[size+i]!=0xa5)
      throw std::runtime_error("GM suffix guard changed");
  }
};
float Float(const uint8_t* ptr,size_t index) {
  float value;std::memcpy(&value,ptr+index*4,4);return value;
}
bool Close(float got,float expected,float atol,float rtol) {
  if(std::isinf(expected))return got==expected;
  return std::isfinite(got) && std::abs(got-expected)<=atol+rtol*std::abs(expected);
}
}  // namespace

int main(int argc,char** argv) {
  try {
    if(argc!=3)throw std::runtime_error("usage: wide_cpu CASE_DIR valid|faults|grid2");
    const std::string dir=argv[1],mode=argv[2];
    if(mode!="valid" && mode!="faults" && mode!="grid2")throw std::runtime_error("unsupported mode");
    int rows,kv,dim;float scale,atol,rtol;
    std::ifstream shape(dir+"/shape.txt"),tolerance(dir+"/tolerance.txt");
    if(!(shape>>rows>>kv>>dim>>scale) || rows!=kRows || dim!=kDim ||
       (kv!=512 && kv!=1024) || scale!=0.0625F)
      throw std::runtime_error("requires M126,D256,B512/B1024,scale=1/16");
    if(!(tolerance>>atol>>rtol) || atol!=0.005F || rtol!=0.005F)
      throw std::runtime_error("frozen fused_attention tolerance changed");
    Gm q(dir+"/qrot.bin",rows*dim*4),k(dir+"/krot.bin",kv*dim*4),
       v(dir+"/vrot.bin",kv*dim*4),rotation(dir+"/rv.bin",dim*dim*4),
       visible(dir+"/visible.bin",rows*4),work((rows*kv+rows*dim)*4),
       output(rows*dim*4),lse(rows*4),status(rows*4);
    std::vector<int32_t> expectedStatus(rows,0);
    if(mode=="faults") {
      const int32_t low=-1,high=kv+1;
      std::memcpy(visible.ptr,&low,4);std::memcpy(visible.ptr+64*4,&high,4);
      const uint32_t nan=0x7fc12345;std::memcpy(q.ptr+2*dim*4,&nan,4);
      expectedStatus[0]=1;expectedStatus[64]=1;expectedStatus[2]=2;
    } else if(mode=="grid2")std::fill(expectedStatus.begin(),expectedStatus.end(),1);
    std::cout<<"WIDE_FP32_CPU_START b="<<kv<<" mode="<<mode<<std::endl;
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    const uint32_t grid=mode=="grid2"?2:1;
#define ARGS q.ptr,k.ptr,v.ptr,rotation.ptr,visible.ptr,work.ptr,output.ptr,lse.ptr,status.ptr,scale
    if(kv==512) { ICPU_RUN_KF(oscar_wide_fp32_b512_kernel,grid,ARGS); }
    else { ICPU_RUN_KF(oscar_wide_fp32_b1024_kernel,grid,ARGS); }
#undef ARGS
    for(const Gm* gm:{&q,&k,&v,&rotation,&visible,&work,&output,&lse,&status})gm->CheckGuard();
    const auto expected=Read(dir+"/expected_output.bin",rows*dim*4);
    const auto expectedLse=Read(dir+"/expected_lse.bin",rows*4);
    float maxOutput=0,maxLse=0,maxScaledError=0;
    for(int row=0;row<rows;++row) {
      int32_t code;std::memcpy(&code,status.ptr+row*4,4);
      if(code!=expectedStatus[row])throw std::runtime_error("status mismatch row="+std::to_string(row));
      if(mode=="grid2")continue;
      const float gotLse=Float(lse.ptr,row),wantLse=Float(expectedLse.data(),row);
      if(code!=0) {
        if(!std::isnan(gotLse))throw std::runtime_error("error LSE must be NaN");
      } else {
        if(!Close(gotLse,wantLse,atol,rtol))throw std::runtime_error("LSE oracle mismatch row="+std::to_string(row));
        if(std::isfinite(wantLse))maxLse=std::max(maxLse,std::abs(gotLse-wantLse));
      }
      for(int col=0;col<dim;++col) {
        const int index=row*dim+col;
        const float got=Float(output.ptr,index),want=Float(expected.data(),index);
        if(code!=0) {
          if(!std::isnan(got))throw std::runtime_error("error output must be NaN");
        } else {
          maxOutput=std::max(maxOutput,std::abs(got-want));
          maxScaledError=std::max(maxScaledError,std::abs(got-want)/(atol+rtol*std::abs(want)));
          if(!Close(got,want,atol,rtol)) {
            std::cerr<<"mismatch row="<<row<<" col="<<col<<" got="<<got<<" expected="<<want<<'\n';
            throw std::runtime_error("frozen output oracle failed");
          }
        }
      }
    }
    std::cout<<std::setprecision(10)<<"WIDE_FP32_CPU_RESULT {\"b\":"<<kv
      <<",\"mode\":\""<<mode<<"\",\"status\":\"passed\",\"rows\":"<<rows
      <<",\"output_max_abs\":"<<maxOutput<<",\"lse_max_abs\":"<<maxLse
      <<",\"max_tolerance_fraction\":"<<maxScaledError
      <<",\"guards\":\"unchanged\",\"npu\":\"not_run\",\"performance\":\"not_run\"}"<<std::endl;
    return 0;
  } catch(const std::exception& error) {
    std::cerr<<"WIDE_FP32_CPU_ERROR "<<error.what()<<std::endl;return 1;
  }
}
