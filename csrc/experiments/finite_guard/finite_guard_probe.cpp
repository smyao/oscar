// SPDX-License-Identifier: Apache-2.0
// Archive #126/#129/#140-153; startup D.4 four questions:
// (1) Run actual bounded source0 fused INT2 CV old/new kernels, not a mock.
// (2) D.4's failed full-history restore cost 6.5s/device; no such buffer here.
// (3) Both kernels get the same task/raw/Q bytes; valid output/LSE/status must
//     be bitwise, invalid final status bitwise with at least one real error.
// (4) CPU-debug checks correctness only, never target NPU speed or graph.
#include "tikicpulib.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "../../include/oscar_attention_launch.h"

extern "C" void oscar_prepare_attention_tasks_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,bool,uint8_t*,int64_t,bool);
extern "C" void oscar_attention_cv_fast_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,float);
extern "C" void oscar_attention_cv_checked_fast_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,float);
extern "C" void oscar_merge_lse_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    int64_t,int64_t,int64_t);

namespace {
std::vector<uint8_t> Read(const std::string& path,size_t size) {
  std::vector<uint8_t> bytes(size);std::ifstream file(path,std::ios::binary);
  if(!file.read(reinterpret_cast<char*>(bytes.data()),size) || file.peek()!=EOF)
    throw std::runtime_error("incorrect golden byte count: "+path);
  return bytes;
}
struct Gm {
  uint8_t* ptr;size_t size;
  explicit Gm(size_t bytes):ptr(static_cast<uint8_t*>(AscendC::GmAlloc(bytes))),size(bytes) {
    if(!ptr)throw std::runtime_error("GmAlloc failed");std::memset(ptr,0x85,bytes);
  }
  Gm(const std::string& path,size_t bytes):Gm(bytes) {
    auto source=Read(path,bytes);std::memcpy(ptr,source.data(),bytes);
  }
  ~Gm(){AscendC::GmFree(ptr);}Gm(const Gm&)=delete;
};
void EqualBytes(const Gm& a,const Gm& b,const char* label) {
  if(a.size!=b.size || std::memcmp(a.ptr,b.ptr,a.size))
    throw std::runtime_error(std::string(label)+" differs bitwise");
}
void SameClass(const Gm& a,const Gm& b,const char* label) {
  if(a.size!=b.size)throw std::runtime_error(std::string(label)+" size changed");
  for(size_t off=0;off<a.size;off+=4) {
    float left,right;std::memcpy(&left,a.ptr+off,4);std::memcpy(&right,b.ptr+off,4);
    if(std::isnan(left)!=std::isnan(right) || std::isfinite(left)!=std::isfinite(right) ||
       (std::isfinite(left) && left!=right))
      throw std::runtime_error(std::string(label)+" invalid domain changed");
  }
}
void StatusZero(const Gm& status) {
  for(size_t off=0;off<status.size;off+=4) {
    int32_t code;std::memcpy(&code,status.ptr+off,4);
    if(code)throw std::runtime_error("normal case wrote nonzero status");
  }
}
void StatusNonzero(const Gm& status) {
  bool found=false;
  for(size_t off=0;off<status.size;off+=4) {
    int32_t code;std::memcpy(&code,status.ptr+off,4);found|=code!=0;
  }
  if(!found)throw std::runtime_error("invalid case did not report status");
}
int32_t StatusWord(const Gm& status,size_t word) {
  if((word+1)*4>status.size)throw std::runtime_error("status word outside tensor");
  int32_t value;std::memcpy(&value,status.ptr+word*4,4);return value;
}
void Poison(Gm& gm) {
  if(gm.size%4)throw std::runtime_error("poison target is not FP32 aligned");
  constexpr uint32_t bits=0x7fc00000U;
  for(size_t off=0;off<gm.size;off+=4)std::memcpy(gm.ptr+off,&bits,4);
}
void Close(const Gm& got,const std::string& path) {
  auto reference=Read(path,got.size);
  for(size_t off=0;off<got.size;off+=4) {
    float a,b;std::memcpy(&a,got.ptr+off,4);std::memcpy(&b,reference.data()+off,4);
    if(a==b)continue;
    if(!std::isfinite(a)||!std::isfinite(b)||std::abs(a-b)>0.005F+0.005F*std::abs(b))
      throw std::runtime_error("frozen oracle mismatch "+path+" at "+std::to_string(off));
  }
}
void HalfBits(uint8_t* at,uint16_t bits) {
  at[0]=static_cast<uint8_t>(bits&255);at[1]=static_cast<uint8_t>(bits>>8);
}
size_t SlotOffset(const Gm& table,int64_t position,int64_t blockTokens,
                  int64_t prefix,int64_t stride,int64_t d,int64_t hk,int64_t head) {
  int32_t virtualPage;std::memcpy(&virtualPage,table.ptr+(position/128)*4,4);
  if(virtualPage<0)throw std::runtime_error("bad golden virtual block");
  const int64_t physical=virtualPage/(blockTokens/128);
  const int64_t inpage=(virtualPage%(blockTokens/128))*128+position%128;
  return static_cast<size_t>(prefix+physical*stride+(inpage*hk+head)*(d/2+8));
}
void SetScale(Gm& raw,const Gm& table,int64_t position,int64_t blockTokens,
              int64_t prefix,int64_t stride,int64_t d,int64_t hk,bool value,
              uint16_t bits) {
  const size_t base=SlotOffset(table,position,blockTokens,prefix,stride,d,hk,0);
  const size_t offset=base+(value?d/4+4:0)+d/4;
  if(offset+2>raw.size)throw std::runtime_error("scale offset out of raw arena");
  HalfBits(raw.ptr+offset,bits);
}
void SetHugeFiniteQuery(Gm& qr,Gm& raw,const Gm& table,int64_t context,
                        int64_t blockTokens,int64_t prefix,int64_t stride,
                        int64_t hq,int64_t hk,int64_t d,int64_t recent,int64_t sink) {
  const float huge=1.0e30F;
  const long double score=static_cast<long double>(d)*huge*262016.0L/std::sqrt(double(d));
  const int64_t historyRows=std::min(context,std::max(sink,context+4-recent))-sink;
  if(!(std::isfinite(static_cast<float>(d*huge)) &&
       std::isfinite(static_cast<float>(score)) &&
       historyRows*score>std::numeric_limits<float>::max()))
    throw std::runtime_error("huge Q failed to isolate finite query/score from old sum overflow");
  for(int64_t column=0;column<d;++column)
    std::memcpy(qr.ptr+column*4,&huge,4);
  const int64_t historyEnd=std::min(context,std::max(sink,context+4-recent));
  for(int64_t position=sink;position<historyEnd;++position) {
    const size_t base=SlotOffset(table,position,blockTokens,prefix,stride,d,hk,0);
    if(base+d/4+4>raw.size)throw std::runtime_error("huge-Q packed offset out of raw arena");
    std::memset(raw.ptr+base,0xff,d/4); // Every INT2 code is exactly 3.
    HalfBits(raw.ptr+base+d/4,0x7bff);  // Largest finite FP16 scale.
    HalfBits(raw.ptr+base+d/4+2,0x7bff); // Largest finite FP16 zero.
  }
  (void)hq;
}
void SetHugeCancellingQuery(Gm& qr,Gm& raw,const Gm& table,int64_t context,
                            int64_t blockTokens,int64_t prefix,int64_t stride,
                            int64_t hk,int64_t d,int64_t recent,int64_t sink) {
  if(d%2)throw std::runtime_error("canceling Q requires an even head dimension");
  const float huge=2.0e30F;
  for(int64_t column=0;column<d;++column) {
    const float value=column%2?-huge:huge;
    std::memcpy(qr.ptr+column*4,&value,4);
  }
  const int64_t historyEnd=std::min(context,std::max(sink,context+4-recent));
  for(int64_t position=sink;position<historyEnd;++position) {
    const size_t base=SlotOffset(table,position,blockTokens,prefix,stride,d,hk,0);
    if(base+d/4+4>raw.size)throw std::runtime_error("canceling-Q packed offset out of raw arena");
    std::memset(raw.ptr+base,0x22,d/4); // Codes 2,0,2,0 -> K signs follow Q.
    HalfBits(raw.ptr+base+d/4,0x7bff);   // +65504 FP16 scale.
    HalfBits(raw.ptr+base+d/4+2,0xfbff); // -65504 FP16 zero.
  }
  const long double perScore=static_cast<long double>(d)*huge*65504.0L/std::sqrt(double(d));
  if(!(std::isfinite(static_cast<float>(perScore)) &&
       (historyEnd-sink)*perScore>std::numeric_limits<float>::max()))
    throw std::runtime_error("canceling Q failed to isolate finite scores and overflowed old sum");
}
}

int main(int argc,char** argv) {
  try {
    if(argc!=3)throw std::runtime_error("usage: oscar_finite_guard_cpu mode golden_dir");
    const std::string mode=argv[1],dir=argv[2];
    if(mode!="normal" && mode!="poison" && mode!="empty" && mode!="masked_empty" &&
       mode!="nan_q" && mode!="inf_q" &&
       mode!="nan_scale" && mode!="inf_scale" && mode!="neg_scale" &&
       mode!="huge_q" && mode!="huge_cancel_q" &&
       mode!="cross_lane_bad_k" && mode!="cross_lane_bad_v" &&
       mode!="bad_page" && mode!="masked_bad_k")
      throw std::runtime_error("unknown finite-guard case");
    std::ifstream meta(dir+"/shape.txt");
    int64_t n,hq,hk,d,context,sink,recent,spec,blockTokens,blocks,prefix,stride,cores;
    if(!(meta>>n>>hq>>hk>>d>>context>>sink>>recent>>spec>>blockTokens>>blocks>>prefix>>stride>>cores))
      throw std::runtime_error("invalid golden shape");
    int64_t requests=1,splits=1;meta>>requests;meta>>splits;
    if(n<=0||hq<=0||hk<=0||d<=0||requests!=1||splits<=0||cores<=0)
      throw std::runtime_error("finite-guard CPU fixture requires one request and positive geometry");
    const int64_t segments=3*splits,tasksCount=n*hk*segments,windowRows=sink+recent+spec;
    Gm q(dir+"/q.bin",n*hq*d*2),qr(dir+"/qr.bin",n*hq*d*4);
    Gm ck(dir+"/ck.bin",n*hk*d*2),cv(dir+"/cv.bin",n*hk*d*2);
    Gm rv(dir+"/rv.bin",d*d*4),raw(dir+"/raw.bin",prefix+blocks*stride);
    Gm table(dir+"/table.bin",requests*8*4),wk(dir+"/wk.bin",blocks*windowRows*hk*d*2);
    Gm wv(dir+"/wv.bin",blocks*windowRows*hk*d*2),tags(dir+"/tags.bin",blocks*windowRows*8);
    Gm starts(dir+"/starts.bin",(requests+1)*4),lens(dir+"/lens.bin",requests*4);
    Gm slots(dir+"/slots.bin",n*8),tasks(tasksCount*16*8),positions(n*8);
    const size_t partialBytes=n*hq*segments*d*4,lseBytes=n*hq*segments*4;
    Gm oldPartial(partialBytes),oldLse(lseBytes),oldStatus(tasksCount*2*4);
    Gm newPartial(partialBytes),newLse(lseBytes),newStatus(tasksCount*2*4);
    const int64_t oldCoreBytes=oscar_ascend::attention_workspace_per_core(d);
    Gm oldWorkspace(cores*oldCoreBytes),newWorkspace(cores*oldCoreBytes);
    Gm output(n*hq*d*4),outputLse(n*hq*4),mergeStatus(n*hq*4);
    if(mode=="poison") {
      Poison(oldWorkspace);Poison(newWorkspace);
      Poison(oldPartial);Poison(newPartial);Poison(oldLse);Poison(newLse);
    }
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(oscar_prepare_attention_tasks_kernel,4,starts.ptr,lens.ptr,slots.ptr,
        tasks.ptr,positions.ptr,requests,n,hq,hk,sink,recent,splits,true,
        static_cast<uint8_t*>(nullptr),int64_t{0},false);
    if(mode=="masked_empty"||mode=="masked_bad_k") {
      if(n!=4 || splits!=1 || context+3-recent<sink || context+4-recent>context)
        throw std::runtime_error("masked case requires a four-query live history fixture");
      // One compressed key is invisible to the first three queries and
      // visible to the fourth. This invokes Softmax with no visible rows,
      // unlike the context=0 empty source that never enters its KV loop.
      auto* leader=reinterpret_cast<int64_t*>(tasks.ptr);
      const int64_t key=mode=="masked_bad_k"?context-1:context+3-recent;
      if(leader[1]!=4 || leader[3]>key ||
          (mode=="masked_empty" && leader[4]<=key))
        throw std::runtime_error("masked case missed source0 leader domain");
      leader[3]=key;leader[4]=key+1;
      if(mode=="masked_bad_k")
        SetScale(raw,table,key,blockTokens,prefix,stride,d,hk,false,0x7e01);
    }
    if(mode=="nan_q"||mode=="inf_q") {
      const float bad=mode=="nan_q"?std::nanf(""):std::numeric_limits<float>::infinity();
      std::memcpy(qr.ptr,&bad,4);
    } else if(mode=="nan_scale"||mode=="inf_scale"||mode=="neg_scale") {
      SetScale(raw,table,64,blockTokens,prefix,stride,d,hk,false,
               mode=="nan_scale"?0x7e01:mode=="inf_scale"?0x7c00:0xbc00);
    } else if(mode=="cross_lane_bad_k"||mode=="cross_lane_bad_v") {
      SetScale(raw,table,80,blockTokens,prefix,stride,d,hk,
               mode=="cross_lane_bad_v",0x7e01);
    } else if(mode=="huge_q") {
      SetHugeFiniteQuery(qr,raw,table,context,blockTokens,prefix,stride,hq,hk,d,recent,sink);
    } else if(mode=="huge_cancel_q") {
      SetHugeCancellingQuery(qr,raw,table,context,blockTokens,prefix,stride,hk,d,recent,sink);
    } else if(mode=="bad_page") {
      // First source0 KV unit has no valid physical page. AIV may overwrite
      // dequant UB without an intervening MTE2, exercising the #126 fence.
      const int32_t invalid=-1;std::memcpy(table.ptr,&invalid,sizeof(invalid));
    }
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_RUN_KF(oscar_attention_cv_fast_kernel,cores,q.ptr,qr.ptr,ck.ptr,cv.ptr,
        rv.ptr,raw.ptr,table.ptr,wk.ptr,wv.ptr,tags.ptr,tasks.ptr,oldPartial.ptr,
        oldLse.ptr,oldStatus.ptr,oldWorkspace.ptr,n,hq,hk,d,requests,int64_t{8},tasksCount,
        blockTokens,blocks,prefix,stride,windowRows*hk*d,windowRows,sink,recent,
        spec,splits,1.0F/std::sqrt(float(d)));
    ICPU_RUN_KF(oscar_attention_cv_checked_fast_kernel,cores,q.ptr,qr.ptr,ck.ptr,cv.ptr,
        rv.ptr,raw.ptr,table.ptr,wk.ptr,wv.ptr,tags.ptr,tasks.ptr,newPartial.ptr,
        newLse.ptr,newStatus.ptr,newWorkspace.ptr,n,hq,hk,d,requests,int64_t{8},tasksCount,
        blockTokens,blocks,prefix,stride,windowRows*hk*d,windowRows,sink,recent,
        spec,splits,1.0F/std::sqrt(float(d)));
    EqualBytes(oldStatus,newStatus,"final two-AIV status");
    if(mode=="normal"||mode=="poison"||mode=="empty"||mode=="masked_empty") {
      EqualBytes(oldPartial,newPartial,"partial");EqualBytes(oldLse,newLse,"LSE");
      StatusZero(oldStatus);
      if(mode!="masked_empty") {
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        ICPU_RUN_KF(oscar_merge_lse_kernel,4,newPartial.ptr,newLse.ptr,
            output.ptr,outputLse.ptr,mergeStatus.ptr,n*hq,segments,d);
        StatusZero(mergeStatus);
        Close(output,dir+"/expected_output.bin");Close(outputLse,dir+"/expected_lse.bin");
      }
      if(mode=="empty"||mode=="masked_empty") {
        const int64_t emptyTokens=mode=="masked_empty"?3:n;
        for(int64_t token=0;token<emptyTokens;++token)for(int64_t head=0;head<hq;++head) {
          const int64_t offset=((token*hq+head)*segments)*4;
          float lse;std::memcpy(&lse,newLse.ptr+offset,4);
          if(!std::isinf(lse)||lse>0)throw std::runtime_error("empty history LSE is not -inf");
          for(int64_t column=0;column<d;++column) {
            float value;std::memcpy(&value,newPartial.ptr+((token*hq+head)*segments*d+column)*4,4);
            if(value!=0.0F)throw std::runtime_error("empty history partial is not zero");
          }
        }
      }
    } else {
      StatusNonzero(oldStatus);
      if((mode=="huge_q"||mode=="huge_cancel_q") &&
          StatusWord(oldStatus,0)!=2)
        throw std::runtime_error("old score finite check did not emit expected code 2; got "+
            std::to_string(StatusWord(oldStatus,0))+"/"+
            std::to_string(StatusWord(oldStatus,1)));
      if(mode=="neg_scale"||mode=="masked_bad_k") {
        const auto left=StatusWord(oldStatus,0),right=StatusWord(oldStatus,1);
        if((left!=3 && right!=3) || left==2 || right==2)
          throw std::runtime_error("old metadata error priority was not code 3");
      }
      if((mode=="cross_lane_bad_k"||mode=="cross_lane_bad_v") &&
         StatusWord(oldStatus,0)!=3 && StatusWord(oldStatus,1)!=3)
        throw std::runtime_error("old cross-lane metadata case did not report code 3");
      // Invalid paths can publish different NaN payload bytes. Compare exact
      // final error status and finite/nonfinite classification, not NaN bits.
      SameClass(oldPartial,newPartial,"partial");SameClass(oldLse,newLse,"LSE");
    }
    std::cout<<"{\"backend\":\"ascendc_cpu_debug\",\"op\":\"finite_guard\",\"mode\":\""
             <<mode<<"\",\"status\":\"passed\"}"<<std::endl;
    return 0;
  } catch(const std::exception& error) {
    std::cerr<<"FINITE_GUARD_CPU_FAILED: "<<error.what()<<std::endl;return 1;
  }
}
