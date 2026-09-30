// Official AscendC CPU-debug: same task table, canonical old fast CV versus
// losslessly permuted striped M32/HalfKv64 CV. Archive #126/#129/#145/#154.
// D.4's 6.5s all-history restore is absent from both bounded KV256 kernels.
// CPU-debug bit parity and the frozen merged oracle do not imply NPU timing.
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
#include "../../include/oscar_attention_launch.h"

#define CV_ARGS uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*, \
    uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*, \
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t, \
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,float
extern "C" void oscar_attention_cv_fast_kernel(CV_ARGS);
extern "C" void oscar_attention_cv_striped_decode_kernel(CV_ARGS);
extern "C" void oscar_attention_cv_striped_q1_kernel(CV_ARGS);
extern "C" void oscar_attention_cv_striped_decode_simd_kernel(CV_ARGS);
extern "C" void oscar_attention_cv_striped_q1_simd_kernel(CV_ARGS);
#undef CV_ARGS
extern "C" void oscar_prepare_attention_tasks_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,bool,uint8_t*,int64_t,bool);
extern "C" void oscar_merge_lse_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    int64_t,int64_t,int64_t);

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
void CheckOracle(const Gm& partial,const Gm& partLse,const std::string& dir,
    const Gm& slots,int64_t n,int64_t hq,int64_t d,int64_t segments) {
  Gm output(n*hq*d*4),lse(n*hq*4),status(n*hq*4);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(oscar_merge_lse_kernel,4,partial.ptr,partLse.ptr,output.ptr,lse.ptr,
      status.ptr,n*hq,segments,d);
  const auto expectedOutput=Read(dir+"/expected_output.bin",output.size);
  const auto expectedLse=Read(dir+"/expected_lse.bin",lse.size);
  int64_t live=0;
  for(int64_t token=0;token<n;++token) {
    int64_t slot;std::memcpy(&slot,slots.ptr+token*8,8);if(slot<0)continue;
    ++live;
    for(int64_t head=0;head<hq;++head) {
      const int64_t row=token*hq+head;
      int32_t code;std::memcpy(&code,status.ptr+row*4,4);
      if(code)throw std::runtime_error("live merge status is nonzero");
      float actual,expected;
      std::memcpy(&actual,lse.ptr+row*4,4);
      std::memcpy(&expected,expectedLse.data()+row*4,4);
      if(!std::isfinite(actual) || std::abs(actual-expected)>0.005F+0.005F*std::abs(expected))
        throw std::runtime_error("LSE exceeds frozen tolerance");
      for(int64_t dim=0;dim<d;++dim) {
        const int64_t index=row*d+dim;
        std::memcpy(&actual,output.ptr+index*4,4);
        std::memcpy(&expected,expectedOutput.data()+index*4,4);
        if(!std::isfinite(actual) || std::abs(actual-expected)>0.005F+0.005F*std::abs(expected))
          throw std::runtime_error("output exceeds frozen tolerance");
      }
    }
  }
  if(live==0)throw std::runtime_error("fixture has no live tokens");
}
}

int main(int argc,char** argv) {
  try {
    if((argc!=3 && argc!=4) ||
        (std::string(argv[1])!="q4" && std::string(argv[1])!="q1" &&
         std::string(argv[1])!="q4simd" && std::string(argv[1])!="q1simd"))
      throw std::runtime_error("usage: oscar_striped_decode_cpu q4|q1|q4simd|q1simd GOLDEN_DIR [EXPORT_DIR]");
    const std::string mode=argv[1],dir=argv[2];
    std::ifstream shape(dir+"/shape.txt");
    int64_t n,hq,hk,d,context,sink,recent,spec,b,blocks,prefix,stride,cores;
    if(!(shape>>n>>hq>>hk>>d>>context>>sink>>recent>>spec>>b>>blocks>>prefix>>stride>>cores))
      throw std::runtime_error("invalid shape file");
    int64_t requests=1,splits=1;shape>>requests;shape>>splits;
    if(d!=256 || hk<=0 || hq%hk || hq/hk>8 || splits<1 || cores<1)
      throw std::runtime_error("striped M32 experiment supports D256 and GQA<=8 only");
    const int64_t segments=3*splits,tasksCount=n*hk*segments;
    const int64_t windowRows=sink+recent+spec;
    Gm q(dir+"/q.bin",n*hq*d*2),qr(dir+"/qr.bin",n*hq*d*4);
    Gm k(dir+"/ck.bin",n*hk*d*2),v(dir+"/cv.bin",n*hk*d*2),rv(dir+"/rv.bin",d*d*4);
    Gm raw(dir+"/raw.bin",prefix+blocks*stride),striped(raw.size),table(dir+"/table.bin",requests*8*4);
    Gm wk(dir+"/wk.bin",blocks*windowRows*hk*d*2),wv(dir+"/wv.bin",blocks*windowRows*hk*d*2);
    Gm tags(dir+"/tags.bin",blocks*windowRows*8),starts(dir+"/starts.bin",(requests+1)*4);
    Gm lens(dir+"/lens.bin",requests*4),slots(dir+"/slots.bin",n*8);
    Gm tasks(tasksCount*16*8),positions(n*8);
    Gm oldPartial(n*hq*segments*d*4),oldLse(n*hq*segments*4),oldStatus(tasksCount*2*4);
    Gm newPartial(oldPartial.size),newLse(oldLse.size),newStatus(oldStatus.size);
    Gm oldWork(cores*oscar_ascend::attention_workspace_per_core(d));
    Gm newWork(oldWork.size);
    StripeRaw(raw,striped,prefix,blocks,stride,b,hk);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(oscar_prepare_attention_tasks_kernel,4,starts.ptr,lens.ptr,slots.ptr,
        tasks.ptr,positions.ptr,requests,n,hq,hk,sink,recent,splits,true,
        static_cast<uint8_t*>(nullptr),int64_t{0},false);
    for(int64_t id=0;id<tasksCount;++id) {
      const auto* row=reinterpret_cast<const int64_t*>(tasks.ptr)+id*16;
      if(row[1]>0 && row[1]>((mode=="q1" || mode=="q1simd")?1:4))
        throw std::runtime_error("fixture exceeds short-query owner contract");
    }
    const float scale=1.0F/std::sqrt(float(d));
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
#define CV_CALL(kernel,rawArg,partArg,lseArg,statusArg,workArg) \
    ICPU_RUN_KF(kernel,cores,q.ptr,qr.ptr,k.ptr,v.ptr,rv.ptr,rawArg,table.ptr, \
        wk.ptr,wv.ptr,tags.ptr,tasks.ptr,partArg,lseArg,statusArg,workArg, \
        n,hq,hk,d,requests,int64_t{8},tasksCount,b,blocks,prefix,stride, \
        windowRows*hk*d,windowRows,sink,recent,spec,splits,scale)
    CV_CALL(oscar_attention_cv_fast_kernel,raw.ptr,oldPartial.ptr,oldLse.ptr,oldStatus.ptr,oldWork.ptr);
    if(mode=="q4") {
      CV_CALL(oscar_attention_cv_striped_decode_kernel,striped.ptr,newPartial.ptr,newLse.ptr,
          newStatus.ptr,newWork.ptr);
    } else if(mode=="q1") {
      CV_CALL(oscar_attention_cv_striped_q1_kernel,striped.ptr,newPartial.ptr,newLse.ptr,
          newStatus.ptr,newWork.ptr);
    } else if(mode=="q4simd") {
      CV_CALL(oscar_attention_cv_striped_decode_simd_kernel,striped.ptr,newPartial.ptr,
          newLse.ptr,newStatus.ptr,newWork.ptr);
    } else {
      CV_CALL(oscar_attention_cv_striped_q1_simd_kernel,striped.ptr,newPartial.ptr,
          newLse.ptr,newStatus.ptr,newWork.ptr);
    }
#undef CV_CALL
    CheckEqual(oldPartial,newPartial,"partial");
    CheckEqual(oldLse,newLse,"LSE");
    CheckEqual(oldStatus,newStatus,"status");
    for(size_t offset=0;offset<oldStatus.size;offset+=4) {
      int32_t code;std::memcpy(&code,oldStatus.ptr+offset,4);
      if(code)throw std::runtime_error("valid golden has nonzero CV status");
    }
    CheckOracle(newPartial,newLse,dir,slots,n,hq,d,segments);
    if(argc==4) {
      const std::string exportDir=argv[3];
      Write(exportDir+"/tasks.bin",tasks.ptr,tasks.size);
      Write(exportDir+"/raw_striped.bin",striped.ptr,striped.size);
      std::ofstream attrs(exportDir+"/attrs.txt",std::ios::trunc);
      if(!(attrs<<n<<' '<<hq<<' '<<hk<<' '<<d<<' '<<requests<<' '<<8<<' '
          <<tasksCount<<' '<<b<<' '<<blocks<<' '<<prefix<<' '<<stride<<' '
          <<windowRows*hk*d<<' '<<windowRows<<' '<<sink<<' '<<recent<<' '
          <<spec<<' '<<splits<<' '<<scale<<' '<<cores<<'\n') || !attrs.flush())
        throw std::runtime_error("failed to write CAModel attrs");
    }
    std::cout<<"{\"backend\":\"ascendc_cpu_debug\",\"op\":\"striped_decode_"<<mode
             <<"\",\"status\":\"passed\",\"bitwise\":true,\"oracle\":true}"<<std::endl;
    return 0;
  }catch(const std::exception& e) {
    std::cerr<<"STRIPED_DECODE_CPU_FAILED: "<<e.what()<<std::endl;return 1;
  }
}
