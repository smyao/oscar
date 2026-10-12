// Archive #126/#129/#145/#148-155; D.4: official CPU-debug is numerical
// evidence only. Compare the actual fe0 CV source0 implementation to bounded
// producer/consumer launches, including FP32 recurrence across 8192 rows.
#include "../striped_prefill/probe_common.h"
#include "../../include/oscar_attention_launch.h"
#include <filesystem>
#define BASE_ARGS uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*, \
 uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*, \
 int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t, \
 int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,float
extern "C" void oscar_attention_cv_kernel(BASE_ARGS);
extern "C" void oscar_shared_fp32_history_consumer_kernel(BASE_ARGS,uint8_t*,uint8_t*,uint8_t*,uint8_t*,int64_t,int64_t);
extern "C" void oscar_shared_fp32_history_producer_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
 int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);
extern "C" void oscar_prepare_attention_tasks_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
 int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,bool,uint8_t*,int64_t,bool);
#undef BASE_ARGS
int main(int argc,char** argv) {
 try {
  if(argc!=3)throw std::runtime_error("usage: consumer_cpu GOLDEN_DIR valid|future_bad|visible_bad|carry_bad");
  const std::string dir=argv[1],mode=argv[2];
  if(mode!="valid" && mode!="future_bad" && mode!="visible_bad" && mode!="carry_bad")throw std::runtime_error("bad mode");
  std::ifstream shape(dir+"/shape.txt");
  int64_t n,hq,hk,d,context,sink,recent,spec,b,blocks,prefix,stride,cores,requests,splits,columns;
  if(!(shape>>n>>hq>>hk>>d>>context>>sink>>recent>>spec>>b>>blocks>>prefix>>stride>>cores>>requests>>splits>>columns)
      || d!=256 || hq!=6 || hk!=1 || requests!=1 || splits!=1 || context<=sink)
    throw std::runtime_error("requires one D256 GQA6 S1 request with history");
  cores=20;const int64_t count=n*3,windowRows=sink+recent+spec;
  Gm q(dir+"/q.bin",n*hq*d*2),qr(dir+"/qr.bin",n*hq*d*4);
  Gm k(dir+"/ck.bin",n*hk*d*2),v(dir+"/cv.bin",n*hk*d*2),rv(dir+"/rv.bin",d*d*4);
  Gm raw(dir+"/raw.bin",prefix+blocks*stride),striped(raw.size),table(dir+"/table.bin",columns*4);
  Gm wk(dir+"/wk.bin",blocks*windowRows*d*2),wv(dir+"/wv.bin",blocks*windowRows*d*2);
  Gm tags(dir+"/tags.bin",blocks*windowRows*8),starts(dir+"/starts.bin",8),lens(dir+"/lens.bin",4),slots(dir+"/slots.bin",n*8);
  Gm tasks(count*16*8),positions(n*8),oldP(n*hq*3*d*4),newP(oldP.size),oldL(n*hq*3*4),newL(oldL.size);
  Gm oldS(count*2*4),newS(oldS.size),work(cores*oscar_ascend::attention_workspace_per_core(d));
  Gm sharedK(8192*d*4),sharedV(sharedK.size),errors(8192*4);
  Gm state(((n+20)/21)*2*(64*d+128+8)*4);
  std::memset(oldS.ptr,0,oldS.size);std::memset(newS.ptr,0,newS.size);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(oscar_prepare_attention_tasks_kernel,4,starts.ptr,lens.ptr,slots.ptr,tasks.ptr,positions.ptr,
      requests,n,hq,hk,sink,recent,splits,true,static_cast<uint8_t*>(nullptr),columns,false);
  auto* task=reinterpret_cast<int64_t*>(tasks.ptr);
  int64_t maxEnd=sink;
  for(int64_t id=0;id<count;++id) {
    if(task[id*16+7]!=0)task[id*16+1]=0;
    else if(task[id*16+1]>0)maxEnd=std::max(maxEnd,task[id*16+4]);
  }
  if(mode=="carry_bad") {
    if(context-sink<8192)throw std::runtime_error("carry_bad requires more than 8192 history rows");
    const float large=1.0e38F;
    for(size_t off=0;off<qr.size;off+=4)std::memcpy(qr.ptr+off,&large,4);
    for(int64_t block=0;block<blocks;++block)for(int64_t row=0;row<b;++row) {
      auto* slot=raw.ptr+prefix+block*stride+row*136;
      std::memset(slot,0,68);const uint16_t scaleOne=0x3c00;
      std::memcpy(slot+64,&scaleOne,2);
    }
    const int64_t badRow=sink+32;int32_t virtualBlock;
    std::memcpy(&virtualBlock,table.ptr+badRow/128*4,4);
    const int64_t ratio=b/128,physical=virtualBlock/ratio,inpage=(virtualBlock%ratio)*128+badRow%128;
    const uint16_t zero=0;std::memcpy(raw.ptr+prefix+physical*stride+inpage*136+132,&zero,2);
  } else if(mode!="valid") {
    const int64_t badRow=mode=="future_bad"?maxEnd:std::min(maxEnd-1,sink+32);
    if(badRow>=context)throw std::runtime_error("fixture has no producer-only future row");
    int32_t virtualBlock;std::memcpy(&virtualBlock,table.ptr+badRow/128*4,4);
    const int64_t ratio=b/128,physical=virtualBlock/ratio,inpage=(virtualBlock%ratio)*128+badRow%128;
    // FP16 NaN zero: future-only K would poison QK before the causal mask
    // without the private tail copy. Visible failure must retain fe0 status.
    const uint16_t nan=0x7e01;
    std::memcpy(raw.ptr+prefix+physical*stride+inpage*136+66,&nan,2);
    std::memcpy(raw.ptr+prefix+physical*stride+inpage*136+134,&nan,2);
  }
  StripeRaw(raw,striped,prefix,blocks,stride,b,hk);
  const float scale=1.0F/std::sqrt(float(d));
#define COMMON(P,L,S) q.ptr,qr.ptr,k.ptr,v.ptr,rv.ptr,raw.ptr,table.ptr,wk.ptr,wv.ptr,tags.ptr,tasks.ptr,P.ptr,L.ptr,S.ptr,work.ptr,n,hq,hk,d,requests,columns,count,b,blocks,prefix,stride,windowRows*d,windowRows,sink,recent,spec,splits,scale
  AscendC::SetKernelMode(KernelMode::MIX_MODE);
  ICPU_RUN_KF(oscar_attention_cv_kernel,cores,COMMON(oldP,oldL,oldS));
  for(int64_t begin=sink;begin<maxEnd;begin+=8192) {
    const int64_t end=std::min(begin+8192,context);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(oscar_shared_fp32_history_producer_kernel,40,striped.ptr,table.ptr,sharedK.ptr,sharedV.ptr,
      errors.ptr,context,begin,end,columns,b,blocks,prefix,stride);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_RUN_KF(oscar_shared_fp32_history_consumer_kernel,cores,COMMON(newP,newL,newS),
      sharedK.ptr,sharedV.ptr,errors.ptr,state.ptr,begin,end);
  }
#undef COMMON
  CheckEqual(oldP,newP,"source0 partial");CheckEqual(oldL,newL,"source0 LSE");CheckEqual(oldS,newS,"source0 status");
  int64_t failures=0;for(size_t i=0;i<oldS.size;i+=4) {int32_t code;std::memcpy(&code,oldS.ptr+i,4);failures+=code!=0;}
  if((mode=="visible_bad" || mode=="carry_bad")!=(failures>0))throw std::runtime_error("wrong error domain");
  float maxOut=0,maxLse=0;
  if(mode!="visible_bad" && mode!="carry_bad") {
    const auto expected=Read(dir+"/source0_output.bin",n*hq*d*4),expectedLse=Read(dir+"/source0_lse.bin",n*hq*4);
    for(int64_t row=0;row<n*hq;++row) {
      float a,e;std::memcpy(&a,newL.ptr+row*3*4,4);std::memcpy(&e,expectedLse.data()+row*4,4);
      maxLse=std::max(maxLse,std::abs(a-e));
      if(!std::isfinite(a)||std::abs(a-e)>0.005F+0.005F*std::abs(e))throw std::runtime_error("frozen LSE oracle failed");
      for(int64_t col=0;col<d;++col) {
        std::memcpy(&a,newP.ptr+(row*3*d+col)*4,4);std::memcpy(&e,expected.data()+(row*d+col)*4,4);
        maxOut=std::max(maxOut,std::abs(a-e));
        if(!std::isfinite(a)||std::abs(a-e)>0.005F+0.005F*std::abs(e))throw std::runtime_error("frozen output oracle failed");
      }
    }
  }
  // Actual bytes and tasks retained for same-source CAModel replay.
  if(mode=="valid") {
  Write(dir+"/tasks_source0.bin",tasks.ptr,tasks.size);Write(dir+"/raw_striped.bin",striped.ptr,striped.size);
  Write(dir+"/shared_cpu_partial.bin",newP.ptr,newP.size);Write(dir+"/shared_cpu_lse.bin",newL.ptr,newL.size);
  Write(dir+"/shared_cpu_status.bin",newS.ptr,newS.size);
  }
  std::cout<<"{\"backend\":\"ascendc_cpu_debug\",\"status\":\"passed\",\"tokens\":"<<n<<",\"context\":"<<context
   <<",\"mode\":\""<<mode<<"\",\"tiles\":"<<(maxEnd-sink+8191)/8192<<",\"bitwise\":true,\"oracle_maxout\":"<<maxOut
   <<",\"oracle_maxlse\":"<<maxLse<<",\"nonzero_status\":"<<failures<<"}"<<std::endl;
  return 0;
 } catch(const std::exception& e) {std::cerr<<"SHARED_FP32_CONSUMER_FAILED: "<<e.what()<<std::endl;return 1;}
}
