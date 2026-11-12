// Archive #145/#150/#151/#154/#164/#172 and complete startup D.4 reviewed.
// D.4: this tests the bounded fused-score path, never history restoration.
// Python injects actual old/new Softmax and actual finite helper bodies.
// Host recording primitives prove control, input/error and bounded scratch
// equivalence; they do NOT simulate CANN reduction trees or asynchronous DMA.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#define __aicore__
#define __ubuf__
#define ASCEND_IS_AIV (true)
constexpr int PIPE_V=0;
using bfloat16_t=uint16_t;
void require(bool v,const char* message) {if(!v)throw std::runtime_error(message);}
int64_t Min64(int64_t a,int64_t b){return std::min(a,b);}
int64_t Max64(int64_t a,int64_t b){return std::max(a,b);}
uint32_t bits(float x){uint32_t b;std::memcpy(&b,&x,4);return b;}
float value(uint32_t b){float x;std::memcpy(&x,&b,4);return x;}

struct Counters {
  int scalarReduces=0,certificateReduces=0,fallbackReduces=0,maskCalls=0;
  int copies=0,softmaxCalls=0,casts=0;
  std::vector<int> fences;
  std::vector<uint32_t> softmaxInput;
};
thread_local Counters* counters=nullptr;
thread_local int64_t vectorCount=0;
thread_local float accValue=0;

namespace AscendC {
enum class HardEvent {MTE2_V,V_S,S_V,V_MTE3,MTE3_MTE2};
enum class MaskMode {COUNTER};
template<HardEvent E> void Fence(){counters->fences.push_back(int(E));}
template<int> void PipeBarrier(){}
void SetMaskCount(){}
void SetMaskNorm(){}
void ResetMask(){vectorCount=0;}
template<class T,MaskMode> void SetVectorMask(int64_t count){vectorCount=count;}
template<class T> struct LocalTensor {
  unsigned char* ptr;
  size_t bytes;
  LocalTensor operator[](int64_t offset)const {
    require(offset>=0 && size_t(offset)*sizeof(T)<=bytes,"tensor offset out of bounds");
    return {ptr+offset*sizeof(T),bytes-size_t(offset)*sizeof(T)};
  }
  T GetValue(int64_t offset)const {
    require(offset>=0 && (size_t(offset)+1)*sizeof(T)<=bytes,"tensor read out of bounds");
    T item;std::memcpy(&item,ptr+offset*sizeof(T),sizeof(T));return item;
  }
  void SetValue(int64_t offset,T item)const {
    require(offset>=0 && (size_t(offset)+1)*sizeof(T)<=bytes,"tensor write out of bounds");
    std::memcpy(ptr+offset*sizeof(T),&item,sizeof(T));
  }
  int64_t GetSize()const{return bytes/sizeof(T);}
  void* GetPhyAddr()const{return ptr;}
  template<class U> LocalTensor<U> ReinterpretCast()const{return {ptr,bytes};}
};
template<class T> void Abs(LocalTensor<T> dst,LocalTensor<T> src,int64_t count) {
  for(int64_t i=0;i<count;++i)dst.SetValue(i,std::abs(src.GetValue(i)));
}
}
using namespace AscendC;
namespace oscar_ascend_device {
using AscendC::Fence;
bool Finite(float x){return std::isfinite(x);}
}
using oscar_ascend_device::Finite;

// Both old ReduceSum and direct vcadd use the SAME deterministic host sum.
// Actual A2 accumulator-tree arithmetic remains the existing NPU proof gate.
float hostSum(const float* p,int64_t count) {
  float result=0;
  for(int64_t i=0;i<count;++i)result+=p[i];
  return result;
}
void vcadd(float* dst,float* src,int,int,int,int,int) {
  require(vectorCount>0,"vcadd missing count mask");
  accValue=hostSum(src,vectorCount);*dst=accValue;
  if(vectorCount>256)++counters->certificateReduces;else ++counters->fallbackReduces;
}
int64_t get_acc_val(){return int64_t(bits(accValue));}

@@ROW_FINITE_HELPER@@
@@CERTIFICATE_HELPER@@

struct Buffer {
  size_t size;
  std::vector<uint64_t> backing;
  explicit Buffer(size_t bytes):size(bytes),backing((bytes+7)/8+2,0) {
    backing.front()=backing.back()=0x7193aaf015739badULL;
  }
  template<class T> LocalTensor<T> Get(int64_t count=-1) {
    if(count<0)count=size/sizeof(T);
    require(size_t(count)*sizeof(T)<=size,"buffer request exceeds allocation");
    return {reinterpret_cast<unsigned char*>(backing.data()+1),size_t(count)*sizeof(T)};
  }
  void guard()const {
    require(backing.front()==0x7193aaf015739badULL && backing.back()==0x7193aaf015739badULL,
            "scratch buffer canary damaged");
  }
};
template<class T> void Duplicate(LocalTensor<T> dst,T x,int64_t n) {
  for(int64_t i=0;i<n;++i)dst.SetValue(i,x);
}
template<class T> void DataCopy(LocalTensor<T> dst,LocalTensor<T> src,int64_t n) {
  ++counters->copies;
  for(int64_t i=0;i<n;++i)dst.SetValue(i,src.GetValue(i));
}
void Muls(LocalTensor<float> dst,LocalTensor<float> src,float scale,int64_t n) {
  for(int64_t i=0;i<n;++i)dst.SetValue(i,src.GetValue(i)*scale);
}
void Adds(LocalTensor<float> dst,LocalTensor<float> src,float x,int64_t n) {
  for(int64_t i=0;i<n;++i)dst.SetValue(i,src.GetValue(i)+x);
}
void ReduceSum(LocalTensor<float> dst,LocalTensor<float> src,LocalTensor<float>,int64_t n) {
  ++counters->scalarReduces;
  require(n==256,"original finite row width changed");
  dst.SetValue(0,hostSum(static_cast<float*>(src.GetPhyAddr()),n));
}
namespace oscar_range_mask {
void MaskComplement(LocalTensor<float> scores,int lo0,int hi0,int lo1,int hi1) {
  ++counters->maskCalls;
  for(int i=0;i<256;++i)
    if(!((i>=lo0 && i<hi0)||(i>=lo1 && i<hi1)))scores.SetValue(i,-INFINITY);
}
}
struct SoftMaxShapeInfo {uint32_t rows,columns,originalRows,originalColumns;};
int SoftMaxFlashV2TilingFunc(SoftMaxShapeInfo,int,int,size_t,bool,bool){return 0;}
constexpr int kCvSoftmaxConfig=0;
template<class T,bool,bool,bool,bool,int>
void SoftmaxFlashV2(LocalTensor<float> out,LocalTensor<float> sum,LocalTensor<float> maximum,
    LocalTensor<float> scores,LocalTensor<float> alpha,LocalTensor<float>,LocalTensor<float>,
    LocalTensor<uint8_t>,int,SoftMaxShapeInfo shape) {
  ++counters->softmaxCalls;
  for(uint32_t r=0;r<shape.rows;++r) {
    for(uint32_t j=0;j<shape.columns;++j)
      counters->softmaxInput.push_back(bits(scores.GetValue(r*shape.columns+j)));
    // Recording substitute: its only purpose is to detect changed shape,
    // scores or downstream ownership. It is not an attention oracle.
    sum.SetValue(r,1.0F);maximum.SetValue(r,2.0F);alpha.SetValue(r,0.5F);
    for(uint32_t j=0;j<shape.columns;++j)
      out.SetValue(r*shape.columns+j,float((r+j)%17)/32.0F);
  }
}
struct BrcbParams {uint8_t a,b;};
void Brcb(LocalTensor<float> dst,LocalTensor<float> src,uint8_t repeats,BrcbParams) {
  for(int i=0;i<repeats*8;++i)for(int j=0;j<8;++j)dst.SetValue(i*8+j,src.GetValue(i));
}
struct BinaryRepeatParams {uint8_t a,b,c,d,e,f;};
void Mul(LocalTensor<float> dst,LocalTensor<float> lhs,LocalTensor<float> rhs,uint64_t count,
    uint8_t repeats,BinaryRepeatParams p) {
  for(int r=0;r<repeats;++r)for(uint64_t i=0;i<count;++i)
    dst.SetValue(r*p.d*8+i,lhs.GetValue(r*p.e*8+i)*rhs.GetValue(r*p.f*8));
}
namespace oscar_bf16 {
void CastRne(LocalTensor<bfloat16_t> dst,LocalTensor<float> src,int64_t n) {
  ++counters->casts;
  for(int64_t i=0;i<n;++i) {
    uint32_t b=bits(src.GetValue(i));b+=0x7fff+((b>>16)&1);
    dst.SetValue(i,b>>16);
  }
}
}

constexpr int D=256,kHalfRows=32,kLaneRows=64,kKvRows=256,kElements=16*D,kHistoryElements=32*D;
constexpr float kNegativeInfinity=-INFINITY;
struct Geometry {int64_t hq=6,hk=1,sink=64,recent=256;float scale=0.125F;};
struct Kernel {
  Geometry g;
  int64_t qcount=21,qbegin=37,requestBegin=37,kind=0,context=4096,kvend=320;
  int64_t qOffset=0,scoreOffset=0;
  int32_t lane=0,error=0;
  Buffer scoreBuf{32*256*4},accBuf{64*256*4},statsBuf{192*4},scratchBuf{128*4};
  Buffer naturalBuf{32*(256+16)*2},dequantBuf{32*256*4};
  Buffer gmScores{128*256*4},gmP{128*256*2};
  LocalTensor<float> work=gmScores.Get<float>();
  LocalTensor<bfloat16_t> operands=gmP.Get<bfloat16_t>();
  Counters count;
  int64_t ProbabilityOffset(int64_t q)const{return q;}
  @@ADD_VISIBLE_RANGE@@
  @@OLD_SOFTMAX@@
  @@NEW_SOFTMAX@@
  void guard()const {
    scoreBuf.guard();accBuf.guard();statsBuf.guard();scratchBuf.guard();
    naturalBuf.guard();dequantBuf.guard();gmScores.guard();gmP.guard();
  }
};
void compare(Buffer& a,Buffer& b) {
  require(a.size==b.size,"buffer sizes differ");
  require(std::memcmp(a.backing.data()+1,b.backing.data()+1,a.size)==0,"published buffer differs");
}

int main() {
  int cases=0,certified=0,fallback=0,masked=0,empty=0;
  const std::vector<int64_t> starts={64,3712,4096,0,64,400,4096,4096,4352};
  const std::vector<int64_t> kinds={0,0,0,1,1,1,2,2,2};
  const std::vector<int64_t> ends={320,3968,4352,256,319,656,4352,4200,4608};
  const std::vector<int64_t> offsets={0,0,0,0,0,0,512,0,200};
  for(size_t mode=0;mode<starts.size();++mode)
  for(int rowBase:{0,32,64,96})
  for(int qcount:{0,1,4,21})
  for(int poison=0;poison<7;++poison)
  for(int priorError:{0,4}) {
    Kernel old,newer;
    for(Kernel* k:{&old,&newer}) {
      k->kind=kinds[mode];k->kvend=ends[mode];k->qbegin+=offsets[mode];
      k->qcount=qcount;k->lane=rowBase/64;k->error=priorError;
      auto score=k->gmScores.Get<float>();
      for(int64_t i=0;i<score.GetSize();++i)score.SetValue(i,float(i%31-15)/8.0F);
      // Poison a live row including columns later masked on a diagonal.
      // Complete/tail/empty paths retain original pre-mask check semantics.
      int64_t p=rowBase*256;
      if(poison==1)score.SetValue(p+255,std::numeric_limits<float>::quiet_NaN());
      if(poison==2)score.SetValue(p+17,INFINITY);
      if(poison==3)for(int j=0;j<256;++j)score.SetValue(p+j,std::numeric_limits<float>::max());
      if(poison==4)for(int j=0;j<256;++j)score.SetValue(p+j,
          j%2? -std::numeric_limits<float>::max():std::numeric_limits<float>::max());
      if(poison==5)for(int j=0;j<256;++j)score.SetValue(p+j,
          j<128?std::numeric_limits<float>::max():-std::numeric_limits<float>::max());
      if(poison==6)for(int j=0;j<256;++j)score.SetValue(p+j,std::numeric_limits<float>::denorm_min());
      auto acc=k->accBuf.Get<float>();for(int i=0;i<64*256;++i)acc.SetValue(i,float(i%13)/7.0F);
      auto stats=k->statsBuf.Get<float>();for(int i=0;i<192;++i)stats.SetValue(i,i<64?-1e20F:0.0F);
      auto out=k->gmP.Get<bfloat16_t>();for(int i=0;i<128*256;++i)out.SetValue(i,0xabcd);
    }
    counters=&old.count;old.SoftmaxOld(starts[mode],rowBase);
    counters=&newer.count;newer.SoftmaxNew(starts[mode],rowBase);
    old.guard();newer.guard();
    require(old.error==newer.error,"finite/error result changed");
    require(old.count.softmaxInput==newer.count.softmaxInput,"masked Softmax input changed");
    require(old.count.softmaxCalls==newer.count.softmaxCalls,"Softmax shape/count changed");
    require(old.count.copies==newer.count.copies && old.count.casts==newer.count.casts,
            "published DMA/Cast changed");
    compare(old.accBuf,newer.accBuf);compare(old.statsBuf,newer.statsBuf);compare(old.gmP,newer.gmP);
    compare(old.gmScores,newer.gmScores);
    if(old.count.softmaxCalls) {
      require(old.count.fences.back()==int(HardEvent::MTE3_MTE2) &&
              newer.count.fences.back()==int(HardEvent::MTE3_MTE2),"score/P DMA ownership lost");
    } else {require(old.count.fences.empty() && newer.count.fences.empty(),"empty block changed");++empty;}
    if(newer.count.certificateReduces){++certified;}
    if(newer.count.fallbackReduces){++fallback;}
    if(newer.count.maskCalls){++masked;}
    if(!newer.count.certificateReduces && !newer.count.fallbackReduces) {
      require(old.count.scalarReduces==newer.count.scalarReduces,"partial row finite count changed");
      require(old.count.fences==newer.count.fences,"partial/empty path fences changed");
    }
    ++cases;
  }
  require(certified>0 && fallback>0 && masked>0 && empty>0,"missing branch coverage");
  std::cout<<"{\"cases\":"<<cases<<",\"certificate_cases\":"<<certified
      <<",\"fallback_cases\":"<<fallback<<",\"masked_cases\":"<<masked
      <<",\"empty_cases\":"<<empty<<",\"host_control_only\":true}\n";
}
