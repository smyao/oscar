// Archive #126/#145/#150/#151/#154/#164/#172 and full startup D.4 reviewed.
// Actual LoadPrecise/PublishKv and finite helper bodies execute with bounded
// recording stubs. This validates dead-write elimination, not CANN, DMA
// concurrency, device FP conversion semantics or hardware performance.
#include <algorithm>
#include <array>
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
constexpr int PIPE_V=1;
using bfloat16_t=uint16_t;
void require(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
int64_t Min64(int64_t a,int64_t b){return std::min(a,b);}
int64_t Max64(int64_t a,int64_t b){return std::max(a,b);}
float from_bits(uint32_t b){float x;std::memcpy(&x,&b,4);return x;}
uint32_t to_bits(float x){uint32_t b;std::memcpy(&b,&x,4);return b;}
using Event=std::array<int64_t,8>;
struct Counters {
  int64_t fp32ZeroBytes=0,bf16ZeroBytes=0;
  std::vector<Event> events;
};
thread_local Counters* counters=nullptr;
thread_local int64_t vectorCount=0;
thread_local float accumulator=0;
void record(int64_t code,int64_t a=0,int64_t b=0,int64_t c=0,int64_t d=0,
            int64_t e=0,int64_t f=0,int64_t g=0) {
  counters->events.push_back({code,a,b,c,d,e,f,g});
}
namespace AscendC {
enum class HardEvent {V_S,V_MTE2,MTE2_V,S_V,MTE2_S,V_MTE3,MTE3_V};
enum class MaskMode {COUNTER};
template<HardEvent E> void Fence(){record(1,int(E));}
template<int P> void PipeBarrier(){record(2,P);}
void SetMaskCount(){}
void SetMaskNorm(){}
void ResetMask(){vectorCount=0;}
template<class T,MaskMode> void SetVectorMask(int64_t n){vectorCount=n;}
template<class T> struct LocalTensor {
  unsigned char* ptr;size_t bytes;int id;size_t offset;
  LocalTensor operator[](int64_t n)const {
    require(n>=0 && size_t(n)*sizeof(T)<=bytes,"tensor offset outside buffer");
    return {ptr+n*sizeof(T),bytes-n*sizeof(T),id,offset+n*sizeof(T)};
  }
  T GetValue(int64_t n)const {
    require(n>=0 && (size_t(n)+1)*sizeof(T)<=bytes,"tensor read outside buffer");
    T value;std::memcpy(&value,ptr+n*sizeof(T),sizeof(T));return value;
  }
  void SetValue(int64_t n,T value)const {
    require(n>=0 && (size_t(n)+1)*sizeof(T)<=bytes,"tensor write outside buffer");
    std::memcpy(ptr+n*sizeof(T),&value,sizeof(T));
  }
  void* GetPhyAddr()const{return ptr;}
  template<class U> LocalTensor<U> ReinterpretCast()const{return {ptr,bytes,id,offset};}
};
void Abs(LocalTensor<float> dst,LocalTensor<float> src,int64_t n) {
  record(3,dst.id,dst.offset,src.id,src.offset,n);
  for(int64_t i=0;i<n;++i)dst.SetValue(i,std::abs(src.GetValue(i)));
}
}
using namespace AscendC;
namespace oscar_ascend_device {
using AscendC::Fence;
bool Finite(float x){return std::isfinite(x);}
}
using oscar_ascend_device::Finite;
void vcadd(float*,float* source,int,int,int,int,int) {
  record(4,vectorCount);float sum=0;
  for(int64_t i=0;i<vectorCount;++i)sum+=source[i];
  accumulator=sum;
}
int64_t get_acc_val(){return int64_t(to_bits(accumulator));}
@@ROW_HELPER@@
@@CERTIFICATE_HELPER@@
struct Buffer {
  int id;size_t size;std::vector<uint64_t> data;
  Buffer(int identity,size_t bytes):id(identity),size(bytes),data((bytes+7)/8+2,0xdededeabcd123456ULL){}
  template<class T> LocalTensor<T> Get(int64_t n=-1) {
    if(n<0)n=size/sizeof(T);
    require(size_t(n)*sizeof(T)<=size,"buffer request exceeds capacity");
    return {reinterpret_cast<unsigned char*>(data.data()+1),size_t(n)*sizeof(T),id,0};
  }
  void check()const {require(data.front()==0xdededeabcd123456ULL &&
      data.back()==0xdededeabcd123456ULL,"buffer canary overwritten");}
};
template<class T> void Duplicate(LocalTensor<T> dst,T x,int64_t n) {
  if(sizeof(T)==4)counters->fp32ZeroBytes+=n*sizeof(T);
  else counters->bf16ZeroBytes+=n*sizeof(T);
  for(int64_t i=0;i<n;++i)dst.SetValue(i,x);
}
struct DataCopyExtParams {uint16_t count;uint32_t len,srcStride,dstStride,reserved;};
template<class T> struct DataCopyPadExtParams {bool pad;uint8_t left,right;T value;};
template<class T> void DataCopyPad(LocalTensor<T> dst,LocalTensor<T> src,
    DataCopyExtParams copy,DataCopyPadExtParams<T>) {
  record(5,src.id,src.offset,copy.count,copy.len,copy.srcStride,dst.id,dst.offset);
  for(int row=0;row<copy.count;++row) {
    const size_t in=row*(copy.len+copy.srcStride)/sizeof(T);
    const size_t out=row*(copy.len+copy.dstStride)/sizeof(T);
    for(size_t i=0;i<copy.len/sizeof(T);++i)dst.SetValue(out+i,src.GetValue(in+i));
  }
}
template<class T> void DataCopy(LocalTensor<T> dst,LocalTensor<T> src,int64_t n) {
  record(6,src.id,src.offset,n*sizeof(T),dst.id,dst.offset);
  for(int64_t i=0;i<n;++i)dst.SetValue(i,src.GetValue(i));
}
enum class RoundMode {CAST_NONE};
void Cast(LocalTensor<float> dst,LocalTensor<bfloat16_t> src,RoundMode,int64_t n) {
  record(7,src.id,src.offset,dst.id,dst.offset,n);
  for(int64_t i=0;i<n;++i)dst.SetValue(i,from_bits(uint32_t(src.GetValue(i))<<16));
}
void ReduceSum(LocalTensor<float> dst,LocalTensor<float> src,LocalTensor<float>,int n) {
  record(8,src.id,src.offset,n);float sum=0;
  for(int i=0;i<n;++i)sum+=src.GetValue(i);
  dst.SetValue(0,sum);
}
namespace oscar_bf16 {
void CastRne(LocalTensor<bfloat16_t> dst,LocalTensor<float> src,int64_t n) {
  record(9,src.id,src.offset,dst.id,dst.offset,n);
  for(int64_t i=0;i<n;++i) {
    uint32_t b=to_bits(src.GetValue(i));b+=0x7fff+((b>>16)&1);
    dst.SetValue(i,b>>16);
  }
}
}
struct Geometry {
  int64_t tokens=64,hk=1,sink=64,recent=256,speculative=3;
  int64_t tagStride=323,windowStride=323*2*256;
};
struct Kernel {
  static constexpr int D=256,kHalfKv=16,kElements=16*256,kHistoryHalfKv=32;
  Geometry g;
  int64_t kind=2,kvend=16,requestBegin=16,context=32,kvhead=0;
  int64_t kOffset=0,vOffset=256*256;
  int32_t lane=0,error=0;
  Buffer dequantBuf{1,32*256*4},qBf16Buf{2,256*2},naturalBuf{3,32*272*2};
  Buffer scratchBuf{4,32*8*4},addressBuf{5,128};
  Buffer keys{6,64*2*256*2},values{7,64*2*256*2};
  Buffer windowsK{8,323*2*256*2},windowsV{9,323*2*256*2},tags{10,323*8};
  Buffer stage{11,2*2*256*256*2};
  LocalTensor<bfloat16_t> ck=keys.Get<bfloat16_t>(),cv=values.Get<bfloat16_t>();
  LocalTensor<bfloat16_t> wk=windowsK.Get<bfloat16_t>(),wv=windowsV.Get<bfloat16_t>();
  LocalTensor<int64_t> wt=tags.Get<int64_t>();
  LocalTensor<bfloat16_t> operands=stage.Get<bfloat16_t>();
  Counters count;
  int64_t LogicalPosition(int64_t p){return p;}
  bool Physical(int64_t p,int64_t& block,int64_t& inpage) {
    record(10,p);if(p<0 || p>=64){error=1;return false;}
    block=0;inpage=p;return true;
  }
  void LoadWindowRuns(int64_t start,bool val) {
    record(11,start,val);
    // This unchanged delegate must still receive the original zero tile.
    auto dst=dequantBuf.Get<float>();
    for(int i=0;i<kElements;++i)require(to_bits(dst.GetValue(i))==0,"window initial zero changed");
  }
  @@OLD_LOAD@@
  @@NEW_LOAD@@
  @@PUBLISH@@
  void checks()const {
    dequantBuf.check();qBf16Buf.check();naturalBuf.check();scratchBuf.check();addressBuf.check();
    keys.check();values.check();windowsK.check();windowsV.check();tags.check();stage.check();
  }
};
void equal(Buffer& a,Buffer& b,const char* reason) {
  require(a.size==b.size && std::memcmp(a.data.data()+1,b.data.data()+1,a.size)==0,reason);
}
int main() {
  int cases=0,full=0,partial=0,invalid=0,empty=0,window=0;
  // Fixed allocations are reused while metadata, live rows and source bits
  // change. No specialization may assume a previous call's eligibility.
  Kernel old,newer;
  for(int kind:{0,1,2})for(int rows=0;rows<=16;++rows)
  for(int lane:{0,1})for(int hk:{1,2})for(bool val:{false,true})
  for(int location:{0,1,2})for(int poison=0;poison<7;++poison) {
    const int64_t start=32;
    const int64_t request=location==0?16:location==1?-100:60;
    for(Kernel* k:{&old,&newer}) {
      k->count=Counters{};
      k->kind=kind;k->lane=lane;k->g.hk=hk;k->kvhead=hk-1;k->requestBegin=request;
      k->kvend=start+lane*16+rows;k->error=4;
      for(Buffer* source:{&k->keys,&k->values,&k->windowsK,&k->windowsV}) {
        if((kind==0)!=(source==&k->windowsK || source==&k->windowsV))continue;
        auto bf=source->Get<bfloat16_t>();
        for(int64_t i=0;i<int64_t(bf.bytes/2);++i)bf.SetValue(i,uint16_t(0x3f00+(i%63)));
        // Distinct sources make any key/value or head/stride mistake visible.
        bf.SetValue(0,source->id%2?0x4000:0x3f80);
        const uint16_t special[]={0x3f91,0x7fc1,0x7f80,0xff80,0x8000,0x0001,0x0000};
        for(int64_t i=31;i<int64_t(bf.bytes/2);i+=257)bf.SetValue(i,special[poison]);
      }
      for(int i=0;i<323;++i)k->wt.SetValue(i,i);
      // Poison old UB on every call. Valid DMA/Cast must overwrite it;
      // tail, no-row and invalid-token outputs must still become exact +0.
      auto dst=k->dequantBuf.Get<float>();
      for(int64_t i=0;i<int64_t(dst.bytes/4);++i)dst.SetValue(i,from_bits(0x7fc01234));
    }
    const auto originalKeys=old.keys.data,originalValues=old.values.data;
    counters=&old.count;old.LoadOld(start,val);old.PublishKv(val,0);old.checks();
    counters=&newer.count;newer.LoadNew(start,val);newer.PublishKv(val,0);newer.checks();
    require(old.error==newer.error,"LoadPrecise error changed");
    require(old.count.events==newer.count.events,"DMA/Cast/fence/helper/source-binding sequence changed");
    equal(old.dequantBuf,newer.dequantBuf,"FP32 load result changed");
    equal(old.stage,newer.stage,"published BF16 staging changed");
    require(old.keys.data==originalKeys && newer.keys.data==originalKeys &&
        old.values.data==originalValues && newer.values.data==originalValues,"input modified");
    const int64_t token=request+lane*16;
    const bool overwrites=kind==2 && rows>0 && token>=0 && token+rows<=64;
    require(old.count.fp32ZeroBytes-newer.count.fp32ZeroBytes==(overwrites?16384:0),
        "wrong FP32 dead-write count");
    require(old.count.bf16ZeroBytes-newer.count.bf16ZeroBytes==
        (overwrites && rows==16?8192:0),"wrong BF16 dead-write count");
    if(kind==2) {
      if(rows==0)++empty;else if(!overwrites)++invalid;else if(rows==16)++full;else ++partial;
    } else if(kind==1)++window;
    ++cases;
  }
  require(full && partial && invalid && empty && window,"missing semantic coverage");
  std::cout<<"{\"cases\":"<<cases<<",\"full\":"<<full<<",\"partial\":"<<partial
    <<",\"invalid\":"<<invalid<<",\"empty\":"<<empty<<",\"window\":"<<window
    <<",\"host_control_only\":true,\"fixed_address_metadata_refresh\":true"
    <<",\"full_kv256_zero_bytes_removed\":786432}\n";
}
