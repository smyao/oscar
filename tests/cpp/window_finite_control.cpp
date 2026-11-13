// Archive #126/#145/#150/#151/#154/#164/#172; startup D.4 reviewed in full.
// Actual LoadWindowRuns/Physical/LogicalPosition and finite helpers are injected.
// Bounded synchronous host stubs validate bits, errors, reads and structural
// fence placement. They do not simulate CANN, asynchronous pipes or NPU timing.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#define __aicore__
#define __ubuf__
#define ASCEND_IS_AIV (true)
using bfloat16_t=uint16_t;
constexpr int PIPE_V=1;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
float from_bits(uint32_t x){float y;std::memcpy(&y,&x,4);return y;}
int reduction_order=0,mask_count=0;
bool count_mode=false,natural_read_pending=false;
int certificate_calls=0,row_reductions=0;
float acc_value=0;
std::vector<std::string> trace;
struct Span {uintptr_t start;size_t bytes;};
std::vector<Span> spans;
void raw_bounds(const void* pointer,size_t bytes){
  uintptr_t p=reinterpret_cast<uintptr_t>(pointer);
  for(auto s:spans)if(p>=s.start && p<=s.start+s.bytes && bytes<=s.start+s.bytes-p)return;
  throw std::runtime_error("raw vector access exceeded UB");
}
float reduce(const float* p,int count){
  raw_bounds(p,size_t(count)*4);
  if(reduction_order==0){float x=0;for(int i=0;i<count;++i)x=x+p[i];return x;}
  std::vector<float> v(p,p+count);
  while(v.size()>1){size_t n=0;for(size_t i=0;i<v.size();i+=2)
    v[n++]=i+1<v.size()?v[i]+v[i+1]:v[i];v.resize(n);}
  return v.empty()?0:v[0];
}
namespace AscendC {
enum class HardEvent {V_S,MTE2_S,MTE2_V,V_MTE2,S_MTE2,S_V};
enum class MaskMode {COUNTER};
template<class T> struct LocalTensor {
  T* p;size_t capacity;std::string label;
  LocalTensor operator[](size_t offset)const{
    require(offset<capacity,"local slice overflow");return {p+offset,capacity-offset,label};}
  template<class U> LocalTensor<U> ReinterpretCast()const{
    require(capacity*sizeof(T)%sizeof(U)==0,"unaligned reinterpret size");
    return {reinterpret_cast<U*>(p),capacity*sizeof(T)/sizeof(U),label};}
  T* GetPhyAddr()const{return p;}
  T GetValue(size_t i)const{require(i<capacity,"scalar UB overflow");return p[i];}
};
template<int> void PipeBarrier(){natural_read_pending=false;}
void SetMaskCount(){count_mode=true;}
template<class T,MaskMode> void SetVectorMask(int n){
  static_assert(std::is_same_v<T,float>);require(n>0,"bad mask");mask_count=n;}
void SetMaskNorm(){count_mode=false;}
void ResetMask(){mask_count=0;}
void Abs(LocalTensor<float> dst,LocalTensor<float> src,int count){
  require(!natural_read_pending,"Abs overwrote unfinished BF16 Cast source");
  require(count>=0 && size_t(count)<=dst.capacity && size_t(count)<=src.capacity,"Abs bounds");
  ++certificate_calls;
  for(int i=0;i<count;++i)dst.p[i]=std::fabs(src.p[i]);
}
}
using AscendC::LocalTensor;
using AscendC::HardEvent;
using AscendC::PipeBarrier;
namespace oscar_ascend_device {
bool Finite(float x){return std::isfinite(x);}
template<HardEvent event> void Fence(){
  if constexpr(event!=HardEvent::V_S)trace.push_back("fence:"+std::to_string(int(event)));
}
}
using oscar_ascend_device::Finite;
using oscar_ascend_device::Fence;
void vcadd(float* dst,const float* src,int repeat,int d,int s,int stride,int mode){
  require(count_mode && repeat==1 && d==1 && s==1 && stride==8 && mode==1,"vcadd signature drift");
  raw_bounds(dst,4);++row_reductions;acc_value=reduce(src,mask_count);
}
int64_t get_acc_val(){uint32_t x;std::memcpy(&x,&acc_value,4);return x;}
void ReduceSum(LocalTensor<float> dst,LocalTensor<float> src,LocalTensor<float>,int count){
  require(size_t(count)<=src.capacity,"ReduceSum bounds");++row_reductions;dst.p[0]=reduce(src.p,count);
}

@@ROW_HELPER@@
@@CERTIFICATE_HELPER@@

struct Buffer {
  static constexpr uint64_t canary=0x5a5aa5a534126789ULL;
  size_t bytes;std::string name;std::vector<uint64_t> storage;
  Buffer(size_t n,std::string label):bytes(n),name(std::move(label)),storage(n/8+16,canary){require(n%8==0,"buffer alignment");}
  template<class T> LocalTensor<T> Get(size_t count=0){
    size_t cap=bytes/sizeof(T);require(count<=cap,"Get capacity");
    return {reinterpret_cast<T*>(storage.data()+8),count?count:cap,name};}
  void register_span(){spans.push_back({reinterpret_cast<uintptr_t>(storage.data()+8),bytes});}
  void check()const{for(size_t i=0;i<8;++i)
    require(storage[i]==canary && storage[8+bytes/8+i]==canary,"UB guard changed");}
};
template<class T> struct GlobalTensor {
  std::vector<T>* values;std::string name;size_t offset=0;
  GlobalTensor operator[](size_t n)const{require(n<values->size()-offset,"GM slice bounds");return {values,name,offset+n};}
};
struct DataCopyExtParams {uint16_t rows;uint32_t bytes,srcStride,dstStride;uint32_t reserved;};
template<class T> struct DataCopyPadExtParams {bool enabled;uint8_t left,right;T value;};
template<class T> void DataCopyPad(LocalTensor<T> dst,GlobalTensor<T> src,
                                  DataCopyExtParams copy,DataCopyPadExtParams<T>){
  require(copy.bytes%sizeof(T)==0,"DMA element size");
  trace.push_back("dma:"+src.name+":"+std::to_string(src.offset*sizeof(T))+":"+
    std::to_string(copy.rows)+":"+std::to_string(copy.bytes)+":"+std::to_string(copy.srcStride)+":"+dst.label);
  for(size_t r=0;r<copy.rows;++r){
    const size_t gm=src.offset*sizeof(T)+r*(copy.bytes+copy.srcStride);
    const size_t ub=r*(copy.bytes+copy.dstStride);
    require(gm+copy.bytes<=src.values->size()*sizeof(T) && ub+copy.bytes<=dst.capacity*sizeof(T),"DMA extent overflow");
    std::memcpy(reinterpret_cast<uint8_t*>(dst.p)+ub,reinterpret_cast<const uint8_t*>(src.values->data())+gm,copy.bytes);
  }
}
template<class T> void Duplicate(LocalTensor<T> dst,T value,int count){
  require(count>=0 && size_t(count)<=dst.capacity,"Duplicate bounds");
  std::fill(dst.p,dst.p+count,value);
}
enum class RoundMode {CAST_NONE};
void Cast(LocalTensor<float> dst,LocalTensor<bfloat16_t> src,RoundMode,int count){
  require(count>=0 && size_t(count)<=dst.capacity && size_t(count)<=src.capacity,"Cast extent");
  trace.push_back("cast:"+std::to_string(count));
  for(int i=0;i<count;++i)dst.p[i]=from_bits(uint32_t(src.p[i])<<16);
  natural_read_pending=true;
}
int64_t Min64(int64_t a,int64_t b){return std::min(a,b);}
int64_t Max64(int64_t a,int64_t b){return std::max(a,b);}
struct Kernel {
  static constexpr int D=256,kHalfKv=16,kHistoryHalfKv=32,kElements=D*kHalfKv;
  struct Geometry {int64_t sink=64,recent=256,speculative=3,hk=2,columns=8,blocks=2,
    blockTokens=512,windowStride=0,tagStride=323;} g;
  int64_t request=0,kind=1,context=768,qbegin=0,requestBegin=0,kvhead=0,kvend=319;
  int lane=0,error=0;
  int64_t physicalCacheRequest=-1,physicalCacheColumn=-1,physicalCacheValue=0;
  Buffer dequantBuf{kHistoryHalfKv*D*4,"dst"},naturalBuf{kHistoryHalfKv*(D+16)*2,"natural"},
    packedBuf{kHistoryHalfKv*160,"tags"},scratchBuf{kHistoryHalfKv*8*4,"check"},addressBuf{128,"address"};
  std::vector<int32_t> table;
  std::vector<int64_t> tags;
  std::vector<bfloat16_t> keys,values;
  GlobalTensor<int32_t> bt{&table,"table"};
  GlobalTensor<int64_t> wt{&tags,"tags"};
  GlobalTensor<bfloat16_t> wk{&keys,"K"},wv{&values,"V"};
  Kernel(){
    g.windowStride=g.tagStride*g.hk*D;
    table.resize(g.columns);for(int i=0;i<g.columns;++i)table[i]=i;
    tags.assign(g.blocks*g.tagStride,-1);
    keys.assign(g.blocks*g.windowStride,0x3f00);values.assign(keys.size(),0xbf00);
    for(int64_t c=0;c<kvend;++c){auto a=address(c);if(a.block>=0)tags[a.tag]=a.inpage;}
  }
  struct Address {int64_t block,inpage,tag,data;};
  Address address(int64_t c)const{
    int64_t sink=std::min(g.sink,context),tail=std::max(sink,context+1-g.recent);
    int64_t p=c<sink?c:tail+c-sink;
    if(p<0 || p/128>=g.columns || table[p/128]<0)return {-1,0,0,0};
    int64_t vb=table[p/128],ratio=g.blockTokens/128;
    if(vb>=g.blocks*ratio)return {-1,0,0,0};
    int64_t b=vb/ratio,ip=(vb%ratio)*128+p%128;
    int64_t wr=p<g.sink?p:g.sink+ip%(g.recent+g.speculative);
    return {b,ip,b*g.tagStride+wr,b*g.windowStride+(wr*g.hk+kvhead)*D};
  }
  void register_spans(){spans.clear();for(auto b:{&dequantBuf,&naturalBuf,&packedBuf,&scratchBuf,&addressBuf})b->register_span();}
  void guards()const{for(auto b:{&dequantBuf,&naturalBuf,&packedBuf,&scratchBuf,&addressBuf})b->check();}
@@LOGICAL_POSITION@@
@@PHYSICAL@@
@@OLD_METHOD@@
@@NEW_METHOD@@
};
struct Result {std::vector<uint32_t> words;std::vector<int> errors;std::vector<std::string> events;
  int certs,reductions;};
Result run(Kernel& k,int64_t start,bool candidate){
  k.register_spans();trace.clear();certificate_calls=row_reductions=0;natural_read_pending=false;
  Result out;
  for(bool value:{false,true}){
    // Exact caller LoadPrecise(kind1) clears this destination before reader.
    auto dst=k.dequantBuf.Get<float>();std::fill(dst.p,dst.p+Kernel::kElements,0.0F);
    if(candidate)k.New(start,value);else k.Old(start,value);
    for(int i=0;i<Kernel::kElements;++i){uint32_t w;std::memcpy(&w,dst.p+i,4);out.words.push_back(w);}
    out.errors.push_back(k.error);k.guards();
  }
  require(!count_mode && mask_count==0,"helper left nonnormal mask");
  out.events=trace;out.certs=certificate_calls;out.reductions=row_reductions;return out;
}
int cases=0,certified=0,fallback=0,tag_cases=0,physical_cases=0;
void check(int64_t start,int lane,int hk,int head,int limit,uint16_t tag_mask,
           int fault_row,uint16_t fault_bits,bool fault_value,int original_error,int bad_column){
  Kernel a,b;
  for(auto k:{&a,&b}){
    k->lane=lane;k->kvhead=head;k->error=original_error;
    // Keep allocation/strides as hk2; hk1 is a separate physical interleave
    // using the same large guarded backing allocation.
    k->g.hk=hk;k->g.windowStride=k->g.tagStride*hk*Kernel::D;k->kvend=limit;
    for(int r=0;r<16;++r){const int64_t c=start+lane*16+r;if(c>=limit)continue;
      auto addr=k->address(c);if(addr.block<0)continue;
      if(tag_mask&(uint16_t(1)<<r))k->tags[addr.tag]=-9;
      if(r==fault_row){auto& data=fault_value?k->values:k->keys;
        for(int col=0;col<256;++col)data[addr.data+col]=fault_bits;
        // Cancellation must retain the original reduction on extreme input.
        if(fault_bits==0x7f7f)for(int col=128;col<256;++col)data[addr.data+col]=0xff7f;
        if(fault_bits==0x7f7e)for(int col=0;col<256;++col)data[addr.data+col]=col%2?0xff7f:0x7f7f;
      }
    }
    if(bad_column>=0)k->table[size_t(bad_column)]=-1;
  }
  const auto table=a.table;const auto tags=a.tags;const auto keys=a.keys;const auto values=a.values;
  auto old=run(a,start,false),now=run(b,start,true);
  require(old.words==now.words,"window destination bits differ");
  require(old.errors==now.errors,"physical/tag/finite error precedence differs");
  if(start==64 && lane==0 && bad_column<0 && tag_mask==0x0101 &&
      fault_bits==0x7fc1 && (fault_row==7 || fault_row==15)){
    const int after_bad_finite=fault_row==7?4:2;
    require(now.errors[0]==(fault_value?4:after_bad_finite) &&
            now.errors[1]==(fault_value?after_bad_finite:4),"independent last-error counterexample failed");
  }
  if(fault_bits==0x7f7e && tag_mask==0 && bad_column<0)
    require(now.errors[0]==original_error && now.errors[1]==original_error,
            "finite extreme cancellation was incorrectly rejected by certificate");
  require(old.events==now.events,"GM reads/Casts/non-V_S fences differ");
  require(a.table==table && b.table==table && a.tags==tags && b.tags==tags &&
          a.keys==keys && b.keys==keys && a.values==values && b.values==values,"input mutation");
  if(now.certs){++certified;if(now.reductions>now.certs)++fallback;}
  if(tag_mask)++tag_cases;if(bad_column>=0)++physical_cases;++cases;
}
int main(int argc,char** argv){try{
  const int first=argc==2?std::stoi(argv[1]):0,last=argc==2?first+1:2;
  require(first>=0 && last<=2,"unknown reduction order");
  for(reduction_order=first;reduction_order<last;++reduction_order){
    // Boundary runs: sink/tail jump, virtual128, ring259, final/empty rows.
    for(int start:{0,48,56,64,80,176,256,288,304,320})
      for(int lane:{0,1})for(int hk:{1,2})for(int error:{0,1,2,3,4,7})
        check(start,lane,hk,hk-1,319,0,-1,0,false,error,-1);
    for(int n=0;n<=16;++n)for(int lane:{0,1})
      check(0,lane,2,1,lane*16+n,0,-1,0,false,4,-1);
    // Every row can be invalid, nonfinite, extreme, tiny or signed zero;
    // invalid row data must never be read, and K/V order must stay exact.
    for(uint16_t mask:{uint16_t(0),uint16_t(0xaaaa),uint16_t(0xffff),uint16_t(0x0101)})
      for(int row=0;row<16;++row)for(bool value:{false,true})
        for(uint16_t bits:{uint16_t(0x7fc1),uint16_t(0x7f81),uint16_t(0x7f80),uint16_t(0xff80),
                          uint16_t(0x7f7f),uint16_t(0x8000),uint16_t(1)})
          check(64,0,2,1,319,mask,row,bits,value,7,-1);
    for(int start:{0,56,64,176,288})for(int column:{0,4,5,6})for(int row:{0,7,15})
      check(start,0,2,0,319,0x8001,row,0x7fc1,false,4,column);
    for(int row:{0,7,8,15})for(bool value:{false,true})
      check(64,0,2,1,319,0,row,0x7f7e,value,7,-1);
  }
  require(certified>0 && fallback>0 && tag_cases>0 && physical_cases>0,"coverage missing");
  std::cout<<"{\"host_control_only\":true,\"cases\":"<<cases
    <<",\"reduction_orders\":"<<(last-first)<<",\"certificate_cases\":"<<certified
    <<",\"fallback_cases\":"<<fallback<<",\"tag_cases\":"<<tag_cases
    <<",\"physical_cases\":"<<physical_cases
    <<",\"value_bits_and_errors\":\"identical\",\"DMA_Cast_fences\":\"identical_except_finite_V_S\","
    <<"\"guards\":\"passed\",\"CANN\":\"not_run\",\"NPU\":\"not_run\",\"performance\":\"not_run\"}\n";
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
