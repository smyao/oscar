// Archive #145/#150/#151/#154/#164/#172 and startup D.4: host-only proof.
// Python injects the real old/new current finite block and both real helpers.
// Primitive stubs check bounds and FP32 predicate/control semantics under two
// reduction orders. They do NOT simulate CANN, DMA, NPU pipes or device timing.
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
void require(bool ok,const char* reason) {if(!ok)throw std::runtime_error(reason);}
float bits(uint32_t value) {float result;std::memcpy(&result,&value,4);return result;}
bool bit_equal(const std::vector<float>& a,const std::vector<float>& b) {
  return a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*4)==0;
}
struct Span {float* begin;size_t size;};
std::vector<Span> spans;
void raw_bounds(const float* p,size_t count) {
  const uintptr_t address=reinterpret_cast<uintptr_t>(p);
  for(const auto& span:spans) {
    const uintptr_t begin=reinterpret_cast<uintptr_t>(span.begin);
    if(address>=begin && address<=begin+span.size*4 &&
        count*4<=begin+span.size*4-address)return;
  }
  throw std::runtime_error("raw helper access outside bounded storage");
}
int reduce_order=0,mask_count=0;
bool count_mode=false;
int explicit_v_s=0,abs_calls=0,reduce_calls=0;
float accumulator=0;
float reduce(const float* source,int count) {
  raw_bounds(source,count);
  if(reduce_order==0) {
    float total=0;
    for(int i=0;i<count;++i)total=total+source[i];
    return total;
  }
  std::vector<float> tree(source,source+count);
  while(tree.size()>1) {
    size_t next=0;
    for(size_t i=0;i<tree.size();i+=2)
      tree[next++]=i+1<tree.size()?tree[i]+tree[i+1]:tree[i];
    tree.resize(next);
  }
  return tree.empty()?0:tree[0];
}
namespace AscendC {
enum class HardEvent {V_S};
enum class MaskMode {COUNTER};
template<typename T> struct LocalTensor {
  T* data;size_t capacity;
  LocalTensor operator[](size_t offset) const {
    require(offset<capacity,"tensor offset out of bounds");
    return {data+offset,capacity-offset};
  }
  T* GetPhyAddr() const {require(capacity>0,"empty physical address");return data;}
  T GetValue(size_t index) const {require(index<capacity,"scalar read out of bounds");return data[index];}
};
template<int> void PipeBarrier() {}
inline void SetMaskCount() {count_mode=true;}
template<typename T,MaskMode> void SetVectorMask(int count) {
  static_assert(std::is_same_v<T,float>);
  require(count>0,"invalid counter mask");mask_count=count;
}
inline void SetMaskNorm() {count_mode=false;}
inline void ResetMask() {mask_count=0;}
inline void Abs(LocalTensor<float> dst,LocalTensor<float> source,int count) {
  require(count>=0 && size_t(count)<=dst.capacity && size_t(count)<=source.capacity,
      "Abs exceeded live tensor");
  ++abs_calls;
  for(int i=0;i<count;++i)dst.data[i]=std::fabs(source.data[i]);
}
}
namespace oscar_ascend_device {
inline bool Finite(float value) {return std::isfinite(value);}
template<AscendC::HardEvent> void Fence() {++explicit_v_s;}
}
void vcadd(float* dst,const float* source,int repeat,int dst_stride,int src_stride,
           int repeat_stride,int acc_mode) {
  require(count_mode && repeat==1 && dst_stride==1 && src_stride==1 &&
      repeat_stride==8 && acc_mode==1,"actual helper reduction signature changed");
  raw_bounds(dst,1);++reduce_calls;accumulator=reduce(source,mask_count);
}
int64_t get_acc_val() {
  uint32_t word;std::memcpy(&word,&accumulator,4);return word;
}
using AscendC::HardEvent;
using oscar_ascend_device::Fence;
using oscar_ascend_device::Finite;
inline void ReduceSum(AscendC::LocalTensor<float> dst,AscendC::LocalTensor<float> source,
                      AscendC::LocalTensor<float>,int count) {
  require(size_t(count)<=source.capacity,"original ReduceSum exceeded input");
  ++reduce_calls;dst.data[0]=reduce(source.data,count);
}

@@ROW_HELPER@@
@@CERTIFICATE_HELPER@@

struct Buffer {
  static constexpr int guard=8;
  static constexpr float canary=123.25F;
  std::vector<float> data;size_t capacity;
  explicit Buffer(size_t count):data(count+2*guard,canary),capacity(count) {}
  template<typename T> AscendC::LocalTensor<T> Get(size_t count=0) {
    static_assert(std::is_same_v<T,float>);
    require(count<=capacity,"requested tensor exceeds real UB buffer");
    return {data.data()+guard,count?count:capacity};
  }
  void register_span() {spans.push_back({data.data()+guard,capacity});}
  void check() const {
    for(int i=0;i<guard;++i)
      require(data[i]==canary && data[guard+capacity+i]==canary,"UB canary changed");
  }
};
struct Kernel {
  static constexpr int D=256,kHalfKv=16,kElements=kHalfKv*D,kHistoryHalfKv=32;
  Buffer naturalBuf{kHistoryHalfKv*(D+16)*2/4};
  Buffer scratchBuf{kHistoryHalfKv*8};
  Buffer dequantBuf{kHistoryHalfKv*D};
  int error=0;
  void Baseline(int rows) {
    if(rows==0)return; // Real LoadPrecise returns before its finite block.
    auto dst=dequantBuf.Get<float>();
@@BASELINE_BLOCK@@
  }
  void Candidate(int rows) {
    if(rows==0)return;
    auto dst=dequantBuf.Get<float>();
@@CANDIDATE_BLOCK@@
  }
  void register_spans() {
    spans.clear();naturalBuf.register_span();scratchBuf.register_span();dequantBuf.register_span();
  }
  void check() const {naturalBuf.check();scratchBuf.check();dequantBuf.check();}
};
constexpr int max_count=Kernel::kHistoryHalfKv*Kernel::D;
std::vector<float> base() {
  std::vector<float> values(max_count);
  for(int i=0;i<max_count;++i)values[i]=((i*31)%511-255)*0.03125F;
  return values;
}
int cases=0,certified_cases=0,fallback_cases=0;
void check(const std::vector<float>& values,int rows,int original_error,int expected,
           const std::string& name) {
  Kernel old,candidate;
  std::copy(values.begin(),values.end(),old.dequantBuf.Get<float>().data);
  std::copy(values.begin(),values.end(),candidate.dequantBuf.Get<float>().data);
  old.error=candidate.error=original_error;
  const auto before=old.dequantBuf.data;
  old.register_spans();old.Baseline(rows);old.check();
  explicit_v_s=abs_calls=reduce_calls=0;count_mode=false;mask_count=0;
  candidate.register_spans();candidate.Candidate(rows);candidate.check();
  require(candidate.error==old.error,(name+": error predicate differs").c_str());
  if(expected>=0)require(candidate.error==expected,(name+": wrong expected error").c_str());
  require(bit_equal(before,old.dequantBuf.data) && bit_equal(before,candidate.dequantBuf.data),
      "input or padding bits changed");
  require(!count_mode && mask_count==0,"normal vector mask was not restored");
  const int written=rows>=8?rows*Kernel::D:0;
  for(size_t i=written;i<candidate.naturalBuf.capacity;++i)
    require(candidate.naturalBuf.Get<float>().GetValue(i)==Buffer::canary,
        "certificate wrote outside active scratch rows");
  require(abs_calls==(rows>=8?1:0),"certificate row eligibility drift");
  if(rows>=8) {
    if(reduce_calls==1) {++certified_cases;require(explicit_v_s==1,"certified block has extra scalar waits");}
    else {++fallback_cases;require(reduce_calls==rows+1,"uncertified block lost original rows");}
  } else require(reduce_calls==rows,"small block lost original row reduction");
  ++cases;
}
int main() {
  try {
    for(reduce_order=0;reduce_order<2;++reduce_order) {
      for(int rows=0;rows<=16;++rows)for(int error:{0,1,2,3,4,7}) {
        auto values=base();values[rows*256]=bits(0x7fc00123);
        check(values,rows,error,error,"inactive poison and original error");
      }
      for(int rows:{1,7,8,9,15,16})for(int row=0;row<rows;++row)
        for(int repeat=0;repeat<4;++repeat)for(int edge:{0,63})
          for(uint32_t bad:{0x7fc00123U,0x7f800000U,0xff800000U,0x7f800001U}) {
            auto values=base();values[row*256+repeat*64+edge]=bits(bad);
            check(values,rows,4,2,"live nonfinite input");
          }
      const float boundary=std::ldexp(1.0F,110);
      for(int rows:{7,8,9,16})for(float scale:{std::nextafter(boundary,0.0F),boundary,
          std::nextafter(boundary,std::numeric_limits<float>::infinity())}) {
        for(int offset:{0,255,rows*128,rows*256-1})for(float sign:{1.0F,-1.0F}) {
          std::vector<float> values(max_count,0);values[offset]=sign*scale;
          check(values,rows,3,3,"sufficient bound edge");
        }
        std::vector<float> values(max_count,scale/(rows*256));
        check(values,rows,0,0,"distributed bound edge");
      }
      for(int rows:{1,7,8,16})for(int row=0;row<rows;++row) {
        auto values=base();
        for(int column=0;column<256;++column)values[row*256+column]=std::numeric_limits<float>::max();
        check(values,rows,4,2,"finite row overflow");
        for(int period:{2,4,8,16,128,256}) {
          for(int column=0;column<256;++column)values[row*256+column]=
              (column%period<period/2?1.0F:-1.0F)*std::numeric_limits<float>::max();
          check(values,rows,4,-1,"extreme cancellation follows original reduction");
        }
        for(uint32_t tiny:{0U,1U,0x10000U}) {
          for(int column=0;column<256;++column)values[row*256+column]=bits(tiny|(column%2?0x80000000:0));
          check(values,rows,1,1,"signed zero and subnormal bits");
        }
      }
    }
    std::cout<<"{\"status\":\"passed\",\"host_control_only\":true,\"reduction_orders\":2,\"cases\":"
      <<cases<<",\"certified_cases\":"<<certified_cases<<",\"fallback_cases\":"<<fallback_cases
      <<",\"input_bits_unchanged\":true,\"guards\":\"passed\",\"CANN\":\"not_run\",\"NPU\":\"not_run\",\"performance\":\"not_run\"}\n";
    return 0;
  }catch(const std::exception& error) {
    std::cerr<<"PRECISE_FINITE_CONTROL_FAILED cases="<<cases<<" "<<error.what()<<"\n";return 1;
  }
}
