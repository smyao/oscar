// Archive #126/#145/#154/#155/#164/#172; startup D.4 reviewed in full.
// Host execution of the actual old/new bounded helper sources, not an SDK
// simulator, CANN CPU-debug, device timing, or NPU qualification. D.4's
// 6499.8--6655.1ms full-history materialization is not this tile mechanism.
// Compare exact BF16 bits, errors, read bounds and explicit scalar handoffs;
// source/index/destination Gather overlap is rejected rather than guessed.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#define __aicore__
constexpr int PIPE_V=1;
struct bfloat16_t {uint16_t bits;};
void require(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
float from_bits(uint32_t bits) {float f;std::memcpy(&f,&bits,4);return f;}
uint32_t to_bits(float f) {uint32_t b;std::memcpy(&b,&f,4);return b;}
uint16_t vector_rne(float f) {
  uint32_t b=to_bits(f);
  if((b&0x7f800000)==0x7f800000 && (b&0x007fffff))return uint16_t((b>>16)|0x0040);
  return uint16_t((b+0x7fff+((b>>16)&1))>>16);
}
// Independent tie decision, not the vector stub's add-and-shift expression.
uint16_t scalar_rne(float f) {
  const uint32_t b=to_bits(f);uint16_t hi=b>>16;
  if((b&0x7f800000)==0x7f800000 && (b&0x007fffff))return hi|0x0040;
  const uint16_t lo=uint16_t(b);
  if(lo>0x8000 || (lo==0x8000 && (hi&1)))++hi;
  return hi;
}
float scalar_bf16(float f) {return from_bits(uint32_t(scalar_rne(f))<<16);}
struct Buffer {
  size_t size;std::vector<uint64_t> words;
  explicit Buffer(size_t bytes):size(bytes),words((bytes+7)/8+2,UINT64_C(0xa5dedeab12ad34ef)){}
  unsigned char* ptr() {return reinterpret_cast<unsigned char*>(words.data()+1);}
  void check()const {require(words.front()==UINT64_C(0xa5dedeab12ad34ef) &&
      words.back()==UINT64_C(0xa5dedeab12ad34ef),"UB canary changed");}
};
struct Range {const Buffer* owner;size_t offset,bytes;};
bool overlaps(Range a,Range b) {
  return a.owner==b.owner && a.offset<b.offset+b.bytes && b.offset<a.offset+a.bytes;
}
std::vector<Range> vector_writes;
size_t gather_calls=0,gather_largest=0,gather_index_max=0,scalar_reads=0;
size_t raw_gather_calls=0,raw_gather_index_max=0;

namespace AscendC {
enum class TPosition {VECCALC};
enum class RoundMode {CAST_NONE,CAST_RINT,CAST_TRUNC};
enum class HardEvent {V_S,S_V,V_MTE2,MTE2_V,V_MTE3,MTE3_V,MTE3_MTE2};
template<HardEvent E> void Fence() {if(E==HardEvent::V_S)vector_writes.clear();}
template<int> void PipeBarrier() {}
void SetMaskNorm() {}
void ResetMask() {}
struct UnaryRepeatParams {uint16_t dstBlkStride,srcBlkStride;uint8_t dstRepStride,srcRepStride;};
struct BinaryRepeatParams {
  uint8_t dstBlkStride,src0BlkStride,src1BlkStride,dstRepStride,src0RepStride,src1RepStride;
};
struct BrcbRepeatParams {uint8_t dstBlkStride,dstRepStride;};
template<class T> struct LocalTensor {
  std::shared_ptr<Buffer> owner;size_t offset=0;
  void bounds(size_t i)const {require(offset+(i+1)*sizeof(T)<=owner->size,"UB access outside allocation");}
  T read(size_t i)const {bounds(i);T v;std::memcpy(&v,owner->ptr()+offset+i*sizeof(T),sizeof(T));return v;}
  void write(size_t i,T v)const {bounds(i);std::memcpy(owner->ptr()+offset+i*sizeof(T),&v,sizeof(T));}
  T GetValue(size_t i)const {
    for(const auto& r:vector_writes)
      require(!overlaps(r,{owner.get(),offset+i*sizeof(T),sizeof(T)}),"scalar read before V_S handoff");
    ++scalar_reads;return read(i);
  }
  void SetValue(size_t i,T v)const {write(i,v);}
  LocalTensor operator[](size_t i)const {
    require(offset+i*sizeof(T)<=owner->size,"UB view outside allocation");return {owner,offset+i*sizeof(T)};
  }
  template<class U> LocalTensor<U> ReinterpretCast()const {return {owner,offset};}
};
template<TPosition> struct TBuf {
  std::shared_ptr<Buffer> owner;
  explicit TBuf(size_t bytes):owner(std::make_shared<Buffer>(bytes)){}
  template<class T> LocalTensor<T> Get() {return {owner};}
};
template<class T> void mark(LocalTensor<T> x,size_t last) {
  x.bounds(last);vector_writes.push_back({x.owner.get(),x.offset,(last+1)*sizeof(T)});
}
template<class T> size_t index(int rep,int elem,int block_stride,int repeat_stride) {
  constexpr int per=32/sizeof(T);
  return rep*repeat_stride*per+(elem/per)*block_stride*per+elem%per;
}
template<class T> void Duplicate(LocalTensor<T> d,T value,int n) {
  for(int i=0;i<n;++i)d.write(i,value);if(n)mark(d,n-1);
}
template<class T> void ShiftRight(LocalTensor<T> d,LocalTensor<T> s,T shift,
    uint64_t mask,uint8_t repeats,UnaryRepeatParams p) {
  size_t last=0;
  for(int r=0;r<repeats;++r)for(int e=0;e<int(mask);++e) {
    const size_t di=index<T>(r,e,p.dstBlkStride,p.dstRepStride);
    d.write(di,static_cast<T>(s.read(index<T>(r,e,p.srcBlkStride,p.srcRepStride))>>shift));last=std::max(last,di);
  }
  if(repeats && mask)mark(d,last);
}
template<class T> void And(LocalTensor<T> d,LocalTensor<T> a,LocalTensor<T> b,
    uint64_t mask,uint8_t repeats,BinaryRepeatParams p) {
  size_t last=0;
  for(int r=0;r<repeats;++r)for(int e=0;e<int(mask);++e) {
    const size_t di=index<T>(r,e,p.dstBlkStride,p.dstRepStride);
    d.write(di,static_cast<T>(a.read(index<T>(r,e,p.src0BlkStride,p.src0RepStride))&
                            b.read(index<T>(r,e,p.src1BlkStride,p.src1RepStride))));last=std::max(last,di);
  }
  if(repeats && mask)mark(d,last);
}
template<class T> float number(T x) {
  if constexpr(std::is_same_v<T,bfloat16_t>)return from_bits(uint32_t(x.bits)<<16);
  else return static_cast<float>(x);
}
template<class T,class U> T converted(U value,RoundMode mode) {
  if constexpr(std::is_same_v<T,bfloat16_t>) {require(mode==RoundMode::CAST_RINT,"BF16 rounding mode");return {vector_rne(number(value))};}
  else if constexpr(std::is_integral_v<T>) {
    float f=number(value);require(std::isfinite(f),"nonfinite integer conversion");
    if(mode==RoundMode::CAST_RINT)f=std::nearbyint(f);
    require(double(f)>=double(std::numeric_limits<T>::min()) && double(f)<=double(std::numeric_limits<T>::max()),"integer Cast overflow");
    return static_cast<T>(f);
  } else return static_cast<T>(number(value));
}
template<class T,class U> void Cast(LocalTensor<T> d,LocalTensor<U> s,RoundMode mode,
    uint64_t mask,uint8_t repeats,UnaryRepeatParams p) {
  std::vector<std::pair<size_t,T>> results;size_t last=0;
  for(int r=0;r<repeats;++r)for(int e=0;e<int(mask);++e) {
    size_t di=index<T>(r,e,p.dstBlkStride,p.dstRepStride);
    results.push_back({di,converted<T>(s.read(index<U>(r,e,p.srcBlkStride,p.srcRepStride)),mode)});
    last=std::max(last,di);
  }
  for(const auto& x:results)d.write(x.first,x.second);
  if(!results.empty())mark(d,last);
}
template<class T,class U> void Cast(LocalTensor<T> d,LocalTensor<U> s,RoundMode mode,int n) {
  std::vector<T> results;for(int i=0;i<n;++i)results.push_back(converted<T>(s.read(i),mode));
  for(int i=0;i<n;++i)d.write(i,results[i]);if(n)mark(d,n-1);
}
template<class T> void Gather(LocalTensor<T> d,LocalTensor<T> s,LocalTensor<uint32_t> indices,
    uint32_t byte_base,uint32_t n) {
  require(d.offset%32==0 && s.offset%32==0 && indices.offset%32==0,"Gather base must be 32B aligned");
  require(!overlaps({d.owner.get(),d.offset,n*sizeof(T)},{indices.owner.get(),indices.offset,n*4}),"Gather destination aliases offsets");
  require(!overlaps({d.owner.get(),d.offset,n*sizeof(T)},{s.owner.get(),s.offset,s.owner->size-s.offset}),"Gather destination aliases source");
  ++gather_calls;gather_largest=std::max(gather_largest,size_t(n));
  if constexpr(std::is_same_v<T,uint16_t>)++raw_gather_calls;
  for(uint32_t i=0;i<n;++i) {
    size_t byte=byte_base+indices.read(i);
    require(byte%sizeof(T)==0,"Gather byte offset misaligned");
    gather_index_max=std::max(gather_index_max,byte);
    if constexpr(std::is_same_v<T,uint16_t>)raw_gather_index_max=std::max(raw_gather_index_max,byte);
    d.write(i,s.read(byte/sizeof(T)));
  }
  if(n)mark(d,n-1);
}
void Brcb(LocalTensor<float> d,LocalTensor<float> s,uint8_t repeats,BrcbRepeatParams p) {
  size_t last=0;
  for(int r=0;r<repeats;++r)for(int block=0;block<8;++block)for(int k=0;k<8;++k) {
    size_t di=r*p.dstRepStride*8+block*p.dstBlkStride*8+k;
    d.write(di,s.read(r*8+block));last=std::max(last,di);
  }
  if(repeats)mark(d,last);
}
template<class F> void binary(LocalTensor<float> d,LocalTensor<float> a,LocalTensor<float> b,
    uint64_t mask,uint8_t repeats,BinaryRepeatParams p,F f) {
  size_t last=0;
  for(int r=0;r<repeats;++r)for(int e=0;e<int(mask);++e) {
    size_t di=index<float>(r,e,p.dstBlkStride,p.dstRepStride);
    volatile float result=f(a.read(index<float>(r,e,p.src0BlkStride,p.src0RepStride)),
                            b.read(index<float>(r,e,p.src1BlkStride,p.src1RepStride)));
    d.write(di,result);last=std::max(last,di);
  }
  if(repeats && mask)mark(d,last);
}
#define BINARY(NAME,EXPR) \
void NAME(LocalTensor<float> d,LocalTensor<float> a,LocalTensor<float> b,uint64_t m,uint8_t r,BinaryRepeatParams p) { \
  binary(d,a,b,m,r,p,[](float x,float y){return EXPR;}); \
} \
void NAME(LocalTensor<float> d,LocalTensor<float> a,LocalTensor<float> b,int n) { \
  for(int i=0;i<n;++i){volatile float x=a.read(i),y=b.read(i);d.write(i,EXPR);}if(n)mark(d,n-1); \
}
BINARY(Sub,x-y)
BINARY(Mul,x*y)
BINARY(Add,x+y)
#undef BINARY
void Muls(LocalTensor<float> d,LocalTensor<float> s,float scalar,uint64_t mask,uint8_t repeats,UnaryRepeatParams p) {
  size_t last=0;
  for(int r=0;r<repeats;++r)for(int e=0;e<int(mask);++e) {
    size_t di=index<float>(r,e,p.dstBlkStride,p.dstRepStride);
    volatile float f=s.read(index<float>(r,e,p.srcBlkStride,p.srcRepStride))*scalar;
    d.write(di,f);last=std::max(last,di);
  }
  if(repeats && mask)mark(d,last);
}
template<class T> void ArithProgression(LocalTensor<T> d,T first,T step,int n) {
  for(int i=0;i<n;++i)d.write(i,static_cast<T>(first+i*step));if(n)mark(d,n-1);
}
}
namespace oscar_ascend_device {
using AscendC::Fence;
bool Finite(float x) {return x==x && x<=3.402823466e38F && x>=-3.402823466e38F;}
}
using namespace AscendC;
@@CAST_HEADER@@
@@OLD_HEADER@@
@@NEW_HEADER@@

constexpr int H=64,D=256;
struct Fixture {
  TBuf<TPosition::VECCALC> packed{H*160},word{32},plane{32},natural{H*272*2},mask{64};
  TBuf<TPosition::VECCALC> word_index{32},lane_index{32},meta_index{2*H*4},meta_half{2*H*2};
  TBuf<TPosition::VECCALC> dequant{H*D*4},scratch{H*8*4},output{H*D*2};
  std::vector<uint8_t> permanent_mask,permanent_index;
  void init() {
    vector_writes.clear();oscar_ascend_aligned::InitIndices<D,H>(word_index,lane_index,meta_index,mask);
    permanent_mask.assign(mask.owner->ptr(),mask.owner->ptr()+mask.owner->size);
    permanent_index.assign(meta_index.owner->ptr(),meta_index.owner->ptr()+meta_index.owner->size);
  }
  void check() {
    for(auto* b:{&packed,&word,&plane,&natural,&mask,&word_index,&lane_index,&meta_index,&meta_half,&dequant,&scratch,&output})b->owner->check();
    require(std::memcmp(mask.owner->ptr(),permanent_mask.data(),permanent_mask.size())==0 &&
            std::memcmp(meta_index.owner->ptr(),permanent_index.data(),permanent_index.size())==0,
            "permanent unpack mask/index changed across calls");
  }
  std::vector<uint16_t> run(bool newer,bool value,int live,int mode,int& error) {
    vector_writes.clear();
    if(newer) {
      oscar_ascend_codebook::Unpack<D,H>(value,live,error,packed,word,plane,natural,mask,
          word_index,lane_index,meta_index,meta_half,dequant,scratch,mode);
    } else {
      oscar_ascend_aligned::Unpack<D,H>(value,live,error,packed,word,plane,natural,mask,
          word_index,lane_index,meta_index,meta_half,dequant,scratch,mode);
      oscar_bf16::CastRne(output.Get<bfloat16_t>(),dequant.Get<float>(),H*D);
    }
    Fence<HardEvent::V_S>();
    auto result=newer?natural.Get<uint16_t>():output.Get<uint16_t>();
    std::vector<uint16_t> bits(H*D);
    for(int i=0;i<H*D;++i)bits[i]=result.read(i);
    check();return bits;
  }
};
uint32_t read_u32(const std::vector<uint8_t>& bytes,size_t i) {uint32_t x;std::memcpy(&x,bytes.data()+i,4);return x;}
std::vector<uint16_t> reference(const std::vector<uint8_t>& data,bool value,int live,int mode,int& error) {
  std::vector<uint16_t> out(H*D);std::array<float,H> scales,zeros;
  for(int r=0;r<H;++r) {
    scales[r]=from_bits(read_u32(data,r*160+128+int(value)*8));
    zeros[r]=from_bits(read_u32(data,r*160+132+int(value)*8));
    if(r<live && (!(scales[r]>0) || !std::isfinite(scales[r]) || !std::isfinite(zeros[r])))error=3;
  }
  if(mode==1)for(int r=0;r<H;++r) {
    scales[r]=scalar_bf16(scales[r]);zeros[r]=scalar_bf16(zeros[r]);
    if(r<live && (!(scales[r]>0) || !std::isfinite(scales[r]) || !std::isfinite(zeros[r])))error=3;
  }
  else if(mode!=0)error=1;
  for(int r=0;r<H;++r)for(int d=0;d<D;++d) {
    int byte=r*160+int(value)*64+(d%32)*2;
    uint16_t word=data[byte]|(uint16_t(data[byte+1])<<8);
    int code=(word>>(2*(d/32)))&3;
    volatile float diff=float(code)-zeros[r];
    if(mode==1)diff=scalar_bf16(diff);
    volatile float result=diff*scales[r];out[r*D+d]=scalar_rne(result);
  }
  return out;
}
int main(int argc,char** argv) {
  try {
    require(argc==6,"mode live initial-error input output expected");
    int mode=std::stoi(argv[1]),live=std::stoi(argv[2]),initial=std::stoi(argv[3]);
    require(live>=0 && live<=H,"invalid live rows");
    std::ifstream input(argv[4],std::ios::binary);std::ofstream output(argv[5],std::ios::binary);
    require(bool(input) && bool(output),"fixture IO failed");
    Fixture old,next;old.init();next.init();
    std::vector<int> statuses;size_t records=0;
    for(;;) {
      std::vector<uint8_t> data(H*160);input.read(reinterpret_cast<char*>(data.data()),data.size());
      if(input.gcount()==0)break;require(size_t(input.gcount())==data.size(),"truncated packed fixture");
      std::memcpy(old.packed.owner->ptr(),data.data(),data.size());
      std::memcpy(next.packed.owner->ptr(),data.data(),data.size());
      int old_error=initial,new_error=initial,reference_error=initial;
      for(bool value:{false,true}) {
        auto a=old.run(false,value,live,mode,old_error);
        auto b=next.run(true,value,live,mode,new_error);
        auto expected=reference(data,value,live,mode,reference_error);
        require(old_error==new_error && old_error==reference_error,"metadata error/order changed");
        if(a!=b || a!=expected) {
          for(size_t i=0;i<a.size();++i)if(a[i]!=b[i] || a[i]!=expected[i]) {
            std::cerr<<"mismatch row="<<i/D<<" dim="<<i%D<<" side="<<value<<" old="<<a[i]<<" new="<<b[i]<<" ref="<<expected[i]<<"\n";break;
          }
          throw std::runtime_error("BF16 bits differ");
        }
        statuses.push_back(new_error);
        output.write(reinterpret_cast<const char*>(b.data()),b.size()*2);
        require(std::memcmp(old.packed.owner->ptr(),data.data(),data.size())==0 &&
                std::memcmp(next.packed.owner->ptr(),data.data(),data.size())==0,"packed input was modified");
      }
      ++records;
    }
    require(records>0,"no packed fixtures");
    std::cout<<"{\"records\":"<<records<<",\"gather_calls\":"<<gather_calls
      <<",\"gather_largest\":"<<gather_largest<<",\"gather_index_max\":"<<gather_index_max
      <<",\"raw_gather_calls\":"<<raw_gather_calls<<",\"raw_gather_index_max\":"<<raw_gather_index_max<<",\"statuses\":[";
    for(size_t i=0;i<statuses.size();++i){if(i)std::cout<<',';std::cout<<statuses[i];}
    std::cout<<"],\"host_model_only\":true}\n";return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
