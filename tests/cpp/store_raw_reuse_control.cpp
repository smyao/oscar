// Archive #126/#145/#154/#155/#164/#172 and startup D.4.
// Host-only actual full AlignedStore comparison, not CANN/NPU or timing.
// Quantizer is a destructive deterministic byte/error stub; BF16 values,
// bounds, all GM transactions, all fences, and raw bytes are checked.
// Recorded fence order cannot certify asynchronous device scheduling.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define __aicore__
#define __gm__
using GM_ADDR = uint8_t*;
using bfloat16_t = uint16_t;
constexpr int32_t kMaxDim = 256;
constexpr int32_t kRowTile = 8;
enum class TPosition { VECCALC };
enum class RoundMode { CAST_NONE };
enum class HardEvent { MTE2_V, V_MTE2, S_MTE3, MTE3_S, MTE2_S, S_MTE2,
                       V_MTE3, MTE3_MTE2 };
constexpr int PIPE_V = 0;
struct Borrow {const void* buffer;size_t offset,bytes;};
std::vector<Borrow> mte3_borrows;
void check_unborrowed(const void* buffer,size_t offset,size_t bytes) {
  for(const auto& b:mte3_borrows)
    if(buffer==b.buffer && offset<b.offset+b.bytes && b.offset<offset+bytes)
      throw std::runtime_error("UB overwritten before MTE3 completion");
}
std::vector<int> fence_trace;
template <HardEvent E> void Fence() {
  fence_trace.push_back(static_cast<int>(E));
  // Conservative event model: remember an issued GM write's UB read until
  // an explicit MTE3 completion handoff. This detects missing reuse fences,
  // not the hardware event allocator or actual asynchronous device timing.
  if(E==HardEvent::MTE3_MTE2 || E==HardEvent::MTE3_S) mte3_borrows.clear();
}
template <int P> void PipeBarrier() { fence_trace.push_back(1000+P); }
int core_index = 0, core_count = 1;
int GetBlockIdx() { return core_index; }
int GetBlockNum() { return core_count; }

void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
template <class T> struct LocalTensor {
  std::shared_ptr<std::vector<uint64_t>> owner;
  size_t byte_offset = 0;
  size_t origin_offset = 0;
  T* data(size_t count) const {
    require(byte_offset + count * sizeof(T) <= owner->size() * 8,
            "stub local tensor bounds");
    return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(owner->data()) + byte_offset);
  }
  LocalTensor operator[](size_t i) const {
    return {owner, byte_offset + i * sizeof(T), origin_offset + i * sizeof(T)};
  }
  T GetValue(size_t i) const { return data(i + 1)[i]; }
  void SetValue(size_t i, T x) { data(i + 1)[i] = x; }
};
template <TPosition> struct TBuf {
  std::shared_ptr<std::vector<uint64_t>> owner;
  template <class T> LocalTensor<T> Get() { return {owner}; }
};
std::vector<size_t> allocations;
struct TPipe {
  template <TPosition P> void InitBuffer(TBuf<P>& buf, size_t bytes) {
    allocations.push_back(bytes);
    buf.owner = std::make_shared<std::vector<uint64_t>>((bytes + 7) / 8,
                                                      UINT64_C(0xa5a5a5a5a5a5a5a5));
  }
};
struct Region { uintptr_t begin; size_t bytes; std::string name; };
struct Copy {
  std::string region; size_t offset; std::vector<uint8_t> bytes;
  bool operator==(const Copy& other) const {
    return region==other.region && offset==other.offset && bytes==other.bytes;
  }
};
std::vector<Copy> loads, writes;
std::vector<Region> regions;
void check_global(const void* ptr, size_t bytes) {
  const auto address = reinterpret_cast<uintptr_t>(ptr);
  for (const auto& r : regions)
    if (address >= r.begin && address - r.begin <= r.bytes &&
        bytes <= r.bytes - (address - r.begin)) return;
  throw std::runtime_error("stub global tensor bounds");
}
Copy transaction(const void* ptr, size_t bytes) {
  check_global(ptr,bytes);
  auto address=reinterpret_cast<uintptr_t>(ptr);
  for(const auto& r:regions) if(address>=r.begin && address-r.begin<r.bytes) {
    const auto* data=reinterpret_cast<const uint8_t*>(ptr);
    return {r.name,address-r.begin,std::vector<uint8_t>(data,data+bytes)};
  }
  throw std::runtime_error("unregistered transaction");
}
template <class T> struct GlobalTensor {
  T* base = nullptr;
  void SetGlobalBuffer(T* ptr) { base = ptr; }
  GlobalTensor operator[](size_t i) const { return {base + i}; }
};
uintptr_t key_base, value_base;
size_t input_bytes;
int dim_current, loaded_side;
size_t loaded_row;
void record_load(const void* pointer) {
  auto p = reinterpret_cast<uintptr_t>(pointer);
  if (p >= key_base && p - key_base < input_bytes) {
    loaded_side = 0; loaded_row = (p - key_base) / (2 * dim_current);
  } else if (p >= value_base && p - value_base < input_bytes) {
    loaded_side = 1; loaded_row = (p - value_base) / (2 * dim_current);
  }
}
template <class T> void DataCopy(LocalTensor<T> dst, GlobalTensor<T> src, size_t count) {
  check_global(src.base, count * sizeof(T));
  check_unborrowed(dst.owner.get(),dst.byte_offset,count*sizeof(T));
  loads.push_back(transaction(src.base,count*sizeof(T)));
  record_load(src.base);
  std::memcpy(dst.data(count), src.base, count * sizeof(T));
}
template <class T> void DataCopy(GlobalTensor<T> dst, LocalTensor<T> src, size_t count) {
  check_global(dst.base, count * sizeof(T));
  mte3_borrows.push_back({src.owner.get(),src.byte_offset,count*sizeof(T)});
  std::memcpy(dst.base, src.data(count), count * sizeof(T));
  writes.push_back(transaction(dst.base,count*sizeof(T)));
}
struct DataCopyExtParams { uint32_t blocks, length, src_stride, dst_stride, reserved; };
template <class T> struct DataCopyPadExtParams { bool enabled; int left, right; T value; };
template <class T> void DataCopyPad(LocalTensor<T> dst, GlobalTensor<T> src,
                                  DataCopyExtParams p, DataCopyPadExtParams<T>) {
  require(p.blocks == 1 && p.length % sizeof(T) == 0, "stub read DMA geometry");
  DataCopy(dst, src, p.length / sizeof(T));
}
template <class T> void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> src,
                                  DataCopyExtParams p) {
  require(p.blocks == 1 && p.length % sizeof(T) == 0, "stub write DMA geometry");
  DataCopy(dst, src, p.length / sizeof(T));
}
float bf16_float(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result; std::memcpy(&result, &bits, 4); return result;
}
uint16_t bf16(float value) {
  uint32_t bits; std::memcpy(&bits, &value, 4); return static_cast<uint16_t>(bits >> 16);
}
uint32_t float_bits(float value) {
  uint32_t result; std::memcpy(&result, &value, 4); return result;
}
void Cast(LocalTensor<float> dst, LocalTensor<bfloat16_t> src, RoundMode, size_t count) {
  for (size_t i = 0; i < count; ++i) dst.SetValue(i, bf16_float(src.GetValue(i)));
}
struct QuantCall {
  size_t row; int side; uint32_t ratio; int packed_offset, metadata_offset;
  std::vector<uint32_t> values;
  bool operator==(const QuantCall& other) const {
    return row == other.row && side == other.side && ratio == other.ratio &&
        packed_offset == other.packed_offset && metadata_offset == other.metadata_offset &&
        values == other.values;
  }
};
std::vector<QuantCall> quant_calls;
class ClipQuantize {
 public:
  void Init(TPipe*, int32_t dim) { dim_ = dim; }
  int32_t Process(LocalTensor<float> x, float ratio, LocalTensor<uint8_t> packed,
                  int packed_offset, int metadata_offset) {
    QuantCall call{loaded_row + x.origin_offset / sizeof(float) / dim_, loaded_side,
                   float_bits(ratio), packed_offset, metadata_offset, {}};
    uint32_t hash = 2166136261u;
    bool invalid = false;
    for (int i = 0; i < dim_; ++i) {
      auto bits = float_bits(x.GetValue(i));
      call.values.push_back(bits); hash = (hash ^ bits) * 16777619u;
      invalid |= !std::isfinite(x.GetValue(i));
    }
    quant_calls.push_back(call);
    if (invalid) return 2;
    if (x.GetValue(0) == -776.0f) return 3;  // Explicit synthetic quantizer-error injection.
    for (int i = 0; i < dim_ / 4; ++i)
      packed.SetValue(packed_offset + i, static_cast<uint8_t>((hash >> (8 * (i % 4))) ^ i));
    for (int i = 0; i < 8; ++i)
      packed.SetValue(metadata_offset + i, static_cast<uint8_t>(hash >> (8 * (i % 4))));
    // A raw publication must never use this clipped/modified float buffer.
    for(int i=0;i<dim_;++i) x.SetValue(i,12345.0f+i);
    return 0;
  }
 private: int dim_;
};

#include "store_raw_reuse_classes.inc"

struct Case {
  int tokens,heads,cores,alias_mode,sink=4,recent=3,blocks=3;
  static constexpr int dim=256,block=128,packed_offset=64;
  std::vector<uint16_t> key,value,original_key,original_value;
  std::vector<int64_t> slots,positions;
  // alias_mode bits 0/1 independently alias originals; bit 2 shares K/V input.
  Case(int n,int h,int c,int a):tokens(n),heads(h),cores(c),alias_mode(a),
      key(n*h*dim),value(n*h*dim),original_key(n*h*dim),original_value(n*h*dim),
      slots(n),positions(n) {
    for(int i=0;i<n*h*dim;++i) {
      key[i]=bf16(static_cast<float>((i%113)-55)/8);
      value[i]=bf16(static_cast<float>((i%67)-31)/4);
      original_key[i]=static_cast<uint16_t>(key[i]^0x8000);
      original_value[i]=static_cast<uint16_t>(value[i]^0x8000);
    }
    for(int t=0;t<n;++t) {slots[t]=t;positions[t]=t;}
    for(int row=0;row<n*h;++row) {
      key[row*dim+1]=0x8000; value[row*dim+1]=0;
      key[row*dim+2]=0; value[row*dim+2]=0x8000;
      // Distinct raw originals may contain arbitrary BF16 payloads. The
      // writer validates rotated inputs, not these exact original bytes.
      const uint16_t special[]={0x8000,0,0x7fc1,0xffc3,0x7f80,0xff80,1,0x8001};
      for(int j=0;j<8;++j) {
        original_key[row*dim+j]=special[j];
        original_value[row*dim+j]=special[7-j];
      }
    }
  }
  std::vector<uint16_t>& input_v() {return alias_mode&4?key:value;}
  std::vector<uint16_t>& original_k() {return alias_mode&1?key:original_key;}
  std::vector<uint16_t>& original_v() {return alias_mode&2?input_v():original_value;}
  bool reuses() const {return (alias_mode&3)==3;}
};
struct Result {
  std::vector<uint8_t> packed;
  std::vector<uint16_t> wk,wv;
  std::vector<int64_t> tags;
  std::vector<int32_t> status;
  std::vector<QuantCall> calls;
  std::vector<Copy> reads,stores;
  std::vector<int> fences;
  std::vector<size_t> ub;
  explicit Result(const Case& c):
      packed(Case::packed_offset+c.blocks*Case::block*c.heads*144+64,0xa5),
      wk(c.blocks*(c.sink+c.recent)*c.heads*Case::dim,0x7fc1),
      wv(wk.size(),0x8000),tags(c.blocks*(c.sink+c.recent),-999),
      status(c.tokens*c.heads,99) {}
};
template <class T> void region(std::vector<T>& values,const char* name) {
  regions.push_back({reinterpret_cast<uintptr_t>(values.data()),values.size()*sizeof(T),name});
}
template <class Kernel> void execute(Case& c,Result& r) {
  regions.clear();
  region(c.key,"key");region(c.value,"value");
  region(c.original_key,"original_key");region(c.original_value,"original_value");
  region(c.slots,"slots");region(c.positions,"positions");region(r.packed,"packed");
  region(r.wk,"wk");region(r.wv,"wv");region(r.tags,"tags");region(r.status,"status");
  key_base=reinterpret_cast<uintptr_t>(c.key.data());
  value_base=reinterpret_cast<uintptr_t>(c.input_v().data());input_bytes=c.key.size()*2;
  dim_current=Case::dim;quant_calls.clear();loads.clear();writes.clear();
  fence_trace.clear();allocations.clear();mte3_borrows.clear();core_count=c.cores;
  auto ptr=[](auto& v){return reinterpret_cast<GM_ADDR>(v.data());};
  for(core_index=0;core_index<core_count;++core_index) {
    Kernel op;
    op.Run(ptr(c.key),ptr(c.input_v()),ptr(c.original_k()),ptr(c.original_v()),
        ptr(c.slots),ptr(c.positions),ptr(r.packed),ptr(r.wk),ptr(r.wv),ptr(r.tags),
        ptr(r.status),c.tokens,c.heads,Case::dim,Case::block,c.blocks,
        Case::packed_offset,Case::block*c.heads*144,
        (c.sink+c.recent)*c.heads*Case::dim,(c.sink+c.recent)*c.heads*Case::dim,
        c.sink+c.recent,c.sink,c.recent,.96f,.92f);
  }
  r.calls=quant_calls;r.reads=loads;r.stores=writes;r.fences=fence_trace;r.ub=allocations;
}
size_t byte_count(const std::vector<Copy>& items) {
  size_t result=0;for(const auto& item:items) result+=item.bytes.size();return result;
}
void compare(Case& c,Result& old,Result& next) {
  execute<BeforeStore>(c,old);execute<ReuseStore>(c,next);
  require(old.packed==next.packed,"packed bytes changed");
  require(old.wk==next.wk && old.wv==next.wv,"raw BF16 bits changed");
  require(old.tags==next.tags && old.status==next.status,"tags/status changed");
  require(old.calls==next.calls,"quantizer sequence/operands/ratio changed");
  require(old.stores==next.stores,"GM writes/address/payload/order changed");
  require(old.fences==next.fences,"DMA/vector/scalar fence sequence changed");
  require(old.ub.size()==next.ub.size(),"allocation count changed");
  require(old.ub.size()==static_cast<size_t>(6*c.cores),"unexpected class allocation count");
  for(size_t i=0;i<old.ub.size();++i) {
    if(i%6==5) require(old.ub[i]==512 && next.ub[i]==4096,"bounded raw allocation changed");
    else require(old.ub[i]==next.ub[i],"unrelated allocation changed");
  }
  size_t active=std::count_if(c.slots.begin(),c.slots.end(),[](int64_t s){return s>=0;});
  require(next.calls.size()==2*active*c.heads,"quantization was skipped or duplicated");
  size_t raw_writes=0;
  for(const auto& w:old.stores) if(w.region=="wk" || w.region=="wv") {
    require(w.bytes.size()==512,"unexpected raw DMA geometry");++raw_writes;
  }
  if(c.reuses()) {
    require(old.reads.size()==next.reads.size()+raw_writes,"GM read count did not omit exactly raw rereads");
    require(byte_count(old.reads)==byte_count(next.reads)+raw_writes*512,"GM read byte saving incorrect");
    // Every remaining transaction must occur in the old stream, with exact
    // source address/payload/order; no new GM source or speculative fetch.
    size_t next_index=0;
    for(const auto& old_read:old.reads)
      if(next_index<next.reads.size() && old_read==next.reads[next_index]) ++next_index;
    require(next_index==next.reads.size(),"new or reordered GM read introduced");
  } else require(old.reads==next.reads,"distinct/partial alias reload path changed");
  for(int t=0;t<c.tokens;++t) for(int h=0;h<c.heads;++h) {
    int row=t*c.heads+h;
    if(c.slots[t]<0) require(next.status[row]==0,"negative slot must stay padding");
    else if(c.slots[t]>=c.blocks*Case::block) require(next.status[row]==1,"slot error priority");
    else if(c.positions[t]<0) require(next.status[row]==4,"position error priority");
  }
}
void run(const std::string& name) {
  int n=25,h=1,cores=2,alias=3;
  if(name=="distinct_originals") alias=0;
  if(name=="ownership_negative") alias=0;
  if(name=="key_alias_only") alias=1;
  if(name=="value_alias_only") alias=2;
  if(name=="same_kv_source") alias=7;
  if(name=="many_heads") {h=6;n=37;cores=3;}
  if(name=="tail_tile") {n=11;h=3;}
  if(name=="one_row") n=1;
  if(name=="bs16") n=16;
  if(name=="bs28") n=28;
  if(name=="zero_tokens") n=0;
  if(name=="page_boundary") {n=139;h=2;cores=3;}
  Case c(n,h,cores,alias);
  if(name=="sink_only") {c.sink=128;c.recent=0;}
  if(name=="recent_only") c.sink=0;
  if(name=="all_padding") std::fill(c.slots.begin(),c.slots.end(),-1);
  if(name=="interleaved" || name=="errors")
    for(int t=0;t<n;++t) if(t%3==0) c.slots[t]=-1;
  if(name=="bs16" || name=="bs28") {
    // One current row per independent request/page: every raw row is live
    // without manufacturing conflicting owners of a shared ring slot.
    c.blocks=n;
    for(int t=0;t<n;++t) {c.slots[t]=t*Case::block+4;c.positions[t]=4;}
  }
  for(int t=0;t<n;++t) if(c.slots[t]<0) {
    c.key[t*h*Case::dim]=0x7fc1;c.value[t*h*Case::dim]=0xff80;
  }
  if(name=="errors") {
    c.slots[0]=c.blocks*Case::block;c.positions[0]=-1;c.key[0]=0x7fc1;
    c.slots[1]=1;c.positions[1]=-1;c.key[h*Case::dim]=0xff80;
    c.slots[2]=2;c.key[2*h*Case::dim]=0x7fc1;
    c.slots[3]=3;c.key[3*h*Case::dim]=bf16(-776);c.value[3*h*Case::dim]=0x7fc1;
    c.slots[4]=4;c.key[4*h*Case::dim]=bf16(1);c.value[4*h*Case::dim]=bf16(-776);
  }
  Result old(c),next(c);
  if(name=="ownership_negative") {
    execute<UnsafeReuseStore>(c,next);
    throw std::runtime_error("negative ownership control unexpectedly completed");
  }
  compare(c,old,next);
  if(name=="errors") require(next.status[2*h]==2 && next.status[3*h]==3 && next.status[4*h]==3,
                              "K-before-V quantizer error priority");
  if(name=="same_address_replay") {
    auto key_ptr=c.key.data();auto slot_ptr=c.slots.data();auto output_ptr=next.wk.data();
    for(int t=0;t<n;++t) {
      c.slots[t]=(t%2==0?t:-1);c.positions[t]=t;
      for(int i=0;i<h*Case::dim;++i) {c.key[t*h*Case::dim+i]=bf16(2);c.value[t*h*Case::dim+i]=bf16(-2);}
    }
    compare(c,old,next);
    require(key_ptr==c.key.data() && slot_ptr==c.slots.data() && output_ptr==next.wk.data(),
            "fixed-address replay reallocated buffers");
  }
  std::cout<<name<<" passed quant="<<next.calls.size()<<" old_reads="<<old.reads.size()
           <<" new_reads="<<next.reads.size()<<" saved_bytes="<<byte_count(old.reads)-byte_count(next.reads)<<"\n";
}
int main(int argc,char** argv) {
  try {require(argc==2,"one case required");run(argv[1]);return 0;}
  catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}

