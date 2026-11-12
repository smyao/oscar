// Host-only control-flow harness: extracted production AlignedStore classes.
// NOT AscendC CPU-debug, CANN arithmetic, DMA scheduling, or a speed test.
// Quantization is a deterministic byte-writing/error stub. Both real Run
// bodies, Read/WriteRaw/WriteTag, and row publication execute unchanged.
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
template <HardEvent> void Fence() {}
template <int> void PipeBarrier() {}
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
struct TPipe {
  template <TPosition P> void InitBuffer(TBuf<P>& buf, size_t bytes) {
    buf.owner = std::make_shared<std::vector<uint64_t>>((bytes + 7) / 8,
                                                      UINT64_C(0xa5a5a5a5a5a5a5a5));
  }
};
struct Region { uintptr_t begin; size_t bytes; };
std::vector<Region> regions;
void check_global(const void* ptr, size_t bytes) {
  const auto address = reinterpret_cast<uintptr_t>(ptr);
  for (const auto& r : regions)
    if (address >= r.begin && address - r.begin <= r.bytes &&
        bytes <= r.bytes - (address - r.begin)) return;
  throw std::runtime_error("stub global tensor bounds");
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
  record_load(src.base);
  std::memcpy(dst.data(count), src.base, count * sizeof(T));
}
template <class T> void DataCopy(GlobalTensor<T> dst, LocalTensor<T> src, size_t count) {
  check_global(dst.base, count * sizeof(T));
  std::memcpy(dst.base, src.data(count), count * sizeof(T));
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
    return 0;
  }
 private: int dim_;
};

#include "active_store_classes.inc"

struct Case {
  int tokens, heads, cores; static constexpr int dim = 256, block = 128, blocks = 3;
  static constexpr int sink = 4, recent = 3, packed_offset = 64;
  bool alias;
  std::vector<uint16_t> key, value, original_key, original_value;
  std::vector<int64_t> slots, positions;
  Case(int n, int h, int c, bool a) : tokens(n), heads(h), cores(c), alias(a),
      key(n*h*dim), value(n*h*dim), original_key(n*h*dim), original_value(n*h*dim),
      slots(n), positions(n) {
    for (int i = 0; i < n*h*dim; ++i) {
      key[i] = bf16(static_cast<float>((i % 113) - 55) / 8);
      value[i] = bf16(static_cast<float>((i % 67) - 31) / 4);
      original_key[i] = static_cast<uint16_t>(key[i] ^ 0x8000);
      original_value[i] = static_cast<uint16_t>(value[i] ^ 0x8000);
    }
    for (int t = 0; t < n; ++t) { slots[t] = t; positions[t] = t; }
    if (n) { key[0] = 0x8000; value[0] = 0; original_key[0] = 0; original_value[0] = 0x8000; }
  }
};
struct Result {
  std::vector<uint8_t> packed;
  std::vector<uint16_t> wk, wv;
  std::vector<int64_t> tags;
  std::vector<int32_t> status;
  std::vector<QuantCall> calls;
  explicit Result(const Case& c) :
      packed(Case::packed_offset + Case::blocks*Case::block*c.heads*144 + 64, 0xa5),
      wk(Case::blocks*(Case::sink+Case::recent)*c.heads*Case::dim, 0x7fc1),
      wv(wk.size(), 0x8000), tags(Case::blocks*(Case::sink+Case::recent), -999),
      status(c.tokens*c.heads, 99) {}
};
template <class T> void region(std::vector<T>& values) {
  regions.push_back({reinterpret_cast<uintptr_t>(values.data()), values.size()*sizeof(T)});
}
template <class Kernel> void execute(Case& c, Result& r) {
  regions.clear();
  region(c.key); region(c.value); region(c.original_key); region(c.original_value);
  region(c.slots); region(c.positions); region(r.packed); region(r.wk); region(r.wv);
  region(r.tags); region(r.status);
  key_base = reinterpret_cast<uintptr_t>(c.key.data());
  value_base = reinterpret_cast<uintptr_t>(c.value.data()); input_bytes = c.key.size()*2;
  dim_current = Case::dim; quant_calls.clear(); core_count = c.cores;
  auto ptr = [](auto& v) { return reinterpret_cast<GM_ADDR>(v.data()); };
  for (core_index = 0; core_index < core_count; ++core_index) {
    Kernel op;
    op.Run(ptr(c.key), ptr(c.value), ptr(c.alias?c.key:c.original_key),
        ptr(c.alias?c.value:c.original_value), ptr(c.slots), ptr(c.positions), ptr(r.packed),
        ptr(r.wk), ptr(r.wv), ptr(r.tags), ptr(r.status), c.tokens, c.heads, Case::dim,
        Case::block, Case::blocks, Case::packed_offset, Case::block*c.heads*144,
        (Case::sink+Case::recent)*c.heads*Case::dim,
        (Case::sink+Case::recent)*c.heads*Case::dim, Case::sink+Case::recent,
        Case::sink, Case::recent, .96f, .92f);
  }
  r.calls = quant_calls;
}
void compare(Case& c, Result& old, Result& next) {
  execute<MotherStore>(c, old); execute<ActiveStore>(c, next);
  require(old.packed == next.packed, "packed mismatch");
  require(old.wk == next.wk && old.wv == next.wv, "raw window mismatch");
  require(old.tags == next.tags, "tags mismatch");
  require(old.status == next.status, "status mismatch");
  std::vector<QuantCall> wanted;
  for (const auto& call : old.calls)
    if (c.slots.at(call.row / c.heads) >= 0) wanted.push_back(call);
  require(wanted == next.calls, "nonnegative quantizer sequence/operands changed");
  require(old.calls.size() == static_cast<size_t>(2*c.tokens*c.heads), "mother count");
  size_t active = std::count_if(c.slots.begin(), c.slots.end(), [](int64_t s){return s>=0;});
  require(next.calls.size() == 2*active*c.heads, "active quantizer count");
  for (int t = 0; t < c.tokens; ++t) for (int h = 0; h < c.heads; ++h) {
    const int row = t*c.heads+h;
    if (c.slots[t] < 0) require(next.status[row] == 0, "padding status must be zero");
    else if (c.slots[t] >= Case::blocks*Case::block)
      require(next.status[row] == 1, "invalid positive slot priority");
    else if (c.positions[t] < 0) require(next.status[row] == 4, "negative position priority");
  }
}
void run(std::string name) {
  int n=17, h=1, cores=3; bool alias=true;
  if (name=="many_heads") h=6;
  if (name=="tail_tile") { n=11; h=3; }
  if (name=="distinct_originals") { h=2; alias=false; }
  if (name=="bs16") n=64;
  if (name=="bs28") n=112;
  if (name=="zero_tokens") n=0;
  Case c(n,h,cores,alias);
  if (name=="bs16" || name=="bs28")
    for (int t=n/4; t<n; ++t) c.slots[t]=-1;
  else if (name=="all_padding")
    std::fill(c.slots.begin(), c.slots.end(), -1);
  else if (name!="all_live" && name!="zero_tokens")
    for (int t=0;t<n;++t) if (t%3!=1) c.slots[t]=-1-(t%2);
  // Padding poison must stay discarded; source signed-zero payloads are
  // compared in the actual raw-window writes and quantizer input trace.
  for(int t=0;t<n;++t) if(c.slots[t]<0) {
    c.key[t*h*Case::dim]=0x7fc1; c.value[t*h*Case::dim]=0xff80;
  }
  if (name=="errors") {
    c.slots[0]=Case::blocks*Case::block; c.positions[0]=-1; c.key[0]=0x7fc1;
    c.slots[1]=1; c.positions[1]=-1; c.key[h*Case::dim]=0x7fc1;
    c.slots[2]=2; c.key[2*h*Case::dim]=0x7fc1;
    c.slots[3]=3; c.key[3*h*Case::dim]=bf16(-776); c.value[3*h*Case::dim]=0x7fc1;
    c.slots[4]=4; c.key[4*h*Case::dim]=bf16(1); c.value[4*h*Case::dim]=bf16(-776);
  }
  Result old(c), next(c); compare(c,old,next);
  if (name=="errors") {
    require(next.status[2*h]==2 && next.status[3*h]==3 && next.status[4*h]==3,
            "K-before-V quantizer error priority");
  }
  if (name=="same_address_replay") {
    auto pointer=c.slots.data();
    for(int t=0;t<n;++t) {c.slots[t]=(t%2==0 ? t : -1); c.key[t*h*Case::dim]=bf16(2); c.value[t*h*Case::dim]=bf16(-2);}
    require(pointer==c.slots.data(),"slot address changed");
    compare(c,old,next);  // Existing output allocations also stay fixed.
  }
  std::cout << name << " passed old_quant=" << old.calls.size()
            << " active_quant=" << next.calls.size() << "\n";
}
int main(int argc,char** argv) {
  try {require(argc==2,"one case name required");run(argv[1]);return 0;}
  catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
