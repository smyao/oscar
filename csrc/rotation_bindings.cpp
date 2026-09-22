// SPDX-License-Identifier: Apache-2.0
// Archive G6/#17/#22/#34/#53/#69/#87/#108/#111: checked out-only ABI,
// retained FP32 transform arithmetic, no allocation/sync or CPU numerical path.
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <torch/extension.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include "include/oscar_rotation_launch.h"

namespace {
constexpr int64_t kLimit=std::numeric_limits<int64_t>::max();
int64_t Product(int64_t a,int64_t b,const char* what) {
  TORCH_CHECK(a>=0 && b>=0 && (b==0 || a<=kLimit/b),what," overflows int64");
  return a*b;
}
void Tensor(const at::Tensor& value,const at::Tensor& owner,at::ScalarType dtype,
            const char* name,bool contiguous=true) {
  TORCH_CHECK(value.device().type()==c10::DeviceType::PrivateUse1 && value.device()==owner.device(),
              name," must share the selected NPU");
  TORCH_CHECK(value.scalar_type()==dtype,name," has incorrect dtype");
  TORCH_CHECK(!contiguous || value.is_contiguous(),name," must be contiguous");
}
int32_t Input(const at::Tensor& x) {
  const auto dtype=x.scalar_type();
  TORCH_CHECK(dtype==at::kFloat || dtype==at::kHalf || dtype==at::kBFloat16,
              "rotation input must be FP32/FP16/BF16");
  Tensor(x,x,dtype,"input");
  TORCH_CHECK(x.dim()==3 && x.size(1)>0,"rotation input must be [N,H,D]");
  TORCH_CHECK(x.size(2)==64 || x.size(2)==128 || x.size(2)==256,"D must be 64/128/256");
  return dtype==at::kFloat ? 0 : dtype==at::kHalf ? 1 : 2;
}
void Matrix(const at::Tensor& matrix,const at::Tensor& x) {
  Tensor(matrix,x,at::kFloat,"rotation_transposed");
  TORCH_CHECK(matrix.dim()==2 && matrix.size(0)==x.size(2) && matrix.size(1)==x.size(2),
              "rotation_transposed must be FP32 [D,D]");
}
struct Region {uintptr_t start;uint64_t size;};
Region Bytes(const at::Tensor& t) {
  int64_t extent=t.numel()==0 ? 0 : 1;
  if (extent) for (int64_t axis=0;axis<t.dim();++axis) {
    TORCH_CHECK(t.stride(axis)>=0,"negative tensor strides are forbidden");
    const int64_t part=Product(t.size(axis)-1,t.stride(axis),"tensor extent");
    TORCH_CHECK(extent<=kLimit-part,"tensor extent overflows int64"); extent+=part;
  }
  return {reinterpret_cast<uintptr_t>(t.data_ptr()),
          static_cast<uint64_t>(Product(extent,t.element_size(),"tensor bytes"))};
}
bool Separate(Region a,Region b) {
  return !a.size || !b.size || (a.start<=b.start ? b.start-a.start>=a.size : a.start-b.start>=b.size);
}
void NoOverlap(Region a,Region b) {TORCH_CHECK(Separate(a,b),"rotation tensors have overlapping active bytes");}
void NoOverlap(const at::Tensor& a,const at::Tensor& b) {NoOverlap(Bytes(a),Bytes(b));}
void NoOverlapPages(const at::Tensor& a,const at::Tensor& b,int64_t aRowBytes,int64_t bRowBytes) {
  const Region ar=Bytes(a),br=Bytes(b);
  if (Separate(ar,br)) return;
  const uint64_t strideA=Product(a.stride(0),a.element_size(),"page stride");
  const uint64_t strideB=Product(b.stride(0),b.element_size(),"page stride");
  TORCH_CHECK(strideA==strideB && strideA>0,"overlapping page views need the same byte stride");
  const uint64_t distance=ar.start<=br.start ? (br.start-ar.start)%strideA : (ar.start-br.start)%strideA;
  const uint64_t earlier=ar.start<=br.start ? aRowBytes : bRowBytes;
  const uint64_t later=ar.start<=br.start ? bRowBytes : aRowBytes;
  TORCH_CHECK(distance>=earlier && later<=strideA-distance,"raw page views overlap");
}
uint32_t Cores(int64_t rows) {
  return static_cast<uint32_t>(std::min<int64_t>(32,rows>0 ? (rows-1)/8+1 : 1));
}
void Rotate(const at::Tensor& input,const at::Tensor& rotation,at::Tensor output,
            at::Tensor status,bool hadamard,const std::optional<at::Tensor>& slots) {
  const int32_t dtype=Input(input); Matrix(rotation,input);
  Tensor(output,input,at::kFloat,"output"); Tensor(status,input,at::kInt,"status");
  TORCH_CHECK(output.sizes()==input.sizes(),"rotation output must match input [N,H,D]");
  TORCH_CHECK(status.dim()==2 && status.size(0)==input.size(0) && status.size(1)==input.size(1),
              "rotation status must be int32[N,H]");
  for (const auto& dst:{output,status}) {NoOverlap(dst,input);NoOverlap(dst,rotation);}
  NoOverlap(output,status);
  if (slots.has_value()) {
    Tensor(*slots,input,at::kLong,"slot_mapping");
    TORCH_CHECK(slots->dim()==1 && slots->size(0)==input.size(0),"rotation slots must be int64[N]");
    NoOverlap(output,*slots);NoOverlap(status,*slots);
  }
  if (input.size(0)==0) return;
  const c10_npu::OptionalNPUGuard guard(input.device());
  const int64_t rows=Product(input.size(0),input.size(1),"rows");
  oscar_ascend::rotate_launch(c10_npu::getCurrentNPUStream().stream(),input.data_ptr(),
      rotation.data_ptr(),output.data_ptr(),status.data_ptr(),rows,input.size(2),dtype,
      hadamard,slots.has_value() ? slots->data_ptr() : nullptr,input.size(1),Cores(rows));
}
void RawView(const at::Tensor& raw,const at::Tensor& owner,int64_t blocks,int64_t rows,
             int64_t heads,int64_t dim,const char* name) {
  Tensor(raw,owner,at::kBFloat16,name,false);
  TORCH_CHECK(raw.dim()==4 && raw.size(0)==blocks && raw.size(1)==rows
      && raw.size(2)==heads && raw.size(3)==dim,name," must be BF16[blocks,S+R,H,D]");
  const int64_t row=Product(heads,dim,"raw row elements");
  TORCH_CHECK(raw.stride(3)==1 && raw.stride(2)==dim && raw.stride(1)==row
      && raw.stride(0)>=Product(rows,row,"raw page elements"),name," invalid page strides");
  TORCH_CHECK(raw.stride(0)%16==0,"raw BF16 page stride must be 32-byte aligned");
}
void RotateStore(const at::Tensor& key,const at::Tensor& value,const at::Tensor& rk,
    const at::Tensor& rv,const at::Tensor& slots,const at::Tensor& positions,
    at::Tensor packed,at::Tensor rawKey,at::Tensor rawValue,at::Tensor tags,
    at::Tensor status,int64_t blockTokens,int64_t blocks,int64_t offset,
    int64_t pageStride,int64_t sink,int64_t recent,double kClip,double vClip,bool hadamard) {
  const int32_t dtype=Input(key);Tensor(value,key,key.scalar_type(),"value");
  TORCH_CHECK(value.sizes()==key.sizes(),"K/V must share [N,H,D]");
  Matrix(rk,key);Matrix(rv,key);
  Tensor(slots,key,at::kLong,"slot_mapping");Tensor(positions,key,at::kLong,"positions");
  Tensor(packed,key,at::kByte,"packed");Tensor(status,key,at::kInt,"status");
  const int64_t n=key.size(0),h=key.size(1),d=key.size(2);
  TORCH_CHECK(slots.dim()==1 && slots.size(0)==n && positions.sizes()==slots.sizes(),
              "slots/positions must be int64[N]");
  TORCH_CHECK(status.dim()==2 && status.size(0)==n && status.size(1)==h,"status must be int32[N,H]");
  TORCH_CHECK(blockTokens>0 && blockTokens%128==0 && blocks>0 && offset>=0 && pageStride>0,
              "invalid packed page geometry");
  Product(blocks,blockTokens,"slot capacity");
  const int64_t payload=Product(Product(blockTokens,h,"page rows"),d/2+8,"packed payload");
  TORCH_CHECK(pageStride>=payload && packed.dim()==1 && offset<=packed.numel()
      && blocks<=(packed.numel()-offset)/pageStride,"packed allocation does not cover page geometry");
  TORCH_CHECK(sink>=0 && recent>=0 && sink<=blockTokens && recent<=blockTokens
      && sink<=kLimit-recent && sink+recent>0,"invalid exact window geometry");
  TORCH_CHECK(std::isfinite(kClip) && std::isfinite(vClip) && kClip>=0 && kClip<=1
      && vClip>=0 && vClip<=1,"clip ratios must be finite in [0,1]");
  const int64_t window=sink+recent;
  RawView(rawKey,key,blocks,window,h,d,"raw_key");
  RawView(rawValue,key,blocks,window,h,d,"raw_value");
  Tensor(tags,key,at::kLong,"raw_tags",false);
  TORCH_CHECK(tags.dim()==2 && tags.size(0)==blocks && tags.size(1)==window
      && tags.stride(1)==1 && tags.stride(0)>=window,"raw_tags must be strided int64[blocks,S+R]");
  const uint64_t rawBytes=Product(Product(Product(window,h,"window rows"),d,"window dims"),2,"window bytes");
  const uint64_t tagBytes=Product(window,8,"tag bytes");
  const Region packedActive{reinterpret_cast<uintptr_t>(packed.data_ptr())+static_cast<uint64_t>(offset),
      static_cast<uint64_t>(Product(blocks-1,pageStride,"packed span")+payload)};
  // Raw page padding may alias the backing uint8 allocation, but never the
  // active compressed region. K/V/tags can interleave in the same page stride.
  for (const auto& raw:{rawKey,rawValue,tags}) {NoOverlap(Bytes(raw),packedActive);NoOverlap(raw,status);}
  NoOverlapPages(rawKey,rawValue,rawBytes,rawBytes);
  NoOverlapPages(rawKey,tags,rawBytes,tagBytes);NoOverlapPages(rawValue,tags,rawBytes,tagBytes);
  NoOverlap(Bytes(status),packedActive);
  for (const auto& source:{key,value,rk,rv,slots,positions}) {
    NoOverlap(Bytes(source),packedActive);NoOverlap(source,status);
    NoOverlap(source,rawKey);NoOverlap(source,rawValue);NoOverlap(source,tags);
  }
  if (n==0) return;
  const c10_npu::OptionalNPUGuard guard(key.device());
  oscar_ascend::rotate_clip_store_launch(c10_npu::getCurrentNPUStream().stream(),
      key.data_ptr(),value.data_ptr(),rk.data_ptr(),rv.data_ptr(),slots.data_ptr(),
      positions.data_ptr(),packed.data_ptr(),rawKey.data_ptr(),rawValue.data_ptr(),
      tags.data_ptr(),status.data_ptr(),n,h,d,dtype,blockTokens,blocks,offset,pageStride,
      rawKey.stride(0),rawValue.stride(0),tags.stride(0),sink,recent,
      static_cast<float>(kClip),static_cast<float>(vClip),hadamard,Cores(Product(n,h,"rows")));
}
}
TORCH_LIBRARY_FRAGMENT(oscar_ascend_ops,m) {
  m.def("rotate_out(Tensor input, Tensor rotation_transposed, Tensor(a!) output, "
        "Tensor(b!) status, bool hadamard=False, Tensor? slots=None) -> ()");
  m.def("rotate_clip_store_out(Tensor key, Tensor value, Tensor rk_transposed, "
        "Tensor rv_transposed, Tensor slots, Tensor positions, Tensor(a!) packed, "
        "Tensor(b!) raw_key, Tensor(c!) raw_value, Tensor(d!) raw_tags, Tensor(e!) status, "
        "int block_tokens, int blocks, int ssm_offset, int page_stride, int sink_tokens, "
        "int recent_capacity, float k_clip, float v_clip, bool hadamard=False) -> ()");
}
TORCH_LIBRARY_IMPL(oscar_ascend_ops,PrivateUse1,m) {
  m.impl("rotate_out",&Rotate);m.impl("rotate_clip_store_out",&RotateStore);
}
