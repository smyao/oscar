// SPDX-License-Identifier: Apache-2.0
// Archive G6/G18/#53/#98-110: strict schema, explicit linking, one load path.
// Native references/vllm-ascend/csrc/torch_binding.cpp:19-27,302-305 (headers/guard/stream).
#include <algorithm>
#include <limits>
#include <torch/extension.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include "include/oscar_launch.h"

namespace {
void Check(const at::Tensor& t, const at::Tensor& owner, at::ScalarType dtype,
           const char* name) {
  TORCH_CHECK(t.device().type()==c10::DeviceType::PrivateUse1,
              name," must be on NPU");
  TORCH_CHECK(t.device()==owner.device(),name," must share the input NPU");
  TORCH_CHECK(t.scalar_type()==dtype,name," has incorrect dtype");
  TORCH_CHECK(t.is_contiguous(),name," must be contiguous");
}
void CheckDim(int64_t d) {
  TORCH_CHECK(d==64 || d==128 || d==256,"head dimension must be 64/128/256");
}
uint32_t Cores(int64_t rows) {
  return static_cast<uint32_t>(std::min<int64_t>(32,std::max<int64_t>(1,rows)));
}
void NoOverlap(const at::Tensor& a,const at::Tensor& b) {
  if (a.numel()==0 || b.numel()==0) return;
  const uintptr_t first=reinterpret_cast<uintptr_t>(a.data_ptr());
  const uintptr_t second=reinterpret_cast<uintptr_t>(b.data_ptr());
  const uint64_t aBytes=static_cast<uint64_t>(a.numel())*a.element_size();
  const uint64_t bBytes=static_cast<uint64_t>(b.numel())*b.element_size();
  TORCH_CHECK(first<=second ? second-first>=aBytes : first-second>=bBytes,
              "AscendC out tensors must not overlap inputs or other outputs");
}
void Store(const at::Tensor& key,const at::Tensor& value,const at::Tensor& slots,
           at::Tensor raw,at::Tensor status,int64_t blockTokens,int64_t blocks,
           int64_t ssmOffset,int64_t pageStride) {
  Check(key,key,at::kFloat,"key_rot"); Check(value,key,at::kFloat,"value_rot");
  Check(raw,key,at::kByte,"raw"); Check(status,key,at::kInt,"status");
  TORCH_CHECK(slots.scalar_type()==at::kInt || slots.scalar_type()==at::kLong,
              "slot_mapping must be int32/int64");
  Check(slots,key,slots.scalar_type(),"slot_mapping");
  TORCH_CHECK(key.dim()==3 && value.sizes()==key.sizes(),"K/V shape must be [N,H,D]");
  const int64_t n=key.size(0), h=key.size(1), d=key.size(2); CheckDim(d);
  TORCH_CHECK(h>0 && slots.dim()==1 && slots.size(0)==n,"invalid token/head/slot dimensions");
  TORCH_CHECK(status.dim()==2 && status.size(0)==n && status.size(1)==h,
              "status must be int32 [N,H]");
  TORCH_CHECK(raw.dim()==1,"raw storage must be flat uint8");
  TORCH_CHECK(blockTokens>0 && blockTokens%128==0 && blocks>0 && ssmOffset>=0,
              "invalid physical page geometry");
  const int64_t limit=std::numeric_limits<int64_t>::max();
  const int64_t headBytes=d/2+8;
  TORCH_CHECK(h<=limit/headBytes && blockTokens<=limit/(h*headBytes),
              "physical row geometry overflows signed int64");
  TORCH_CHECK(blocks<=limit/blockTokens,"physical slot capacity overflows signed int64");
  TORCH_CHECK(pageStride>0 && pageStride>=blockTokens*h*headBytes,
              "physical page cannot fit packed rows");
  TORCH_CHECK(ssmOffset<=raw.numel() && blocks<=(raw.numel()-ssmOffset)/pageStride,
              "raw storage cannot cover all SSM pages");
  NoOverlap(raw,key); NoOverlap(raw,value); NoOverlap(raw,slots); NoOverlap(raw,status);
  NoOverlap(status,key); NoOverlap(status,value); NoOverlap(status,slots);
  if (n==0) return;
  const c10_npu::OptionalNPUGuard guard(key.device());
  oscar_ascend::store_int2_launch(c10_npu::getCurrentNPUStream().stream(),
      key.data_ptr(),value.data_ptr(),slots.data_ptr(),raw.data_ptr(),
      status.data_ptr(),n,h,d,blockTokens,blocks,ssmOffset,pageStride,
      slots.scalar_type()==at::kLong,Cores(n*h));
}
void Merge(const at::Tensor& partial,const at::Tensor& partialLse,
           at::Tensor output,at::Tensor lse,at::Tensor status) {
  Check(partial,partial,at::kFloat,"partial_out");
  Check(partialLse,partial,at::kFloat,"partial_lse");
  Check(output,partial,at::kFloat,"output"); Check(lse,partial,at::kFloat,"lse");
  Check(status,partial,at::kInt,"status");
  TORCH_CHECK(partial.dim()==3,"partial_out must be [R,S,D]");
  const int64_t r=partial.size(0),s=partial.size(1),d=partial.size(2); CheckDim(d);
  TORCH_CHECK(s>0 && s<=128,"split count must be in [1,128]");
  TORCH_CHECK(partialLse.dim()==2 && partialLse.size(0)==r && partialLse.size(1)==s,
              "partial_lse must be [R,S]");
  TORCH_CHECK(output.dim()==2 && output.size(0)==r && output.size(1)==d,
              "output must be [R,D]");
  TORCH_CHECK(lse.dim()==1 && lse.size(0)==r && status.dim()==1 && status.size(0)==r,
              "lse/status must be [R]");
  for (const auto& written : {output,lse,status}) {
    NoOverlap(written,partial); NoOverlap(written,partialLse);
  }
  NoOverlap(output,lse); NoOverlap(output,status); NoOverlap(lse,status);
  if (r==0) return;
  const c10_npu::OptionalNPUGuard guard(partial.device());
  oscar_ascend::merge_lse_launch(c10_npu::getCurrentNPUStream().stream(),
      partial.data_ptr(),partialLse.data_ptr(),output.data_ptr(),lse.data_ptr(),
      status.data_ptr(),r,s,d,Cores(r));
}
}
TORCH_LIBRARY(oscar_ascend_ops,m) {
  m.def("store_int2_out(Tensor key_rot, Tensor value_rot, Tensor slot_mapping, "
        "Tensor(a!) raw, Tensor(b!) status, int physical_block_tokens, "
        "int physical_num_blocks, int raw_ssm_offset, int physical_page_stride) -> ()");
  m.def("merge_lse_out(Tensor partial_out, Tensor partial_lse, Tensor(a!) output, "
        "Tensor(b!) lse, Tensor(c!) status) -> ()");
}
TORCH_LIBRARY_IMPL(oscar_ascend_ops,PrivateUse1,m) {
  m.impl("store_int2_out",&Store); m.impl("merge_lse_out",&Merge);
}
PYBIND11_MODULE(_oscar_ascend_ops,m) {
  m.def("abi_version",[]{return 1;});
  m.def("capabilities",[]{return std::vector<std::string>{"store_int2_out","merge_lse_out",
      "rotate_out","rotate_clip_store_out","prepare_attention_tasks_out","attention_cv_out",
      "attention_cv_profile_out","status_guard"};});
}
