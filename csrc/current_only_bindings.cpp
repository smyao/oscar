// SPDX-License-Identifier: Apache-2.0
// Archive G5/G6/G30-G34/#91/#125/#145 and startup D.4: optional strict
// current-only BF16/FP32 out ABI. Compile/bind does not enable the route.
// No fallback for a non-FP32 vendor LSE, unaligned output, or unavailable
// device vector-core count. Fresh-only proof remains in the Python dispatcher.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <acl/acl_rt.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include "include/oscar_current_only_launch.h"

namespace {
void Check(const at::Tensor& value, const at::Tensor& owner,
           at::ScalarType dtype, const char* name) {
  TORCH_CHECK(value.device().type()==c10::DeviceType::PrivateUse1 &&
      value.device()==owner.device(), name, " must share the input NPU");
  TORCH_CHECK(value.scalar_type()==dtype, name, " has incorrect dtype");
  TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
}
void Disjoint(const at::Tensor& first, const at::Tensor& second) {
  if (!first.numel() || !second.numel()) return;
  const uintptr_t a=reinterpret_cast<uintptr_t>(first.data_ptr());
  const uintptr_t b=reinterpret_cast<uintptr_t>(second.data_ptr());
  const uint64_t an=static_cast<uint64_t>(first.numel())*first.element_size();
  const uint64_t bn=static_cast<uint64_t>(second.numel())*second.element_size();
  TORCH_CHECK(a<=b ? b-a>=an : a-b>=bn,
      "current-only input/output tensors must not overlap");
}
uint32_t VectorCores(const at::Tensor& owner,int64_t rows) {
  const int index=owner.device().index();
  TORCH_CHECK(index>=0,"selected NPU has no device index");
  // Cache only a successful query. An ACL failure remains visible on every
  // attempt, while ordinary 16-layer FULL passes pay no repeated host query.
  static std::mutex lock;
  static std::unordered_map<int,uint32_t> byDevice;
  uint32_t available=0;
  {
    std::lock_guard<std::mutex> guard(lock);
    const auto known=byDevice.find(index);
    if (known!=byDevice.end()) {
      available=known->second;
    } else {
      int64_t queried=0;
      const aclError rc=aclrtGetDeviceInfo(static_cast<uint32_t>(index),
          ACL_DEV_ATTR_VECTOR_CORE_NUM,&queried);
      TORCH_CHECK(rc==ACL_SUCCESS && queried>0 &&
          queried<=std::numeric_limits<uint32_t>::max(),
          "cannot query the selected NPU vector-core count; ACL rc=",rc);
      available=static_cast<uint32_t>(queried);
      byDevice.emplace(index,available);
    }
  }
  const int64_t batches=rows/16+(rows%16!=0);
  return static_cast<uint32_t>(std::min<int64_t>(available,batches));
}
void CopyValidateCurrent(const at::Tensor& currentOut,
    const at::Tensor& currentLse,at::Tensor output,at::Tensor outputLse,
    at::Tensor rowStatus) {
  Check(currentOut,currentOut,at::kBFloat16,"current_out");
  Check(currentLse,currentOut,at::kFloat,"current_lse");
  Check(output,currentOut,at::kBFloat16,"output");
  Check(outputLse,currentOut,at::kFloat,"output_lse");
  Check(rowStatus,currentOut,at::kInt,"row_status");
  TORCH_CHECK(currentOut.dim()==3,"current_out must be BF16[N,H,D]");
  const int64_t n=currentOut.size(0),h=currentOut.size(1),d=currentOut.size(2);
  TORCH_CHECK(h>0 && (d==64 || d==128 || d==256),
      "current-only requires H>0 and D in 64/128/256");
  TORCH_CHECK(n<=std::numeric_limits<int64_t>::max()/h,
      "current-only row count overflows int64");
  TORCH_CHECK(currentLse.dim()==3 && currentLse.size(0)==n &&
      currentLse.size(1)==h && currentLse.size(2)==1,
      "current_lse must be FP32[N,H,1]");
  TORCH_CHECK(output.sizes()==currentOut.sizes(),
      "output must be BF16[N,H,D]");
  TORCH_CHECK(outputLse.dim()==2 && outputLse.size(0)==n &&
      outputLse.size(1)==h,"output_lse must be FP32[N,H]");
  TORCH_CHECK(rowStatus.sizes()==outputLse.sizes(),
      "row_status must be int32[N,H]");
  for (const auto& written:{output,outputLse,rowStatus}) {
    Disjoint(written,currentOut);Disjoint(written,currentLse);
  }
  Disjoint(output,outputLse);Disjoint(output,rowStatus);
  Disjoint(outputLse,rowStatus);
  if (n==0) return;
  TORCH_CHECK(reinterpret_cast<uintptr_t>(outputLse.data_ptr())%64==0 &&
      reinterpret_cast<uintptr_t>(rowStatus.data_ptr())%64==0,
      "current-only FP32 LSE/status outputs must start on 64-byte lines");
  const c10_npu::OptionalNPUGuard guard(currentOut.device());
  const uint32_t cores=VectorCores(currentOut,n*h);
  oscar_ascend::copy_validate_current_launch(
      c10_npu::getCurrentNPUStream().stream(),currentOut.data_ptr(),
      currentLse.data_ptr(),output.data_ptr(),outputLse.data_ptr(),
      rowStatus.data_ptr(),n*h,d,cores);
}
}

TORCH_LIBRARY_FRAGMENT(oscar_ascend_ops,m) {
  m.def("copy_validate_current_out(Tensor current_out, Tensor current_lse, "
        "Tensor(a!) output, Tensor(b!) output_lse, Tensor(c!) row_status) -> ()");
}
TORCH_LIBRARY_IMPL(oscar_ascend_ops,PrivateUse1,m) {
  m.impl("copy_validate_current_out",&CopyValidateCurrent);
}
