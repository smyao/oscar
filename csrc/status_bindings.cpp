// Archive #27/#44/#69/#108: one same-stream asynchronous device guard, no host tensor readback.
#include <torch/extension.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
extern "C" void oscar_status_guard_launch(void*,void*,void*,void*,void*,int64_t,int64_t,int64_t,int64_t);
namespace {
void Guard(at::Tensor a,const at::Tensor& b,const at::Tensor& c,const at::Tensor& d) {
    for(const auto& tensor:{a,b,c,d}) {
        TORCH_CHECK(tensor.device().type()==c10::DeviceType::PrivateUse1 && tensor.device()==a.device(),"status tensors must share NPU");
        TORCH_CHECK(tensor.scalar_type()==at::kInt && tensor.is_contiguous(),"status tensors must be contiguous int32");
    }
    const c10_npu::OptionalNPUGuard guard(a.device());
    oscar_status_guard_launch(c10_npu::getCurrentNPUStream().stream(),a.data_ptr(),b.data_ptr(),c.data_ptr(),d.data_ptr(),a.numel(),b.numel(),c.numel(),d.numel());
}
}
TORCH_LIBRARY_FRAGMENT(oscar_ascend_ops,m) {m.def("status_guard(Tensor(a!) attention_status, Tensor rotate_status, Tensor merge_status, Tensor store_status) -> ()");}
TORCH_LIBRARY_IMPL(oscar_ascend_ops,PrivateUse1,m) {m.impl("status_guard",&Guard);}
