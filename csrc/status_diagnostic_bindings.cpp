// Archive #27/#94/#95/#118/#125/#157: opt-in diagnostics, original Trap retained.
// Native reference: references/vllm-ascend/csrc/torch_binding.cpp device_print:
// pinned D2H -> aclrtLaunchHostFunc on the same stream, payload survives replay.
// No AscendC kernel, DMA address, precision, status or inference output changes.
// D.4 four questions: (1) diagnoses the existing CV/history/window/current path;
// (2) does not revive the failed full-history BF16 dequantization route;
// (3) reads only statuses/metadata and copies 1 KiB, never model Q/K/V/history;
// (4) extra reductions/D2H/callback are debug overhead, not a speed optimization.
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <cstdio>
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>
#include <unistd.h>
#include "status_diagnostic_packet.h"
extern "C" void oscar_status_guard_launch(void*,void*,void*,void*,void*,int64_t,int64_t,int64_t,int64_t);
namespace {
struct Payload {
    at::Tensor host;
    std::string context;
    int pid, device;
    uint64_t invocation = 0;
};
struct Registry {
    std::mutex mutex;
    std::vector<std::unique_ptr<Payload>> entries;
};
Registry& GetRegistry() {
    // Graphs retain raw callback pointers. Process-lifetime ownership avoids
    // first-replay UAF and shutdown/destructor races. Bound retained memory.
    static auto* registry = new Registry;
    return *registry;
}
std::mutex& OutputMutex() { static auto* mutex = new std::mutex; return *mutex; }
void Report(void* opaque) {
    auto* p = static_cast<Payload*>(opaque);
    // No ACL/Python/device operations inside this callback. Each recorded call
    // owns its snapshot, including graphs replayed on a different stream.
    ++p->invocation;
    const auto message = oscar_diagnostic::ErrorLines(p->host.data_ptr<int64_t>(),
        p->context, p->pid, p->device, p->invocation);
    std::lock_guard<std::mutex> lock(OutputMutex());
    static std::atomic<bool> announced{false};
    if (!announced.exchange(true))
        std::fprintf(stderr, "OSCAR_STATUS_DIAGNOSTICS_ACTIVE pid=%d device=%d bytes=1024 context=%s\n",
                     p->pid, p->device, p->context.c_str());
    if (!message.empty()) std::fwrite(message.data(), 1, message.size(), stderr);
    std::fflush(stderr);  // complete before original same-stream Trap executes
}
void GuardDiagnostic(at::Tensor a, const at::Tensor& b, const at::Tensor& c,
                     const at::Tensor& d, const at::Tensor& report, const std::string& context) {
    for (const auto& tensor : {a,b,c,d}) {
        TORCH_CHECK(tensor.device().type()==c10::DeviceType::PrivateUse1 && tensor.device()==a.device(),
                    "status tensors must share NPU");
        TORCH_CHECK(tensor.scalar_type()==at::kInt && tensor.is_contiguous(),
                    "status tensors must be contiguous int32");
    }
    TORCH_CHECK(report.device()==a.device() && report.scalar_type()==at::kLong &&
                report.is_contiguous() && report.dim()==2 && report.size(0)==4 && report.size(1)==32,
                "status diagnostic report must be contiguous NPU int64 [4,32]");
    TORCH_CHECK(context.size() <= 4096, "status diagnostic context exceeds 4096 bytes");
    const c10_npu::OptionalNPUGuard guard(a.device());
    const auto stream = c10_npu::getCurrentNPUStream().stream();
    auto& registry = GetRegistry();
    Payload* p;
    {
        std::lock_guard<std::mutex> lock(registry.mutex);
        // Do NOT share snapshots between separately captured graphs, even if
        // they were recorded on the same stream: replay may use another stream.
        // Up to 64 MiB pinned per worker; this is a bounded short diagnostic,
        // never a long-running production/performance configuration.
        TORCH_CHECK(registry.entries.size() < 65536, "status diagnostic snapshot limit reached; restart debug worker");
        auto value = std::make_unique<Payload>();
        value->host = at::empty_like(report, report.options().device(at::kCPU).pinned_memory(true));
        value->context = context;
        value->pid = getpid(); value->device = a.device().index();
        p = value.get();
        registry.entries.push_back(std::move(value));
    }
    constexpr size_t bytes = 4 * 32 * sizeof(int64_t);
    const auto copied = aclrtMemcpyAsync(p->host.data_ptr(), bytes, report.data_ptr(), bytes,
                                        ACL_MEMCPY_DEVICE_TO_HOST, stream);
    TORCH_CHECK(copied == ACL_SUCCESS, "OSCAR diagnostic D2H enqueue failed: ", copied);
    const auto launched = aclrtLaunchHostFunc(stream, Report, p);
    TORCH_CHECK(launched == ACL_SUCCESS, "OSCAR diagnostic host callback enqueue failed: ", launched);
    // Keep snapshot + callback + guard inside ONE mutable dispatcher op: the
    // compiler cannot reorder the guard ahead of the evidence it destroys.
    oscar_status_guard_launch(stream,a.data_ptr(),b.data_ptr(),c.data_ptr(),d.data_ptr(),
                              a.numel(),b.numel(),c.numel(),d.numel());
}
}
TORCH_LIBRARY_FRAGMENT(oscar_ascend_ops,m) {
    m.def("status_guard_diagnostic(Tensor(a!) attention_status, Tensor rotate_status, Tensor merge_status, Tensor store_status, Tensor report, str context) -> ()");
}
TORCH_LIBRARY_IMPL(oscar_ascend_ops,PrivateUse1,m) {m.impl("status_guard_diagnostic",&GuardDiagnostic);}
