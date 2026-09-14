#include <torch/library.h>
#include <torch/types.h>

#include "aclnn_torch_adapter/op_api_common.h"

// op_api_common.h declares these buffers extern because vLLM-Ascend normally
// defines them in its monolithic extension.  This binding is intentionally
// standalone, so it owns one thread-local copy.
thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;

namespace oscar_ascend {

at::Tensor int2_paged_attention(
    const at::Tensor& q, const at::Tensor& kNew, const at::Tensor& vNew,
    const at::Tensor& kCache, const at::Tensor& vCache,
    const at::Tensor& blockTables, const at::Tensor& qStarts,
    const at::Tensor& qLens, const at::Tensor& prefixes,
    const at::Tensor& stageK, const at::Tensor& stageV,
    const at::Tensor& owner, double scaleValue, int64_t numKvHeads,
    int64_t headDim) {
  TORCH_CHECK(q.scalar_type() == at::kHalf,
              "OscarInt2PagedAttention device ABI requires FP16 Q/K/V");
  TORCH_CHECK(kNew.scalar_type() == at::kHalf &&
                  vNew.scalar_type() == at::kHalf,
              "OscarInt2PagedAttention fresh K/V must be FP16");
  auto out = at::empty_like(q);
  EXEC_NPU_CMD(aclnnOscarInt2PagedAttention,
               q, kNew, vNew, kCache, vCache, blockTables, qStarts, qLens,
               prefixes, stageK, stageV, owner, scaleValue, numKvHeads,
               headDim, out);
  return out;
}

}  // namespace oscar_ascend

TORCH_LIBRARY(oscar_ascend, m) {
  m.def("int2_paged_attention(Tensor q, Tensor k_new, Tensor v_new, "
        "Tensor k_cache, Tensor v_cache, Tensor block_tables, "
        "Tensor q_starts, Tensor q_lens, Tensor prefixes, Tensor stage_k, "
        "Tensor stage_v, Tensor owner, float scale_value, int num_kv_heads, "
        "int head_dim) -> Tensor");
}

TORCH_LIBRARY_IMPL(oscar_ascend, PrivateUse1, m) {
  m.impl("int2_paged_attention", &oscar_ascend::int2_paged_attention);
}
