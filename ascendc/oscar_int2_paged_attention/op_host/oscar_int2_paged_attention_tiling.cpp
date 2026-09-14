#include "oscar_int2_paged_attention_tiling.h"

#include <algorithm>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr uint32_t kHeadDim = 256;
constexpr uint32_t kMaxQuery = 4;
constexpr uint32_t kTargetKvPerSplit = 1024;

uint32_t DtypeBytes(ge::DataType dtype) {
  return dtype == ge::DT_INT8 ? 1U : 2U;
}

ge::graphStatus Tiling(gert::TilingContext* context) {
  if (context == nullptr || context->GetAttrs() == nullptr) return ge::GRAPH_FAILED;
  const auto* q = context->GetInputShape(0);
  const auto* kc = context->GetInputShape(3);
  const auto* bt = context->GetInputShape(5);
  const auto* ql = context->GetInputShape(7);
  const auto* stage = context->GetInputShape(9);
  if (q == nullptr || kc == nullptr || bt == nullptr || ql == nullptr || stage == nullptr)
    return ge::GRAPH_FAILED;
  const auto& qs = q->GetStorageShape();
  const auto& ks = kc->GetStorageShape();
  const auto& bs = bt->GetStorageShape();
  if (qs.GetDimNum() != 3 || ks.GetDimNum() != 4 || bs.GetDimNum() != 2)
    return ge::GRAPH_FAILED;

  const uint32_t tokens = static_cast<uint32_t>(qs.GetDim(0));
  const uint32_t hq = static_cast<uint32_t>(qs.GetDim(1));
  const uint32_t d = static_cast<uint32_t>(qs.GetDim(2));
  const uint32_t blocks = static_cast<uint32_t>(ks.GetDim(0));
  const uint32_t blockSize = static_cast<uint32_t>(ks.GetDim(1));
  const uint32_t hk = static_cast<uint32_t>(ks.GetDim(2));
  const uint32_t requests = static_cast<uint32_t>(ql->GetStorageShape().GetShapeSize());
  if (d != kHeadDim || hk == 0 || hq == 0 || hq % hk != 0 || requests == 0 ||
      tokens == 0 || blockSize == 0 || blocks == 0) return ge::GRAPH_FAILED;

  const int64_t* attrHk = context->GetAttrs()->GetInt(1);
  const int64_t* attrD = context->GetAttrs()->GetInt(2);
  const float* scale = context->GetAttrs()->GetFloat(0);
  if (attrHk == nullptr || attrD == nullptr || scale == nullptr ||
      *attrHk != hk || *attrD != d) return ge::GRAPH_FAILED;

  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t cores = platform.GetCoreNum();
  if (cores == 0) return ge::GRAPH_FAILED;
  const uint32_t workItems = requests * hk;
  context->SetBlockDim(std::min(cores, std::max(1U, workItems)));
  context->SetTilingKey(0);

  OscarInt2PagedAttentionTilingData data;
  data.set_numTokens(tokens);
  data.set_numRequests(requests);
  data.set_numQueryHeads(hq);
  data.set_numKvHeads(hk);
  data.set_gqaSize(hq / hk);
  data.set_headDim(d);
  data.set_blockSize(blockSize);
  data.set_numBlocks(blocks);
  data.set_maxBlocksPerRequest(static_cast<uint32_t>(bs.GetDim(1)));
  data.set_cacheSlotBytes(static_cast<uint32_t>(ks.GetDim(3)) *
                          DtypeBytes(context->GetInputDesc(3)->GetDataType()));
  const bool hasStage = stage->GetStorageShape().GetShapeSize() > 0;
  data.set_hasStage(hasStage ? 1U : 0U);
  data.set_stageRows(hasStage ? static_cast<uint32_t>(stage->GetStorageShape().GetDim(0)) : 0U);
  data.set_splitKv(1U);
  data.set_scaleValue(*scale);
  size_t* workspace = context->GetWorkspaceSizes(1);
  workspace[0] = platform.GetLibApiWorkSpaceSize();
  data.SaveToBuffer(context->GetRawTilingData()->GetData(),
                    context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(data.GetDataSize());
  return ge::GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_OPTILING(OscarInt2PagedAttention).Tiling(Tiling);
}  // namespace optiling
