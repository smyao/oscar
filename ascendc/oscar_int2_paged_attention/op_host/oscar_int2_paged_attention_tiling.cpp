#include "oscar_int2_paged_attention_tiling.h"

#include <algorithm>
#include <limits>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {
namespace {
constexpr uint32_t kHeadDim = 256;
constexpr uint32_t kMaxQuery = 4;
constexpr uint32_t kTargetKvPerSplit = 1024;
constexpr uint32_t kLongThreshold = 256;
constexpr uint32_t kRowsPerItem = 16;
constexpr uint32_t kCubeM = 16;
constexpr uint32_t kCubeN = 256;
constexpr uint64_t kWorkspacePerItem =
    (kRowsPerItem * kHeadDim + 2 * kCubeN * kHeadDim +
     kRowsPerItem * kCubeN) * sizeof(uint16_t) +
    2ULL * kRowsPerItem * kCubeN * sizeof(float);

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
  const int64_t* maxSeqLen = context->GetAttrs()->GetInt(3);
  if (attrHk == nullptr || attrD == nullptr || scale == nullptr ||
      maxSeqLen == nullptr || *attrHk != hk || *attrD != d ||
      *maxSeqLen <= 0) return ge::GRAPH_FAILED;

  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t cores = platform.GetCoreNum();
  if (cores == 0) return ge::GRAPH_FAILED;
  const uint32_t gqa = hq / hk;
  if (gqa > 8U) return ge::GRAPH_FAILED;
  const uint32_t gqaGroups = (gqa + 3U) / 4U;
  const uint64_t workItems64 =
      static_cast<uint64_t>(requests) * hk * gqaGroups;
  if (workItems64 == 0 ||
      workItems64 > std::numeric_limits<uint32_t>::max()) {
    return ge::GRAPH_FAILED;
  }
  const uint32_t workItems = static_cast<uint32_t>(workItems64);
  // KFC Matmul clients from two GQA tiles of the same request/KV head must
  // not execute concurrently on CANN 9.1: they share the operator's system
  // workspace/event channels and corrupt each other's Cube result.  Launch at
  // most one AIV client per request/KV-head; that client processes its GQA
  // tiles serially.  User workspace remains per tile to keep addresses
  // disjoint and to preserve the flattened device work mapping.
  const uint32_t concurrentItems = requests * hk;
  context->SetBlockDim(std::min(cores, std::max(1U, concurrentItems)));
  const bool longContext = static_cast<uint64_t>(*maxSeqLen) > kLongThreshold;
  context->SetTilingKey(longContext ? 2 : 0);

  using namespace matmul_tiling;
  MatmulApiTiling cube(platform);
  // Host-side matmul tiling owns a separate enum namespace from the
  // device-side AscendC API.  CANN 9.1 deliberately does not expose
  // AscendC::TPosition/CubeFormat to op_host translation units.
  cube.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                matmul_tiling::DataType::DT_FLOAT16);
  cube.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                matmul_tiling::DataType::DT_FLOAT16);
  cube.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                matmul_tiling::DataType::DT_FLOAT);
  cube.SetBias(false);
  cube.SetShape(kCubeM, kCubeN, kHeadDim);
  // CANN 9.1 reliably exposes one 16-row result per KFC client on 910B.  A
  // work item therefore owns four GQA heads (q<=4 => M<=16); GQA=8 is split
  // into two independently scheduled work items.
  cube.SetFixSplit(kCubeM, 128, 128);
  cube.SetOrgShape(kCubeM, kCubeN, kHeadDim);
  cube.SetBufferSpace(-1, -1, -1);

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
  data.set_maxSeqLen(static_cast<uint32_t>(*maxSeqLen));
  data.set_kvTile(kCubeN);
  if (workItems64 > std::numeric_limits<uint64_t>::max() /
                        kWorkspacePerItem) {
    return ge::GRAPH_FAILED;
  }
  const uint64_t workspaceBytes = workItems64 * kWorkspacePerItem;
  data.set_userWorkspaceBytes(workspaceBytes);
  data.set_scaleValue(*scale);
  if (cube.GetTiling(data.cubeTiling) == -1) return ge::GRAPH_FAILED;
  size_t* workspace = context->GetWorkspaceSizes(1);
  workspace[0] = platform.GetLibApiWorkSpaceSize() + workspaceBytes;
  data.SaveToBuffer(context->GetRawTilingData()->GetData(),
                    context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(data.GetDataSize());
  return ge::GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_OPTILING(OscarInt2PagedAttention).Tiling(Tiling);
}  // namespace optiling
