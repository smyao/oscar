#pragma once

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(OscarInt2PagedAttentionTilingData)
TILING_DATA_FIELD_DEF(uint32_t, numTokens);
TILING_DATA_FIELD_DEF(uint32_t, numRequests);
TILING_DATA_FIELD_DEF(uint32_t, numQueryHeads);
TILING_DATA_FIELD_DEF(uint32_t, numKvHeads);
TILING_DATA_FIELD_DEF(uint32_t, gqaSize);
TILING_DATA_FIELD_DEF(uint32_t, headDim);
TILING_DATA_FIELD_DEF(uint32_t, blockSize);
TILING_DATA_FIELD_DEF(uint32_t, numBlocks);
TILING_DATA_FIELD_DEF(uint32_t, maxBlocksPerRequest);
TILING_DATA_FIELD_DEF(uint32_t, cacheSlotBytes);
TILING_DATA_FIELD_DEF(uint32_t, hasStage);
TILING_DATA_FIELD_DEF(uint32_t, stageRows);
TILING_DATA_FIELD_DEF(uint32_t, splitKv);
TILING_DATA_FIELD_DEF(uint32_t, maxSeqLen);
TILING_DATA_FIELD_DEF(uint32_t, kvTile);
TILING_DATA_FIELD_DEF(uint64_t, userWorkspaceBytes);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTiling);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(OscarInt2PagedAttention,
                           OscarInt2PagedAttentionTilingData)
}  // namespace optiling
