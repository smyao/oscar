#pragma once

#include "kernel_operator.h"
#include "../op_host/oscar_int2_paged_attention_tiling.h"

namespace OscarAscendC {

constexpr uint32_t HEAD_DIM = 256;
constexpr uint32_t VALUES_PER_BYTE = 4;
constexpr uint32_t PACKED_BYTES = HEAD_DIM / VALUES_PER_BYTE;
constexpr uint32_t K_META_OFFSET = 0;
constexpr uint32_t K_PACKED_OFFSET = 32;
constexpr uint32_t V_PACKED_OFFSET = 0;
constexpr uint32_t K_SCALE_OFFSET = 0;
constexpr uint32_t K_ZERO_OFFSET = 2;
constexpr uint32_t V_SCALE_OFFSET = 4;
constexpr uint32_t V_ZERO_OFFSET = 6;
constexpr uint32_t MAX_QUERY = 4;
constexpr uint32_t KV_TILE = 128;

struct RuntimeShape {
  uint32_t tokens;
  uint32_t requests;
  uint32_t queryHeads;
  uint32_t kvHeads;
  uint32_t gqa;
  uint32_t blockSize;
  uint32_t blocks;
  uint32_t maxBlocks;
  uint32_t slotBytes;
  uint32_t stageRows;
  bool hasStage;
  float scale;
};

__aicore__ inline RuntimeShape ReadShape(
    const optiling::OscarInt2PagedAttentionTilingData* tiling) {
  RuntimeShape s{};
  s.tokens = tiling->get_numTokens();
  s.requests = tiling->get_numRequests();
  s.queryHeads = tiling->get_numQueryHeads();
  s.kvHeads = tiling->get_numKvHeads();
  s.gqa = tiling->get_gqaSize();
  s.blockSize = tiling->get_blockSize();
  s.blocks = tiling->get_numBlocks();
  s.maxBlocks = tiling->get_maxBlocksPerRequest();
  s.slotBytes = tiling->get_cacheSlotBytes();
  s.hasStage = tiling->get_hasStage() != 0;
  s.stageRows = tiling->get_stageRows();
  s.scale = tiling->get_scaleValue();
  return s;
}

}  // namespace OscarAscendC
