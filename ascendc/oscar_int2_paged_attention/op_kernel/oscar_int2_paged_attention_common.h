#pragma once

#include "kernel_operator.h"

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
  uint32_t maxSeqLen;
  uint32_t kvTile;
  bool hasStage;
  float scale;
};

template <typename TilingData>
__aicore__ inline RuntimeShape ReadShape(const TilingData* tiling) {
  RuntimeShape s{};
  s.tokens = tiling->numTokens;
  s.requests = tiling->numRequests;
  s.queryHeads = tiling->numQueryHeads;
  s.kvHeads = tiling->numKvHeads;
  s.gqa = tiling->gqaSize;
  s.blockSize = tiling->blockSize;
  s.blocks = tiling->numBlocks;
  s.maxBlocks = tiling->maxBlocksPerRequest;
  s.slotBytes = tiling->cacheSlotBytes;
  s.hasStage = tiling->hasStage != 0;
  s.stageRows = tiling->stageRows;
  s.maxSeqLen = tiling->maxSeqLen;
  s.kvTile = tiling->kvTile;
  s.scale = tiling->scaleValue;
  return s;
}

}  // namespace OscarAscendC
