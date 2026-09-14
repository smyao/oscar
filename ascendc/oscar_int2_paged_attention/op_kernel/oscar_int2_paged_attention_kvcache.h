#pragma once

#include "oscar_int2_paged_attention_common.h"

namespace OscarAscendC {

// Loads one logical K/V vector from OSCAR's split physical cache.  Metadata
// lives in the first eight bytes of the K slot, K payload at K+32 and V
// payload at V+0.  This is deliberately one loader for both tensors so scale,
// zero, page and staging tag are fetched only once per token/KV-head.
template <typename T>
class Int2KvCacheLoader {
 public:
  __aicore__ inline void Init(
      GM_ADDR kCache, GM_ADDR vCache, GM_ADDR blockTables,
      GM_ADDR stageK, GM_ADDR stageV, GM_ADDR owner,
      const RuntimeShape& shape) {
    shape_ = shape;
    kCache_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(kCache));
    vCache_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(vCache));
    blockTables_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(blockTables));
    if (shape_.hasStage) {
      stageK_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(stageK));
      stageV_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(stageV));
      owner_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(owner));
    }
  }

  __aicore__ inline bool Resolve(uint32_t request, uint32_t position,
                                 uint32_t& block, uint32_t& offset) const {
    const uint32_t page = position / shape_.blockSize;
    if (request >= shape_.requests || page >= shape_.maxBlocks) return false;
    const int32_t signedBlock =
        blockTables_.GetValue(request * shape_.maxBlocks + page);
    if (signedBlock < 0 || static_cast<uint32_t>(signedBlock) >= shape_.blocks)
      return false;
    block = static_cast<uint32_t>(signedBlock);
    offset = position - page * shape_.blockSize;
    return true;
  }

  __aicore__ inline bool StageHit(uint32_t block, uint32_t offset) const {
    if (!shape_.hasStage || shape_.stageRows == 0) return false;
    const uint32_t row = block % shape_.stageRows;
    return owner_.GetValue(row * shape_.blockSize + offset) ==
           static_cast<int64_t>(block);
  }

  __aicore__ inline void Load(uint32_t block, uint32_t offset,
                              uint32_t kvHead, LocalTensor<T> k,
                              LocalTensor<T> v) const {
    if (StageHit(block, offset)) {
      LoadStage(block, offset, kvHead, k, v);
      return;
    }
    const uint64_t slot =
        (static_cast<uint64_t>(block) * shape_.blockSize + offset) *
            shape_.kvHeads * shape_.slotBytes +
        static_cast<uint64_t>(kvHead) * shape_.slotBytes;
    const float ks = DecodeHalf(slot + K_SCALE_OFFSET);
    const float kz = DecodeHalf(slot + K_ZERO_OFFSET);
    const float vs = DecodeHalf(slot + V_SCALE_OFFSET);
    const float vz = DecodeHalf(slot + V_ZERO_OFFSET);
    for (uint32_t packed = 0; packed < PACKED_BYTES; ++packed) {
      const uint8_t kb = kCache_.GetValue(slot + K_PACKED_OFFSET + packed);
      const uint8_t vb = vCache_.GetValue(slot + V_PACKED_OFFSET + packed);
      const uint32_t d = packed * VALUES_PER_BYTE;
      k.SetValue(d + 0, static_cast<T>(static_cast<float>(kb & 3U) * ks + kz));
      k.SetValue(d + 1, static_cast<T>(static_cast<float>((kb >> 2) & 3U) * ks + kz));
      k.SetValue(d + 2, static_cast<T>(static_cast<float>((kb >> 4) & 3U) * ks + kz));
      k.SetValue(d + 3, static_cast<T>(static_cast<float>((kb >> 6) & 3U) * ks + kz));
      v.SetValue(d + 0, static_cast<T>(static_cast<float>(vb & 3U) * vs + vz));
      v.SetValue(d + 1, static_cast<T>(static_cast<float>((vb >> 2) & 3U) * vs + vz));
      v.SetValue(d + 2, static_cast<T>(static_cast<float>((vb >> 4) & 3U) * vs + vz));
      v.SetValue(d + 3, static_cast<T>(static_cast<float>((vb >> 6) & 3U) * vs + vz));
    }
  }

 private:
  __aicore__ inline float DecodeHalf(uint64_t byteOffset) const {
    union HalfBits {
      uint16_t bits;
      half value;
    } decoded;
    decoded.bits = static_cast<uint16_t>(kCache_.GetValue(byteOffset)) |
                   (static_cast<uint16_t>(kCache_.GetValue(byteOffset + 1)) << 8);
    return static_cast<float>(decoded.value);
  }

  __aicore__ inline void LoadStage(uint32_t block, uint32_t offset,
                                   uint32_t kvHead, LocalTensor<T> k,
                                   LocalTensor<T> v) const {
    const uint32_t row = block % shape_.stageRows;
    const uint64_t base =
        (static_cast<uint64_t>(row) * shape_.blockSize + offset) *
            shape_.kvHeads * HEAD_DIM +
        static_cast<uint64_t>(kvHead) * HEAD_DIM;
    for (uint32_t d = 0; d < HEAD_DIM; ++d) {
      k.SetValue(d, stageK_.GetValue(base + d));
      v.SetValue(d, stageV_.GetValue(base + d));
    }
  }

  RuntimeShape shape_{};
  GlobalTensor<uint8_t> kCache_;
  GlobalTensor<uint8_t> vCache_;
  GlobalTensor<int32_t> blockTables_;
  GlobalTensor<float> stageK_;
  GlobalTensor<float> stageV_;
  GlobalTensor<int64_t> owner_;
};

}  // namespace OscarAscendC
