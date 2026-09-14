#include <cmath>

#include "kernel_operator.h"
#include "oscar_int2_paged_attention_common.h"
#include "oscar_int2_paged_attention_kvcache.h"

using namespace AscendC;
using namespace OscarAscendC;

namespace {
constexpr uint32_t MAX_GQA = 8;
constexpr uint32_t MAX_LANES = MAX_QUERY * MAX_GQA;

template <typename T>
class OscarInt2AttentionReference {
 public:
  __aicore__ inline void Init(
      GM_ADDR qRot, GM_ADDR kNew, GM_ADDR vNew, GM_ADDR kCache,
      GM_ADDR vCache, GM_ADDR blockTables, GM_ADDR qStarts, GM_ADDR qLens,
      GM_ADDR prefixes, GM_ADDR stageK, GM_ADDR stageV, GM_ADDR owner,
      GM_ADDR attentionOut,
      const optiling::OscarInt2PagedAttentionTilingData* tiling,
      TPipe* pipe) {
    shape_ = ReadShape(tiling);
    q_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(qRot));
    kNew_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(kNew));
    vNew_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(vNew));
    qStarts_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(qStarts));
    qLens_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(qLens));
    prefixes_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(prefixes));
    out_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(attentionOut));
    cache_.Init(kCache, vCache, blockTables, stageK, stageV, owner, shape_);
    pipe->InitBuffer(kBuf_, HEAD_DIM * sizeof(T));
    pipe->InitBuffer(vBuf_, HEAD_DIM * sizeof(T));
    pipe->InitBuffer(accBuf_, MAX_LANES * HEAD_DIM * sizeof(float));
    pipe->InitBuffer(stateBuf_, MAX_LANES * 2 * sizeof(float));
    kLocal_ = kBuf_.Get<T>();
    vLocal_ = vBuf_.Get<T>();
    acc_ = accBuf_.Get<float>();
    state_ = stateBuf_.Get<float>();
  }

  __aicore__ inline void Process() {
    if (shape_.gqa > MAX_GQA) return;
    const uint32_t workItems = shape_.requests * shape_.kvHeads;
    for (uint32_t work = GetBlockIdx(); work < workItems;
         work += GetBlockNum()) {
      const uint32_t request = work / shape_.kvHeads;
      const uint32_t kvHead = work % shape_.kvHeads;
      ProcessGroup(request, kvHead);
    }
  }

 private:
  __aicore__ inline void LoadFresh(uint32_t token, uint32_t kvHead) {
    const uint64_t base =
        (static_cast<uint64_t>(token) * shape_.kvHeads + kvHead) * HEAD_DIM;
    for (uint32_t d = 0; d < HEAD_DIM; ++d) {
      kLocal_.SetValue(d, kNew_.GetValue(base + d));
      vLocal_.SetValue(d, vNew_.GetValue(base + d));
    }
  }

  __aicore__ inline void ProcessGroup(uint32_t request, uint32_t kvHead) {
    const int32_t qStartSigned = qStarts_.GetValue(request);
    const int32_t qLenSigned = qLens_.GetValue(request);
    const int32_t prefixSigned = prefixes_.GetValue(request);
    if (qStartSigned < 0 || qLenSigned <= 0 || qLenSigned > MAX_QUERY ||
        prefixSigned < 0) return;
    const uint32_t qStart = static_cast<uint32_t>(qStartSigned);
    const uint32_t qLen = static_cast<uint32_t>(qLenSigned);
    const uint32_t prefix = static_cast<uint32_t>(prefixSigned);
    const uint32_t lanes = qLen * shape_.gqa;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      state_.SetValue(lane, -3.402823466e+38F);
      state_.SetValue(MAX_LANES + lane, 0.0F);
      for (uint32_t d = 0; d < HEAD_DIM; ++d)
        acc_.SetValue(lane * HEAD_DIM + d, 0.0F);
    }

    const uint32_t total = prefix + qLen;
    for (uint32_t pos = 0; pos < total; ++pos) {
      if (pos < prefix) {
        uint32_t block = 0;
        uint32_t offset = 0;
        if (!cache_.Resolve(request, pos, block, offset)) continue;
        cache_.Load(block, offset, kvHead, kLocal_, vLocal_);
      } else {
        LoadFresh(qStart + pos - prefix, kvHead);
      }
      for (uint32_t qi = 0; qi < qLen; ++qi) {
        if (pos > prefix + qi) continue;
        for (uint32_t group = 0; group < shape_.gqa; ++group) {
          const uint32_t lane = qi * shape_.gqa + group;
          const uint32_t qHead = kvHead * shape_.gqa + group;
          const uint64_t qBase =
              (static_cast<uint64_t>(qStart + qi) * shape_.queryHeads + qHead) *
              HEAD_DIM;
          float score = 0.0F;
          for (uint32_t d = 0; d < HEAD_DIM; ++d)
            score += static_cast<float>(q_.GetValue(qBase + d)) *
                     static_cast<float>(kLocal_.GetValue(d));
          score *= shape_.scale;
          const float oldMax = state_.GetValue(lane);
          const float newMax = score > oldMax ? score : oldMax;
          const float oldFactor = oldMax < -3.0e+38F ? 0.0F : expf(oldMax - newMax);
          const float weight = expf(score - newMax);
          const float oldSum = state_.GetValue(MAX_LANES + lane);
          state_.SetValue(lane, newMax);
          state_.SetValue(MAX_LANES + lane, oldSum * oldFactor + weight);
          for (uint32_t d = 0; d < HEAD_DIM; ++d) {
            const uint32_t index = lane * HEAD_DIM + d;
            acc_.SetValue(index, acc_.GetValue(index) * oldFactor +
                                     weight * static_cast<float>(vLocal_.GetValue(d)));
          }
        }
      }
    }
    for (uint32_t qi = 0; qi < qLen; ++qi) {
      for (uint32_t group = 0; group < shape_.gqa; ++group) {
        const uint32_t lane = qi * shape_.gqa + group;
        const uint32_t qHead = kvHead * shape_.gqa + group;
        const uint64_t outBase =
            (static_cast<uint64_t>(qStart + qi) * shape_.queryHeads + qHead) *
            HEAD_DIM;
        const float denom = state_.GetValue(MAX_LANES + lane);
        const float inverse = denom > 0.0F ? 1.0F / denom : 0.0F;
        for (uint32_t d = 0; d < HEAD_DIM; ++d)
          out_.SetValue(outBase + d,
                        static_cast<T>(acc_.GetValue(lane * HEAD_DIM + d) * inverse));
      }
    }
  }

  RuntimeShape shape_{};
  GlobalTensor<T> q_, kNew_, vNew_, out_;
  GlobalTensor<int32_t> qStarts_, qLens_, prefixes_;
  Int2KvCacheLoader<T> cache_;
  TBuf<TPosition::VECCALC> kBuf_, vBuf_, accBuf_, stateBuf_;
  LocalTensor<T> kLocal_, vLocal_;
  LocalTensor<float> acc_, state_;
};
}  // namespace

extern "C" __global__ __aicore__ void oscar_int2_paged_attention(
    GM_ADDR qRot, GM_ADDR kNew, GM_ADDR vNew, GM_ADDR kCache,
    GM_ADDR vCache, GM_ADDR blockTables, GM_ADDR qStarts, GM_ADDR qLens,
    GM_ADDR prefixes, GM_ADDR stageK, GM_ADDR stageV, GM_ADDR owner,
    GM_ADDR attentionOut, GM_ADDR workspace, GM_ADDR tiling) {
  TPipe pipe;
  GET_TILING_DATA(tilingData, tiling);
  if (TILING_KEY_IS(1)) {
    OscarInt2AttentionReference<bfloat16_t> op;
    op.Init(qRot, kNew, vNew, kCache, vCache, blockTables, qStarts, qLens,
            prefixes, stageK, stageV, owner, attentionOut, &tilingData, &pipe);
    op.Process();
  } else {
    OscarInt2AttentionReference<half> op;
    op.Init(qRot, kNew, vNew, kCache, vCache, blockTables, qStarts, qLens,
            prefixes, stageK, stageV, owner, attentionOut, &tilingData, &pipe);
    op.Process();
  }
}
