#pragma once

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "oscar_int2_paged_attention_common.h"
#include "oscar_int2_paged_attention_kvcache.h"

namespace OscarAscendC {

constexpr uint32_t LONG_M = 32;
constexpr uint32_t LONG_CUBE_M = 16;
constexpr uint32_t LONG_N = 256;
constexpr uint32_t LONG_Q_ELEMS = LONG_M * HEAD_DIM;
constexpr uint32_t LONG_KV_ELEMS = LONG_N * HEAD_DIM;
constexpr uint32_t LONG_MATRIX_ELEMS = LONG_M * LONG_N;
constexpr uint64_t LONG_Q_BYTES = LONG_Q_ELEMS * sizeof(half);
constexpr uint64_t LONG_KV_BYTES = LONG_KV_ELEMS * sizeof(half);
constexpr uint64_t LONG_SCORE_BYTES = LONG_MATRIX_ELEMS * sizeof(float);
constexpr uint64_t LONG_PROB_BYTES = LONG_MATRIX_ELEMS * sizeof(half);
constexpr uint64_t LONG_PV_BYTES = LONG_MATRIX_ELEMS * sizeof(float);
constexpr uint64_t LONG_WORK_BYTES =
    LONG_Q_BYTES + 2 * LONG_KV_BYTES + LONG_SCORE_BYTES +
    LONG_PROB_BYTES + LONG_PV_BYTES;

// CubeFormat is intentionally a global device-side enum in CANN 9.1, while
// TPosition and Matmul live in AscendC.  Use the public Matmul interface here;
// Avoid internal implementation aliases that are not shipped by every CANN SDK.
using LongA = AscendC::MatmulType<AscendC::TPosition::GM,
                                  CubeFormat::ND, half, false>;
using LongBK = AscendC::MatmulType<AscendC::TPosition::GM,
                                   CubeFormat::ND, half, true>;
using LongBV = AscendC::MatmulType<AscendC::TPosition::GM,
                                   CubeFormat::ND, half, false>;
using LongC = AscendC::MatmulType<AscendC::TPosition::GM,
                                  CubeFormat::ND, float, false>;
using LongBias = AscendC::MatmulType<AscendC::TPosition::GM,
                                     CubeFormat::ND, half, false>;
using LongQkMatmul = AscendC::Matmul<LongA, LongBK, LongC, LongBias>;
using LongPvMatmul = AscendC::Matmul<LongA, LongBV, LongC, LongBias>;

// Long-context implementation.  One work item owns a request/KV-head pair;
// all q<=4 rows and its complete GQA group share every decompressed KV tile.
template <class QkMatmul, class PvMatmul, class TilingData>
class OscarInt2AttentionLong {
 public:
  __aicore__ inline OscarInt2AttentionLong(QkMatmul& qk, PvMatmul& pv)
      : qk_(qk), pv_(pv) {}

  __aicore__ inline void Init(
      GM_ADDR qRot, GM_ADDR kNew, GM_ADDR vNew, GM_ADDR kCache,
      GM_ADDR vCache, GM_ADDR blockTables, GM_ADDR qStarts, GM_ADDR qLens,
      GM_ADDR prefixes, GM_ADDR stageK, GM_ADDR stageV, GM_ADDR owner,
      GM_ADDR attentionOut, GM_ADDR userWorkspace, const TilingData* tiling,
      AscendC::TPipe* pipe) {
    shape_ = ReadShape(tiling);
    q_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(qRot));
    kNew_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(kNew));
    vNew_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(vNew));
    qStarts_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(qStarts));
    qLens_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(qLens));
    prefixes_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(prefixes));
    out_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(attentionOut));
    workspaceAddr_ = reinterpret_cast<__gm__ uint8_t*>(userWorkspace);
    cache_.Init(kCache, vCache, blockTables, stageK, stageV, owner, shape_);
    pipe->InitBuffer(kBuf_, HEAD_DIM * sizeof(half));
    pipe->InitBuffer(vBuf_, HEAD_DIM * sizeof(half));
    pipe->InitBuffer(accBuf_, LONG_M * HEAD_DIM * sizeof(float));
    pipe->InitBuffer(stateBuf_, LONG_M * 2 * sizeof(float));
    pipe->InitBuffer(expBuf_, 32);
    pipe->InitBuffer(rowExpBuf_, LONG_N * sizeof(float));
    kLocal_ = kBuf_.Get<half>();
    vLocal_ = vBuf_.Get<half>();
    acc_ = accBuf_.Get<float>();
    state_ = stateBuf_.Get<float>();
    exp_ = expBuf_.Get<float>();
    rowExp_ = rowExpBuf_.Get<float>();
  }

  __aicore__ inline void Process() {
    const uint32_t workItems = shape_.requests * shape_.kvHeads;
    for (uint32_t work = AscendC::GetBlockIdx(); work < workItems;
         work += AscendC::GetBlockNum()) {
      ProcessGroup(work, work / shape_.kvHeads, work % shape_.kvHeads);
    }
    qk_.End();
    pv_.End();
  }

 private:
  __aicore__ inline float DeviceExp(float value) {
    AscendC::Duplicate(exp_, value, 8);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(exp_, exp_, 8);
    const event_t vs = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
    AscendC::SetFlag<AscendC::HardEvent::V_S>(vs);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(vs);
    const float result = exp_.GetValue(0);
    const event_t sv = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
    AscendC::SetFlag<AscendC::HardEvent::S_V>(sv);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(sv);
    return result;
  }

  __aicore__ inline float PrepareProbabilityRow(
      uint32_t row, uint32_t qi, uint32_t tileStart, uint32_t total,
      float newMax) {
    // Build one FP32 score row with causal/tail masking, then issue a single
    // 256-wide vector Exp.  This replaces 256 scalar-sized Exp launches per
    // row/tile, which is the dominant small-op cost at 16K-32K.
    for (uint32_t col = 0; col < LONG_N; ++col) {
      const bool valid = tileStart + col < total &&
                         tileStart + col <= statePrefix_ + qi;
      const float value = valid
          ? scoreWork_.GetValue(row * LONG_N + col) * shape_.scale - newMax
          : -3.402823466e+38F;
      rowExp_.SetValue(col, value);
    }
    const event_t scalarToVector = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
    AscendC::SetFlag<AscendC::HardEvent::S_V>(scalarToVector);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(scalarToVector);
    AscendC::Exp(rowExp_, rowExp_, LONG_N);
    const event_t vectorToScalar = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
    AscendC::SetFlag<AscendC::HardEvent::V_S>(vectorToScalar);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(vectorToScalar);
    float sum = 0.0F;
    for (uint32_t col = 0; col < LONG_N; ++col) {
      const float weight = rowExp_.GetValue(col);
      probWork_.SetValue(row * LONG_N + col, static_cast<half>(weight));
      sum += weight;
    }
    return sum;
  }

  __aicore__ inline void BindWorkspace(uint32_t work) {
    const uint64_t base = static_cast<uint64_t>(work) * LONG_WORK_BYTES;
    uint64_t offset = base;
    qWork_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(workspaceAddr_ + offset));
    offset += LONG_Q_BYTES;
    kWork_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(workspaceAddr_ + offset));
    offset += LONG_KV_BYTES;
    vWork_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(workspaceAddr_ + offset));
    offset += LONG_KV_BYTES;
    scoreWork_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspaceAddr_ + offset));
    offset += LONG_SCORE_BYTES;
    probWork_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(workspaceAddr_ + offset));
    offset += LONG_PROB_BYTES;
    pvWork_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspaceAddr_ + offset));
  }

  __aicore__ inline void LoadFresh(uint32_t token, uint32_t kvHead) {
    const uint64_t base =
        (static_cast<uint64_t>(token) * shape_.kvHeads + kvHead) * HEAD_DIM;
    for (uint32_t d = 0; d < HEAD_DIM; ++d) {
      kLocal_.SetValue(d, kNew_.GetValue(base + d));
      vLocal_.SetValue(d, vNew_.GetValue(base + d));
    }
  }

  __aicore__ inline void PrepareQueries(uint32_t qStart, uint32_t qLen,
                                        uint32_t kvHead) {
    const uint32_t lanes = qLen * shape_.gqa;
    for (uint32_t row = 0; row < LONG_M; ++row) {
      const uint32_t qi = row / shape_.gqa;
      const uint32_t group = row % shape_.gqa;
      const bool valid = row < lanes;
      const uint32_t qHead = kvHead * shape_.gqa + group;
      const uint64_t src =
          (static_cast<uint64_t>(qStart + qi) * shape_.queryHeads + qHead) *
          HEAD_DIM;
      for (uint32_t d = 0; d < HEAD_DIM; ++d)
        qWork_.SetValue(row * HEAD_DIM + d,
                        valid ? q_.GetValue(src + d) : half(0));
    }
  }

  __aicore__ inline void PrepareKvTile(uint32_t request, uint32_t kvHead,
                                       uint32_t qStart, uint32_t prefix,
                                       uint32_t total, uint32_t tileStart) {
    for (uint32_t col = 0; col < LONG_N; ++col) {
      const uint32_t pos = tileStart + col;
      bool valid = pos < total;
      if (valid && pos < prefix) {
        uint32_t block = 0, offset = 0;
        valid = cache_.Resolve(request, pos, block, offset);
        if (valid) cache_.Load(block, offset, kvHead, kLocal_, vLocal_);
      } else if (valid) {
        LoadFresh(qStart + pos - prefix, kvHead);
      }
      for (uint32_t d = 0; d < HEAD_DIM; ++d) {
        kWork_.SetValue(col * HEAD_DIM + d,
                        valid ? kLocal_.GetValue(d) : half(0));
        vWork_.SetValue(col * HEAD_DIM + d,
                        valid ? vLocal_.GetValue(d) : half(0));
      }
    }
  }

  __aicore__ inline void CubeQk() {
    qk_.SetOrgShape(LONG_CUBE_M, LONG_N, HEAD_DIM);
    qk_.SetSingleShape(LONG_CUBE_M, LONG_N, HEAD_DIM);
    for (uint32_t batch = 0; batch < LONG_M / LONG_CUBE_M; ++batch) {
      const uint32_t row = batch * LONG_CUBE_M;
      qk_.SetTensorA(qWork_[row * HEAD_DIM], false);
      qk_.SetTensorB(kWork_, true);
      // Submit one stable 16-row Cube program.  Both programs reuse kWork_, so
      // compressed historical KV is still loaded exactly once per KV tile.
      qk_.template IterateAll<false>(scoreWork_[row * LONG_N], 0, false, true);
      qk_.WaitIterateAll();
    }
  }

  __aicore__ inline void CubePv() {
    pv_.SetOrgShape(LONG_CUBE_M, HEAD_DIM, LONG_N);
    pv_.SetSingleShape(LONG_CUBE_M, HEAD_DIM, LONG_N);
    for (uint32_t batch = 0; batch < LONG_M / LONG_CUBE_M; ++batch) {
      const uint32_t row = batch * LONG_CUBE_M;
      pv_.SetTensorA(probWork_[row * LONG_N], false);
      pv_.SetTensorB(vWork_, false);
      pv_.template IterateAll<false>(pvWork_[row * HEAD_DIM], 0, false, true);
      pv_.WaitIterateAll();
    }
  }

  __aicore__ inline void ProcessGroup(uint32_t work, uint32_t request,
                                      uint32_t kvHead) {
    const int32_t qStartRaw = qStarts_.GetValue(request);
    const int32_t qLenRaw = qLens_.GetValue(request);
    const int32_t prefixRaw = prefixes_.GetValue(request);
    if (qStartRaw < 0 || qLenRaw <= 0 || qLenRaw > MAX_QUERY ||
        prefixRaw < 0 || shape_.gqa > 8) return;
    const uint32_t qStart = static_cast<uint32_t>(qStartRaw);
    const uint32_t qLen = static_cast<uint32_t>(qLenRaw);
    const uint32_t prefix = static_cast<uint32_t>(prefixRaw);
    statePrefix_ = prefix;
    const uint32_t total = prefix + qLen;
    const uint32_t lanes = qLen * shape_.gqa;
    BindWorkspace(work);
    PrepareQueries(qStart, qLen, kvHead);
    for (uint32_t row = 0; row < lanes; ++row) {
      state_.SetValue(row, -3.402823466e+38F);
      state_.SetValue(LONG_M + row, 0.0F);
      for (uint32_t d = 0; d < HEAD_DIM; ++d)
        acc_.SetValue(row * HEAD_DIM + d, 0.0F);
    }

    for (uint32_t start = 0; start < total; start += LONG_N) {
      PrepareKvTile(request, kvHead, qStart, prefix, total, start);
      CubeQk();
      for (uint32_t row = 0; row < LONG_M; ++row) {
        const uint32_t qi = row / shape_.gqa;
        const bool rowValid = row < lanes;
        float tileMax = -3.402823466e+38F;
        for (uint32_t col = 0; col < LONG_N; ++col) {
          const bool valid = rowValid && start + col < total &&
                             start + col <= prefix + qi;
          float score = valid
              ? scoreWork_.GetValue(row * LONG_N + col) * shape_.scale
              : -3.402823466e+38F;
          if (score > tileMax) tileMax = score;
        }
        const float oldMax = rowValid ? state_.GetValue(row) : 0.0F;
        const float newMax = tileMax > oldMax ? tileMax : oldMax;
        const float oldFactor =
            rowValid && oldMax > -3.0e+38F ? DeviceExp(oldMax - newMax) : 0.0F;
        const float tileSum = rowValid
            ? PrepareProbabilityRow(row, qi, start, total, newMax) : 0.0F;
        if (!rowValid) {
          for (uint32_t col = 0; col < LONG_N; ++col)
            probWork_.SetValue(row * LONG_N + col, half(0));
        }
        if (rowValid) {
          state_.SetValue(row, newMax);
          state_.SetValue(LONG_M + row,
                          state_.GetValue(LONG_M + row) * oldFactor + tileSum);
          for (uint32_t d = 0; d < HEAD_DIM; ++d)
            acc_.SetValue(row * HEAD_DIM + d,
                          acc_.GetValue(row * HEAD_DIM + d) * oldFactor);
        }
      }
      CubePv();
      for (uint32_t row = 0; row < lanes; ++row)
        for (uint32_t d = 0; d < HEAD_DIM; ++d)
          acc_.SetValue(row * HEAD_DIM + d,
                        acc_.GetValue(row * HEAD_DIM + d) +
                        pvWork_.GetValue(row * HEAD_DIM + d));
    }
    for (uint32_t row = 0; row < lanes; ++row) {
      const uint32_t qi = row / shape_.gqa;
      const uint32_t group = row % shape_.gqa;
      const uint32_t qHead = kvHead * shape_.gqa + group;
      const uint64_t dst =
          (static_cast<uint64_t>(qStart + qi) * shape_.queryHeads + qHead) *
          HEAD_DIM;
      const float denom = state_.GetValue(LONG_M + row);
      const float inv = denom > 0.0F ? 1.0F / denom : 0.0F;
      for (uint32_t d = 0; d < HEAD_DIM; ++d)
        out_.SetValue(dst + d,
                      static_cast<half>(acc_.GetValue(row * HEAD_DIM + d) * inv));
    }
  }

  RuntimeShape shape_{};
  uint32_t statePrefix_ = 0;
  QkMatmul& qk_;
  PvMatmul& pv_;
  Int2KvCacheLoader<half> cache_;
  __gm__ uint8_t* workspaceAddr_ = nullptr;
  AscendC::GlobalTensor<half> q_, kNew_, vNew_, out_;
  AscendC::GlobalTensor<int32_t> qStarts_, qLens_, prefixes_;
  AscendC::GlobalTensor<half> qWork_, kWork_, vWork_, probWork_;
  AscendC::GlobalTensor<float> scoreWork_, pvWork_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> kBuf_, vBuf_, accBuf_, stateBuf_, expBuf_, rowExpBuf_;
  AscendC::LocalTensor<half> kLocal_, vLocal_;
  AscendC::LocalTensor<float> acc_, state_, exp_, rowExp_;
};

}  // namespace OscarAscendC
