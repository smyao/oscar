// Archive #126/#129/#145/#148-153 and startup D.4: bounded experimental
// C4 QK/PV batching. This header is intentionally outside the production ABI.
#pragma once
#include <cstdint>

namespace oscar_ascend_experiment {
constexpr int64_t kQueryRows = 128;
constexpr int64_t kKvRows = 256;
constexpr int64_t kMembers = 4;

constexpr int64_t batched4_workspace_per_core(int64_t dim) {
  // Q[4,M,D], K/V[B,D], shared score/P[4,M,B], PV/rotation[4,M,D],
  // acc[4,M,D], max/sum[4,2,M], saved alpha[4,M]. Fixed-size, no history.
  return ((3 * kMembers * kQueryRows + 2 * kKvRows) * dim +
          kMembers * kQueryRows * kKvRows +
          3 * kMembers * kQueryRows) * 4;
}

}
