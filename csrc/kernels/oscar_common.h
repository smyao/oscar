// SPDX-License-Identifier: Apache-2.0
// Archive #3/#21/#81-83/#91: native A2 API shapes and no unsigned/float casts.
#pragma once
#include "kernel_operator.h"

namespace oscar_ascend_device {
using namespace AscendC;
constexpr int32_t kMaxDim = 256;
constexpr int32_t kMaxSplits = 128;

// Event idiom: native moe_gating_top_k_generalized.h, lines 186-204.
template <HardEvent event>
__aicore__ inline void Fence() {
    const event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
    SetFlag<event>(id);
    WaitFlag<event>(id);
}
__aicore__ inline bool Finite(float x) {
    return x == x && x <= 3.402823466e38F && x >= -3.402823466e38F;
}
}  // namespace oscar_ascend_device
