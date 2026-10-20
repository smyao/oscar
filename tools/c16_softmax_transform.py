# Archive #126/#134/#145/#148/#164: source-bound C16 finite-check candidate;
# keep original count-mode reduction, FP32 softmax and cross-pipe ownership.
"""Transform only the measured C16 softmax finite-check hotspot.

The original production source is an input, never modified by this module.
No symbol/ABI renaming occurs here: the composing generator owns it.
"""
from __future__ import annotations

import re

MARKER = "// C16_TIMER_TARGETED_SOFTMAX_FINITE_V1"
HEADER = """// C16_TIMER_TARGETED_SOFTMAX_FINITE_V1
// Archive #126/#134/#145/#148/#164; D.4 complete raw log reread.
// D.4: fused history FIA finite-check only; old full-history dequant
// 6499.8-6655.1ms/95.4% and stores209-215ms remain forbidden structures.
// Same bounded KV256 tile, FP32 ACC_VAL reduction and finite predicate;
// no history materialization, precision change or cross-core flag change.
// Removes unused scalar UB result publication in complete-history rows.
// Target softmax_math23.4-23.8ms is measured; its finite subshare and this
// candidate's full-C16/device/model speed remain unmeasured.
"""

OLD = """      // Natural unpack is dead until later alpha Brcb. Each scalar result
      // has its own32B-aligned slot; the original tmp[16] reduction work
      // area stays separate. PIPE_V keeps its reuse in original row order.
      auto finiteSums=naturalBuf.Get<float>(kHalfRows*8);
      for(int32_t r=0;r<activeRows;++r) {
        PipeBarrier<PIPE_V>();
        ReduceSum(finiteSums[r*8],scores[r*kKvRows],tmp[16],kKvRows);
      }
      Fence<HardEvent::V_S>();
      for(int32_t r=0;r<activeRows;++r)
        if(!Finite(finiteSums.GetValue(r*8)))error=2;
      // Protect scalar reads before any later Vector reuse of naturalBuf.
      Fence<HardEvent::S_V>();
"""

NEW = """      // Same A2 count-mode vcadd/ACC_VAL per row, consumed as a scalar
      // finite predicate. No scalar-to-UB publication has a consumer here.
      auto finiteSums=naturalBuf.Get<float>(kHalfRows*8);
      if(!oscar_c16_softmax::FiniteRows256(finiteSums,scores,activeRows))error=2;
"""


def _replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"C16 softmax source contract drift: expected one match, got {count}")
    return text.replace(old, new, 1)


def transform(text: str) -> str:
    """Return a same-ABI clone; reject drift or accidental double application."""
    if MARKER in text:
        raise ValueError("C16 softmax transform was already applied")
    if not re.search(r"static_assert\(D==256 && kKvRows==D,", text):
        raise ValueError("C16 softmax candidate requires the D256/KV256 source contract")
    if "#define OSCAR_STRIPED_D256_ONLY 1" not in text:
        raise ValueError("C16 softmax candidate requires the existing D256-only route")
    result = _replace_once(text, OLD, NEW)
    result = _replace_once(
        result, '#include "attention_bf16_cast.h"',
        '#include "attention_bf16_cast.h"\n#include "attention_c16_softmax_helpers.h"')
    # No tiling rewrite: the official SDK's tiling function is constexpr and
    # its compile-time portions may already fold. Such work is not new gain.
    return HEADER + result
