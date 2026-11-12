"""Archive #145/#150/#151/#154/#164/#172: preserve the C4 mother source.

Port only C16's existing complete-visibility/finite-certificate branch. The
original per-row path remains byte-identical for diagonals, tails and empty
source rows; all numerical methods, flags, Casts and public symbols stay put.
"""
from hashlib import sha256


BASELINE_SHA256 = "5069fa48b89a72d78b269b42c50be833007dd24e3bd05eceda0a62ceec722858"
DERIVED_HEADER = """// Derived by tools/cluster_visibility_transform.py from protected C4.
// Archive #145/#150/#151/#154/#164/#172; startup D.4 full raw log reviewed.
// D.4(1): fused C4 score finite validation and visibility setup only.
// D.4(2): old full-history dequant6499.8-6655.1ms/card and stores209-215ms
// remain forbidden structural regressions, not current measured shares.
// D.4(3): reuse the existing C16 full-visibility certificate and finite helper;
// original partial/empty row path, FP32 softmax, Casts, flags and ABI remain.
// No history restore, new allocation, query routing or precision change.
// D.4(4): certified complete blocks replace per-row scalar waits with one
// bounded proof; exceptions retain exact row checks. NPU speed unmeasured.
"""

_INCLUDE = '#include "attention_fast_unpack.h"\n#include "attention_striped_unpack.h"'
_DERIVED_INCLUDE = '#include "attention_official_finite.h"\n' + _INCLUDE
_LOOP_START = "    for(int32_t r=0;r<activeRows;++r) {\n"
_BEFORE_LOOP = "    const int64_t tailBegin=Max64(Min64(g.sink,context),context+1-g.recent);\n"
_LOOP_END = "    }\n    // The two 64-row max/sum arrays persist across KV work units."

# This is the existing official C16 branch verbatim, including the same
# sufficient certificate, row fallback, scratch capacities and fixed cut.
_VISIBLE_PREFIX = """    // Complete history tile: the earliest live query must see every
    // column. Later live rows have monotone query position/cut, so their
    // original any=true and mask-complement branch is empty as well.
    // jEnd==256 proves tileEnd is bounded by this task's kvend.
    const int64_t earliestPosition=context+qbegin+rowBase/group-requestBegin;
    const int64_t earliestCut=Max64(Min64(g.sink,context),context+1-g.recent);
    const bool fullyVisibleHistory=kind==0 && jEnd==kKvRows && start>=g.sink &&
        tileEnd<=Min64(context,Min64(earliestCut,earliestPosition+1));
    // V4 keeps cut==tailBegin for the entire forward. Window candidates
    // concatenate [0,sinkCount) and [tailBegin,context), with no hidden gap.
    // Every query position is >=context, so the complete candidate tile is
    // visible to every live row. Current tiles must additionally precede
    // the earliest query of THIS score block; the diagonal keeps its mask.
    const bool fullyVisibleWindow=kind==1 && jEnd==kKvRows && start>=0 &&
        tileEnd<=sinkCount+context-tailBegin;
    const bool fullyVisibleCurrent=kind==2 && jEnd==kKvRows && start>=context &&
        tileEnd<=earliestPosition+1;
    if(fullyVisibleHistory || fullyVisibleWindow || fullyVisibleCurrent) {
      // Same A2 count-mode vcadd/ACC_VAL per row, consumed as a scalar
      // finite predicate. No scalar-to-UB publication has a consumer here.
      auto finiteSums=naturalBuf.Get<float>(kHalfRows*8);
      // OFFICIAL_FINITE_CERTIFICATE_V1
      // KV staging is complete; this UB is dead until the following V2 call.
      static_assert(kHistoryElements>=kHalfRows*kKvRows,
          "finite certificate needs one complete score block of dead KV UB");
      auto finiteWork=dequantBuf.Get<float>(kHalfRows*kKvRows);
      if(!oscar_official_finite::FiniteRows256(
          finiteSums,scores,finiteWork,activeRows))error=2;
    } else {
"""


def _require_baseline(text: str) -> None:
    if not isinstance(text, str) or sha256(text.encode()).hexdigest() != BASELINE_SHA256:
        raise ValueError("cluster visibility requires the exact protected C4 mother source")
    for anchor in (_INCLUDE, _BEFORE_LOOP + _LOOP_START, _LOOP_END):
        if text.count(anchor) != 1:
            raise ValueError("cluster visibility mother source anchor changed")


def transform(text: str) -> str:
    _require_baseline(text)
    return DERIVED_HEADER + text.replace(_INCLUDE, _DERIVED_INCLUDE, 1).replace(
        _BEFORE_LOOP + _LOOP_START, _BEFORE_LOOP + _VISIBLE_PREFIX + _LOOP_START, 1).replace(
        _LOOP_END, "    }\n" + _LOOP_END, 1)


def restore(text: str) -> str:
    if (not isinstance(text, str) or not text.startswith(DERIVED_HEADER)
            or text.count(_VISIBLE_PREFIX) != 1
            or text.count(_DERIVED_INCLUDE) != 1):
        raise ValueError("cluster visibility derived header or branch changed")
    baseline = text[len(DERIVED_HEADER):].replace(_DERIVED_INCLUDE, _INCLUDE, 1).replace(
        _VISIBLE_PREFIX, "", 1).replace("    }\n" + _LOOP_END, _LOOP_END, 1)
    _require_baseline(baseline)
    if transform(baseline) != text:
        raise ValueError("cluster visibility derived source is not canonical")
    return baseline
