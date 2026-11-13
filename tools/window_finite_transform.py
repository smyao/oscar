"""Archive #126/#145/#150/#151/#154/#164/#172: ordered window checks.

D.4(1): valid-tag exact-window K/V finite checks in bounded C4/C16 FIA.
D.4(2): no whole-history restore (historical6499.8--6655.1ms/card).
D.4(3): certify only consecutive valid-tag rows, retaining original fallback,
tag/physical/finite error order, values, Cast, DMA and ownership fences.
D.4(4): dead natural BF16 staging becomes Abs scratch; no added UB/GM.
Scalar-wait reduction does not certify CANN/NPU correctness or speed.
"""
from __future__ import annotations

from hashlib import sha256

from tools.cluster_state_transform import method

# Complete outputs of the preceding physical-page transform; accepting only
# this composition prevents silently dropping either ownership contract.
INPUT_SHA256 = {
    4: "1f4c77f87ba49e5749496d3528a719974183aa9e6dc8d687612934336d1784ce",
    16: "8ea2316a6e1f050dde6b30656ae1c630c0f1036555877e16c23cda930e639b40",
}
METHOD_SHA256 = "d465149391c99c27095a83e57ef9169a9af3d8b9e76e53b0ac532a38f2d4e916"
HEADER = """// WINDOW_ORDERED_FINITE_CERTIFICATE_V1
// Archive #126/#145/#150/#151/#154/#164/#172; startup D.4 reviewed in full.
// D.4(1): finite validation of exact-window K/V, bounded16-row DMA runs.
// D.4(2): no historical6.5-second whole-history restore or layout change.
// D.4(3): consecutive valid-tag runs only; invalid tags and physical errors
// retain original order against finite errors. Same source values, count-mode
// row fallback, Cast, every DMA and ownership fence, K then V publication.
// D.4(4): completed Cast frees naturalBuf for Abs scratch; no new allocation.
// The scalar-wait work reduction is not a measured speed or NPU certificate.
"""
OLD = """      auto check=scratchBuf.Get<float>(128);
      for(int32_t row=0;row<rows;++row) {
        if(tags.GetValue(row)!=inpage+row) {error=4;continue;}
        ReduceSum(check,dst[(j+row)*D],check[16],D);Fence<HardEvent::V_S>();
        if(!Finite(check.GetValue(0)))error=2;
      }
"""
NEW = """      // Cast has consumed every BF16 batch row. Its completed PIPE_V
      // permits naturalBuf to serve as disjoint FP32 Abs work. Tags remain
      // in packedBuf, and dst remains untouched. Keep both trailing reuse
      // fences before the next physical/tag/DMA run.
      static_assert(D==256 && kHalfKv==16,
          "window finite certificate requires D256/16-row runs");
      static_assert(kHistoryHalfKv*(D+16)*sizeof(bfloat16_t)>=kElements*sizeof(float),
          "dead window BF16 staging must fit the FP32 certificate work");
      static_assert(kHistoryHalfKv*8>=kHalfKv*8,
          "window finite result scratch must fit original row slots");
      auto check=scratchBuf.Get<float>(kHalfKv*8);
      auto finiteWork=naturalBuf.Get<float>(kElements);
      for(int32_t row=0;row<rows;) {
        if(tags.GetValue(row)!=inpage+row) {error=4;++row;continue;}
        int32_t end=row+1;
        while(end<rows && tags.GetValue(end)==inpage+end)++end;
        // Only error2 is possible inside this valid-tag run. Aggregate its
        // unchanged row predicates before applying the next tag/physical
        // error, preserving original last-error ownership (4 then2,2 then4).
        if(!oscar_official_finite::FiniteRows256(
            check,dst[(j+row)*D],finiteWork,end-row))error=2;
        row=end;
      }
"""


def _check(text: str) -> None:
    digest = sha256(text.encode()).hexdigest()
    if digest not in INPUT_SHA256.values():
        raise ValueError("window finite requires its complete pinned input source")
    body = method(text, "LoadWindowRuns")
    if sha256(body.encode()).hexdigest() != METHOD_SHA256 or body.count(OLD) != 1:
        raise ValueError("window finite complete method/row-check drift")
    if '#include "attention_official_finite.h"' not in text:
        raise ValueError("window finite requires the existing exact certificate dependency")
    if "Cast(dst[j*D],batch,RoundMode::CAST_NONE,rows*D);PipeBarrier<PIPE_V>();" not in body:
        raise ValueError("window finite scratch lifetime drift")


def transform(text: str) -> str:
    _check(text)
    return HEADER + text.replace(OLD, NEW, 1)


def restore(text: str) -> str:
    if not text.startswith(HEADER) or text.count(NEW) != 1:
        raise ValueError("window finite derivative/header drift")
    original = text.removeprefix(HEADER).replace(NEW, OLD, 1)
    _check(original)
    if transform(original) != text:
        raise ValueError("window finite inverse is not exact")
    return original
