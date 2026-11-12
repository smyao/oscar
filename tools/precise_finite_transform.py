"""Archive #145/#150/#151/#154/#164/#172: exact current-source finite check.

D.4(1): finite validation in bounded fused C4/C16 current K/V staging.
D.4(2): retain fused KV256; never restore the old 6.5-second full history.
D.4(3): reuse the existing sufficient-bound helper and original row-reduction
fallback; preserve Cast, values, errors, all DMA and cross-pipe ownership.
D.4(4): reuse dead BF16 staging, with capacity assertions; no speed claim.

The owner generator pins the complete source before composing transforms.
This local transform checks its exact method/geometry anchors and provides an
inverse so the owner can verify every unrelated byte against that pinned source.
"""
from __future__ import annotations


MARKER = "// PRECISE_CURRENT_FINITE_CERTIFICATE_V1"
HEADER = """// PRECISE_CURRENT_FINITE_CERTIFICATE_V1
// Archive #145/#150/#151/#154/#164/#172; complete startup D.4 reviewed.
// D.4(1): current-source finite validation inside bounded fused C4/C16 FIA.
// D.4(2): no full-history restore (old6499.8--6655.1ms) or new launch.
// D.4(3): same FP32 finite/error predicate through existing certificate and
// exact row-reduction fallback; unchanged Cast, masks, math and DMA fences.
// D.4(4): reuse dead BF16 staging, no added UB/GM; device speed unmeasured.
"""
DEPENDENCY = '#include "attention_official_finite.h" // PRECISE_FINITE_DEPENDENCY_V1\n'
OLD = """      auto check=scratchBuf.Get<float>(128);
      for(int32_t row=0;row<rows;++row) {
        ReduceSum(check,dst[row*D],check[16],D);Fence<HardEvent::V_S>();
        if(!Finite(check.GetValue(0)))error=2;
      }
      return;
"""
NEW = """      // Cast has consumed BF16 staging. Its storage is dead until PublishKv;
      // the preceding Vector barrier also orders its reuse as FP32 scratch.
      static_assert(D==256 && kHalfKv==16,
          "current finite certificate requires the D256/16-row precise tile");
      static_assert(kHistoryHalfKv*(D+16)*sizeof(bfloat16_t)>=kElements*sizeof(float),
          "dead BF16 staging must fit the precise finite certificate");
      static_assert(kHistoryHalfKv*8>=kHalfKv*8,
          "finite-result scratch must fit every original row-reduction slot");
      auto check=scratchBuf.Get<float>(kHalfKv*8);
      auto finiteWork=naturalBuf.Get<float>(kElements);
      if(!oscar_official_finite::FiniteRows256(check,dst,finiteWork,rows))error=2;
      return;
"""


def _bounds(text: str) -> tuple[int, int]:
    signature = "  __aicore__ void LoadPrecise(int64_t start,bool value) {"
    if text.count(signature) != 1:
        raise ValueError("precise finite LoadPrecise source contract drift")
    start = text.index(signature)
    opening = text.index("{", start)
    depth = 0
    for index in range(opening, len(text)):
        depth += (text[index] == "{") - (text[index] == "}")
        if depth == 0:
            return start, index + 1
    raise ValueError("precise finite LoadPrecise is unclosed")


def _check(text: str) -> None:
    symbols = ("oscar_attention_cv_official_cluster4_kernel",
               "oscar_attention_cv_official_cluster16_kernel")
    required = (
        "static_assert(D==256 && kKvRows==D,",
        "constexpr int32_t kHalfRows=32, kHalfKv=16;",
        "pipe.InitBuffer(naturalBuf,kHistoryHalfKv*(D+16)*2);",
        "pipe.InitBuffer(scratchBuf,kHistoryHalfKv*8*4);",
    )
    if sum(symbol in text for symbol in symbols) != 1 or any(s not in text for s in required):
        raise ValueError("precise finite requires the complete C4/C16 D256 source contract")
    start, end = _bounds(text)
    method = text[start:end]
    current = "    if(kind==2) {"
    window = "    if(kind==1) {LoadWindowRuns(start,value);return;}"
    guards = (
        "      if(rows==0)return;",
        "      if(token<0 || token+rows>g.tokens) {error=1;return;}",
        "      Fence<HardEvent::MTE2_V>();Cast(dst,batch,RoundMode::CAST_NONE,kElements);\n"
        "      PipeBarrier<PIPE_V>();\n",
    )
    if (method.count(current) != 1 or method.count(window) != 1 or
            any(s not in method for s in guards)):
        raise ValueError("precise finite current-source guard/Cast contract drift")
    if method.count(OLD) != 1 or text.count(OLD) != 1:
        raise ValueError("precise finite row-check anchor drift")
    if not method.index(current) < method.index(OLD) < method.index(window):
        raise ValueError("precise finite replacement escaped current-source branch")


def transform(text: str) -> str:
    if MARKER in text or DEPENDENCY in text:
        raise ValueError("precise finite transform already applied")
    _check(text)
    if '#include "attention_official_finite.h"' not in text:
        anchor = '#include "oscar_common.h"'
        if text.count(anchor) != 1:
            raise ValueError("precise finite dependency anchor drift")
        text = text.replace(anchor, DEPENDENCY + anchor, 1)
    return HEADER + text.replace(OLD, NEW, 1)


def restore(text: str) -> str:
    if not text.startswith(HEADER) or text.count(NEW) != 1:
        raise ValueError("precise finite derived source contract drift")
    original = text.removeprefix(HEADER).replace(NEW, OLD, 1).replace(DEPENDENCY, "", 1)
    _check(original)
    if transform(original) != text:
        raise ValueError("precise finite inverse is not exact")
    return original
