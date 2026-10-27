# Archive #126/#129/#134/#145/#148/#164: preserve causal masks, pre-mask
# finite checks, FP32 online state and every cross-pipe ownership fence.
"""Use the existing finite-row path for fully visible official C16 tiles.

The fixed per-forward cut makes every complete exact-window candidate tile
visible. Current tiles before the earliest query are also fully visible.
Only those already unmasked rows use the same count-mode finite reduction;
partially visible tiles retain their original mask and error-domain path.
"""
from __future__ import annotations


MARKER = "// OFFICIAL_C16_COMPLETE_SOURCE_FINITE_V1"
HEADER = """// OFFICIAL_C16_COMPLETE_SOURCE_FINITE_V1
// Archive #126/#129/#134/#145/#148/#164; complete D.4 raw log reviewed.
// D.4(1): finite/mask work inside bounded fused current/window/history FIA.
// D.4(2): old whole-history dequant6499.8-6655.1ms and stores209-215ms
// prohibit full-history materialization or a separate dequant launch.
// D.4(3): exact fixed-cut/current visibility proof only; unchanged KV256,
// same count-mode vcadd finite predicate, FP32 softmax and ownership fences.
// Incomplete tiles retain the original per-row range and finite checks.
// D.4(4): removes redundant range/result-publication work on proven full
// tiles; no extra UB/GM, no changed precision, no measured speed claim.
"""

OLD = """    const bool fullyVisibleHistory=kind==0 && jEnd==kKvRows && start>=g.sink &&
        tileEnd<=Min64(context,Min64(earliestCut,earliestPosition+1));
    if(fullyVisibleHistory) {
"""

NEW = """    const bool fullyVisibleHistory=kind==0 && jEnd==kKvRows && start>=g.sink &&
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
"""


def transform(text: str) -> str:
    """Change one predicate in an official D256/KV256 C16 source clone."""
    if MARKER in text:
        raise ValueError("official visibility transform was already applied")
    required = (
        "static_assert(D==256 && kKvRows==D,",
        "#define OSCAR_STRIPED_D256_ONLY 1",
        "oscar_attention_cv_official_cluster16_kernel",
        "const int64_t tailBegin=Max64(Min64(g.sink,context),context+1-g.recent);",
        "const int64_t cut=Max64(Min64(g.sink,context),context+1-g.recent);",
        "oscar_c16_softmax::FiniteRows256(finiteSums,scores,activeRows)",
    )
    if any(anchor not in text for anchor in required):
        raise ValueError("official visibility requires the fixed-cut D256 C16 source contract")
    if text.count(OLD) != 1:
        raise ValueError("official visibility predicate source contract drift")
    return HEADER + text.replace(OLD, NEW, 1)
