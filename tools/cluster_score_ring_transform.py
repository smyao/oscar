"""Archive #145/#150/#151/#154/#164/#172: C16 score ownership, not math.

D.4(1): bounded fused FIA Cube-score/Vector-softmax intermediate addresses.
D.4(2): avoid the historical 6499.8--6655.1ms full-history restoration.
D.4(3): reuse four existing score slots after both AIV P acknowledgements;
keep every cross flag, DMA extent, FP32 operation, error and allocator byte.
D.4(4): score address-set capacity 2MiB -> 512KiB/core; traffic is unchanged.
This does not establish L2 hits, HBM savings, NPU correctness or speed.

The composing generator owns full-source pinning. The protocol and consumers
are additionally pinned here: changing their lifetime invalidates this proof.
"""
from __future__ import annotations

from hashlib import sha256

from tools.cluster_state_transform import method

MARKER = "// CLUSTER_SCORE_RING4_V1"
HEADER = """// CLUSTER_SCORE_RING4_V1
// Archive #145/#150/#151/#154/#164/#172; full startup D.4 reviewed.
// D.4(1): C16 bounded fused-FIA score address reuse, no attention math change.
// D.4(2): no full-history restore (historical dequant6499.8--6655.1ms).
// D.4(3): P(i-4) acknowledges BOTH AIV score consumers before slot overwrite;
// Q/P and online-state slots remain per member, with all existing fences/flags.
// D.4(4): score address-set capacity2MiB->512KiB/core; same allocator and DMA
// bytes. No measured L2-hit, HBM-bandwidth, NPU-correctness or speed claim.
"""
OLD_SELECT = """  __aicore__ void SelectClusterSlots(int32_t buffer,int32_t slot) {
    kOffset=buffer?extraKvOffset:baseKOffset;
    vOffset=kOffset+kKvRows*D;
    scoreOffset=slot?extraScoreOffset+(slot-1)*kQueryRows*kKvRows:baseScoreOffset; // P lives in each Q slot second half, never in FP32 score.
  }"""
NEW_SELECT = """  __aicore__ void SelectClusterSlots(int32_t buffer,int32_t slot) {
    kOffset=buffer?extraKvOffset:baseKOffset;
    vOffset=kOffset+kKvRows*D;
    // Cube waits P(i-4), from both AIVs, before replacing Score(i-4).
    // P remains in the member's Q half-slot; only score uses the four-slot ring.
    const int32_t scoreSlot=slot%4;
    scoreOffset=scoreSlot?extraScoreOffset+(scoreSlot-1)*kQueryRows*kKvRows:baseScoreOffset;
  }"""
PROTOCOL_SHA256 = {
    ("CubeCluster", "void"): "e0963d4fea74631a8343edb77f8fd764b6dc83e04ad2958b3ed9bfe9264ec859",
    ("VectorCluster", "void"): "b07c8baca9e5e15067ee9edf647b4f77a90a23f3970d8ea26a3332896deeae2f",
    ("Softmax", "void"): "de32ba20a9321065c154c1f6ecb0ee9a153034955d2f3a3ba3907066d96f489d",
    ("SpillState", "void"): "4c8729cf8cdec5cfc502caa949a06e638dd3a414e911cd440a74280987f60567",
    ("ProbabilityOffset", "int64_t"): "1ef63c5cd8fafb48faee7e38ec75fbce256d417265d17d02d05be8b087120a1f",
    ("IsClusterMember", "bool"): "829a5d93acbe1135af37de64b8ccc18e149cacd25c155cab05b03f52d16c2d6d",
}


def _check(text: str) -> None:
    if "oscar_attention_cv_official_cluster16_kernel" not in text:
        raise ValueError("score ring requires the complete C16 source")
    if text.count(OLD_SELECT) != 1:
        raise ValueError("score ring slot selection anchor drift")
    for (name, result), expected in PROTOCOL_SHA256.items():
        if sha256(method(text, name, result).encode()).hexdigest() != expected:
            raise ValueError(f"score ring {name} ownership/consumer drift")


def transform(text: str) -> str:
    if MARKER in text:
        raise ValueError("score ring transform already applied")
    _check(text)
    return HEADER + text.replace(OLD_SELECT, NEW_SELECT, 1)


def restore(text: str) -> str:
    if not text.startswith(HEADER) or text.count(NEW_SELECT) != 1:
        raise ValueError("score ring derived header/selection drift")
    original = text.removeprefix(HEADER).replace(NEW_SELECT, OLD_SELECT, 1)
    _check(original)
    if transform(original) != text:
        raise ValueError("score ring inverse is not exact")
    return original
