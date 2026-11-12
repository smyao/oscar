"""Remove only overwritten current-K/V zero fills after finite optimization.

Archive #126/#145/#150/#151/#154/#164/#172 and full startup D.4 reviewed.
The owner pins the whole C4/C16 mother; this transform additionally pins the
complete common LoadPrecise method and restores it byte for byte. No Cast,
finite check, DMA, fence, window path or public interface changes.
"""
from hashlib import sha256


METHOD_SHA256 = "ced157ebdf277e99a79d7dc1f3a084ee5b1b2c3ab41c7874ec86c48c0c79c2cf"
HEADER = """// PRECISE_CURRENT_ZERO_FILL_V1
// Archive #126/#145/#150/#151/#154/#164/#172; full startup D.4 reviewed.
// D.4(1): bounded current-source K/V staging in fused C4/C16 attention.
// D.4(2): no whole-history restoration (old6499.8-6655.1ms/card).
// D.4(3): skip only zeros overwritten by the original full-size Cast/DMA.
// Empty/error/window paths, partial BF16 padding, finite predicate, Casts,
// all DMA/fences and output ownership remain; no new GM/UB allocation.
// D.4(4): full KV256 K+V avoids768KiB logical UB zero writes, with extra
// integer eligibility checks. This is not measured HBM traffic or speed.
"""
OLD_DST = """    Duplicate(dst,0.0F,kElements);PipeBarrier<PIPE_V>();
    if(kind==2) {
"""
NEW_DST = """    // A valid nonempty current load always Casts all kElements, including
    // its explicitly zeroed BF16 tail. That Cast overwrites every dst bit.
    // Keep the original Vector barrier and all original error checks below.
    bool currentOverwritesDst=false;
    if(kind==2) {
      const int32_t writeRows=static_cast<int32_t>(Max64(0,Min64(kHalfKv,kvend-start-lane*kHalfKv)));
      if(writeRows>0) {
        const int64_t writeToken=requestBegin+start+lane*kHalfKv-context;
        currentOverwritesDst=writeToken>=0 && writeToken+writeRows<=g.tokens;
      }
    }
    if(!currentOverwritesDst)Duplicate(dst,0.0F,kElements);
    PipeBarrier<PIPE_V>();
    if(kind==2) {
"""
OLD_BATCH = """      Duplicate(batch.ReinterpretCast<uint16_t>(),static_cast<uint16_t>(0),kElements);
      Fence<HardEvent::V_MTE2>();
"""
NEW_BATCH = """      // A full 16-row DMA writes the entire BF16 staging tile. Partial
      // tiles retain their exact original full zero-fill and padded Cast.
      if(rows<kHalfKv)
        Duplicate(batch.ReinterpretCast<uint16_t>(),static_cast<uint16_t>(0),kElements);
      Fence<HardEvent::V_MTE2>();
"""


def _bounds(text: str) -> tuple[int, int]:
    signature = "  __aicore__ void LoadPrecise(int64_t start,bool value) {"
    if text.count(signature) != 1:
        raise ValueError("precise zero-fill LoadPrecise signature changed")
    start = text.index(signature)
    depth = 0
    for pos in range(text.index("{", start), len(text)):
        depth += (text[pos] == "{") - (text[pos] == "}")
        if depth == 0:
            return start, pos + 1
    raise ValueError("precise zero-fill LoadPrecise is unclosed")


def _check(text: str) -> tuple[int, int]:
    if "static_assert(D==256 && kHalfKv==16," not in text:
        raise ValueError("precise zero-fill requires existing D256 finite certificate")
    start, end = _bounds(text)
    method = text[start:end]
    if sha256(method.encode()).hexdigest() != METHOD_SHA256:
        raise ValueError("precise zero-fill requires the exact protected LoadPrecise method")
    for anchor in (OLD_DST, OLD_BATCH):
        if method.count(anchor) != 1:
            raise ValueError("precise zero-fill source anchor changed")
    return start, end


def transform(text: str) -> str:
    if HEADER in text:
        raise ValueError("precise zero-fill transform already applied")
    start, end = _check(text)
    method = text[start:end].replace(OLD_DST, NEW_DST, 1).replace(OLD_BATCH, NEW_BATCH, 1)
    return HEADER + text[:start] + method + text[end:]


def restore(text: str) -> str:
    if not text.startswith(HEADER) or text.count(NEW_DST) != 1 or text.count(NEW_BATCH) != 1:
        raise ValueError("precise zero-fill derived source changed")
    original = text.removeprefix(HEADER).replace(NEW_DST, OLD_DST, 1).replace(NEW_BATCH, OLD_BATCH, 1)
    _check(original)
    if transform(original) != text:
        raise ValueError("precise zero-fill inverse is not canonical")
    return original
