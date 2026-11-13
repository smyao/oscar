"""Archive #126/#145/#150/#151/#154/#164/#172: scalar page-map reuse.

D.4(1): repeated physical-page metadata loads inside bounded fused FIA.
D.4(2): no full-history restoration or expansion; the old 6499.8--6655.1ms
failure is not a measurement of this helper. D.4(3): reuse one immutable table
entry per kernel instance, preserve every bounds/error check and MTE2_S fence.
D.4(4): three int64 scalar fields, no UB/GM allocation or floating-point change.
The input table is read-only throughout one launch. Init invalidates the cache
on every new launch, including fixed-address graph replay with changed inputs.
"""
from __future__ import annotations

from hashlib import sha256

BASELINES = {
    "81f677c45c5c49b3d92b2ae3c007c66eafb6acbaa03f1aa72c9e0ce5dedca019": "decode",
    "4141c96580820f19db8d7fbe34ed6f0b73ef85a773452336645d846e4d83642a": "cluster4",
    "c073f2608b89533df60e1f1f0d5f46d830374a2c7a0d9aa9055baee259be1a1f": "cluster16",
}
HEADER = """// PHYSICAL_PAGE_SCALAR_CACHE_V1
// Archive #126/#145/#150/#151/#154/#164/#172; full startup D.4 reviewed.
// D.4(1): bounded FIA physical-page metadata lookup, not attention math.
// D.4(2): no full-history restore (historical dequant6499.8--6655.1ms).
// D.4(3): cache one (request,virtual128-column) -> scalar table value only
// within this read-only-input launch; Init resets it for every graph replay.
// Keep every bounds/error check and MTE2_S fence, including cache hits:
// the old fence can also complete a preceding packed-data DMA.
// D.4(4): three int64 scalar fields, no additional UB/GM, no FP change.
// Metadata DMA/read counts may fall; no NPU speed or HBM claim follows.
"""
OLD_PHYSICAL = """  __aicore__ bool Physical(int64_t position,int64_t& block,int64_t& inpage) {
    if(position<0 || position/128>=g.columns) {error=1;return false;}
    auto local=addressBuf.Get<int32_t>();
    DataCopyExtParams copy{1,4,0,0,0};DataCopyPadExtParams<int32_t> pad{false,0,0,0};
    DataCopyPad(local,bt[request*g.columns+position/128],copy,pad);
    Fence<HardEvent::MTE2_S>();
    const int64_t virtualBlock=local.GetValue(0);
    const int64_t ratio=g.blockTokens/128;
    if(virtualBlock<0 || virtualBlock>=g.blocks*ratio) {error=1;return false;}
    block=virtualBlock/ratio;inpage=(virtualBlock%ratio)*128+position%128;
    return true;
  }"""
NEW_PHYSICAL = """  __aicore__ bool Physical(int64_t position,int64_t& block,int64_t& inpage) {
    if(position<0 || position/128>=g.columns) {error=1;return false;}
    const int64_t column=position/128;
    const bool cached=physicalCacheRequest==request && physicalCacheColumn==column;
    if(!cached) {
      auto local=addressBuf.Get<int32_t>();
      DataCopyExtParams copy{1,4,0,0,0};DataCopyPadExtParams<int32_t> pad{false,0,0,0};
      DataCopyPad(local,bt[request*g.columns+column],copy,pad);
    }
    // Preserve incidental completion of preceding packed-data MTE2 work.
    Fence<HardEvent::MTE2_S>();
    if(!cached) {
      physicalCacheValue=addressBuf.Get<int32_t>().GetValue(0);
      physicalCacheRequest=request;physicalCacheColumn=column;
    }
    // The saved value is scalar, independent of tag/status addressBuf reuse.
    const int64_t virtualBlock=physicalCacheValue;
    const int64_t ratio=g.blockTokens/128;
    if(virtualBlock<0 || virtualBlock>=g.blocks*ratio) {error=1;return false;}
    block=virtualBlock/ratio;inpage=(virtualBlock%ratio)*128+position%128;
    return true;
  }"""
INIT_RESET = "    physicalCacheRequest=-1;physicalCacheColumn=-1;physicalCacheValue=0;\n"
FIELDS = "  int64_t physicalCacheRequest,physicalCacheColumn,physicalCacheValue;\n"
REPLACEMENTS = (
    (OLD_PHYSICAL, NEW_PHYSICAL),
    ("    g=geometry;\n", "    g=geometry;\n" + INIT_RESET),
    ("  TPipe pipe;CubeMatmul mm;", FIELDS + "  TPipe pipe;CubeMatmul mm;"),
)


def family(text: str) -> str:
    result = BASELINES.get(sha256(text.encode()).hexdigest())
    if result is None:
        raise ValueError("physical page cache requires exact de0c85b active source")
    return result


def transform(text: str) -> str:
    family(text)
    for before, after in REPLACEMENTS:
        if text.count(before) != 1:
            raise ValueError("physical page cache source anchor drift")
        text = text.replace(before, after, 1)
    return HEADER + text


def restore(text: str) -> str:
    if not text.startswith(HEADER):
        raise ValueError("physical page cache derived header drift")
    original = text.removeprefix(HEADER)
    for before, after in reversed(REPLACEMENTS):
        if original.count(after) != 1:
            raise ValueError("physical page cache derived anchor drift")
        original = original.replace(after, before, 1)
    family(original)
    if transform(original) != text:
        raise ValueError("physical page cache inverse is not exact")
    return original
