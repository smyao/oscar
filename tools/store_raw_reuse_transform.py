"""Retain exact BF16 input rows for publication within one writer tile.

Archive #126/#145/#154/#155/#164/#172 and startup D.4 were read in full
where required.  This composes after active_store_transform: no quantizer,
input arithmetic, source ABI, error order, or native window owner changes.
The binding rejects input/output overlap.  Only both identical original
input pointers authorize reuse; distinct or partially aliased originals
retain their original reload path.  No state survives this invocation.
"""
from __future__ import annotations

import hashlib


INPUT_SHA256 = "fd35153511b9eacb2e63980d6aed712c8d4d1f8f2fe33f82ecb044eb4be53842"

DERIVED_HEADER = """// Derived same-invocation raw-input reuse; archive #126/#145/#154/#155/#164/#172.
// D.4 four questions (complete original log read):
// 1. This removes bounded exact-window rereads in phase1_stores only.
// 2. Old full-history dequant6499.8-6655.1ms/card and stores209-215ms
//    describe rejected historical structures, not current measurements.
// 3. Retain the existing K DMA in rawBuf and V DMA in inputBuf only when
//    key==originalKey and value==originalValue. Quantizers modify floatBuf,
//    never these BF16 bytes. Keep every DMA fence and error/owner decision;
//    distinct originals use the original reload. Binding forbids input/output
//    overlap. Do not reuse across calls, tokens, layers, or cache generations.
// 4. UB adds3584B/core atD256; no new GM/graph allocation or UB copy.
//    Each published exact KV row avoids two GM reads (1024B atD256), while
//    all quantization and writes remain. Host checks do not certify NPU speed.
"""

_EDITS = (
    (
        "    pipe_.InitBuffer(metaBuf_,32); pipe_.InitBuffer(rawBuf_,kMaxDim*2);",
        "    pipe_.InitBuffer(metaBuf_,32); pipe_.InitBuffer(rawBuf_,kRowTile*kMaxDim*2);\n"
        "    const bool reuseExact=key==originalKey && value==originalValue;",
    ),
    (
        "        auto input=inputBuf_.Get<bfloat16_t>();auto values=floatBuf_.Get<float>();",
        "        auto input=(reuseExact && side==0) ? rawBuf_.Get<bfloat16_t>()\n"
        "                                          : inputBuf_.Get<bfloat16_t>();\n"
        "        auto values=floatBuf_.Get<float>();",
    ),
    (
        "          WriteRaw(token,head,page,position,kStride,vStride,heads,dim);",
        "          WriteRaw(token,head,page,position,kStride,vStride,heads,dim,row,reuseExact);",
    ),
    (
        "          WriteRaw(token,head,page,index,kStride,vStride,heads,dim);",
        "          WriteRaw(token,head,page,index,kStride,vStride,heads,dim,row,reuseExact);",
    ),
    (
        "      int64_t index,int64_t kStride,int64_t vStride,int64_t heads,int32_t dim) {\n"
        "    // The source K/V are read again only for the bounded exact window, never\n"
        "    // for history. No rotation or quantizer rounding can touch these bytes.",
        "      int64_t index,int64_t kStride,int64_t vStride,int64_t heads,int32_t dim,\n"
        "      int32_t row,bool reuseExact) {\n"
        "    // Retained BF16 rows precede clipping; only floatBuf was quantized.\n"
        "    // Both raw publications finish before the next tile's MTE2 overwrite.\n"
        "    // Distinct originals keep the original bounded exact-window reload.",
    ),
    (
        "      DataCopy(converted,(side==0?originalK_:originalV_)[(token*heads+head)*dim],dim);",
        "      if (reuseExact) {\n"
        "        converted=(side==0 ? rawBuf_.Get<bfloat16_t>()\n"
        "                           : inputBuf_.Get<bfloat16_t>())[row*dim];\n"
        "      } else {\n"
        "        DataCopy(converted,(side==0?originalK_:originalV_)[(token*heads+head)*dim],dim);\n"
        "      }",
    ),
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def transform(text: str) -> str:
    """Apply after the pinned active-slot transform; reject unrelated drift."""
    if _digest(text) != INPUT_SHA256:
        raise ValueError("raw reuse input source SHA256 mismatch")
    for before, after in _EDITS:
        if text.count(before) != 1 or after in text:
            raise ValueError("raw reuse edit anchor drift")
        text = text.replace(before, after, 1)
    return DERIVED_HEADER + text


def restore(text: str) -> str:
    """Recover the exact prior active-slot source, including all fences."""
    if not text.startswith(DERIVED_HEADER):
        raise ValueError("raw reuse derived header mismatch")
    text = text[len(DERIVED_HEADER):]
    for before, after in reversed(_EDITS):
        if text.count(after) != 1:
            raise ValueError("raw reuse derived edit drift")
        text = text.replace(after, before, 1)
    if _digest(text) != INPUT_SHA256:
        raise ValueError("raw reuse restored source SHA256 mismatch")
    return text
