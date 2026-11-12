"""Derive a padding-aware writer without changing the qualified mother source.

Archive #37-49/#126/#145/#157/#164; startup D.4 was read in full.  Only a
negative native slot makes a row inactive.  Invalid positive slots retain
their original quantization and slot/position/quantizer error precedence.
The whole source hash and reversible edits keep every arithmetic helper,
DMA handoff, window owner, public symbol and graph address unchanged.
"""
from __future__ import annotations

import hashlib


MOTHER_SHA256 = "b9c44ddd1cf6c003173c6470d8596e5d1b8d10d8a51862a22694077d4050e134"

DERIVED_HEADER = """// Derived active-slot writer; archive #37-49/#126/#145/#157/#164.
// D.4 four questions (complete original log read):
// 1. This changes phase1_stores for current tokens only.
// 2. Failed historical full-history recovery cost6499.8-6655.1ms/card and
//    stores209-215ms; those timings are not current-kernel measurements.
// 3. Cache native slots within one eight-row tile and omit only negative-slot
//    quantizers whose results were already discarded; never infer padding
//    from a malformed positive slot. Keep all arithmetic and DMA fences.
// 4. Quantizer calls change from2*N*Hkv to2*active_rows; at most eight int64
//    slot values are retained locally, with no new GM/graph allocation.
//    Host control-flow checks are not CANN/NPU or measured speed evidence.
"""

_EDITS = (
    (
        "      int32_t errors[kRowTile];\n      auto bytes=packBuf_.Get<uint8_t>();",
        "      int32_t errors[kRowTile];\n"
        "      int64_t cachedSlots[kRowTile];\n"
        "      for (int32_t row=0;row<count;++row) {\n"
        "        cachedSlots[row]=Read(slots_,(base+row)/heads);\n"
        "        errors[row]=0;\n"
        "      }\n"
        "      auto bytes=packBuf_.Get<uint8_t>();",
    ),
    (
        "        for (int32_t row=0;row<count;++row) {\n"
        "          const int32_t error=quant_.Process(values[row*dim],",
        "        for (int32_t row=0;row<count;++row) {\n"
        "          if (cachedSlots[row]<0) continue;\n"
        "          const int32_t error=quant_.Process(values[row*dim],",
    ),
    (
        "        const int64_t slot=Read(slots_,token);",
        "        const int64_t slot=cachedSlots[row];",
    ),
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def transform(text: str) -> str:
    """Apply only the audited Run edits; reject drift or double application."""
    if _digest(text) != MOTHER_SHA256:
        raise ValueError("active store mother source SHA256 mismatch")
    for before, after in _EDITS:
        if text.count(before) != 1 or after in text:
            raise ValueError("active store Run edit anchor drift")
        text = text.replace(before, after, 1)
    return DERIVED_HEADER + text


def restore(text: str) -> str:
    """Recover the exact mother, rejecting changes outside the audited edits."""
    if not text.startswith(DERIVED_HEADER):
        raise ValueError("active store derived header mismatch")
    text = text[len(DERIVED_HEADER):]
    for before, after in reversed(_EDITS):
        if text.count(after) != 1:
            raise ValueError("active store derived Run edit drift")
        text = text.replace(after, before, 1)
    if _digest(text) != MOTHER_SHA256:
        raise ValueError("active store restored source SHA256 mismatch")
    return text
