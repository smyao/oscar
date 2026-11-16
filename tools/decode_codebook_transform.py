"""Archive #126/#145/#154/#155/#164/#172: exact history decode derivative.

D.4: bounded INT2 reconstruction inside fused attention; the historical
6.5-second whole-history expansion remains prohibited. Keep the accepted
mother and all nonhistory math, task ownership, DMA fences and ABI intact.
Only the history pipeline consumes the new BF16 codebook result. This is
source protection, not CANN, NPU, graph or performance acceptance.
"""
from __future__ import annotations

from hashlib import sha256


INPUT_SHA256 = "c286404e59a9242541dfa047b50ac698a808f53d45bdc2e0419f3c37b6efe7ba"
HEADER = """// HISTORY_CODEBOOK_DECODE_V1
// Archive #126/#145/#154/#155/#164/#172; startup D.4 fully reviewed.
// D.4(1): bounded source0 INT2 reconstruction inside fused q4/q1 FIA.
// D.4(2): never materialize full-history BF16 (old dequant6499.8-6655.1ms).
// D.4(3): compute four levels per row with the original Sub/Mul/RNE order,
// select raw BF16 bits into existing natural UB, retain every DMA fence.
// Only history pipeline publication skips the now-redundant final Cast.
// Current/window/softmax/Cube, masks, error commitment and codec stay fixed.
// D.4(4): same UB/GM allocation; fewer level conversions but added byte-index
// construction and Gather. Neither API counts nor CPU tests prove speed.
"""

_EDITS = (
    ('#include "attention_aligned_unpack.h"',
     '#include "attention_aligned_unpack.h"\n#include "attention_codebook_unpack.h"'),
    (
        '  __aicore__ void PipelineUnpack(TBuf<TPosition::VECCALC>& packedInput,bool value) {\n'
        '    oscar_ascend_aligned::Unpack<D,kHalfKv>(value,liveKvRows,error,packedInput,wordBuf,',
        '  __aicore__ void PipelineUnpack(TBuf<TPosition::VECCALC>& packedInput,bool value) {\n'
        '    oscar_ascend_codebook::Unpack<D,kHalfKv>(value,liveKvRows,error,packedInput,wordBuf,',
    ),
    (
        '    oscar_bf16::CastRne(rounded,dequantBuf.Get<float>(),kElements);',
        '    // PipelineUnpack already produced exact BF16 bits in naturalBuf.',
    ),
    (
        '      Unpack(false);PipelinePublishKv(false,sub,0);\n'
        '      Unpack(true);PipelinePublishKv(true,sub,0);',
        '      PipelineUnpack(packedBuf,false);PipelinePublishKv(false,sub,0);\n'
        '      PipelineUnpack(packedBuf,true);PipelinePublishKv(true,sub,0);',
    ),
)


def transform(text: str) -> str:
    if sha256(text.encode()).hexdigest() != INPUT_SHA256:
        raise ValueError("codebook decode requires exact prior active source")
    for before, after in _EDITS:
        if text.count(before) != 1 or after in text:
            raise ValueError("codebook decode input anchor drift")
        text = text.replace(before, after, 1)
    return HEADER + text


def restore(text: str) -> str:
    if not text.startswith(HEADER):
        raise ValueError("codebook decode header drift")
    original = text.removeprefix(HEADER)
    for before, after in reversed(_EDITS):
        if original.count(after) != 1:
            raise ValueError("codebook decode derived anchor drift")
        original = original.replace(after, before, 1)
    if sha256(original.encode()).hexdigest() != INPUT_SHA256:
        raise ValueError("codebook decode inverse source drift")
    if transform(original) != text:
        raise ValueError("codebook decode inverse is not exact")
    return original
