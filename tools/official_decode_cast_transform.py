"""Archive #126/#145/#164: batch only official q4/q1 operand publication.

D.4 four questions: this is the fused-FIA publication phase; the rejected
6499.8--6655.1 ms full-history restore is never introduced; only independent
normal-mask FP32-to-BF16 repeats use the already qualified C16 helper; UB/GM,
addresses, rounding and caller-owned DMA fences do not change. Decode's
16384-element publication uses two explicit repeat batches (255 + 1), not
the unsafe count overload. Instruction-issue reduction is not device timing.

Use qualified names at exactly three publication sites after all original
headers have been parsed. No namespace macro can change the shared unpack
helper, and LoadQueries' zero-padding conversion retains its original helper.
"""

_INCLUDE = '#include "attention_bf16_cast.h"'
_LAST_INCLUDE = '#include "adv_api/activation/softmaxflashv2.h"'
_HELPER_BLOCK = '''
// Archive #126/#145/#164, D.4: bounded q4/q1 publication only; preserve
// all fences and numerical work, with no history restore or extra UB/GM.
// The normal-mask helper batches at most255 repeats, including D256/HalfKv64.
#include "attention_c16_cast.h"'''
_PUBLICATIONS = (
    "oscar_bf16::CastRne(rounded,src,kElements);",
    "oscar_bf16::CastRne(rounded,scores,activeRows*kKvRows);",
    "oscar_bf16::CastRne(rounded,dequantBuf.Get<float>(),kElements);",
)
_PADDING = "oscar_bf16::CastRne(rounded,zeros,count*D);"


def transform(text: str) -> str:
    """Bind the three official decode publications; reject unaudited drift."""
    if not all(symbol in text for symbol in (
        "void oscar_attention_cv_official_decode_kernel(",
        "void oscar_attention_cv_official_decode_q1_kernel(",
    )):
        raise ValueError("official decode cast transform requires q4/q1 source")
    if (text.count(_INCLUDE) != 1 or text.count(_LAST_INCLUDE) != 1
            or '#include "attention_c16_cast.h"' in text
            or "oscar_c16_cast::" in text or "#define oscar_bf16" in text):
        raise ValueError("official decode cast include/binding drift or already transformed")
    last_header = text.index(_LAST_INCLUDE) + len(_LAST_INCLUDE)
    if "#include" in text[last_header:]:
        raise ValueError("official decode cast last-header anchor drift")
    if text.count("oscar_bf16::CastRne(") != len(_PUBLICATIONS) + 1:
        raise ValueError("official decode cast call count drift")
    for call in (*_PUBLICATIONS, _PADDING):
        if text.count(call) != 1:
            raise ValueError(f"official decode cast call site drift: {call}")
    text = text.replace(_LAST_INCLUDE, _LAST_INCLUDE + _HELPER_BLOCK)
    for call in _PUBLICATIONS:
        text = text.replace(call, call.replace("oscar_bf16::", "oscar_c16_cast::"))
    return text
