"""Archive #164/#126/#145: C16-only normal-mask repeat batching.

D.4: bind only six bounded-tile conversion call sites; no history restore,
buffer allocation, DMA fence or shared q1/q4/C4 helper changes. Keep their
literal spelling so the existing timer continues to find every original site.
"""

_INCLUDE = '#include "attention_bf16_cast.h"'
_CANDIDATE_INCLUDE = '#include "attention_c16_cast.h"'
_LAST_INCLUDE = '#include "adv_api/activation/softmaxflashv2.h"'
_ALIAS = "#define oscar_bf16 oscar_c16_cast"
_UNDEF = "\n#undef oscar_bf16\n"
_CALLS = (
    "oscar_bf16::CastRne(rounded,acc[row*D],count*D);",
    "oscar_bf16::CastRne(rounded,dst,D);",
    "oscar_bf16::CastRne(rounded,zeros,count*D);",
    "oscar_bf16::CastRne(rounded,src,kHistoryElements);",
    "oscar_bf16::CastRne(rounded,src,kElements);",
    "oscar_bf16::CastRne(roundedP,scores,activeRows*kKvRows);",
)


def transform(text: str) -> str:
    """Return C16 source using the isolated Cast helper; reject source drift."""
    if "attention_cv_aligned_cluster16" not in text:
        raise ValueError("C16 cast transform requires aligned cluster16 source")
    if text.count(_INCLUDE) != 1 or _CANDIDATE_INCLUDE in text:
        raise ValueError("C16 cast include missing, duplicated, or already transformed")
    if text.count(_LAST_INCLUDE) != 1 or _ALIAS in text:
        raise ValueError("C16 cast alias anchor missing or already transformed")
    if text.count("oscar_bf16::CastRne(") != len(_CALLS):
        raise ValueError("C16 cast call count drift")
    for call in _CALLS:
        if text.count(call) != 1:
            raise ValueError(f"C16 cast call site drift: {call}")
    # Parse all existing headers first. Their oscar_bf16 calls retain the
    # original helper; the alias affects only this translation unit's six
    # calls, preserving existing timer anchors verbatim.
    text = text.replace(
        _LAST_INCLUDE, f"{_LAST_INCLUDE}\n{_CANDIDATE_INCLUDE}\n{_ALIAS}"
    )
    return text + _UNDEF
