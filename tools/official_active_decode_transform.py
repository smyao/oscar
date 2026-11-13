"""Archive #126/#145/#150/#151/#154/#164/#172: exact decode derivatives.

D.4(1): task and page-map metadata in fused q4/q1 attention. D.4(2): no
full-history dequantization. D.4(3): compose the audited zero-task read skip
with launch-local page-map reuse; FP math, errors and all fences remain.
D.4(4): three scalar fields, no extra UB/GM. Source reversibility is not
CANN compilation, NPU precision or a measured performance improvement.
"""
from tools import active_decode_transform as active
from tools import physical_page_cache_transform as page_cache


def transform(text: str) -> str:
    return page_cache.transform(active.transform(text))


def restore(text: str) -> str:
    original = active.restore(page_cache.restore(text))
    if transform(original) != text:
        raise ValueError("official active decode inverse is not exact")
    return original
