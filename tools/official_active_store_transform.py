"""Archive #126/#145/#157/#164: same-ABI current-token store derivatives.

D.4(1): current-token writes, not history materialization. D.4(2): the
historical 6.5s recovery is not a measurement of this writer. D.4(3): retain
original BF16 inputs within one tile after quantization; no persistent
history expansion. D.4(4): 3584 additional UB bytes, unchanged arithmetic,
all fences and GM/graph storage. Host checks do not certify device speed.
"""
from tools import active_store_transform as active
from tools import store_raw_reuse_transform as raw_reuse


def transform(text: str) -> str:
    return raw_reuse.transform(active.transform(text))


def restore(text: str) -> str:
    original = active.restore(raw_reuse.restore(text))
    if transform(original) != text:
        raise ValueError("official active store inverse is not exact")
    return original
