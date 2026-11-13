"""Archive #145/#154/#164/#172: reversible same-ABI C4/C16 derivatives.

D.4(1): finite checks and bounded data movement in fused prefill/mixed CV.
D.4(2): the historical 6.5-second full-history restore remains prohibited.
D.4(3): compose finite checks, overwritten current zero fills, initial state
transfers, C4 B1 reuse and C16 score-slot reuse; numerical work and routing stay.
D.4(4): C4 adds one128KiB B1, no extra GM/UB or launch; host proof is not
device acceptance. C16 keeps its allocator and all existing cross flags.
"""
from __future__ import annotations

from hashlib import sha256

from tools import cluster_visibility_transform as visibility
from tools import precise_finite_transform as precise
from tools import precise_zero_fill_transform as zero_fill
from tools import cluster_state_transform as state
from tools import cluster4_l1_transform as c4_l1
from tools import cluster_score_ring_transform as score_ring


MOTHERS = {
    "5069fa48b89a72d78b269b42c50be833007dd24e3bd05eceda0a62ceec722858": "c4",
    "1bcf9ccfe0181f3103b255f67589845a9d83b4e232baa9aa11561b61326d8264": "c16",
}


def _family(text: str) -> str:
    family = MOTHERS.get(sha256(text.encode()).hexdigest())
    if family is None:
        raise ValueError("cluster prefill requires an exact protected C4/C16 mother")
    return family


def transform(text: str) -> str:
    family = _family(text)
    finite = precise.transform(visibility.transform(text) if family == "c4" else text)
    active = state.transform(zero_fill.transform(finite))
    return c4_l1.transform(active) if family == "c4" else score_ring.transform(active)


def restore(text: str) -> str:
    if text.startswith(c4_l1.HEADER):
        active = c4_l1.restore(text)
    else:
        active = score_ring.restore(text)
    original = precise.restore(zero_fill.restore(state.restore(active)))
    if original.startswith(visibility.DERIVED_HEADER):
        original = visibility.restore(original)
    _family(original)
    if transform(original) != text:
        raise ValueError("cluster prefill derived source is not canonical")
    return original
