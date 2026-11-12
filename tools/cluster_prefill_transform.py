"""Archive #145/#154/#164/#172: reversible same-ABI C4/C16 derivatives.

D.4(1): finite checks within the existing fused prefill/mixed CV kernels.
D.4(2): the historical 6.5-second full-history restore remains prohibited.
D.4(3): compose only complete-visible C4 checks and current K/V checks;
all numerical work, task routing and ownership stay in the protected mothers.
D.4(4): no added buffer or launch; host proof is not device acceptance.
"""
from __future__ import annotations

from hashlib import sha256

from tools import cluster_visibility_transform as visibility
from tools import precise_finite_transform as precise


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
    return precise.transform(visibility.transform(text) if family == "c4" else text)


def restore(text: str) -> str:
    original = precise.restore(text)
    if original.startswith(visibility.DERIVED_HEADER):
        original = visibility.restore(original)
    _family(original)
    if transform(original) != text:
        raise ValueError("cluster prefill derived source is not canonical")
    return original
