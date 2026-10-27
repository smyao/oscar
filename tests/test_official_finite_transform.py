"""Archive #126/#134/#145/#148/#164: source identity around bounded proof.

The real CANN probe separately checks arithmetic/errors and memory guards.
These tests prevent the generator changing any other statement or old helper.
"""
import pytest

from tools import generate_official_attention as generated
from tools import official_finite_transform as finite
from tools.official_softmax_visibility_transform import transform as visibility


def test_certificate_only_replaces_complete_tile_check_and_adds_header():
    old = visibility(generated.cluster(16, optimize=False))
    new = finite.transform(old)
    restored = new.replace(finite.NEW, finite.OLD).replace(
        '#include "attention_c16_softmax_helpers.h"\n#include "attention_official_finite.h"',
        '#include "attention_c16_softmax_helpers.h"')
    assert restored == old
    assert new == generated.cluster(16)


def test_certificate_rejects_double_application_and_anchor_drift():
    old = visibility(generated.cluster(16, optimize=False))
    for source in (finite.transform(old), old.replace(finite.OLD, ''), old + finite.OLD):
        with pytest.raises(ValueError):
            finite.transform(source)
