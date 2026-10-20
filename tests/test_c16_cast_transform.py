"""Archive #164/#126/#145: isolate the audited conversion change to C16."""
from pathlib import Path

import pytest

from tools.c16_cast_transform import transform

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "csrc/kernels/attention_cv_aligned_cluster16_control.cpp"
ALIAS_BLOCK = '\n#include "attention_c16_cast.h"\n#define oscar_bf16 oscar_c16_cast'
UNDEF = "\n#undef oscar_bf16\n"


def test_c16_cast_preserves_all_timer_anchors_and_header_binding():
    original = CONTROL.read_text()
    candidate = transform(original)
    assert candidate.count("oscar_bf16::CastRne(") == 6
    assert "oscar_c16_cast::CastRne(" not in candidate
    assert '#include "attention_bf16_cast.h"' in candidate
    alias_position = candidate.index("#define oscar_bf16 oscar_c16_cast")
    assert "#include" not in candidate[alias_position:]
    assert candidate.endswith(UNDEF)
    restored = candidate.removesuffix(UNDEF).replace(ALIAS_BLOCK, "")
    # #126/#145: every lifetime fence, address, arithmetic and error branch
    # outside the six helper calls remains byte-for-byte identical.
    assert restored == original


def test_c16_cast_rejects_c4():
    source = (ROOT / "csrc/kernels/attention_cv_aligned_cluster4.cpp").read_text()
    with pytest.raises(ValueError, match="aligned cluster16"):
        transform(source)


@pytest.mark.parametrize("drift", ["missing", "extra", "changed", "include", "twice"])
def test_c16_cast_rejects_changed_or_already_transformed_source(drift):
    source = CONTROL.read_text()
    call = "oscar_bf16::CastRne(rounded,dst,D);"
    if drift == "missing":
        source = source.replace(call, "")
    elif drift == "extra":
        source += "\n" + call
    elif drift == "changed":
        source = source.replace(call, "oscar_bf16::CastRne(rounded,dst,D+1);")
    elif drift == "include":
        source = source.replace('#include "attention_bf16_cast.h"', "")
    else:
        source = transform(source)
    with pytest.raises(ValueError, match="C16 cast"):
        transform(source)
