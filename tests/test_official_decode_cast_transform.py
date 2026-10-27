"""Archive #126/#145/#164: publication binding and unchanged DMA/error domain.

These source checks protect exact transformation scope. They do not replace
official CPU-debug comparison, device compilation, NPU or graph validation.
"""
from pathlib import Path

import pytest

from tools.official_decode_cast_transform import transform
from tools.generate_official_attention import decode

ROOT = Path(__file__).resolve().parents[1]
HELPER_BLOCK = '''
// Archive #126/#145/#164, D.4: bounded q4/q1 publication only; preserve
// all fences and numerical work, with no history restore or extra UB/GM.
// The normal-mask helper batches at most255 repeats, including D256/HalfKv64.
#include "attention_c16_cast.h"'''
PUBLICATIONS = (
    "CastRne(rounded,src,kElements);",
    "CastRne(rounded,scores,activeRows*kKvRows);",
    "CastRne(rounded,dequantBuf.Get<float>(),kElements);",
)


def original_source():
    return decode(optimize=False)


def test_only_three_publications_change_and_headers_keep_original_binding():
    original = original_source()
    candidate = transform(original)
    assert candidate.count("oscar_c16_cast::CastRne(") == 3
    assert candidate.count("oscar_bf16::CastRne(") == 1
    assert "oscar_bf16::CastRne(rounded,zeros,count*D);" in candidate
    assert "#define oscar_bf16" not in candidate
    helper = candidate.index('#include "attention_c16_cast.h"')
    assert helper > candidate.index('#include "attention_aligned_unpack.h"')
    assert "#include" not in candidate[helper + len('#include "attention_c16_cast.h"'):]
    restored = candidate.replace(HELPER_BLOCK, "")
    for call in PUBLICATIONS:
        assert candidate.count("oscar_c16_cast::" + call) == 1
        restored = restored.replace("oscar_c16_cast::" + call, "oscar_bf16::" + call)
    # #145: every existing DMA fence, cross-core flag, address and error
    # branch survives byte-for-byte; even the padding helper stays unchanged.
    assert restored == original


@pytest.mark.parametrize("drift", [
    "missing", "extra", "changed", "padding", "include", "late_include", "twice",
])
def test_rejects_changed_or_already_transformed_source(drift):
    source = original_source()
    call = "oscar_bf16::CastRne(rounded,src,kElements);"
    if drift == "missing":
        source = source.replace(call, "")
    elif drift == "extra":
        source += "\n" + call
    elif drift == "changed":
        source = source.replace(call, call.replace("kElements", "kElements+1"))
    elif drift == "padding":
        source = source.replace("rounded,zeros,count*D", "rounded,zeros,count*D+1")
    elif drift == "include":
        source = source.replace('#include "attention_bf16_cast.h"', "")
    elif drift == "late_include":
        source += '\n#include "attention_aligned_unpack.h"\n'
    else:
        source = transform(source)
    with pytest.raises(ValueError, match="official decode cast"):
        transform(source)


@pytest.mark.parametrize("family", ["aligned_decode", "official_cluster4", "official_cluster16"])
def test_rejects_other_numerical_families(family):
    source = (ROOT / f"csrc/kernels/attention_cv_{family}.cpp").read_text()
    with pytest.raises(ValueError, match="requires q4/q1"):
        transform(source)
