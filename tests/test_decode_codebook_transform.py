"""Archive #126/#145/#154/#164/#172: history-only wiring and inverse checks.

The actual old/new helper arithmetic is exercised separately by the C++ host
instruction model. Here we protect all other methods, allocation and the
production same-ABI route; these checks are not device acceptance.
"""
from hashlib import sha256
from pathlib import Path
import re

import pytest

from tools import active_decode_transform as active
from tools import decode_codebook_transform as codebook
from tools import physical_page_cache_transform as page
from tools.cluster_state_transform import method
from tools import generate_active_kernels as generated


ROOT = Path(__file__).resolve().parents[1]


def baseline():
    mother = (ROOT / "csrc/kernels/attention_cv_official_decode.cpp").read_text()
    return page.transform(active.transform(mother))


def test_complete_inverse_and_only_history_pipeline_methods_change():
    original = baseline()
    assert sha256(original.encode()).hexdigest() == codebook.INPUT_SHA256
    derived = codebook.transform(original)
    assert codebook.restore(derived) == original
    assert derived == (ROOT / generated.KERNELS[0]["active"]).read_text()
    changes = {"PipelineUnpack", "PipelinePublishKv", "HistoryVectorTask"}
    names = re.findall(r"  __aicore__ (?:void|bool|int32_t) (\w+)\(", original)
    assert changes <= set(names)
    for name in names:
        kind = "bool" if name in ("Physical", "TaskValid") else "int32_t" if name == "TailColumns" else "void"
        if name not in changes:
            assert method(original, name, kind) == method(derived, name, kind), name
    # The public ABI and all existing buffers are byte-identical; the scratch
    # ownership proof is in the helper test, not a changed allocation budget.
    assert method(original, "Init") == method(derived, "Init")
    for name in ("PipelinePublishKv", "HistoryVectorTask"):
        fences = lambda s: re.findall(r"(?:Fence<[^>]+>\(\)|CrossCore\w+[^;]+);", method(s, name))
        assert fences(original) == fences(derived)
    old_publish = method(original, "PipelinePublishKv")
    new_publish = method(derived, "PipelinePublishKv")
    assert "CastRne" in old_publish and "CastRne" not in new_publish
    assert 'auto rounded=naturalBuf.Get<bfloat16_t>();' in new_publish
    assert 'dequantBuf.Get<float>()' not in new_publish
    # Generic current/window publication still converts its FP32 inputs.
    assert "CastRne" in method(derived, "PublishKv")
    assert "oscar_ascend_aligned::Unpack" in method(derived, "Unpack")
    assert "oscar_ascend_codebook::Unpack" in method(derived, "PipelineUnpack")
    history = method(derived, "HistoryVectorTask")
    assert "PipelineUnpack(packedBuf,false);PipelinePublishKv(false,sub,0);" in history
    assert "PipelineUnpack(packedBuf,true);PipelinePublishKv(true,sub,0);" in history
    assert method(derived, "PrefetchKeys").count("PipelineUnpack(packedInput,false)") == 1
    assert method(derived, "PrefetchValues").count("PipelineUnpack(packedInput,true)") == 1


@pytest.mark.parametrize("damage", ["input", "header", "publisher", "initial", "prefetch", "fence", "tail"])
def test_incomplete_or_drifted_wiring_is_rejected(damage):
    original = baseline()
    if damage == "input":
        with pytest.raises(ValueError):
            codebook.transform(original + "\n")
        return
    derived = codebook.transform(original)
    changed = {
        "header": lambda s: s.removeprefix(codebook.HEADER),
        "publisher": lambda s: s.replace(codebook._EDITS[2][1], codebook._EDITS[2][0], 1),
        "initial": lambda s: s.replace("PipelineUnpack(packedBuf,false)", "Unpack(false)", 1),
        "prefetch": lambda s: s.replace("PipelineUnpack(packedInput,false)", "Unpack(false)", 1),
        "fence": lambda s: s.replace("Fence<HardEvent::MTE3_V>();", "", 1),
        "tail": lambda s: s + "\n",
    }[damage](derived)
    with pytest.raises(ValueError):
        codebook.restore(changed)


def test_new_header_participates_in_build_and_all_numerical_probe_bindings():
    from tools.environment import file_fingerprint
    from tools import probe_aligned_attention, probe_official_attention, probe_prefix_cache
    name = "csrc/kernels/attention_codebook_unpack.h"
    expected = sha256((ROOT / name).read_bytes()).hexdigest()
    assert file_fingerprint(ROOT / "csrc")[name.removeprefix("csrc/")] == expected
    for probe in (probe_aligned_attention, probe_official_attention, probe_prefix_cache):
        assert probe.source_binding()[name] == expected
        assert "tools/decode_codebook_transform.py" in probe.source_binding()
