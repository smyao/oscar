# Archive #126/#145/#148/#164: preserve the measured source outside the
# complete-history finite check, original timer boundaries, ABI and errors.
"""Source-contract checks; actual A2 reduction parity needs the CANN probe."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tools.c16_softmax_transform import HEADER, MARKER, NEW, OLD, transform

ROOT = Path(__file__).resolve().parents[1]


def original_source():
    # Once integrated, the control source is the immutable pre-change input.
    control = ROOT / "csrc/kernels/attention_cv_aligned_cluster16_control.cpp"
    if control.is_file():
        return control.read_text()
    return (ROOT / "csrc/kernels/attention_cv_aligned_cluster16.cpp").read_text()


def test_only_complete_history_finite_branch_changes():
    original = original_source()
    candidate = transform(original)
    restored = candidate.removeprefix(HEADER)
    restored = restored.replace(
        '#include "attention_bf16_cast.h"\n#include "attention_c16_softmax_helpers.h"',
        '#include "attention_bf16_cast.h"', 1)
    restored = restored.replace(NEW, OLD, 1)
    assert restored == original
    assert candidate.count(MARKER) == 1
    # Precision, mask-domain and same-output protocol are inherited bytewise.
    for preserved in ("SoftmaxFlashV2<float,true,true,false,false,kCvSoftmaxConfig>",
                      "Fence<HardEvent::MTE3_MTE2>();",
                      "if(fullyVisibleHistory)", "oscar_range_mask::MaskComplement"):
        assert candidate.count(preserved) == original.count(preserved)


def test_reject_duplicate_transform_and_source_drift():
    original = original_source()
    with pytest.raises(ValueError, match="already applied"):
        transform(transform(original))
    with pytest.raises(ValueError, match="source contract drift"):
        transform(original.replace("ReduceSum(finiteSums[r*8]", "OtherReduce(finiteSums[r*8]"))
    with pytest.raises(ValueError, match="D256/KV256"):
        transform(original.replace("static_assert(D==256 && kKvRows==D,", "static_assert(D==128,"))


def test_existing_timer_generator_accepts_candidate_without_timer_changes(tmp_path, monkeypatch):
    path = ROOT / "csrc/experiments/c16_timer_20261019/generate.py"
    spec = importlib.util.spec_from_file_location("c16_original_timer_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidate = transform(original_source())
    # The unchanged generator resolves includes relative to SOURCE.parent.
    # Feed the candidate contents while retaining the real source's parent.
    real_source = ROOT / "csrc/kernels/attention_cv_aligned_cluster16.cpp"
    real_read = Path.read_text

    def read_text(self, *args, **kwargs):
        return candidate if self == real_source else real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    module.generate(tmp_path)
    generated = real_read(tmp_path / "c16_timer_kernel.cpp")
    assert "oscar_c16_softmax::FiniteRows256" in generated
    assert "timer,oscar_c16_timer::SoftmaxMath" in generated
    assert "timer,oscar_c16_timer::SoftmaxLoad" in generated
    assert "timer,oscar_c16_timer::SoftmaxPublish" in generated
    assert "timer,oscar_c16_timer::Bf16Cast" in generated
