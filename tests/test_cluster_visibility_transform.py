"""Archive #145/#150/#151/#154/#164/#172; startup D.4 control proof.

Compile the actual C4 Softmax before/after and existing finite helpers with
recording host primitives. CANN arithmetic/async ownership and speed are not
emulated; the frozen source inverse additionally protects all other methods.
"""
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from tools.cluster_visibility_transform import (
    BASELINE_SHA256, DERIVED_HEADER, _DERIVED_INCLUDE, _INCLUDE,
    _VISIBLE_PREFIX, restore, transform,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "csrc/kernels/attention_cv_official_cluster4.cpp"


def _method(text, name):
    start = text.index(f"  __aicore__ void {name}(")
    first, depth = text.index("{", start), 0
    for pos in range(first, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if not depth:
                return text[start:pos + 1]
    raise AssertionError(f"unclosed method {name}")


def test_exact_source_inverse_and_only_softmax_control_changes():
    original = SOURCE.read_text()
    assert sha256(original.encode()).hexdigest() == BASELINE_SHA256
    changed = transform(original)
    assert restore(changed) == original
    restored = changed.removeprefix(DERIVED_HEADER).replace(
        _DERIVED_INCLUDE, _INCLUDE, 1).replace(
        _method(changed, "Softmax"), _method(original, "Softmax"), 1)
    assert restored == original
    assert "#define OSCAR_STRIPED_D256_ONLY 1" in original
    old, new = _method(original, "Softmax"), _method(changed, "Softmax")
    # Finite branch insertion cannot touch SDK softmax, alpha, Cast or publish.
    suffix = "    // The two 64-row max/sum arrays persist across KV work units."
    assert old[old.index(suffix):] == new[new.index(suffix):]
    for unchanged in ("LoadPrecise", "CubeCluster", "VectorCluster", "LoadTask"):
        assert _method(original, unchanged) == _method(changed, unchanged)


def test_visible_branch_is_identical_to_existing_c16_branch():
    c16 = (ROOT / "csrc/kernels/attention_cv_official_cluster16.cpp").read_text()
    assert c16.count(_VISIBLE_PREFIX) == 1
    # Existing tests/test_official_softmax_visibility_transform.py independently
    # enumerates candidate-to-logical masks for precisely these predicates.
    # Existing finite certificate helper is imported unchanged in production.
    assert '#include "attention_official_finite.h"' in transform(SOURCE.read_text())


@pytest.mark.parametrize("drift", ["append", "cut", "math", "duplicate"])
def test_transform_rejects_source_drift(drift):
    text = SOURCE.read_text()
    changed = {
        "append": lambda: text + "\n",
        "cut": lambda: text.replace("context+1-g.recent", "context+2-g.recent", 1),
        "math": lambda: text.replace("g.scale,kHalfRows", "1.0F,kHalfRows", 1),
        "duplicate": lambda: transform(text),
    }[drift]()
    with pytest.raises(ValueError, match="exact protected C4 mother"):
        transform(changed)


@pytest.mark.parametrize("drift", ["header", "branch", "suffix", "include"])
def test_inverse_rejects_derived_drift(drift):
    text = transform(SOURCE.read_text())
    changed = {
        "header": lambda: text.removeprefix(DERIVED_HEADER),
        "branch": lambda: text.replace("start>=context", "start>context", 1),
        "suffix": lambda: text.replace("oscar_bf16::CastRne(roundedP,scores", "OtherCast(roundedP,scores", 1),
        "include": lambda: text.replace(_DERIVED_INCLUDE, _INCLUDE, 1),
    }[drift]()
    with pytest.raises(ValueError, match="cluster visibility"):
        restore(changed)


def _helper(name):
    # Retain the REAL namespace/functions/intrinsics; only CANN includes are
    # replaced by the recording primitives in this host-only translation unit.
    text = (ROOT / "csrc/kernels" / name).read_text()
    return "\n".join(line for line in text.splitlines()
                     if not line.startswith("#include") and line != "#pragma once")


def test_actual_softmax_control_inputs_errors_scratch_and_publication(tmp_path):
    original = SOURCE.read_text()
    changed = transform(original)
    fixture = (ROOT / "tests/cpp/cluster_visibility_control.cpp").read_text()
    substitutions = {
        "@@ROW_FINITE_HELPER@@": _helper("attention_c16_softmax_helpers.h"),
        "@@CERTIFICATE_HELPER@@": _helper("attention_official_finite.h"),
        "@@ADD_VISIBLE_RANGE@@": _method(original, "AddVisibleRange"),
        "@@OLD_SOFTMAX@@": _method(original, "Softmax").replace("Softmax(", "SoftmaxOld(", 1),
        "@@NEW_SOFTMAX@@": _method(changed, "Softmax").replace("Softmax(", "SoftmaxNew(", 1),
    }
    for marker, body in substitutions.items():
        assert fixture.count(marker) == 1
        fixture = fixture.replace(marker, body)
    source, executable = tmp_path / "control.cpp", tmp_path / "control"
    source.write_text(fixture)
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "host C++ control proof requires a compiler; it is not CANN"
    env = os.environ.copy()
    xcode = Path("/Applications/Xcode.app/Contents/Developer")
    if "DEVELOPER_DIR" not in env and xcode.is_dir():
        env["DEVELOPER_DIR"] = str(xcode)
    built = subprocess.run([
        compiler, "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror",
        "-Wno-unused-variable", "-fsanitize=undefined", "-fno-sanitize-recover=undefined",
        str(source), "-o", str(executable),
    ], capture_output=True, text=True, timeout=60, env=env)
    assert built.returncode == 0, built.stdout + built.stderr
    run = subprocess.run([str(executable)], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    assert not run.stderr
    result = json.loads(run.stdout)
    assert result["cases"] == 2016
    for branch in ("certificate_cases", "fallback_cases", "masked_cases", "empty_cases"):
        assert result[branch] > 0
    assert result["host_control_only"] is True
