"""Archive #126/#145/#157/#164, D.4: active-slot writer control-flow checks.

The C++ harness executes extracted old/new production AlignedStore classes
with a deterministic quantizer stub, bounds-checked host copies, and inert
fences. It does not implement or qualify CANN arithmetic or DMA concurrency.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from tools.active_store_transform import DERIVED_HEADER, MOTHER_SHA256, restore, transform


ROOT = Path(__file__).resolve().parents[1]
MOTHER = ROOT / "csrc/kernels/store_aligned_int2.cpp"


def source() -> str:
    return MOTHER.read_text()


def extract_class(text: str, name: str) -> str:
    begin = text.index("class AlignedStore {")
    end = text.index("\n};\n}", begin) + len("\n};")
    return text[begin:end].replace("class AlignedStore {", f"class {name} {{", 1)


def test_transform_is_reversible_and_bound_to_the_full_mother():
    before = source()
    assert hashlib.sha256(before.encode()).hexdigest() == MOTHER_SHA256
    after = transform(before)
    assert after.startswith(DERIVED_HEADER)
    assert restore(after) == before
    with pytest.raises(ValueError, match="SHA256"):
        transform(before + "\n")
    with pytest.raises(ValueError, match="SHA256"):
        transform(after)
    with pytest.raises(ValueError, match="header"):
        restore(before)


@pytest.mark.parametrize("old,new", [
    ("if (slot<0)", "if (slot<=0)"),
    ("count*dim);PipeBarrier", "count*dim); PipeBarrier"),
    ("if (position<0)", "if (position<=0)"),
    ("errors[row]=0;", "errors[row]=1;"),
    ("cachedSlots[row]<0", "cachedSlots[row]<=0"),
    ("Fence<HardEvent::MTE3_MTE2>();", "Fence<HardEvent::MTE2_V>();"),
])
def test_restore_rejects_derived_drift(old, new):
    after = transform(source())
    assert old in after
    with pytest.raises(ValueError):
        restore(after.replace(old, new, 1))


def test_only_run_changes_and_public_abi_quantizer_copy_and_helpers_stay_exact():
    before = source()
    after = transform(before)[len(DERIVED_HEADER):]
    start = before.index("class AlignedStore {")
    private = before.index("\n private:", start)
    derived_private = after.index("\n private:", start)
    assert after[:start] == before[:start]  # Entire ClipQuantize body unchanged.
    assert after[derived_private:] == before[private:]  # DMA helpers, ABI and Q-copy unchanged.
    run_before = before[start:private]
    run_after = after[start:derived_private]
    for exact in (
        "DataCopy(input,(side==0?k_:v_)[base*dim],count*dim);",
        "Cast(values,input,RoundMode::CAST_NONE,count*dim);PipeBarrier<PIPE_V>();",
        "        Fence<HardEvent::V_MTE2>();",
        "        if (slot<0) {errors[row]=0;continue;}",
        "        if (slot>=blocks*blockTokens) {errors[row]=1;continue;}",
        "        if (position<0) {errors[row]=4;continue;}",
        "        if (errors[row]) continue;",
    ):
        assert run_before.count(exact) == run_after.count(exact) == 1
    assert run_after.count("Read(slots_,(base+row)/heads)") == 1
    assert "const int64_t slot=cachedSlots[row];" in run_after
    assert "if (cachedSlots[row]<0) continue;" in run_after


@pytest.fixture(scope="module")
def control_binary(tmp_path_factory):
    compiler = shutil.which("clang++")
    if compiler is None:
        pytest.fail("host clang++ is required for extracted C++ control-flow validation")
    build = tmp_path_factory.mktemp("active-store-control")
    before = source()
    (build / "active_store_classes.inc").write_text(
        extract_class(before, "MotherStore") + "\n" +
        extract_class(transform(before), "ActiveStore") + "\n")
    binary = build / "active_store_control"
    env = os.environ.copy()
    xcode = Path("/Applications/Xcode.app/Contents/Developer")
    if "DEVELOPER_DIR" not in env and xcode.is_dir():
        # Select an already installed toolchain for this child only. Some
        # hosts retain an xcode-select pointer to removed CommandLineTools.
        env["DEVELOPER_DIR"] = str(xcode)
    result = subprocess.run(
        [compiler, "-std=c++17", "-O0", "-Wall", "-Wextra", "-Werror",
         "-I", str(build), str(ROOT / "tests/cpp/active_store_control.cpp"),
         "-o", str(binary)],
        text=True, capture_output=True, timeout=60, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    return binary


@pytest.mark.parametrize("case", [
    "all_live", "interleaved", "all_padding", "many_heads", "tail_tile",
    "errors", "distinct_originals", "same_address_replay", "zero_tokens",
    "bs16", "bs28",
])
def test_extracted_run_preserves_outputs_and_nonnegative_quantizer_sequence(control_binary, case):
    result = subprocess.run([str(control_binary), case], text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"{case} passed" in result.stdout
    if case == "bs16":
        assert "old_quant=128 active_quant=32" in result.stdout
    elif case == "bs28":
        assert "old_quant=224 active_quant=56" in result.stdout
    elif case in {"all_padding", "zero_tokens"}:
        assert "active_quant=0" in result.stdout
