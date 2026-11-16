"""Archive #126/#145/#154/#155/#164/#172, D.4: raw writer input reuse.

These tests run the complete extracted before/after AlignedStore classes.
The destructive quantizer stub tests buffer independence, not CANN math;
recorded fences test retained ordering, not asynchronous device execution.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from tools.active_store_transform import transform as active_transform
from tools.store_raw_reuse_transform import DERIVED_HEADER, INPUT_SHA256, restore, transform


ROOT = Path(__file__).resolve().parents[1]


def source() -> str:
    # Derive from the frozen mother so integration may regenerate the active
    # file without making the test accidentally compare candidate to itself.
    return active_transform((ROOT / "csrc/kernels/store_aligned_int2.cpp").read_text())


def extract_class(text: str, name: str) -> str:
    begin = text.index("class AlignedStore {")
    end = text.index("\n};\n}", begin) + len("\n};")
    return text[begin:end].replace("class AlignedStore {", f"class {name} {{", 1)


def test_whole_source_identity_reversible_composition():
    before = source()
    assert hashlib.sha256(before.encode()).hexdigest() == INPUT_SHA256
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
    ("key==originalKey && value==originalValue", "key==originalKey || value==originalValue"),
    ("kRowTile*kMaxDim*2);\n    const bool reuseExact", "kMaxDim*2);\n    const bool reuseExact"),
    ("(reuseExact && side==0)", "(reuseExact && side==1)"),
    ("[row*dim];", "[0];"),
    ("Fence<HardEvent::MTE3_MTE2>();", "Fence<HardEvent::MTE3_S>();"),
    ("if (position<0)", "if (position<=0)"),
    ("float range=maximum-minimum;", "float range=maximum+minimum;"),
])
def test_restore_rejects_all_drift(old, new):
    derived = transform(source())
    assert old in derived
    with pytest.raises(ValueError):
        restore(derived.replace(old, new, 1))


def test_entire_quantizer_query_bridge_and_public_abi_are_unchanged():
    before = source()
    after = transform(before)[len(DERIVED_HEADER):]
    start = before.index("class AlignedStore {")
    before_end = before.index("\n};\n}", start) + len("\n};")
    after_end = after.index("\n};\n}", start) + len("\n};")
    assert before[:start] == after[:start]
    assert before[before_end:] == after[after_end:]
    old = extract_class(before, "Store")
    new = extract_class(after, "Store")
    assert re.findall(r"Fence<[^;]+;|PipeBarrier<[^;]+;", old) == re.findall(
        r"Fence<[^;]+;|PipeBarrier<[^;]+;", new)
    for exact in (
        "DataCopy(input,(side==0?k_:v_)[base*dim],count*dim);",
        "Cast(values,input,RoundMode::CAST_NONE,count*dim);PipeBarrier<PIPE_V>();",
        "if (cachedSlots[row]<0) continue;",
        "if (slot>=blocks*blockTokens) {errors[row]=1;continue;}",
        "if (position<0) {errors[row]=4;continue;}",
        "if (errors[row]) continue;",
        "if (later>=0 && later/blockTokens==page && later==slot+recent) keepRecent=false;",
        "DataCopy(converted,(side==0?originalK_:originalV_)[(token*heads+head)*dim],dim);",
    ):
        assert old.count(exact) == new.count(exact) == 1
    old_read = old[old.index("  __aicore__ inline int64_t Read("):old.index("  __aicore__ inline void WriteRaw(")]
    new_read = new[new.index("  __aicore__ inline int64_t Read("):new.index("  __aicore__ inline void WriteRaw(")]
    assert old_read == new_read


def test_production_alias_qualification_and_input_immutability_contract():
    math = (ROOT / "oscar_ascend/integration/official_math.py").read_text()
    binding = (ROOT / "csrc/aligned_store_bindings.cpp").read_text()
    assert "ops.store_aligned_int2_out(kr, vr, kr, vr, slots, positions," in math
    assert "for (const auto& source:{key,value,rk,rv,slots,positions})" in binding
    assert "NoOverlap(Bytes(source),packedActive);NoOverlap(source,status);" in binding
    assert "NoOverlap(source,rawKey);NoOverlap(source,rawValue);NoOverlap(source,tags);" in binding
    assert 'rk.sizes()==key.sizes() && rv.sizes()==key.sizes()' in binding
    assert '"original_key");Tensor(rv,key,at::kBFloat16,"original_value")' in binding


@pytest.fixture(scope="module")
def control_binary(tmp_path_factory):
    compiler = shutil.which("clang++")
    if compiler is None:
        pytest.fail("host clang++ is required for full extracted writer verification")
    build = tmp_path_factory.mktemp("store-raw-reuse")
    before = source()
    (build / "store_raw_reuse_classes.inc").write_text(
        extract_class(before, "BeforeStore") + "\n" +
        extract_class(transform(before), "ReuseStore") + "\n" +
        extract_class(transform(before), "UnsafeReuseStore").replace(
            "Fence<HardEvent::MTE3_MTE2>();", "Fence<HardEvent::V_MTE2>();", 1) + "\n")
    binary = build / "store_raw_reuse_control"
    env = os.environ.copy()
    xcode = Path("/Applications/Xcode.app/Contents/Developer")
    if "DEVELOPER_DIR" not in env and xcode.is_dir():
        env["DEVELOPER_DIR"] = str(xcode)
    result = subprocess.run(
        [compiler, "-std=c++17", "-O0", "-Wall", "-Wextra", "-Werror",
         "-I", str(build), str(ROOT / "tests/cpp/store_raw_reuse_control.cpp"),
         "-o", str(binary)],
        text=True, capture_output=True, timeout=60, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    return binary


@pytest.mark.parametrize("case", [
    "all_live", "distinct_originals", "key_alias_only", "value_alias_only",
    "same_kv_source", "many_heads", "tail_tile", "one_row", "interleaved",
    "all_padding", "errors", "same_address_replay", "zero_tokens",
    "sink_only", "recent_only", "page_boundary", "bs16", "bs28",
])
def test_actual_classes_preserve_bytes_errors_fences_and_only_remove_reloads(control_binary, case):
    result = subprocess.run([str(control_binary), case], text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"{case} passed" in result.stdout
    if case in {"distinct_originals", "key_alias_only", "value_alias_only", "all_padding", "zero_tokens"}:
        assert "saved_bytes=0" in result.stdout
    elif case == "bs16":
        assert "quant=32" in result.stdout and "saved_bytes=16384" in result.stdout
    elif case == "bs28":
        assert "quant=56" in result.stdout and "saved_bytes=28672" in result.stdout


def test_negative_control_detects_unfenced_mte3_source_reuse(control_binary):
    result = subprocess.run([str(control_binary), "ownership_negative"],
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 1
    assert "UB overwritten before MTE3 completion" in result.stderr
