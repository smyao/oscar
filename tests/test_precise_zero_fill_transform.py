"""Actual C4/C16 current-load control with dead-zero-write accounting.

Archive #126/#145/#150/#151/#154/#164/#172 and full startup D.4. Host
recording does not certify CANN conversion, asynchronous DMA or NPU speed.
"""
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from tools import precise_finite_transform as finite
from tools import precise_zero_fill_transform as zero


ROOT = Path(__file__).resolve().parents[1]


def source(family):
    mother = (ROOT / f"csrc/kernels/attention_cv_official_cluster{family}.cpp").read_text()
    if family == 4:
        from tools.cluster_visibility_transform import transform
        mother = transform(mother)
    return finite.transform(mother)


def method(text, name):
    first = text.index(f"  __aicore__ void {name}(")
    depth = 0
    for pos in range(text.index("{", first), len(text)):
        depth += (text[pos] == "{") - (text[pos] == "}")
        if depth == 0:
            return text[first:pos + 1]
    raise AssertionError(f"unclosed {name}")


@pytest.mark.parametrize("family", [4, 16])
def test_only_loadprecise_dead_writes_change_and_restore_exactly(family):
    old = source(family)
    old_method = method(old, "LoadPrecise")
    assert sha256(old_method.encode()).hexdigest() == zero.METHOD_SHA256
    new = zero.transform(old)
    assert zero.restore(new) == old
    restored = new.removeprefix(zero.HEADER).replace(method(new, "LoadPrecise"), old_method, 1)
    assert restored == old
    new_method = method(new, "LoadPrecise")
    assert new_method.replace(zero.NEW_DST, zero.OLD_DST).replace(zero.NEW_BATCH, zero.OLD_BATCH) == old_method
    for unchanged in ("LoadWindowRuns", "PublishKv", "VectorTask", "CubeTask", "Softmax"):
        assert method(old, unchanged) == method(new, unchanged)
    # Existing finite branch and every original error guard are unchanged.
    assert finite.NEW in new_method
    suffix = "    if(kind==1) {LoadWindowRuns(start,value);return;}"
    assert old_method[old_method.index(suffix):] == new_method[new_method.index(suffix):]


@pytest.mark.parametrize("family", [4, 16])
@pytest.mark.parametrize("drift", ["double", "cast", "fence", "row_guard", "geometry", "finite"])
def test_drift_is_rejected(family, drift):
    text = source(family)
    changed = {
        "double": lambda: zero.transform(text),
        "cast": lambda: text.replace("Cast(dst,batch,RoundMode::CAST_NONE,kElements)",
                                      "Cast(dst,batch,RoundMode::CAST_NONE,rows*D)", 1),
        "fence": lambda: text.replace(zero.OLD_DST, zero.OLD_DST.replace("PipeBarrier<PIPE_V>();", ""), 1),
        "row_guard": lambda: text.replace("if(rows==0)return;", "if(rows<=0)return;", 1),
        "geometry": lambda: text.replace("static_assert(D==256 && kHalfKv==16,", "static_assert(D==128,", 1),
        "finite": lambda: text.replace("FiniteRows256(check,dst,finiteWork,rows)",
                                       "FiniteRows256(check,dst,finiteWork,rows-1)", 1),
    }[drift]()
    with pytest.raises(ValueError, match="precise zero-fill"):
        zero.transform(changed)


@pytest.mark.parametrize("drift", ["header", "eligibility", "batch", "error"])
def test_inverse_rejects_noncanonical_modified_method(drift):
    text = zero.transform(source(16))
    changed = {
        "header": lambda: text.removeprefix(zero.HEADER),
        "eligibility": lambda: text.replace("if(writeRows>0)", "if(writeRows>=0)", 1),
        "batch": lambda: text.replace("if(rows<kHalfKv)", "if(rows<=kHalfKv)", 1),
        "error": lambda: text.replace("{error=1;return;}", "{error=4;return;}", 1),
    }[drift]()
    with pytest.raises(ValueError, match="precise zero-fill"):
        zero.restore(changed)


def helper(name):
    text = (ROOT / "csrc/kernels" / name).read_text()
    return "\n".join(line for line in text.splitlines()
                     if not line.startswith("#include") and line != "#pragma once")


def test_actual_complete_load_publish_and_exact_zero_byte_accounting(tmp_path):
    before = source(4)
    # C4/C16 use byte-identical LoadPrecise and PublishKv; run that real body.
    for name in ("LoadPrecise", "PublishKv"):
        assert method(before, name) == method(source(16), name)
    after = zero.transform(before)
    fixture = (ROOT / "tests/cpp/precise_zero_fill_control.cpp").read_text()
    replacements = {
        "@@ROW_HELPER@@": helper("attention_c16_softmax_helpers.h"),
        "@@CERTIFICATE_HELPER@@": helper("attention_official_finite.h"),
        "@@OLD_LOAD@@": method(before, "LoadPrecise").replace("LoadPrecise(", "LoadOld(", 1),
        "@@NEW_LOAD@@": method(after, "LoadPrecise").replace("LoadPrecise(", "LoadNew(", 1),
        "@@PUBLISH@@": method(before, "PublishKv"),
    }
    for marker, body in replacements.items():
        assert fixture.count(marker) == 1
        fixture = fixture.replace(marker, body)
    cpp, executable = tmp_path / "control.cpp", tmp_path / "control"
    cpp.write_text(fixture)
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "host control requires a compiler; it is not a CANN substitute"
    env = os.environ.copy()
    xcode = Path("/Applications/Xcode.app/Contents/Developer")
    if "DEVELOPER_DIR" not in env and xcode.is_dir():
        env["DEVELOPER_DIR"] = str(xcode)
    built = subprocess.run([
        compiler, "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror",
        "-fno-fast-math", "-fno-strict-aliasing", "-fsanitize=undefined",
        "-fno-sanitize-recover=undefined", str(cpp), "-o", str(executable),
    ], capture_output=True, text=True, timeout=60, env=env)
    assert built.returncode == 0, built.stdout + built.stderr
    run = subprocess.run([str(executable)], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    assert not run.stderr
    result = json.loads(run.stdout)
    assert result["cases"] == 8568
    for key in ("full", "partial", "invalid", "empty", "window"):
        assert result[key] > 0
    assert result["host_control_only"]
    assert result["fixed_address_metadata_refresh"]
    assert result["full_kv256_zero_bytes_removed"] == 768 * 1024
