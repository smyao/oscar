"""Actual window reader/helper host controls; no CANN/NPU/performance claim.

Archive #126/#145/#150/#151/#154/#164/#172 and full startup D.4 reviewed.
The baseline source is reconstructed independently of the composing generator.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from tools import cluster_visibility_transform as visibility
from tools import precise_finite_transform as precise
from tools import precise_zero_fill_transform as zero
from tools import cluster_state_transform as state
from tools import cluster4_l1_transform as l1
from tools import cluster_score_ring_transform as ring
from tools import physical_page_cache_transform as physical
from tools import window_finite_transform as window

ROOT = Path(__file__).resolve().parents[1]


def baseline(family):
    source = (ROOT / f"csrc/kernels/attention_cv_official_cluster{family}.cpp").read_text()
    if family == 4:
        source = visibility.transform(source)
    source = state.transform(zero.transform(precise.transform(source)))
    source = l1.transform(source) if family == 4 else ring.transform(source)
    return physical.transform(source)


@pytest.mark.parametrize("family", [4, 16])
def test_complete_source_inverse_only_changes_ordered_window_checks(family):
    source = baseline(family)
    assert sha256(source.encode()).hexdigest() == window.INPUT_SHA256[family]
    candidate = window.transform(source)
    assert window.restore(candidate) == source
    assert candidate.removeprefix(window.HEADER).replace(window.NEW, window.OLD, 1) == source
    for name in ("LoadPrecise", "LoadPacked", "PublishKv", "PublishHistoryKv", "VectorTask",
                 "VectorCluster", "Softmax", "CubeTask", "CubeCluster", "Init", "Process"):
        assert state.method(candidate, name) == state.method(source, name)
    before, after = (state.method(text, "LoadWindowRuns") for text in (source, candidate))
    assert before.replace(window.OLD, window.NEW, 1) == after
    assert after.count("Cast(dst[j*D],batch,RoundMode::CAST_NONE,rows*D);PipeBarrier<PIPE_V>();") == 1
    assert "Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();\n      j+=rows;" in after
    assert "if(tags.GetValue(row)!=inpage+row) {error=4;++row;continue;}" in after
    assert after.index("Cast(dst[j*D],batch") < after.index("auto finiteWork=naturalBuf")


@pytest.mark.parametrize("fault", ["double", "cast_fence", "tag", "physical", "math", "tail"])
def test_source_drift_rejected(fault):
    source = baseline(4)
    changed = {
        "double": lambda: window.transform(source),
        "cast_fence": lambda: source.replace("Cast(dst[j*D],batch,RoundMode::CAST_NONE,rows*D);PipeBarrier<PIPE_V>();",
                                              "Cast(dst[j*D],batch,RoundMode::CAST_NONE,rows*D);", 1),
        "tag": lambda: source.replace("if(tags.GetValue(row)!=inpage+row)", "if(false)", 1),
        "physical": lambda: source.replace("if(!Physical(position,block,inpage))", "if(false)", 1),
        "math": lambda: source.replace("ReduceSum(check,dst[(j+row)*D],check[16],D)",
                                        "ReduceSum(check,dst[(j+row)*D],check[16],D-1)", 1),
        "tail": lambda: source + "\n",
    }[fault]()
    with pytest.raises(ValueError):
        window.transform(changed)


@pytest.mark.parametrize("fault", ["header", "run", "order", "capacity", "tail"])
def test_inverse_drift_rejected(fault):
    candidate = window.transform(baseline(16))
    changed = {
        "header": lambda: candidate.removeprefix(window.HEADER),
        "run": lambda: candidate.replace("finiteWork,end-row)", "finiteWork,rows)", 1),
        "order": lambda: candidate.replace("{error=4;++row;continue;}", "{++row;continue;}", 1),
        "capacity": lambda: candidate.replace("auto finiteWork=naturalBuf.Get<float>(kElements);",
                                             "auto finiteWork=naturalBuf.Get<float>(1);", 1),
        "tail": lambda: candidate + "\n",
    }[fault]()
    with pytest.raises(ValueError):
        window.restore(changed)


def header(name, dependency):
    text = (ROOT / "csrc/kernels" / name).read_text()
    assert text.count(f'#include "{dependency}"') == 1
    return text.replace("#pragma once", "").replace(f'#include "{dependency}"', "")


@pytest.fixture(scope="module")
def host_results(tmp_path_factory):
    old = baseline(4)
    new = window.transform(old)
    assert state.method(old, "LoadWindowRuns") == state.method(baseline(16), "LoadWindowRuns")
    fixture = (ROOT / "tests/cpp/window_finite_control.cpp").read_text()
    replacements = {
        "@@ROW_HELPER@@": header("attention_c16_softmax_helpers.h", "oscar_common.h"),
        "@@CERTIFICATE_HELPER@@": header("attention_official_finite.h", "attention_c16_softmax_helpers.h"),
        "@@LOGICAL_POSITION@@": state.method(old, "LogicalPosition", "int64_t"),
        "@@PHYSICAL@@": state.method(old, "Physical", "bool"),
        "@@OLD_METHOD@@": state.method(old, "LoadWindowRuns").replace("LoadWindowRuns(", "Old(", 1),
        "@@NEW_METHOD@@": state.method(new, "LoadWindowRuns").replace("LoadWindowRuns(", "New(", 1),
    }
    for marker, code in replacements.items():
        assert fixture.count(marker) == 1
        fixture = fixture.replace(marker, code)
    assert "@@" not in fixture
    directory = tmp_path_factory.mktemp("window-finite")
    cpp, exe = directory / "control.cpp", directory / "control"
    cpp.write_text(fixture)
    compiler = shutil.which("clang++")
    assert compiler, "actual window controls require a host C++ compiler"
    env = os.environ.copy()
    xcode = Path("/Applications/Xcode.app/Contents/Developer")
    if "DEVELOPER_DIR" not in env and xcode.is_dir():
        env["DEVELOPER_DIR"] = str(xcode)
    built = subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-Wno-unused-variable",
                            "-fno-fast-math", "-fno-strict-aliasing", "-fsanitize=undefined",
                            "-fno-sanitize-recover=undefined", str(cpp), "-o", str(exe)],
                           capture_output=True, text=True, timeout=60, env=env)
    assert built.returncode == 0, built.stdout + built.stderr
    results = []
    for order in (0, 1):
        ran = subprocess.run([str(exe), str(order)], capture_output=True, text=True, timeout=60)
        assert ran.returncode == 0, ran.stdout + ran.stderr
        assert not ran.stderr
        results.append(json.loads(ran.stdout))
    result = results[0].copy()
    for key in ("cases", "reduction_orders", "certificate_cases", "fallback_cases", "tag_cases", "physical_cases"):
        result[key] = sum(part[key] for part in results)
    return result


def test_actual_window_reader_preserves_ordered_errors_bits_and_memory_reads(host_results):
    assert host_results["host_control_only"]
    assert host_results["cases"] > 2000 and host_results["reduction_orders"] == 2
    assert host_results["certificate_cases"] > 0 and host_results["fallback_cases"] > 0
    assert host_results["tag_cases"] > 0 and host_results["physical_cases"] > 0
    assert host_results["value_bits_and_errors"] == "identical"
    assert host_results["DMA_Cast_fences"] == "identical_except_finite_V_S"
    assert host_results["guards"] == "passed"
    assert host_results["CANN"] == host_results["NPU"] == host_results["performance"] == "not_run"
