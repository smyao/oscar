"""Archive #145/#150/#151/#154/#164/#172 and full startup D.4.

Compile actual finite helpers and emitted old/new blocks with bounded host
stubs. This checks source scope, predicates, scratch bounds and FP32 inputs;
it does not establish CANN reduction order, asynchronous ownership or NPU speed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from tools import precise_finite_transform as precise


ROOT = Path(__file__).resolve().parents[1]


def mother(family: int) -> str:
    return (ROOT / f"csrc/kernels/attention_cv_official_cluster{family}.cpp").read_text()


@pytest.mark.parametrize("family", [4, 16])
def test_only_current_finite_validation_changes_with_exact_inverse(family):
    original = mother(family)
    candidate = precise.transform(original)
    assert precise.restore(candidate) == original
    restored = candidate.removeprefix(precise.HEADER).replace(
        precise.NEW, precise.OLD, 1).replace(precise.DEPENDENCY, "", 1)
    assert restored == original
    original_start, original_end = precise._bounds(original)
    candidate_start, candidate_end = precise._bounds(candidate)
    before, after = original[original_start:original_end], candidate[candidate_start:candidate_end]
    assert before.replace(precise.OLD, precise.NEW, 1) == after
    # Finite work follows the unchanged completed input Cast. The two zero
    # initializations and all early-return errors deliberately remain intact.
    assert "PipeBarrier<PIPE_V>();\n" + precise.NEW in after
    for statement in ("if(rows==0)return;", "if(token<0 || token+rows>g.tokens)",
                      "Duplicate(dst,0.0F,kElements)", "Duplicate(batch.ReinterpretCast",
                      "Fence<HardEvent::V_MTE2>();", "DataCopyPad(batch,cv[off]",
                      "if(kind==1) {LoadWindowRuns(start,value);return;}"):
        assert before.count(statement) == after.count(statement)
    assert after[after.index("    if(kind==1)"):] == before[before.index("    if(kind==1)"):]
    assert candidate.count('#include "attention_official_finite.h"') == 1


def test_transform_composes_after_existing_finite_dependency():
    # Complete C4 with a preceding transformation's dependency is accepted;
    # precise restore must not remove the dependency that it did not add.
    original = mother(4).replace('#include "oscar_common.h"',
        '#include "attention_official_finite.h"\n#include "oscar_common.h"', 1)
    candidate = precise.transform(original)
    assert precise.DEPENDENCY not in candidate
    assert precise.restore(candidate) == original


def test_actual_c4_visibility_and_precise_transforms_compose_with_exact_inverse():
    from tools import cluster_visibility_transform as visibility

    original = mother(4)
    visible = visibility.transform(original)
    candidate = precise.transform(visible)
    assert precise.DEPENDENCY not in candidate
    assert precise.restore(candidate) == visible
    assert visibility.restore(precise.restore(candidate)) == original


@pytest.mark.parametrize("family", [4, 16])
@pytest.mark.parametrize("drift", ["duplicate", "row_count", "cast_fence", "window", "guard", "scratch", "family"])
def test_source_contract_drift_is_rejected(family, drift):
    original = mother(family)
    changed = {
        "duplicate": lambda: precise.transform(original),
        "row_count": lambda: original.replace("kHalfKv=16;", "kHalfKv=32;", 1),
        "cast_fence": lambda: original.replace(
            "Cast(dst,batch,RoundMode::CAST_NONE,kElements);\n      PipeBarrier<PIPE_V>();",
            "Cast(dst,batch,RoundMode::CAST_NONE,kElements);", 1),
        "window": lambda: original.replace("if(kind==1) {LoadWindowRuns(start,value);return;}", "if(kind==1)return;", 1),
        "guard": lambda: original.replace("if(rows==0)return;", "if(rows<0)return;", 1),
        "scratch": lambda: original.replace("pipe.InitBuffer(scratchBuf,kHistoryHalfKv*8*4);",
                                             "pipe.InitBuffer(scratchBuf,128);", 1),
        "family": lambda: original.replace(f"oscar_attention_cv_official_cluster{family}_kernel", "unknown_kernel"),
    }[drift]()
    with pytest.raises(ValueError):
        precise.transform(changed)


@pytest.mark.parametrize("drift", ["header", "body", "capacity"])
def test_restore_rejects_changed_precise_contract(drift):
    candidate = precise.transform(mother(16))
    changed = {
        "header": lambda: candidate.removeprefix(precise.HEADER),
        "body": lambda: candidate.replace("FiniteRows256(check,dst,finiteWork,rows)",
                                          "FiniteRows256(check,dst,finiteWork,rows-1)", 1),
        "capacity": lambda: candidate.replace("kElements*sizeof(float)", "1*sizeof(float)", 1),
    }[drift]()
    with pytest.raises(ValueError):
        precise.restore(changed)


def _actual_header(name: str, dependency: str) -> str:
    source = (ROOT / "csrc/kernels" / name).read_text()
    assert source.count(f'#include "{dependency}"') == 1
    # Only dependency directives are removed. Both real helper function bodies
    # compile unchanged against explicit host primitive stubs in this test.
    return source.replace("#pragma once", "").replace(f'#include "{dependency}"', "")


@pytest.fixture(scope="module")
def host_control_results(tmp_path_factory):
    original = mother(16)
    candidate = precise.transform(original)
    assert precise.OLD in original and precise.NEW in candidate
    fixture = (ROOT / "tests/cpp/precise_finite_control.cpp").read_text()
    replacements = {
        "@@ROW_HELPER@@": _actual_header("attention_c16_softmax_helpers.h", "oscar_common.h"),
        "@@CERTIFICATE_HELPER@@": _actual_header("attention_official_finite.h", "attention_c16_softmax_helpers.h"),
        "@@BASELINE_BLOCK@@": precise.OLD,
        "@@CANDIDATE_BLOCK@@": precise.NEW,
    }
    for marker, body in replacements.items():
        assert fixture.count(marker) == 1
        fixture = fixture.replace(marker, body)
    work = tmp_path_factory.mktemp("precise-finite-control")
    cpp, executable = work / "control.cpp", work / "control"
    cpp.write_text(fixture)
    compiler = shutil.which("clang++")
    assert compiler, "host finite control requires clang++; it is not a CANN substitute"
    env = os.environ.copy()
    xcode = Path("/Applications/Xcode.app/Contents/Developer")
    if "DEVELOPER_DIR" not in env and xcode.is_dir():
        env["DEVELOPER_DIR"] = str(xcode)
    built = subprocess.run([compiler, "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror",
                            "-fno-fast-math", "-fno-strict-aliasing", "-fsanitize=undefined",
                            "-fno-sanitize-recover=undefined", str(cpp), "-o", str(executable)],
                           capture_output=True, text=True, timeout=60, env=env)
    assert built.returncode == 0, built.stdout + built.stderr
    ran = subprocess.run([str(executable)], capture_output=True, text=True, timeout=60)
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert not ran.stderr
    return json.loads(ran.stdout)


def test_actual_helpers_preserve_errors_inputs_and_bounded_scratch(host_control_results):
    result = host_control_results
    assert result["status"] == "passed"
    assert result["host_control_only"] and result["input_bits_unchanged"]
    assert result["reduction_orders"] == 2 and result["cases"] > 4000
    assert result["certified_cases"] > 0 and result["fallback_cases"] > 0
    assert result["guards"] == "passed"
    assert result["CANN"] == result["NPU"] == result["performance"] == "not_run"
