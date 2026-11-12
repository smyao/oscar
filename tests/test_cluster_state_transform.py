"""Archive #145/#150/#151/#154/#164/#172 and complete startup D.4.

Run actual state lifecycle and cluster eligibility methods with bounded host
DMA/math stubs. This proves source scope, address work, initialization and
control equivalence; CANN/NPU asynchronous ownership remains unverified.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from tools import cluster_state_transform as state
from tools import cluster_visibility_transform as visibility
from tools import precise_finite_transform as precise


ROOT = Path(__file__).resolve().parents[1]


def baseline(family: int) -> str:
    original = (ROOT / f"csrc/kernels/attention_cv_official_cluster{family}.cpp").read_text()
    return precise.transform(visibility.transform(original) if family == 4 else original)


@pytest.mark.parametrize("family", [4, 16])
def test_complete_inverse_and_unchanged_math_cube_flags_and_publication(family):
    original = baseline(family)
    candidate = state.transform(original)
    assert state.restore(candidate) == original
    for method in ("LoadQueries", "LoadClusterKv", "CubeTask", "CubeCluster", "Softmax", "PublishRows"):
        assert state.method(candidate, method) == state.method(original, method)
    assert state.method(candidate, "IsClusterMember", "bool") == state.method(original, "IsClusterMember", "bool")
    for method in ("SpillState", "RestoreState"):
        before, after = state.method(original, method), state.method(candidate, method)
        assert [line for line in before.splitlines() if "Fence<" in line] == [
            line for line in after.splitlines() if "Fence<" in line]
    before, after = state.method(original, "VectorCluster"), state.method(candidate, "VectorCluster")
    assert [line for line in before.splitlines() if "CrossCore" in line] == [
        line for line in after.splitlines() if "CrossCore" in line]
    # Final normalization must restore Cube's atomic update, and every tile
    # must still spill alpha*acc before allowing that atomic update.
    assert "error=memberErrors[member];RestoreState(member);" in after
    assert "        SpillState(member);\n        CrossCoreSetFlag<2,PIPE_MTE3>" in after
    tail = "    // No final next-KV token;"
    assert before[before.index(tail):] == after[after.index(tail):]


@pytest.mark.parametrize("family", [4, 16])
@pytest.mark.parametrize("fault", ["double", "empty", "range", "query_state", "query_fence", "final_restore", "spill"])
def test_transform_rejects_lifecycle_or_eligibility_drift(family, fault):
    original = baseline(family)
    changed = {
        "double": lambda: state.transform(original),
        "empty": lambda: original.replace("kvbegin>=kvend", "kvbegin>kvend", 1),
        "range": lambda: original.replace("kvbegin!=original.kvbegin || kvend!=original.kvend", "false", 1),
        "query_state": lambda: original.replace("  __aicore__ void LoadQueries() {", "  __aicore__ void LoadQueries() {\n    accBuf.Get<float>();", 1),
        "query_fence": lambda: original.replace(state.method(original, "LoadQueries"),
            state.method(original, "LoadQueries").replace("Fence<HardEvent::MTE3_V>();", ""), 1),
        "final_restore": lambda: original.replace("error=memberErrors[member];RestoreState(member);", "error=memberErrors[member];", 1),
        "spill": lambda: original.replace(state.OLD_SPILL, "", 1),
    }[fault]()
    with pytest.raises(ValueError):
        state.transform(changed)


@pytest.mark.parametrize("fault", ["header", "initial_value", "stride", "first"])
def test_restore_rejects_modified_state_derivative(fault):
    candidate = state.transform(baseline(16))
    changed = {
        "header": lambda: candidate.removeprefix(state.HEADER),
        "initial_value": lambda: candidate.replace(state.NEW_RESTORE,
            state.NEW_RESTORE.replace("Duplicate(acc,0.0F", "Duplicate(acc,1.0F"), 1),
        "stride": lambda: candidate.replace("statCopy{2,statBlocks,0,statBlocks}", "statCopy{2,statBlocks,0,0}", 1),
        "first": lambda: candidate.replace("if(start==first)", "if(start<=first)", 1),
    }[fault]()
    with pytest.raises(ValueError):
        state.restore(changed)


@pytest.fixture(scope="module")
def host_results(tmp_path_factory):
    sources = {family: baseline(family) for family in (4, 16)}
    candidates = {family: state.transform(source) for family, source in sources.items()}
    fixture = (ROOT / "tests/cpp/cluster_state_control.cpp").read_text()
    replacements = {}
    for family in (4, 16):
        for arm, source in (("BASELINE", sources[family]), ("ACTIVE", candidates[family])):
            method = state.method(source, "VectorCluster")
            replacements[f"@@{arm}_CLUSTER{family}@@"] = method.replace(
                "VectorCluster(", f"VectorCluster{family}{arm.title()}(", 1)
        replacements[f"@@QUALIFY{family}@@"] = state.method(sources[family], "IsClusterMember", "bool").replace(
            "IsClusterMember(", f"IsClusterMember{family}(", 1)
    for name in ("SpillState", "RestoreState"):
        assert state.method(sources[4], name) == state.method(sources[16], name)
        assert state.method(candidates[4], name) == state.method(candidates[16], name)
        short = "SPILL" if name == "SpillState" else "RESTORE"
        for arm, source in (("BASELINE", sources[16]), ("ACTIVE", candidates[16])):
            replacements[f"@@{arm}_{short}@@"] = state.method(source, name).replace(
                name + "(", name + arm.title() + "(", 1)
    for marker, name, result in (("@@CURRENT_TASK@@", "CurrentTask", "TaskState"),
                                 ("@@USE_TASK@@", "UseTask", "void"),
                                 ("@@TASK_VALID@@", "TaskValid", "bool")):
        assert state.method(sources[4], name, result) == state.method(sources[16], name, result)
        replacements[marker] = state.method(sources[16], name, result)
    for marker, body in replacements.items():
        assert fixture.count(marker) == 1
        fixture = fixture.replace(marker, body)
    assert "@@" not in fixture
    work = tmp_path_factory.mktemp("cluster-state-control")
    cpp, executable = work / "control.cpp", work / "control"
    cpp.write_text(fixture)
    compiler = shutil.which("clang++")
    assert compiler, "host lifecycle proof requires clang++; this is not a CANN substitute"
    env = os.environ.copy()
    xcode = Path("/Applications/Xcode.app/Contents/Developer")
    if "DEVELOPER_DIR" not in env and xcode.is_dir():
        env["DEVELOPER_DIR"] = str(xcode)
    built = subprocess.run([compiler, "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror",
                            "-fno-fast-math", "-fsanitize=undefined", "-fno-sanitize-recover=undefined",
                            str(cpp), "-o", str(executable)],
                           capture_output=True, text=True, timeout=60, env=env)
    assert built.returncode == 0, built.stdout + built.stderr
    ran = subprocess.run([str(executable)], capture_output=True, text=True, timeout=60)
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert not ran.stderr
    return json.loads(ran.stdout)


def test_actual_state_lifecycle_and_strided_dma_are_equivalent(host_results):
    assert host_results == {
        "host_control_only": True, "cases": 114, "uninitialized_GM_reads": 0,
        "state_and_error_bits": "identical", "cross_flags": "identical", "guards": "passed",
        "CANN": "not_run", "NPU": "not_run", "performance": "not_run",
    }
