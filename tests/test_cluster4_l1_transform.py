"""Archive #145/#150/#151/#154/#164/#172 and full startup D.4.

Pinned source/C16 resource identity plus actual-method three-actor host proof.
Host transport/math stubs do not prove SDK floating-point reduction, CANN
compilation, NPU timing, graph replay, model quality or performance.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from tools import cluster4_l1_transform as l1
from tools import cluster_visibility_transform as visible
from tools import precise_finite_transform as finite
from tools import precise_zero_fill_transform as zero
from tools import cluster_state_transform as state

ROOT = Path(__file__).resolve().parents[1]


def baseline():
    source = (ROOT / "csrc/kernels/attention_cv_official_cluster4.cpp").read_text()
    return state.transform(zero.transform(finite.transform(visible.transform(source))))


def test_whole_source_inverse_and_pinned_c16_resource_implementation():
    original = baseline()
    assert sha256(original.encode()).hexdigest() == l1.BASELINE_SHA256
    candidate = l1.transform(original)
    assert l1.restore(candidate) == original
    c16 = l1._c16()
    assert l1._config(candidate) == l1._config(c16)
    assert "kCubeConfig.shareL1Size+128*1024<=512*1024" in l1._config(candidate)
    assert candidate.count("pipe.InitBuffer(residentB,128*1024);") == 1
    assert candidate.count("CubeMatmul mm;") == 1
    assert candidate.count("mm.Init(") == 1
    for name in ("LoadResidentB", "MatmulResidentB", "Matmul", "MatmulAtomic"):
        assert state.method(candidate, name) == state.method(c16, name)
    for name in ("Process", "LoadQueries", "LoadClusterKv", "CubeTask", "VectorTask", "VectorCluster",
                 "Softmax", "SpillState", "RestoreState", "PublishRows", "SelectClusterSlots"):
        assert state.method(candidate, name) == state.method(original, name)
    assert state.method(candidate, "IsClusterMember", "bool") == state.method(original, "IsClusterMember", "bool")
    assert candidate.count("extraScoreOffset=baseFloats+2*kKvRows*D;") == 1


def test_c1_keeps_fixed_d256_shapes_transpose_atomic_and_handshakes():
    original, candidate = baseline(), l1.transform(baseline())
    body = state.method(candidate, "CubeTask")
    assert body == state.method(original, "CubeTask")
    assert "Matmul(qOffset,kOffset,scoreOffset,rows,kKvRows,D);" in body
    assert "Matmul(pOffset,vOffset,pvOffset,rows,D,kKvRows,false,false);" in body
    assert "static_assert(D==256 && kKvRows==D," in candidate
    old = l1._config(original)
    for tiling in ("cfg.basicM=64;cfg.basicN=64;cfg.basicK=128;",
                   "cfg.singleCoreM=128;cfg.singleCoreN=256;cfg.singleCoreK=256;"):
        assert tiling in old and tiling in l1._config(candidate)
    transport = state.method(candidate, "MatmulResidentB")
    assert "SetHF32Mode(false);" in transport
    assert "mm.SetOrgShape(m,n,k);mm.SetSingleShape(m,n,k);" in transport
    assert "mm.SetTensorA(operands[2*a],false);" in transport
    assert "mm.SetTensorB(residentB.Get<bfloat16_t>(),transposeB);" in transport
    assert "mm.template IterateAll<true>(work[c],atomic);" in transport
    assert "mm.End();" in transport


@pytest.mark.parametrize("change", ["trailing", "dma", "math", "double"])
def test_transform_rejects_source_drift(change):
    original = baseline()
    source = {"trailing": lambda: original + "\n",
              "dma": lambda: original.replace("Fence<HardEvent::MTE3_MTE2>();", "", 1),
              "math": lambda: original.replace("SetHF32Mode(false)", "SetHF32Mode(true)", 1),
              "double": lambda: l1.transform(original)}[change]()
    with pytest.raises(ValueError, match="exact b49a689"):
        l1.transform(source)


@pytest.mark.parametrize("change", ["header", "flag", "capacity", "math", "tail"])
def test_restore_rejects_derivative_drift(change):
    candidate = l1.transform(baseline())
    source = {"header": lambda: candidate.removeprefix(l1.HEADER),
              "flag": lambda: candidate.replace("if(member>=2)CrossCoreWaitFlag", "if(member>2)CrossCoreWaitFlag", 1),
              "capacity": lambda: candidate.replace("residentB,128*1024", "residentB,256*1024", 1),
              "math": lambda: candidate.replace("SetHF32Mode(false)", "SetHF32Mode(true)", 1),
              "tail": lambda: candidate + "\n"}[change]()
    with pytest.raises(ValueError):
        l1.restore(source)


@pytest.fixture(scope="module")
def host_results(tmp_path_factory):
    original, candidate = baseline(), l1.transform(baseline())
    fixture = (ROOT / "tests/cpp/cluster4_l1_control.cpp").read_text()
    replace = {}
    for marker, name, result in (("@@SELECT@@", "SelectClusterSlots", "void"),
                                 ("@@PROBABILITY@@", "ProbabilityOffset", "int64_t"),
                                 ("@@VECTOR_CLUSTER@@", "VectorCluster", "void"),
                                 ("@@SPILL@@", "SpillState", "void"),
                                 ("@@RESTORE@@", "RestoreState", "void"),
                                 ("@@CUBE_TASK@@", "CubeTask", "void")):
        assert state.method(original, name, result) == state.method(candidate, name, result)
        replace[marker] = state.method(original, name, result)
    for marker, name in (("@@RESIDENT_LOAD@@", "LoadResidentB"),
                         ("@@RESIDENT_MATMUL@@", "MatmulResidentB")):
        replace[marker] = state.method(candidate, name)
    for arm, source in (("BASELINE", original), ("ACTIVE", candidate)):
        for suffix, name in (("MATMUL", "Matmul"), ("CUBE", "CubeCluster")):
            replace[f"@@{arm}_{suffix}@@"] = state.method(source, name).replace(
                name + "(", name + arm.title() + "(", 1)
    replace["@@BASELINE_ATOMIC@@"] = state.method(original, "MatmulAtomic").replace(
        "MatmulAtomic(", "MatmulAtomicBaseline(", 1)
    for marker, source in replace.items():
        assert fixture.count(marker) == 1
        fixture = fixture.replace(marker, source)
    assert "@@" not in fixture
    work = tmp_path_factory.mktemp("c4-l1-control")
    cpp, executable = work / "control.cpp", work / "control"
    cpp.write_text(fixture)
    compiler = shutil.which("clang++")
    assert compiler, "clang++ is required for host control, not a CANN substitute"
    env = os.environ.copy()
    xcode = Path("/Applications/Xcode.app/Contents/Developer")
    if "DEVELOPER_DIR" not in env and xcode.is_dir():
        env["DEVELOPER_DIR"] = str(xcode)
    built = subprocess.run([compiler, "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror", "-pthread",
                            "-fno-fast-math", "-fsanitize=undefined", "-fno-sanitize-recover=undefined",
                            str(cpp), "-o", str(executable)], capture_output=True, text=True,
                           timeout=60, env=env)
    assert built.returncode == 0, built.stdout + built.stderr
    ran = subprocess.run([str(executable)], capture_output=True, text=True, timeout=60)
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert not ran.stderr
    return json.loads(ran.stdout)


def test_actual_three_actor_methods_preserve_inputs_and_state(host_results):
    assert host_results == {
        "host_control_only": True, "actors": 3, "cases": 36,
        "matrix_input_bits": "identical", "state_error_bits": "identical",
        "B1_bytes": 131072, "cluster_B_loads_per_tile": 2, "C1_B_loads_per_tile": 2,
        "CANN": "not_run", "NPU": "not_run", "performance": "not_run",
    }
