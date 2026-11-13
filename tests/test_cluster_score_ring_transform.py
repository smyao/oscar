"""Archive #145/#150/#151/#154/#164/#172 and full startup D.4.

Protect score read/write ownership with actual extracted C16 control methods.
The three-actor model explores legal interleavings and deliberately rejects a
one-AIV acknowledgement. It is host control evidence, not NPU validation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from tools import cluster_score_ring_transform as ring
from tools import cluster_state_transform as state
from tools import precise_finite_transform as finite
from tools import precise_zero_fill_transform as zero


ROOT = Path(__file__).resolve().parents[1]


def baseline() -> str:
    # Reconstruct b49's source independently of the composing owner, which
    # will subsequently include this transform. No self-derived expected text.
    mother = (ROOT / "csrc/kernels/attention_cv_official_cluster16.cpp").read_text()
    return state.transform(zero.transform(finite.transform(mother)))


def test_full_inverse_changes_only_shared_slot_selection_and_header():
    original = baseline()
    candidate = ring.transform(original)
    assert ring.restore(candidate) == original
    assert candidate.removeprefix(ring.HEADER).replace(ring.NEW_SELECT, ring.OLD_SELECT, 1) == original
    # All numerical, finite, error, mask, ordinary-task and allocation paths
    # remain byte-identical, stronger than reproducing them in a host mock.
    for name in ("CubeCluster", "VectorCluster", "CubeTask", "VectorTask", "Softmax",
                 "LoadPacked", "LoadPrecise", "Unpack", "SpillState", "RestoreState",
                 "PublishRows", "LoadQueries", "Init"):
        assert state.method(candidate, name) == state.method(original, name)
    for name, result in (("IsClusterMember", "bool"), ("TaskValid", "bool"),
                         ("ProbabilityOffset", "int64_t")):
        assert state.method(candidate, name, result) == state.method(original, name, result)


@pytest.mark.parametrize("fault", ["cube_wait", "vector_flag", "softmax", "spill", "probability",
                                   "eligibility", "slot", "duplicate", "c4"])
def test_transform_rejects_unproven_protocol_or_consumers(fault):
    source = baseline()
    changed = {
        "cube_wait": lambda: source.replace("if(member>=4)CrossCoreWaitFlag", "if(member>=5)CrossCoreWaitFlag", 1),
        "vector_flag": lambda: source.replace("SpillState(member);\n        CrossCoreSetFlag<2,PIPE_MTE3>",
                                               "SpillState(member);\n        CrossCoreSetFlag<2,PIPE_V>", 1),
        "softmax": lambda: source.replace("  __aicore__ void Softmax(", "  __aicore__ void SoftmaxChanged(", 1),
        "spill": lambda: source.replace("statCopy{2,statBlocks,0,statBlocks}", "statCopy{2,statBlocks,0,0}", 1),
        "probability": lambda: source.replace("return queryBase+kQueryRows*D/2;", "return queryBase;", 1),
        "eligibility": lambda: source.replace("kvbegin>=kvend", "kvbegin>kvend", 1),
        "slot": lambda: source.replace(ring.OLD_SELECT, ring.OLD_SELECT.replace("slot-1", "slot"), 1),
        "duplicate": lambda: ring.transform(source),
        "c4": lambda: (ROOT / "csrc/kernels/attention_cv_official_cluster4.cpp").read_text(),
    }[fault]()
    assert changed != source
    with pytest.raises(ValueError):
        ring.transform(changed)


@pytest.mark.parametrize("fault", ["header", "modulo", "offset", "protocol", "math"])
def test_inverse_rejects_modified_derivative(fault):
    candidate = ring.transform(baseline())
    changed = {
        "header": lambda: candidate.removeprefix(ring.HEADER),
        "modulo": lambda: candidate.replace("scoreSlot=slot%4", "scoreSlot=slot%3", 1),
        "offset": lambda: candidate.replace("(scoreSlot-1)*kQueryRows", "scoreSlot*kQueryRows", 1),
        "protocol": lambda: candidate.replace("if(member>=4)CrossCoreWaitFlag", "if(member>=5)CrossCoreWaitFlag", 1),
        "math": lambda: candidate.replace("  __aicore__ void Softmax(", "  __aicore__ void SoftmaxChanged(", 1),
    }[fault]()
    with pytest.raises(ValueError):
        ring.restore(changed)


@pytest.fixture(scope="module")
def host_results(tmp_path_factory):
    original = baseline()
    active = ring.transform(original)
    fixture = (ROOT / "tests/cpp/cluster_score_ring_control.cpp").read_text()
    replacements = {
        "@@BASE_SELECT@@": state.method(original, "SelectClusterSlots").replace("SelectClusterSlots(", "SelectBase(", 1),
        "@@RING_SELECT@@": state.method(active, "SelectClusterSlots").replace("SelectClusterSlots(", "SelectRing(", 1),
        "@@PROBABILITY@@": state.method(original, "ProbabilityOffset", "int64_t"),
        "@@CUBE_CLUSTER@@": state.method(original, "CubeCluster"),
        "@@VECTOR_CLUSTER@@": state.method(original, "VectorCluster"),
    }
    for marker, body in replacements.items():
        assert fixture.count(marker) == 1
        fixture = fixture.replace(marker, body)
    assert "@@" not in fixture
    work = tmp_path_factory.mktemp("cluster-score-ring-control")
    cpp, executable = work / "control.cpp", work / "control"
    cpp.write_text(fixture)
    compiler = shutil.which("clang++")
    assert compiler, "host ownership proof requires clang++; it is not a CANN substitute"
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


def test_three_actor_ownership_round_drain_and_changed_input(host_results):
    assert host_results["host_control_only"] is True
    assert host_results["exhaustive_states"] > 1000
    assert host_results["random_or_adversarial_schedules"] == 1024
    assert host_results["double_AIV_required"] is True
    assert host_results["changed_input_epochs"] == 2
    assert host_results["persistent_address_shadow"] is True
    assert host_results["stale_payload_rejected"] is True
    assert host_results["old_score_address_bytes"] == 2 * 1024 * 1024
    assert host_results["new_score_address_bytes"] == 512 * 1024
    assert host_results["score_DMA_bytes_per_tile"] == 4_161_536
    assert {host_results[key] for key in ("CANN", "NPU", "performance")} == {"not_run"}
