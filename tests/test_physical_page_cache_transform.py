"""Pinned inverse and actual-method host checks; never CANN/NPU evidence."""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess

import pytest

from tools import active_decode_transform as active_decode
from tools import cluster4_l1_transform as l1
from tools import cluster_score_ring_transform as ring
from tools import cluster_state_transform as state
from tools import cluster_visibility_transform as visibility
from tools import precise_finite_transform as precise
from tools import precise_zero_fill_transform as zero
from tools import physical_page_cache_transform as cache

ROOT = Path(__file__).resolve().parents[1]


def baseline(family):
    """Rebuild de0c85b without using a wrapper that may add this transform."""
    text = (ROOT / f"csrc/kernels/attention_cv_official_{family}.cpp").read_text()
    if family == "decode":
        return active_decode.transform(text)
    source = visibility.transform(text) if family == "cluster4" else text
    source = state.transform(zero.transform(precise.transform(source)))
    return l1.transform(source) if family == "cluster4" else ring.transform(source)


@pytest.mark.parametrize("family", ["decode", "cluster4", "cluster16"])
def test_whole_source_inverse_only_physical_init_and_scalar_fields(family):
    source = baseline(family)
    assert cache.BASELINES[sha256(source.encode()).hexdigest()] == family
    derived = cache.transform(source)
    assert cache.restore(derived) == source
    assert state.method(source, "Physical", "bool") == cache.OLD_PHYSICAL
    assert state.method(derived, "Physical", "bool") == cache.NEW_PHYSICAL
    assert state.method(derived, "Init").replace(cache.INIT_RESET, "", 1) == state.method(source, "Init")
    assert derived.count(cache.INIT_RESET) == 1
    assert derived.count(cache.FIELDS) == 1
    # Every caller, numerical method, actor flag and pre-existing UB allocation
    # remains identical; the inverse additionally checks the complete source.
    for method in ("Process", "LoadQueries", "LoadPrecise", "LoadWindowRuns", "PublishStatus",
                   "Softmax", "CubeTask", "VectorTask"):
        assert state.method(source, method) == state.method(derived, method)
    assert source.count("pipe.InitBuffer(") == derived.count("pipe.InitBuffer(")
    assert cache.NEW_PHYSICAL.count("Fence<HardEvent::MTE2_S>();") == 1
    assert cache.NEW_PHYSICAL.index("}\n    // Preserve") < cache.NEW_PHYSICAL.index("Fence<HardEvent::MTE2_S>();")


@pytest.mark.parametrize("family", ["decode", "cluster4", "cluster16"])
@pytest.mark.parametrize("change", ["input", "header", "fence", "cache_key", "reset", "tail"])
def test_rejects_any_input_or_derivative_drift(family, change):
    source = baseline(family)
    if change == "input":
        with pytest.raises(ValueError):
            cache.transform(source + "\n")
        return
    derived = cache.transform(source)
    altered = {
        "header": lambda: derived.removeprefix(cache.HEADER),
        "fence": lambda: derived.replace(cache.NEW_PHYSICAL, cache.NEW_PHYSICAL.replace("Fence<HardEvent::MTE2_S>();", "")),
        "cache_key": lambda: derived.replace("physicalCacheRequest==request && ", ""),
        "reset": lambda: derived.replace(cache.INIT_RESET, ""),
        "tail": lambda: derived + "\n",
    }[change]()
    with pytest.raises(ValueError):
        cache.restore(altered)


def test_actual_physical_and_packed_loaders_with_deferred_dma(tmp_path):
    sources = {family: baseline(family) for family in ("decode", "cluster4", "cluster16")}
    for source in sources.values():
        assert state.method(source, "Physical", "bool") == cache.OLD_PHYSICAL
    assert state.method(sources["cluster4"], "LoadPacked") == state.method(sources["cluster16"], "LoadPacked")
    fixture = (ROOT / "tests/cpp/physical_page_cache_control.cpp").read_text()
    substitutions = {
        "@@FIELDS@@": cache.FIELDS,
        "@@RESET@@": cache.INIT_RESET,
        "@@ORIGINAL_PHYSICAL@@": state.method(sources["decode"], "Physical", "bool").replace("Physical(", "PhysicalOriginal(", 1),
        "@@CACHED_PHYSICAL@@": state.method(cache.transform(sources["decode"]), "Physical", "bool").replace("Physical(", "PhysicalCached(", 1),
        "@@DECODE_LOAD@@": state.method(sources["decode"], "LoadPackedInto"),
        "@@CLUSTER_LOAD@@": state.method(sources["cluster4"], "LoadPacked"),
    }
    for marker, code in substitutions.items():
        assert fixture.count(marker) == 1
        fixture = fixture.replace(marker, code)
    assert "@@" not in fixture
    cpp, executable = tmp_path / "physical.cpp", tmp_path / "physical"
    cpp.write_text(fixture)
    compiler = shutil.which("clang++")
    assert compiler, "clang++ is required for host control; not a CANN substitute"
    env = os.environ.copy()
    xcode = Path("/Applications/Xcode.app/Contents/Developer")
    if "DEVELOPER_DIR" not in env and xcode.is_dir():
        env["DEVELOPER_DIR"] = str(xcode)
    built = subprocess.run([compiler, "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror",
                            "-fsanitize=undefined", "-fno-sanitize-recover=undefined", str(cpp),
                            "-o", str(executable)], capture_output=True, text=True, timeout=60, env=env)
    assert built.returncode == 0, built.stdout + built.stderr
    ran = subprocess.run([str(executable)], capture_output=True, text=True, timeout=60)
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert not ran.stderr
    assert json.loads(ran.stdout) == {
        "physical_calls": 39228, "decode_packed_cases": 360, "cluster_packed_cases": 360,
        "fences": "identical", "packed_bytes": "identical", "control": "identical",
        "CANN": "not_run", "NPU": "not_run", "performance": "not_run",
    }


def test_committed_static_work_count_is_reproducible_and_retains_zero_hit_case():
    directory = ROOT / "reports/bs_scaling_20261113"
    generated = runpy.run_path(str(directory / "physical_page_cache_workload.py"))["measure"]()
    assert generated == json.loads((directory / "physical_page_cache_workload.json").read_text())
    aligned = next(case for case in generated["isolated_history_intervals"]
                   if case["family"] == "decode" and case["begin"] == 0)
    assert aligned["table_dma_before"] == aligned["table_dma_after"]
    assert aligned["hit_pct"] == 0.0
    assert all(case["lookups_and_fences_before_and_after"] == case["table_dma_before"]
               for case in generated["decode_cases"])
