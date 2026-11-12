"""Archive #145/#150/#154/#164/#172 and startup D.4 control-flow regression.

Real protected/derived C++ methods run with counting host stubs. This proves
descriptor access, dispatch and status equivalence, not AscendC arithmetic,
pipe timing, device compilation, NPU execution or actual graph replay.
"""
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from tools.active_decode_transform import BASELINE_SHA256, DERIVED_HEADER, restore, transform


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "csrc/kernels/attention_cv_official_decode.cpp"


def _method(source, name):
    """Extract the actual simple C++ method with its balanced brace body."""
    start = source.index(f"  __aicore__ {'bool' if name == 'TaskValid' else 'void'} {name}(")
    first = source.index("{", start)
    depth = 0
    for pos in range(first, len(source)):
        if source[pos] == "{":
            depth += 1
        elif source[pos] == "}":
            depth -= 1
            if depth == 0:
                return source[start:pos + 1]
    raise AssertionError(f"unclosed real method {name}")


def test_exact_mother_inverse_and_unchanged_numerical_body():
    original = SOURCE.read_text()
    assert sha256(original.encode()).hexdigest() == BASELINE_SHA256
    active = transform(original)
    assert restore(active) == original
    # Independent scope check: there is exactly one changed method and one
    # derived comment prefix. Symbols, arithmetic, flags and DMA sites remain.
    restored = active.removeprefix(DERIVED_HEADER).replace(
        _method(active, "LoadTask"), _method(original, "LoadTask"), 1)
    assert restored == original
    for method in ("Process", "TaskValid"):
        assert _method(active, method) == _method(original, method)
    load = _method(active, "LoadTask")
    assert load.count("DataCopy(row,taskGm[id*16],16)") == 1
    assert load.count("Fence<HardEvent::MTE2_S>();") == 1


@pytest.mark.parametrize("drift", ["trailing", "arithmetic", "already_derived"])
def test_transform_rejects_every_nonmother_source(drift):
    original = SOURCE.read_text()
    changed = {"trailing": lambda: original + "\n",
               "arithmetic": lambda: original.replace("if(qcount==0)", "if(qcount<=0)", 1),
               "already_derived": lambda: transform(original)}[drift]()
    with pytest.raises(ValueError, match="exact protected mother"):
        transform(changed)


@pytest.mark.parametrize("drift", ["header", "load", "math", "trailing"])
def test_restore_rejects_modified_derived_source(drift):
    active = transform(SOURCE.read_text())
    changed = {"header": lambda: active.removeprefix(DERIVED_HEADER),
               "load": lambda: active.replace("if(data[1]==0)", "if(data[1]<=0)", 1),
               "math": lambda: active.replace("qcount*(g.hq/g.hk)<=kQueryRows", "qcount*(g.hq/g.hk)<kQueryRows", 1),
               "trailing": lambda: active + "\n"}[drift]()
    with pytest.raises(ValueError, match="active decode"):
        restore(changed)


@pytest.fixture(scope="module")
def host_control_results(tmp_path_factory):
    original = SOURCE.read_text()
    active = transform(original)
    fixture = (ROOT / "tests/cpp/active_decode_control.cpp").read_text()
    for name in ("Process", "TaskValid"):
        assert _method(active, name) == _method(original, name)
    replacements = {
        "@@BASELINE_LOAD@@": _method(original, "LoadTask").replace("LoadTask(", "LoadTaskBaseline(", 1),
        "@@ACTIVE_LOAD@@": _method(active, "LoadTask").replace("LoadTask(", "LoadTaskActive(", 1),
        "@@PROCESS@@": _method(original, "Process"),
        "@@TASK_VALID@@": _method(original, "TaskValid"),
    }
    for marker, body in replacements.items():
        assert fixture.count(marker) == 1
        fixture = fixture.replace(marker, body)
    work = tmp_path_factory.mktemp("active-decode-control")
    cpp, executable = work / "control.cpp", work / "control"
    cpp.write_text(fixture)
    compiler = shutil.which("clang++")
    assert compiler, "host C++ control test requires clang++; it is not a CANN substitute"
    env = os.environ.copy()
    xcode = Path("/Applications/Xcode.app/Contents/Developer")
    if "DEVELOPER_DIR" not in env and xcode.is_dir():
        # Select the installed Xcode for this compiler subprocess only; never
        # mutate the machine's xcode-select setting or hide compiler failures.
        env["DEVELOPER_DIR"] = str(xcode)
    built = subprocess.run([compiler, "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror",
                            "-fsanitize=undefined", "-fno-sanitize-recover=undefined",
                            "-I", str(ROOT / "csrc/include"), str(cpp), "-o", str(executable)],
                           capture_output=True, text=True, timeout=60, env=env)
    assert built.returncode == 0, built.stdout + built.stderr
    ran = subprocess.run([str(executable)], capture_output=True, text=True, timeout=30)
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert not ran.stderr
    return [json.loads(line) for line in ran.stdout.splitlines()]


@pytest.mark.parametrize("batch,q,zero,before,after", [
    (16, 4, 720, 10560, 3360), (28, 4, 1008, 14784, 4704),
    # Real later-MTP negative slots prepare qcount=-1, not zero. Their full
    # descriptors remain necessary for PublishEmpty: no saving is claimed.
    (16, 1, 0, 10560, 10560), (28, 1, 0, 14784, 14784),
])
def test_actual_cpp_q4_q1_workload_counts(host_control_results, batch, q, zero, before, after):
    row = next(r for r in host_control_results if r.get("batch") == batch and r["q"] == q)
    assert row == {"batch": batch, "q": q, "splits": 5 if batch == 16 else 4,
                   "zero_tasks": zero, "aic_reads_before": before, "aic_reads_after": after,
                   "aiv_reads_before": before * 2, "aiv_reads_after": after * 2}


def test_actual_cpp_error_poison_and_same_address_metadata_refresh(host_control_results):
    assert host_control_results[-1] == {
        "adversarial": True, "same_address_refresh": True, "host_control_only": True}
