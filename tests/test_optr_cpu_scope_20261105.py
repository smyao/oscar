"""Archive #172/#175-177: CPU preparation may not change online execution."""
import ast
import hashlib
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
BASE = "7720e3d"
OFFLINE = {
    "oscar_ascend/optr_artifacts.py", "oscar_ascend/optr_calibration.py",
    "oscar_ascend/optr_checkpoint.py", "oscar_ascend/optr_native_residual.py",
    "oscar_ascend/optr_native_residual_certificate.py", "oscar_ascend/optr_policy.py",
    "oscar_ascend/optr_traces.py", "oscar_ascend/optr_trust_region.py",
    "tools/optr_fit_worker.py", "tools/optr_parallel.py", "tools/optr_recipe3_sampling.py",
    "tools/prepare_optr_rotations.py",
    "tools/optr_rejection_report.py", "tools/optr_rejection_summary.py",
}


def _without_profile(source):
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "calibration_cache_identity")
    lines = source.splitlines(keepends=True)
    return b"".join(lines[:node.lineno - 1] + lines[node.end_lineno:])


def _front_card_target(source):
    # User 2026-11-06: permit exactly the placement update; freeze other bytes.
    devices = b'"devices": [\n    4,\n    5,\n    6,\n    7\n  ]'
    policy = b'User 2026-11-05: physical NPU 4-7;'
    assert source.count(devices) == source.count(policy) == 1
    return source.replace(devices, b'"devices": [\n    0,\n    1,\n    2,\n    3\n  ]').replace(
        policy, b'User 2026-11-06: physical NPU 0-3;')


def test_online_sources_launch_timer_and_base_fitter_are_unchanged():
    records = subprocess.check_output(["git", "ls-tree", "-r", "-z", BASE,
        "csrc", "cmake", "oscar_ascend", "configs", "scripts", "tools"], cwd=ROOT).split(b"\0")
    changed = []
    for record in filter(None, records):
        header, encoded = record.split(b"\t", 1)
        _, kind, expected = header.split()
        name = encoded.decode()
        if kind != b"blob" or name in OFFLINE:
            continue
        path = ROOT / name
        if not path.is_file():
            changed.append(name)
            continue
        data = path.read_bytes()
        if name == "configs/target.json":
            previous = subprocess.check_output(["git", "show", f"{BASE}:{name}"], cwd=ROOT)
            assert data == _front_card_target(previous)
            continue
        if name == "oscar_ascend/rotations.py":
            previous = subprocess.check_output(["git", "show", f"{BASE}:{name}"], cwd=ROOT)
            assert _without_profile(data) == _without_profile(previous)
            continue
        digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if digest != expected.decode():
            changed.append(name)
    assert changed == []


def test_ordinary_profile_identity_does_not_use_transport(tmp_path, monkeypatch):
    import json
    import types
    import sys
    from oscar_ascend.rotations import calibration_cache_identity
    old = types.ModuleType("old_profile")
    old.__package__ = "oscar_ascend"
    monkeypatch.setitem(sys.modules, old.__name__, old)
    source = subprocess.check_output(["git", "show", f"{BASE}:oscar_ascend/rotations.py"], cwd=ROOT)
    exec(compile(source, "old_rotations.py", "exec"), old.__dict__)
    config = json.loads((ROOT / "configs/target.json").read_text())
    prompts = tmp_path / "prompts.json"
    prompts.write_text("[]")
    for extra in ({}, {"rotation_calibration_prompts": str(prompts)}):
        current = {**config, **extra}
        assert calibration_cache_identity(current, "fingerprint") == old.calibration_cache_identity(current, "fingerprint")
