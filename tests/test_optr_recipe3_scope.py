"""Protect all pre-diagnosis production files outside named offline seams."""
import hashlib
from pathlib import Path
import subprocess

from test_optr_cpu_scope_20261105 import OFFLINE, _without_profile, _front_card_target


ROOT = Path(__file__).resolve().parents[1]
BASE = "7c4a621"
OFFLINE_SEAMS = {
    "oscar_ascend/optr_artifacts.py", "oscar_ascend/optr_checkpoint.py",
    "oscar_ascend/optr_policy.py", "oscar_ascend/optr_sequential.py",
    "oscar_ascend/optr_traces.py", "tools/calibrate_optr_traces.py",
    "tools/optr_fit_worker.py", "tools/prepare_optr_rotations.py",
    "tools/serving_variants.py",
    "tools/prepare_rotations.py",  # #176: explain an existing failed fit; no fitting identity change.
}


def test_existing_production_bytes_change_only_at_offline_recipe3_seams():
    records = subprocess.check_output(["git", "ls-tree", "-r", "-z", BASE,
        "csrc", "cmake", "oscar_ascend", "configs", "scripts", "tools"], cwd=ROOT).split(b"\0")
    changed = []
    for record in filter(None, records):
        header, path_bytes = record.split(b"\t", 1)
        _, kind, expected = header.split()
        path = path_bytes.decode()
        if kind != b"blob" or path in (OFFLINE_SEAMS | OFFLINE):
            continue
        current = ROOT / path
        if not current.is_file():
            changed.append(path)
            continue
        data = current.read_bytes()
        if path in ("configs/target.json", "tools/probe_history_reuse.py"):
            # User 2026-11-06 authorized only the new target placement.
            previous = subprocess.check_output(["git", "show", f"7720e3d:{path}"], cwd=ROOT)
            assert data == (_front_card_target(previous) if path == "configs/target.json" else previous)
            continue
        if path == "oscar_ascend/rotations.py":
            previous = subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT)
            assert _without_profile(data) == _without_profile(previous)
            continue
        actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if actual != expected.decode():
            changed.append(path)
    assert changed == []
