"""Protect all pre-diagnosis production files outside named offline seams."""
import hashlib
from pathlib import Path
import subprocess

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
        if kind != b"blob" or path in OFFLINE_SEAMS:
            continue
        current = ROOT / path
        if not current.is_file():
            changed.append(path)
            continue
        data = current.read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if actual != expected.decode():
            changed.append(path)
    assert changed == []
