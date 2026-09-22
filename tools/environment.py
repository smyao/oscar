# 档案 #74–#78/#96/#104：环境只记录；完整性只比较本轮前后，不按版本字符串误杀。
"""Inspect environment without importing torch or creating a device context."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from .phase import atomic_json


def file_fingerprint(root: Path) -> dict[str, str]:
    # No symlink traversal into unrelated trees. Exclude generated/runtime caches.
    result = {}
    if not root.is_dir():
        return result
    for parent, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in {".git", "__pycache__", ".pytest_cache", "build", ".venv"})
        for name in sorted(files):
            path = Path(parent) / name
            if path.is_symlink() or name == ".DS_Store" or path.suffix in {".pyc", ".pyo"}:
                continue
            result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def inspect_environment() -> dict:
    packages = {}
    for name in ("torch", "torch_npu", "vllm", "vllm-ascend", "pybind11", "triton"):
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = {"installed": False}
        else:
            packages[name] = {"installed": True, "version": dist.version, "location": str(dist.locate_file(""))}
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "packages": packages, "tools": {x: shutil.which(x) for x in ("cmake", "bisheng", "npu-smi", "c++")},
            "environment": {x: os.environ.get(x) for x in ("ASCEND_HOME_PATH", "ASCEND_CANN_PACKAGE_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH", "ASCEND_CUSTOM_OPP_PATH", "ASCEND_RT_VISIBLE_DEVICES", "OMP_NUM_THREADS")},
            "policy": "observations_only", "npu_acceptance": "not_run"}


def compare_integrity(before: dict, after: dict) -> dict:
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    return {"status": "failed" if changed else "passed", "changed": changed}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--native-root", action="append", type=Path, default=[])
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    report = inspect_environment()
    report["native_trees"] = {str(p.resolve()): file_fingerprint(p) for p in args.native_root}
    failed = False
    if args.compare:
        before = json.loads(args.compare.read_text())["native_trees"]
        after = report["native_trees"]
        report["integrity"] = {k: compare_integrity(before.get(k, {}), after.get(k, {})) for k in before.keys() | after.keys()}
        failed = any(v["status"] == "failed" for v in report["integrity"].values())
    atomic_json(args.output, report)
    print(f"environment recorded: {args.output}; native mutation={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
