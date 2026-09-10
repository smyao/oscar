"""Fail-fast environment check for the optional OSCAR AscendC operator."""

from __future__ import annotations

import importlib.metadata as metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys

from delivery.runtime_versions import check_runtime_version


def _distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def main() -> int:
    versions = {
        name: _distribution_version(name)
        for name in ("torch", "torch-npu", "vllm", "vllm-ascend")
    }
    for name, version in versions.items():
        print(f"{name}: {version}")
    try:
        check_runtime_version("vllm", versions["vllm"])
        check_runtime_version("vllm-ascend", versions["vllm-ascend"])
    except (TypeError, ValueError) as exc:
        print(f"Unsupported runtime ABI: {exc}", file=sys.stderr)
        return 3

    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/latest"))
    required = (
        cann / "aarch64-linux" / "ascendc" / "include" / "basic_api"
        / "kernel_operator.h",
        cann / "aarch64-linux" / "asc" / "include" / "adv_api" / "matmul"
        / "matmul_intf.h",
    )
    print(f"ASCEND_HOME_PATH: {cann}")
    missing = [str(path) for path in required if not path.exists()]
    ccec = shutil.which("ccec") or shutil.which("ccec_compiler")
    if ccec:
        first_line = subprocess.run(
            [ccec, "--version"], check=False, capture_output=True, text=True
        ).stdout.splitlines()
        print("ccec:", first_line[0] if first_line else ccec)
    else:
        missing.append("ccec/ccec_compiler in PATH")
    if missing:
        print("Missing AscendC build prerequisites:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
