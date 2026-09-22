# Archive #74-85/#94-97/#100/#117/#120: isolated VM workspace, actual CANN builds, CPU-debug evidence kept separate.
"""Reproduce CANN compilation and official CPU debugging in the local Lima VM."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import tarfile
import tempfile
from .phase import atomic_json,run_phase

ROOT=Path(__file__).resolve().parents[1]
SOURCES=("oscar_ascend","csrc","tools","tests","configs","pyproject.toml")


def source_paths(root=ROOT):
    for name in SOURCES:
        base=root/name
        for path in sorted(base.rglob("*")) if base.is_dir() else [base]:
            if (path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts
                    and not path.name.startswith("._") and path.name!=".DS_Store"):
                yield path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instance",default="oscar")
    p.add_argument("--workspace",default="/home/sunao2000.linux/gpt_new_oscar")
    p.add_argument("--cann",default="/usr/local/Ascend/cann-9.1.0")
    p.add_argument("--soc",default="ascend910b4")
    p.add_argument("--bootstrap",action="store_true",help="install isolated VM Python build/test dependencies")
    p.add_argument("--output",type=Path,default=ROOT/"reports/vm")
    args=p.parse_args()
    remote=PurePosixPath(args.workspace)
    if not args.workspace.startswith("/home/") or remote.name!="gpt_new_oscar" or ".." in remote.parts:
        raise ValueError("VM validation only writes a dedicated /home/.../gpt_new_oscar workspace")
    args.output.mkdir(parents=True,exist_ok=True)
    prefix=["limactl","shell","--workdir",args.workspace,args.instance,"--"]
    subprocess.run(["limactl","shell","--workdir","/tmp",args.instance,"--","mkdir","-p",args.workspace],check=True,timeout=30)
    with tempfile.TemporaryFile() as archive:
        fingerprints={}
        with tarfile.open(fileobj=archive,mode="w") as bundle:
            for path in source_paths():
                relative=str(path.relative_to(ROOT));bundle.add(path,arcname=relative,recursive=False)
                fingerprints[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
        archive.seek(0)
        subprocess.run([*prefix,"tar","-xf","-"],stdin=archive,check=True,timeout=60)
    atomic_json(args.output/"source_sha256.json",fingerprints)
    shell=lambda script:[*prefix,"timeout","--signal=TERM","--kill-after=10s","1900","bash","-lc",script]
    quoted=shlex.quote
    stages=[]
    if args.bootstrap:
        stages.append(("bootstrap","python3 -m venv --without-pip --system-site-packages .venv && "
            ".venv/bin/python -m pip install pybind11 pytest wheel && "
            ".venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu --no-deps --ignore-installed torch==2.12.0"))
    activate=f"source {quoted(args.cann)}/set_env.sh; export TORCH_DEVICE_BACKEND_AUTOLOAD=0; "
    stages += [
        ("install",activate+".venv/bin/python -m pip install --no-build-isolation --no-deps -e ."),
        ("device-build",activate+f".venv/bin/python -m tools.build_ops --soc {quoted(args.soc)} --log-dir logs/vm-verify-device"),
        ("cpu-build",activate+f"cmake -S csrc/cpu -B build/cpu -DASCEND_HOME_PATH={quoted(args.cann)} -Dtikicpulib_DIR={quoted(args.cann)}/tools/tikicpulib/lib/cmake && cmake --build build/cpu --parallel 4"),
        ("primitive-goldens",activate+".venv/bin/python -m tools.generate_cpu_cases --output artifacts/cpu_cases"),
        ("primitive-debug",activate+".venv/bin/python -m tools.run_cpu_debug --log-dir logs/vm-primitives"),
        ("rotation-goldens",activate+".venv/bin/python -m tools.generate_rotation_cpu_cases --output artifacts/rotation_cpu_cases"),
        ("rotation-debug",activate+".venv/bin/python -m tools.run_cpu_debug --executable build/cpu/oscar_rotation_cpu --cases artifacts/rotation_cpu_cases/cases.json --output reports/rotation_cpu_debug.json --log-dir logs/vm-rotation"),
        ("cv-goldens",activate+".venv/bin/python tests/test_cv_contracts.py artifacts/cv_cpu_cases"),
        ("cv-debug",activate+".venv/bin/python -m tools.run_cpu_debug --executable build/cpu/oscar_cv_cpu --cases artifacts/cv_cpu_cases/cases.json --output reports/cv_cpu_debug.json --log-dir logs/vm-cv"),
    ]
    report={"scope":"CANN compiler and official CPU debugger","instance":args.instance,
            "workspace":args.workspace,"status":"running","npu_acceptance":"not_run","phases":[]}
    rc=0
    try:
        for name,script in stages:
            result=run_phase(name,shell(script),cwd=ROOT,log_dir=args.output/"logs",timeout=1950)
            report["phases"].append({"phase":name,"returncode":result.returncode,"log":result.log})
            atomic_json(args.output/"validation.json",report)
            if result.returncode:
                rc=result.returncode;report.update(status="failed",failed_phase=name);break
        if not rc:report["status"]="passed"
    finally:
        # Read only this task's reports, never other mounted/failed projects.
        result=subprocess.run([*prefix,"tar","-cf","-","reports","logs/vm-verify-device",
                               "logs/vm-primitives","logs/vm-rotation","logs/vm-cv"],capture_output=True,timeout=60)
        if result.returncode==0:
            target=args.output/"guest";target.mkdir(exist_ok=True)
            with tarfile.open(fileobj=io.BytesIO(result.stdout),mode="r:") as bundle:
                bundle.extractall(target,filter="data")
        else:
            report["collection_error"]=result.stderr.decode(errors="replace")
        atomic_json(args.output/"validation.json",report)
    return rc


if __name__=="__main__":
    sys.exit(main())
