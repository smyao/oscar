"""Archive #79-85/#96/#125/#126/#145/#164: local CPU/CANN paired validation.

D.4: execute only bounded fused-reader fixtures; never restore production
history. Preserve every first failure and real phase RC. This does not time
kernels, contact a target NPU, or claim graph/model/performance acceptance.
Only the current repository sources are staged; an existing VM interpreter
supplies dependencies without reading any other project's source.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[3]
GUEST = "/tmp/oscar-long-context-20261027-decode"
VM_PYTHON = "/home/sunao2000.linux/gpt_new_oscar/.venv/bin/python"
OUTPUT = Path("reports/long_context_20261027/decode")
SDK = "/usr/local/Ascend/cann-9.1.0"
BASE_COMMIT = "232b563b69e01495046a9b0025ac9150bc11b9d7"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def stage():
    paths = [
        "tools/official_reference.py", "tools/phase.py",
        "configs/acceptance_official_bf16.json",
        "csrc/experiments/official_math_20261021/fixture.py",
        "csrc/experiments/official_math_20261021/probe.cpp",
        "csrc/experiments/long_context_20261027/decode_pair/CMakeLists.txt",
        "csrc/experiments/long_context_20261027/validate_decode_pair.py",
        "csrc/kernels/attention_cv_official_decode.cpp",
        "csrc/kernels/attention_official_tasks.cpp",
        "csrc/kernels/store_aligned_int2.cpp", "csrc/kernels/merge_lse.cpp",
    ]
    paths += [str(p.relative_to(ROOT)) for folder in ("csrc/kernels", "csrc/include")
              for p in sorted((ROOT / folder).glob("*.h"))]
    payloads = {name: (ROOT / name).read_bytes() for name in paths}
    payloads["csrc/kernels/long_context_decode_baseline.cpp"] = subprocess.check_output(
        ["git", "show", BASE_COMMIT + ":csrc/kernels/attention_cv_official_decode.cpp"], cwd=ROOT)
    # #148: rerunning after delivery must still compare against the original
    # control, never accidentally compare the new candidate against itself.
    manifest = {"guest": GUEST, "baseline_commit": BASE_COMMIT,
        "sha256": {name: sha(data) for name, data in payloads.items()}}
    out = ROOT / OUTPUT
    out.mkdir(parents=True, exist_ok=True)
    (out / "source_stage.json").write_text(json.dumps(manifest, indent=2) + "\n")
    payloads[str(OUTPUT / "source_stage.json")] = (out / "source_stage.json").read_bytes()
    subprocess.run(["limactl", "shell", "--workdir", "/tmp", "oscar", "--", "mkdir", GUEST], check=True, timeout=30)
    with tempfile.TemporaryFile() as stream:
        with tarfile.open(fileobj=stream, mode="w") as archive:
            for name, data in payloads.items():
                info = tarfile.TarInfo(name); info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        stream.seek(0)
        subprocess.run(["limactl", "shell", "--workdir", GUEST, "oscar", "--", "tar", "-xf", "-"],
                       stdin=stream, check=True, timeout=60)
    print(json.dumps({"staged_files": len(payloads), "guest": GUEST, "manifest": str(out / "source_stage.json")}))


def clean_completed_sdk_debug():
    """Only after the owned phase exits; preserve its real log/result files."""
    deleted = {}
    for name in ("npuchk", "cceprint"):
        folder = ROOT / name
        if folder.exists():
            deleted[name] = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
            shutil.rmtree(folder)
    return deleted


def run(resume=False):
    from tools.phase import run_phase, atomic_json
    out = ROOT / OUTPUT
    report_path = out / ("validation_resume.json" if resume else "validation.json")
    if report_path.exists():
        raise RuntimeError("retain the first run; validation.json already exists")
    report = {"scope": "local CANN and official SDK CPU-debug paired functionality",
              "npu": "not_run", "graph": "not_run", "performance": "not_run",
              "status": "running", "phases": [], "cases": []}
    if resume:
        previous = json.loads((out / "validation.json").read_text())
        if not all(any(row["name"] == name and row["returncode"] == 0 for row in previous["phases"])
                   for name in ("compile-cpu", "compile-device")):
            raise RuntimeError("resume requires completed CPU/device builds")
        report.update(cases=previous["cases"], build_reused_from="validation.json",
            first_failure="q4_mode0_fault-after: SDK GmAlloc SIGBUS before CV; disk100%, subsequent report write ENOSPC",
            cleanup_before_resume=clean_completed_sdk_debug())
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

    def phase(name, command, timeout=180):
        if resume: name = "resume-" + name
        result = run_phase(name, command, cwd=ROOT, log_dir=out / "logs", timeout=timeout)
        report["phases"].append({"name": name, "returncode": result.returncode, "log": result.log,
                                 "timed_out": result.timed_out, "cleanup_complete": result.cleanup_complete})
        atomic_json(report_path, report)
        if result.returncode:
            raise RuntimeError(f"{name}: rc={result.returncode}")
        report["phases"][-1]["completed_sdk_debug_removed_bytes"] = clean_completed_sdk_debug()
        atomic_json(report_path, report)

    try:
        source = "csrc/experiments/long_context_20261027/decode_pair"
        for arm, cpu in (() if resume else (("cpu", "ON"), ("device", "OFF"))):
            build = "build/decode-pair-" + arm
            phase("configure-" + arm, ["cmake", "-S", source, "-B", build,
                "-DSOC_VERSION=ascend910b4", "-DASCEND_HOME_PATH=" + SDK,
                "-DOSCAR_DECODE_PAIR_CPU_DEBUG=" + cpu])
            phase("compile-" + arm, ["cmake", "--build", build, "--parallel", "2"], 900)
        before = ROOT / "build/decode-pair-cpu/oscar_decode_before"
        after = ROOT / "build/decode-pair-cpu/oscar_decode_after"
        for family in ("q4", "q1"):
            for mode in (0, 1):
                for fault in (False, True):
                    case = family + "_mode" + str(mode) + ("_fault" if fault else "")
                    if any(row["case"] == case and row["status"] == "passed" for row in report["cases"]):
                        continue
                    fixture = out / "fixtures" / case
                    goldens = out / "goldens" / case
                    goldens.mkdir(parents=True, exist_ok=resume)
                    command = [sys.executable, "-m", "csrc.experiments.official_math_20261021.fixture",
                               "make", family, str(fixture)]
                    if fault: command.append("--fault")
                    if not fixture.exists(): phase(case + "-fixture", command)
                    base = [family, str(fixture), str(goldens), str(mode)]
                    if not (goldens / "tasks.bin").exists():
                        phase(case + "-prepare", [str(before), *base, "prepare-only"])
                    phase(case + "-oracle", [sys.executable, "-m", "csrc.experiments.official_math_20261021.fixture",
                        "golden", family, str(fixture), "--output", str(goldens), "--mode", str(mode)])
                    files = ["partial.bin", "lse.bin", "status.bin"]
                    if not fault: files += ["output.bin", "output_lse.bin"]
                    phase(case + "-before", [str(before), *base])
                    original = {name: (goldens / name).read_bytes() for name in files}
                    phase(case + "-after", [str(after), *base])
                    comparison = {name: {"before_sha256": sha(data), "after_sha256": sha((goldens / name).read_bytes()),
                                        "bytes": len(data), "bitwise_equal": data == (goldens / name).read_bytes()}
                                  for name, data in original.items()}
                    if not all(row["bitwise_equal"] for row in comparison.values()):
                        for name, data in original.items():
                            (goldens / ("before_" + name)).write_bytes(data)
                        atomic_json(out / (case + "_mismatch.json"), comparison)
                        raise RuntimeError(case + ": paired bytes differ")
                    report["cases"].append({"case": case, "status": "passed", "files": comparison,
                        "same_fixture_and_prepared_tasks": True, "independent_v4_oracle": not fault,
                        "exact_error_checks": fault})
                    atomic_json(report_path, report)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        atomic_json(report_path, report)
    print(json.dumps({"status": report["status"], "paired_cases": len(report["cases"]), "report": str(report_path)}))


def collect():
    out = ROOT / OUTPUT
    # Export logs/reports only; fixtures and tensors stay in the independent VM.
    names = [str(OUTPUT / "validation.json"), str(OUTPUT / "validation_resume.json"),
             str(OUTPUT / "long_validation.json"), str(OUTPUT / "build_artifacts.json"), str(OUTPUT / "logs")]
    data = subprocess.check_output(["limactl", "shell", "--workdir", GUEST, "oscar", "--", "tar", "-czf", "-", *names], timeout=60)
    (out / "vm_evidence.tar.gz").write_bytes(data)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.name.startswith(str(OUTPUT) + "/") or member.issym() or member.islnk() or ".." in Path(member.name).parts:
                raise RuntimeError("unexpected evidence archive path")
            if member.isfile():
                destination = out / "vm_evidence" / Path(member.name).relative_to(OUTPUT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.extractfile(member).read())
    print(json.dumps({"evidence": str(out / "vm_evidence"), "bytes": len(data)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("stage", "run", "resume", "collect"))
    args = parser.parse_args()
    {"stage": stage, "run": run, "resume": lambda: run(True), "collect": collect}[args.action]()
