# Archive G25-G34/#4-22/#51/#52: official AscendC CPU-debug execution with external goldens and bounded processes.
"""Run compiled CANN CPU-debug kernels, distinctly from CPU math or NPU acceptance."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
from .phase import atomic_json, run_phase


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable",type=Path,default=Path("build/cpu/oscar_primitive_cpu"))
    parser.add_argument("--cases",type=Path,default=Path("artifacts/cpu_cases/cases.json"))
    parser.add_argument("--output",type=Path,default=Path("reports/ascendc_cpu_debug.json"))
    parser.add_argument("--log-dir",type=Path,default=Path("logs/cpu-debug"))
    args=parser.parse_args()
    cases=json.loads(args.cases.read_text())
    report={"backend":"official_ascendc_cpu_debug","status":"running","cases":[],"npu_acceptance":"not_run"}
    atomic_json(args.output,report)
    for i,case in enumerate(cases):
        result=run_phase(f"{i:03d}-{case['op']}",[str(args.executable.resolve()),case['op'],str(Path(case['path']).resolve())],
                         cwd=Path.cwd(),log_dir=args.log_dir,timeout=120,grace=5)
        log=Path(result.log).read_text()
        # The official debugger may return zero after a failed child. Require
        # the golden-comparison completion marker from our parent harness.
        markers=[line for line in log.splitlines() if line.startswith('{"backend":"ascendc_cpu_debug"')]
        passed=(result.returncode==0 and len(markers)==1
                and json.loads(markers[0]).get("status")=="passed"
                and "[ERROR]" not in log and "error happened!" not in log)
        if case.get("expect_kernel_error"):
            passed=(not result.timed_out and "oscar_status_guard_kernel" in log
                    and ("[ERROR]" in log or "error happened!" in log))
        report["cases"].append({**case,"status":"passed" if passed else "failed","returncode":result.returncode,"log":result.log})
        atomic_json(args.output,report)
        if not passed:
            report["status"]="failed";atomic_json(args.output,report)
            return 1
    report["status"]="passed";atomic_json(args.output,report)
    print(json.dumps(report,indent=2))
    return 0


if __name__=="__main__":
    sys.exit(main())
