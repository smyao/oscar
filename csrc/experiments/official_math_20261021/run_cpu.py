"""Archive #126/#145/#148/#164: bounded actual CPU-debug functional matrix.

Uses the official SDK CPU executor, not an NPU or performance experiment.
All first failures, complete phase logs, source hashes and real RCs persist.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

from tools.phase import run_phase, atomic_json

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "reports/official-cpu")
    parser.add_argument("--case", action="append", help="family:mode[:fault|empty]; default complete bounded matrix")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    cases = args.case or ["q4:0", "q4:1", "q1:1", "c4:0", "c4:1", "c16:0", "c16:1", "q4:1:fault", "c4:1:fault", "c16:1:fault", "c16:0:empty"]
    report = {"backend": "official_AscendC_CPU_debug", "npu": "not_run", "performance": "not_run",
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted((ROOT / "csrc/kernels").glob("*official*.cpp"))},
              "phases": [], "cases": [], "status": "running"}
    def phase(name, command, timeout=180):
        result = run_phase(name, command, cwd=ROOT, log_dir=out / "logs", timeout=timeout)
        report["phases"].append({"name": name, "returncode": result.returncode, "log": result.log,
                                 "timed_out": result.timed_out, "cleanup_complete": result.cleanup_complete})
        atomic_json(out / "summary.json", report)
        if result.returncode:
            report.update(status="failed", failed_phase=name)
            atomic_json(out / "summary.json", report)
            raise RuntimeError(f"{name}: rc={result.returncode}")
    if not args.skip_build:
        phase("configure-cpu", ["cmake", "-S", "csrc/experiments/official_math_20261021", "-B", "build/official-cpu",
              "-DSOC_VERSION=ascend910b4", "-DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0", "-DOSCAR_OFFICIAL_CPU_DEBUG=ON"])
        phase("compile-cpu", ["cmake", "--build", "build/official-cpu", "--parallel", "2"], 900)
    for item in cases:
        fields = item.split(":")
        family, mode = fields[:2]; fault = len(fields) == 3 and fields[2] == "fault"
        empty_history = len(fields) == 3 and fields[2] == "empty"
        name = family + "_mode" + mode + ("_fault" if fault else "_empty_history" if empty_history else "")
        fixture, golden = out / "fixtures" / name, out / "goldens" / name
        golden.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, "-m", "csrc.experiments.official_math_20261021.fixture", "make", family, str(fixture)]
        if fault: command.append("--fault")
        if empty_history: command.append("--empty-history")
        phase(name + "-fixture", command)
        binary = ROOT / "build/official-cpu" / ("oscar_official_decode_cpu" if family in ("q4", "q1") else "oscar_official_clusters_cpu")
        base = [str(binary), family, str(fixture), str(golden), mode]
        phase(name + "-prepare", base + ["prepare-only"])
        phase(name + "-golden", [sys.executable, "-m", "csrc.experiments.official_math_20261021.fixture",
              "golden", family, str(fixture), "--output", str(golden), "--mode", mode])
        phase(name + "-execute", base)
        if empty_history:
            import torch
            saved = torch.load(fixture / "fixture.pt", weights_only=False)
            n, heads, dim = saved["cpu"]["q"].shape
            splits = saved["spec"].splits
            partial = torch.frombuffer(bytearray((golden / "partial.bin").read_bytes()), dtype=torch.float32).reshape(n, heads, 3 * splits, dim)
            lse = torch.frombuffer(bytearray((golden / "lse.bin").read_bytes()), dtype=torch.float32).reshape(n, heads, 3 * splits)
            if bool(partial[:, :, :splits].contiguous().view(torch.int32).ne(0).any()) or not bool(torch.isneginf(lse[:, :, :splits]).all()):
                report.update(status="failed", failed_phase=name + "-exact-empty-history")
                atomic_json(out / "summary.json", report)
                raise RuntimeError("all-masked history must publish exact positive zero and negative-infinite LSE")
        scope = "writer/query-copy/prepare/CV" if family in ("q4", "q1") else "prepare/mature-cluster/tail-C1/CV"
        report["cases"].append({"case": name, "status": "passed", "scope": scope + ("/exact-error-domain" if fault else "/FP32-merge"),
                                **({"empty_history_exact_zero_and_negative_infinity": True} if empty_history else {})})
        atomic_json(out / "summary.json", report)
    report["status"] = "passed"
    atomic_json(out / "summary.json", report)
    print(json.dumps({"status": "passed", "cases": len(report["cases"]), "summary": str(out / "summary.json")}))


if __name__ == "__main__":
    main()
