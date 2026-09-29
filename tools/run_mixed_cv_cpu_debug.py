"""Run complete mixed CV CPU-debug cases with bounded per-kernel C16 phases.

Archive #126/#129/#148-151 and startup D.4: a long official CPU simulation
can exceed the existing 120 s per-process limit when fe0 and C16 run in one
process. This runner preserves the same full golden and actual kernels, saving
the checked fe0 bytes in a fresh artifact directory before a separate C16
process compares them bitwise and checks the independent merge oracle. It
does not provide target-NPU precision, graph, or performance evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import uuid

from .phase import atomic_json, run_phase


TIMEOUT_SECONDS = 120  # Existing official CPU-debug bound, per actual kernel.


def _fixture_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(p for p in path.iterdir() if p.is_file() and
                   p.suffix in {".bin", ".txt"})
    if not files or not (path / "shape.txt").is_file():
        raise ValueError(f"missing CPU golden files: {path}")
    for item in files:
        digest.update(item.name.encode())
        digest.update(b"\0")
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _reference_sizes(path: Path) -> dict[str, int]:
    values = [int(value) for value in (path / "shape.txt").read_text().split()]
    if len(values) < 15:
        raise ValueError("mixed CPU shape lacks requests/splits")
    tokens, query_heads, kv_heads, dim = values[:4]
    splits = values[14]
    if min(tokens, query_heads, kv_heads, dim, splits) <= 0:
        raise ValueError("mixed CPU shape has nonpositive dimensions")
    segments = 3 * splits
    return {"partial.bin": tokens * query_heads * segments * dim * 4,
            "lse.bin": tokens * query_heads * segments * 4,
            "status.bin": tokens * kv_heads * segments * 2 * 4}


def _reference_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for name in ("partial.bin", "lse.bin", "status.bin"):
        digest.update(name.encode())
        with (path / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _passed_phase(result, expected_mode: str) -> tuple[bool, list[str]]:
    log = Path(result.log).read_text(errors="replace")
    markers = [line for line in log.splitlines()
               if line.startswith('{"backend":"ascendc_cpu_debug"')]
    stages = [line for line in log.splitlines() if line.startswith("CPU_STAGE ")]
    try:
        marker = json.loads(markers[0]) if len(markers) == 1 else {}
    except json.JSONDecodeError:
        marker = {}
    passed = (result.returncode == 0 and not result.timed_out and
              result.cleanup_complete and
              len(markers) == 1 and marker.get("op") == expected_mode and
              marker.get("status") == "passed" and
              "[ERROR]" not in log and "error happened!" not in log)
    return passed, stages


def run_cases(*, executable: Path, cases_path: Path, output: Path,
              log_dir: Path, artifact_root: Path,
              select: set[str] | None = None) -> int:
    executable = executable.resolve(strict=True)
    cases = json.loads(cases_path.read_text())
    if not isinstance(cases, list):
        raise ValueError("CPU cases manifest must be a list")
    if select is not None:
        available = {Path(case["path"]).name for case in cases}
        missing = select - available
        if missing:
            raise ValueError(f"unknown CPU case selection: {sorted(missing)}")
    report = {"backend": "official_ascendc_cpu_debug", "status": "running",
              "timeout_seconds_per_phase": TIMEOUT_SECONDS,
              "npu_acceptance": "not_run", "cases": []}
    atomic_json(output, report)
    chosen = 0
    for index, case in enumerate(cases):
        mode = case["op"]
        case_dir = Path(case["path"]).resolve(strict=True)
        case_name = case_dir.name
        if select is not None and case_name not in select:
            continue
        chosen += 1
        initial_sha = _fixture_sha256(case_dir)
        split = mode == "mixed_cluster16" and int((case_dir / "shape.txt").read_text().split()[0]) >= 335
        reference_dir = None
        if split:
            reference_dir = artifact_root.resolve() / f"{index:03d}-{case_name}-{uuid.uuid4().hex}"
            reference_dir.mkdir(parents=True, exist_ok=False)
            phases = ("mixed_C16_reference", "mixed_C16_candidate")
        else:
            phases = (mode,)
        case_report = {"case": case_name, "op": mode,
                       "golden_sha256": initial_sha,
                       "execution": "separate_complete_kernels" if split else "single_process",
                       "reference_dir": str(reference_dir) if reference_dir else None,
                       "phases": [], "status": "running"}
        report["cases"].append(case_report)
        atomic_json(output, report)
        for phase in phases:
            if _fixture_sha256(case_dir) != initial_sha:
                raise RuntimeError(f"CPU golden changed before {phase}: {case_dir}")
            if phase == "mixed_C16_candidate" and (
                    reference_dir is None or
                    _reference_sha256(reference_dir) != case_report.get("reference_sha256")):
                raise RuntimeError("fe0 reference bytes changed before C16 candidate")
            command = [str(executable), phase, str(case_dir)]
            if reference_dir is not None:
                command.append(str(reference_dir))
            result = run_phase(f"{index:03d}-{case_name}-{phase}", command,
                               cwd=Path.cwd(), log_dir=log_dir,
                               timeout=TIMEOUT_SECONDS, grace=5)
            passed, stages = _passed_phase(result, phase)
            phase_report = {"phase": phase, "status": "passed" if passed else "failed",
                            "returncode": result.returncode,
                            "timed_out": result.timed_out,
                            "elapsed_seconds": result.elapsed_seconds,
                            "log": result.log, "stages": stages}
            case_report["phases"].append(phase_report)
            if _fixture_sha256(case_dir) != initial_sha:
                phase_report["status"] = "failed_input_changed"
                passed = False
            if passed and phase == "mixed_C16_reference":
                assert reference_dir is not None
                sizes = _reference_sizes(case_dir)
                for name, expected_size in sizes.items():
                    if (reference_dir / name).stat().st_size != expected_size:
                        raise RuntimeError(f"wrong fe0 reference artifact size: {name}")
                phase_report["reference_sizes"] = sizes
                case_report["reference_sha256"] = _reference_sha256(reference_dir)
            if passed and phase == "mixed_C16_candidate" and (
                    reference_dir is None or
                    _reference_sha256(reference_dir) != case_report["reference_sha256"]):
                phase_report["status"] = "failed_reference_changed"
                passed = False
            atomic_json(output, report)
            print(f"CPU_MIXED case={case_name} phase={phase} status={phase_report['status']} "
                  f"rc={result.returncode} elapsed_s={result.elapsed_seconds:.3f} "
                  f"last_stage={stages[-1] if stages else 'none'}", flush=True)
            if not passed:
                case_report["status"] = "failed"
                report["status"] = "failed"
                atomic_json(output, report)
                return 1
        case_report["status"] = "passed"
        atomic_json(output, report)
    if chosen == 0:
        raise ValueError("no requested mixed CV CPU cases matched")
    report["status"] = "passed"
    atomic_json(output, report)
    print(f"CPU_MIXED_SUMMARY status=passed cases={chosen} output={output}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=Path("build/cpu/oscar_cv_cpu"))
    parser.add_argument("--cases", type=Path, default=Path("reports/vm-cv-mixed-balanced/goldens/cases.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/vm-cv-mixed-balanced/validation.json"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs/vm-cv-mixed-balanced"))
    parser.add_argument("--artifact-root", type=Path,
                        default=Path("reports/vm-cv-mixed-balanced/reference-artifacts"))
    parser.add_argument("--select", action="append", help="Select a case directory basename; repeatable")
    args = parser.parse_args()
    try:
        return run_cases(executable=args.executable, cases_path=args.cases,
                         output=args.output, log_dir=args.log_dir,
                         artifact_root=args.artifact_root,
                         select=set(args.select) if args.select else None)
    except Exception as error:
        try:
            report = json.loads(args.output.read_text())
            report["status"] = "failed"
            report["error"] = f"{type(error).__name__}: {error}"
            atomic_json(args.output, report)
        except (OSError, ValueError) as report_error:
            print(f"CPU_MIXED_REPORT_ERROR original={type(error).__name__}: {error}; "
                  f"report={type(report_error).__name__}: {report_error}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
