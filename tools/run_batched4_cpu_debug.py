"""Bounded official CPU-debug A/B for the isolated M512 batched4 experiment.

Archive #126/#129/#145/#148-153 and startup D.4: normal modes run old fast
C4 and new batched4 in one process. The unchanged full old-only golden can
run as two independently bounded real kernel processes, with the checked
old-C4 bytes read and compared by the new-kernel process. CPU-debug is not
target NPU precision, graph, service quality, or a speed measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import uuid

from .phase import atomic_json, run_phase


TIMEOUT_SECONDS = 120


def _sha256_files(directory: Path, names: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode())
        digest.update(b"\0")
        with (directory / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _fixture_sha(directory: Path) -> str:
    names = tuple(sorted(p.name for p in directory.iterdir()
                         if p.is_file() and p.suffix in {".bin", ".txt"}))
    if "shape.txt" not in names:
        raise ValueError("CPU golden lacks shape.txt")
    return _sha256_files(directory, names)


def _artifact_sizes(directory: Path) -> dict[str, int]:
    values = [int(value) for value in (directory / "shape.txt").read_text().split()]
    if len(values) < 13:
        raise ValueError("invalid CPU golden shape")
    tokens, heads, kv_heads, dim, cores = values[0], values[1], values[2], values[3], values[12]
    splits = values[14] if len(values) >= 15 else 1
    if min(tokens, heads, kv_heads, dim, cores, splits) <= 0:
        raise ValueError("nonpositive CPU golden shape")
    segments = 3 * splits
    return {"partial.bin": tokens * heads * segments * dim * 4,
            "lse.bin": tokens * heads * segments * 4,
            "status.bin": tokens * kv_heads * segments * 2 * 4,
            "stats.bin": cores * 8 * 8}


def _phase_passed(result) -> tuple[bool, list[str]]:
    log = Path(result.log).read_text(errors="replace")
    markers = [line for line in log.splitlines()
               if line.startswith('{"backend":"ascendc_cpu_debug"')]
    stages = [line for line in log.splitlines() if line.startswith("BATCHED4_STAGE ")]
    try:
        marker = json.loads(markers[0]) if len(markers) == 1 else {}
    except json.JSONDecodeError:
        marker = {}
    passed = (result.returncode == 0 and not result.timed_out and
              result.cleanup_complete and len(markers) == 1 and
              marker == {"backend": "ascendc_cpu_debug", "op": "batched4",
                         "status": "passed"} and
              "[ERROR]" not in log and "error happened!" not in log)
    return passed, stages


def run_case(*, executable: Path, fixture: Path, mode: str,
             output: Path, log_dir: Path, artifact_root: Path) -> int:
    executable = executable.resolve(strict=True)
    fixture = fixture.resolve(strict=True)
    if mode not in {"normal", "poison", "bad_meta", "nan_query", "c1",
                    "old_only", "poison_old_only"}:
        raise ValueError(f"unsupported batched4 CPU mode: {mode}")
    fixture_sha = _fixture_sha(fixture)
    split = mode == "old_only"
    artifact_dir = None
    if split:
        artifact_dir = artifact_root.resolve() / f"{fixture.name}-{uuid.uuid4().hex}"
        artifact_dir.mkdir(parents=True, exist_ok=False)
        phases = ("old_only_reference", "old_only_candidate")
    else:
        phases = (mode,)
    report = {"scope": "official_ascendc_cpu_debug", "status": "running",
              "mode": mode, "fixture": str(fixture), "fixture_sha256": fixture_sha,
              "phase_timeout_seconds": TIMEOUT_SECONDS,
              "execution": "separate_complete_kernels" if split else "same_process",
              "artifact_dir": str(artifact_dir) if artifact_dir else None,
              "npu_acceptance": "not_run", "phases": []}
    atomic_json(output, report)
    artifact_sha = None
    for phase in phases:
        if _fixture_sha(fixture) != fixture_sha:
            raise RuntimeError("frozen CPU golden changed between phases")
        if phase == "old_only_candidate" and (
                artifact_dir is None or
                _sha256_files(artifact_dir, tuple(_artifact_sizes(fixture))) != artifact_sha):
            raise RuntimeError("checked old-C4 reference bytes changed before candidate")
        command = [str(executable), phase, str(fixture)]
        if artifact_dir is not None:
            command.append(str(artifact_dir))
        result = run_phase(f"batched4-{fixture.name}-{phase}", command,
                           cwd=Path.cwd(), log_dir=log_dir,
                           timeout=TIMEOUT_SECONDS, grace=5)
        passed, stages = _phase_passed(result)
        phase_report = {"phase": phase, "status": "passed" if passed else "failed",
                        "returncode": result.returncode, "timed_out": result.timed_out,
                        "cleanup_complete": result.cleanup_complete,
                        "elapsed_seconds": result.elapsed_seconds,
                        "log": result.log, "stages": stages}
        report["phases"].append(phase_report)
        if _fixture_sha(fixture) != fixture_sha:
            phase_report["status"] = "failed_input_changed"
            passed = False
        if passed and phase == "old_only_reference":
            assert artifact_dir is not None
            sizes = _artifact_sizes(fixture)
            for name, expected in sizes.items():
                if (artifact_dir / name).stat().st_size != expected:
                    raise RuntimeError(f"wrong old-C4 reference byte count: {name}")
            artifact_sha = _sha256_files(artifact_dir, tuple(sizes))
            report["reference_sha256"] = artifact_sha
            report["reference_byte_counts"] = sizes
        if passed and phase == "old_only_candidate" and (
                artifact_dir is None or
                _sha256_files(artifact_dir, tuple(_artifact_sizes(fixture))) != artifact_sha):
            phase_report["status"] = "failed_reference_changed"
            passed = False
        atomic_json(output, report)
        print(f"BATCHED4_CPU mode={mode} phase={phase} status={phase_report['status']} "
              f"rc={result.returncode} elapsed_s={result.elapsed_seconds:.3f} "
              f"last_stage={stages[-1] if stages else 'none'}", flush=True)
        if not passed:
            report["status"] = "failed"
            atomic_json(output, report)
            return 1
    report["status"] = "passed"
    atomic_json(output, report)
    print(f"BATCHED4_CPU_SUMMARY status=passed mode={mode} output={output}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        return run_case(executable=args.executable, fixture=args.fixture,
                        mode=args.mode, output=args.output, log_dir=args.log_dir,
                        artifact_root=args.artifact_root)
    except Exception as error:
        if args.output.exists():
            try:
                report = json.loads(args.output.read_text())
                report["status"] = "failed"
                report["error"] = f"{type(error).__name__}: {error}"
                atomic_json(args.output, report)
            except (ValueError, OSError):
                pass
        raise


if __name__ == "__main__":
    sys.exit(main())
