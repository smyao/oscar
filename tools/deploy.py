# 档案 #27/#51/#52/#74–#86/#94–#97/#117/#120：独立相位、有界退出、原生完整性和真实就绪判据。
"""Target workflow. Incomplete runtime gates stop deployment explicitly."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from .phase import atomic_json, run_phase
from .target_cli import target_env

ROOT = Path(__file__).resolve().parents[1]


def plan(config_path: Path, log_dir: Path) -> list[tuple[str, list[str]]]:
    python = sys.executable
    cfg = str(config_path.resolve())
    return [
        ("environment", [python, "-m", "tools.environment", "--output", str(log_dir / "environment.json"), "--native-root", "/vllm-workspace/vllm", "--native-root", "/vllm-workspace/vllm-ascend"]),
        ("runtime-readiness", [python, "-m", "tools.readiness", "--output", str(log_dir / "readiness.json")]),
        ("install-plugin", [python, "-m", "pip", "install", "--no-deps", "-e", str(ROOT)]),
        ("build-ops", [python, "-m", "tools.build_ops", "--log-dir", str(log_dir / "build")]),
        ("probe-ops", [python, "-m", "tools.probe_ops", "--output", str(log_dir / "operators.json")]),
        ("prepare-rotations", [python, "-m", "tools.prepare_rotations", "--config", cfg]),
        ("native-integrity", [python, "-m", "tools.environment", "--output", str(log_dir / "environment-after.json"), "--native-root", "/vllm-workspace/vllm", "--native-root", "/vllm-workspace/vllm-ascend", "--compare", str(log_dir / "environment.json")]),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/target.json")
    parser.add_argument("--plan", action="store_true", help="print commands without installing or touching an NPU")
    parser.add_argument("--only", choices=["environment", "build-ops", "probe-ops", "prepare-rotations", "runtime-readiness"])
    parser.add_argument("--log-dir", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    log_dir = (args.log_dir or ROOT / "logs" / stamp).resolve()
    status = {"status": "running", "npu_acceptance": "not_run", "phases": []}
    if not args.plan:
        atomic_json(log_dir / "status.json", status)
    try:
        config = json.loads(args.config.read_text())
    except (OSError, ValueError) as exc:
        if not args.plan:
            (log_dir / "config.log").write_text(f"FAILED phase=config: {exc}\n")
            status.update(status="failed", failed_phase="config", error=str(exc))
            atomic_json(log_dir / "status.json", status)
        raise
    stages = plan(args.config, log_dir)
    if args.only:
        stages = [x for x in stages if x[0] == args.only]
    if args.plan:
        print(json.dumps({"stages": stages, "serve": [sys.executable, "-m", "tools.target_cli", "--config", str(args.config.resolve())],
                          "target_devices": config["devices"], "production_status": "incomplete_runtime; not deployable yet"}, indent=2))
        return 0
    # Observation can run without hardware ownership; actual NPU phases require it.
    try:
        env = os.environ.copy() if args.only in {"environment", "runtime-readiness"} else target_env(config)
    except (ValueError, KeyError) as exc:
        (log_dir / "config.log").write_text(f"FAILED phase=config: {exc}\n")
        status.update(status="failed", failed_phase="config", error=str(exc))
        atomic_json(log_dir / "status.json", status)
        raise
    rc = 0
    try:
        for name, command in stages:
            if name == "native-integrity":
                continue  # Always executed in finally, including on earlier failures.
            result = run_phase(name, command, cwd=ROOT, log_dir=log_dir,
                               timeout=config["phase_timeout_seconds"], env=env,
                               grace=config["shutdown_timeout_seconds"])
            status["phases"].append(asdict(result))
            atomic_json(log_dir / "status.json", status)
            if result.returncode:
                status.update(status="failed", failed_phase=name)
                rc = result.returncode
                break
        if rc == 0:
            if args.only:
                status["status"] = "selected_phase_passed"
            else:
                error = "full-service TP4/MTP/graph and NPU-resource-release probes are not implemented; formal serve prohibited"
                (log_dir / "full-service-probe.log").write_text("FAILED phase=full-service-probe: " + error + "\n")
                status.update(status="failed", failed_phase="full-service-probe", error=error)
                print(error, file=sys.stderr)
                rc = 1
    finally:
        if not args.only and (log_dir / "environment.json").is_file():
            _, command = next(stage for stage in stages if stage[0] == "native-integrity")
            integrity = run_phase("native-integrity", command, cwd=ROOT, log_dir=log_dir,
                                  timeout=config["phase_timeout_seconds"], env=env,
                                  grace=config["shutdown_timeout_seconds"])
            status["phases"].append(asdict(integrity))
            if integrity.returncode and rc == 0:
                rc = integrity.returncode
                status.update(status="failed", failed_phase="native-integrity")
        atomic_json(log_dir / "status.json", status)
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"[oscar] FAILED deployment: {exc}", file=sys.stderr)
        sys.exit(1)
