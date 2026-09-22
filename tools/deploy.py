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
        ("runtime-readiness", [python, "-m", "tools.readiness", "--source-only", "--output", str(log_dir / "source-readiness.json")]),
        ("build-dependencies", [python,"-m","pip","install","pybind11>=3","cmake>=3.26","ninja","pytest"]),
        ("install-plugin", [python, "-m", "pip", "install", "--no-deps", "-e", str(ROOT)]),
        ("build-ops", [python, "-m", "tools.build_ops", "--log-dir", str(log_dir / "build")]),
        ("binary-readiness", [python,"-m","tools.readiness","--output",str(log_dir/"binary-readiness.json")]),
        ("probe-ops", [python, "-m", "tools.probe_ops", "--output", str(log_dir / "operators.json")]),
        ("probe-cv", [python,"-m","pytest","-q","--maxfail=1",str(ROOT/"tests/test_cv_contracts.py"),str(ROOT/"tests/test_rotation_npu.py"),
                      "--junitxml="+str(log_dir/"cv-npu.xml")]),
        ("prepare-rotations", [python, "-m", "tools.prepare_rotations", "--config", cfg]),
        ("service-probe", [python,"-m","tools.service_probe","--config",cfg,
                           "--log-dir",str(log_dir/"service-probe"),"--output",str(log_dir/"service-probe.json")]),
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
                          "target_devices": config["devices"], "production_status": "requires_target_probes_and_resource_release"}, indent=2))
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
    env["OSCAR_TARGET_CONFIG"]=str(args.config.resolve())
    env["OSCAR_RUN_NPU_TESTS"]="1"
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
                evidence=log_dir/"service-probe.json"
                probe=json.loads(evidence.read_text()) if evidence.is_file() else {}
                if probe.get("status")!="passed" or probe.get("resource_release")!="passed":
                    error="full-service evidence or owned NPU resource release is missing; formal serve prohibited"
                    (log_dir / "full-service-probe.log").write_text("FAILED phase=full-service-probe: "+error+"\n")
                    status.update(status="failed",failed_phase="full-service-probe",error=error)
                    rc=1
                else:
                    status["status"]="probes_passed"
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
    if rc==0 and not args.only:
        # The managed server preserves ownership and validates a real request
        # before staying in the foreground. It never starts after failed probes.
        from .service_probe import main as service_main
        sys.argv=["service_probe","--config",str(args.config.resolve()),"--serve",
                  "--log-dir",str(log_dir/"serve"),"--output",str(log_dir/"serve.json")]
        return service_main()
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"[oscar] FAILED deployment: {exc}", file=sys.stderr)
        sys.exit(1)
