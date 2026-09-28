# 档案 #75/#94/#95/#116/#117/#120/#125：相位输出实时到终端、完整日志落盘、保留真实退出码；本入口按用户要求只编译安装并直接启动，不含任何测试或probe。
"""Install, compile and prepare rotations, then exec the formal service; no tests, no probes."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from .deploy import plan as deploy_plan
from .phase import atomic_json, run_phase, terminal_line
from .target_cli import target_env

ROOT = Path(__file__).resolve().parents[1]

# The probe-free prefix of the deploy plan. probe-ops, native-synthetic and
# service-probe belong to scripts/install_probe_serve.sh only.
BUILD_PHASES = ("build-dependencies", "install-plugin", "build-ops", "prepare-rotations")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/target.json")
    parser.add_argument("--plan", action="store_true", help="print commands without installing or touching an NPU")
    parser.add_argument("--log-dir", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    log_dir = (args.log_dir or ROOT / "logs" / stamp).resolve()
    status = {"status": "running", "phases": [], "probes": "none"}
    if not args.plan:
        atomic_json(log_dir / "status.json", status)
        print(f"[oscar] install → build → rotations → serve (no tests, no probes); logs={log_dir}", flush=True)
    try:
        config = json.loads(args.config.read_text())
    except (OSError, ValueError) as exc:
        if not args.plan:
            (log_dir / "config.log").write_text(f"FAILED phase=config: {exc}\n")
            status.update(status="failed", failed_phase="config", error=str(exc))
            atomic_json(log_dir / "status.json", status)
        raise
    stages = [(name, command) for name, command in deploy_plan(args.config, log_dir) if name in BUILD_PHASES]
    if [name for name, _ in stages] != list(BUILD_PHASES):
        raise RuntimeError(f"deploy plan no longer provides the expected build phases {BUILD_PHASES}; refusing to improvise")
    serve_command = [sys.executable, "-m", "tools.target_cli", "--config", str(args.config.resolve())]
    if args.plan:
        print(json.dumps({"stages": stages, "serve": serve_command, "probes": "none",
                          "target_devices": config["devices"]}, indent=2))
        return 0
    try:
        env = target_env(config)
    except (ValueError, KeyError) as exc:
        (log_dir / "config.log").write_text(f"FAILED phase=config: {exc}\n")
        status.update(status="failed", failed_phase="config", error=str(exc))
        atomic_json(log_dir / "status.json", status)
        raise
    env["OSCAR_TARGET_CONFIG"] = str(args.config.resolve())
    env["PYTHONUNBUFFERED"] = "1"
    print(f"[oscar] devices={env.get('ASCEND_RT_VISIBLE_DEVICES', 'diagnostic')} port={config['port']}", flush=True)
    rc = 0
    try:
        for name, command in stages:
            result = run_phase(name, command, cwd=ROOT, log_dir=log_dir,
                               timeout=config["phase_timeout_seconds"], env=env,
                               grace=config["shutdown_timeout_seconds"], heartbeat=60)
            status["phases"].append(asdict(result))
            atomic_json(log_dir / "status.json", status)
            if result.returncode:
                status.update(status="failed", failed_phase=name)
                rc = result.returncode
                break
        if rc == 0:
            status["status"] = "build_passed"
    finally:
        atomic_json(log_dir / "status.json", status)
    if rc != 0:
        terminal_line(f"[oscar] build phase failed rc={rc}; formal serve prohibited; logs={log_dir}", stderr=True)
        return rc
    status["status"] = "serving"
    atomic_json(log_dir / "status.json", status)
    terminal_line("[oscar] build complete; starting formal service in the foreground (no probes)")
    os.execvpe(sys.executable, serve_command, env)
    return 127  # unreachable: execvpe replaces this process


if __name__ == "__main__":
    sys.exit(main())
