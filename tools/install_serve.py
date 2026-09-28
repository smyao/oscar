# 档案 #75/#94/#95/#116/#117/#120/#125：相位输出实时到终端、完整日志落盘、保留真实退出码；本入口按用户要求只编译安装并直接启动，不含任何测试或probe。
# #148/#150：安装最新产物不等于启用优化；显式candidate将同一有效配置传给所有相位和服务。
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
from .serving_variants import variant_config, variant_features

ROOT = Path(__file__).resolve().parents[1]

# The probe-free prefix of the deploy plan. probe-ops, native-synthetic and
# service-probe belong to scripts/install_probe_serve.sh only.
BUILD_PHASES = ("build-dependencies", "install-plugin", "build-ops", "prepare-rotations")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/target.json")
    parser.add_argument("--plan", action="store_true", help="print commands without installing or touching an NPU")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--variant", choices=("baseline", "candidate"),
                        help="candidate enables C4, later-MTP q1 and fast unpack; baseline selects fe0; omitted respects config")
    parser.add_argument("--rear-cards", action="store_true",
                        help="use physical Ascend devices 4,5,6,7 and port 7878 for this launch only")
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    log_dir = (args.log_dir or ROOT / "logs" / stamp).resolve()
    status = {"status": "running", "phases": [], "probes": "none"}
    if not args.plan:
        atomic_json(log_dir / "status.json", status)
        print(f"[oscar] install → build → rotations → serve (no tests, no probes); logs={log_dir}", flush=True)
    try:
        config = variant_config(json.loads(args.config.read_text()), args.variant)
        if args.rear_cards:
            # Explicit user-selected placement, never inherited from a prior
            # task's environment. Validate the supplied device config first.
            target_env(config, base={})
            config.update(devices=[4, 5, 6, 7], port=7878,
                          device_policy="Explicit install_serve --rear-cards: physical NPU 4-7, port 7878")
    except (OSError, ValueError) as exc:
        if not args.plan:
            (log_dir / "config.log").write_text(f"FAILED phase=config: {exc}\n")
            status.update(status="failed", failed_phase="config", error=str(exc))
            atomic_json(log_dir / "status.json", status)
        raise
    enabled = config.get("experimental_history_reuse", False)
    variant = "candidate" if enabled else "baseline"
    write_effective = args.variant is not None or args.rear_cards
    effective_path = (log_dir / "effective-target.json" if write_effective
                      else args.config.resolve())
    stages = []
    for name, command in deploy_plan(args.config, log_dir):
        if name not in BUILD_PHASES:
            continue
        # Plan mode remains read-only: derive commands from the original
        # config and bind config-taking phases to the future effective file.
        if "--config" in command:
            command[command.index("--config") + 1] = str(effective_path)
        stages.append((name, command))
    if [name for name, _ in stages] != list(BUILD_PHASES):
        raise RuntimeError(f"deploy plan no longer provides the expected build phases {BUILD_PHASES}; refusing to improvise")
    serve_command = [sys.executable, "-m", "tools.target_cli", "--config", str(effective_path)]
    status.update(variant=variant, original_config=str(args.config.resolve()),
                  effective_config=str(effective_path),
                  placement="rear" if args.rear_cards else "configured",
                  target_devices=config["devices"], port=config["port"],
                  optimizations=variant_features(config))
    if args.plan:
        print(json.dumps({"stages": stages, "serve": serve_command, "probes": "none",
                          "variant": variant, "optimizations": status["optimizations"],
                          "target_devices": config["devices"], "port": config["port"],
                          "placement": status["placement"]}, indent=2))
        return 0
    try:
        env = target_env(config)
    except (ValueError, KeyError) as exc:
        (log_dir / "config.log").write_text(f"FAILED phase=config: {exc}\n")
        status.update(status="failed", failed_phase="config", error=str(exc))
        atomic_json(log_dir / "status.json", status)
        raise
    if write_effective:
        atomic_json(effective_path, config)
    env["OSCAR_TARGET_CONFIG"] = str(effective_path)
    env["PYTHONUNBUFFERED"] = "1"
    print(f"[oscar] SERVE_MODE variant={variant} C4={'on' if enabled else 'off'} "
          f"Q1={'on' if enabled else 'off'} FAST_UNPACK={'on' if config.get('experimental_fast_unpack', False) else 'off'} "
          f"placement={status['placement']} config={effective_path}", flush=True)
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
