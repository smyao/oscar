# Archive G21/G22/#94/#95/#125/#135: supervise only this launch's process
# group, keep live/durable failures, and diagnose startup memory without
# changing utilization, touching an existing service, or claiming NPU release.
"""Foreground service ownership, with no installation, probe or readiness gate."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import sys

from .phase import PhaseResult, atomic_json, run_phase, terminal_line
from .target_cli import ROOT, serve_argv, target_env


_STARTUP_MEMORY = re.compile(
    r"Free memory on device \((?P<free>[0-9.]+)/(?P<total>[0-9.]+) GiB\) "
    r"on startup is less than desired GPU memory utilization "
    r"\((?P<utilization>[0-9.]+), (?P<required>[0-9.]+) GiB\)"
)


def startup_diagnostic(log: Path, config: dict) -> dict | None:
    """Read this failed launch's bounded log prefix, without any device query.

    Native worker._init_device checks this snapshot before ModelRunner/model
    loading. The message establishes the free-memory deficit, never its owner.
    This optional explanation cannot override the real process return code.
    """
    try:
        with log.open("rb") as stream:
            prefix = stream.read(8 * 1024 * 1024).decode("utf-8", errors="replace")
    except OSError:
        return None
    match = _STARTUP_MEMORY.search(prefix)
    if match is None:
        return None
    return {
        "kind": "native_startup_memory_insufficient",
        "source": str(log),
        "native_message": match.group(0),
        "reported_gib": {key: match[key] for key in ("free", "total", "required")},
        "reported_utilization": match["utilization"],
        "configured_devices": config["devices"],
        "memory_owner": "not_established",
        "npu_resource_release": "not_measured",
        "read_only_checks": ["npu-smi info", "ps -eo pid,ppid,pgid,etime,args"],
    }


def shell_returncode(code: int) -> int:
    """Preserve POSIX signal meaning; subprocess -SIGTERM becomes shell 143."""
    return 128 - code if code < 0 else code


def run_service(config: dict, config_path: Path, *, log_dir: Path,
                env: dict[str, str] | None = None, native: bool = False) -> PhaseResult:
    environment = target_env(config, base=env, native=native)
    config_path = Path(config_path).resolve()
    environment["OSCAR_TARGET_CONFIG"] = str(config_path)
    # Direct serving historically prints every line. Retain that contract
    # even when a prior probe exported its compact terminal preference.
    environment["OSCAR_TERMINAL_LOG_MODE"] = "full"
    command = [sys.executable, "-m", "tools.target_cli", "--config", str(config_path)]
    if native:
        command.append("--native")
    result = run_phase("serve", command, cwd=ROOT, log_dir=Path(log_dir),
                       timeout=None, env=environment,
                       grace=config["shutdown_timeout_seconds"], heartbeat=0)
    if result.returncode:
        diagnostic = startup_diagnostic(Path(result.log), config)
        if diagnostic is not None:
            diagnostic["owned_group_cleanup_complete"] = result.cleanup_complete
            try:
                atomic_json(Path(log_dir) / "serve-diagnostic.json", diagnostic)
            except OSError as error:
                terminal_line(f"[oscar] diagnostic write failed: {error}; original rc={result.returncode}", stderr=True)
            terminal_line(
                "[oscar] STARTUP_MEMORY_INSUFFICIENT " + diagnostic["native_message"] +
                f"; selected physical devices={config['devices']}; "
                "failure is in native _init_device before model loading; "
                "memory owner is not established. "
                f"This launch's owned-group cleanup_complete={result.cleanup_complete}; "
                "NPU memory release was not measured.", stderr=True)
            terminal_line("[oscar] Read-only checks: npu-smi info", stderr=True)
            terminal_line("[oscar] Read-only checks: ps -eo pid,ppid,pgid,etime,args", stderr=True)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/target.json")
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--log-dir", type=Path)
    args = parser.parse_args(argv)
    from .serving_variants import variant_config
    config = variant_config(json.loads(args.config.read_text()), "native" if args.native else None)
    if args.print_command:
        print("vllm " + shlex.join(serve_argv(config, native=args.native)))
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    log_dir = (args.log_dir or ROOT / "logs" / f"serve-{stamp}").resolve()
    result = run_service(config, args.config, log_dir=log_dir, native=args.native)
    atomic_json(log_dir / "status.json", {
        "status": "interrupted" if result.interrupted else "failed" if result.returncode else "stopped",
        "service": asdict(result), "probes": "none", "npu_resource_release": "not_measured",
    })
    return shell_returncode(result.returncode)


if __name__ == "__main__":
    sys.exit(main())
