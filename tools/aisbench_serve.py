# 档案 #94/#95/#125/#140-147：独立实验入口复用签名构建、真NPU精度门和有主服务；
# 失败实时可见且落盘，不把跳过配对性能门称为验收通过。
"""Install and serve OSCAR for a user-owned AISBench run."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import traceback

from .deploy import _temporary_process_context
from .phase import atomic_json, live_log, run_phase, terminal_line
from .target_cli import target_env

ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE = ROOT / "configs/acceptance.json"
_DIAGNOSTIC_ENV = (
    "OSCAR_DEBUG_SYNC", "OSCAR_TIMING", "OSCAR_TIMING_STDERR", "OSCAR_PROFILER",
    "OSCAR_PROFILE_DIR", "OSCAR_DEVICE_TIMING_CONTROL",
)


def _config(path: Path) -> tuple[dict, dict[str, str]]:
    config = json.loads(path.read_text())
    env = target_env(config)
    if config.get("diagnostic_device_timing") is True or config.get("profiler_config") is not None:
        raise ValueError("AISBench service requires target.json without diagnostic timing/profiler settings")
    for key in _DIAGNOSTIC_ENV:
        env.pop(key, None)
    env.update(OSCAR_TARGET_CONFIG=str(path), OSCAR_RUN_NPU_TESTS="1",
               OSCAR_TERMINAL_LOG_MODE="compact", PYTHONUNBUFFERED="1")
    return config, env


def phase_plan(config_path: Path, log_dir: Path) -> list[tuple[str, list[str]]]:
    """No native server, paired K4, long-request, or CV profile invocation."""
    python = sys.executable
    return [
        ("build-dependencies", [python, "-m", "pip", "install", "setuptools>=69", "wheel",
                                "pybind11>=3", "cmake>=3.26", "ninja", "pytest"]),
        ("install-plugin", [python, "-m", "pip", "install", "--no-deps", "--no-build-isolation",
                            "-e", str(ROOT)]),
        ("probe-ops", [python, "-m", "tools.probe_ops", "--output",
                       str(log_dir / "operators.json")]),
        ("prepare-rotations", [python, "-m", "tools.prepare_rotations", "--config",
                               str(config_path)]),
    ]


def _run_phase(name: str, command: list[str], config: dict, env: dict[str, str],
               log_dir: Path, status: dict) -> None:
    result = run_phase(name, command, cwd=ROOT, log_dir=log_dir,
                       timeout=float(config["phase_timeout_seconds"]), env=env,
                       grace=float(config["shutdown_timeout_seconds"]), heartbeat=60)
    status["phases"].append(asdict(result))
    atomic_json(log_dir / "status.json", status)
    if result.returncode != 0 or result.timed_out or not result.cleanup_complete:
        raise PhaseFailure(name, result.returncode or (124 if result.timed_out else 125),
                           f"phase did not complete cleanly; log={result.log}")


class PhaseFailure(RuntimeError):
    def __init__(self, phase: str, returncode: int, message: str):
        super().__init__(message)
        self.phase = phase
        self.returncode = returncode or 1


def _read_report(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise RuntimeError(f"missing or malformed phase report {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"phase report must be an object: {path}")
    return value


def _accuracy_gates(config_path: Path, config: dict, log_dir: Path, status: dict) -> None:
    # Import only after pip install. The helper checks artifact hashes and
    # exact source/config/oracle identity before reusing prior true NPU proof.
    from .paired_concurrency_probe import (ensure_current_operators,
                                           ensure_native_current_attention)

    acceptance = _read_report(ACCEPTANCE)
    if acceptance.get("frozen_before_measurement") is not True:
        raise PhaseFailure("acceptance-policy", 1, "operator accuracy policy is not frozen")
    gate_dir = log_dir / "accuracy"
    gate_dir.mkdir(parents=True, exist_ok=True)
    gate = ensure_current_operators(config_path, config, acceptance, gate_dir)
    if (gate.get("status") != "passed" or gate.get("build") not in {"rebuilt", "reused"}
            or gate.get("accuracy") not in {"fresh_device_completion", "reused_prior_evidence"}
            or gate.get("resource_release") not in {"passed", "reused_prior_evidence"}):
        raise PhaseFailure("operator-accuracy", 1, "signed build or true NPU accuracy evidence is incomplete")
    status["operator_gate"] = {"status": "passed", "build": gate["build"],
                               "accuracy": gate["accuracy"],
                               "resource_release": gate["resource_release"],
                               "report": str(gate_dir / "operator-gate.json")}
    atomic_json(log_dir / "status.json", status)
    current = ensure_native_current_attention(config_path, config, ACCEPTANCE, gate_dir)
    if current.get("status") != "passed" or current.get("resource_release") != "passed":
        raise PhaseFailure("native-current-accuracy", 1,
                           "current-source output/LSE/merge NPU accuracy evidence is incomplete")
    status["native_current_gate"] = {"status": "passed", "resource_release": "passed",
                                     "report": current.get("report")}
    atomic_json(log_dir / "status.json", status)


def _ready_url(config: dict) -> str:
    from .service_probe import _bind_host
    host = _bind_host(str(config["host"]))
    return f"http://{'[' + host + ']' if ':' in host else host}:{int(config['port'])}/v1/chat/completions"


def _foreground_line(line: str) -> None:
    # redirect_stdout sends service detail into the durable compact log. The
    # watcher still needs to show READY directly on the user's terminal.
    data = memoryview((line + "\n").encode(errors="replace"))
    while data:
        count = os.write(1, data)
        if count <= 0:
            raise OSError("foreground terminal write returned zero bytes")
        data = data[count:]


def _watch_ready(stop: threading.Event, *, log_dir: Path, config: dict, status: dict,
                 ready: threading.Event) -> None:
    """Mirror one readiness line after the short functional gate and observer."""
    report_path = log_dir / "serve.json"
    started = time.monotonic()
    next_heartbeat = started + 60
    while not stop.wait(0.25):
        try:
            report = _read_report(report_path)
        except RuntimeError:
            report = {}
        observer = report.get("external_observation")
        server = report.get("server")
        if (report.get("status") == "serving" and isinstance(observer, dict)
                and observer.get("status") == "recording" and isinstance(server, dict)
                and server.get("status") == "healthy"):
            ready_info = {"url": _ready_url(config), "model": config["served_model_name"],
                          "bind": f"{config['host']}:{config['port']}",
                          "serve_report": str(report_path),
                          "server_log": str(log_dir / "serve" / "server.log"),
                          "observer_report": str(log_dir / "serve" / "external-load.json"),
                          "performance_acceptance": "not_run",
                          "mode": "experimental_aisbench"}
            atomic_json(log_dir / "ready.json", ready_info)
            status.update(status="serving", ready=ready_info)
            atomic_json(log_dir / "status.json", status)
            _foreground_line("[oscar] AISBENCH_READY url={url} model={model} bind={bind} "
                             "mode=experimental_aisbench performance_acceptance=not_run "
                             "log={server_log}".format(**ready_info))
            ready.set()
            return
        if time.monotonic() >= next_heartbeat:
            _foreground_line(f"[oscar] AISBENCH_STARTUP_WAIT elapsed={int(time.monotonic()-started)}s "
                             f"log={log_dir / 'serve' / 'server.log'}")
            next_heartbeat = time.monotonic() + 60


def _serve(config_path: Path, log_dir: Path) -> dict:
    from .service_probe import run_service
    return run_service(config_path, output=log_dir / "serve.json", log_dir=log_dir / "serve",
                       serve=True)


def run(config_path: Path, log_dir: Path) -> int:
    config_path, log_dir = config_path.resolve(), log_dir.resolve()
    status = {"status": "preparing", "mode": "experimental_aisbench",
              "performance_acceptance": "not_run", "paired_performance": "not_run",
              "full_service_acceptance": "not_run", "phases": []}
    atomic_json(log_dir / "status.json", status)
    (log_dir / "serve.json").unlink(missing_ok=True)
    (log_dir / "ready.json").unlink(missing_ok=True)
    current_phase = "config"
    try:
        config, env = _config(config_path)
        timeout = float(config["phase_timeout_seconds"])
        grace = float(config["shutdown_timeout_seconds"])
        if not (math.isfinite(timeout) and timeout > 0 and math.isfinite(grace) and grace > 0):
            raise ValueError("phase and shutdown timeouts must be finite positive seconds")
        terminal_line(f"[oscar] AISBENCH_EXPERIMENT devices={','.join(map(str,config['devices']))} "
                      f"port={config['port']} logs={log_dir} performance_acceptance=not_run")
        with _temporary_process_context(env, sys.argv):
            # deploy's context updates the parent environment; explicitly
            # remove inherited debug switches for helpers that copy os.environ
            # themselves. The context restores the original values on exit.
            for key in _DIAGNOSTIC_ENV:
                os.environ.pop(key, None)
            stages = phase_plan(config_path, log_dir)
            for name, command in stages[:2]:
                current_phase = name
                _run_phase(name, command, config, env, log_dir, status)
            current_phase = "operator-accuracy"
            _accuracy_gates(config_path, config, log_dir, status)
            for name, command in stages[2:]:
                current_phase = name
                if name == "probe-ops":
                    (log_dir / "operators.json").unlink(missing_ok=True)
                _run_phase(name, command, config, env, log_dir, status)
                if name == "probe-ops":
                    report = _read_report(log_dir / "operators.json")
                    cases = report.get("cases")
                    if (report.get("status") != "primitive_probe_passed" or
                            not isinstance(cases, list) or not cases or
                            any(case.get("device_completion") != "passed" for case in cases)):
                        raise PhaseFailure(name, 1, "real NPU store/merge evidence incomplete")
            status["status"] = "accuracy_passed_starting_service"
            status["accuracy_acceptance"] = "passed_operator_gates"
            atomic_json(log_dir / "status.json", status)
            current_phase = "serve"
            stop, ready = threading.Event(), threading.Event()
            watcher = threading.Thread(target=_watch_ready, kwargs={"stop": stop, "log_dir": log_dir,
                "config": config, "status": status, "ready": ready}, daemon=True,
                name="oscar-aisbench-ready")
            watcher.start()
            try:
                with live_log(log_dir / "serve-console.log", mode="compact") as stream:
                    with redirect_stdout(stream), redirect_stderr(stream):
                        service = _serve(config_path, log_dir)
            finally:
                stop.set()
                watcher.join(timeout=5)
            status["service_report"] = str(log_dir / "serve.json")
            status["resource_release"] = service.get("resource_release", "not_run")
            status["server_cleanup_complete"] = service.get("server", {}).get("cleanup_complete")
            if (service.get("status") != "stopped" or not ready.is_set() or
                    status["resource_release"] != "passed" or
                    status["server_cleanup_complete"] is not True):
                reason = service.get("error", "service returned without a clean stop and NPU release")
                raise PhaseFailure("serve", 1, reason)
            status["status"] = "stopped"
            atomic_json(log_dir / "status.json", status)
            terminal_line(f"[oscar] AISBENCH_STOPPED cleanup=passed report={log_dir / 'serve.json'}")
            return 0
    except BaseException as error:
        phase = getattr(error, "phase", current_phase)
        rc = getattr(error, "returncode", 130 if isinstance(error, KeyboardInterrupt) else 1)
        if type(rc) is not int or rc == 0:
            rc = 1
        status.update(status="failed", failed_phase=phase,
                      error=f"{type(error).__name__}: {error}", returncode=rc)
        atomic_json(log_dir / "status.json", status)
        gate_log = log_dir / f"{phase}-gate.log"
        with gate_log.open("w") as stream:
            stream.write(status["error"] + "\n")
            if not isinstance(error, (PhaseFailure, KeyboardInterrupt)):
                traceback.print_exception(error, file=stream)
                # #94/#95/#125: a new failure must show its traceback now,
                # alongside the durable gate and phase logs.
                traceback.print_exception(error, file=sys.stderr)
        terminal_line(f"[oscar] AISBENCH_FAILED phase={phase} rc={rc}: {error}; "
                      f"status={log_dir / 'status.json'}", stderr=True)
        return rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/target.json")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--plan", action="store_true", help="print phase names without installation or NPU access")
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    log_dir = (args.log_dir or ROOT / "logs" / f"aisbench-{stamp}").resolve()
    if args.plan:
        config, _ = _config(args.config.resolve())
        print(json.dumps({"mode": "experimental_aisbench",
            "phases": ["build-dependencies", "install-plugin", "signed-build-and-real-npu-accuracy",
                       "native-current-partial-accuracy", "probe-ops", "prepare-rotations",
                       "short-functional-service-and-passive-observer"],
            "port": config["port"], "performance_acceptance": "not_run"}, indent=2))
        return 0
    return run(args.config, log_dir)


if __name__ == "__main__":
    raise SystemExit(main())
