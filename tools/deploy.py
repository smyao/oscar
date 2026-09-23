# 档案 #27/#51/#52/#70–73/#94–95/#125/#131–139：单入口、配对性能门、真实退出码和精简终端证据。
"""Install, build, probe and serve using the target's existing Python environment."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from .phase import atomic_json, run_phase, terminal_line
from .target_cli import target_env

ROOT = Path(__file__).resolve().parents[1]


def plan(config_path: Path, log_dir: Path) -> list[tuple[str, list[str]]]:
    python = sys.executable
    cfg = str(config_path.resolve())
    config = json.loads(config_path.read_text())
    return [
        ("build-dependencies", [python,"-m","pip","install","setuptools>=69","wheel","pybind11>=3","cmake>=3.26","ninja","pytest"]),
        ("install-plugin", [python, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", str(ROOT)]),
        ("build-ops", [python, "-m", "tools.build_ops", "--soc", config.get("soc_version", "ascend910b4"),
                       "--log-dir", str(log_dir / "build")]),
        ("probe-ops", [python, "-m", "tools.probe_ops", "--output", str(log_dir / "operators.json")]),
        ("probe-cv", [python,"-m","pytest","-q","--maxfail=1",str(ROOT/"tests/test_cv_contracts.py"),str(ROOT/"tests/test_rotation_npu.py"),
                      "--junitxml="+str(log_dir/"cv-npu.xml")]),
        ("prepare-rotations", [python, "-m", "tools.prepare_rotations", "--config", cfg]),
        # A separate native baseline must finish and release its owned NPU
        # resources before the OSCAR full-service probe starts (#131/#136).
        ("native-synthetic", [python, "-m", "tools.paired_concurrency_probe", "--native-only",
                              "--config", cfg, "--acceptance", str(ROOT / "configs/acceptance.json"),
                              "--log-dir", str(log_dir / "native-synthetic"),
                              "--output", str(log_dir / "native-synthetic-report.json")]),
        ("service-probe", [python,"-m","tools.service_probe","--config",cfg,
                           "--log-dir",str(log_dir/"service-probe"),"--output",str(log_dir/"service-probe-report.json")]),
    ]


def diagnostic_plan(log_dir: Path) -> list[tuple[str, list[str]]]:
    """Explicit developer diagnostics; never part of install/probe/serve."""
    return [
        ("environment", [sys.executable, "-m", "tools.environment", "--output", str(log_dir / "environment.json"),
                         "--native-root", "/vllm-workspace/vllm", "--native-root", "/vllm-workspace/vllm-ascend"]),
        ("runtime-readiness", [sys.executable, "-m", "tools.readiness", "--source-only", "--output", str(log_dir / "source-readiness.json")]),
    ]


def _report(path: Path) -> dict:
    """Reject missing, overwritten or malformed evidence instead of serving."""
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise RuntimeError(f"missing or invalid probe evidence {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"probe evidence must be a JSON object: {path}")
    return value


def _compare_performance(native: dict, oscar: dict) -> dict:
    from .paired_concurrency_probe import compare_synthetic_reports

    acceptance = _report(ROOT / "configs/acceptance.json")
    if acceptance.get("frozen_before_measurement") is not True:
        raise RuntimeError("performance acceptance policy is not frozen")
    return compare_synthetic_reports(native, oscar, acceptance)


def _performance_lines(native: dict, oscar: dict, comparison: dict, *,
                       native_path: Path, oscar_path: Path, comparison_path: Path) -> list[str]:
    from .performance_summary import format_performance_summary

    return format_performance_summary(native, oscar, comparison,
                                      native_path=native_path, oscar_path=oscar_path,
                                      comparison_path=comparison_path)


def _gate_failed(status: dict, log_dir: Path, phase: str, error: str) -> int:
    # Archive #94/#95/#125: the brief failure must be visible immediately and
    # the same diagnosis must survive in a dedicated file and status ledger.
    (log_dir / f"{phase}-gate.log").write_text(f"FAILED phase={phase}: {error}\n")
    terminal_line(f"[oscar] FAILED phase={phase}: {error}", stderr=True)
    status.update(status="failed", failed_phase=phase, error=error)
    atomic_json(log_dir / "status.json", status)
    return 1


@contextmanager
def _temporary_process_context(env: dict[str, str], argv: list[str]):
    """Give the foreground server target settings only for its own lifetime."""
    prior_env, prior_argv = os.environ.copy(), sys.argv
    try:
        os.environ.update(env)
        sys.argv = argv
        yield
    finally:
        os.environ.clear()
        os.environ.update(prior_env)
        sys.argv = prior_argv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/target.json")
    parser.add_argument("--plan", action="store_true", help="print commands without installing or touching an NPU")
    parser.add_argument("--only", choices=["environment", "build-ops", "probe-ops", "probe-cv", "native-synthetic", "service-probe", "prepare-rotations", "runtime-readiness"])
    parser.add_argument("--log-dir", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    log_dir = (args.log_dir or ROOT / "logs" / stamp).resolve()
    status = {"status": "running", "npu_acceptance": "not_run", "phases": []}
    if not args.plan:
        atomic_json(log_dir / "status.json", status)
        if not args.only:
            (log_dir / "paired-performance-report.json").unlink(missing_ok=True)
        print(f"[oscar] install → build → probes → serve; logs={log_dir}", flush=True)
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
        stages = [x for x in stages + diagnostic_plan(log_dir) if x[0] == args.only]
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
    if not args.only:
        env["OSCAR_TERMINAL_LOG_MODE"] = "compact"
    print(f"[oscar] devices={env.get('ASCEND_RT_VISIBLE_DEVICES', 'diagnostic')} port={config['port']}", flush=True)
    validated = set()
    try:
        for name, command in stages:
            # A repeated --log-dir must not let stale success reports satisfy
            # a new phase if its child forgets to write evidence (#136).
            if not args.only and name in {"native-synthetic", "service-probe"}:
                (log_dir / f"{name}-report.json").unlink(missing_ok=True)
                if name == "native-synthetic":
                    (log_dir / "native-synthetic" / "native" / "report.json").unlink(missing_ok=True)
            result = run_phase(name, command, cwd=ROOT, log_dir=log_dir,
                               timeout=config["phase_timeout_seconds"], env=env,
                               grace=config["shutdown_timeout_seconds"], heartbeat=60)
            status["phases"].append(asdict(result))
            atomic_json(log_dir / "status.json", status)
            if result.returncode:
                status.update(status="failed", failed_phase=name)
                rc = result.returncode
                break
            if args.only:
                continue
            if name == "native-synthetic":
                native_wrapper_path = log_dir / "native-synthetic-report.json"
                native_path = log_dir / "native-synthetic" / "native" / "report.json"
                try:
                    native_wrapper = _report(native_wrapper_path)
                    variant = native_wrapper.get("native")
                    if (native_wrapper.get("status") != "passed" or not isinstance(variant, dict)
                            or variant.get("status") != "passed" or variant.get("returncode") != 0
                            or variant.get("resource_release") != "passed"
                            or variant.get("owned_server_cleanup_complete") is not True
                            or variant.get("runner_cleanup_complete") is not True):
                        raise RuntimeError(f"native synthetic requests or owned NPU release incomplete; report={native_wrapper_path}")
                    native = _report(native_path)
                    if (native.get("mode") != "synthetic_mixed" or native.get("variant") != "native"
                            or native.get("status") != "measured"):
                        raise RuntimeError(f"native synthetic request evidence incomplete; report={native_path}")
                    validated.add("native-synthetic")
                except RuntimeError as error:
                    rc = _gate_failed(status, log_dir, "native-synthetic", str(error))
                    break
            if name == "service-probe":
                oscar_path = log_dir / "service-probe-report.json"
                try:
                    if "native-synthetic" not in validated:
                        raise RuntimeError("fresh native synthetic baseline was not validated before OSCAR service probe")
                    oscar = _report(oscar_path)
                    if oscar.get("status") != "passed" or oscar.get("resource_release") != "passed":
                        raise RuntimeError("full-service evidence or owned NPU resource release is missing "
                            f"(status={oscar.get('status')!r} resource_release={oscar.get('resource_release')!r} report={oscar_path})")
                    if not isinstance(oscar.get("performance"), dict) or oscar["performance"].get("status") != "measured":
                        raise RuntimeError(f"OSCAR synthetic performance evidence is incomplete; report={oscar_path}")
                except RuntimeError as error:
                    rc = _gate_failed(status, log_dir, "full-service-probe", str(error))
                    break
                comparison_path = log_dir / "paired-performance-report.json"
                try:
                    comparison = _compare_performance(native, oscar)
                    if not isinstance(comparison, dict):
                        raise RuntimeError("paired comparator returned no JSON object")
                    comparison["native_release_report"] = str(native_wrapper_path)
                    atomic_json(comparison_path, comparison)
                    for line in _performance_lines(native, oscar, comparison,
                                                   native_path=native_path, oscar_path=oscar_path,
                                                   comparison_path=comparison_path):
                        terminal_line(line)
                    if comparison.get("status") != "passed":
                        issues = comparison.get("issues")
                        first = issues[0] if isinstance(issues, list) and issues else "comparison incomplete"
                        reason = ("paired 20–30K/K4 speed evidence inconclusive" if comparison.get("status") == "needs_evidence"
                                  else "paired 20–30K/K4 speed gate failed")
                        raise RuntimeError(f"{reason}: {first}; report={comparison_path}")
                    status["paired_performance"] = {"status": "passed", "report": str(comparison_path),
                                                    "warmup_pairing": comparison.get("warmup_pairing", "unpaired"),
                                                    "performance_acceptance": comparison.get("performance_acceptance", "not_run")}
                    validated.update(("service-probe", "paired-performance"))
                except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
                    rc = _gate_failed(status, log_dir, "paired-performance",
                                      f"{type(error).__name__}: {error}; report={comparison_path}")
                    break
        if rc == 0:
            if args.only:
                status["status"] = "selected_phase_passed"
            else:
                missing = {"native-synthetic", "service-probe", "paired-performance"} - validated
                if missing:
                    rc = _gate_failed(status, log_dir, "full-service-probe",
                                      f"fresh one-click evidence missing: {', '.join(sorted(missing))}; formal serve prohibited")
                else:
                    status["status"]="probes_passed"
    finally:
        atomic_json(log_dir / "status.json", status)
    if rc==0 and not args.only:
        # The managed server preserves ownership and validates a real request
        # before staying in the foreground. It never starts after failed probes.
        from .service_probe import main as service_main
        serve_argv = ["service_probe","--config",str(args.config.resolve()),"--serve",
                      "--log-dir",str(log_dir/"serve"),"--output",str(log_dir/"serve.json")]
        try:
            with _temporary_process_context(env, serve_argv):
                serve_rc = service_main()
        except BaseException as error:
            status.update(status="failed", failed_phase="serve",
                          error=f"{type(error).__name__}: {error}")
            atomic_json(log_dir / "status.json", status)
            raise
        if serve_rc != 0:
            status.update(status="failed", failed_phase="serve", service_returncode=serve_rc,
                          error=f"formal service exited with code {serve_rc}")
            atomic_json(log_dir / "status.json", status)
        return serve_rc
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"[oscar] FAILED deployment: {exc}", file=sys.stderr)
        sys.exit(1)
