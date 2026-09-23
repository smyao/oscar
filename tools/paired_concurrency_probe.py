# 档案 #70-#73/#94/#95/#125/#129-#139：同配置配对测量、逐次释放和失败证据。
"""Run one synthetic 20/23/27/30K K4 native→OSCAR diagnostic.

This is a single HTTP diagnostic batch, not the frozen multi-case performance
acceptance matrix. No private user dataset or historical baseline is used.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from benchmarks.compare import canonical_sha256, read_json
from benchmarks.mixed import compare_mixed
from .npu_resources import (DEFAULT_RELEASE_TOLERANCE, read_npu_resources,
                            wait_for_release)
from .phase import atomic_json, cleanup_group, live_log, terminal_line
from .service_probe import SYNTHETIC_MIXED_LENGTHS
from .target_cli import ROOT, target_env


def _normalize_synthetic(report: dict, variant: str) -> dict:
    """Accept the stand-alone diagnostic or full-service performance evidence."""
    if report.get("mode") == "synthetic_mixed":
        if report.get("variant") != variant:
            raise ValueError(f"{variant} report has variant={report.get('variant')!r}")
        return report
    performance = report.get("performance")
    if not isinstance(performance, dict):
        raise ValueError(f"{variant} report has no synthetic_mixed or performance evidence")
    if report.get("server", {}).get("variant") != variant:
        raise ValueError(f"{variant} full-service report has wrong server variant")
    return {"mode": "synthetic_mixed", "variant": variant,
            "status": "measured" if report.get("status") == "passed" and
            performance.get("status") == "measured" else "failed",
            "synthetic_mixed": performance}


def compare_synthetic_reports(native: dict, oscar: dict, acceptance: dict) -> dict:
    """Apply frozen ratios to exact paired prompts; never claim full acceptance.

    Archive #73/#137-#139: a prior native log is not a matching baseline, and
    one K4 batch cannot satisfy the frozen warmup/repeat and coverage matrix.
    """
    left = _normalize_synthetic(native, "native")
    right = _normalize_synthetic(oscar, "oscar")
    warmup_pairing = ("fresh_service_before_batch" if native.get("mode") == "synthetic_mixed"
                      and oscar.get("mode") == "synthetic_mixed" else "unpaired")
    diagnostic = compare_mixed(left, right)
    policy = acceptance.get("performance")
    if not isinstance(policy, dict) or acceptance.get("frozen_before_measurement") is not True:
        raise ValueError("frozen performance acceptance policy is required")
    max_latency = policy.get("max_latency_ratio")
    min_throughput = policy.get("min_throughput_ratio")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
           for v in (max_latency, min_throughput)):
        raise ValueError("performance latency/throughput ratios must be finite and positive")
    issues = list(diagnostic["issues"])
    unresolved = False
    if diagnostic["status"] == "diagnostic_measured":
        native_section, oscar_section = left["synthetic_mixed"], right["synthetic_mixed"]
        for name, section in (("native", native_section), ("oscar", oscar_section)):
            identity = section.get("pair_identity")
            if not isinstance(identity, dict) or canonical_sha256(identity) != section.get("pair_sha256"):
                issues.append(f"{name} pair identity is missing or does not hash to pair_sha256")
        if native_section.get("pair_identity") != oscar_section.get("pair_identity"):
            issues.append("native and OSCAR pair identity fields differ")
        for name, side in (("native", left), ("oscar", right)):
            if side["synthetic_mixed"].get("prompt_lengths") != list(SYNTHETIC_MIXED_LENGTHS):
                issues.append(f"{name} prompt lengths differ from fixed 20/23/27/30K")
        if len(diagnostic["batches"]) != 1:
            issues.append("the diagnostic requires exactly one batch per variant")
        for batch in diagnostic["batches"]:
            if len(batch["requests"]) != len(SYNTHETIC_MIXED_LENGTHS):
                issues.append("paired batch lacks one of the four requests")
            for metric in ("prompt_throughput_ratio", "generation_throughput_ratio"):
                ratio = batch.get(metric)
                if type(ratio) not in (int, float) or not math.isfinite(ratio):
                    issues.append(f"{metric} ratio missing or nonfinite")
                elif ratio < min_throughput:
                    issues.append(f"{metric}={ratio:.4f} < {min_throughput:.4f}")
            for request in batch["requests"]:
                request_id = request["request_id"]
                source = {row["request_id"]: row for row in native_section["sample"]["requests"]}
                candidate = {row["request_id"]: row for row in oscar_section["sample"]["requests"]}
                n_salt = source[request_id].get("cache_salt_sha256")
                o_salt = candidate[request_id].get("cache_salt_sha256")
                if not isinstance(n_salt, str) or len(n_salt) != 64 or n_salt != o_salt:
                    issues.append(f"{request_id} cache salt identity missing or differs")
                for metric in ("ttft_ms", "tpot_ms", "e2e_ms"):
                    ratio = request["oscar_over_native"].get(metric)
                    if type(ratio) not in (int, float) or not math.isfinite(ratio):
                        if metric == "tpot_ms" and (source[request_id].get(metric) == 0
                                                    or candidate[request_id].get(metric) == 0):
                            issues.append(f"{request_id} tpot_unresolved_SSE_burst")
                            unresolved = True
                        else:
                            issues.append(f"{request_id} {metric} ratio missing or nonfinite")
                    elif ratio > max_latency:
                        issues.append(f"{request['request_id']} {metric}={ratio:.4f} > {max_latency:.4f}")
    status = ("passed" if not issues and diagnostic["status"] == "diagnostic_measured"
              else "needs_evidence" if unresolved and all("tpot_unresolved_SSE_burst" in item
                                                           for item in issues) else "failed")
    return {"status": status,
            "scope": "single-batch client-side synthetic K4 directional ratio screen",
            "warmup_pairing": warmup_pairing,
            "warmup_note": ("OSCAR synthetic follows functional probes while native synthetic follows startup"
                            if warmup_pairing == "unpaired" else
                            "both synthetic batches follow fresh service startup"),
            "performance_acceptance": "not_run",
            "max_latency_ratio": max_latency, "min_throughput_ratio": min_throughput,
            "issues": issues, "pair_sha256": diagnostic.get("pair_sha256"),
            "batches": diagnostic.get("batches", []),
            "diagnostic_status": diagnostic["status"]}


def _observe_resources(config: dict, directory: Path, *, before=None):
    """Retain observer logs without printing their device import chatter."""
    directory.mkdir(parents=True, exist_ok=True)
    with live_log(directory / "observer-console.log", mode="compact") as stream:
        with redirect_stdout(stream), redirect_stderr(stream):
            if before is None:
                return read_npu_resources(config, log_dir=directory,
                    timeout=float(config.get("resource_observation_timeout_seconds", 30)))
            return wait_for_release(config, before, log_dir=directory,
                timeout=float(config.get("resource_release_timeout_seconds", 30)),
                tolerance_bytes=config.get("resource_release_tolerance_bytes",
                                           DEFAULT_RELEASE_TOLERANCE))


def _run_variant(variant: str, config_path: Path, config: dict, directory: Path) -> dict:
    """Capture full service output and always reclaim the runner's process group."""
    directory.mkdir(parents=True, exist_ok=True)
    report_path, console_path = directory / "report.json", directory / "console.log"
    command = [sys.executable, "-m", "tools.service_probe", "--synthetic",
               "--config", str(config_path), "--log-dir", str(directory),
               "--output", str(report_path)]
    if variant == "native":
        command.append("--native")
    env = target_env(config)
    env["OSCAR_TERMINAL_LOG_MODE"] = "compact"
    duration = float(config.get("synthetic_timeout_seconds", 1800)) + float(
        config.get("shutdown_timeout_seconds", 30)) + 90
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("paired probe timeout must be finite and positive")
    started = time.monotonic()
    # Archive #125: full child log is durable, but traceback/error lines are
    # forwarded while the request is still running rather than after timeout.
    with live_log(console_path, mode="compact") as stream:
        stream.write("START " + json.dumps({"variant": variant, "command": command}) + "\n")
        stream.flush()
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        timed_out, returncode = False, 1
        try:
            try:
                returncode = process.wait(timeout=duration)
            except subprocess.TimeoutExpired:
                timed_out, returncode = True, 124
                stream.write(f"TIMEOUT variant={variant} limit={duration}s\n")
                stream.flush()
        finally:
            cleaned = cleanup_group(process, float(config.get("shutdown_timeout_seconds", 30)))
        if not cleaned and returncode == 0:
            returncode = 125
        stream.write("RESULT " + json.dumps({"returncode": returncode,
            "timed_out": timed_out, "runner_cleanup_complete": cleaned}) + "\n")
    try:
        probe = read_json(report_path)
    except (OSError, ValueError) as error:
        probe = {"status": "failed", "error": f"report missing or invalid: {error}"}
    lifecycle = probe.get("server") if isinstance(probe.get("server"), dict) else {}
    status = ("passed" if returncode == 0 and probe.get("status") == "measured"
              and lifecycle.get("cleanup_complete") is True and cleaned else "failed")
    return {"status": status, "variant": variant, "returncode": returncode,
            "timed_out": timed_out, "runner_cleanup_complete": cleaned,
            "owned_server_cleanup_complete": lifecycle.get("cleanup_complete") is True,
            "probe_status": probe.get("status"), "probe_error": probe.get("error"),
            "report": str(report_path), "log": str(console_path),
            "elapsed_seconds": round(time.monotonic() - started, 6)}


def run_paired(config_path: Path, *, output: Path, log_dir: Path,
               acceptance_path: Path, native_only: bool = False) -> dict:
    """Run native→OSCAR sequentially, with release proof between owners."""
    config_path, output = config_path.resolve(), output.resolve()
    acceptance_path, log_dir = acceptance_path.resolve(), log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "mode": "native_synthetic_mixed" if native_only else "paired_synthetic_mixed",
              "config": str(config_path), "acceptance": str(acceptance_path),
              "native": "not_run", "oscar": "not_run", "comparison": "not_run",
              "performance_acceptance": "not_run", "exit_code": 1,
              "warmup_scope": "fresh_service_before_batch"}
    atomic_json(output, report)
    try:
        config, acceptance = read_json(config_path), read_json(acceptance_path)
        target_env(config)  # Archive #123: reject invalid selection before NPU touch.
        report["config_sha256"] = canonical_sha256(config)
        report["acceptance_sha256"] = canonical_sha256(acceptance)
        baseline = _observe_resources(config, log_dir / "resources-before-native")
        atomic_json(log_dir / "resources-before-native.json", baseline)
        for variant in (("native",) if native_only else ("native", "oscar")):
            terminal_line(f"[oscar] PERF_VARIANT start={variant} lengths=20K,23K,27K,30K K=4")
            result = _run_variant(variant, config_path, config, log_dir / variant)
            report[variant] = result
            atomic_json(output, report)
            release = _observe_resources(config, log_dir / f"resources-after-{variant}",
                                         before=baseline)
            result["resource_release"] = release["status"]
            result["release_evidence"] = str(log_dir / f"resources-after-{variant}.json")
            atomic_json(log_dir / f"resources-after-{variant}.json", release)
            atomic_json(output, report)
            if result["status"] != "passed" or release["status"] != "passed":
                reason = result.get("probe_error") or release.get("reason") or "request, cleanup or release gate failed"
                report.update(status="failed", error=f"{variant}: {reason}",
                              exit_code=result["returncode"] if result["returncode"] != 0 else 1)
                return report
            terminal_line(f"[oscar] PERF_VARIANT done={variant} result=4/4 release=passed")
        if native_only:
            report.update(status="passed", exit_code=0)
            return report
        native, oscar = read_json(report["native"]["report"]), read_json(report["oscar"]["report"])
        comparison = compare_synthetic_reports(native, oscar, acceptance)
        report["comparison"] = comparison
        atomic_json(log_dir / "comparison.json", comparison)
        from .performance_summary import format_performance_summary
        for line in format_performance_summary(native, oscar, comparison,
                                               native_path=report["native"]["report"],
                                               oscar_path=report["oscar"]["report"],
                                               comparison_path=log_dir / "comparison.json"):
            terminal_line(line)
        report["status"] = "passed" if comparison["status"] == "passed" else "failed"
        report["exit_code"] = 0 if report["status"] == "passed" else 2
        if comparison["issues"]:
            report["error"] = "; ".join(comparison["issues"][:6])
    except KeyboardInterrupt:
        report.update(status="interrupted", error="interrupted", exit_code=130)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", exit_code=1)
    finally:
        atomic_json(output, report)
        if report["status"] != "passed":
            terminal_line(f"[oscar] PERF_ERROR {report.get('error', 'probe failed')} report={output}", stderr=True)
        terminal_line(f"[oscar] PERF_RESULT status={report['status']} rc={report['exit_code']} report={output}")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/target.json")
    parser.add_argument("--acceptance", type=Path, default=ROOT / "configs/acceptance.json")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--native-only", action="store_true",
                        help="one-click pipeline: collect native side; compare with its existing OSCAR service probe")
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    log_dir = args.log_dir or ROOT / "logs" / f"paired-concurrency-{stamp}"
    output = args.output or log_dir / "paired-report.json"
    return run_paired(args.config, output=output, log_dir=log_dir,
                      acceptance_path=args.acceptance, native_only=args.native_only)["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
