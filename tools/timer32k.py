# Archive #70-73/#94-95/#125/#129/#131-140: owned bounded lifecycles,
# exact paired requests, separate control/profile clocks, durable quiet logs.
# No native HTTP start/stop profiler (#133), no global kill, no source edits.
"""Run a bounded 32K native BS4 / OSCAR BS4+BS8 attribution diagnostic."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid

from benchmarks.compare import canonical_sha256
from benchmarks.measure import metrics_delta, metrics_snapshot
from benchmarks.mixed import run_batch
from .deploy import plan as deploy_plan
from .install_serve import BUILD_PHASES
from .model_config import add_model_arguments, apply_model_options
from .npu_resources import read_npu_resources, wait_for_release
from .phase import atomic_json, cleanup_group, run_phase, terminal_line
from .prepare_rotations import model_geometry
from .service_probe import (exact_prompt, load_tokenizer, managed_server,
                            parse_gauges, ServiceProbeError)
from .serving_variants import runtime_feature_summary, variant_config
from .target_cli import ROOT, diagnose_guard_enabled, serve_argv, target_env

PROMPT_TOKENS = 32768
SHARED_BATCH = 4
CAPACITY_BATCH = 8
MAX_SUMMARY_LINES = 200
COUNTERS = ("vllm:prompt_tokens_total", "vllm:generation_tokens_total",
            "vllm:num_preemptions_total")
_ACTIVE_SERVER = None


class TimerRunError(RuntimeError):
    def __init__(self, message, returncode=1, log=None):
        super().__init__(message)
        self.returncode = returncode
        self.log = log


def positive_seconds(value):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("seconds must be finite and positive")
    return parsed


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_rows(tokenizer, count=CAPACITY_BATCH):
    """Natural text IDs, identical across variants; salt isolates each request."""
    body = exact_prompt(tokenizer, PROMPT_TOKENS)
    rows = []
    for index in range(count):
        prefix = tokenizer.encode(f"Observation record {index + 1}. ", add_special_tokens=False)
        if (not prefix or any(type(token) is not int or token < 0 for token in prefix)
                or len(prefix) >= PROMPT_TOKENS):
            raise ValueError("tokenizer produced an invalid record prefix")
        rows.append({"id": f"timer32k-{index:03d}",
                     "token_ids": prefix + body[:PROMPT_TOKENS - len(prefix)]})
    return rows


def prompt_manifest(rows):
    return [{"id": row["id"], "prompt_tokens": len(row["token_ids"]),
             "prompt_sha256": canonical_sha256(row["token_ids"])} for row in rows]


def _source_identity():
    def git(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, check=True)
        return result.stdout.strip()
    try:
        return {"git_commit": git("rev-parse", "HEAD"),
                "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.SubprocessError) as error:
        return {"git_commit": None, "dirty": None, "error": str(error)}


@contextmanager
def _environment(changes):
    prior = {key: os.environ.get(key) for key in changes}
    try:
        for key, value in changes.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _signal_target(server, grace):
    global _ACTIVE_SERVER
    prior = _ACTIVE_SERVER
    _ACTIVE_SERVER = (server, grace)
    try:
        yield
    finally:
        _ACTIVE_SERVER = prior


@contextmanager
def _heartbeat(label, interval=60):
    # Keep this captured stream even while run_phase output is redirected.
    terminal = sys.stdout
    stop = threading.Event()
    started = time.monotonic()
    def report():
        while not stop.wait(interval):
            terminal.write(f"[timer32k] {label} running_seconds={time.monotonic()-started:.0f}\n")
            terminal.flush()
    thread = threading.Thread(target=report, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1)


def _quiet_call(path, label, function, *args, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    terminal_line(f"[timer32k] START {label}")
    with _heartbeat(label), path.open("a", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            return function(*args, **kwargs)


def _failure_tail(path, limit=60):
    if path is None or not Path(path).is_file():
        return
    # Bound memory as well as terminal output for large compiler/server logs.
    with Path(path).open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 64 * 1024))
        lines = stream.read().decode("utf-8", errors="replace").splitlines()
    terminal_line(f"[timer32k] FAILURE_LOG {path}", stderr=True)
    for line in lines[-limit:]:
        terminal_line(line[:2000], stderr=True)


def _counter_delta(before, after):
    if before.get("status") != "observed" or after.get("status") != "observed":
        return {"status": "unavailable", "missing": list(COUNTERS)}
    left = parse_gauges(Path(before["path"]).read_text(), COUNTERS)
    right = parse_gauges(Path(after["path"]).read_text(), COUNTERS)
    missing = sorted(set(COUNTERS) - (set(left) & set(right)))
    deltas = {key: right[key] - left[key] for key in set(left) & set(right)}
    return {"status": "failed" if any(value < 0 for value in deltas.values())
            else "observed" if not missing else "partial", "delta": deltas, "missing": missing}


def _completion_check(sample, rows, output_tokens):
    expected = {row["id"]: row for row in rows}
    requests = sample.get("requests", [])
    errors = []
    if len(requests) != len(expected) or len({r.get("request_id") for r in requests}) != len(expected):
        errors.append("missing or duplicate request result")
    for request in requests:
        row = expected.get(request.get("request_id"))
        if row is None:
            errors.append("unexpected request ID")
            continue
        usage = request.get("usage") or {}
        if (request.get("status") != "completed" or request.get("done_received_ms") is None
                or usage.get("prompt_tokens") != len(row["token_ids"])
                or usage.get("completion_tokens") != output_tokens
                or len(request.get("generated_token_ids", [])) != output_tokens
                or request.get("prompt_sha256") != canonical_sha256(row["token_ids"])):
            errors.append(f"{row['id']}: incomplete DONE/usage/exact-token evidence")
    if sample.get("status") != "completed":
        errors.append("batch did not complete")
    return {"status": "passed" if not errors else "failed", "errors": errors}


def _disarm(control, run_id=""):
    atomic_json(control, {"enabled": False, "run_id": run_id})


def flush_disarmed_control(server, config, token_ids, *, control, run_id, timeout, report=None):
    """Make all TP runners observe disarm before process shutdown/next control.

    Multiprocess worker teardown need not execute atexit. The actual short
    request enters each runner once; its timing and counters are kept outside
    every paired batch. No profiling is armed for this request.
    """
    _disarm(control, run_id)
    record = {} if report is None else report
    record.update(scope="disarmed marker observation only; excluded from paired wall and metrics",
                  profile_enabled=False, status="running")
    rows = [{"id": "timer32k-control-flush", "token_ids": token_ids[:128]}]
    server.check_alive()
    with _heartbeat("disarmed-control-flush"):
        sample = run_batch(server.base_url, rows, model=config["served_model_name"],
            output_tokens=1, timeout=min(timeout, 120), repeat=f"{run_id}-flush", sample_interval=1)
    check = _completion_check(sample, rows, 1)
    record.update(sample=sample, completion_check=check, status=check["status"])
    if check["status"] != "passed":
        raise TimerRunError(f"{run_id}: disarmed control flush request failed", log=server.log)
    server.check_alive()
    return record


def run_one_batch(server, config, rows, *, root, variant, mode, control,
                  output_tokens, timeout, pair_hash, run_prefix, report=None):
    batch_size = len(rows)
    directory = root / variant / f"b{batch_size}-{mode}"
    directory.mkdir(parents=True, exist_ok=True)
    run_id = f"{run_prefix}-{variant}-b{batch_size}-{mode}"
    profile_dir = directory / "profile"
    profile_control = {"enabled": mode == "profile", "run_id": run_id,
        "variant": variant, "output_dir": str(profile_dir), "profile": mode == "profile",
        "target_batch": batch_size, "min_decode_requests": batch_size}
    row = {} if report is None else report
    row.update(run_id=run_id, variant=variant, concurrency=batch_size, mode=mode,
        status="running", record_path=str(directory / "batch.json"),
        profile_dir=str(profile_dir), profile_control=profile_control,
        pair_hash=pair_hash, prompt_manifest_sha256=canonical_sha256(prompt_manifest(rows)),
        client_timing_scope="SSE client receive times; profile wall is diagnostic only")
    atomic_json(directory / "batch.json", row)
    server.check_alive()
    before = metrics_snapshot(server.base_url, timeout=10, path=directory / "before.prom")
    row["metrics_before"] = before
    atomic_json(control, profile_control)
    started = time.time()
    terminal_line(f"[timer32k] BATCH variant={variant} bs={batch_size} mode={mode} prompt=32768 output={output_tokens}")
    try:
        with _heartbeat(f"{variant}/bs{batch_size}/{mode}"):
            sample = run_batch(server.base_url, rows, model=config["served_model_name"],
                output_tokens=output_tokens, timeout=timeout, repeat=run_id, sample_interval=1.0)
        row["sample"] = sample
        row["window"] = [started, time.time()]
        server.check_alive()
        after = metrics_snapshot(server.base_url, timeout=10, path=directory / "after.prom")
        row.update(metrics_after=after, mtp_counter_delta=metrics_delta(before, after),
                   counter_delta=_counter_delta(before, after))
        observations = sample.get("running_waiting_samples", [])
        row["peak_running"] = max((s["running"] for s in observations if "running" in s), default=None)
        row["peak_waiting"] = max((s["waiting"] for s in observations if "waiting" in s), default=None)
        row["request_completion_check"] = _completion_check(sample, rows, output_tokens)
        row["status"] = "completed" if row["request_completion_check"]["status"] == "passed" else "failed"
        if row["status"] != "completed":
            raise TimerRunError(f"{run_id}: incomplete request evidence", log=server.log)
        return row
    except BaseException as error:
        row.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        _disarm(control, run_id)
        atomic_json(directory / "batch.json", row)


def _build_stages(config_path, effective_path, root):
    stages = []
    for name, command in deploy_plan(config_path, root / "phases"):
        if name in BUILD_PHASES:
            if "--config" in command:
                command[command.index("--config") + 1] = str(effective_path)
            stages.append((name, command))
    if [name for name, _ in stages] != list(BUILD_PHASES):
        raise RuntimeError("deployment build/rotation phase contract changed")
    return stages


def _validate_config(config, output_tokens):
    target_env(config, base={})
    if diagnose_guard_enabled(config):
        raise ValueError("timer comparison requires diagnose_guard disabled")
    if config.get("profiler_config") or config.get("diagnostic_device_timing"):
        raise ValueError("timer comparison rejects existing profiler_config/diagnostic_device_timing")
    if config.get("enable_prefix_caching") is not False:
        raise ValueError("this paired diagnostic requires the current APC-off launch configuration")
    if (config.get("use_native_token_budget") or config.get("max_num_batched_tokens") != 16384
            or config.get("scheduler_cls")):
        raise ValueError("timer diagnostic preserves the native scheduler and explicit 16384 token budget")
    if config.get("speculative_config", {}).get("num_speculative_tokens") != 3:
        raise ValueError("timer diagnostic requires the existing 3-token MTP configuration")
    if config["max_model_len"] < PROMPT_TOKENS + output_tokens or config["max_num_seqs"] < CAPACITY_BATCH:
        raise ValueError("configured context/request capacity cannot contain the requested diagnostic")


def _run_service(config, config_path, rows, *, root, variant, control, manifest, args):
    directory = root / variant
    directory.mkdir(parents=True, exist_ok=True)
    service = {"variant": variant, "lifecycle": {}, "batches": []}
    manifest["services"].append(service)
    atomic_json(root / "run.json", manifest)
    service["resource_before"] = _quiet_call(directory / "resource-supervisor.log",
        f"{variant}/resources-before", read_npu_resources, config, log_dir=directory / "resources-before")
    try:
        with managed_server(config, config_path, log_dir=directory / "server",
                lifecycle=service["lifecycle"], native=variant == "native", quiet=True) as server:
            with _signal_target(server, config["shutdown_timeout_seconds"]):
                _disarm(control)
                warm = [{"id": "timer32k-warmup", "token_ids": rows[0]["token_ids"][:128]}]
                with _heartbeat(f"{variant}/warmup"):
                    service["warmup"] = run_batch(server.base_url, warm,
                        model=config["served_model_name"], output_tokens=16, timeout=args.request_timeout,
                        repeat=f"{manifest['run_id']}-{variant}-warmup", sample_interval=1)
                if _completion_check(service["warmup"], warm, 16)["status"] != "passed":
                    raise TimerRunError(f"{variant}: warmup did not complete", log=server.log)
                for count in ((SHARED_BATCH,) if variant == "native" else (SHARED_BATCH, CAPACITY_BATCH)):
                    for mode in ("control", "profile"):
                        batch = {}
                        service["batches"].append(batch)
                        try:
                            run_one_batch(server, config, rows[:count], root=root, variant=variant,
                                mode=mode, control=control, output_tokens=args.output_tokens,
                                timeout=args.request_timeout, pair_hash=manifest["pair_hash"],
                                run_prefix=manifest["run_id"], report=batch)
                            if mode == "profile":
                                batch["control_flush"] = {}
                                flush_disarmed_control(server, config, rows[0]["token_ids"],
                                    control=control, run_id=batch["run_id"], timeout=args.request_timeout,
                                    report=batch["control_flush"])
                        finally:
                            if batch.get("record_path"):
                                atomic_json(Path(batch["record_path"]), batch)
                            atomic_json(root / "run.json", manifest)
    except BaseException as error:
        service["error"] = f"{type(error).__name__}: {error}"
        if not isinstance(error, (TimerRunError, KeyboardInterrupt)):
            raise TimerRunError(str(error), log=directory / "server" / "server.log") from error
        raise
    finally:
        _disarm(control)
        try:
            service["resource_release"] = _quiet_call(directory / "resource-supervisor.log",
                f"{variant}/resources-release", wait_for_release, config, service["resource_before"],
                log_dir=directory / "resources-release", timeout=config["shutdown_timeout_seconds"])
        except BaseException as error:
            service["resource_release"] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        atomic_json(root / "run.json", manifest)
    if (service["lifecycle"].get("cleanup_complete") is not True
            or service["resource_release"].get("status") != "passed"):
        raise TimerRunError(f"{variant}: owned service/resource cleanup failed", log=directory / "server" / "server.log")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/target.json")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--plan", action="store_true", help="read-only plan; no installation, tokenizer, model or NPU load")
    parser.add_argument("--rear-cards", action="store_true")
    parser.add_argument("--output-tokens", type=positive_int, default=256)
    parser.add_argument("--request-timeout", type=positive_seconds, default=900)
    parser.add_argument("--startup-timeout", type=positive_seconds)
    add_model_arguments(parser)
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    root = (args.log_dir or ROOT / "logs" / f"timer32k-{stamp}").resolve()
    if not args.plan and root.exists() and any(root.iterdir()):
        parser.error("--log-dir must be empty or new; previous timer evidence is never overwritten")
    config = apply_model_options(json.loads(args.config.read_text()), args, require_model=not args.plan)
    config = variant_config(config, "candidate")
    if args.rear_cards:
        target_env(config, base={})
        config.update(devices=[4, 5, 6, 7], port=7878)
    if args.startup_timeout:
        config["service_startup_timeout_seconds"] = args.startup_timeout
    _validate_config(config, args.output_tokens)
    effective_path = root / "effective-target.json"
    stages = _build_stages(args.config, effective_path, root)
    control = root / "timer-control.json"
    if args.plan:
        print(json.dumps({"kind": "timer32k", "stages": stages,
            "vllm_argv": serve_argv(config), "runtime_selection": runtime_feature_summary(config, "candidate"),
            "devices": config["devices"], "port": config["port"],
            "timer_control": str(control), "prompt_tokens": PROMPT_TOKENS,
            "output_tokens": args.output_tokens, "warmup": {"prompt_tokens": 128, "output_tokens": 16},
            "profile_flush": {"prompt_tokens": 128, "output_tokens": 1,
                              "profile_enabled": False, "excluded_from_batch_wall_and_metrics": True},
            "arms": [{"variant": variant, "concurrency": count, "modes": ["control", "profile"]}
                     for variant, count in (("native", SHARED_BATCH), ("candidate", SHARED_BATCH), ("candidate", CAPACITY_BATCH))],
            "request_timeout_seconds": args.request_timeout,
            "policy": "same config and exact prompts; control wall only for comparison; no HTTP profiler"}, indent=2))
        return 0
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(effective_path, config)
    _disarm(control)
    manifest = {"schema_version": 1, "kind": "timer32k", "status": "running", "root": str(root),
        "created_at": stamp, "run_id": uuid.uuid4().hex, "config_path": str(effective_path),
        "effective_config": config, "source": _source_identity(), "phases": [], "services": []}
    atomic_json(root / "run.json", manifest)
    terminal_line(f"[timer32k] logs={root}; native BS4 / OSCAR BS4+BS8, 32K, output={args.output_tokens}")
    rc = 0
    previous_signals = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    def interrupt(signum, frame):
        # Close the owned server's sockets before ThreadPoolExecutor.__exit__
        # waits for HTTP readers. Otherwise Ctrl-C waits the request deadline.
        if _ACTIVE_SERVER is not None:
            server, grace = _ACTIVE_SERVER
            cleanup_group(server.process, grace)
        raise KeyboardInterrupt(f"signal {signum}")
    for sig in previous_signals:
        signal.signal(sig, interrupt)
    try:
        # Explicit user-requested diagnostic only. Never inherit older sync,
        # phase-print or profiler switches into the uninstrumented controls.
        clean = {"OSCAR_DEBUG_SYNC": "0", "OSCAR_GUARD_HOST_SYNC": "0", "OSCAR_TIMING": "0",
                 "OSCAR_PROFILER": "0", "OSCAR_TIMING_STDERR": "0",
                 "OSCAR_DEVICE_TIMING_CONTROL": None, "OSCAR_PASSIVE_TIMING_CONTROL": None,
                 "OSCAR_TIMER_CONTROL": None, "OSCAR_PROFILE_DIR": None}
        with _environment(clean):
            env = target_env(config)
            env["OSCAR_TARGET_CONFIG"] = str(effective_path)
            env["OSCAR_TERMINAL_LOG_MODE"] = "full"
            for name, command in stages:
                result = _quiet_call(root / "phase-supervisor.log", name, run_phase,
                    name, command, cwd=ROOT, log_dir=root / "phases", env=env,
                    timeout=config["phase_timeout_seconds"], grace=config["shutdown_timeout_seconds"], heartbeat=60)
                manifest["phases"].append(asdict(result))
                atomic_json(root / "run.json", manifest)
                if result.returncode or not result.cleanup_complete:
                    raise TimerRunError(f"{name} failed rc={result.returncode}", result.returncode or 125, result.log)
                terminal_line(f"[timer32k] DONE {name}")
            rows = build_rows(load_tokenizer(config["model"]))
            layers, dim, fingerprint = model_geometry(Path(config["model"]))
            manifest["model_identity"] = {"fingerprint": fingerprint, "full_layers": layers, "head_dim": dim}
            manifest["workload"] = {"prompt_tokens": PROMPT_TOKENS, "output_tokens": args.output_tokens,
                "source": "synthetic_repeated_natural_sentence_with_record_prefix", "prompt_manifest": prompt_manifest(rows),
                "manifest_sha256": canonical_sha256(prompt_manifest(rows)), "prompt_ids_file": str(root / "prompt-ids.json"),
                "cache_policy": "APC disabled and distinct run/request cache salts; same exact IDs in paired arms"}
            atomic_json(root / "prompt-ids.json", rows)
            manifest["pair_hash"] = canonical_sha256({"model": manifest["model_identity"],
                "source": manifest["source"], "argv": serve_argv(config), "workload": manifest["workload"]})
            atomic_json(root / "run.json", manifest)
            with _environment({"OSCAR_TIMER_CONTROL": str(control)}):
                for variant in ("native", "candidate"):
                    _run_service(config, effective_path, rows, root=root, variant=variant,
                        control=control, manifest=manifest, args=args)
        manifest["status"] = "completed"
    except BaseException as error:
        rc = 130 if isinstance(error, KeyboardInterrupt) else getattr(error, "returncode", 1)
        manifest.update(status="interrupted" if rc == 130 else "failed", error=f"{type(error).__name__}: {error}")
        (root / "failure.txt").write_text(traceback.format_exc())
        terminal_line(f"[timer32k] FAILED rc={rc} {type(error).__name__}: {error}", stderr=True)
        _failure_tail(getattr(error, "log", None) or root / "failure.txt")
    finally:
        _disarm(control)
        manifest["returncode"] = rc
        atomic_json(root / "run.json", manifest)
        for sig, handler in previous_signals.items():
            signal.signal(sig, handler)
    try:
        from .summarize_timer import summarize, summary_lines
        summary = summarize(root)
        if summary.get("evidence_status") == "missing" and rc == 0:
            rc = 2
            manifest["returncode"] = rc
            manifest["timer_evidence"] = "missing"
            atomic_json(root / "run.json", manifest)
        atomic_json(root / "timer-summary.json", summary)
        lines = list(summary_lines(summary))
        if len(lines) > MAX_SUMMARY_LINES:
            raise RuntimeError(f"timer summary exceeded {MAX_SUMMARY_LINES} lines")
        (root / "timer-summary.txt").write_text("\n".join(lines) + "\n")
        for line in lines:
            terminal_line(line)
    except Exception as error:
        manifest["summary_error"] = f"{type(error).__name__}: {error}"
        rc = rc or 1
        manifest["returncode"] = rc
        atomic_json(root / "run.json", manifest)
        terminal_line(f"[timer32k] SUMMARY_FAILED {error}; evidence={root}", stderr=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
