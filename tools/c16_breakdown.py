# Archive #70-73/#94-95/#125/#133/#143/#145/#164: bounded owned diagnostics,
# preserved real exits/precision, no HTTP/torch profiler, no production kernel edits.
"""One real candidate 32K request with a separately built C16 cycle diagnostic."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback
import uuid

from benchmarks.measure import metrics_delta, metrics_snapshot
from benchmarks.mixed import run_batch
from .model_config import add_model_arguments, apply_model_options
from .npu_resources import read_npu_resources, wait_for_release
from .phase import atomic_json, cleanup_group, run_phase, terminal_line
from .prepare_rotations import model_geometry
from .service_probe import exact_prompt, load_tokenizer, managed_server
from .serving_variants import runtime_feature_summary, variant_config
from .target_cli import ROOT, diagnose_guard_enabled, serve_argv, target_env
from .timer32k import (TimerRunError, _build_stages, _completion_check, _counter_delta,
                      _environment, _failure_tail, _source_identity, positive_int,
                      positive_seconds, prompt_manifest)

PROMPT_TOKENS = 32768
MAX_SUMMARY_LINES = 150
MAX_SUMMARY_BYTES = 40 * 1024
_ACTIVE_SERVER = None


@contextmanager
def _heartbeat(label, interval=60):
    terminal = sys.stdout
    stop = threading.Event()
    started = time.monotonic()
    def report():
        while not stop.wait(interval):
            terminal.write(f"[c16-breakdown] {label} running_seconds={time.monotonic()-started:.0f}\n")
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
    terminal_line(f"[c16-breakdown] START {label}")
    with _heartbeat(label), path.open("a", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            return function(*args, **kwargs)


@contextmanager
def _signal_target(server, grace):
    global _ACTIVE_SERVER
    prior = _ACTIVE_SERVER
    _ACTIVE_SERVER = (server, grace)
    try:
        yield
    finally:
        _ACTIVE_SERVER = prior


def validate_config(config, output_tokens):
    target_env(config, base={})
    if diagnose_guard_enabled(config) or config.get("profiler_config") or config.get("diagnostic_device_timing"):
        raise ValueError("C16 cycle diagnostic rejects eager guard/other profiler modes")
    if (config.get("dtype") != "bfloat16" or config.get("enable_prefix_caching") is not False
            or config.get("use_native_token_budget") or config.get("max_num_batched_tokens") != 16384
            or config.get("scheduler_cls")):
        raise ValueError("C16 diagnostic preserves BF16, APC-off, native scheduler and explicit 16384 budget")
    if config.get("speculative_config", {}).get("num_speculative_tokens") != 3:
        raise ValueError("C16 diagnostic requires the existing three-token MTP configuration")
    if config.get("compilation_config", {}).get("cudagraph_mode") != "FULL_DECODE_ONLY" or config.get("enforce_eager"):
        raise ValueError("C16 diagnostic preserves FULL_DECODE_ONLY graph configuration")
    if config["max_model_len"] < PROMPT_TOKENS + output_tokens or config["max_num_seqs"] < 1:
        raise ValueError("configured context/request capacity cannot contain this diagnostic")


def diagnostic_build_command(config, root):
    return [sys.executable, "-m", "tools.build_c16_timer", "--soc", config.get("soc_version", "ascend910b4"),
            "--output", str(root / "timer-build.json"), "--log-dir", str(root / "timer-build")]


def load_build_receipt(path):
    receipt = json.loads(Path(path).read_text())
    if receipt.get("status") not in {"built", "reused"}:
        raise TimerRunError("independent C16 library did not publish a successful build receipt", log=path)
    library = receipt.get("library")
    if (not isinstance(library, str) or not Path(library).is_absolute()
            or not Path(library).is_file() or not Path(library).stat().st_size):
        raise TimerRunError("C16 library receipt must identify a nonempty absolute shared-library file", log=path)
    return receipt


def _run_request(server, config, rows, *, output_tokens, timeout, run_id, record):
    record.update(status="running", prompt_manifest=prompt_manifest(rows), output_tokens=output_tokens,
                  timing_scope="instrumented one-request diagnostic only; not a performance comparison")
    started = time.time()
    try:
        server.check_alive()
        with _heartbeat(f"request prompt={len(rows[0]['token_ids'])} output={output_tokens}"):
            sample = run_batch(server.base_url, rows, model=config["served_model_name"],
                output_tokens=output_tokens, timeout=timeout, repeat=run_id, sample_interval=1)
        record.update(sample=sample, completion_check=_completion_check(sample, rows, output_tokens))
        if record["completion_check"]["status"] != "passed":
            raise TimerRunError("C16 diagnostic request lacks complete DONE/usage/token evidence", log=server.log)
        server.check_alive()
        record["status"] = "completed"
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["window"] = [started, time.time()]


def run_service(config, config_path, rows, *, root, manifest, timeout):
    directory = root / "candidate"
    directory.mkdir(parents=True, exist_ok=True)
    service = {"variant": "candidate", "lifecycle": {}, "warmup": {}, "request": {}}
    manifest.update(service=service, services=[service])
    atomic_json(root / "run.json", manifest)
    service["resource_before"] = _quiet_call(directory / "resource-supervisor.log", "resources-before",
        read_npu_resources, config, log_dir=directory / "resources-before")
    try:
        with managed_server(config, config_path, log_dir=directory / "server", lifecycle=service["lifecycle"],
                            native=False, quiet=True) as server:
            with _signal_target(server, config["shutdown_timeout_seconds"]):
                warmup = [{"id": "c16-warmup", "token_ids": rows[0]["token_ids"][:128]}]
                _run_request(server, config, warmup, output_tokens=16, timeout=timeout,
                    run_id=f"{manifest['run_id']}-warmup", record=service["warmup"])
                atomic_json(root / "run.json", manifest)
                before = metrics_snapshot(server.base_url, timeout=10, path=directory / "before.prom")
                service["request"]["metrics_before"] = before
                terminal_line(f"[c16-breakdown] REQUEST candidate bs=1 prompt={PROMPT_TOKENS} output={manifest['workload']['output_tokens']}")
                _run_request(server, config, rows, output_tokens=manifest["workload"]["output_tokens"], timeout=timeout,
                    run_id=f"{manifest['run_id']}-32k", record=service["request"])
                after = metrics_snapshot(server.base_url, timeout=10, path=directory / "after.prom")
                service["request"].update(metrics_after=after, mtp_counter_delta=metrics_delta(before, after),
                                          counter_delta=_counter_delta(before, after))
    except BaseException as error:
        service["error"] = f"{type(error).__name__}: {error}"
        if not isinstance(error, (TimerRunError, KeyboardInterrupt)):
            raise TimerRunError(str(error), log=directory / "server" / "server.log") from error
        raise
    finally:
        try:
            service["resource_release"] = _quiet_call(directory / "resource-supervisor.log", "resources-release",
                wait_for_release, config, service["resource_before"], log_dir=directory / "resources-release",
                timeout=config["shutdown_timeout_seconds"])
        except Exception as error:
            service["resource_release"] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        atomic_json(root / "run.json", manifest)
        atomic_json(directory / "requests.json", {"warmup": service["warmup"], "request": service["request"]})
    if (service["lifecycle"].get("cleanup_complete") is not True
            or service["resource_release"].get("status") != "passed"):
        raise TimerRunError("C16 diagnostic owned service/resource cleanup failed", log=directory / "server" / "server.log")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/target.json")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--plan", action="store_true", help="read-only; no installation, tokenizer, library or model load")
    parser.add_argument("--rear-cards", action="store_true")
    parser.add_argument("--output-tokens", type=positive_int, default=16)
    parser.add_argument("--request-timeout", type=positive_seconds, default=900)
    parser.add_argument("--startup-timeout", type=positive_seconds)
    add_model_arguments(parser)
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    root = (args.log_dir or ROOT / "logs" / f"c16-breakdown-{stamp}").resolve()
    if not args.plan and root.exists() and any(root.iterdir()):
        parser.error("--log-dir must be empty or new; previous diagnostic evidence is never overwritten")
    config = variant_config(apply_model_options(json.loads(args.config.read_text()), args,
                                              require_model=not args.plan), "candidate")
    if args.rear_cards:
        target_env(config, base={})
        config.update(devices=[4, 5, 6, 7], port=7878)
    if args.startup_timeout:
        config["service_startup_timeout_seconds"] = args.startup_timeout
    validate_config(config, args.output_tokens)
    effective_path = root / "effective-target.json"
    stages = _build_stages(args.config, effective_path, root)
    stages.append(("build-c16-timer", diagnostic_build_command(config, root)))
    if args.plan:
        print(json.dumps({"kind": "c16_breakdown", "stages": stages, "variant": "candidate",
            "vllm_argv": serve_argv(config), "runtime_selection": runtime_feature_summary(config, "candidate"),
            "devices": config["devices"], "port": config["port"], "device_dir": str(root / "device"),
            "library_manifest": str(root / "timer-build.json"),
            "services": [{"variant": "candidate", "warmup": {"requests": 1, "prompt_tokens": 128, "output_tokens": 16},
                          "workload": {"requests": 1, "prompt_tokens": PROMPT_TOKENS, "output_tokens": args.output_tokens}}],
            "capture_selection": {"query_tokens": 16384, "context_tokens": 32768,
                                  "warm_history_required": True, "max_calls_per_rank": 20,
                                  "policy": "first matching call per actual layer; observed coverage is reported"},
            "measurement": "independent C16 kernel cycle counters, no HTTP or torch profiler",
            "request_timeout_seconds": args.request_timeout}, indent=2))
        return 0
    root.mkdir(parents=True, exist_ok=True)
    (root / "device").mkdir()
    atomic_json(effective_path, config)
    manifest = {"schema_version": 1, "kind": "c16_breakdown", "status": "running", "root": str(root),
        "run_id": uuid.uuid4().hex, "created_at": stamp, "config_path": str(effective_path),
        "effective_config": config, "source": _source_identity(), "phases": [], "services": [],
        "expected_ranks": config["tensor_parallel_size"], "device_dir": str(root / "device"),
        "measurement": "instrumented kernel cycles, not model throughput or wall attribution"}
    atomic_json(root / "run.json", manifest)
    terminal_line(f"[c16-breakdown] logs={root}; candidate only, one 32K request, no profiler")
    rc = 0
    previous_signals = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    def interrupt(signum, frame):
        if _ACTIVE_SERVER is not None:
            server, grace = _ACTIVE_SERVER
            cleanup_group(server.process, grace)
        raise KeyboardInterrupt(f"signal {signum}")
    for sig in previous_signals:
        signal.signal(sig, interrupt)
    try:
        clean = {"OSCAR_DEBUG_SYNC": "0", "OSCAR_GUARD_HOST_SYNC": "0", "OSCAR_TIMING": "0",
                 "OSCAR_PROFILER": "0", "OSCAR_TIMING_STDERR": "0", "OSCAR_DEVICE_TIMING_CONTROL": None,
                 "OSCAR_PASSIVE_TIMING_CONTROL": None, "OSCAR_TIMER_CONTROL": None, "OSCAR_PROFILE_DIR": None,
                 "OSCAR_C16_BREAKDOWN_DIR": None, "OSCAR_C16_TIMER_LIBRARY": None, "OSCAR_C16_TIMER_MANIFEST": None}
        with _environment(clean):
            env = target_env(config)
            env.update(OSCAR_TARGET_CONFIG=str(effective_path), OSCAR_TERMINAL_LOG_MODE="full")
            for name, command in stages:
                result = _quiet_call(root / "phase-supervisor.log", name, run_phase, name, command,
                    cwd=ROOT, log_dir=root / "phases", env=env, timeout=config["phase_timeout_seconds"],
                    grace=config["shutdown_timeout_seconds"], heartbeat=60)
                manifest["phases"].append(asdict(result))
                atomic_json(root / "run.json", manifest)
                if result.returncode or not result.cleanup_complete:
                    raise TimerRunError(f"{name} failed rc={result.returncode}", result.returncode or 125, result.log)
                terminal_line(f"[c16-breakdown] DONE {name}")
            receipt_path = root / "timer-build.json"
            manifest["timer_build"] = load_build_receipt(receipt_path)
            manifest["diag_build"] = {**manifest["timer_build"], "manifest": str(receipt_path)}
            tokenizer = load_tokenizer(config["model"])
            rows = [{"id": "c16-breakdown-32k", "token_ids": exact_prompt(tokenizer, PROMPT_TOKENS)}]
            layers, dim, fingerprint = model_geometry(Path(config["model"]))
            manifest["model_identity"] = {"fingerprint": fingerprint, "full_layers": layers, "head_dim": dim}
            manifest["workload"] = {"requests": 1, "prompt_tokens": PROMPT_TOKENS, "output_tokens": args.output_tokens,
                "source": "deterministic_repeated_natural_sentence", "prompt_manifest": prompt_manifest(rows),
                "prompt_ids_file": str(root / "prompt-ids.json"), "cache_policy": "APC-off; distinct request salt"}
            atomic_json(root / "prompt-ids.json", rows)
            atomic_json(root / "run.json", manifest)
            with _environment({"OSCAR_C16_BREAKDOWN_DIR": str(root / "device"),
                    "OSCAR_C16_TIMER_LIBRARY": manifest["timer_build"]["library"],
                    "OSCAR_C16_TIMER_MANIFEST": str(receipt_path)}):
                run_service(config, effective_path, rows, root=root, manifest=manifest, timeout=args.request_timeout)
        manifest["status"] = "completed"
    except BaseException as error:
        rc = 130 if isinstance(error, KeyboardInterrupt) else getattr(error, "returncode", 1)
        manifest.update(status="interrupted" if rc == 130 else "failed", error=f"{type(error).__name__}: {error}")
        (root / "failure.txt").write_text(traceback.format_exc())
        terminal_line(f"[c16-breakdown] FAILED rc={rc}: {error}", stderr=True)
        _failure_tail(getattr(error, "log", None) or root / "failure.txt")
    finally:
        manifest["returncode"] = rc
        atomic_json(root / "run.json", manifest)
        for sig, handler in previous_signals.items():
            signal.signal(sig, handler)
    try:
        from .summarize_c16_breakdown import summarize, summary_lines
        summary = summarize(root)
        atomic_json(root / "c16-breakdown-summary.json", summary)
        lines = list(summary_lines(summary))
        text = "\n".join(lines) + "\n"
        if len(lines) > MAX_SUMMARY_LINES or len(text.encode("utf-8")) > MAX_SUMMARY_BYTES:
            raise RuntimeError("C16 summary exceeded 150 lines/40KiB")
        (root / "c16-breakdown-summary.txt").write_text(text)
        for line in lines:
            terminal_line(line)
        if summary.get("status") != "complete":
            rc = rc or 2
    except Exception as error:
        manifest["summary_error"] = f"{type(error).__name__}: {error}"
        (root / "summary-error.txt").write_text(traceback.format_exc())
        terminal_line(f"[c16-breakdown] SUMMARY_FAILED {error}; evidence={root}", stderr=True)
        rc = rc or 1
    manifest["returncode"] = rc
    atomic_json(root / "run.json", manifest)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
