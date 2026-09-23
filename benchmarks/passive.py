"""Observe a user's external load on an already running local service.

Archive #70-#73/#130-#139: no synthetic traffic is sent here. Metrics and
host progress are diagnostic observations; client request lengths, TTFT,
ITL, completion rate and device phase time require the user's client output
or independent device evidence. Never infer them from Running/Waiting gauges.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import threading
import time

from benchmarks.measure import local_url
from tools.phase import atomic_json
from tools.service_probe import _http, parse_gauges, parse_mtp_metrics


METRICS = ("vllm:num_requests_running", "vllm:num_requests_waiting",
           "vllm:prompt_tokens_total", "vllm:generation_tokens_total")
COUNTERS = METRICS[2:]


def _snapshot(raw, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw.encode()).hexdigest()}


def _finite_positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _read_metrics(url, timeout):
    raw = _http(url + "/metrics", timeout=timeout)
    values = parse_gauges(raw, METRICS)
    if METRICS[0] not in values or METRICS[1] not in values:
        raise RuntimeError("/metrics lacks running/waiting gauges; external load window cannot be detected")
    return raw, values


def _close_window(window, raw, values, when, *, output, end_reason, mtp_after):
    window["end_unix"] = when
    window["end_reason"] = end_reason
    window["metrics_after"] = _snapshot(raw, output.parent / f"{output.stem}-window{window['index']}-after.prom")
    window["status"] = ("observed" if end_reason == "idle_after_activity"
        and not window["sampling_errors"] and not window["partial_start"] else "partial")
    wall = max(when - window["start_unix"], 1e-9)
    window["wall_seconds"] = wall
    window["counter_delta"] = {}
    for key in COUNTERS:
        before, after = window["counter_before"].get(key), values.get(key)
        delta = after - before if before is not None and after is not None else None
        window["counter_delta"][key] = delta
        if delta is not None and delta < 0:
            window["status"] = "failed"
        elif delta is None and window["status"] == "observed":
            window["status"] = "partial"
    window["throughput_tps"] = {
        "prompt": None if window["counter_delta"][COUNTERS[0]] is None else
            window["counter_delta"][COUNTERS[0]] / wall,
        "generation": None if window["counter_delta"][COUNTERS[1]] is None else
            window["counter_delta"][COUNTERS[1]] / wall}
    window["throughput_scope"] = ("clipped_start_not_comparable" if window["partial_start"]
        else "observed_external_window_from_idle_baseline")
    if window["partial_start"]:
        window["throughput_tps"] = {"prompt": None, "generation": None}
    mtp_before = window["mtp_before"]
    window["mtp_delta"] = {key: value - mtp_before.get(key, 0) for key, value in mtp_after.items()}
    samples = window["samples"]
    window["peak_observed_running"] = max((s["running"] for s in samples if "running" in s), default=None)
    window["peak_observed_waiting"] = max((s["waiting"] for s in samples if "waiting" in s), default=None)
    window["peak_observed_inflight_lower_bound"] = max(
        (s["running"] + s["waiting"] for s in samples if "running" in s), default=None)
    window["client_request_count"] = "unobserved_without_client_artifact"
    window["prompt_length_distribution"] = "unobserved_without_client_artifact"
    window["ttft_itl_e2e_completion"] = "unobserved_without_client_artifact"
    window["device_phase_ms"] = "unobserved_without_device_timing"


def observe(url, *, output, stop=None, sample_interval=1.0, idle_seconds=15.0,
            await_seconds=None, max_seconds=None, variant="oscar", ready=None):
    """Poll metrics; detect one or more external traffic windows without requests.

    `stop` allows the formal service supervisor to terminate this observer.
    Standalone callers should pass a finite await/max bound.
    """
    if variant not in {"native", "oscar"}:
        raise ValueError("variant must be explicitly native or oscar")
    url = local_url(url)
    sample_interval = _finite_positive(sample_interval, "sample_interval")
    idle_seconds = _finite_positive(idle_seconds, "idle_seconds")
    if await_seconds is not None:
        await_seconds = _finite_positive(await_seconds, "await_seconds")
    if max_seconds is not None:
        max_seconds = _finite_positive(max_seconds, "max_seconds")
    if stop is None and max_seconds is None:
        raise ValueError("standalone observation requires max_seconds")
    stop = threading.Event() if stop is None else stop
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"schema_version": 1, "kind": variant, "mode": "passive_external_load",
              "status": "waiting_for_external_load", "service_url": url,
              "sample_interval_seconds": sample_interval, "idle_close_seconds": idle_seconds,
              "measurement_type": "passive_http_metrics", "performance_acceptance": "not_run",
              "scope": "external client traffic only; this observer sends GET /metrics and no inference requests",
              "windows": [], "errors": [], "client_workload_identity": "unverified_until_client_artifact_attached"}
    atomic_json(output, report)
    prior = None
    active = None
    quiet_since = None
    ready_announced = False
    try:
        while not stop.is_set():
            elapsed = time.monotonic() - started
            if max_seconds is not None and elapsed >= max_seconds:
                break
            if await_seconds is not None and not report["windows"] and active is None and elapsed >= await_seconds:
                break
            try:
                raw, values = _read_metrics(url, min(5.0, sample_interval + 2.0))
            except Exception as error:
                failure = {"elapsed_seconds": elapsed,
                           "error": f"{type(error).__name__}: {error}"}
                report["errors"].append(failure)
                if active is not None:
                    active["sampling_errors"].append(failure)
                atomic_json(output, report)
                if len(report["errors"]) >= 5 and active is None:
                    report["status"] = "failed"
                    break
                stop.wait(sample_interval)
                continue
            now = time.time()
            counters_changed = bool(prior and any(values.get(key) is not None
                and prior[1].get(key) is not None and values[key] > prior[1][key] for key in COUNTERS))
            busy = values[METRICS[0]] + values[METRICS[1]] > 0 or counters_changed
            if active is None and not busy and not ready_announced:
                if ready is not None:
                    ready.set()
                print(f"[oscar-observe] OBSERVER_READY url={url} report={output}; start your existing load client", flush=True)
                ready_announced = True
            if active is None and busy:
                index = len(report["windows"])
                before_raw, before_values = prior if prior is not None else (raw, values)
                active = {"index": index, "status": "observing", "start_unix": now,
                    "partial_start": prior is None,
                    "counter_before": {key: before_values.get(key) for key in COUNTERS},
                    "mtp_before": parse_mtp_metrics(before_raw), "samples": [],
                    "sampling_errors": [],
                    "metrics_before": _snapshot(before_raw,
                        output.parent / f"{output.stem}-window{index}-before.prom")}
                report["windows"].append(active)
                report["status"] = "observing"
                print(f"[oscar-observe] external traffic window={index} started report={output}", flush=True)
            if active is not None:
                sample = {"unix": now, "elapsed_seconds": now - active["start_unix"],
                          "running": values[METRICS[0]], "waiting": values[METRICS[1]],
                          "prompt_tokens_total": values.get(COUNTERS[0]),
                          "generation_tokens_total": values.get(COUNTERS[1])}
                active["samples"].append(sample)
                quiet_since = None if busy else (now if quiet_since is None else quiet_since)
                if quiet_since is not None and now - quiet_since >= idle_seconds:
                    _close_window(active, raw, values, quiet_since, output=output,
                                  end_reason="idle_after_activity", mtp_after=parse_mtp_metrics(raw))
                    print(f"[oscar-observe] external window={active['index']} "
                          f"status={active['status']} peak_inflight_observed={active['peak_observed_inflight_lower_bound']} "
                          f"prompt_tps={active['throughput_tps']['prompt']} "
                          f"gen_tps={active['throughput_tps']['generation']}", flush=True)
                    active = None
                    states = {window["status"] for window in report["windows"]}
                    report["status"] = ("failed" if "failed" in states else
                                        "partial" if "partial" in states else "observed")
            prior = (raw, values)
            atomic_json(output, report)
            stop.wait(sample_interval)
        if active is not None and prior is not None:
            raw, values = prior
            _close_window(active, raw, values, time.time(), output=output,
                          end_reason="observer_stopped_while_busy", mtp_after=parse_mtp_metrics(raw))
        if report["windows"]:
            states = {window["status"] for window in report["windows"]}
            report["status"] = ("failed" if "failed" in states else
                                "partial" if "partial" in states else "observed")
        elif report["status"] != "failed":
            report["status"] = "not_run"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("native", "oscar"), required=True)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--idle-seconds", type=float, default=15.0)
    parser.add_argument("--await-seconds", type=float, default=600.0)
    parser.add_argument("--max-seconds", type=float, default=1800.0)
    args = parser.parse_args(argv)
    report = observe(args.url, output=args.output, variant=args.variant,
                     sample_interval=args.sample_interval, idle_seconds=args.idle_seconds,
                     await_seconds=args.await_seconds, max_seconds=args.max_seconds)
    print(f"[oscar-observe] status={report['status']} report={args.output}", flush=True)
    return 0 if report["status"] == "observed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
