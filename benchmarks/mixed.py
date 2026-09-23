"""Bounded 20-30K mixed-length concurrency diagnostic for an explicit service.

Archive #70-#73/#130-#139: preserve exact request IDs, client SSE times,
incomplete requests, scheduler gauges, and native counters. This is an HTTP
diagnostic only; it cannot establish device phase time, graph, or accuracy.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import math
from pathlib import Path
import statistics
import threading
import time

from benchmarks.compare import canonical_sha256, read_json
from benchmarks.measure import stream_request
from tools.phase import atomic_json
from tools.service_probe import parse_gauges


GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting")
MAX_METRICS_BYTES = 4 * 1024 * 1024


def _percentile(values, fraction):
    return sorted(values)[math.ceil(fraction * len(values)) - 1] if values else None


def _metrics_gauges(url, timeout):
    # Deliberately independent of the inference request socket.
    from benchmarks.measure import _connection
    conn = _connection(url, timeout)
    try:
        conn.request("GET", "/metrics")
        response = conn.getresponse()
        raw = response.read(MAX_METRICS_BYTES + 1)
        if response.status != 200 or len(raw) > MAX_METRICS_BYTES:
            raise RuntimeError(f"/metrics status={response.status} or response too large")
        values = parse_gauges(raw.decode("utf-8"), GAUGES)
        if any(name not in values for name in GAUGES):
            raise RuntimeError("/metrics lacks running or waiting gauge")
        return values
    finally:
        conn.close()


def _gauge_sampler(url, started, stop, samples, interval):
    while not stop.is_set():
        timestamp = time.perf_counter() - started
        try:
            gauges = _metrics_gauges(url, min(3.0, interval))
            samples.append({"elapsed_seconds": timestamp,
                            "running": gauges[GAUGES[0]], "waiting": gauges[GAUGES[1]]})
        except Exception as error:
            samples.append({"elapsed_seconds": timestamp,
                            "error": f"{type(error).__name__}: {error}"})
        stop.wait(interval)


def run_batch(url, rows, *, model, output_tokens, timeout, repeat,
              sample_interval=2.0):
    """One simultaneous arrival burst with a hard deadline for each request."""
    count = len(rows)
    barrier = threading.Barrier(count + 1, action=lambda: release.append(time.perf_counter_ns()))
    release = []
    stop = threading.Event()
    gauges = []
    started = time.perf_counter()
    sampler = threading.Thread(target=_gauge_sampler,
        args=(url, started, stop, gauges, sample_interval), daemon=True)
    sampler.start()
    workload_hash = canonical_sha256([{"id": row["id"],
        "prompt_sha256": canonical_sha256(row["token_ids"])} for row in rows])
    payloads = []
    for index, row in enumerate(rows):
        payloads.append({"model": model, "prompt": row["token_ids"],
            "max_tokens": output_tokens, "min_tokens": output_tokens,
            "ignore_eos": True, "temperature": 0, "seed": 46774,
            "stream": True, "stream_options": {"include_usage": True},
            "return_token_ids": True, "add_special_tokens": False,
            "cache_salt": f"mixed:{workload_hash[:16]}:{repeat}:{index}"})
    requests = []
    with ThreadPoolExecutor(max_workers=count) as pool:
        futures = [pool.submit(stream_request, url, payload, timeout=timeout,
            request_id=row["id"], barrier=barrier)
            for payload, row in zip(payloads, rows)]
        try:
            barrier.wait(timeout=timeout)
        except threading.BrokenBarrierError:
            barrier.abort()
        for row, future in zip(rows, futures):
            try:
                result = future.result()
            except Exception as error:
                result = {"request_id": row["id"], "status": "failed",
                          "error": f"{type(error).__name__}: {error}"}
            result["prompt_tokens"] = len(row["token_ids"])
            result["prompt_sha256"] = canonical_sha256(row["token_ids"])
            result["cache_salt_sha256"] = hashlib.sha256(payloads[len(requests)]["cache_salt"].encode()).hexdigest()
            if release and "request_start_monotonic_ns" in result:
                result["release_offset_ms"] = (result["request_start_monotonic_ns"] - release[0]) / 1e6
            requests.append(result)
    stop.set()
    sampler.join(timeout=sample_interval + 3)
    elapsed = (time.perf_counter_ns() - release[0]) / 1e9 if release else time.perf_counter() - started
    completed = [r for r in requests if r["status"] == "completed"]
    failed = [r for r in requests if r["status"] == "failed"]
    timed_out = [r for r in requests if r["status"] == "timeout"]
    summary = {"repeat": repeat, "status": "completed" if len(completed) == count else "failed",
        "arrival": "simultaneous_barrier", "clock": "client_monotonic_SSE_receive",
        "prefix_cache_policy": "distinct deterministic cache_salt requested per prompt; actual cache hits need server metrics",
        "release_monotonic_ns": release[0] if release else None,
        "batch_elapsed_seconds": elapsed, "completed_requests": len(completed),
        "failed_requests": len(failed), "timeouts": len(timed_out),
        "requests": requests, "running_waiting_samples": gauges,
        "latency_ms": {}, "throughput_tps": {}}
    if completed:
        summary["latency_ms"] = {
            name: {"p50": statistics.median(r[name] for r in completed if r.get(name) is not None),
                   "p95": _percentile([r[name] for r in completed if r.get(name) is not None], .95)}
            for name in ("ttft_ms", "tpot_ms", "e2e_ms")}
        summary["throughput_tps"] = {
            "prompt": sum(r["usage"]["prompt_tokens"] for r in completed) / elapsed,
            "generation": sum(r["usage"]["completion_tokens"] for r in completed) / elapsed}
    return summary


def compare_mixed(native, oscar):
    """Pair exact HTTP workloads and show regression ratios without acceptance."""
    def normalize(report):
        if report.get("mode") != "synthetic_mixed":
            return report
        sample = report.get("synthetic_mixed")
        if not isinstance(sample, dict):
            return report
        return {"kind": report.get("variant"), "pair_sha256": sample.get("pair_sha256"),
                "prompt_manifest": sample.get("prompt_manifest"),
                "status": "needs_evidence" if report.get("status") == "measured" else "failed",
                "batches": [sample["sample"]] if isinstance(sample.get("sample"), dict) else []}

    native, oscar = normalize(native), normalize(oscar)
    issues = []
    if native.get("kind") != "native" or oscar.get("kind") != "oscar":
        issues.append("reports must be explicitly native and oscar")
    if not native.get("pair_sha256") or native.get("pair_sha256") != oscar.get("pair_sha256"):
        issues.append("workload/config/model/prompt token identity differs")
    if native.get("prompt_manifest") != oscar.get("prompt_manifest"):
        issues.append("per-request exact prompt IDs or order differ")
    if native.get("status") != "needs_evidence" or oscar.get("status") != "needs_evidence":
        issues.append("both runs need every request completed")
    if len(native.get("batches", [])) != len(oscar.get("batches", [])):
        issues.append("repeat counts differ")
    result = {"schema_version": 1, "scope": "paired HTTP diagnostic",
              "performance_acceptance": "not_run", "status": "failed" if issues else "diagnostic_measured",
              "issues": issues, "pair_sha256": native.get("pair_sha256"), "batches": []}
    if issues:
        return result
    for left, right in zip(native["batches"], oscar["batches"]):
        n_requests = {r["request_id"]: r for r in left["requests"]}
        o_requests = {r["request_id"]: r for r in right["requests"]}
        if n_requests.keys() != o_requests.keys():
            result["issues"].append("request IDs differ between paired batches")
            continue
        rows = []
        for request_id in n_requests:
            n, o = n_requests[request_id], o_requests[request_id]
            if n["prompt_sha256"] != o["prompt_sha256"] or n["status"] != "completed" or o["status"] != "completed":
                result["issues"].append(f"request {request_id} prompt/status mismatch")
                continue
            rows.append({"request_id": request_id, "prompt_tokens": n["prompt_tokens"],
                "native_ms": {k: n.get(k) for k in ("ttft_ms", "tpot_ms", "e2e_ms")},
                "oscar_ms": {k: o.get(k) for k in ("ttft_ms", "tpot_ms", "e2e_ms")},
                "oscar_over_native": {k: o[k] / n[k] if n.get(k) and o.get(k) is not None else None
                                      for k in ("ttft_ms", "tpot_ms", "e2e_ms")}})
        result["batches"].append({"repeat": left["repeat"], "requests": rows,
            "prompt_throughput_ratio": right["throughput_tps"]["prompt"] / left["throughput_tps"]["prompt"],
            "generation_throughput_ratio": right["throughput_tps"]["generation"] / left["throughput_tps"]["generation"],
            "native_completed": left["completed_requests"], "oscar_completed": right["completed_requests"]})
    if result["issues"]:
        result["status"] = "failed"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pair = sub.add_parser("compare", help="pair two completed HTTP diagnostic reports")
    pair.add_argument("--native", type=Path, required=True)
    pair.add_argument("--oscar", type=Path, required=True)
    pair.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare_mixed(read_json(args.native), read_json(args.oscar))
    atomic_json(args.output, report)
    print(f"[oscar-benchmark] paired status={report['status']} report={args.output}")
    return 0 if report["status"] == "diagnostic_measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
