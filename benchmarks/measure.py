"""Measure an explicitly selected native/OSCAR HTTP service, never a fallback.

Archive G31/#70-73: bounded requests, actual streaming timing, paired inputs,
no invented device-phase durations. #94/#95: persist errors and partial runs.
Native protocol evidence: completion/protocol.py return_token_ids documents
delta IDs per SSE chunk; receive timing is not NPU execution timing.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import ipaddress
import json
import math
from pathlib import Path
import socket
import statistics
import threading
import time
from urllib.parse import urlsplit
import uuid

from benchmarks.compare import canonical_sha256, read_json, SOFTWARE_FIELDS
from tools.phase import atomic_json
from tools.prepare_rotations import model_geometry
from tools.service_probe import PROMPT_SENTENCE, load_tokenizer, parse_mtp_metrics

ROOT = Path(__file__).resolve().parents[1]
WARMUPS, REPEATS = 2, 5
STANDARD_INPUT_LENGTHS = (16384, 32768, 50000)
MAX_SSE_EVENT_BYTES = 4 * 1024 * 1024


def local_url(url):
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("service URL must be an http(s) localhost URL without credentials")
    if parsed.hostname != "localhost":
        try:
            loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            loopback = False
        if not loopback:
            raise ValueError("the benchmark client only connects to a local loopback service")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("service URL must be its base URL, without an API path/query/fragment")
    return url.rstrip("/")


def _connection(url, timeout):
    parsed = urlsplit(local_url(url))
    cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    return cls(parsed.hostname, parsed.port, timeout=timeout)


def _deadline(connection, end):
    left = end - time.perf_counter()
    if left <= 0:
        raise TimeoutError("stream exceeded its total request deadline")
    connection.timeout = left
    if connection.sock is not None:
        connection.sock.settimeout(left)


def stream_request(url, payload, *, timeout, request_id, barrier=None):
    """Record client receipt of delta-token SSE events and terminal usage.

    Tokens in one MTP burst share its observed arrival time; their internal
    device emission times are unknown, not interpolated or fabricated.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("request timeout must be finite and positive")
    body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
    if barrier is not None:
        barrier.wait(timeout=timeout)
    started = time.perf_counter()
    end = started + timeout
    result = {"request_id": request_id, "status": "failed", "prompt_tokens_requested": len(payload["prompt"]),
              "output_tokens_requested": payload["max_tokens"], "token_arrivals_ms": [], "bursts": [],
              "timing_clock": "client_monotonic_SSE_receive", "usage": None}
    connection = _connection(url, timeout)
    output_ids, text_parts, usage, finish, done = [], [], None, None, False
    try:
        connection.request("POST", "/v1/completions", body, {"Content-Type": "application/json", "Accept": "text/event-stream"})
        _deadline(connection, end)
        response = connection.getresponse()
        result["http_status"] = response.status
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {response.read(65536).decode('utf-8', errors='replace')}")
        if "text/event-stream" not in response.getheader("Content-Type", ""):
            raise RuntimeError("successful streaming response must use text/event-stream")
        buffered, event_lines = b"", []
        while not done:
            _deadline(connection, end)
            data = response.read1(8192)
            received = time.perf_counter()
            if not data:
                raise RuntimeError("SSE stream ended before the mandatory [DONE] marker")
            buffered += data
            if len(buffered) > MAX_SSE_EVENT_BYTES:
                raise RuntimeError("SSE line exceeded the explicit size limit")
            while b"\n" in buffered:
                line, buffered = buffered.split(b"\n", 1)
                line = line.rstrip(b"\r")
                if line.startswith(b"data:"):
                    event_lines.append(line[5:].lstrip())
                    if sum(map(len, event_lines)) > MAX_SSE_EVENT_BYTES:
                        raise RuntimeError("SSE event exceeded the explicit size limit")
                elif not line and event_lines:
                    event = b"\n".join(event_lines)
                    event_lines.clear()
                    if event == b"[DONE]":
                        done = True
                        result["done_received_ms"] = (received - started) * 1000
                        break
                    message = json.loads(event)
                    if not isinstance(message, dict) or message.get("error"):
                        raise RuntimeError(f"stream reported an error: {message}")
                    if message.get("usage") is not None:
                        usage = message["usage"]
                    choices = message.get("choices", [])
                    if not isinstance(choices, list) or len(choices) > 1:
                        raise RuntimeError("benchmark requires one completion choice per request")
                    for choice in choices:
                        prompt_ids = choice.get("prompt_token_ids")
                        if prompt_ids is not None and prompt_ids != payload["prompt"]:
                            raise RuntimeError("stream prompt_token_ids differ from submitted exact IDs")
                        ids = choice.get("token_ids")
                        text = choice.get("text", "")
                        if ids is None and text:
                            raise RuntimeError("server ignored return_token_ids; token arrival timing is unavailable")
                        if ids is not None:
                            if not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids):
                                raise RuntimeError("invalid delta token_ids in streaming response")
                            if ids:
                                arrival = (received - started) * 1000
                                output_ids.extend(ids)
                                result["token_arrivals_ms"].extend([arrival] * len(ids))
                                result["bursts"].append({"arrival_ms": arrival, "token_count": len(ids)})
                        if isinstance(text, str):
                            text_parts.append(text)
                        if choice.get("finish_reason") is not None:
                            finish = choice["finish_reason"]
        result["usage"] = usage
        if (not isinstance(usage, dict) or usage.get("prompt_tokens") != len(payload["prompt"])
                or usage.get("completion_tokens") != payload["max_tokens"]
                or usage.get("completion_tokens") != len(output_ids)):
            raise RuntimeError(f"terminal usage/token IDs do not match exact workload: usage={usage}, received={len(output_ids)}")
        if finish not in {"length", "stop"}:
            raise RuntimeError(f"missing/invalid terminal finish_reason: {finish!r}")
        arrivals = result["token_arrivals_ms"]
        if not arrivals:
            raise RuntimeError("stream completed without generated token IDs")
        intervals = [b - a for a, b in zip(arrivals, arrivals[1:])]
        result.update(status="completed", ttft_ms=arrivals[0],
            tpot_ms=(arrivals[-1] - arrivals[0]) / (len(arrivals) - 1) if len(arrivals) > 1 else None,
            itl_ms=intervals, generated_token_ids=output_ids, text="".join(text_parts), finish_reason=finish)
    except (TimeoutError, socket.timeout) as error:
        result.update(status="timeout", error=f"{type(error).__name__}: {error}")
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        connection.close()
        result["e2e_ms"] = result.get("done_received_ms", (time.perf_counter() - started) * 1000)
        result["usage"] = usage
        result["generated_token_ids"] = output_ids
    return result


def metrics_snapshot(url, *, timeout, path):
    connection = _connection(url, timeout)
    try:
        connection.request("GET", "/metrics")
        response = connection.getresponse()
        text = response.read(MAX_SSE_EVENT_BYTES + 1).decode("utf-8")
        if response.status != 200 or len(text.encode()) > MAX_SSE_EVENT_BYTES:
            raise RuntimeError(f"metrics response status={response.status} or size exceeds limit")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return {"status": "observed", "path": str(path.resolve()),
                "sha256": hashlib.sha256(text.encode()).hexdigest(), "mtp": parse_mtp_metrics(text)}
    except Exception as error:
        return {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    finally:
        connection.close()


def metrics_delta(before, after):
    if before.get("status") != "observed" or after.get("status") != "observed":
        return {"status": "needs_evidence", "reason": "native metrics unavailable"}
    first, last = before["mtp"], after["mtp"]
    if not last:
        return {"status": "needs_evidence", "reason": "native MTP counters unavailable"}
    delta = {key: value - first.get(key, 0) for key, value in last.items()}
    return {"status": "observed" if all(value >= 0 for value in delta.values()) else "failed",
            "mtp_counter_delta": delta, "baseline_missing": sorted(set(last)-set(first)),
            "device_timing": "not_observed"}


def _p95(values):
    return sorted(values)[math.ceil(.95 * len(values)) - 1] if values else None


def batch_sample(url, *, token_ids, concurrency, output_tokens, model, timeout,
                 run_id, case_id, phase, repeat):
    release = []
    barrier = threading.Barrier(concurrency + 1, action=lambda: release.append(time.perf_counter()))
    payloads = [{"model": model, "prompt": token_ids, "max_tokens": output_tokens,
        "min_tokens": output_tokens, "ignore_eos": True, "temperature": 0, "seed": 46774,
        "stream": True, "stream_options": {"include_usage": True}, "return_token_ids": True,
        "add_special_tokens": False,
        "cache_salt": f"{run_id}:{case_id}:{phase}:{repeat}:{index}"} for index in range(concurrency)]
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(stream_request, url, payload, timeout=timeout,
            request_id=f"{phase}-{repeat}-{index}", barrier=barrier) for index, payload in enumerate(payloads)]
        try:
            barrier.wait(timeout=timeout)
        except threading.BrokenBarrierError:
            barrier.abort()
        requests = []
        for index, future in enumerate(futures):
            try:
                requests.append(future.result())
            except Exception as error:
                requests.append({"request_id": f"{phase}-{repeat}-{index}", "status": "failed",
                                 "error": f"{type(error).__name__}: {error}"})
    elapsed = time.perf_counter() - (release[0] if release else started)
    completed = [item for item in requests if item["status"] == "completed"]
    sample = {"repeat": repeat, "status": "completed" if len(completed) == concurrency else "failed",
        "completed_requests": len(completed), "failed_requests": sum(r["status"] == "failed" for r in requests),
        "timeouts": sum(r["status"] == "timeout" for r in requests), "requests": requests,
        "batch_elapsed_seconds": elapsed, "latency_clock": "client_monotonic_receive",
        "phase_ms": {}, "phase_status": "needs_evidence", "latency_ms": {}, "throughput_tps": {}}
    if completed:
        intervals = [value for result in completed for value in result["itl_ms"]]
        sample["latency_ms"] = {"ttft": statistics.median(r["ttft_ms"] for r in completed),
            "tpot": statistics.median(r["tpot_ms"] for r in completed), "itl_p95": _p95(intervals),
            "e2e": statistics.median(r["e2e_ms"] for r in completed)}
        sample["throughput_tps"] = {"prompt": sum(r["usage"]["prompt_tokens"] for r in completed) / elapsed,
                                    "generation": sum(r["usage"]["completion_tokens"] for r in completed) / elapsed}
    return sample


def measure(config_path, *, variant, output, url=None, lengths=None, concurrency=None,
            output_tokens=32, timeout=300.0, run_timeout=7200.0, dataset=None,
            identity=None, tokenizer_factory=None):
    if variant not in {"native", "oscar"}:
        raise ValueError("variant must be explicitly native or oscar")
    if type(output_tokens) is not int or output_tokens < 2:
        raise ValueError("output_tokens must be an integer >=2 for TPOT/ITL measurement")
    if any(not math.isfinite(x) or x <= 0 for x in (timeout, run_timeout)):
        raise ValueError("request and run timeouts must be finite and positive")
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "kind": variant, "run_id": uuid.uuid4().hex,
        "measurement_type": "http_client", "status": "running", "cases": [], "http_cases": [],
        "phase_definitions": {}, "needs_evidence": ["npu", "route", "graph", "accuracy", "memory", "device_phase_ms", "device_q_len"]}
    atomic_json(output, report)
    started = time.perf_counter()
    try:
        config = read_json(config_path)
        policy = read_json(ROOT / "configs/acceptance.json")
        if policy.get("frozen_before_measurement") is not True or policy["performance"]["warmup"] != WARMUPS or policy["performance"]["repeats"] != REPEATS:
            raise ValueError("measurement requires the frozen two-warmup/five-repeat policy")
        default_scope = "required_standard_inputs" if lengths is None else "explicit_input_lengths"
        lengths = list(STANDARD_INPUT_LENGTHS if lengths is None else lengths)
        concurrency = list(policy["required_concurrency"] if concurrency is None else concurrency)
        for values in (lengths, concurrency):
            if not values or any(type(i) is not int or i < 1 for i in values) or len(values) != len(set(values)):
                raise ValueError("length/concurrency axes must be nonempty unique positive integers")
        if max(concurrency) > config["max_num_seqs"]:
            raise ValueError("requested concurrency exceeds configured max_num_seqs")
        url = local_url(url or f"http://127.0.0.1:{config['port']}")
        _, _, model_hash = model_geometry(Path(config["model"]))
        dataset_bytes = Path(dataset).read_bytes() if dataset is not None else PROMPT_SENTENCE.encode("utf-8")
        text = dataset_bytes.decode("utf-8")
        tokenizer = (load_tokenizer if tokenizer_factory is None else tokenizer_factory)(config["model"])
        piece = tokenizer.encode(text, add_special_tokens=False)
        if not isinstance(piece, list) or not piece or any(type(i) is not int or i < 0 for i in piece):
            raise ValueError("dataset did not encode to nonempty token IDs")
        dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()
        workload = {"output_tokens": output_tokens, "sampling": {"temperature": 0, "seed": 46774, "ignore_eos": True},
            "arrival": "closed_loop_simultaneous_batch", "prefix_cache": "isolated_per_request_unique_cache_salt",
            "max_model_len": config["max_model_len"], "max_num_seqs": config["max_num_seqs"], "async_scheduling": True,
            "lengths": lengths, "concurrency": concurrency, "warmup": WARMUPS, "repeats": REPEATS,
            "stream": {"include_usage": True, "return_token_ids": True}, "timing_clock": "client_monotonic_receive"}
        declared = {} if identity is None else read_json(identity)
        pair = {"model_fingerprint": model_hash, "model_fingerprint_scope": "local_config_and_weight_index",
            "dataset_sha256": dataset_hash, "config_sha256": canonical_sha256(config),
            "workload_sha256": canonical_sha256(workload), "workload": workload,
            "devices": config.get("devices"), "tp": config["tensor_parallel_size"],
            "mtp": config["speculative_config"], "graph": config["compilation_config"]["cudagraph_mode"],
            "draft_graph_scope": "eager" if config["speculative_config"].get("enforce_eager") else "unknown",
            "software": declared.get("software"), "npu_model": declared.get("npu_model")}
        for field in ("devices", "tp"):
            if field in declared and declared[field] != pair[field]:
                raise ValueError(f"external identity {field} differs from target configuration")
        if pair["software"] is not None and not isinstance(pair["software"], dict):
            raise ValueError("external identity software must be an object")
        missing_identity = [name for name in SOFTWARE_FIELDS if not isinstance((pair["software"] or {}).get(name), str)
                            or not (pair["software"] or {}).get(name)]
        if missing_identity or not pair["npu_model"]:
            report["needs_evidence"].append("server_hardware_and_software_identity")
        report.update(pair=pair, service_url=url, acceptance_sha256=canonical_sha256(policy),
            default_workload_scope=default_scope,
            identity_source=None if identity is None else {"path": str(Path(identity).resolve()), "sha256": hashlib.sha256(Path(identity).read_bytes()).hexdigest()},
            matrix={"requested_input_lengths": lengths, "requested_concurrency": concurrency,
                "required_input_lengths": policy["required_input_lengths"], "required_concurrency": policy["required_concurrency"],
                "required_q_lens": policy["required_q_lens"], "q_len": "unobserved_by_http",
                "full_acceptance_matrix_complete": False,
                "http_axes_complete": set(lengths) == set(policy["required_input_lengths"]) and set(concurrency) == set(policy["required_concurrency"])})
        artifacts = output.parent / (output.stem + "-artifacts") / report["run_id"]
        artifacts.mkdir(parents=True)
        (artifacts / "dataset.txt").write_bytes(dataset_bytes)
        deadline = started + run_timeout
        for length in lengths:
            ids = (piece * ((length + len(piece) - 1) // len(piece)))[:length]
            input_path = artifacts / f"input-{length}.json"
            atomic_json(input_path, ids)
            for count in concurrency:
                case_id = f"len-{length}-concurrency-{count}"
                case = {"input_tokens": length, "concurrency": count, "q_len": None,
                    "input_sha256": canonical_sha256(ids), "input_file": str(input_path),
                    "status": "running", "warmup": [], "samples": [], "evidence": {}, "phase_status": "needs_evidence"}
                # HTTP cannot control or observe device MTP verification q_len;
                # do not fabricate four kernel cases from one request stream.
                report["http_cases"].append(case)
                if length + output_tokens > config["max_model_len"]:
                    case.update(status="not_run", reason="input plus requested output exceeds max_model_len; input was not silently shortened")
                    atomic_json(output, report)
                    continue
                left = deadline - time.perf_counter()
                if left <= 0:
                    raise TimeoutError("benchmark reached its bounded whole-run deadline")
                before = metrics_snapshot(url, timeout=min(timeout, 10, left), path=artifacts / f"{case_id}-before.prom")
                case["native_metrics_before"] = before
                for phase, repeats in (("warmup", WARMUPS), ("samples", REPEATS)):
                    for repeat in range(repeats):
                        left = deadline - time.perf_counter()
                        if left <= 0:
                            raise TimeoutError("benchmark reached its bounded whole-run deadline")
                        sample = batch_sample(url, token_ids=ids, concurrency=count, output_tokens=output_tokens,
                            model=config["served_model_name"], timeout=min(timeout, left), run_id=report["run_id"],
                            case_id=case_id, phase=phase, repeat=repeat)
                        case[phase].append(sample)
                        atomic_json(output, report)
                left = deadline - time.perf_counter()
                if left <= 0:
                    raise TimeoutError("benchmark reached its bounded whole-run deadline")
                after = metrics_snapshot(url, timeout=min(timeout, 10, left), path=artifacts / f"{case_id}-after.prom")
                case["native_metrics_after"] = after
                case["native_metrics_delta"] = metrics_delta(before, after)
                case["status"] = "needs_evidence" if all(x["status"] == "completed" for x in case["warmup"] + case["samples"]) else "failed"
                if case["native_metrics_delta"]["status"] == "failed":
                    case["status"] = "failed"
                atomic_json(output, report)
                print(f"[oscar-benchmark] {variant} {case_id}: {case['status']}", flush=True)
        report["status"] = "failed" if any(c["status"] == "failed" for c in report["http_cases"]) else "needs_evidence"
        report["client_measurements_complete"] = all(c["status"] == "needs_evidence" for c in report["http_cases"])
    except KeyboardInterrupt as error:
        report.update(status="interrupted", error=str(error), client_measurements_complete=False)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", client_measurements_complete=False)
    finally:
        for case in report["http_cases"]:
            if case["status"] == "running":
                case.update(status="not_run", reason="measurement interrupted before this case completed")
        report["elapsed_seconds"] = time.perf_counter() - started
        atomic_json(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=("native", "oscar"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/target.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url")
    parser.add_argument("--lengths", type=int, nargs="+", help="default: startup §10.1 standard inputs 16384 32768 50000; full policy remains separately reported")
    parser.add_argument("--concurrency", type=int, nargs="+")
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--run-timeout", type=float, default=7200)
    parser.add_argument("--dataset", type=Path, help="local UTF-8 text encoded into repeatable exact prompt IDs")
    parser.add_argument("--identity", type=Path, help="existing audited server software/NPU identity JSON, not a hardware certificate")
    args = parser.parse_args(argv)
    try:
        report = measure(args.config, variant=args.variant, output=args.output, url=args.url, lengths=args.lengths,
            concurrency=args.concurrency, output_tokens=args.output_tokens, timeout=args.timeout,
            run_timeout=args.run_timeout, dataset=args.dataset, identity=args.identity)
    except Exception as error:
        report = {"status": "failed", "kind": args.variant, "measurement_type": "http_client",
                  "client_measurements_complete": False, "error": f"{type(error).__name__}: {error}"}
        atomic_json(args.output, report)
    print(json.dumps({"status": report["status"], "client_measurements_complete": report.get("client_measurements_complete", False),
        "report": str(args.output.resolve()), "performance_acceptance": "not_run"}, indent=2))
    return 0 if report.get("client_measurements_complete") and report["status"] == "needs_evidence" else 1


if __name__ == "__main__":
    raise SystemExit(main())
