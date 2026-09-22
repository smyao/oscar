# Archive G21/G22/G31/#27/#50-52/#68/#72-78/#84/#94-95/#117: owned server
# lifecycle, bounded real requests, per-rank route/graph evidence, always-final
# logs and release accounting. #118/#122: only current owned-PID CANN plog evidence.
# #129: per-request wall deadlines, active progress and visible cleanup stages.
"""Exercise the configured TP4/MTP service and preserve independent evidence."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

from .npu_resources import DEFAULT_RELEASE_TOLERANCE, read_npu_resources, wait_for_release
from .phase import atomic_json, cleanup_group, live_log, terminal_line
from .plog import OwnedProcessGroup, attach_plog
from .prepare_rotations import model_geometry
from .target_cli import ROOT, serve_argv, target_env

PROMPT_SENTENCE = (
    "A small research team records river levels each morning. "
    "They compare the measurements carefully, explain changes, and preserve the original observations. "
)
PROBE_LENGTHS = (128, 16384, 32768, 50000)
MIN_OUTPUT_TOKENS = 16
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
REQUEST_HEARTBEAT_SECONDS = 15.0

_PRINT_LOCK = threading.Lock()


def _terminal(line, *, stderr=False):
    # Mixed-phase request threads print concurrently; keep each line whole.
    with _PRINT_LOCK:
        terminal_line(line, stderr=stderr)


class ServiceProbeError(RuntimeError):
    pass


def _positive_time(config, name, default):
    value = float(config.get(name, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _remaining(deadline, maximum):
    seconds = min(maximum, deadline - time.monotonic())
    if seconds <= 0:
        raise TimeoutError("whole-service probe exceeded its configured phase bound")
    return seconds


def _http(url, *, payload=None, timeout=5.0):
    data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
    request = urllib.request.Request(url, data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"})
    # A localhost service must not be routed through an inherited HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            status = response.status
    except urllib.error.HTTPError as error:
        body = error.read(65536).decode("utf-8", errors="replace")
        raise ServiceProbeError(f"HTTP {error.code} from {url}: {body}") from error
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ServiceProbeError(f"response from {url} exceeded {MAX_RESPONSE_BYTES} bytes")
    if status != 200:
        raise ServiceProbeError(f"HTTP {status} from {url}")
    return raw.decode("utf-8")


def exact_prompt(tokenizer, length):
    if type(length) is not int or length < 1:
        raise ValueError("prompt length must be a positive integer")
    piece = tokenizer.encode(PROMPT_SENTENCE, add_special_tokens=False)
    if not isinstance(piece, list) or not piece or any(type(x) is not int or x < 0 for x in piece):
        raise ServiceProbeError("local tokenizer did not produce a nonempty token ID sequence")
    return (piece * ((length + len(piece) - 1) // len(piece)))[:length]


def load_tokenizer(model):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(model), trust_remote_code=True, local_files_only=True)


def _bind_host(host):
    return "127.0.0.1" if host in {"0.0.0.0", ""} else "::1" if host == "::" else host


def _assert_port_free(host, port):
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError as error:
            raise ServiceProbeError(f"configured address {host}:{port} is already in use; no existing process was touched") from error


@dataclass
class Server:
    process: subprocess.Popen
    base_url: str
    log: Path
    trace_dir: Path
    lifecycle: dict
    ownership: OwnedProcessGroup | None = None

    def check_alive(self):
        if self.ownership is not None:
            self.ownership.refresh()
            self.lifecycle["owned_pids"] = sorted(self.ownership.pids)
            if self.ownership.error is not None:
                self.lifecycle["pid_observation_error"] = self.ownership.error
        code = self.process.poll()
        if code is not None:
            raise ServiceProbeError(f"owned vLLM server exited rc={code}; log={self.log}")


@contextmanager
def managed_server(config, config_path, *, log_dir, lifecycle=None, command=None):
    """Launch one owned process group; cleanup also runs before failed startup.

    ``command`` is dependency injection for subprocess fault tests; the CLI
    never accepts an alternate command or a replacement inference backend.
    """
    log_dir = Path(log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    lifecycle = {} if lifecycle is None else lifecycle
    trace_dir = log_dir / f"trace-{uuid.uuid4().hex}"
    trace_dir.mkdir()
    environment = target_env(config)
    environment["OSCAR_TRACE_DIR"] = str(trace_dir)
    environment["OSCAR_TARGET_CONFIG"] = str(Path(config_path).resolve())
    environment["PYTHONUNBUFFERED"] = "1"
    command = list(command) if command is not None else [sys.executable, "-m", "tools.target_cli", "--config", str(Path(config_path).resolve())]
    host, port = str(config["host"]), int(config["port"])
    if not 0 < port < 65536:
        raise ValueError("configured service port is outside 1..65535")
    _assert_port_free(host, port)
    health_host = _bind_host(host)
    base_url = f"http://{'[' + health_host + ']' if ':' in health_host else health_host}:{port}"
    log = log_dir / "server.log"
    started = time.monotonic()
    startup_timeout = _positive_time(config, "service_startup_timeout_seconds", config.get("phase_timeout_seconds", 1800))
    grace = _positive_time(config, "shutdown_timeout_seconds", 30)
    process = None
    ownership = None
    lifecycle.update(status="starting", command=command, target_argv=serve_argv(config),
                     log=str(log), trace_dir=str(trace_dir), cleanup_complete=False,
                     debug_sync=environment.get("OSCAR_DEBUG_SYNC", "0").lower() in {"1", "true"})
    print(f"[oscar] worker checkpoints debug_sync={lifecycle['debug_sync']} trace={trace_dir}", flush=True)
    atomic_json(log_dir / "server_lifecycle.json", lifecycle)
    with live_log(log) as stream:
        stream.write(f"START owned_service cwd={ROOT} command={json.dumps(command)}\n")
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            lifecycle["pid"] = process.pid
            ownership = OwnedProcessGroup(process.pid)
            server = Server(process, base_url, log, trace_dir, lifecycle, ownership)
            deadline = started + startup_timeout
            heartbeat = started + 15
            last_health_error = None
            while True:
                server.check_alive()
                remaining = _remaining(deadline, 2)
                try:
                    _http(base_url + "/health", timeout=remaining)
                    server.check_alive()
                    break
                except (OSError, urllib.error.URLError, ServiceProbeError, TimeoutError) as error:
                    last_health_error = str(error)
                    lifecycle["last_health_error"] = f"{type(error).__name__}: {error}"
                if time.monotonic() >= heartbeat:
                    stream.write(f"STARTUP_WAIT elapsed={time.monotonic()-started:.1f} health={last_health_error}\n")
                    print(f"[oscar] waiting for service health log={log}", flush=True)
                    heartbeat = time.monotonic() + 15
                time.sleep(min(0.2, max(0, deadline - time.monotonic())))
            lifecycle.update(status="healthy", startup_seconds=round(time.monotonic() - started, 6))
            atomic_json(log_dir / "server_lifecycle.json", lifecycle)
            yield server
            lifecycle["status"] = "finished"
        except BaseException as error:
            lifecycle.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                             error=f"{type(error).__name__}: {error}")
            stream.write("SERVICE_ERROR " + lifecycle["error"] + "\n")
            raise
        finally:
            print(f"[oscar] CLEANUP_START owned_server pid={None if process is None else process.pid} log={log}", flush=True)
            if ownership is not None:
                ownership.refresh(force=True)
                lifecycle["owned_pids"] = sorted(ownership.pids)
                if ownership.error is not None:
                    lifecycle["pid_observation_error"] = ownership.error
            try:
                lifecycle["cleanup_complete"] = process is None or cleanup_group(process, grace)
            except BaseException as error:
                lifecycle["cleanup_complete"] = False
                lifecycle["cleanup_error"] = f"{type(error).__name__}: {error}"
            lifecycle["exit_code"] = None if process is None else process.poll()
            lifecycle["elapsed_seconds"] = round(time.monotonic() - started, 6)
            if not lifecycle["cleanup_complete"]:
                lifecycle["status"] = "failed"
                lifecycle.setdefault("cleanup_error", "owned process group remains after bounded SIGTERM/SIGKILL")
            stream.write("SERVICE_RESULT " + json.dumps(lifecycle, allow_nan=False) + "\n")
            atomic_json(log_dir / "server_lifecycle.json", lifecycle)
            print(f"[oscar] CLEANUP_END owned_server complete={lifecycle['cleanup_complete']} exit={lifecycle['exit_code']}", flush=True)


def worker_progress(server):
    """Read bounded host-side progress files, never query a blocked NPU."""
    result = []
    for path in sorted(server.trace_dir.glob("phase-*.json"))[:16]:
        try:
            if path.stat().st_size > 65536:
                continue
            record = json.loads(path.read_text())
            result.append({key: record.get(key) for key in
                           ("pid", "rank", "state", "phase", "layer", "tokens", "wall_time")})
        except (OSError, ValueError) as error:
            result.append({"path": str(path), "read_error": str(error)})
    if not result:
        # Normal serving does not force synchronization. Show the latest host
        # event instead of an ambiguous empty list; it is NOT device completion.
        for path in sorted(server.trace_dir.glob("worker-*.jsonl"))[:16]:
            try:
                with path.open("rb") as stream:
                    stream.seek(0, 2);stream.seek(max(0, stream.tell()-16384))
                    lines = stream.read(16384).decode("utf-8", errors="replace").splitlines()
                for line in reversed(lines):
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    result.append({"state": "last_host_event_only", "event": record.get("event"),
                                   **{k: record.get(k) for k in ("pid", "rank", "layer", "tokens", "max_seq_len", "wall_time")},
                                   "device_completion": "not_established"})
                    break
            except OSError as error:
                result.append({"path": str(path), "read_error": str(error)})
    if not result:
        result.append({"state": "no_worker_progress_yet", "debug_sync": server.lifecycle.get("debug_sync", False),
                       "device_completion": "not_established"})
    return result


def request_with_deadline(server, payload, *, label, timeout):
    """One child per HTTP request: socket progress cannot extend the deadline.

    The client inherits the supervisor's group (no detached subprocess). It
    imports no NPU backend. This also works in the mixed-request worker threads.
    """
    if not math.isfinite(timeout) or timeout <= 0 or Path(label).name != label:
        raise ValueError("request needs a positive finite timeout and a simple label")
    directory = server.log.parent / "requests" / f"{label}-{uuid.uuid4().hex}"
    directory.mkdir(parents=True)
    state_path, response_path = directory / "status.json", directory / "response.json"
    started = time.monotonic()
    deadline = started + timeout
    state = {"label": label, "status": "running", "prompt_tokens": len(payload["prompt"]),
             "timeout_seconds": timeout, "started_unix": time.time()}
    atomic_json(state_path, state)
    atomic_json(directory / "request.json", {"url": server.base_url + "/v1/completions",
                                             "payload": payload, "timeout": timeout})
    _terminal(f"[oscar] REQUEST_START {label} prompt={state['prompt_tokens']} timeout={timeout:.1f}s status={state_path}")
    process = None
    environment = dict(os.environ, PYTHONUNBUFFERED="1", TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    try:
        with (directory / "client.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-m", "tools.http_request",
                str(directory / "request.json"), str(response_path)], cwd=ROOT, env=environment,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            state["client_pid"] = process.pid
            heartbeat = started + REQUEST_HEARTBEAT_SECONDS
            while process.poll() is None:
                now = time.monotonic()
                if now >= deadline:
                    raise TimeoutError(f"{label}: wall deadline {timeout:.1f}s exceeded")
                server.check_alive()
                if now >= heartbeat:
                    client_state_path = directory / "client-state.json"
                    client_state = json.loads(client_state_path.read_text()) if client_state_path.is_file() else {"state": "client_starting"}
                    state.update(elapsed_seconds=now-started, remaining_seconds=deadline-now,
                                 worker_progress=worker_progress(server), client_state=client_state)
                    atomic_json(state_path, state)
                    _terminal(f"[oscar] REQUEST_WAIT {label} elapsed={now-started:.1f}s remaining={deadline-now:.1f}s "
                              f"client={client_state.get('state')} workers={json.dumps(state['worker_progress'], ensure_ascii=False)}")
                    heartbeat = now + REQUEST_HEARTBEAT_SECONDS
                time.sleep(min(.1, max(0, deadline-now)))
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{label}: wall deadline {timeout:.1f}s exceeded")
        response = json.loads(response_path.read_text()) if response_path.is_file() else {}
        if process.returncode or response.get("status") != "passed":
            raise ServiceProbeError(f"{label}: HTTP client rc={process.returncode}: "
                                    f"{response.get('error', 'missing response')} log={directory/'client.log'}")
        state["status"] = "received"
        return response["body"]
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}",
                     worker_progress=worker_progress(server))
        atomic_json(state_path, state)  # visible before any potentially slow cleanup
        _terminal(f"[oscar] REQUEST_ERROR {label}: {state['error']}; workers={json.dumps(state['worker_progress'])}",
                  stderr=True)
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)
        state["elapsed_seconds"] = time.monotonic() - started
        state["client_returncode"] = None if process is None else process.returncode
        atomic_json(state_path, state)


def completion(server, config, tokenizer, length, *, label, timeout, prompt_ids=None):
    server.check_alive()
    ids = exact_prompt(tokenizer, length) if prompt_ids is None else prompt_ids
    if len(ids) != length or any(type(x) is not int or x < 0 for x in ids):
        raise ServiceProbeError("prepared prompt IDs do not match the exact requested length")
    payload = {"model": config["served_model_name"], "prompt": ids,
               "max_tokens": MIN_OUTPUT_TOKENS, "min_tokens": MIN_OUTPUT_TOKENS,
               "ignore_eos": True, "temperature": 0, "seed": 46774,
               "stream": False, "add_special_tokens": False}
    started = time.monotonic()
    response = request_with_deadline(server, payload, label=label, timeout=timeout)
    if not isinstance(response, dict) or response.get("error"):
        raise ServiceProbeError(f"{label}: completion returned an error: {response}")
    usage, choices = response.get("usage", {}), response.get("choices")
    if usage.get("prompt_tokens") != length:
        raise ServiceProbeError(f"{label}: requested exactly {length} prompt IDs but server reported {usage}")
    if type(usage.get("completion_tokens")) is not int or usage["completion_tokens"] < MIN_OUTPUT_TOKENS:
        raise ServiceProbeError(f"{label}: fewer than {MIN_OUTPUT_TOKENS} generated tokens: {usage}")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0].get("text"), str) or not choices[0]["text"]:
        raise ServiceProbeError(f"{label}: invalid or empty completion choice")
    if choices[0].get("finish_reason") not in {"length", "stop"}:
        raise ServiceProbeError(f"{label}: unexpected finish_reason {choices[0].get('finish_reason')!r}")
    server.check_alive()
    return {"label": label, "status": "passed", "prompt_tokens": length,
            "completion_tokens": usage["completion_tokens"], "elapsed_seconds": time.monotonic() - started,
            "prompt_ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
            "response": response}


def read_telemetry(directory):
    records = []
    for path in sorted(Path(directory).glob("worker-*.jsonl")):
        data = path.read_text()
        for line_no, line in enumerate(data.splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ServiceProbeError(f"malformed telemetry {path}:{line_no}") from error
            if not isinstance(record, dict):
                raise ServiceProbeError(f"non-object telemetry {path}:{line_no}")
            records.append(record)
    return records


def evaluate_telemetry(records, *, tensor_parallel_size, expected_layers, require_mtp=True):
    expected_ranks = set(range(tensor_parallel_size))
    errors, details = [], {}
    coverage = {}
    canonical = re.compile(r"(?:^|\.)model\.layers\.(\d+)\.self_attn\.attn$")
    for rank in sorted(expected_ranks):
        local = [record for record in records if type(record.get("rank")) is int and record["rank"] == rank]
        layouts = {r.get("layer"): r for r in local if r.get("event") == "cache_layout" and isinstance(r.get("layer"), str)}
        routed = {r.get("layer") for r in local if r.get("event") == "attention_dispatched" and r.get("route") == "ascendc_int2_cv"}
        target = {f"model.layers.{match.group(1)}.self_attn.attn" for name in layouts
                  if "mtp" not in name and (match := canonical.search(name))}
        mtp = {name for name in layouts if "mtp" in name}
        if target != set(expected_layers):
            errors.append(f"rank {rank} target FULL layer coverage differs from the actual model config")
        if require_mtp and not mtp:
            errors.append(f"rank {rank} has no MTP FULL layer cache evidence")
        if not layouts or set(layouts) - routed:
            errors.append(f"rank {rank} has unexecuted FULL routes: {sorted(set(layouts)-routed)}")
        for name, layout in layouts.items():
            if (layout.get("gdn_reshape") != "native" or type(layout.get("physical_blocks")) is not int
                    or layout["physical_blocks"] <= 0 or type(layout.get("physical_block_tokens")) is not int
                    or layout["physical_block_tokens"] <= 0 or type(layout.get("snapshot_bytes_per_page")) is not int
                    or type(layout.get("page_bytes")) is not int
                    or not 0 < layout["snapshot_bytes_per_page"] <= layout["page_bytes"]):
                errors.append(f"rank {rank} has invalid native cache layout: {name}")
        capture = [r for r in local if r.get("event") == "graph_capture_return"
                   and "FULL" in str(r.get("mode")) and r.get("descriptor")]
        replay = [r for r in local if r.get("event") == "graph_replay_launch_return"
                  and "FULL" in str(r.get("mode")) and r.get("descriptor")]
        if not capture:
            errors.append(f"rank {rank} graph capture did not return")
        if not replay:
            errors.append(f"rank {rank} graph replay launch did not return")
        if capture and replay and not ({r["descriptor"] for r in capture} & {r["descriptor"] for r in replay}):
            errors.append(f"rank {rank} graph replay has no matching captured descriptor")
        coverage[rank] = set(layouts)
        details[str(rank)] = {"cache_layers": sorted(layouts), "routed_layers": sorted(routed),
                             "capture_records": capture, "replay_records": replay}
    if coverage and any(layers != coverage[0] for layers in coverage.values()):
        errors.append("TP ranks disagree on FULL layer identities")
    return {"status": "passed" if not errors else "failed", "errors": errors,
            "ranks": details, "event_count": len(records),
            "graph_capture": "returned" if not any("capture" in e for e in errors) else "missing",
            "graph_replay": "launch_returned" if not any("replay" in e for e in errors) else "missing",
            "device_completion": "established separately by completed service requests, not telemetry alone"}


def parse_mtp_metrics(text):
    wanted = {"vllm:spec_decode_num_drafts", "vllm:spec_decode_num_draft_tokens", "vllm:spec_decode_num_accepted_tokens"}
    values = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([^\s{]+)(?:\{.*\})?\s+([^\s]+)(?:\s+[^\s]+)?", line)
        if not match:
            continue
        name = match[1].removesuffix("_total")
        if name not in wanted:
            continue
        value = float(match[2])
        if not math.isfinite(value) or value < 0:
            raise ServiceProbeError(f"invalid native MTP metric {name}={value}")
        values[name] = values.get(name, 0.0) + value
    return values


def evaluate_mtp_metrics(before, after):
    names = ("vllm:spec_decode_num_drafts", "vllm:spec_decode_num_draft_tokens", "vllm:spec_decode_num_accepted_tokens")
    if any(name not in after for name in names):
        return {"status": "failed", "reason": "native MTP counters are missing", "before": before, "after": after}
    deltas = {name: after[name] - before.get(name, 0) for name in names}
    drafts, proposed, accepted = (deltas[name] for name in names)
    passed = drafts > 0 and proposed > 0 and 0 <= accepted <= proposed
    return {"status": "passed" if passed else "failed", "before": before, "after": after, "delta": deltas,
            "before_missing": [name for name in names if name not in before],
            "acceptance_rate": accepted / proposed if proposed > 0 else None,
            "mean_acceptance_length": 1 + accepted / drafts if drafts > 0 else None,
            "scope": "native MTP execution observation, not a quality comparison"}


def run_service(config_path, *, output, log_dir, serve=False, command=None,
                tokenizer_factory=None, resource_reader=None):
    """Run real service requests; injectable boundaries are only for fault tests."""
    config_path, output, log_dir = Path(config_path).resolve(), Path(output).resolve(), Path(log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    started_wall = time.time()
    report = {"status": "running", "mode": "serve" if serve else "probe", "config": str(config_path),
              "requests": [], "server": {}, "device_completion": "not_run", "quality": "not_run", "performance": "not_run",
              "resource_release": "not_run", "resource_evidence": {}}
    atomic_json(output, report)
    before = None
    resource_reader = read_npu_resources if resource_reader is None else resource_reader
    tokenizer_factory = load_tokenizer if tokenizer_factory is None else tokenizer_factory
    old_signals = {}
    config = None

    def interrupt(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")

    try:
        config = json.loads(config_path.read_text())
        target_env(config)  # Fail before any NPU touch when selection is absent.
        report["devices"] = config["devices"]
        report["target_config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
        if config.get("tensor_parallel_size") != 4 or config.get("data_parallel_size") != 1:
            raise ServiceProbeError("this acceptance probe requires the configured TP4/DP1 target")
        if config.get("compilation_config", {}).get("cudagraph_mode") != "FULL_DECODE_ONLY":
            raise ServiceProbeError("target graph mode must remain FULL_DECODE_ONLY")
        speculative = config.get("speculative_config", {})
        if (speculative.get("num_speculative_tokens") != 3 or speculative.get("method") != "qwen3_5_mtp"
                or speculative.get("enforce_eager") is not True):
            raise ServiceProbeError("target MTP must retain three speculative tokens")
        duration = _positive_time(config, "phase_timeout_seconds", 1800)
        deadline = started + duration
        lengths = (PROBE_LENGTHS[0],) if serve else PROBE_LENGTHS
        if max(lengths) + MIN_OUTPUT_TOKENS > config["max_model_len"]:
            raise ServiceProbeError("configured model length cannot cover mandatory service requests")
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_signals[sig] = signal.signal(sig, interrupt)
        expected_layers, _, fingerprint = model_geometry(Path(config["model"]))
        report["model_fingerprint"] = fingerprint
        before = resource_reader(config, log_dir=log_dir / "resources", timeout=_remaining(deadline, 30))
        report["resource_evidence"]["before"] = before
        atomic_json(output, report)
        tokenizer = tokenizer_factory(config["model"])
        prompts = {length: exact_prompt(tokenizer, length) for length in lengths}
        managed_config = dict(config)
        managed_config["service_startup_timeout_seconds"] = _remaining(deadline,
            _positive_time(config, "service_startup_timeout_seconds", duration))
        with managed_server(managed_config, config_path, log_dir=log_dir, lifecycle=report["server"], command=command) as server:
            metrics_before_text = _http(server.base_url + "/metrics", timeout=_remaining(deadline, 10))
            (log_dir / "metrics_before.prom").write_text(metrics_before_text)
            metrics_before = parse_mtp_metrics(metrics_before_text)
            request_limit = _positive_time(config, "service_request_timeout_seconds", 300)
            for length in lengths:
                record = completion(server, config, tokenizer, length, label=f"serial-{length}",
                                    timeout=_remaining(deadline, request_limit), prompt_ids=prompts[length])
                report["requests"].append(record)
                atomic_json(output, report)
                _terminal(f"[oscar] service request complete prompt={length} generated={record['completion_tokens']}")
            if not serve:
                pool = ThreadPoolExecutor(max_workers=4)
                futures = []
                try:
                    for i, length in enumerate(PROBE_LENGTHS):
                        futures.append(pool.submit(completion, server, config, tokenizer, length,
                            label=f"mixed-{i}-{length}", timeout=_remaining(deadline, request_limit), prompt_ids=prompts[length]))
                    for future in as_completed(futures, timeout=_remaining(deadline, request_limit)):
                        report["requests"].append(future.result())
                        atomic_json(output, report)
                finally:
                    for future in futures:
                        future.cancel()
                    pool.shutdown(wait=False, cancel_futures=True)
            # Native counters are updated asynchronously; the bounded polling
            # observes their publication and never changes numerical gates.
            metric_deadline = min(deadline, time.monotonic() + _positive_time(config, "mtp_metrics_timeout_seconds", 15))
            while True:
                metrics_text = _http(server.base_url + "/metrics", timeout=_remaining(metric_deadline, 5))
                (log_dir / "metrics_after.prom").write_text(metrics_text)
                report["mtp"] = evaluate_mtp_metrics(metrics_before, parse_mtp_metrics(metrics_text))
                if report["mtp"]["status"] == "passed" or time.monotonic() + 0.2 >= metric_deadline:
                    break
                server.check_alive()
                time.sleep(0.2)
            records = read_telemetry(server.trace_dir)
            report["telemetry"] = evaluate_telemetry(records, tensor_parallel_size=4, expected_layers=expected_layers)
            atomic_json(log_dir / "telemetry_records.json", records)
            if report["mtp"]["status"] != "passed" or report["telemetry"]["status"] != "passed":
                raise ServiceProbeError("HTTP completed, but mandatory TP4/FULL/graph/MTP evidence did not pass")
            report["device_completion"] = "real_service_requests_completed"
            report["status"] = "serving" if serve else "requests_passed_cleanup_pending"
            atomic_json(output, report)
            if serve:
                print(f"[oscar] validated server ready {server.base_url}; report={output}", flush=True)
                while True:
                    server.check_alive()
                    time.sleep(0.2)
    except KeyboardInterrupt as error:
        report.update(status="stopped" if serve and report.get("status") == "serving" else "interrupted", error=str(error))
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        # Restore handlers before bounded cleanup observations so a second
        # Ctrl-C can interrupt a broken driver instead of being swallowed.
        for sig, handler in old_signals.items():
            signal.signal(sig, handler)
        if before is not None and config is not None:
            try:
                report["resource_evidence"]["release"] = wait_for_release(config, before, log_dir=log_dir / "resources",
                    timeout=_positive_time(config, "resource_release_timeout_seconds", 30),
                    tolerance_bytes=config.get("resource_release_tolerance_bytes", DEFAULT_RELEASE_TOLERANCE),
                    reader=resource_reader)
            except BaseException as error:
                report["resource_evidence"]["release"] = {"status": "failed", "reason": f"{type(error).__name__}: {error}"}
            report["resource_release"] = report["resource_evidence"]["release"]["status"]
            if report["resource_release"] != "passed":
                report["status"] = "failed"
                report.setdefault("error", "NPU free memory did not pass the bounded resource-release check")
        if report["server"].get("cleanup_complete") is False:
            report["status"] = "failed"
            report.setdefault("error", report["server"].get("cleanup_error", "owned service process group cleanup failed"))
        if report["status"] == "requests_passed_cleanup_pending":
            report["status"] = "passed"
        if report["status"] in {"failed", "interrupted"}:
            # The owned process-group ledger is captured before cleanup, so
            # departed workers' logs remain attributable after CANN flushes.
            owned = {os.getpid(), *report["server"].get("owned_pids", ())}
            attach_plog(report, started_at=started_wall, owned_pids=owned)
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(output, report)
        (log_dir / "probe_result.log").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if report.get("error"):
            print(f"[oscar] service detail: {report['error']}", file=sys.stderr, flush=True)
        print(f"[oscar] whole-service {report['status']} report={output}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/target.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/service_probe.json")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs/service-probe")
    parser.add_argument("--serve", action="store_true", help="keep the validated owned service running until interrupted")
    args = parser.parse_args()
    report = run_service(args.config, output=args.output, log_dir=args.log_dir, serve=args.serve)
    return 0 if report["status"] in {"passed", "stopped"} else 130 if report["status"] == "interrupted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
