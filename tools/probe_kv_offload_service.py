"""Real short TP4/MTP service offload acceptance; never a CPU substitute.

Archive #27/#37-49/#127/#128/#157/#163/#171: verify external-prefix and
active-request resume through actual HTTP generation and native preemption.
Archive #94/#95/#125/#129: live logs, bounded requests and owned-server cleanup.
Native cache.py documents num_gpu_blocks_override specifically for preemption
testing. Probe-only pool/token/model caps do not change production defaults.
No timer hooks, device profiling, throughput or model-quality claims.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import json
from math import lcm
import os
from pathlib import Path
import re
import signal
import traceback

from .phase import atomic_json
from .service_probe import (
    ServiceProbeError, managed_server, request_with_deadline, exact_prompt,
    load_tokenizer, _http, parse_gauges, parse_mtp_metrics, evaluate_mtp_metrics,
)
from .npu_resources import read_npu_resources, wait_for_release
from .target_cli import serve_argv, target_env, log_stats_disabled
from .probe_evidence import ProbeEvidence, evaluate_mtp_events, preempted_requests
from oscar_ascend.offload.config import offload_options

PREEMPTIONS = "vllm:num_preemptions_total"
GEOMETRY_MARKER = "KV_OFFLOAD_GEOMETRY "
POOL_STATE_MARKER = "KV_OFFLOAD_POOL_STATE "
OUTPUT_TOKENS = 128
CHUNK_BUDGET = 1024


def read_geometry(text: str, world_size: int = 4) -> dict:
    records = {}
    for line in text.splitlines():
        if GEOMETRY_MARKER not in line:
            continue
        value = json.loads(line.split(GEOMETRY_MARKER, 1)[1])
        rank = value.get("rank")
        if type(rank) is not int or not 0 <= rank < world_size:
            raise ServiceProbeError("invalid offload geometry rank")
        if rank in records and records[rank] != value:
            raise ServiceProbeError("offload pool geometry changed during the service")
        records[rank] = value
    if set(records) != set(range(world_size)):
        raise ServiceProbeError("missing TP worker offload geometry evidence")
    first = records[0]
    if (not first.get("groups") or type(first.get("num_blocks")) is not int
            or first.get("mamba_cache_mode") not in {"none", "align"}):
        raise ServiceProbeError("incomplete physical group geometry")
    if any(row["groups"] != first["groups"] or row["num_blocks"] != first["num_blocks"]
           or row.get("mamba_cache_mode") != first["mamba_cache_mode"]
           for row in records.values()):
        raise ServiceProbeError("TP workers report inconsistent physical groups")
    return first


def pressure_workload(geometry: dict, mode: str) -> dict:
    if mode not in {"none", "align"} or geometry.get("mamba_cache_mode") != mode:
        raise ServiceProbeError("observed mamba_cache_mode does not match the requested probe arm")
    groups = geometry["groups"]
    full = [g for g in groups if g["family"] == "oscar_full"]
    gdn = [g for g in groups if g["family"] == "gdn"]
    if not full or not gdn or len(full) + len(gdn) != len(groups):
        raise ServiceProbeError("pressure probe requires complete FULL/GDN groups")
    if [g.get("group_id") for g in groups] != list(range(len(groups))):
        raise ServiceProbeError("physical group IDs must preserve native group order")
    blocks = {g["block_size"] for g in full}
    if len(blocks) != 1 or any(type(b) is not int or b <= 128 for b in blocks):
        raise ServiceProbeError("pressure probe requires uniform physical FULL pages")
    block = next(iter(blocks))
    if any(g["num_speculative_blocks"] != 3 for g in gdn):
        raise ServiceProbeError("pressure probe must retain MTP3 GDN allocation")
    return {"mode": mode, "prompt_tokens": block - 64, "output_tokens": OUTPUT_TOKENS,
            "concurrency": 2, "full_block_tokens": block,
            "full_groups": len(full), "gdn_groups": len(gdn)}


def read_pool_states(text: str, geometry: dict) -> list[dict]:
    """Validate actual scheduler-owned pages; never infer GDN allocation counts."""
    result = []
    for line in text.splitlines():
        if POOL_STATE_MARKER not in line:
            continue
        value = json.loads(line.split(POOL_STATE_MARKER, 1)[1])
        groups = value.get("groups")
        integers = ("base", "scheduled", "end", "live_pages", "null_id")
        if (not isinstance(value.get("request"), str) or not value["request"]
                or any(type(value.get(name)) is not int for name in integers)
                or value["base"] < 0 or value["scheduled"] <= 0
                or value["end"] != value["base"] + value["scheduled"]
                or not 0 <= value["null_id"] < geometry["num_blocks"]
                or not isinstance(groups, list) or len(groups) != len(geometry["groups"])
                or any(not isinstance(group, list) for group in groups)
                or any(type(page) is not int or not 0 <= page < geometry["num_blocks"]
                       for group in groups for page in group)):
            raise ServiceProbeError("invalid native scheduler pool-state evidence")
        actual = {page for group in groups for page in group if page != value["null_id"]}
        if value["live_pages"] != len(actual):
            raise ServiceProbeError("pool-state live_pages disagrees with native physical IDs")
        result.append(value)
    if not result:
        raise ServiceProbeError("missing native scheduler pool-state evidence")
    if len({row["null_id"] for row in result}) != 1:
        raise ServiceProbeError("native shared null page changed during the baseline")
    return result


def pressure_plan(geometry: dict, mode: str, pool_states: list[dict]) -> dict:
    plan = pressure_workload(geometry, mode)
    full_ids = [g["group_id"] for g in geometry["groups"] if g["family"] == "oscar_full"]
    by_request = {}
    for row in pool_states:
        counts = [len(set(row["groups"][gid]) - {row["null_id"]}) for gid in full_ids]
        # MTP lookahead can allocate the next page before end reaches B.
        # Classify the actual allocation transition, not a token-only estimate.
        phase = "pre" if all(n == 1 for n in counts) else "post" if all(n >= 2 for n in counts) else None
        if phase is None:
            raise ServiceProbeError("baseline FULL groups did not cross their physical boundary together")
        record = by_request.setdefault(row["request"], {"pre": [], "post": []})
        record[phase].append(row)
    if len(by_request) != plan["concurrency"]:
        raise ServiceProbeError("pool evidence must cover both serial baseline requests")
    measured = {}
    for request, phases in by_request.items():
        if not phases["pre"] or not phases["post"]:
            raise ServiceProbeError("baseline did not observe both sides of the FULL allocation boundary")
        measured[request] = {
            "pre_cross_peak": max(row["live_pages"] for row in phases["pre"]),
            "post_cross_peak": max(row["live_pages"] for row in phases["post"]),
            "pre_steps": len(phases["pre"]), "post_steps": len(phases["post"]),
        }
    before = max(row["pre_cross_peak"] for row in measured.values())
    after = max(row["post_cross_peak"] for row in measured.values())
    if after <= before:
        raise ServiceProbeError("measured FULL crossing did not increase native owned-page demand")
    capacity = 1 + plan["concurrency"] * before
    if capacity >= geometry["num_blocks"]:
        raise ServiceProbeError("baseline pool is not larger than the measured pressure capacity")
    plan.update(num_gpu_blocks=capacity, pre_cross_peak=before, post_cross_peak=after,
                measured_requests=measured, baseline_pool_records=pool_states,
                both_crossed_pages=1 + plan["concurrency"] * after,
                scope="measured native owned-page pressure trigger; observed preemption mandatory")
    return plan


def event_records(text: str) -> list[dict]:
    result = []
    for line in text.splitlines():
        match = re.search(r"KV_OFFLOAD_(SWAP_SAVE|RESTORE|PREFIX_HIT|PREFIX_PUBLISHED)\b(.*)", line)
        if match:
            row = dict(re.findall(r"(\w+)=([^\s]+)", match[2]))
            row["event"] = match[1]
            result.append(row)
    return result


def require_swap_evidence(text: str, before: str, after: str, world_size=4, *, native_events=None) -> dict:
    if native_events is None:
        first, last = parse_gauges(before, {PREEMPTIONS}), parse_gauges(after, {PREEMPTIONS})
        if PREEMPTIONS not in first or PREEMPTIONS not in last or last[PREEMPTIONS] <= first[PREEMPTIONS]:
            raise ServiceProbeError("native preemption counter did not increase")
        evidence = {"native_preemptions_delta": last[PREEMPTIONS] - first[PREEMPTIONS]}
        requests = None
    else:
        requests = preempted_requests(native_events)
        if not requests:
            raise ServiceProbeError("no completed native preemption event")
        evidence = {"source": "native_scheduler_probe_events", "completed_preemptions": len(requests)}
    save, restored = {}, {}
    for row in event_records(text):
        if row["event"] != "SWAP_SAVE" and not (row["event"] == "RESTORE" and row.get("kind") == "swap"):
            continue
        if any(k not in row for k in ("request", "generation", "version", "tokens", "rank")):
            raise ServiceProbeError("swap log lacks request/generation/version/position/rank")
        key = tuple(row[k] for k in ("request", "generation", "version", "tokens"))
        target = save if row["event"] == "SWAP_SAVE" else restored
        target.setdefault(key, set()).add(int(row["rank"]))
    complete = [key for key, ranks in save.items()
                if ranks == set(range(world_size)) and restored.get(key) == ranks
                and (requests is None or key[0] in requests)]
    if not complete:
        raise ServiceProbeError("no same-generation committed swap completed save and restore on every TP rank")
    return {**evidence, "complete_joint_swaps": [list(key) for key in complete]}


def require_prefix_evidence(text: str, world_size=4) -> dict:
    rows = event_records(text)
    hits = [r for r in rows if r["event"] == "PREFIX_HIT"]
    restores = [r for r in rows if r["event"] == "RESTORE" and r.get("kind") == "prefix"]
    by_request = {}
    for row in restores:
        by_request.setdefault((row.get("request"), row.get("tokens")), set()).add(int(row["rank"]))
    joint = [key for key, ranks in by_request.items() if ranks == set(range(world_size))]
    if not hits or not joint:
        raise ServiceProbeError("local reset was not followed by an external joint-prefix restore")
    return {"external_hits": hits, "joint_restores": [list(key) for key in joint]}


def _metrics(server, directory: Path, name: str, timeout: float) -> str:
    value = _http(server.base_url + "/metrics", timeout=timeout)
    (directory / f"{name}.prom").write_text(value)
    return value


def _request(server, config, tokens, *, output_tokens: int, label: str, timeout: float):
    payload = {"model": config["served_model_name"], "prompt": tokens,
               "max_tokens": output_tokens, "min_tokens": output_tokens, "ignore_eos": True,
               "temperature": 0, "seed": 46774, "stream": False,
               "add_special_tokens": False, "return_token_ids": True}
    body = request_with_deadline(server, payload, label=label, timeout=timeout)
    choices, usage = body.get("choices"), body.get("usage", {})
    if not isinstance(choices, list) or len(choices) != 1:
        raise ServiceProbeError(f"{label}: invalid completion choices")
    ids = choices[0].get("token_ids")
    if (not isinstance(ids, list) or len(ids) != output_tokens
            or any(type(token) is not int or token < 0 for token in ids)
            or usage.get("prompt_tokens") != len(tokens)
            or usage.get("completion_tokens") != output_tokens):
        raise ServiceProbeError(f"{label}: missing exact generated-token/usage evidence")
    return {"label": label, "prompt_tokens": len(tokens), "generated_token_ids": ids,
            "completion_tokens": output_tokens, "response": body}


@contextmanager
def _probe_environment():
    key = "OSCAR_OFFLOAD_SERVICE_PROBE"
    previous = os.environ.get(key)
    os.environ[key] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


@contextmanager
def _owned(config, directory: Path, report: dict):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "target.json"
    atomic_json(path, config)
    report["argv"] = serve_argv(config)
    report["effective_config"] = str(path)
    report["server"] = {}
    before = read_npu_resources(config, log_dir=directory / "resources", timeout=30)
    report["resources_before"] = before
    try:
        with managed_server(config, path, log_dir=directory, lifecycle=report["server"],
                            native_probe_evidence=True) as server:
            yield server
    finally:
        report["resource_release"] = wait_for_release(config, before,
            log_dir=directory / "resources", timeout=config.get("resource_release_timeout_seconds", 30))
        if report["resource_release"]["status"] != "passed" or not report["server"].get("cleanup_complete"):
            raise ServiceProbeError("owned probe server failed resource release/cleanup")


def _mode_config(base: dict, mode: str) -> dict:
    config = copy.deepcopy(base)
    config.update(enable_prefix_caching=mode == "align", enable_chunked_prefill=True,
                  max_num_batched_tokens=CHUNK_BUDGET, use_native_token_budget=False,
                  max_num_seqs=2, max_model_len=16384)
    return config


def run_probe(config_path: Path, output_dir: Path, *, request_timeout=300.0) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"schema": 1, "status": "running", "probe": "real_service_kv_offload",
              "production_defaults_changed": False, "arms": {},
              "performance": "not_measured", "model_quality": "not_measured",
              "scope": "short deterministic token equality, native preemption and external-prefix restore"}
    result_path = output_dir / "result.json"
    atomic_json(result_path, report)
    try:
        from hashlib import sha256
        report["target_config_sha256"] = sha256(config_path.read_bytes()).hexdigest()
        base = json.loads(config_path.read_text())
        if offload_options(base) is None:
            raise ServiceProbeError("explicit kv_offload.enabled configuration is required")
        if (base.get("tensor_parallel_size") != 4 or base.get("data_parallel_size") != 1
                or base.get("speculative_config", {}).get("num_speculative_tokens") != 3
                or base.get("compilation_config", {}).get("cudagraph_mode") != "FULL_DECODE_ONLY"):
            raise ServiceProbeError("probe requires the configured TP4/MTP3/FULL_DECODE_ONLY target")
        target_env(base)
        tokenizer = load_tokenizer(base["model"])
        with _probe_environment():
            for mode in ("none", "align"):
                arm = report["arms"][mode] = {"status": "running", "baseline": {}, "pressure": {}}
                config = _mode_config(base, mode)
                directory = output_dir / mode
                with _owned(config, directory / "baseline", arm["baseline"]) as server:
                    geometry = read_geometry(server.log.read_text())
                    arm["geometry"] = geometry
                    workload = pressure_workload(geometry, mode)
                    if workload["prompt_tokens"] + OUTPUT_TOKENS > config["max_model_len"]:
                        raise ServiceProbeError("observed geometry cannot fit the bounded pressure workload")
                    prompts = [exact_prompt(tokenizer, workload["prompt_tokens"]) for _ in range(2)]
                    alternatives = tokenizer.encode("Independent alternate experiment.", add_special_tokens=False)
                    alternate = next((token for token in alternatives if token != prompts[0][0]), None)
                    if alternate is None:
                        raise ServiceProbeError("cannot form two distinct valid-token prompt prefixes")
                    prompts[1][0] = alternate
                    pool_offset = server.log.stat().st_size
                    baselines = [_request(server, config, tokens, output_tokens=OUTPUT_TOKENS,
                                  label=f"{mode}-serial-{i}", timeout=request_timeout)
                                 for i, tokens in enumerate(prompts)]
                    arm["baseline"]["requests"] = baselines
                    with server.log.open("rb") as stream:
                        stream.seek(pool_offset)
                        pool_states = read_pool_states(stream.read().decode(errors="replace"), geometry)
                    plan = arm["pressure_plan"] = pressure_plan(geometry, mode, pool_states)
                    if mode == "align":
                        joint = lcm(*(g["block_size"] for g in geometry["groups"]))
                        prompt_size = 2 * joint + 32
                        if prompt_size <= CHUNK_BUDGET or prompt_size + 16 > config["max_model_len"]:
                            raise ServiceProbeError("observed geometry cannot fit the bounded chunked-prefix probe")
                        prefix = exact_prompt(tokenizer, prompt_size)
                        publish_offset = server.log.stat().st_size
                        first = _request(server, config, prefix, output_tokens=16,
                                         label="prefix-first", timeout=request_timeout)
                        with server.log.open("rb") as stream:
                            stream.seek(publish_offset)
                            publications = [row for row in event_records(stream.read().decode(errors="replace"))
                                            if row["event"] == "PREFIX_PUBLISHED"
                                            and 0 < int(row.get("tokens", 0)) < prompt_size]
                        if not publications:
                            raise ServiceProbeError("no committed non-final chunk checkpoint was published")
                        # Native endpoint deliberately ignores its reset return
                        # value. Require later external-hit/TP-restore evidence;
                        # HTTP200 by itself is never counted as a successful reset.
                        _http(server.base_url + "/reset_prefix_cache?reset_external=false",
                              payload={}, timeout=30)
                        offset = server.log.stat().st_size
                        second = _request(server, config, prefix, output_tokens=16,
                                          label="prefix-external", timeout=request_timeout)
                        if first["generated_token_ids"] != second["generated_token_ids"]:
                            raise ServiceProbeError("external prefix restore changed generated token IDs")
                        with server.log.open("rb") as stream:
                            stream.seek(offset)
                            events = stream.read().decode(errors="replace")
                        arm["prefix"] = {"first": first, "external": second,
                                         "chunked_prompt_exceeds_budget": prompt_size > CHUNK_BUDGET,
                                         "committed_chunk_publications": publications,
                                         "evidence": require_prefix_evidence(events)}
                atomic_json(result_path, report)
                pressure = copy.deepcopy(config)
                # none-mode native Mamba geometry depends on max_model_len.
                # Keep every baseline setting except physical pool capacity.
                pressure["offload_probe_num_gpu_blocks"] = plan["num_gpu_blocks"]
                with _owned(pressure, directory / "pressure", arm["pressure"]) as server:
                    observed = read_geometry(server.log.read_text())
                    if (observed["groups"] != geometry["groups"]
                            or observed["mamba_cache_mode"] != mode):
                        raise ServiceProbeError("pressure geometry differs from the measured baseline")
                    if observed["num_blocks"] != plan["num_gpu_blocks"]:
                        raise ServiceProbeError("native pool did not apply the explicit probe capacity")
                    evidence = ProbeEvidence(server).start() if log_stats_disabled(pressure) else None
                    before = "" if evidence is not None else _metrics(server, directory / "pressure", "before", 30)
                    offset = server.log.stat().st_size
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        futures = [pool.submit(_request, server, pressure, tokens,
                                   output_tokens=OUTPUT_TOKENS, label=f"{mode}-pressure-{i}",
                                   timeout=request_timeout) for i, tokens in enumerate(prompts)]
                        actual = [future.result() for future in futures]
                    after = "" if evidence is not None else _metrics(server, directory / "pressure", "after", 30)
                    native_events = None
                    if evidence is not None:
                        evidence.stop()
                        native_events = evidence.read()
                        atomic_json(directory / "pressure" / "native_scheduler_events.json", native_events)
                    for baseline, resumed in zip(baselines, actual):
                        if baseline["generated_token_ids"] != resumed["generated_token_ids"]:
                            raise ServiceProbeError("preemption/resume changed generated token IDs")
                    with server.log.open("rb") as stream:
                        stream.seek(offset)
                        events = stream.read().decode(errors="replace")
                    arm["pressure"]["requests"] = actual
                    arm["pressure"]["swap_evidence"] = require_swap_evidence(events, before, after,
                                                                            native_events=native_events)
                    mtp = (evaluate_mtp_metrics(parse_mtp_metrics(before), parse_mtp_metrics(after))
                           if evidence is None else evaluate_mtp_events(native_events))
                    if mtp["status"] != "passed":
                        raise ServiceProbeError("native MTP execution was not observed under offload pressure")
                    arm["pressure"]["mtp"] = mtp
                arm["status"] = "passed"
                atomic_json(result_path, report)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        atomic_json(result_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/kv_offload_service"))
    parser.add_argument("--request-timeout", type=float, default=300)
    args = parser.parse_args(argv)
    if not 0 < args.request_timeout <= 600:
        parser.error("request timeout must be in (0,600]")
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    try:
        run_probe(args.config.resolve(), args.output_dir.resolve(), request_timeout=args.request_timeout)
    except BaseException:
        traceback.print_exc()
        print(f"KV OFFLOAD SERVICE FAILED: {args.output_dir / 'result.json'}", flush=True)
        return 1
    print(f"KV OFFLOAD SERVICE PASSED: {args.output_dir / 'result.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
