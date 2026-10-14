"""Archive #94/#95/#130-134/#156/#157: compact existing performance evidence.

Read logs only; never starts a benchmark, initializes an NPU or changes a
running service. Counts describe log samples, not engine steps or completions.
Default discovery is restricted to this project's logs/<run>/status.json.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

from tools.collect_startup_evidence import (
    ANSI, CONFIG_KEYS, ENV_KEYS, KEY, ROOT, STARTUP, clipped, command_snapshot,
    selected_json, snapshot,
)
from tools.serving_variants import FEATURE_FLAGS, RESEARCH_FEATURE_FLAGS

SIGNALS = {
    "engine": re.compile(r"Avg (?:prompt|generation) throughput|Running:.*Waiting:"),
    "spec": re.compile(r"SpecDecoding metrics:|acceptance (?:rate|length)", re.I),
    "capacity": re.compile(r"max_num_batched_tokens|GPU KV cache size|Available KV cache|"
                           r"Maximum concurrency|preempt|recomput", re.I),
    "benchmark": re.compile(r"Benchmark Duration|Total Requests|Success Requests|Failed Requests|"
                            r"Total Input Tokens|Total Generated Tokens|Max Concurrency|"
                            r"Throughput|\bTTFT\b|\bTPOT\b|\bITL\b|\bE2EL\b"),
    "startup": STARTUP,
    "runtime_or_error": KEY,
}
EVENT_FIELDS = (
    "event", "t", "pid", "rank", "wall_time", "ts_unix", "layer", "tokens", "requests",
    "max_query_len", "max_seq_len", "cv_operator", "current_source", "capture_origin",
    "dummy_origin", "draft_index", "source_splits", "computed_tokens", "proposed_tokens",
    "scheduled_tokens", "checkpoint_end", "joint_checkpoint", "full_page_tokens",
    "state_block_tokens", "hash_block_tokens", "physical_blocks", "physical_block_tokens",
    "page_bytes", "snapshot_bytes_per_page", "scratch_bytes", "cache_format", "descriptor",
    "mode", "device_completion", "phase", "phase_end", "state", "host_s", "device_ms",
    "debug_synchronization", "status", "step_device_ms", "scope", "device_evidence",
)


class Samples:
    def __init__(self):
        self.count = 0
        self.first = []
        self.last = deque(maxlen=8)

    def add(self, row):
        self.count += 1
        if len(self.first) < 8:
            self.first.append(row)
        else:
            self.last.append(row)

    def report(self):
        return {"matched": self.count, "omitted": self.count - len(self.first) - len(self.last),
                "first_then_last": self.first + list(self.last)}


def scan(path):
    samples = {name: Samples() for name in SIGNALS}
    events = {}
    counts = Counter()
    malformed = oversized = lines = size = 0
    digest = hashlib.sha256()
    with path.open("rb") as source:
        # A live service may append forever; read only the size at open time.
        remaining = os.fstat(source.fileno()).st_size
        while remaining:
            raw = source.readline(min(remaining, 1024 * 1024))
            if not raw:
                break
            remaining -= len(raw)
            size += len(raw)
            digest.update(raw)
            prefix = raw
            large = len(raw) == 1024 * 1024 and not raw.endswith(b"\n")
            while large and remaining and raw and not raw.endswith(b"\n"):
                raw = source.readline(min(remaining, 1024 * 1024))
                remaining -= len(raw)
                size += len(raw)
                digest.update(raw)
            lines += 1
            oversized += int(large)
            line = ANSI.sub("", prefix.decode("utf-8", errors="replace")).strip()
            if line.startswith("{"):
                try:
                    row = json.loads(line)
                except ValueError:
                    malformed += 1
                    continue
                if not isinstance(row, dict):
                    continue
                tag = row.get("event") or row.get("t")
                if not isinstance(tag, str):
                    continue
                # Bound unknown event cardinality as well as retained samples.
                tag = tag if tag in events or len(events) < 24 else "other_events"
                counts[tag] += 1
                events.setdefault(tag, Samples()).add({"line": lines, **{
                    k: (clipped(v, 500) if isinstance(v, str) else v)
                    for k, v in row.items() if k in EVENT_FIELDS
                    and (v is None or type(v) in (str, int, float, bool))}})
                continue
            for name, pattern in SIGNALS.items():
                if pattern.search(line):
                    samples[name].add({"line": lines, "text": clipped(line, 1800)})
    return {"path": str(path.resolve()), "bytes_scanned": size, "lines": lines,
            "sha256_of_scanned_bytes": digest.hexdigest(), "oversized_lines_prefix_only": oversized,
            "malformed_json_lines": malformed, "event_sample_counts": dict(counts),
            "events": {k: v.report() for k, v in events.items()},
            "log_signals": {k: v.report() for k, v in samples.items() if v.count}}


def latest_run(root):
    candidates = [p for p in root.glob("*/status.json") if p.is_file()]
    if not candidates:
        raise ValueError(f"no logs/<run>/status.json below {root}; use --run-dir or --log")
    return max(candidates, key=lambda p: p.stat().st_mtime_ns).parent.resolve()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="explicit run; default: newest project logs/*/status.json")
    parser.add_argument("--trace-dir", type=Path, help="explicit OSCAR_TRACE_DIR override used by that service")
    parser.add_argument("--log", type=Path, action="append", default=[], help="existing server/client log, repeatable; max 8")
    parser.add_argument("--snapshot", action="store_true", help="read current npu-smi, worker env and stdout links")
    parser.add_argument("--output", type=Path, required=True, help="new compact text file; never overwrites")
    args = parser.parse_args(argv)
    if len(args.log) > 8 or args.output.exists():
        parser.error("at most 8 explicit logs; output must not exist")
    run = args.run_dir.resolve() if args.run_dir else None
    if run is None and not args.log:
        try:
            run = latest_run(ROOT / "logs")
        except (ValueError, OSError) as error:
            parser.error(str(error))
    paths = set(p.resolve() for p in args.log)
    trace = args.trace_dir.resolve() if args.trace_dir else (run / "trace" if run else None)
    if trace:
        paths.update(p.resolve() for p in trace.glob("worker-*.jsonl") if p.is_file())
    if run:
        paths.update((run / name).resolve() for name in ("console.log", "serve.log", "service.log")
                     if (run / name).is_file())
    # Explicit errors instead of silently selecting a subset of many runs.
    if len(paths) > 24:
        parser.error("more than 24 input files; select a single run or explicit --log files")
    selected = list(CONFIG_KEYS) + ["model", "max_num_batched_tokens", "scheduler_cls"]
    selected += list(FEATURE_FLAGS.values()) + list(RESEARCH_FEATURE_FLAGS.values())
    header = {"captured_at_utc": datetime.now(timezone.utc).isoformat(),
              "selected_run": str(run) if run else None, "trace_dir": str(trace) if trace else None,
              "console_files_found": [str(p) for p in paths if p.suffix == ".log"],
              "input_files": sorted(map(str, paths)),
              "collector_shell_env_not_worker_env": {k: os.environ[k] for k in ENV_KEYS if k in os.environ},
              "config": selected_json(run / "effective-target.json" if run else ROOT / "configs/target.json", selected)}
    sections = ["OSCAR PERFORMANCE EVIDENCE v1", json.dumps(header, ensure_ascii=False, indent=2),
                "Newest directory is a candidate, NOT proof of the live service. Verify selected_run.\n"
                "Missing console means server throughput/MTP logs may be unavailable; worker traces are not a replacement.\n"
                "Events are throttled/deduplicated samples, NOT total step counts or phase time shares.\n"
                "No cross-rank sums; host_s is not kernel duration. Capture/replay return is not completion.\n"
                "Current environment/occupancy cannot prove the completed benchmark's environment/occupancy."]
    if run:
        sections.append(json.dumps(selected_json(run / "status.json", (
            "status", "variant", "effective_config", "target_devices", "port", "optimizations")), indent=2))
    failed = not paths
    for path in sorted(paths):
        try:
            report = scan(path)
        except OSError as error:
            failed = True
            report = {"path": str(path), "read_error": str(error)}
        sections.append(clipped(json.dumps(report, ensure_ascii=False, indent=2), 22000))
    if args.snapshot:
        live = snapshot(set())
        live["checkout_head_now_not_benchmark_sha"] = command_snapshot(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
        sections.insert(3, "LIVE SNAPSHOT\n" + clipped(json.dumps(live, ensure_ascii=False, indent=2), 40000))
    content = clipped("\n\n".join(sections) + "\n", 256 * 1024)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as destination:
        destination.write(content)
    print(f"[oscar] selected run={run}; trace={trace}", flush=True)
    print(f"[oscar] existing files={len(paths)}; console files={header['console_files_found']}", flush=True)
    print(f"[oscar] performance evidence={args.output.resolve()} ({len(content.encode())} bytes); send this file only", flush=True)
    if not paths:
        print("[oscar] NO LOG FILES FOUND; metadata/snapshot only, performance cause unknown", flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
