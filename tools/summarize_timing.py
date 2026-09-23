# Archive #70-73/D.4: synchronized phase walls are this stack's reliable
# device-time evidence; the native HTTP profiler window segfaulted all four
# workers on the target (#133) and is not used. Sync values attribute device
# time per phase; they are not native-performance numbers.
"""Aggregate oscar-debug sync checkpoints into per-phase device evidence."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from tools.phase import atomic_json


def _percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def summarize_timing(directory) -> dict:
    records = []
    for path in sorted(Path(directory).glob("timing-*.jsonl")):
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"malformed timing record {path}:{line_no}") from error
    per_pid = {}
    for record in records:
        if record.get("t") == "oscar-debug" and isinstance(record.get("wall_time"), (int, float)):
            per_pid.setdefault(record.get("pid"), []).append(record)
    device = {}
    buckets = {}
    unpaired = 0
    for rows in per_pid.values():
        rows.sort(key=lambda row: row["wall_time"])
        pending = {}
        for row in rows:
            key = (row.get("phase"), row.get("layer"), row.get("tokens"))
            state = row.get("state")
            if state == "waiting_for_device":
                pending[key] = (row["wall_time"], row.get("tokens"), row.get("requests"), row.get("max_seq_len"))
            elif state == "device_completed" and key in pending:
                wall, tokens, requests, max_seq = pending.pop(key)
                value = row["wall_time"] - wall
                device.setdefault(row.get("phase"), []).append(value)
                if isinstance(max_seq, (int, float)) and math.isfinite(max_seq):
                    # Coarse KV buckets (16K granularity) attribute the fia cost
                    # curve without per-request flooding; num_reqs separates the
                    # single- and multi-concurrency ladder arms.
                    kv = int(round(max_seq / 16384.0)) * 16
                    buckets.setdefault((row.get("phase"), tokens, requests, kv), []).append(value)
        for key in pending:
            device.setdefault(key[0], [])
        unpaired += len(pending)
    host = {}
    for record in records:
        if record.get("t") == "oscar-timing" and isinstance(record.get("host_s"), (int, float)):
            host.setdefault(record.get("phase_end"), []).append(record["host_s"])
    phases = {}
    for name in sorted(set(device) | set(host)):
        values = device.get(name, [])
        hosts = host.get(name, [])
        phases[name] = {
            "device_count": len(values),
            "device_s_sum": sum(values) if values else None,
            "device_s_p50": _percentile(values, 0.5) if values else None,
            "device_s_p95": _percentile(values, 0.95) if values else None,
            "device_s_max": max(values) if values else None,
            "host_s_sum": sum(hosts) if hosts else None,
            "host_count": len(hosts),
        }
    observed = any(row["device_count"] for row in phases.values())
    bucket_list = []
    for (phase, tokens, requests, kv), values in buckets.items():
        bucket_list.append({"phase": phase, "tokens": tokens, "requests": requests,
                            "kv_bucket_k": kv,
                            "device_count": len(values), "device_s_sum": sum(values),
                            "device_s_p95": _percentile(values, 0.95),
                            "device_s_max": max(values)})
    bucket_list.sort(key=lambda row: -(row["device_s_sum"] or 0))
    return {"t": "oscar-timing-summary", "source": str(Path(directory).resolve()),
            "status": "observed" if observed else "not_run", "phases": phases,
            "buckets": bucket_list,
            "unpaired_waiting": unpaired,
            "scope": "debug-sync phase walls attribute device time per phase; "
                     "synchronized absolutes are not native-performance numbers"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = summarize_timing(args.trace_dir)
    if args.output is not None:
        atomic_json(args.output, report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "observed" else 2


if __name__ == "__main__":
    sys.exit(main())
