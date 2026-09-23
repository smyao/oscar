# Archive #70-73/#132/#134/#138/#139: production attention_progress is a
# throttled host heartbeat, not a device completion or an operator timer.
"""Summarize host heartbeat gaps without claiming device or kernel time."""
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


def summarize_progress(directory, windows=None) -> dict:
    """Group inter-heartbeat gaps by (arm window, requests, KV bucket).

    Labels come from the first heartbeat; work in the gap may have a different
    state. A gap can include GDN, scheduling, graph waits and host IO. Every TP
    worker contributes its own wall seconds, so their sum is rank-seconds and
    must not be compared directly to the client makespan. Windows split arms;
    cross-window gaps are kept separate rather than charged to either arm.
    """
    rows = []
    for path in sorted(Path(directory).glob("worker-*.jsonl")):
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"malformed progress record {path}:{line_no}") from error
            if record.get("event") == "attention_progress" and isinstance(record.get("wall_time"), (int, float)):
                rows.append(record)
    arms = []
    if windows:
        for index, window in enumerate(windows):
            start, end = float(window[0]), float(window[1])
            if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                raise ValueError("arm windows must be finite increasing epoch pairs")
            arms.append((str(window[2]) if len(window) > 2 else f"arm{index}", start, end))
    buckets = {}
    per_pid = {}
    for record in rows:
        per_pid.setdefault(record.get("pid"), []).append(record)

    def window_of(wall):
        for name, start, end in arms:
            if start <= wall < end:
                return name
        return "outside_arms"

    for pid_rows in per_pid.values():
        pid_rows.sort(key=lambda row: row["wall_time"])
        for previous, current in zip(pid_rows, pid_rows[1:]):
            delta = current["wall_time"] - previous["wall_time"]
            if delta < 0:
                raise ValueError("progress records are not monotonic in wall_time")
            # Inter-arm gaps belong to neither arm; only same-window deltas
            # count toward an arm's residence statistics.
            label = window_of(previous["wall_time"])
            if label != window_of(current["wall_time"]):
                label = "between_arms"
            kv_raw = previous.get("max_seq_len")
            kv = int(round(kv_raw / 16384.0)) * 16 if isinstance(kv_raw, (int, float)) and math.isfinite(kv_raw) else None
            key = (label, previous.get("requests"), previous.get("tokens"), kv)
            buckets.setdefault(key, []).append((previous.get("pid"), delta))
    bucket_list = []
    for (label, requests, tokens, kv), samples in buckets.items():
        values = [delta for _, delta in samples]
        ranks = {pid for pid, _ in samples}
        bucket_list.append({"arm": label, "requests": requests, "tokens": tokens,
                            "kv_bucket_k": kv, "count": len(values),
                            "wall_s_sum": sum(values),
                            "rank_count": len(ranks),
                            "wall_s_mean_per_rank": sum(values) / len(ranks),
                            "wall_s_p50": _percentile(values, 0.5),
                            "wall_s_p95": _percentile(values, 0.95),
                            "wall_s_max": max(values)})
    bucket_list.sort(key=lambda row: (row["arm"], -(row["wall_s_sum"] or 0)))
    return {"t": "oscar-progress-summary", "source": str(Path(directory).resolve()),
            "status": "observed" if bucket_list else "not_run", "buckets": bucket_list,
            "scope": "throttled host attention_progress gaps; includes unobserved work, "
                     "not operator or NPU device latency",
            "wall_s_sum_unit": "TP rank-seconds; divide by rank_count for an approximate per-rank gap sum",
            "synchronization_inserted": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--windows", type=Path, help="JSON list of [start_epoch, end_epoch, label]")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    windows = json.loads(args.windows.read_text()) if args.windows else None
    report = summarize_progress(args.trace_dir, windows)
    if args.output is not None:
        atomic_json(args.output, report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "observed" else 2


if __name__ == "__main__":
    sys.exit(main())
