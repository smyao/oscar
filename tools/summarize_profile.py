"""Summarize real CANN Chrome traces without treating host launches as kernels.

Archive #70-73 and startup D.4: 725ms prepare was host time, whereas 6500ms
restore, 18.7ms FIA and 209ms stores were device time. Missing evidence stays
null/not_run; a fused INT2 name does not prove zero history-restoration bytes.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys

from tools.phase import atomic_json

PHASES = ("prepare", "rotate", "fia", "merge", "phase1_stores", "status_guard",
          "stage_restore", "phase0_store", "materialize", "dequant",
          "current_source_suppress", "current_slot_guard", "current_native_fia")
REQUIRED_PHASES = ("prepare", "rotate", "fia", "merge", "phase1_stores")
KERNEL_PHASES = (
    ("oscar_prepare_attention_tasks", "prepare"),
    ("oscar_rotate_clip_store", "phase1_stores"),
    ("oscar_store_int2", "phase1_stores"),
    ("oscar_attention_cv", "fia"),
    ("oscar_merge_lse", "merge"),
    ("oscar_rotate", "rotate"),
    ("oscar_status_guard", "status_guard"),
    ("fusedinferattentionscore", "fia"),
)
RESTORE_PATTERNS = ("full_dequant", "history_dequant", "dequant_history",
                    "full_history_restore", "inverse_rotation_history", "full_dequant_kv")


class ProfileEvidenceError(ValueError):
    pass


def _number(event, key):
    value = event.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ProfileEvidenceError(f"trace event has invalid {key}: {value!r}")
    if key == "dur" and value < 0:
        raise ProfileEvidenceError("negative trace duration")
    return float(value)


def _phase_marker(name):
    if not isinstance(name, str) or not name.startswith("oscar::"):
        return None
    pieces = name.split("::", 2)
    phase = pieces[1]
    if phase == "history_window":
        phase = "fia"
    return phase if phase in PHASES else None


def _canonical(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _kernel_phase(event):
    name = str(event.get("name", "")).lower()
    args = event.get("args", {})
    declared_name = " ".join(str(value).lower() for key, value in args.items()
                             if _canonical(key) in {"kernelname", "opname", "optype"})
    combined = name + " " + declared_name
    if any(pattern in combined for pattern in RESTORE_PATTERNS):
        return "dequant"
    for needle, phase in KERNEL_PHASES:
        if needle in combined.replace("_kernel", "") or needle in combined:
            return phase
    return None


def _is_device(event, device_pids):
    if event.get("ph") != "X":
        return False
    cat = str(event.get("cat", "")).lower()
    args = {_canonical(k): v for k, v in event.get("args", {}).items()}
    task_type = _canonical(args.get("tasktype", args.get("type", "")))
    known_task = task_type in {"aicore", "aicpu", "aivector", "aic", "aiv", "mix", "mixai"}
    device_category = cat in {"kernel", "kernels", "npu_kernel", "device_kernel", "aicore", "aicpu"}
    # Names alone do not distinguish a CPU launch wrapper from a device event.
    return known_task or (device_category and (event.get("pid") in device_pids or
                          "deviceid" in args or "npu" in cat))


def _correlations(event):
    return {( _canonical(key), str(value)) for key, value in event.get("args", {}).items()
            if _canonical(key) in {"correlationid", "correlation", "corrid", "externalid"}}


def _flow_key(event):
    identity = event.get("id", event.get("id2"))
    if identity is None:
        return None
    return (str(event.get("name", "")), str(event.get("cat", "")),
            json.dumps(identity, sort_keys=True))


def _union(intervals):
    """Sum busy time per hardware process, avoiding overlapping task double-counts."""
    groups = defaultdict(list)
    for hardware, start, stop in intervals:
        groups[str(hardware)].append((start, stop))
    result = 0.0
    for spans in groups.values():
        end = -math.inf
        for start, stop in sorted(spans):
            result += max(0.0, stop - max(start, end))
            end = max(end, stop)
    return result


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lo = math.floor(index)
    hi = math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def _interval_index(items):
    groups = defaultdict(list)
    for item in items:
        event = item["event"]
        groups[(event.get("pid"), event.get("tid"))].append(item)
    index = {}
    for key, entries in groups.items():
        entries.sort(key=lambda x: x["start"])
        maximum = -math.inf
        ends = []
        for entry in entries:
            maximum = max(maximum, entry["stop"])
            ends.append(maximum)
        index[key] = ([x["start"] for x in entries], ends, entries)
    return index


def _containing(index, event, timestamp):
    indexed = index.get((event.get("pid"), event.get("tid")))
    if indexed is None:
        return []
    starts, maximum_ends, entries = indexed
    i = bisect_right(starts, timestamp) - 1
    found = []
    while i >= 0 and maximum_ends[i] >= timestamp:
        if entries[i]["stop"] >= timestamp:
            found.append(entries[i])
        i -= 1
    return found


def summarize_trace(document, *, source="<memory>", time_unit="us", rank=None):
    events = document.get("traceEvents") if isinstance(document, dict) else document
    if not isinstance(events, list):
        raise ProfileEvidenceError("expected CANN Chrome JSON traceEvents list")
    if time_unit not in {"us", "ns"}:
        raise ProfileEvidenceError("only explicit us/ns trace clock units are supported")
    factor = 1e-3 if time_unit == "us" else 1e-6
    device_pids = set()
    for event in events:
        if not isinstance(event, dict):
            raise ProfileEvidenceError("trace events must be objects")
        if event.get("ph") == "M" and event.get("name") == "process_name":
            name = str(event.get("args", {}).get("name", "")).lower()
            if "ascend" in name or "npu" in name:
                device_pids.add(event.get("pid"))
    markers = []
    kernels = []
    for index, event in enumerate(events):
        if event.get("ph") != "X":
            continue
        start, duration = _number(event, "ts"), _number(event, "dur")
        item = {"index": index, "event": event, "start": start, "stop": start + duration,
                "duration": duration, "phase": None, "attribution": None}
        marker = _phase_marker(event.get("name"))
        if marker:
            item["phase"] = marker
            markers.append(item)
        if _is_device(event, device_pids):
            item["phase"] = _kernel_phase(event)
            item["attribution"] = "device_kernel_name" if item["phase"] else None
            kernels.append(item)

    marker_index = _interval_index(markers)
    kernel_index = _interval_index(kernels)

    def containing_marker(event, timestamp):
        choices = _containing(marker_index, event, timestamp)
        return min(choices, key=lambda x: x["duration"])["phase"] if choices else None

    correlations = defaultdict(set)
    for event in events:
        if event.get("ph") != "X" or _is_device(event, device_pids):
            continue
        phase = _phase_marker(event.get("name")) or containing_marker(event, _number(event, "ts"))
        if phase:
            for key in _correlations(event):
                correlations[key].add(phase)
    flows = defaultdict(list)
    for event in events:
        if event.get("ph") in {"s", "f"} and _flow_key(event) is not None:
            flows[_flow_key(event)].append(event)
    for endpoints in flows.values():
        sources = [x for x in endpoints if x["ph"] == "s"]
        destinations = [x for x in endpoints if x["ph"] == "f"]
        for destination in destinations:
            phases = {containing_marker(x, _number(x, "ts")) for x in sources}
            phases.discard(None)
            if len(phases) != 1:
                continue
            stamp = _number(destination, "ts")
            for kernel in _containing(kernel_index, destination, stamp):
                if kernel["phase"] is None:
                    kernel["phase"] = next(iter(phases))
                    kernel["attribution"] = "explicit_trace_flow"
    for kernel in kernels:
        if kernel["phase"] is None:
            choices = set()
            for key in _correlations(kernel["event"]):
                choices.update(correlations[key])
            if len(choices) == 1:
                kernel["phase"] = next(iter(choices))
                kernel["attribution"] = "explicit_correlation_id"

    phase_details = {}
    for phase in PHASES:
        selected = [x for x in kernels if x["phase"] == phase]
        host = [x["duration"] * factor for x in markers if x["phase"] == phase]
        durations = [x["duration"] * factor for x in selected]
        phase_details[phase] = {
            "device_ms": _union([(x["event"].get("pid"), x["start"], x["stop"]) for x in selected]) * factor if selected else None,
            "device_kernel_sum_ms": sum(durations) if selected else None,
            "device_kernel_count": len(selected),
            "device_kernel_p50_ms": _percentile(durations, 0.5),
            "device_kernel_p95_ms": _percentile(durations, 0.95),
            "host_scope_sum_ms": sum(host) if host else None,
            "host_scope_count": len(host),
            "attribution": sorted({x["attribution"] for x in selected}),
            "kernel_names": sorted({str(x["event"].get("name")) for x in selected}),
        }
    unattributed = [str(x["event"].get("name")) for x in kernels
                    if x["phase"] is None and "oscar" in str(x["event"].get("name", "")).lower()]
    forbidden = phase_details["dequant"]["kernel_names"]
    test_fixture = isinstance(document, dict) and document.get("oscar_test_fixture") is True
    state = "test_fixture" if test_fixture else ("observed" if kernels else "not_run")
    return {"t": "oscar-timing", "source": str(source), "rank": rank,
            "evidence_kind": "npu_profiler_trace" if kernels and not test_fixture else state,
            "status": state, "time_unit": time_unit,
            "summary": {"device_ms": {key: value["device_ms"] for key, value in phase_details.items()},
                        "total_device_busy_ms": _union([(x["event"].get("pid"), x["start"], x["stop"]) for x in kernels]) * factor if kernels else None,
                        "npu_kernel_count": len(kernels)},
            "phases": phase_details,
            "missing_required_phases": [p for p in REQUIRED_PHASES if phase_details[p]["device_ms"] is None],
            "unattributed_oscar_kernels": sorted(set(unattributed)),
            "history_restore": {"status": "forbidden_kernel_observed" if forbidden else ("no_standalone_restore_kernel_observed" if kernels else "unverified"),
                                "full_history_restore_bytes": None,
                                "bytes_evidence": "not_measured_by_chrome_duration_trace",
                                "observed_kernel_names": forbidden},
            "graph_capture": "not_inferred_from_trace", "graph_replay": "not_inferred_from_trace",
            "performance_acceptance": "requires_same_config_baseline_and_byte_traffic_evidence"}


def read_trace(path, *, time_unit="us", rank=None):
    path = Path(path).resolve()
    payload = path.read_bytes()
    decoded = gzip.decompress(payload) if path.suffix == ".gz" else payload
    try:
        document = json.loads(decoded)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProfileEvidenceError(f"invalid JSON trace {path}: {exc}") from exc
    report = summarize_trace(document, source=str(path), time_unit=time_unit, rank=rank)
    report["source_sha256"] = hashlib.sha256(payload).hexdigest()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="*", type=Path, help="explicit CANN trace_view.json or .json.gz files")
    parser.add_argument("--output", type=Path, default=Path("reports/oscar_profile.json"))
    parser.add_argument("--jsonl", type=Path, help="oscar-timing summary lines, one per trace")
    parser.add_argument("--time-unit", choices=("us", "ns"), default="us")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    if not args.trace:
        report = {"status": "not_run", "reason": "no real NPU trace supplied", "traces": [],
                  "device_ms": None, "full_history_restore_bytes": None}
        atomic_json(args.output, report)
        print(json.dumps(report, sort_keys=True))
        return 2
    reports = [read_trace(path, time_unit=args.time_unit, rank=args.rank) for path in args.trace]
    complete = all(x["status"] == "observed" and not x["missing_required_phases"] and
                   not x["unattributed_oscar_kernels"] and not x["history_restore"]["observed_kernel_names"] for x in reports)
    report = {"status": "observed" if any(x["status"] == "observed" for x in reports) else "not_run",
              "phase_attribution_complete": complete, "traces": reports,
              "performance_acceptance": "not_established_by_timing_alone"}
    atomic_json(args.output, report)
    lines = [json.dumps({"t": x["t"], "source": x["source"], "rank": x["rank"],
                         "status": x["status"], "summary": x["summary"]}, sort_keys=True, allow_nan=False) for x in reports]
    if args.jsonl:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.jsonl.write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)
    if args.require_complete and not complete:
        return 1
    return 0 if report["status"] == "observed" else 2


if __name__ == "__main__":
    sys.exit(main())
