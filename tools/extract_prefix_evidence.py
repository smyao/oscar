"""Extract source-linked runtime evidence; archive #70-73/#130-135/#152/#155/#157.

No cross-rank duration sum, no HTTP-200 completion inference, no conversion
from host launch time to device time. Truncated/interleaved JSON stays unknown.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from tools.analyze_service_log import ENGINE, SPEC

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
FAULT = re.compile(r"Traceback|\bFAILED\b|\bERROR\b|[Tt]rap|family=|5070(?:15|35)|deadline|timed out")


def scan_json(line):
    decoder = json.JSONDecoder()
    index = 0
    rows = []
    while index < len(line):
        start = line.find("{", index)
        if start < 0:
            break
        try:
            value, used = decoder.raw_decode(line[start:])
        except json.JSONDecodeError:
            # Do not promote a nested object inside a broken outer record.
            return rows, True
        if isinstance(value, dict):
            rows.append(value)
        index = start + used
    return rows, False


def json_objects(line):
    yield from scan_json(line)[0]


def stats(values):
    values = sorted(values)
    return {"count": len(values), "sum": sum(values), "p50": statistics.median(values),
            "p95": values[math.ceil(0.95 * len(values)) - 1], "max": values[-1]}


def extract(paths, output):
    output.mkdir(parents=True, exist_ok=False)
    sources, faults, samples, layouts, selections, tables = [], [], [], [], [], []
    counts, phases = Counter(), defaultdict(list)
    ignored_json_lines = 0
    records_count = 0
    with (output / "records.jsonl").open("w") as records:
        for path in paths:
            digest = hashlib.sha256()
            line_count = 0
            with path.open("rb") as source:
                for line_count, raw in enumerate(source, 1):
                    digest.update(raw)
                    line = ANSI.sub("", raw.decode("utf-8", errors="replace"))
                    where = {"file": str(path.resolve()), "line": line_count}
                    cells = [cell.strip() for cell in line.strip().strip("│").split("│")]
                    if line.lstrip().startswith("│") and cells[0] in {
                            "Benchmark Duration", "Total Requests", "Failed Requests", "Success Requests",
                            "Max Concurrency", "Total Input Tokens", "Total Generated Tokens",
                            "Output Token Throughput", "Input Token Throughput", "Request Throughput",
                            "Prefill Token Throughput", "E2EL", "TTFT", "TPOT", "ITL",
                            "InputTokens", "OutputTokens", "OutputTokenThroughput"}:
                        tables.append({**where, "metric": cells[0], "cells": cells[1:]})
                    if FAULT.search(line):
                        faults.append({**where, "text": line.rstrip()})
                    for regex, label in ((ENGINE, "engine_sample"), (SPEC, "spec_sample")):
                        match = regex.search(line)
                        if match:
                            samples.append({**where, "type": label, **match.groupdict()})
                    found = False
                    rows, incomplete = scan_json(line)
                    for row in rows:
                        tag = row.get("t") or row.get("event")
                        if tag is None and "PERF_RUNTIME_CONFIG" in line:
                            tag = "runtime_selection"
                            selections.append({**where, **row})
                        if tag is None:
                            continue
                        found = True
                        counts[str(tag)] += 1
                        records_count += 1
                        records.write(json.dumps({"source": where, "record": row}, ensure_ascii=False) + "\n")
                        if tag == "cache_layout":
                            layouts.append({**where, **row})
                        value = None
                        if tag == "oscar-timing":
                            measure, value = "host_launch_seconds", row.get("host_s")
                        elif tag == "oscar-device-event" and row.get("device_evidence") == "npu_event":
                            measure = ("synchronized_stream_ms" if row.get("debug_synchronization") is True
                                       and row.get("scope") == "synchronized_npu_stream_interval"
                                       else "npu_event_ms_unqualified")
                            value = row.get("device_ms")
                        elif tag == "oscar-passive-step" and row.get("status") == "measured":
                            measure, value = "passive_step_ms", row.get("step_device_ms")
                        if (type(value) in (int, float) and math.isfinite(value) and value >= 0
                                and not row.get("failed")):
                            key = (row.get("run_id") or str(path.resolve()), row.get("rank"), row.get("pid"), measure,
                                   row.get("phase") or row.get("phase_end") or "step",
                                   row.get("stage") or row.get("bucket"), row.get("draft_index"),
                                   row.get("tokens"), row.get("requests"), row.get("cv_operator"),
                                   row.get("capture_origin"), row.get("debug_synchronization"))
                            phases[key].append(float(value))
                    if incomplete or ("{" in line and not found):
                        ignored_json_lines += 1
            sources.append({"path": str(path.resolve()), "sha256": digest.hexdigest(), "lines": line_count})
    fields = ("run_id_or_source_file", "rank", "pid", "measurement", "phase", "stage", "draft_index",
              "tokens", "requests", "operator", "capture_origin", "debug_synchronization")
    summary = {"status": "extracted" if records_count or samples or faults or tables else "no_matching_evidence",
               "sources": sources, "record_counts": dict(counts), "records": records_count,
               "unrecognized_or_incomplete_json_lines": ignored_json_lines,
               "runtime_selections": selections, "cache_layouts": layouts,
               "sampled_metrics": samples, "benchmark_table_rows": tables, "faults": faults,
               "phase_groups": [{**dict(zip(fields, key)), **stats(values)} for key, values in phases.items()],
               "scope": ["phase sums are within one pid/rank/measurement only, not end-to-end wall",
                         "host_s is not kernel time; debug sync changes scheduling",
                         "APC log percentage is sampled, not per-request hit evidence",
                         "Spec counts remain unaggregated; window/cumulative semantics need confirmation",
                         "HTTP 200, graph launch return and heartbeat do not prove device/request completion",
                         "missing/truncated JSON is not repaired; original file and line remain authoritative"],
               "performance_acceptance": "not_established"}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="explicit files or run/trace directories")
    parser.add_argument("--output", required=True, type=Path, help="new output directory")
    args = parser.parse_args(argv)
    paths = set()
    for source in args.inputs:
        if source.is_file():
            paths.add(source.resolve())
        elif source.is_dir():
            paths.update(p.resolve() for p in source.rglob("*")
                         if p.is_file() and p.suffix in {".jsonl", ".log", ".txt"})
        else:
            parser.error(f"input does not exist: {source}")
    if not paths:
        parser.error("no input log files")
    summary = extract(sorted(paths), args.output.resolve())
    print(json.dumps({"status": summary["status"], "files": len(paths), "records": summary["records"],
                      "fault_lines": len(summary["faults"]), "summary": str(args.output / "summary.json")}))
    return 0 if summary["status"] == "extracted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
