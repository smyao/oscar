"""Bounded 32K diagnostic summary; archive #70-73 separates host/device/wall.

No torch import, NPU access, or inferred roofline verdict. CANN Task Duration is
actual task time; aicore_time is a theoretical per-block metric, never wall.
PMU column meanings: CANN Profiling atlasprofiling_16_0067 (op_summary), and
atlasprofiling_16_0033 (PipeUtilization/Memory). Original names and units survive.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import re
from statistics import mean


SCHEMA_VERSION = 1
COUNTER_SOURCE = "https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta2/devaids/Profiling/atlasprofiling_16_0067.html"
COUNTER_CONFIG_SOURCE = "https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/80RC3alpha003/devaids/auxiliarydevtool/atlasprofiling_16_0033.html"
_UNIT_MS = {"s": 1000.0, "ms": 1.0, "us": .001, "ns": .000001}


def _canon(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _num(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(str(value).strip().replace(",", "").removesuffix("%"))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _unit(name):
    match = re.search(r"[\[(]([^\])]+)[\])]", str(name))
    if match:
        return match.group(1).replace("μ", "u").replace("µ", "u").strip()
    if _canon(name).endswith("ratio"):
        return "ratio"
    if "cycles" in name.lower():
        return "cycles"
    return "unspecified"


def _base(name):
    return _canon(re.sub(r"[\[(][^\])]+[\])]", "", str(name)))


def _field(row, names):
    wanted = {_canon(x) for x in names}
    return next((v for k, v in row.items() if _base(k) in wanted), None)


def _time_ms(row, names):
    wanted = {_canon(x) for x in names}
    for key, value in row.items():
        if _base(key) in wanted:
            number, unit = _num(value), _unit(key).lower()
            if number is not None and unit in _UNIT_MS:
                return number * _UNIT_MS[unit]
    return None


def _counter(name):
    value = _canon(name)
    return _base(name).endswith("ratio") or any(x in value for x in ("utilization", "bandwidth", "mte", "mainmem", "hbm", "ddr", "l2", "l1read", "l1write", "ubread", "ubwrite", "mactime", "vectime", "cubetime", "vector", "cycles", "fops", "flops", "memorybound"))


def classify_kernel(name, op_type=""):
    """Name classification is a category, never a claim about hardware bounds."""
    value = _canon(str(name) + " " + str(op_type))
    patterns = (
        (("storealignedint2", "storeint2", "rotateclipstore"), "oscar.store_quant"),
        (("copyvalidatebf16query",), "oscar.query_validate_copy"),
        (("attentioncvaligned", "attentioncv"), "oscar.history_window_attention"),
        (("mergelse", "mergepartials"), "oscar.lse_merge"),
        (("prepareattentiontasks", "oscarprepare"), "oscar.prepare"),
        (("oscarrotate",), "oscar.rotation"),
        (("statusguard", "currentslotguard", "currentsourcesuppress"), "oscar.guards"),
        (("fullhistoryrestore", "historydequant", "fulldequant"), "oscar.history_materialization"),
        (("gateddelta", "gateddeltanet", "causalconv1d", "chunkgdn", "fusedrecurrent"), "native.GDN"),
        (("fusedinferattention", "flashattention", "promptflashattention", "increflashattention", "pagedattention"), "native.FULL_attention"),
        (("hccl", "allreduce", "allgather", "reducescatter", "broadcast"), "communication.HCCL"),
        (("matmul", "batchmatmul", "gemm", "quantbatchmatmul"), "native.GEMM_unattributed"),
        (("memcpy", "memset", "transdata", "datacopy", "copy"), "copy_layout"),
        (("layernorm", "rmsnorm", "rotary", "rope"), "native.norm_rope"),
    )
    return next((phase for needles, phase in patterns if any(n in value for n in needles)), "unknown")


def _read_json(path, missing):
    try:
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8-sig") as handle:
            return json.load(handle)
    except (OSError, ValueError, UnicodeError) as exc:
        missing.append(f"invalid_or_missing_json:{path}:{type(exc).__name__}")
        return None


def _resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def _union(intervals):
    end, result = -math.inf, 0.0
    for start, stop in sorted(intervals):
        if stop >= start:
            result += max(0.0, stop - max(end, start))
            end = max(end, stop)
    return result


def _exclusive_scopes(scopes):
    """Partition overlaps by innermost scope; parent+child never double count."""
    valid = []
    for index, scope in enumerate(scopes):
        start, end = _num(scope.get("start_ns")), _num(scope.get("end_ns"))
        if start is not None and end is not None and end >= start:
            valid.append((start, end, str(scope.get("phase", "unknown")), index))
    edges = sorted({x for start, end, _, _ in valid for x in (start, end)})
    output = defaultdict(float)
    for start, end in zip(edges, edges[1:]):
        active = [s for s in valid if s[0] <= start and s[1] >= end]
        if active:
            winner = min(active, key=lambda s: (s[1] - s[0], -s[3]))
            output[winner[2]] += (end - start) / 1e6
    return dict(output)


def parse_csv(path, missing=None):
    """Parse per-task CSV only; never use aicore_time as actual duration."""
    missing = missing if missing is not None else []
    result = []
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                row = {str(k).strip(): v for k, v in row.items() if k is not None}
                name = _field(row, ("Name", "Op Name", "Operator Name", "Kernel Name"))
                kind = _field(row, ("Type", "Op Type", "Task Type", "Accelerator Core")) or ""
                duration = _time_ms(row, ("Duration", "Task Duration", "Task Time"))
                if not name:
                    missing.append(f"csv_row_without_name:{path}:{index + 2}")
                    continue
                if duration is None or duration < 0:
                    missing.append(f"csv_actual_duration_or_unit_missing:{path}:{index + 2}")
                    continue
                counters = {k: {"value": n, "unit": _unit(k), "raw": v} for k, v in row.items()
                            if _counter(k) and (n := _num(v)) is not None}
                result.append({"name": str(name), "op_type": str(kind), "duration_ms": duration,
                               "start_ms": _time_ms(row, ("Start Time", "Task Start Time", "Timestamp")),
                               "task_id": _field(row, ("Task Id",)), "stream_id": _field(row, ("Stream Id",)),
                               "step_id": _field(row, ("Step Id",)), "device_id": _field(row, ("Device Id",)),
                               "phase": classify_kernel(name, kind), "attribution": "kernel_name",
                               "counters": counters, "source": str(path), "row": index + 2})
    except (OSError, csv.Error, UnicodeError) as exc:
        missing.append(f"csv_read_error:{path}:{type(exc).__name__}")
    return result


def _marker(name):
    name = str(name)
    for prefix in ("oscar_timer::", "oscar_diagnostic::", "oscar_diag::", "oscar::"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return None


def _correlations(event):
    return {(str(_canon(k)), str(v)) for k, v in event.get("args", {}).items()
            if _canon(k) in {"correlationid", "correlation", "externalid", "corrid"}}


def parse_trace(path, missing=None, time_unit="us"):
    """Chrome complete events. displayTimeUnit is UI-only, not timestamp units."""
    missing = missing if missing is not None else []
    document = _read_json(path, missing)
    events = document.get("traceEvents", []) if isinstance(document, dict) else document
    if not isinstance(events, list) or time_unit not in _UNIT_MS:
        missing.append(f"unsupported_trace_schema_or_clock:{path}")
        return []
    factor = _UNIT_MS[time_unit]
    devices = {str(e.get("pid")) for e in events if isinstance(e, dict) and e.get("ph") == "M"
               and e.get("name") == "process_name" and any(v in str(e.get("args", {}).get("name", "")).lower()
                                                          for v in ("ascend hardware", "npu", "device"))}
    complete = [e for e in events if isinstance(e, dict) and e.get("ph") == "X"
                and _num(e.get("ts")) is not None and _num(e.get("dur")) is not None and _num(e["dur"]) >= 0]
    markers = [e for e in complete if _marker(e.get("name"))]
    correlations = defaultdict(set)
    for event in complete:
        own = _marker(event.get("name"))
        containing = [m for m in markers if m.get("pid") == event.get("pid") and m.get("tid") == event.get("tid")
                      and m["ts"] <= event["ts"] and m["ts"] + m["dur"] >= event["ts"] + event["dur"]]
        phase = own or (_marker(min(containing, key=lambda m: m["dur"])["name"]) if containing else None)
        if phase:
            for key in _correlations(event):
                correlations[key].add(phase)
    output = []
    for event in complete:
        args = {_canon(k): v for k, v in event.get("args", {}).items()}
        cat = str(event.get("cat", "")).lower()
        task_type = _canon(args.get("tasktype", args.get("type", "")))
        is_device = task_type in {"aicore", "aicpu", "aivector", "aivectorcore", "mixaic", "mixaiv", "mix"} or (
            cat in {"kernel", "kernels", "npu_kernel", "device_kernel", "aicore", "aicpu"}
            and (str(event.get("pid")) in devices or "deviceid" in args or "npu" in cat))
        if not is_device:
            continue
        name, kind = str(event.get("name", "unknown")), str(args.get("optype", ""))
        phases = set().union(*(correlations.get(key, set()) for key in _correlations(event)))
        named = classify_kernel(name, kind)
        # A custom kernel's exact identity is more specific than its enclosing host scope.
        phase = next(iter(phases)) if len(phases) == 1 and not named.startswith("oscar.") else named
        output.append({"name": name, "op_type": kind, "duration_ms": float(event["dur"]) * factor,
                       "start_ms": float(event["ts"]) * factor, "task_id": args.get("taskid"),
                       "stream_id": args.get("streamid", event.get("tid")), "device_id": args.get("deviceid"),
                       "phase": phase, "attribution": "trace_correlation" if phase != named else "kernel_name",
                       "counters": {k: {"value": n, "unit": _unit(k), "raw": v}
                                    for k, v in event.get("args", {}).items() if _counter(k) and (n := _num(v)) is not None},
                       "source": str(path)})
    return output


def _task_key(row):
    if row.get("start_ms") is None:
        return None
    return (row.get("device_id"), str(row.get("stream_id")), str(row.get("task_id")),
            round(row["start_ms"], 9), round(row["duration_ms"], 9), row["name"])


def _join_key(row):
    """Cross-export identity requires time+name and compatible task/stream IDs."""
    return (round(row["start_ms"], 6), round(row["duration_ms"], 6), row["name"]) if row.get("start_ms") is not None else None


def _compatible(a, b):
    return all(a.get(k) is None or b.get(k) is None or str(a[k]) == str(b[k])
               for k in ("task_id", "stream_id", "device_id"))


def _load_window(manifest, root, missing):
    directory = _resolve(root, manifest.get("profile_dir", "."))
    if not directory.is_dir():
        missing.append(f"profile_dir_missing:{directory}")
        return [], [], []
    files = sorted(set(directory.rglob("kernel_details*.csv")))
    op_files = sorted(set(directory.rglob("op_summary*.csv")))
    traces = sorted(set(directory.rglob("trace_view*.json"))) or sorted(set(directory.rglob("msprof_*.json")))
    csv_rows = [r for p in files for r in parse_csv(p, missing)]
    op_rows = [r for p in op_files for r in parse_csv(p, missing)]
    trace_rows = [r for p in traces for r in parse_trace(p, missing, manifest.get("trace_time_unit", "us"))]
    rows = csv_rows or op_rows or trace_rows
    # CSV exports and trace describe the same tasks. Never concatenate their durations.
    extras = (op_rows if csv_rows else []) + (trace_rows if rows is not trace_rows else [])
    lookup = defaultdict(list)
    for item in extras:
        if (key := _join_key(item)) is not None:
            lookup[key].append(item)
    merged, seen, matched_extras = [], {}, set()
    for row in rows:
        key = _task_key(row)
        if key is not None and key in seen and seen[key] != row["source"]:
            continue
        if key is not None:
            seen[key] = row["source"]
        matches = [x for x in lookup.get(_join_key(row), []) if _compatible(row, x)]
        for extra in matches:
            matched_extras.add(id(extra))
            for name, value in extra["counters"].items():
                row["counters"].setdefault(name, value)
            if extra["attribution"] == "trace_correlation":
                row["phase"], row["attribution"] = extra["phase"], extra["attribution"]
        merged.append(row)
    if csv_rows and op_rows and any(r["counters"] for r in op_rows) and not any(r["counters"] for r in merged):
        missing.append(f"pmu_csv_not_aligned_to_tasks:{directory}")
    if not merged:
        missing.append(f"device_tasks_missing:{directory}")
    unaligned = [r for r in op_rows if csv_rows and r["counters"] and id(r) not in matched_extras]
    if unaligned:
        missing.append(f"unmatched_pmu_rows:{len(unaligned)}:{directory}")
    return merged, [str(p) for p in files + op_files + traces], unaligned


def _counter_stats(rows):
    groups = defaultdict(list)
    for row in rows:
        for name, item in row["counters"].items():
            groups[name].append((item["value"], row["duration_ms"], item["unit"]))
    result = {}
    for name, values in sorted(groups.items()):
        weight = sum(v[1] for v in values)
        result[name] = {"unit": values[0][2], "samples": len(values), "min": min(v[0] for v in values),
                        "max": max(v[0] for v in values), "duration_weighted_mean":
                        sum(v[0] * v[1] for v in values) / weight if weight else mean(v[0] for v in values),
                        "covered_task_ms": weight}
    return result


def _bound_evidence(rows):
    ratios = []
    for row in rows:
        counters = row["counters"]
        # Compare only the same AIC/AIV family and unit, avoiding MIX lane conflation.
        for prefix in ("", "aic", "aiv"):
            by_unit = defaultdict(dict)
            for name, item in counters.items():
                key = _base(name)
                for label, needles in {"cube": ("macratio", "macexeratio", "mactime", "cubetime"),
                                       "vector": ("vecratio", "vecexeratio", "vectime"),
                                       "mte2": ("mte2ratio", "mte2exeratio", "mte2time")}.items():
                    if key in {prefix + x for x in needles}:
                        by_unit[item["unit"]][label] = item
            for unit in ("ratio", "%", "us", "ms", "ns", "cycles"):
                values = by_unit.get(unit, {})
                compute = [values[k] for k in ("cube", "vector") if k in values]
                memory = values.get("mte2")
                if not memory or not compute:
                    continue
                denom = max(x["value"] for x in compute)
                if denom > 0:
                    ratios.append({"kernel": row["name"], "family": prefix or "generic", "ratio": memory["value"] / denom,
                                   "task_ms": row["duration_ms"], "unit": memory["unit"],
                                   "denominator": "max(cube,vector)" if len(compute) == 2 else "cube" if "cube" in values else "vector"})
                    break
    complete_ratios = [r for r in ratios if r["denominator"] == "max(cube,vector)"]
    covered = sum(r["task_ms"] for r in complete_ratios)
    ratio = sum(r["ratio"] * r["task_ms"] for r in complete_ratios) / covered if covered else None
    families = defaultdict(list)
    for item in ratios:
        families[(item["family"], item["denominator"], item["unit"])].append(item)
    pipe_ratios = [{"family": key[0], "formula": "mte2/" + key[1], "unit": key[2], "samples": len(values),
                    "duration_weighted_mean": sum(v["ratio"] * v["task_ms"] for v in values) / sum(v["task_ms"] for v in values)
                    if sum(v["task_ms"] for v in values) else mean(v["ratio"] for v in values)}
                   for key, values in sorted(families.items())]
    counters = _counter_stats(rows)
    bandwidth = {k: v for k, v in counters.items() if any(s in _canon(k) for s in ("mainmem", "hbm", "ddr"))}
    candidate = "insufficient_device_counters"
    if ratio is not None:
        candidate = "transfer_pipeline_dominant" if ratio > 1 else "compute_pipeline_dominant_or_other_stall"
    elif pipe_ratios:
        candidate = "per_core_pipe_ratio_available_missing_other_compute_pipe"
    return {"classification": "inconclusive", "candidate": candidate, "mte2_over_max_cube_vector": ratio,
            "ratio_sample_count": len(complete_ratios), "per_core_pipe_ratios": pipe_ratios,
            "ratio_scope": "duration-weighted task PMU heuristic; overlapping pipes, not critical path or HBM proof",
            "memory_counters": bandwidth, "evidence_missing": [
                "same-workload aligned hardware roofline/achieved bandwidth or operation rates versus sustainable device ceilings",
                "critical-path stalls and overlap evidence; MTE traffic is not automatically HBM traffic"]}


def _metric_present(window):
    names = [_base(k) for k in window["counters"]]
    if window.get("metric") == "PipeUtilization":
        return any(any(s in k for s in ("macratio", "vecratio", "mactime", "vectime")) for k in names) and any("mte" in k for k in names)
    if window.get("metric") == "Memory":
        return any(any(s in k for s in ("mainmem", "hbm", "ddr", "l2readbw", "l2writebw", "ubreadbw", "ubwritebw")) for k in names)
    return False


def _device_summary(manifest, rows, sources):
    phases = defaultdict(list)
    kernels = defaultdict(list)
    for row in rows:
        phases[row["phase"]].append(row)
        kernels[(row["name"], row["phase"])].append(row)
    total = sum(r["duration_ms"] for r in rows)
    intervals = [(r["start_ms"], r["start_ms"] + r["duration_ms"]) for r in rows if r["start_ms"] is not None]
    all_timed = len(intervals) == len(rows) and bool(rows)
    return {"rank": manifest.get("rank"), "window_id": manifest.get("window_id"), "bucket": manifest.get("bucket", "unknown"),
            "metric": manifest.get("metric"), "status": manifest.get("status"), "steps": manifest.get("steps", []),
            "profile_dir": manifest.get("profile_dir"), "sources": sources, "task_count": len(rows),
            "task_sum_ms": total, "device_busy_union_ms": _union(intervals) if all_timed else None,
            "device_span_ms": max(b for _, b in intervals) - min(a for a, _ in intervals) if all_timed else None,
            "time_scope": "profiled single-rank window; task sums overlap and are not service wall",
            "phases": {k: {"calls": len(v), "task_sum_ms": sum(r["duration_ms"] for r in v),
                            "task_share": sum(r["duration_ms"] for r in v) / total if total else None,
                            "counters": _counter_stats(v), "bound": _bound_evidence(v)} for k, v in sorted(phases.items())},
            "top_ops": sorted(({"name": k[0], "phase": k[1], "calls": len(v),
                                "task_sum_ms": sum(r["duration_ms"] for r in v),
                                "mean_ms": mean(r["duration_ms"] for r in v), "counters": _counter_stats(v)}
                               for k, v in kernels.items()), key=lambda x: x["task_sum_ms"], reverse=True)[:20],
            "counters": _counter_stats(rows), "bound": _bound_evidence(rows)}


def _steps(root, run_id, missing):
    rows = []
    for path in sorted(root.rglob("timer-steps-*.jsonl")):
        try:
            with path.open(encoding="utf-8") as handle:
                for lineno, line in enumerate(handle, 1):
                    try:
                        row = json.loads(line)
                    except ValueError:
                        missing.append(f"invalid_step_json:{path}:{lineno}")
                        continue
                    if isinstance(row, dict) and row.get("run_id") == run_id:
                        rows.append(row)
        except OSError as exc:
            missing.append(f"step_read_error:{path}:{type(exc).__name__}")
    return rows


def _host_summary(steps):
    grouped = defaultdict(list)
    for step in steps:
        grouped[(step.get("rank"), step.get("bucket", "unknown"), bool(step.get("profiled")))].append(step)
    output = []
    for (rank, bucket, profiled), group in sorted(grouped.items(), key=lambda x: str(x[0])):
        phase = defaultdict(float)
        for step in group:
            for key, value in _exclusive_scopes(step.get("scopes", [] )).items():
                phase[key] += value
        times = [n for step in group if (n := _num(step.get("host_ms"))) is not None]
        output.append({"rank": rank, "bucket": bucket, "profiled": profiled, "steps": len(group),
                       "host_ms_mean": mean(times) if times else None, "host_ms_sum": sum(times) if times else None,
                       "exclusive_host_ms": dict(phase), "scheduled_requests_mean": mean(
                           _num(s.get("scheduled_requests")) or 0 for s in group),
                       "scheduled_prompt_tokens": sum(_num(s.get("scheduled_prompt_tokens")) or 0 for s in group),
                       "scheduled_decode_tokens": sum(_num(s.get("scheduled_decode_tokens")) or 0 for s in group),
                       "scheduled_draft_tokens": sum(_num(s.get("scheduled_draft_tokens")) or 0 for s in group),
                       "graph_replays": sum(_num(s.get("graph_replays")) or 0 for s in group),
                       "contexts": [{"min": s.get("context_min"), "max": s.get("context_max"), "source": s.get("context_source")} for s in group],
                       "scope": "host submission, not device completion; exclusive scopes partition nested intervals"})
    return output


def _batches(document):
    if isinstance(document.get("services"), list):
        return [dict(batch, variant=batch.get("variant", service.get("variant")))
                for service in document["services"] for batch in service.get("batches", [])]
    return document.get("runs", [])


def _normal_result(batch):
    sample = batch.get("sample") or {}
    wall = _num(sample.get("batch_elapsed_seconds", batch.get("wall_s")))
    requests = sample.get("requests", [])
    tokens = sum(_num((r.get("usage") or {}).get("completion_tokens")) or 0 for r in requests)
    completed = sample.get("completed_requests", batch.get("completed_requests"))
    profiled = batch.get("mode") == "profile" or batch.get("profiled", False)
    return {"client_wall_s": wall, "completed_requests": completed, "failed_requests": sample.get("failed_requests"),
            "timeouts": sample.get("timeouts"), "output_tokens": tokens if requests else None,
            "output_tokens_per_second": tokens / wall if requests and wall and wall > 0 else None,
            "eligible_for_throughput_comparison": not profiled and batch.get("status") in {"complete", "completed", "passed"}
            and not sample.get("failed_requests", 0) and not sample.get("timeouts", 0),
            "profiled": profiled, "latency_ms": sample.get("latency_ms", {}),
            "reported_throughput_tps": sample.get("throughput_tps", {})}


def summarize(output_dir):
    """Read runner's run.json plus bounded step/window artifacts; never execute."""
    root = Path(output_dir).resolve()
    missing = []
    document = _read_json(root / "run.json", missing) or {}
    report = {"schema_version": SCHEMA_VERSION, "kind": "oscar_timer_summary", "root": str(root),
              "run_id": document.get("run_id"), "status": document.get("status"), "pair_hash": document.get("pair_hash"),
              "error": document.get("error"),
              "source": document.get("source"), "model_identity": document.get("model_identity"),
              "workload": document.get("workload"), "counter_reference": COUNTER_SOURCE, "runs": [], "missing": missing,
              "evidence_sources": [COUNTER_SOURCE, COUNTER_CONFIG_SOURCE],
              "bound": "inconclusive_until_device_counter_and_roofline_review", "comparisons": [], "device_phase_comparisons": [],
              "accounting": "No sum across TP ranks; no parent/child host double count; profiling windows never normal throughput."}
    manifests, runtime_summaries = [], []
    for path in sorted(root.rglob("*.json")):
        if "windows" not in path.parts:
            continue
        item = _read_json(path, missing)
        if isinstance(item, dict) and "window_id" in item and "profile_dir" in item:
            manifests.append(item)
        elif isinstance(item, dict) and item.get("t") == "oscar-diagnostic-run":
            runtime_summaries.append(item)
    for batch in _batches(document):
        run_id = batch.get("run_id", batch.get("label"))
        steps = _steps(root, run_id, missing)
        run = {"run_id": run_id, "variant": batch.get("variant"), "concurrency": batch.get("concurrency", batch.get("client_concurrency")),
               "mode": batch.get("mode", "profile" if batch.get("profiled") else "control"), "status": batch.get("status"),
               "error": batch.get("error"),
               "pair_hash": batch.get("pair_hash"), "prompt_manifest_sha256": batch.get("prompt_manifest_sha256"),
               "normal": _normal_result(batch), "host": _host_summary(steps), "device_windows": [],
               "mtp_counter_delta": batch.get("mtp_counter_delta"), "counter_delta": batch.get("counter_delta"),
               "peak_running": batch.get("peak_running"), "peak_waiting": batch.get("peak_waiting"),
               "record_path": batch.get("record_path"), "profile_dir": batch.get("profile_dir"),
               "request_completion_check": batch.get("request_completion_check"), "missing": []}
        run["control_flush_status"] = (batch.get("control_flush") or {}).get("status")
        run["runtime_summaries"] = [s for s in runtime_summaries if s.get("run_id") == run_id]
        if not steps and run["mode"] == "profile":
            run["missing"].append("bounded_step_records_missing")
        windows = [m for m in manifests if m.get("run_id") == run_id]
        for manifest in windows:
            if manifest.get("status") != "complete":
                run["missing"].append(f"window_{manifest.get('status', 'unknown')}:{manifest.get('window_id')}:{manifest.get('error', manifest.get('unsupported_reason', ''))}")
            rows, sources, unaligned = _load_window(manifest, root, run["missing"])
            item = _device_summary(manifest, rows, sources)
            item["unaligned_counter_exports"] = {"rows": len(unaligned), "counters": _counter_stats(unaligned),
                                                  "scope": "cannot attribute to selected task timings"}
            matched_steps = [s for s in steps if s.get("rank") == manifest.get("rank") and s.get("step_id") in manifest.get("steps", [])]
            item["scheduled_requests"] = [s.get("scheduled_requests") for s in matched_steps]
            item["scheduled_prompt_tokens"] = [s.get("scheduled_prompt_tokens") for s in matched_steps]
            item["scheduled_decode_tokens"] = [s.get("scheduled_decode_tokens") for s in matched_steps]
            item["context_min"] = [s.get("context_min") for s in matched_steps]
            item["context_max"] = [s.get("context_max") for s in matched_steps]
            item["profile_start_ms"] = [s.get("profile_start_ms") for s in matched_steps]
            item["profile_export_ms"] = [s.get("profile_export_ms") for s in matched_steps]
            item["step_alignment"] = "matched" if len(matched_steps) == len(manifest.get("steps", [])) and matched_steps else "missing"
            if item["step_alignment"] == "missing":
                run["missing"].append(f"window_step_alignment_missing:{manifest.get('window_id')}")
            if not item["counters"]:
                run["missing"].append(f"pmu_counters_missing:{manifest.get('window_id')}")
            run["device_windows"].append(item)
        if run["mode"] == "profile" and not windows:
            run["missing"].append("profiler_window_manifests_missing")
        if not run["mtp_counter_delta"]:
            run["missing"].append("MTP_acceptance_window_counters_missing")
        if run["mode"] == "profile" and (not steps or any(s.get("context_min") is None or s.get("context_max") is None for s in steps)):
            run["missing"].append("some_step_context_lengths_missing_from_scheduler_CPU_metadata")
        for runtime in run["runtime_summaries"]:
            if any(runtime.get("missing_windows", {}).values()):
                run["missing"].append(f"worker_windows_missing:rank{runtime.get('rank')}:{runtime.get('missing_reason')}")
        report["runs"].append(run)
    controls = [r for r in report["runs"] if r["normal"]["eligible_for_throughput_comparison"]]
    native = next((r for r in controls if r["variant"] == "native"), None)
    for other in controls:
        if not native or other is native:
            continue
        checks = {"pair_hash": bool(native["pair_hash"]) and native["pair_hash"] == other["pair_hash"],
                  "request_count": native["normal"]["completed_requests"] == other["normal"]["completed_requests"],
                  "prompt_set": bool(native["prompt_manifest_sha256"]) and native["prompt_manifest_sha256"] == other["prompt_manifest_sha256"]}
        n, o = native["normal"], other["normal"]
        comparable = all(checks.values()) if native["concurrency"] == other["concurrency"] else checks["pair_hash"]
        ratio = o["output_tokens_per_second"] / n["output_tokens_per_second"] if comparable and n["output_tokens_per_second"] and o["output_tokens_per_second"] else None
        report["comparisons"].append({"native": native["run_id"], "other": other["run_id"], "checks": checks,
                                      "scope": "same-client-concurrency" if native["concurrency"] == other["concurrency"] else "capacity-concurrency-throughput; different cohort size",
                                      "output_throughput_ratio": ratio,
                                      "wall_ratio": o["client_wall_s"] / n["client_wall_s"] if all(checks.values()) and n["client_wall_s"] and o["client_wall_s"] else None})
    if not native:
        missing.append("completed_unprofiled_native_control_missing")
    native_profile = next((r for r in report["runs"] if r["variant"] == "native" and r["mode"] == "profile"), None)
    if native_profile:
        for other in report["runs"]:
            if other["mode"] != "profile" or other["variant"] == "native" or other["concurrency"] != native_profile["concurrency"]:
                continue
            for window in other["device_windows"]:
                possible = [w for w in native_profile["device_windows"] if w["rank"] == window["rank"]
                            and w["bucket"] == window["bucket"] and w["metric"] == window["metric"]]
                if len(possible) != 1:
                    continue
                baseline = possible[0]
                shape_keys = ("scheduled_requests", "scheduled_prompt_tokens", "scheduled_decode_tokens", "context_min", "context_max")
                checks = {key: bool(window[key]) and all(x is not None for x in window[key]) and window[key] == baseline[key] for key in shape_keys}
                checks["pair_hash"] = bool(other["pair_hash"]) and other["pair_hash"] == native_profile["pair_hash"]
                phases = sorted(set(baseline["phases"]) | set(window["phases"]))
                report["device_phase_comparisons"].append({"native": native_profile["run_id"], "other": other["run_id"],
                    "rank": window["rank"], "bucket": window["bucket"], "metric": window["metric"], "checks": checks,
                    "alignment": "same_scheduler_shape" if all(checks.values()) else "not_aligned",
                    "scope": "profiled per-rank task sums; different instances, not normal wall or full-model speedup",
                    "phase_task_ms": {p: {"native": baseline["phases"].get(p, {}).get("task_sum_ms", 0),
                                           "oscar": window["phases"].get(p, {}).get("task_sum_ms", 0),
                                           "delta": window["phases"].get(p, {}).get("task_sum_ms", 0) - baseline["phases"].get(p, {}).get("task_sum_ms", 0) if all(checks.values()) else None} for p in phases}})
    report["missing"] = list(dict.fromkeys(missing))
    profiles = [r for r in report["runs"] if r["mode"] == "profile"]
    live = [w for r in profiles for w in r["device_windows"] if w["task_count"]]
    report["evidence_status"] = "missing" if not live else "partial"
    expected_ranks = int((document.get("effective_config") or {}).get("tensor_parallel_size") or 4)
    for run in profiles:
        required = {(rank, bucket, metric) for rank in range(expected_ranks)
                    for bucket in ("early", "decode") for metric in ("PipeUtilization", "Memory")}
        present = {(w["rank"], "decode" if w["bucket"] == "decode" else "early", w["metric"])
                   for w in run["device_windows"] if w["task_count"] and _metric_present(w) and w["status"] == "complete"
                   and w["step_alignment"] == "matched"}
        run["required_windows_missing"] = sorted(required - present, key=str)
        if required - present:
            run["missing"].append("required_profile_windows_missing:" + repr(run["required_windows_missing"]))
    if profiles and all(not r["missing"] for r in profiles):
        report["evidence_status"] = "complete"
    return report


def _fmt(value):
    return "NA" if value is None else f"{value:.4g}" if isinstance(value, (int, float)) else str(value)


def _compact(value, limit=600):
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    return text if len(text) <= limit else text[:limit - 16] + "...[see JSON]"


def _bounded_lines(lines):
    """Protect comparison/footer evidence against the historical 65535B limit."""
    mandatory = {"===", "RUN", "RUN_ERROR", "SOURCE", "MODEL", "WORKLOAD", "ACCOUNTING", "BOUND", "FUSED_CV",
                 "BATCH", "BATCH_ERROR", "BATCH_META", "MTP", "LATENCY", "COMPARE", "DEVICE_COMPARE", "GLOBAL_MISSING", "FULL_REPORT"}
    important = {"DEVICE", "DEVICE_PHASES", "PHASE_PMU", "PIPE_EVIDENCE", "WINDOW_COVERAGE", "MISSING", "FILES"}
    candidates, shortened = [], False
    for index, line in enumerate(lines):
        raw = line.encode("utf-8")
        if len(raw) > 1600:
            line = raw[:1570].decode("utf-8", errors="ignore") + "...[full JSON]"
            shortened = True
        token = line.split(" ", 1)[0]
        priority = 0 if token in mandatory else 1 if token in important else 3 if token.startswith("PMU") else 2
        candidates.append((priority, index, line))
    selected, used = [], 0
    for _, index, line in sorted(candidates):
        size = len(line.encode("utf-8")) + 1
        if len(selected) < 198 and used + size <= 50 * 1024 - 256:
            selected.append((index, line))
            used += size
        else:
            shortened = True
    output = [line for _, line in sorted(selected)]
    if shortened:
        output.insert(max(0, len(output) - 1), "SUMMARY_COMPACTED low-priority details shortened; full per-rank/window/PMU evidence remains in timer-summary.json")
    return output


def summary_lines(report):
    """At most 200 lines / 50 KiB UTF-8; keep comparison and footer first."""
    lines = ["=== OSCAR_TIMER_SUMMARY_BEGIN ===", f"RUN {_fmt(report.get('run_id'))} status={report.get('status')} evidence={report.get('evidence_status')} pair_hash={report.get('pair_hash')}",
             f"SOURCE {_compact(report.get('source'))}", f"MODEL {_compact(report.get('model_identity'))}",
             f"WORKLOAD {_compact({k: v for k, v in (report.get('workload') or {}).items() if k not in {'prompt_manifest', 'prompt_ids'}})}",
             "ACCOUNTING control=client wall with NPU profiler/phase sampling off; shared control-file polling/no-op hooks remain. Profile=device diagnosis only; TP ranks separate; task sum != wall; host scopes exclusive.",
             "BOUND timer alone cannot establish compute/HBM bound; PMU ratios are candidates, missing roofline remains inconclusive."]
    lines.append("FUSED_CV fused INT2 unpack/softmax/window work shares one kernel; timer does not invent internal sub-phase times.")
    if report.get("error"):
        lines.append("RUN_ERROR " + _compact(report["error"], 1200))
    for run in report.get("runs", []):
        normal = run["normal"]
        lines.append(f"BATCH {run['run_id']} variant={run['variant']} client_bs={run['concurrency']} mode={run['mode']} status={run['status']} wall_s={_fmt(normal['client_wall_s'])} done={normal['completed_requests']} output={normal['output_tokens']} control_output_tps={_fmt(normal['output_tokens_per_second'] if run['mode'] != 'profile' else None)} compare_eligible={normal['eligible_for_throughput_comparison']}")
        if run.get("error"):
            lines.append("BATCH_ERROR " + _compact(run["error"], 1200))
        lines.append(f"BATCH_META pair={run['pair_hash']} prompts={run['prompt_manifest_sha256']} peak_running={run['peak_running']} peak_waiting={run['peak_waiting']} completion={_compact(run.get('request_completion_check'), 250)} flush={run.get('control_flush_status')}")
        lines.append(f"MTP {run['run_id']} {_compact(run['mtp_counter_delta'])} counter_delta={_compact(run['counter_delta'], 250)}")
        if run["mode"] != "profile":
            lines.append(f"LATENCY {run['run_id']} {_compact(normal['latency_ms'])}")
            continue
        if run["runtime_summaries"]:
            lines.append("WINDOW_COVERAGE " + _compact([{k: s.get(k) for k in ("rank", "target_batch", "max_decode_requests", "missing_windows", "missing_reason")} for s in run["runtime_summaries"]], 1100))
        # Host statistics are per rank and bucket; show largest rank plus full skew.
        host_groups = defaultdict(list)
        for h in run["host"]:
            host_groups[(h["bucket"], h["profiled"])].append(h)
        for (bucket, profiled), group in sorted(host_groups.items(), key=lambda x: str(x[0]))[:3]:
            worst = max(group, key=lambda x: x["host_ms_mean"] or 0)
            lines.append(f"HOST {run['run_id']} bucket={bucket} profiled={profiled} rank_mean_ms={_compact({str(x['rank']):x['host_ms_mean'] for x in group}, 300)} steps={worst['steps']} actual_requests_mean={_fmt(worst['scheduled_requests_mean'])} prompt/decode/draft={worst['scheduled_prompt_tokens']}/{worst['scheduled_decode_tokens']}/{worst['scheduled_draft_tokens']} graph={worst['graph_replays']}")
            top = sorted(worst["exclusive_host_ms"].items(), key=lambda x: x[1], reverse=True)[:10]
            lines.append(f"HOST_EXCLUSIVE rank={worst['rank']} bucket={bucket} " + " ".join(f"{k}={v:.3f}ms" for k, v in top))
        groups = defaultdict(list)
        for window in run["device_windows"]:
            groups[(window["bucket"], window["metric"])].append(window)
        for (bucket, metric), windows in sorted(groups.items(), key=lambda x: str(x[0]))[:6]:
            # Do not sum repeated samples/ranks. A named representative window is explicit.
            representative = max(windows, key=lambda x: x["task_sum_ms"])
            rank_ms = defaultdict(list)
            for item in windows:
                rank_ms[str(item["rank"])].append(item["device_busy_union_ms"])
            lines.append(f"DEVICE {run['run_id']} bucket={bucket} metric={metric} windows={len(windows)} rank_busy_union_ms={_compact(dict(rank_ms), 400)} representative={representative['window_id']} rank={representative['rank']} tasks={representative['task_count']} sum_ms={_fmt(representative['task_sum_ms'])} span_ms={_fmt(representative['device_span_ms'])} aligned={representative['step_alignment']}")
            shapes = {(tuple(w["steps"]), tuple(w["scheduled_requests"]), tuple(w["scheduled_prompt_tokens"]), tuple(w["scheduled_decode_tokens"]), tuple(w["context_min"]), tuple(w["context_max"])) for w in windows}
            known_contexts = all(w["context_min"] and w["context_max"] and all(n is not None for n in w["context_min"] + w["context_max"]) for w in windows)
            means = [mean(n for n in values if n is not None) for values in rank_ms.values() if any(n is not None for n in values)]
            skew = max(means) / min(means) if known_contexts and len(shapes) == 1 and len(means) > 1 and min(means) > 0 else None
            lines[-1] += f" RANK_SKEW={_fmt(skew)} shape_alignment={'matched' if known_contexts and len(shapes) == 1 else 'not_aligned'} actual_requests={representative['scheduled_requests']} decode_tokens={representative['scheduled_decode_tokens']} context={representative['context_min']}..{representative['context_max']}"
            phases = sorted(representative["phases"].items(), key=lambda x: x[1]["task_sum_ms"], reverse=True)
            lines.append("DEVICE_PHASES " + " ".join(f"{k}={v['task_sum_ms']:.3f}ms({100*(v['task_share'] or 0):.1f}%)" for k, v in phases[:12]))
            important = phases[:1]
            attention = next(((p, v) for p, v in phases if any(s in p.lower() for s in ("attention", "full", "gdn")) and p not in {k for k, _ in important}), None)
            if attention:
                important.append(attention)
            for phase, values in important:
                counters = sorted(values["counters"].items(), key=lambda kv: (not any(s in _canon(kv[0]) for s in ("mainmem", "hbm", "mte2", "macratio", "vecratio")), kv[0]))[:6]
                lines.append(f"PHASE_PMU {bucket}/{metric} phase={phase} " + " | ".join(f"{k}={_fmt(v['duration_weighted_mean'])}[{v['unit']}]" for k, v in counters) + f" candidate={values['bound']['candidate']}")
            bound = representative["bound"]
            lines.append(f"PIPE_EVIDENCE bucket={bucket} metric={metric} mte2/max(cube,vector)={_fmt(bound['mte2_over_max_cube_vector'])} per_core={_compact(bound['per_core_pipe_ratios'], 500)} candidate={bound['candidate']} verdict=inconclusive")
            metrics = sorted(representative["counters"].items(), key=lambda kv: (not any(s in _canon(kv[0]) for s in ("mainmem", "hbm", "mte2", "macratio", "vecratio")), kv[0]))
            for offset in range(0, min(len(metrics), 12), 6):
                lines.append(f"PMU rank={representative['rank']} {bucket}/{metric} " + " | ".join(f"{k}={_fmt(v['duration_weighted_mean'])} [{v['unit']}] n={v['samples']}" for k, v in metrics[offset:offset + 6]))
            unaligned = representative.get("unaligned_counter_exports", {})
            if unaligned.get("rows"):
                values = {k: v["duration_weighted_mean"] for k, v in list(unaligned["counters"].items())[:8]}
                lines.append(f"PMU_UNALIGNED rows={unaligned['rows']} values={_compact(values, 700)} not attached to task duration")
        # Split the 20-op budget across early/decode; long prefill cannot hide decode.
        hot_windows = []
        for early in (True, False):
            choices = [w for w in run["device_windows"] if (w["bucket"] != "decode") == early and w["task_count"]]
            pipe_choices = [w for w in choices if w["metric"] == "PipeUtilization"]
            if choices:
                hot_windows.append(max(pipe_choices or choices, key=lambda x: x["task_sum_ms"]))
        budget = 10 if len(hot_windows) == 2 else 20
        for hot in hot_windows:
            lines.append(f"TOP{budget} {run['run_id']} only_window={hot['window_id']} rank={hot['rank']} {hot['bucket']}/{hot['metric']}")
            ops = hot["top_ops"][:budget]
            for offset in range(0, len(ops), 2):
                lines.append(" | ".join(f"OP {op['task_sum_ms']:.3f}ms calls={op['calls']} mean={op['mean_ms']:.4f}ms phase={op['phase']} name={op['name'][:180]}" for op in ops[offset:offset + 2]))
        lines.append(f"MISSING {run['run_id']} {_compact(list(dict.fromkeys(run['missing'])), 1000)}")
        lines.append(f"FILES records={run['record_path']} profile={run['profile_dir']}")
    for comparison in report.get("comparisons", []):
        lines.append("COMPARE " + _compact(comparison, 900))
    for item in [c for c in report.get("device_phase_comparisons", []) if c["rank"] == 0][:4]:
        phases = sorted(item["phase_task_ms"].items(), key=lambda p: max(p[1]["native"], p[1]["oscar"]), reverse=True)[:12]
        lines.append(f"DEVICE_COMPARE rank=0 {item['bucket']}/{item['metric']} alignment={item['alignment']} checks={_compact(item['checks'], 350)} " + " ".join(f"{p}:N={v['native']:.3f},O={v['oscar']:.3f},delta={_fmt(v['delta'])}ms" for p, v in phases))
    footer = ["GLOBAL_MISSING " + _compact(report.get("missing", []), 1200),
              f"FULL_REPORT {report['root']}/timer-summary.json RAW_ROOT {report['root']}"]
    lines.extend(footer)
    lines.append("=== OSCAR_TIMER_SUMMARY_END ===")
    return _bounded_lines(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    report = summarize(args.output_dir)
    destination = args.output_dir / "timer-summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n".join(summary_lines(report)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
