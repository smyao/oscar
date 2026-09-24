# Archive #70-#73/#129/#132-#135/#140-#142: diagnostic-only NPU event
# intervals must be reported per TP rank; sums across ranks are not wall time.
"""Summarize synchronized OSCAR device-phase events from one diagnostic burst."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from .phase import atomic_json


class DeviceEventEvidenceError(ValueError):
    pass


def _finite_ms(value, field, location):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise DeviceEventEvidenceError(f"{location}: {field} must be finite nonnegative milliseconds")
    return float(value)


def _stats(values):
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "sum_ms": None, "p50_ms": None, "p95_ms": None, "max_ms": None}
    return {"count": len(ordered), "sum_ms": sum(ordered),
            "p50_ms": statistics.median(ordered),
            "p95_ms": ordered[math.ceil(.95 * len(ordered)) - 1],
            "max_ms": ordered[-1]}


def summarize_device_events(trace_dir: Path, *, run_id: str,
                            expected_ranks: int = 4, output: Path | None = None) -> dict:
    """Validate host metadata with NPU events; keep replay as whole graph only.

    Each event is already synchronized by the worker's diagnostic collector.
    This trace alters scheduling and is never substituted for the uninstrumented
    native/OSCAR client ratios or for individual CANN kernel profiling (#133).
    """
    directory = Path(trace_dir).resolve()
    if not isinstance(run_id, str) or not run_id or type(expected_ranks) is not int or expected_ranks < 1:
        raise ValueError("diagnostic run_id and expected TP ranks are required")
    paths = sorted(directory.glob("device-events-*.jsonl"))
    by_rank_phase = defaultdict(list)
    by_stage_rank_phase = defaultdict(list)
    by_shape = defaultdict(list)
    incomplete_events = []
    states = set()
    source_files = []
    records = 0
    for path in paths:
        source_files.append(str(path))
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            location = f"{path}:{line_no}"
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise DeviceEventEvidenceError(f"{location}: malformed device event JSON") from error
            if not isinstance(event, dict):
                raise DeviceEventEvidenceError(f"{location}: device event must be an object")
            if event.get("run_id") != run_id:
                continue
            if event.get("t") == "oscar-device-timing-state":
                if event.get("enabled") is True and type(event.get("rank")) is int:
                    states.add(event["rank"])
                continue
            if event.get("t") != "oscar-device-event":
                continue
            rank, phase, pid = event.get("rank"), event.get("phase"), event.get("pid")
            if (not isinstance(phase, str) or not phase
                    or type(pid) is not int or pid <= 0
                    or event.get("device_evidence") != "npu_event"
                    or event.get("scope") != "synchronized_npu_stream_interval"
                    or event.get("debug_synchronization") is not True):
                raise DeviceEventEvidenceError(f"{location}: NPU event provenance/rank/phase is invalid")
            if (rank is None or event.get("failed") is True or event.get("device_ms") is None):
                incomplete_events.append({"source": location, "rank": rank, "phase": phase,
                                          "error": event.get("error")})
                continue
            if type(rank) is not int or not 0 <= rank < expected_ranks:
                raise DeviceEventEvidenceError(f"{location}: NPU event rank is invalid")
            device_ms = _finite_ms(event.get("device_ms"), "device_ms", location)
            _finite_ms(event.get("host_ms"), "host_ms", location)
            stage = "graph" if phase == "graph_replay" else event.get("stage")
            if stage not in {"prefill", "draft", "decode", "graph"}:
                incomplete_events.append({"source": location, "rank": rank, "phase": phase,
                                          "error": "missing/invalid semantic stage"})
                continue
            fields = tuple(event.get(name) for name in (
                "tokens", "max_query_len", "max_seq_len", "requests",
                "source_splits", "splits", "draft_index", "is_draft", "dummy_origin"))
            if any(value is not None and type(value) not in (int, bool) for value in fields):
                raise DeviceEventEvidenceError(f"{location}: shape/draft metadata must be host scalars")
            by_rank_phase[(rank, phase)].append(device_ms)
            by_stage_rank_phase[(stage, rank, phase)].append(device_ms)
            by_shape[(stage, rank, phase, *fields)].append(device_ms)
            records += 1

    rank_totals = {}
    phases = {}
    for rank in range(expected_ranks):
        phase_map = {phase: _stats(values) for (record_rank, phase), values in by_rank_phase.items()
                     if record_rank == rank}
        rank_totals[str(rank)] = {
            "event_count": sum(row["count"] for row in phase_map.values()),
            "phase_sum_ms": sum(row["sum_ms"] for row in phase_map.values()),
            "phases": phase_map,
        }
    for phase in sorted({phase for _, phase in by_rank_phase}):
        per_rank = {str(rank): _stats(by_rank_phase[(rank, phase)]) for rank in range(expected_ranks)}
        phases[phase] = {
            "per_rank": per_rank,
            "critical_rank_sum_ms": max((row["sum_ms"] or 0.0) for row in per_rank.values()),
            "rank_with_max_sum": max(range(expected_ranks),
                                     key=lambda rank: per_rank[str(rank)]["sum_ms"] or 0.0),
            "device_event_count": sum(row["count"] for row in per_rank.values()),
        }
    stages = {}
    for stage in ("prefill", "draft", "decode", "graph"):
        stage_phases = {}
        for phase in sorted({phase for name, _, phase in by_stage_rank_phase if name == stage}):
            per_rank = {str(rank): _stats(by_stage_rank_phase[(stage, rank, phase)])
                        for rank in range(expected_ranks)}
            stage_phases[phase] = {
                "per_rank": per_rank,
                "critical_rank_sum_ms": max((row["sum_ms"] or 0.0) for row in per_rank.values()),
                "rank_with_max_sum": max(range(expected_ranks),
                                         key=lambda rank: per_rank[str(rank)]["sum_ms"] or 0.0),
            }
        if stage_phases:
            stages[stage] = {"phases": stage_phases,
                             "critical_rank_phase_sum_ms": max(
                                 sum(sum(by_stage_rank_phase[(stage, rank, phase)])
                                     for phase in stage_phases)
                                 for rank in range(expected_ranks))}
    shapes = [{"stage": stage, "rank": rank, "phase": phase, "tokens": tokens,
               "max_query_len": max_query_len, "max_seq_len": max_seq_len,
               "requests": requests, "source_splits": source_splits, "splits": splits,
               "draft_index": draft_index, "is_draft": is_draft, "dummy_origin": dummy_origin,
               **_stats(values)}
              for (stage, rank, phase, tokens, max_query_len, max_seq_len, requests,
                   source_splits, splits, draft_index, is_draft, dummy_origin), values
              in by_shape.items()]
    shapes.sort(key=lambda row: (-(row["sum_ms"] or 0), row["phase"], row["rank"]))
    missing = [rank for rank in range(expected_ranks) if rank_totals[str(rank)]["event_count"] == 0]
    required = {str(rank): [phase for phase in ("fia", "current_native_fia", "merge")
                            if not by_rank_phase.get((rank, phase))]
                for rank in range(expected_ranks)}
    missing_graph = [rank for rank in range(expected_ranks)
                     if not by_rank_phase.get((rank, "graph_replay"))]
    status = ("observed" if records and not missing and
              not missing_graph and not any(required.values()) and
              not incomplete_events else "needs_evidence")
    result = {
        "status": status, "run_id": run_id, "source": str(directory), "files": source_files,
        "event_count": records, "expected_ranks": expected_ranks,
        "missing_ranks": missing, "missing_required_eager_phases_by_rank": required,
        "missing_graph_ranks": missing_graph,
        "incomplete_events": incomplete_events,
        "enabled_state_ranks": sorted(states), "ranks": rank_totals, "phases": phases,
        "stages": stages, "shape_groups": shapes,
        "critical_rank_phase_sum_ms": max((row["phase_sum_ms"] for row in rank_totals.values()), default=0.0),
        "graph_replay": {"status": ("whole_graph_event_observed_all_ranks" if not missing_graph
                                    else "incomplete_ranks" if "graph_replay" in phases else "not_observed"),
                         "inner_phases": "not_inferred_from_graph_replay"},
        "scope": "synchronized diagnostic NPU event intervals; changes scheduling; "
                 "critical-rank phase sums are not batch wall time or kernel-level CANN traces",
        "performance_acceptance": "not_established",
    }
    if output is not None:
        atomic_json(Path(output), result)
    return result


def compact_device_event_line(report: dict, *, output: Path) -> str:
    top = sorted(report.get("phases", {}).items(),
                 key=lambda item: -(item[1].get("critical_rank_sum_ms") or 0))[:6]
    text = ",".join(f"{name}:{row['critical_rank_sum_ms']:.1f}" for name, row in top) or "none"
    return (f"[oscar] PERF_DIAG_EVENTS status={report['status']} ranks={report['expected_ranks'] - len(report['missing_ranks'])}/"
            f"{report['expected_ranks']} events={report['event_count']} "
            f"top6_critical_rank_ms={text} graph={report['graph_replay']['status']} "
            f"scope=synchronized_diagnostic_only report={output}")


def compact_device_stage_lines(report: dict) -> list[str]:
    """At most three copyable stage rows, each using its own critical TP rank."""
    lines = []
    for stage in ("prefill", "draft", "graph"):
        group = report.get("stages", {}).get(stage)
        if not isinstance(group, dict):
            continue
        top = sorted(group["phases"].items(),
                     key=lambda item: -(item[1]["critical_rank_sum_ms"] or 0.0))[:3]
        phases = ",".join(f"{name}:{value['critical_rank_sum_ms']:.1f}"
                          for name, value in top)
        lines.append(f"[oscar] PERF_DIAG_STAGE stage={stage} "
                     f"critical_rank_phase_sum_ms={group['critical_rank_phase_sum_ms']:.1f} "
                     f"top3_critical_rank_ms={phases} scope=synchronized_diagnostic_only")
    return lines


def compact_cv_shape_lines(report: dict) -> list[str]:
    """Expose at most three CV shapes needed to test the empty-source hypothesis."""
    exact = defaultdict(list)
    keys = ("stage", "phase", "tokens", "requests", "max_query_len", "max_seq_len",
            "splits", "source_splits", "draft_index", "is_draft", "dummy_origin")
    for row in report.get("shape_groups", []):
        if row.get("phase") == "fia" and row.get("stage") in {"prefill", "draft"}:
            exact[tuple(row.get(key) for key in keys)].append(row)
    candidates = []
    for shape, rows in exact.items():
        # TP ranks execute in parallel: select the slowest rank for this exact
        # shape, not the sum of four rank times.
        critical = max(rows, key=lambda row: row.get("sum_ms") or 0.0)
        candidates.append((shape, critical))

    def slowest(items):
        return max(items, key=lambda item: item[1].get("sum_ms") or 0.0) if items else None

    prefill = [item for item in candidates if item[1]["stage"] == "prefill"]
    no_old_candidate = slowest([item for item in prefill
        if type(item[1].get("max_seq_len")) is int
        and item[1]["max_seq_len"] > 0
        and item[1]["max_seq_len"] == item[1].get("max_query_len")])
    chosen = []
    if no_old_candidate is not None:
        chosen.append(("candidate_no_old_context", no_old_candidate))
    other_prefill = slowest([item for item in prefill
                             if no_old_candidate is None or item[0] != no_old_candidate[0]])
    if other_prefill is not None:
        chosen.append(("largest_other_prefill", other_prefill))
    draft = slowest([item for item in candidates if item[1]["stage"] == "draft"])
    if draft is not None:
        chosen.append(("largest_draft", draft))

    lines = []
    for label, (_, row) in chosen:
        split = row.get("splits") if row.get("splits") is not None else row.get("source_splits")
        context = "unknown_max_seq_zero" if row.get("max_seq_len") == 0 else "not_proven_empty"
        lines.append(
            f"[oscar] PERF_DIAG_CV case={label} stage={row['stage']} rank={row['rank']} "
            f"tokens={row.get('tokens')} requests={row.get('requests')} "
            f"max_query_len={row.get('max_query_len')} max_seq_len={row.get('max_seq_len')} "
            f"splits={split} count={row['count']} "
            f"p50_ms={row['p50_ms']:.2f} p95_ms={row['p95_ms']:.2f} "
            f"rank_sum_ms={row['sum_ms']:.2f} context={context}")
    return lines
