"""Summarize real C16 actor counters without inventing a kernel critical path.

Archive #70-73/#133: device events, local cycles and request wall are separate.
Archive #145/#164: local completion fences perturb the pipeline; diagnostic
phase shares cannot be multiplied by the original 1199 ms or summed over cores.
This reader never imports torch, loads kernels, connects to NPU or runs models.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re


MAX_LINES = 150
MAX_BYTES = 40 * 1024
ENGINES = ("AIC", "AIV0", "AIV1")
CATEGORIES = ("compute", "data_movement", "wait", "mixed")
BITWISE_CHECKS = ("partial", "lse", "status", "cluster_stats", "baseline_status_zero")
CALL_FILE = re.compile(r"call-rank(\d+)-(\d+)\.json$")


def _integer(value):
    return value if type(value) is int and 0 <= value <= (1 << 63) - 1 else None


def _positive(value):
    if type(value) not in (int, float):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _json(path, issues):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as error:
        issues.append(f"json_read_error:{path}:{type(error).__name__}:{error}")
        return None
    if not isinstance(value, dict):
        issues.append(f"json_object_required:{path}")
        return None
    return value


def _resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def _layer_id(name):
    if not isinstance(name, str):
        return None
    match = re.search(r"(?:^|\.)layers\.(\d+)\.self_attn\.attn(?:$|\.)", name)
    return int(match.group(1)) if match else None


def _schema(issues):
    try:
        from oscar_ascend.c16_timer_schema import SCHEMA_ID, PHASE_NAMES, PHASE_CATEGORIES, CLOCK_HZ, CLOCK_SOURCE
    except (ImportError, AttributeError) as error:
        issues.append(f"phase_schema_unavailable:{type(error).__name__}:{error}")
        return None, (), {}, {}
    names = tuple(PHASE_NAMES)
    categories = dict(PHASE_CATEGORIES)
    if (len(set(names)) != len(names) or any(not isinstance(n, str) for n in names)
            or any(categories.get(n) not in CATEGORIES for n in names)):
        issues.append("phase_schema_invalid_names_or_categories")
    return SCHEMA_ID, names, categories, {"hz": CLOCK_HZ, "source": CLOCK_SOURCE}


def _actor(raw, phase_names, categories, issues, label):
    if not isinstance(raw, dict):
        issues.append(f"actor_not_object:{label}")
        return None
    core = _integer(raw.get("core"))
    engine = raw.get("engine") if isinstance(raw.get("engine"), str) else "invalid"
    total, other = _integer(raw.get("total_cycles")), _integer(raw.get("other_cycles"))
    prefix = f"{label}/core{core}/{engine}"
    local = []
    if core is None or engine not in ENGINES:
        local.append("invalid_actor_identity")
    if total is None or other is None:
        local.append("invalid_total_or_other_cycles")
    phases = raw.get("phases")
    if not isinstance(phases, dict):
        phases = {}
        local.append("phase_object_missing")
    absent, unknown = sorted(set(phase_names) - set(phases)), sorted(set(phases) - set(phase_names))
    if absent:
        local.append("phases_missing:" + ",".join(absent))
    if unknown:
        local.append("phases_unknown:" + ",".join(unknown))
    rows, category_cycles = [], {name: 0 for name in CATEGORIES}
    for name in phase_names:
        item = phases.get(name)
        if not isinstance(item, dict):
            if name not in absent:
                local.append(f"phase_not_object:{name}")
            continue
        cycles, count = _integer(item.get("cycles")), _integer(item.get("count"))
        if cycles is None or count is None or cycles > 0 and count == 0:
            local.append(f"invalid_phase_cycles_or_count:{name}")
            continue
        category = categories.get(name, "mixed")
        if category not in CATEGORIES:
            local.append(f"phase_category_invalid:{name}:{category}")
            category_cycles.setdefault(category, 0)
        row = {"name": name, "category": category, "cycles": cycles, "count": count,
               "local_share": cycles / total if total else None,
               "cycles_per_occurrence": cycles / count if count else None}
        # Only explicitly recorded bytes are shown; count * guessed tile size
        # is not measured traffic and never becomes HBM bandwidth here.
        for key in ("bytes", "logical_bytes"):
            if key in item:
                if _integer(item[key]) is None:
                    local.append(f"invalid_{key}:{name}")
                else:
                    row[key] = item[key]
        if "byte_scope" in item:
            row["byte_scope"] = item["byte_scope"]
        rows.append(row)
        category_cycles[category] += cycles
    phase_sum = sum(row["cycles"] for row in rows)
    closed = total is not None and other is not None and phase_sum + other == total and len(rows) == len(phase_names)
    if not closed:
        local.append(f"exclusive_cycle_account_not_closed:phase={phase_sum}:other={other}:total={total}")
    if raw.get("ready") is not None and raw["ready"] not in (True, 1):
        local.append(f"actor_not_ready:{raw['ready']}")
    if raw.get("flags") not in (None, 0):
        local.append(f"actor_flags:{raw['flags']}")
    issues.extend(f"{prefix}:{reason}" for reason in local)
    return {"core": core, "engine": engine, "total_cycles": total, "other_cycles": other,
            "phase_sum_cycles": phase_sum, "account_closed": closed, "valid": not local,
            "ready": raw.get("ready"), "flags": raw.get("flags"), "phases": rows,
            "categories": {name: {"cycles": cycles, "local_share": cycles / total if total else None}
                           for name, cycles in category_cycles.items()},
            "other_share": other / total if total and other is not None else None}


def _call(path, root, schema_id, phase_names, categories, clock_contract, build_hash):
    issues = []
    raw = _json(path, issues)
    if raw is None:
        return {"path": str(path), "valid": False, "issues": issues, "actors": []}
    match = CALL_FILE.fullmatch(path.name)
    rank, index = _integer(raw.get("rank")), _integer(raw.get("call_index"))
    if match and (rank != int(match[1]) or index != int(match[2])):
        issues.append("filename_and_call_identity_mismatch")
    if raw.get("phase_schema_id") != schema_id or not schema_id:
        issues.append("phase_schema_id_mismatch_or_missing")
    if raw.get("status") != "complete":
        issues.append(f"call_status:{raw.get('status')}:{raw.get('error', '')}")
    if raw.get("instrumentation") != "local_completion_fences":
        issues.append("instrumentation_contract_missing_or_unknown")
    if raw.get("inference_uses") != "original_production_outputs":
        issues.append("production_output_ownership_unconfirmed")
    if type(raw.get("is_draft")) is not bool:
        issues.append("target_draft_role_missing")
    validation = raw.get("validation") if isinstance(raw.get("validation"), dict) else {}
    bitwise = validation.get("bitwise") if isinstance(validation.get("bitwise"), dict) else {}
    if validation.get("status") != "passed" or any(bitwise.get(key) is not True for key in BITWISE_CHECKS):
        issues.append("output_or_status_validation_not_passed")
    if raw.get("counters_valid") is not True:
        issues.append("producer_counter_validation_not_passed")
    if raw.get("counter_issues"):
        issues.append("producer_counter_issues:" + str(raw["counter_issues"]))
    baseline, diagnostic = _positive(raw.get("baseline_ms")), _positive(raw.get("diagnostic_ms"))
    if baseline is None or diagnostic is None:
        issues.append("finite_positive_device_event_ms_missing")
    geometry = raw.get("geometry") if isinstance(raw.get("geometry"), dict) else {}
    for key in ("tokens", "hq", "hk", "dim", "splits", "cores", "max_seq_len", "max_query_len"):
        if _integer(geometry.get(key)) is None or geometry[key] <= 0:
            issues.append(f"geometry_missing_or_invalid:{key}")
    if geometry.get("dequant_mode") != 0:
        issues.append("prefill_dequant_mode_not_zero")
    if (geometry.get("tokens") != 16384 or (_integer(geometry.get("max_seq_len")) or 0) < 32768
            or (_integer(geometry.get("max_query_len")) or 0) < 16384):
        issues.append("selected_warm_32k_prefill_geometry_not_confirmed")
    call_build = raw.get("build") if isinstance(raw.get("build"), dict) else {}
    if build_hash and call_build.get("library_sha256") != build_hash:
        issues.append("diagnostic_build_library_hash_mismatch")
    counter_path = raw.get("counter_path")
    if not isinstance(counter_path, str) or not counter_path:
        issues.append("raw_counter_path_missing")
    else:
        resolved = _resolve(root, counter_path)
        if not resolved.is_file() or resolved.stat().st_size == 0:
            issues.append("raw_counter_file_missing_or_empty:" + str(resolved))
    actors = []
    if not isinstance(raw.get("actors"), list):
        issues.append("actor_list_missing")
    else:
        for actor in raw["actors"]:
            parsed = _actor(actor, phase_names, categories, issues, path.name)
            if parsed is not None:
                actors.append(parsed)
    cores = _integer(geometry.get("cores"))
    if cores is not None and cores > 32:
        issues.append("invalid_actor_grid_core_count")
        cores = None
    expected = {(core, engine) for core in range(cores or 0) for engine in ENGINES}
    identities = [(actor["core"], actor["engine"]) for actor in actors]
    if len(set(identities)) != len(identities) or set(identities) != expected or not expected:
        issues.append("actor_grid_incomplete_duplicate_or_unexpected")
    clock_hz = _positive(raw.get("clock_hz"))
    check = raw.get("timebase_check") if isinstance(raw.get("timebase_check"), dict) else {}
    actor_totals = [actor["total_cycles"] for actor in actors if actor["total_cycles"] is not None]
    max_actor_ms = max(actor_totals) * 1000.0 / clock_hz if actor_totals and clock_hz else None
    recorded_actor_ms, recorded_event_ms = _positive(check.get("max_actor_ms")), _positive(check.get("event_ms"))
    timebase_valid = bool(clock_hz and clock_hz == clock_contract.get("hz")
                          and raw.get("clock_source") == clock_contract.get("source")
                          and check.get("status") == "passed" and max_actor_ms and diagnostic
                          and 0 < max_actor_ms <= diagnostic * 1.05 + .1
                          and recorded_actor_ms and math.isclose(recorded_actor_ms, max_actor_ms, rel_tol=1e-9, abs_tol=1e-9)
                          and recorded_event_ms and math.isclose(recorded_event_ms, diagnostic, rel_tol=1e-9, abs_tol=1e-9))
    if not timebase_valid:
        issues.append("native_clock_identity_or_event_upper_bound_check_not_passed")
    for actor in actors:
        def milliseconds(cycles):
            return cycles * 1000.0 / clock_hz if timebase_valid and actor["valid"] and cycles is not None else None
        actor["diagnostic_ms"] = milliseconds(actor["total_cycles"])
        actor["other_diagnostic_ms"] = milliseconds(actor["other_cycles"])
        for phase in actor["phases"]:
            phase["diagnostic_ms"] = milliseconds(phase["cycles"])
        for category in actor["categories"].values():
            category["diagnostic_ms"] = milliseconds(category["cycles"])
    primary = {}
    for family in ("AIC", "AIV"):
        choices = [actor for actor in actors if actor["valid"] and actor["total_cycles"]
                   and (actor["engine"] == "AIC") == (family == "AIC")]
        if choices:
            primary[family] = max(choices, key=lambda actor: actor["total_cycles"])
        else:
            issues.append(f"active_{family}_actor_missing")
    return {"path": str(path), "rank": rank, "call_index": index, "layer": raw.get("layer"),
            "layer_id": _layer_id(raw.get("layer")), "is_draft": raw.get("is_draft"),
            "draft_index": raw.get("draft_index"), "geometry": geometry, "source_status": raw.get("status"),
            "valid": not issues, "issues": issues, "validation": validation, "counters_valid": raw.get("counters_valid"),
            "baseline_ms": baseline, "diagnostic_ms": diagnostic,
            "diagnostic_over_baseline": diagnostic / baseline if baseline and diagnostic else None,
            "counter_path": counter_path, "extra_output_bytes": raw.get("extra_output_bytes"),
            "actors": actors, "primary_actors": primary,
            "instrumentation": raw.get("instrumentation"), "phase_schema_id": raw.get("phase_schema_id"),
            "clock_hz": raw.get("clock_hz"), "clock_source": raw.get("clock_source"),
            "timebase_valid": timebase_valid, "timebase_check": check}


def summarize(root):
    root = Path(root).resolve()
    issues = []
    manifest = _json(root / "run.json", issues) or {}
    schema_id, phase_names, categories, clock_contract = _schema(issues)
    config = manifest.get("effective_config") or {}
    expected_count = manifest.get("expected_ranks", config.get("tensor_parallel_size", 4))
    if _integer(expected_count) is None or not 1 <= expected_count <= 256:
        issues.append("expected_rank_count_missing_or_invalid")
        expected_count = 4
    model = manifest.get("model_identity") or {}
    layer_names = model.get("full_layers") or []
    expected_layers = {_layer_id(name) for name in layer_names}
    if not expected_layers or None in expected_layers or len(expected_layers) != len(layer_names):
        issues.append("expected_full_attention_layers_missing_or_invalid")
        expected_layers.discard(None)
    build = manifest.get("timer_build") or manifest.get("diag_build") or {}
    if build.get("status") not in {"built", "reused"}:
        issues.append("diagnostic_library_build_not_confirmed")
    if build.get("phase_schema_id") not in (None, schema_id):
        issues.append("build_phase_schema_mismatch")
    service = manifest.get("service") or next(iter(manifest.get("services") or []), {})
    request = service.get("request") or {}
    completion = request.get("completion_check") or {}
    if request.get("status") != "completed" or completion.get("status") != "passed":
        issues.append("real_32k_request_completion_not_passed")
    if (service.get("lifecycle") or {}).get("cleanup_complete") is not True or (service.get("resource_release") or {}).get("status") != "passed":
        issues.append("owned_service_or_resource_cleanup_not_passed")
    if manifest.get("status") != "completed":
        issues.append(f"run_not_completed:{manifest.get('status')}:{manifest.get('error', '')}")
    device_value = manifest.get("device_dir") or str(root / "device")
    directory = _resolve(root, device_value)
    paths = sorted(path for path in directory.glob("call-rank*-*.json") if CALL_FILE.fullmatch(path.name)) if directory.is_dir() else []
    calls = [_call(path, root, schema_id, phase_names, categories, clock_contract, build.get("library_sha256")) for path in paths]
    grouped = defaultdict(list)
    for call in calls:
        rank = call.get("rank")
        if type(rank) is int:
            grouped[rank].append(call)
        if not call["valid"]:
            issues.append("invalid_call:" + call["path"])
    unexpected = sorted(set(grouped) - set(range(expected_count)))
    if unexpected:
        issues.append(f"unexpected_ranks:{unexpected}")
    ranks = []
    for rank in range(expected_count):
        group = grouped.get(rank, [])
        valid = [call for call in group if call["valid"]]
        target = [call for call in valid if call.get("is_draft") is False]
        draft = [call for call in valid if call.get("is_draft") is True]
        observed = {call["layer_id"] for call in target if call["layer_id"] is not None}
        missing_layers = sorted(expected_layers - observed)
        if not group:
            issues.append(f"rank_missing:{rank}")
        if missing_layers:
            issues.append(f"target_layers_missing:rank{rank}:{missing_layers}")
        unknown = sorted({str(call.get("layer")) for call in target if call["layer_id"] is None or call["layer_id"] not in expected_layers})
        if unknown:
            issues.append(f"unexpected_target_layers:rank{rank}:{unknown}")
        representative = max(valid, key=lambda call: call["diagnostic_ms"]) if valid else None
        baseline_sum = sum(call["baseline_ms"] for call in valid)
        diagnostic_sum = sum(call["diagnostic_ms"] for call in valid)
        nonwait_winners = Counter()
        for call in target:
            aiv = call["primary_actors"].get("AIV")
            concrete = [phase for phase in aiv["phases"] if phase["cycles"] > 0 and phase["category"] != "wait"
                        and phase["name"] not in {"task_other", "cluster_other", "process_other", "init"}] if aiv else []
            winner = max(concrete, key=lambda phase: phase["cycles"])["name"] if concrete else "no_positive_concrete_nonwait_phase"
            nonwait_winners[winner] += 1
        ranks.append({"rank": rank, "calls": len(group), "valid_calls": len(valid), "target_calls": len(target),
                      "draft_calls": len(draft), "draft_coverage": "observed_only_not_a_fixed_expected_count",
                      "observed_target_layers": sorted(observed), "missing_target_layers": missing_layers,
                      "baseline_event_sum_ms": baseline_sum, "diagnostic_event_sum_ms": diagnostic_sum,
                      "event_sum_scope": "selected calls on this rank only; not TP sum, request wall or original 1199 ms",
                      "diagnostic_over_baseline": diagnostic_sum / baseline_sum if baseline_sum else None,
                      "representative_call": representative["call_index"] if representative else None,
                      "target_aiv_nonwait_phase_winners": dict(sorted(nonwait_winners.items())),
                      "winner_scope": "one largest-local-total AIV per target call; counts only, no across-actor duration sum",
                      "primary_actors": representative["primary_actors"] if representative else {},
                      "representative_geometry": representative["geometry"] if representative else None,
                      "representative_role": "draft" if representative and representative["is_draft"] else "target" if representative else None})
    issues = list(dict.fromkeys(issues))
    if not calls:
        status = "error" if manifest.get("status") in {"failed", "interrupted"} else "missing"
    elif any(not call["valid"] for call in calls) or manifest.get("status") in {"failed", "interrupted"}:
        status = "error"
    else:
        status = "partial" if issues else "complete"
    return {"schema_version": 1, "kind": "c16_breakdown_summary", "status": status, "root": str(root),
            "run_id": manifest.get("run_id"), "source_status": manifest.get("status"), "source": manifest.get("source"),
            "error": manifest.get("error"), "summary_error": manifest.get("summary_error"), "build": build,
            "model_identity": model, "workload": manifest.get("workload"), "request_completion": completion,
            "expected_ranks": expected_count, "expected_target_layers": sorted(expected_layers),
            "phase_schema_id": schema_id, "phase_categories": categories,
            "clock_contract": clock_contract,
            "category_scope": "operation semantics of fenced exclusive intervals, not measured ALU/HBM busy or roofline",
            "timing_scope": "actor-local cycles; no cross-actor phase-max sum, no unperturbed critical path, no conversion to baseline milliseconds",
            "ranks": ranks, "calls": calls, "issues": issues,
            "paths": {"run": str(root / "run.json"), "device": str(directory),
                      "summary": str(root / "c16-breakdown-summary.json")}}


def _fmt(value):
    if value is None:
        return "NA"
    return f"{value:.5g}" if type(value) is float else str(value)


def _compact(value, limit=1200):
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return text if len(text) <= limit else text[:limit] + "...[full JSON]"


def _bounded(lines):
    priorities = {"C16_RUN", "C16_SCOPE", "C16_SCHEMA", "C16_VALIDATION", "C16_COVERAGE", "C16_ERROR", "C16_FILES", "C16_SOURCE", "C16_WORKLOAD", "C16_BUILD"}
    selected, used, shortened = [], 0, False
    candidates = []
    for index, line in enumerate(lines):
        raw = line.encode("utf-8")
        if len(raw) > 1800:
            line = raw[:1760].decode("utf-8", errors="ignore") + "...[full JSON]"
            shortened = True
        token = line.split(" ", 1)[0]
        priority = 0 if token in priorities or token == "===" else 1 if token in {"C16_ACTOR", "C16_PHASES", "C16_CATEGORIES", "C16_PRIORITY", "C16_CONSISTENCY"} else 2
        candidates.append((priority, index, line))
    for _, index, line in sorted(candidates):
        size = len(line.encode("utf-8")) + 1
        if len(selected) < MAX_LINES - 1 and used + size <= MAX_BYTES - 160:
            selected.append((index, line))
            used += size
        else:
            shortened = True
    output = [line for _, line in sorted(selected)]
    if shortened:
        output.insert(max(0, len(output) - 1), "C16_COMPACTED full per-call/per-actor phases and raw-counter paths remain in c16-breakdown-summary.json")
    return output


def summary_lines(report):
    lines = ["=== OSCAR_C16_BREAKDOWN_BEGIN ===",
             f"C16_RUN {report.get('run_id')} status={report['status']} source_status={report.get('source_status')}",
             "C16_SCOPE local completion fences perturb execution. Cycles are GetSystemCycle ticks; phase ms use the native clock after its event-bound sanity check, not frequency calibration. Do not multiply shares by old 1199 ms, sum actor maxima, or claim an unfenced critical path.",
             f"C16_SOURCE {_compact(report.get('source'))}",
             f"C16_WORKLOAD {_compact(report.get('workload'))}",
             f"C16_SCHEMA {report.get('phase_schema_id')} clock={_compact(report.get('clock_contract'))} category=operation_semantics_not_hardware_bound",
             f"C16_BUILD {_compact({k: report.get('build', {}).get(k) for k in ('status', 'library', 'library_sha256', 'phase_schema_id')})}",
             f"C16_VALIDATION request={_compact(report['request_completion'])} issues={len(report['issues'])} calls={len(report['calls'])} expected_ranks={report['expected_ranks']} target_layers={report['expected_target_layers']}"]
    if report.get("error") or report.get("summary_error"):
        lines.append("C16_ERROR " + _compact({"run": report.get("error"), "previous_summary": report.get("summary_error")}))
    for rank in report["ranks"]:
        lines.append(f"C16_COVERAGE rank={rank['rank']} calls={rank['calls']} valid={rank['valid_calls']} target/draft={rank['target_calls']}/{rank['draft_calls']} target_layers={rank['observed_target_layers']} missing={rank['missing_target_layers']} baseline_sum_ms={_fmt(rank['baseline_event_sum_ms'])} diag_sum_ms={_fmt(rank['diagnostic_event_sum_ms'])} diag/base={_fmt(rank['diagnostic_over_baseline'])} sums=selected_calls_this_rank_only")
        lines.append(f"C16_CONSISTENCY rank={rank['rank']} target_calls={rank['target_calls']} largest_local_AIV_concrete_nonwait_winners={_compact(rank['target_aiv_nonwait_phase_winners'])} scope=per_call_winner_counts_not_summed_time;wait_and_other_remain_in_actor_breakdown")
    for call in report["calls"]:
        g = call.get("geometry", {})
        role = "draft" if call.get("is_draft") is True else "target" if call.get("is_draft") is False else "unknown"
        lines.append(f"C16_CALL r{call.get('rank')}/c{call.get('call_index')} {role}[{call.get('draft_index')}] layer={call.get('layer')} N/S={g.get('tokens')}/{g.get('splits')} req={g.get('requests')} context/query={g.get('max_seq_len')}/{g.get('max_query_len')} baseline_ms={_fmt(call.get('baseline_ms'))} diag_ms={_fmt(call.get('diagnostic_ms'))} ratio={_fmt(call.get('diagnostic_over_baseline'))} valid={call['valid']} timebase={call.get('timebase_valid')}")
        if call["issues"]:
            lines.append("C16_ERROR " + _compact({"call": call["path"], "issues": call["issues"]}, 1300))
    for rank in report["ranks"]:
        for family, actor in rank["primary_actors"].items():
            tag = f"r{rank['rank']}/c{rank['representative_call']}/core{actor['core']}/{actor['engine']}"
            lines.append(f"C16_ACTOR {tag} selection=max_local_total_{family}_within_max_diag_ms_call role={rank['representative_role']} total_cycles={actor['total_cycles']} diagnostic_local_ms={_fmt(actor['diagnostic_ms'])} phase_sum={actor['phase_sum_cycles']} other={actor['other_cycles']} closed={actor['account_closed']}")
            categories = " ".join(f"{name}={item['cycles']}cy/{_fmt(item['diagnostic_ms'])}ms({100 * (item['local_share'] or 0):.2f}%)" for name, item in actor["categories"].items())
            lines.append(f"C16_CATEGORIES {tag} {categories} other={actor['other_cycles']}cy/{_fmt(actor['other_diagnostic_ms'])}ms({100 * (actor['other_share'] or 0):.2f}%)")
            phases = sorted((p for p in actor["phases"] if p["cycles"] or p["count"]), key=lambda p: p["cycles"], reverse=True)
            for offset in range(0, len(phases), 6):
                lines.append(f"C16_PHASES {tag} " + " | ".join(f"{p['name']}={_fmt(p['diagnostic_ms'])}ms/{p['cycles']}cy/{p['count']}x({100 * (p['local_share'] or 0):.2f}%,{p['category']})" + (f",recorded_bytes={p.get('bytes', p.get('logical_bytes'))}" if 'bytes' in p or 'logical_bytes' in p else '') for p in phases[offset:offset + 6]))
            if phases:
                top = phases[0]
                focus = "unattributed_other" if actor["other_cycles"] > top["cycles"] else top["name"]
                lines.append(f"C16_PRIORITY {tag} first_inspect={focus} basis=largest_same_actor_exclusive_interval; semantic_mixed_or_wait_requires_dependency_review")
    for issue in report["issues"]:
        lines.append("C16_ERROR " + issue)
    lines.extend(["C16_FILES " + _compact(report["paths"]), "=== OSCAR_C16_BREAKDOWN_END ==="])
    return _bounded(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args(argv)
    report = summarize(args.root)
    path = args.root / "c16-breakdown-summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("\n".join(summary_lines(report)))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
