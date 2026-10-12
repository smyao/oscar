"""Archive #132-135/#150/D.4: real R32/N128/S3 plan and existing PR fixtures.

The default action writes only a stdlib plan, with all execution fields not_run.
Explicit --export reuses this project's independent fixture/oracle offline;
it never launches a kernel, VM, simulator, or service.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def source_work(context, qlen):
    widths = {"history": context + qlen - 256 - 64, "window": 64 + 255, "current": qlen}
    units, tasks = {}, {}
    for name, width in widths.items():
        part = (width + 2) // 3
        pieces = [max(0, min(width, (i + 1) * part) - min(width, i * part)) for i in range(3)]
        units[name] = 32 * sum((piece + 255) // 256 for piece in pieces)
        tasks[name] = 32 * sum(piece > 0 for piece in pieces)
    return {"kv256_units": units, "nonempty_tasks": tasks, "task_slots": 1152,
            "logical_tasks_including_empty_sources": 288,
            "counter_comparison": "sum cores separately for AIC, AIV0, AIV1; never sum engines"}


def plan(contexts):
    # #150/D.4: C2049/S3 is the smallest declared R32 point with three
    # history tiles per split, including prologue/interior/epilogue.
    if len(contexts) not in (1, 2) or len(set(contexts)) != len(contexts) or any(c not in (1025, 2049, 4097, 8193) for c in contexts):
        raise ValueError("select one or two distinct predeclared contexts: 1025,2049,4097,8193")
    rows = []
    for context in contexts:
        for mode, qlen in (("q4", 4), ("q1", 1)):
            rows.append({"name": f"{mode}_r32_n128_c{context}_s3", "mode": mode, "contexts": [context] * 32,
                         "requests": 32, "tokens": 128, "actual_tokens": qlen * 32, "splits": 3,
                         "hq": 6, "hkv": 1, "dim": 256, "cores": 20, "source_work": source_work(context, qlen)})
    return {"schema_version": 1, "cache_format": "shipped striped_v1 per-page window, not dual-region request window",
            "cases": rows, "first_context": contexts[0], "all_sources_and_tasks": True, "task_filtering": False,
            "normal_symbols": ["oscar_bundle_normal_q4_kernel", "oscar_bundle_normal_q1_kernel"],
            "profile_symbols": ["oscar_bundle_profile_q4_kernel", "oscar_bundle_profile_q1_kernel"],
            "workspace_bytes_per_core": 1146880, "profile_bytes_per_core": 2304,
            "max_absolute_instrumentation_perturbation": .05,
            "stages": ["normal/profile complete numeric and instrumentation perturbation at first context",
                       "complete marker collection only after root approves bounded actor/total trace budget",
                       "second context only after complete first context; finite slope, not 64K extrapolation"],
            "trace_budget": {"required_complete_actor_logs": 60, "current_high_core_cap_bytes_per_file": 2097152,
                             "current_cap_is_not_complete_evidence": True, "global_caps_modified": False,
                             "proposed_research_cap_bytes_per_actor": 67108864,
                             "completion_logs_total_at_proposed_cap": 4026531840,
                             "with_popped_logs_total_at_proposed_cap": 8053063680,
                             "proposed_cap_is_not_enabled": True, "c4097_fit": "unknown",
                             "streaming_marker_aggregation": "not_implemented; root may select later"},
            "validation": {key: "not_run" for key in ("cann_compile", "fixture_export", "cpu_debug", "camodel",
                                                       "marker_completeness", "npu", "graph", "whole_service")}}


def export(directory, contexts):
    selected = plan(contexts)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from csrc.experiments.window_batch_decode import fixture as shared
    cases = tuple(shared.Case(row["name"], row["mode"], tuple(row["contexts"]), heads=1,
                              padded=128, splits=3, cores=20) for row in selected["cases"])
    original = shared.case_plan
    try:
        shared.case_plan = lambda _large=False, _long=False: cases
        manifest = shared.export(directory, cores=20)
    finally:
        shared.case_plan = original
    for row in manifest["cases"]:
        expected = next(c for c in selected["cases"] if c["name"] == row["name"])
        if (row["actual_tokens"], row["padded_tokens"], row["cores"], row["hkv"]) != (expected["actual_tokens"], 128, 20, 1):
            raise ValueError("shared fixture did not retain real R32 geometry")
        if row["source_work"]["kv256_units"] != expected["source_work"]["kv256_units"]:
            raise ValueError("logical source work differs from existing fixture")
        row["measurement_scope"] = "32_independent_requests_N128_S3_complete_operator_no_task_filtering"
    manifest["baseline"] = "same_source_oscar_bundle_normal_q4_or_q1_kernel"
    manifest["comparison"] = "normal versus instrumented same shipped bundle, not optimization candidate"
    manifest["profile_plan"] = selected
    (directory / "cases.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=int, nargs="+", default=[1025, 4097])
    parser.add_argument("--output", type=Path, default=OUT / "plan.json")
    parser.add_argument("--export", action="store_true", help="explicit offline fixture/oracle export; output is a directory")
    args = parser.parse_args()
    if args.export:
        export(args.output, args.contexts)
    else:
        args.output.write_text(json.dumps(plan(args.contexts), indent=2) + "\n")
        print(json.dumps({"status": "plan_written", "path": str(args.output), "fixture_export": "not_run", "camodel": "not_run"}))


if __name__ == "__main__":
    main()
