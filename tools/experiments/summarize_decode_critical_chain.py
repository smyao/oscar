"""Archive #129/#132-135/#145/#148/#154-156, D.4: completed decode DAG only.

Summarize existing C1341 normal/profile evidence without running a simulator.
Ready-to-ready differences conserve each history period; no actor durations
are summed and pop-to-complete is never labeled pure compute or transfer busy.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.experiments.analyze_decode_completion import read_trace, scalar_scopes


def run():
    base = ROOT / "reports/decode_completion_20261012/completed/reports"
    analysis = json.loads((base / "analysis.json").read_text())
    pair = json.loads((base / "comparison.json").read_text())
    if analysis["status"] != "complete_bounded_instruction_attribution" or pair["status"] != "passed":
        raise ValueError("complete attribution and normal/profile gate required")
    soft = {}
    for lane in (0, 1):
        actor = f"core0.veccore{lane}"
        path = base / "profile/full_instructions" / (actor + ".instr_log.dump.gz")
        groups = defaultdict(list)
        for scope in scalar_scopes(read_trace(path, actor)):
            if scope["source"] == 0 and scope["field"] == 10:
                groups[scope["task_ordinal"]].append(scope)
        soft[lane] = groups
    results, previous = [], {}
    for item in analysis["history_rounds"]:
        task, index = item["task_ordinal"], item["kv_round"]
        deps = {(d["actor"], d["flag"]): d for d in item["dependencies"]}
        kv = deps["core0.cubecore0", 8]
        score = item["qk"]["ready_complete"]
        p = deps["core0.cubecore0", 10]["producer_ready_complete"]
        pv = item["pv"]["ready_complete"]
        vector = []
        for lane in (0, 1):
            if len(soft[lane][task]) != 5:
                raise ValueError("same-task five history Softmax scopes required")
            scope = soft[lane][task][index]
            if not score <= scope["start"] <= scope["end"] <= p:
                raise ValueError("Softmax scope not inside same round ScoreReady/PReady")
            vector.append({"lane": lane, "start_complete": scope["start"],
                           "end_complete": scope["end"], "span": scope["end"] - scope["start"]})
        row = {"task_ordinal": task, "kv_round": index,
               "class": "first" if index == 0 else "steady_full" if index in (1, 2)
                        else "full_before_tail" if index == 3 else "tail",
               "KVReady_complete": kv["producer_ready_complete"],
               "KV_producer_not_ready_at_wait": kv["observed_producer_not_ready_ticks"],
               "ScoreReady_complete": score, "PReady_complete": p, "PvReady_complete": pv,
               "ScoreReady_to_PReady": p - score, "PReady_to_PvReady": pv - p,
               "softmax_completed_scopes": vector,
               "qk_delivery_span_overlaps_prior_PV": item["qk"]["delivery_span"],
               "pv_delivery_span": item["pv"]["delivery_span"]}
        if task in previous:
            last = previous[task]
            row.update(previous_PvReady_to_ScoreReady=score-last,
                       PvReady_period=pv-last,
                       combined_QK_PV_delivery_fraction=((score-last)+(pv-p))/(pv-last))
            if row["PvReady_period"] != row["previous_PvReady_to_ScoreReady"] + row["ScoreReady_to_PReady"] + row["PReady_to_PvReady"]:
                raise ValueError("ready chain does not conserve period")
        previous[task] = pv
        results.append(row)
    steady = [r for r in results if r["class"] == "steady_full"]
    summary = {key: {"min": min(r[key] for r in steady), "median": statistics.median(r[key] for r in steady),
                     "max": max(r[key] for r in steady)} for key in (
        "previous_PvReady_to_ScoreReady", "ScoreReady_to_PReady", "PReady_to_PvReady",
        "PvReady_period", "combined_QK_PV_delivery_fraction")}
    return {"status": "complete_ready_chain_closed", "source_sha256": analysis["source_sha256"],
            "input_analysis_sha256": hashlib.sha256((base / "analysis.json").read_bytes()).hexdigest(),
            "pair": pair, "rows": results, "steady_full_summary": summary,
            "not_inferences": ["QK_PV_delivery_fraction_is_not_Cube_busy", "not_64K_K32_HBM_roofline",
                "first_KV_wait_includes_query_and_initialization_not_only_dequant",
                "softmax_method_includes_existing_memory_sync", "no_cross_actor_duration_sum"]}


if __name__ == "__main__":
    result = run()
    output = ROOT / "reports/decode_critical_chain_20261012.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "summary": result["steady_full_summary"],
                      "output": str(output)}))
