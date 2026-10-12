"""Archive #132-135/#150/D.4: same-shape instrumentation perturbation gate.

Only the added profile buffer/schema may differ. Preserve completion, frozen
oracle, all-output bitwise, input/library/config/core/deadline identity gates.
This reads completed artifacts only; it never runs a kernel or a VM.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from csrc.experiments.window_batch_decode.compare_camodel import checked_arm


def compare(directory, case):
    baseline, bh = checked_arm(directory, case, "baseline")
    profile, ph = checked_arm(directory, case, "candidate")
    mode = baseline["mode"]
    if mode not in ("q4", "q1"):
        raise ValueError("unknown decode mode")
    for report, arm, enabled in ((baseline, "normal", False), (profile, "profile", True)):
        if report["kernel"] != f"oscar_bundle_{arm}_{mode}_kernel" or report.get("profile_counters") is not enabled:
            raise ValueError("wrong exact-source normal/profile symbol or ABI")
        if report["workspace_bytes_per_core"] != 1146880:
            raise ValueError("current bundle workspace capacity differs")
    attrs = baseline["attrs"]
    if (attrs[0], attrs[1], attrs[2], attrs[3], attrs[4], attrs[6], attrs[-1], baseline["cores"]) != (128, 6, 1, 256, 32, 1152, 3, 20):
        raise ValueError("not the declared real R32/N128/S3/20-core geometry")
    for key in ("case", "mode", "cores", "attrs", "scale", "input_sha256", "oracle_sha256", "cpu_baseline_sha256",
                "library_sha256", "simulator_configuration", "cpu_reference", "cpu_reference_policy"):
        if baseline.get(key) != profile.get(key):
            raise ValueError("normal/profile " + key + " differs")
    if baseline.get("phase_profile_schema") is not None or profile.get("phase_profile_schema") != 2:
        raise ValueError("unknown profile schema")
    if bh["cores"] != 20 or ph["cores"] != 20 or bh["hard_limit_seconds"] != ph["hard_limit_seconds"]:
        raise ValueError("host core/deadline differs")
    for name in ("partial", "lse", "status"):
        if baseline["outputs"][name]["sha256"] != profile["outputs"][name]["sha256"]:
            raise ValueError("normal/profile " + name + " bits differ")
    old, new = baseline["task_ticks"], profile["task_ticks"]
    perturbation = (new - old) / old
    return {"status": "completed_same_shape_instrumentation_comparison", "case": case,
            "cache_format": "shipped striped_v1 per-page window; not dual-region window_map",
            "normal_ticks": old, "instrumented_ticks": new, "profile_over_normal": new / old,
            "perturbation_fraction": perturbation, "partial_lse_status_bitwise": True,
            "perturbation_gate": "passed" if abs(perturbation) <= .05 else "failed",
            "predeclared_max_absolute_perturbation": .05,
            "independent_oracle": "passed_both_arms", "all_sources_and_tasks": True,
            "input_sha256": baseline["input_sha256"], "library_sha256": baseline["library_sha256"],
            "interpretation": "perturbation measurement, not an optimization speedup or uninstrumented phase fraction",
            "phase_evidence": "separate complete marker or nonzero SYS_CNT check required",
            "npu": "not_run", "graph": "not_run", "whole_service": "not_run"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--case", required=True)
    args = parser.parse_args()
    destination = args.directory / (args.case + "-instrumentation-comparison.json")
    try:
        result = compare(args.directory, args.case)
    except (KeyError, OSError, ValueError) as error:
        destination.write_text(json.dumps({"status": "failed", "reason": str(error), "profile_over_normal": None}, indent=2) + "\n")
        raise
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if result["perturbation_gate"] != "passed":
        raise SystemExit(2)
