"""Complete same-library CAModel pair; archive #129/#140/#148/#154-156.

D.4: count all OSCAR work and actual completion, never host-prepack savings.
The existing replay keeps its input, guard, CPU, independent oracle and actor
checks. The supervisor supplies owned-process deadlines and real exit codes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(path, control):
    c = json.loads(control.read_text())
    if (c["status"] != "completed" or c["timed_out"] or
            c["child_returncode"] != 0 or c["supervisor_returncode"] != 0 or
            not c["cleanup_complete"] or c["interrupt_signal"] is not None):
        raise RuntimeError(f"arm failed completion/deadline: {control}")
    r = json.loads((path / "report.json").read_text())
    done = json.loads((path / "completion.json").read_text())
    if (r["status"] != "passed" or r["failures"] or
            done["status"] != "device_complete_outputs_saved" or
            not r["all_sources_and_tasks"] or r["task_filtering"] or
            r["oracle"]["bad_output"] or r["oracle"]["bad_lse"] or
            not isinstance(r["task_ticks"], int) or r["task_ticks"] <= 0):
        raise RuntimeError(f"arm failed full numerical/task/profile gate: {path}")
    for key, meta in r["outputs"].items():
        file = path / meta["path"]
        if (file.stat().st_size != meta["bytes"] or digest(file) != meta["sha256"] or
                done["outputs"][key] != meta):
            raise RuntimeError(f"changed saved output: {file}")
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--fixture-root", required=True)
    p.add_argument("--export-root", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--env-json", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--candidate-workspace", type=int, required=True)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--cpu-reference", choices=("required", "optional"), default="required",
                   help="optional only for explicitly recorded CPU long-shape timeout; full CAModel gates remain")
    a = p.parse_args()
    root = a.root.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    simulator_dir = json.loads(a.env_json.read_text()).get("CAMODEL_CONFIG_PATH")
    if not simulator_dir:
        raise ValueError("explicit paired simulator configuration required")
    arms = {}
    for variant in ("baseline", "candidate"):
        arm = a.output / variant
        control = a.output / (variant + "-control.json")
        cmd = [sys.executable, "tools/experiments/run_camodel_arm.py", "--cwd", str(root),
               "--timeout", str(a.timeout), "--output", str(a.output / (variant + "-logs")),
               "--control-path", str(control), "--env-json", str(a.env_json), "--",
               sys.executable, "csrc/experiments/window_batch_decode/replay_camodel.py",
               "--root", str(root), "--variant", variant, "--case", a.case,
               "--fixture-root", a.fixture_root, "--export-root", a.export_root,
               "--output", str(arm), "--library", str(a.library),
               "--baseline-q4", "oscar_task_overlap_baseline_q4_kernel",
               "--baseline-q1", "oscar_task_overlap_baseline_q1_kernel",
               "--candidate-q4", "oscar_task_overlap_candidate_q4_kernel",
               "--candidate-q1", "oscar_task_overlap_candidate_q1_kernel",
               "--baseline-workspace-bytes-per-core", "1146880",
               "--candidate-workspace-bytes-per-core", str(a.candidate_workspace),
               "--simulator-config-dir", simulator_dir, "--cpu-reference", a.cpu_reference]
        print(f"TASK_OVERLAP_ARM {a.case} {variant}", flush=True)
        subprocess.run(cmd, cwd=root, check=True)
        arms[variant] = checked(arm, control)
    old, new = arms["baseline"], arms["candidate"]
    for key in ("case", "mode", "cores", "attrs", "scale", "input_sha256",
                "oracle_sha256", "cpu_baseline_sha256", "library_sha256",
                "simulator_configuration", "cpu_reference", "profile_counters"):
        if old.get(key) != new.get(key):
            raise RuntimeError(f"unpaired {key}")
    for key in ("partial", "lse", "status"):
        if old["outputs"][key]["sha256"] != new["outputs"][key]["sha256"]:
            raise RuntimeError(f"A/B changed {key} bits")
    result = {"status": "completed_matched_comparison", "case": a.case,
              "task_ticks": {v: r["task_ticks"] for v, r in arms.items()},
              "baseline_over_candidate": old["task_ticks"] / new["task_ticks"],
              "partial_lse_status_bitwise": True, "independent_oracle": "passed_both",
              "all_sources_and_tasks": True, "cores": old["cores"],
              "cpu_reference": old.get("cpu_reference", "available_checked"),
              "input_sha256": old["input_sha256"], "library_sha256": old["library_sha256"],
              "scope": "complete_synthetic_operator_only", "npu": "not_run",
              "graph": "not_run", "whole_service": "not_run"}
    (a.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
