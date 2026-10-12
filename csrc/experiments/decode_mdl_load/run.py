"""Archive #129/#140/#145/#148/#154-156, full D.4: complete MDL A/B.

Reuse existing complete C1341 fixture/prepare, full numeric replay and six
regular-file trace collector. Root owns the external600s supervisor/VM lock.
No process launched by compare, no new benchmark load and no production path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from csrc.experiments.decode_mdl_load import generate
from csrc.experiments.decode_task_overlap.run_pair import checked as numeric_checked
from csrc.experiments.window_batch_decode import replay_camodel as replay
from tools.experiments import decode_completion_probe as collector


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def run(args):
    generate.verify_inputs()
    for choice in (False, True):
        text, binding = generate.generate(choice)
        path = generate.HERE / f"attention_{binding['arm']}.cpp"
        if path.read_text() != text:
            raise ValueError("stale generated candidate/source")
    output, root = args.output.resolve(), args.root.resolve()
    if root != ROOT or output.exists():
        raise ValueError("root mismatch or output already exists")
    plan = output.with_name(output.name + ".plan.json")
    if plan.exists():
        raise ValueError("refuse repeated run plan")
    plan.parent.mkdir(parents=True, exist_ok=True)
    save(plan, {"case": collector.CASE, "arm": args.arm, "source_sha256": generate.PINS[generate.SOURCE],
        "manifest_sha256": digest(generate.HERE / "manifest.json"), "runner_sha256": digest(__file__),
        "library_sha256": digest(args.library), "hard_limit_seconds": 600,
        "per_file_trace_cap": collector.CAP, "cpu_reference_policy": args.cpu_reference,
        "config_difference": "only doNorm/doMultiDataLoad; public API generated tiling"})
    argv = [replay.__file__, "--root", str(root), "--output", str(output), "--case", collector.CASE,
        "--variant", args.arm, "--fixture-root", str(args.fixture_root.resolve()),
        "--export-root", str(args.export_root.resolve()), "--library", str(args.library.resolve()),
        "--simulator-config-dir", str(args.simulator_config_dir.resolve()),
        "--baseline-q4", "oscar_mdl_baseline_q4_kernel", "--candidate-q4", "oscar_mdl_candidate_q4_kernel",
        "--baseline-q1", "oscar_mdl_baseline_q1_kernel", "--candidate-q1", "oscar_mdl_candidate_q1_kernel",
        "--baseline-workspace-bytes-per-core", str(collector.WORKSPACE),
        "--candidate-workspace-bytes-per-core", str(collector.WORKSPACE),
        "--cpu-reference", args.cpu_reference, "--instruction-budget-bytes", str(collector.CAP)]
    limit = resource.getrlimit(resource.RLIMIT_FSIZE)
    if limit[1] != resource.RLIM_INFINITY and limit[1] < collector.CAP:
        raise ValueError("child file cap cannot be applied")
    previous = sys.argv
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (collector.CAP, limit[1]))
        sys.argv = argv
        rc = replay.main()
        report = json.loads((output / "report.json").read_text())
        if rc or report["status"] != "passed" or report["failures"]:
            raise ValueError("full numeric/task/actor gate failed")
        collector.geometry(report)
        trace = collector.capture(output, report, (args.export_root / collector.CASE / "tasks.bin").read_bytes())
        report["mdl_complete_trace"] = {"status": "passed", "plan_sha256": digest(plan),
            "trace_report": "full_instructions.json", "total_raw_bytes": trace["total_raw_bytes"]}
        save(output / "report.json", report)
        print("MDL_COMPLETE", json.dumps({"arm": args.arm, "task_ticks": report["task_ticks"],
              "trace_bytes": trace["total_raw_bytes"]}), flush=True)
    except Exception as error:
        if output.exists():
            save(output / "mdl_failure.json", {"status": "failed", "reason": repr(error)})
        raise
    finally:
        sys.argv = previous
        resource.setrlimit(resource.RLIMIT_FSIZE, limit)


def checked(path, control, arm):
    report = numeric_checked(path, control)
    collector.geometry(report)
    receipt = json.loads(control.read_text())
    if receipt["timeout_seconds"] != 600:
        raise ValueError("changed predeclared deadline")
    plan = path.with_name(path.name + ".plan.json")
    binding = json.loads(plan.read_text())
    if (report.get("mdl_complete_trace", {}).get("status") != "passed" or
        digest(plan) != report["mdl_complete_trace"]["plan_sha256"] or
        binding["source_sha256"] != generate.PINS[generate.SOURCE] or binding["arm"] != arm or
        binding["library_sha256"] != report["library_sha256"] or
        report["kernel"] != f"oscar_mdl_{arm}_q4_kernel" or report["profile_counters"]):
        raise ValueError("source/arm/library/plan/trace binding failed")
    command = receipt["command"]
    owner = json.loads((path / "owner.json").read_text())
    if (command[command.index("--arm")+1] != arm or
        command[command.index("--output")+1] != owner["output"]):
        raise ValueError("supervisor belongs to another arm")
    trace = json.loads((path / "full_instructions.json").read_text())
    expected = {f"core0.{engine}.{kind}.dump" for engine in collector.ENGINES
                for kind in ("instr_log", "instr_popped_log")}
    if trace["status"] != "passed" or len(trace["files"]) != 6 or {r["source"] for r in trace["files"]} != expected:
        raise ValueError("incomplete full trace set")
    for item in trace["files"]:
        file = path / item["path"]
        if not item["complete"] or digest(file) != item["gzip_sha256"]:
            raise ValueError("trace hash/completion failure")
        collector.verify_gzip(file, item["raw_bytes"], item["raw_sha256"])
    for kind, count in (("task", 1), ("aic", 1), ("aiv", 2)):
        rows = report["profiles"][kind]
        if len(rows) != count or any(r["duration"] <= 0 or r["end_cycle"]-r["start_cycle"] != r["duration"] for r in rows):
            raise ValueError("incomplete actor profile")
    return report


def compare(args):
    old = checked(args.baseline, args.baseline_control, "baseline")
    new = checked(args.candidate, args.candidate_control, "candidate")
    for key in ("case", "mode", "cores", "attrs", "scale", "input_sha256", "oracle_sha256",
                "cpu_baseline_sha256", "library_sha256", "simulator_configuration", "cpu_reference",
                "cpu_reference_policy", "workspace_bytes_per_core", "trace_policy"):
        if old[key] != new[key]:
            raise ValueError("unpaired " + key)
    for key in ("partial", "lse", "status"):
        if old["outputs"][key]["sha256"] != new["outputs"][key]["sha256"]:
            raise ValueError("candidate changed " + key + " bits")
    result = {"status": "completed_same_library_pair", "source_sha256": generate.PINS[generate.SOURCE],
        "task_ticks": {"baseline": old["task_ticks"], "candidate": new["task_ticks"]},
        "ratio": old["task_ticks"] / new["task_ticks"], "all_partial_lse_status_bitwise": True,
        "independent_oracle": "passed_both", "all_sources_and_tasks": True,
        "source_abi_publisher_and_geometry": "unchanged", "full_trace_files_per_arm": 6,
        "load_geometry_verdict": "inspect_complete_instructions_before_claiming_full_B_DMA",
        "CPU": old["cpu_reference"], "CANN": "loaded_same_compiled_library",
        "NPU": "not_run", "graph": "not_run", "whole_service": "not_established"}
    save(args.output, result)
    print("MDL_COMPARISON", json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    arm = commands.add_parser("run")
    for name in ("root", "output", "fixture-root", "export-root", "library", "simulator-config-dir"):
        arm.add_argument("--" + name, type=Path, required=True)
    arm.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    arm.add_argument("--cpu-reference", choices=("required", "optional"), default="required")
    pair = commands.add_parser("compare")
    for name in ("baseline", "candidate", "baseline-control", "candidate-control", "output"):
        pair.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    return run(args) if args.command == "run" else compare(args)


if __name__ == "__main__":
    main()
