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
import struct
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from csrc.experiments.decode_mdl_load import generate
from csrc.experiments.decode_task_overlap.run_pair import checked as numeric_checked
from csrc.experiments.window_batch_decode import replay_camodel as replay
from tools.experiments import decode_completion_probe as collector

Q1_CASE = "q1_r2_n8_c1344_1376_s1"


def geometry(report, tasks=None):
    """Frozen complete q1 domain; the previously measured q4 stays unchanged."""
    if report["case"] == collector.CASE:
        return collector.geometry(report)
    a = report["attrs"]
    if (report["case"], report["mode"], report["cores"], a[0], a[1], a[2],
            a[3], a[4], a[6], a[-1]) != (Q1_CASE, "q1", 1, 8, 6, 1, 256, 2, 24, 1):
        raise ValueError("not a declared complete MDL case")
    if not report["all_sources_and_tasks"] or report["task_filtering"]:
        raise ValueError("task filtering prohibited")
    if tasks is None or len(tasks) != 24 * 128 or hashlib.sha256(tasks).hexdigest() != report["input_sha256"]["tasks"]:
        raise ValueError("q1 actual prepare task witness missing")
    rows = list(struct.iter_unpack("<16q", tasks))
    for token in range(8):
        for source in range(3):
            row = rows[token * 3 + source]
            if row[0] != token or row[2] != 0 or row[6] != source or row[7] != source or row[10]:
                raise ValueError("q1 task identity/error drift")
            if token >= 2:
                # attention_tasks.cpp initializes count=-1. A negative slot
                # bypasses the live-row branch with error=0 and empty bounds;
                # q1 CV's padding owner still publishes empty output/status.
                # count=0 denotes a nonleader, not an actual padding witness.
                if row[1] != -1 or tuple(row[3:6]) != (0, 0, 0) or row[8] or row[9]:
                    raise ValueError("q1 padding task became live")
                continue
            context = (1344, 1376)[token]
            bounds = ((64, context + 1 - 256), (0, 319), (context, context + 1))[source]
            if (row[1], row[3], row[4], row[5], row[8], row[9]) != (
                    1, *bounds, token, context, token):
                raise ValueError("q1 full source/owner/tail domain changed")


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
    case = args.case
    if case == Q1_CASE and args.cpu_reference != "required":
        raise ValueError("new q1 qualification requires completed CPU reference")
    save(plan, {"case": case, "arm": args.arm, "source_sha256": generate.PINS[generate.SOURCE],
        "manifest_sha256": digest(generate.HERE / "manifest.json"), "runner_sha256": digest(__file__),
        "library_sha256": digest(args.library), "hard_limit_seconds": 600,
        "per_file_trace_cap": collector.CAP, "cpu_reference_policy": args.cpu_reference,
        "config_difference": "only doNorm/doMultiDataLoad; public API generated tiling"})
    argv = [replay.__file__, "--root", str(root), "--output", str(output), "--case", case,
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
        task_bytes = (args.export_root / case / "tasks.bin").read_bytes()
        geometry(report, task_bytes)
        if case == Q1_CASE:
            (output / "task_witness.bin").write_bytes(task_bytes)
        trace = collector.capture(output, report, task_bytes)
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
    correction = path / 'postprocess-correction.json'
    if correction.exists():
        from csrc.experiments.decode_mdl_load.recover_q1_postprocess import validate_correction
        report = validate_correction(path, control)
    else:
        report = numeric_checked(path, control)
    geometry(report, (path / "task_witness.bin").read_bytes() if report["case"] == Q1_CASE else None)
    receipt = json.loads(control.read_text())
    if receipt["timeout_seconds"] != 600:
        raise ValueError("changed predeclared deadline")
    plan = path.with_name(path.name + ".plan.json")
    binding = json.loads(plan.read_text())
    if (report.get("mdl_complete_trace", {}).get("status") != "passed" or
        digest(plan) != report["mdl_complete_trace"]["plan_sha256"] or
        binding["source_sha256"] != generate.PINS[generate.SOURCE] or binding["arm"] != arm or
        binding["case"] != report["case"] or
        binding["library_sha256"] != report["library_sha256"] or
        report["kernel"] != f"oscar_mdl_{arm}_{report['mode']}_kernel" or report["profile_counters"]):
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
        "case": old["case"], "mode": old["mode"],
        "task_ticks": {"baseline": old["task_ticks"], "candidate": new["task_ticks"]},
        "ratio": old["task_ticks"] / new["task_ticks"], "all_partial_lse_status_bitwise": True,
        "minimum_local_speedup": 1.15,
        "local_speed_gate": "passed" if old["task_ticks"] / new["task_ticks"] >= 1.15 else "failed",
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
    arm.add_argument("--case", choices=(collector.CASE, Q1_CASE), default=collector.CASE)
    arm.add_argument("--cpu-reference", choices=("required", "optional"), default="required")
    pair = commands.add_parser("compare")
    for name in ("baseline", "candidate", "baseline-control", "candidate-control", "output"):
        pair.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    return run(args) if args.command == "run" else compare(args)


if __name__ == "__main__":
    main()
