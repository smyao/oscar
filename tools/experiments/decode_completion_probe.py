"""Archive #129/#132-135/#140/#145/#148/#154-156; startup D.4.

Complete, bounded diagnostic of the unchanged d817 per-page decode bundle.
R2/q4/C1341 covers four full history tiles plus a tail, every source/task,
and one AIC/two AIV. Scalar markers are NOT bare hardware execution times.
No kernel, SDK, production route, frozen tolerance, or old probe is changed.
Root supervises each run with run_camodel_arm.py, 600 seconds per arm.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import stat
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CASE = "q4_r2_n8_c1341_s1"
CAP = 64 * 1024 * 1024
ENGINES = ("cubecore0", "veccore0", "veccore1")
WORKSPACE = 1146880
SOURCE_SHA = "d8171f0403cf176087e1c76c64e454e6fb1e74fd79252aab3b59e848cca19401"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(65536):
            result.update(chunk)
    return result.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def fixture(output):
    from csrc.experiments.window_batch_decode import fixture as shared

    if output.exists():
        raise ValueError("refuse to overwrite fixture")
    selected = shared.Case(CASE, "q4", (1341, 1341), padded=8, splits=1, cores=1)
    previous = shared.case_plan
    shared.case_plan = lambda *_: (selected,)
    try:
        manifest = shared.export(output, cores=1)
    finally:
        shared.case_plan = previous
    row = manifest["cases"][0]
    if row["source_work"]["kv256_units"] != {"history": 10, "window": 4, "current": 2}:
        raise ValueError("unexpected source bounds")
    manifest.update(baseline="same d817 source normal/profile; diagnostic, not optimization",
                    measurement_scope="R2_q4_N8_C1341_S1_one_cube_all_sources_and_tasks")
    write(output / "cases.json", manifest)


def geometry(report):
    a = report["attrs"]
    if (report["case"], report["mode"], report["cores"], a[0], a[1], a[2], a[3], a[4], a[6], a[-1]) != (
            CASE, "q4", 1, 8, 6, 1, 256, 2, 24, 1):
        raise ValueError("not the declared small complete fixture")
    if not report["all_sources_and_tasks"] or report["task_filtering"]:
        raise ValueError("task filtering prohibited")


def expected_counts(report, tasks, profile):
    """Use actual prepare output, not guessed counts or the R32-only gate."""
    geometry(report)
    if len(tasks) != 24 * 128 or hashlib.sha256(tasks).hexdigest() != report["input_sha256"]["tasks"]:
        raise ValueError("task witness identity")
    if len(profile) != 2304 or hashlib.sha256(profile).hexdigest() != report["outputs"]["profile"]["sha256"]:
        raise ValueError("profile witness identity")
    rows = list(struct.iter_unpack("<16q", tasks))
    values = struct.unpack("<288q", profile)
    if any(v < 0 for v in values):
        raise ValueError("profile poison or negative counts")
    expected = {}
    for engine, name in enumerate(ENGINES):
        counts = Counter({(3, 23): 1})
        work = [[0, 0, 0] for _ in range(3)]
        # CvTaskSchedule: all six source/query work items and all 24 slots.
        for wid in range(6):
            begin, segment = (wid % 2) * 4, wid // 2
            for token in range(begin, begin + 4):
                row = rows[token * 3 + segment]
                qcount, first, last, source = row[1], row[3], row[4], row[7]
                counts[3, 20] += 1
                if row[10] or source not in (0, 1, 2) or last < first:
                    raise ValueError("invalid task in valid fixture")
                if qcount == 0:
                    continue
                if qcount != 4 or row[8] != 1341:
                    raise ValueError("unexpected request leader/context")
                if source == 0 and (first, last) != (64, 1089):
                    raise ValueError("history must retain four full tiles plus one row")
                units = (last - first + 255) // 256
                work[source][0] += 1
                work[source][1] += units
                work[source][2] += last - first
                counts[source, 3] += 1
                if engine == 0:
                    for field in (13, 14, 15, 16):
                        counts[source, field] += units
                    if source != 0:
                        counts[source, 17] += units
                    else:
                        counts[source, 18] += 1
                        counts[source, 22] += 2
                else:
                    for field in (4, 19, 21):
                        counts[source, field] += 1
                    for field in (9, 10, 11, 12):
                        counts[source, field] += units
                    if source == 0:
                        counts[source, 5] += 2 * units
                        counts[source, 6] += 4 * units
                        counts[source, 8] += 4 * units
                        counts[source, 22] += 1
                    else:
                        live = sum(start + sub * 128 + (engine - 1) * 64 < last
                                   for start in range(first, last, 256) for sub in (0, 1))
                        counts[source, 7] += 2 * live
                        counts[source, 8] += 2 * live
        if [r[:2] for r in work] != [[2, 10], [2, 4], [2, 2]]:
            raise ValueError("incomplete source work")
        for source in range(4):
            want = work[source] if source < 3 else [sum(r[k] for r in work) for k in range(3)]
            offset = (engine * 4 + source) * 24
            if list(values[offset:offset + 3]) != want:
                raise ValueError("profile counters differ from actual task ownership")
        expected[name] = +counts
    return expected


def verify_gzip(path, size, expected_sha):
    h = hashlib.sha256()
    count = 0
    with gzip.open(path, "rb") as stream:
        while chunk := stream.read(65536):
            count += len(chunk)
            if count > CAP:
                raise ValueError("gzip exceeds declared trace cap")
            h.update(chunk)
    if count != size or h.hexdigest() != expected_sha:
        raise ValueError("saved gzip does not reproduce complete raw trace")


def capture(output, report, tasks):
    from csrc.experiments.window_batch_decode.replay_camodel import marker_events
    from tools.experiments.stream_camodel_markers import Intervals

    enabled = report["profile_counters"]
    expected = expected_counts(report, tasks, (output / "profile.bin").read_bytes()) if enabled else None
    known = {r["source"]: r for r in report["instruction_samples"]}
    names = [f"core0.{engine}.{kind}.dump" for engine in ENGINES for kind in ("instr_log", "instr_popped_log")]
    rawdir, savedir = output / "log_ca", output / "full_instructions"
    savedir.mkdir(exist_ok=False)
    actual = {p.name for p in rawdir.glob("core*.instr*dump*") if p.stat().st_size}
    if actual != set(names):
        raise ValueError("missing, extra, or rotated instruction files")
    evidence = []
    for name in names:
        source = rawdir / name
        sample = known.get(name)
        if not sample or not sample["complete_for_actor"] or sample["prefix_capped"]:
            raise ValueError("replayer reached cap or did not retain full trace: " + name)
        intervals = Intervals()
        h, count, events = hashlib.sha256(), 0, 0
        target = savedir / (name + ".gz")
        fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as raw, gzip.open(target, "wb", compresslevel=1) as compressed:
            before = os.fstat(raw.fileno())
            if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size < CAP:
                raise ValueError("trace not regular, empty, or cap reached")
            for line in raw:
                if len(line) > 65536 or not line.endswith(b"\n"):
                    raise ValueError("partial or oversized instruction line")
                count += len(line)
                if count >= CAP:
                    raise ValueError("raw trace reached cap")
                h.update(line)
                compressed.write(line)
                for event in marker_events([line.decode("utf-8", errors="strict")]):
                    intervals.accept(event)
                    events += 1
            after = os.fstat(raw.fileno())
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or count != before.st_size:
            raise ValueError("trace changed after SDK finalization")
        if count != sample["source_bytes"] or count != sample["captured_bytes"]:
            raise ValueError("full file differs from replayer completion witness")
        engine = name.split(".")[1]
        if enabled:
            intervals.finish()
            if Counter({key: value["count"] for key, value in intervals.groups.items()}) != expected[engine]:
                raise ValueError("marker calls differ from actual task/profile witness")
        elif events:
            raise ValueError("normal symbol unexpectedly emitted profile markers")
        verify_gzip(target, count, h.hexdigest())
        evidence.append({"source": name, "path": str(target.relative_to(output)), "raw_bytes": count,
                         "raw_sha256": h.hexdigest(), "gzip_sha256": digest(target), "complete": True,
                         "marker_events": events, "marker_closure": "passed" if enabled else "not_instrumented"})
        # Byte-identical compressed copy is verified before removing owned raw.
        source.unlink()
    result = {"status": "passed", "files": evidence, "per_file_cap_bytes": CAP,
              "total_raw_bytes": sum(r["raw_bytes"] for r in evidence), "actor_count": 3,
              "method": "regular_files_after_SDK_finalization_all_bytes_gzip_verified",
              "interpretation": "markers locate submission scopes; correlate popped/completion instructions before attribution"}
    write(output / "full_instructions.json", result)
    return result


def run(args):
    from csrc.experiments.request_window_decode_profile import generate
    from csrc.experiments.window_batch_decode import replay_camodel as replay

    root, output = args.root.resolve(), args.output.resolve()
    if root != ROOT or output.exists():
        raise ValueError("run from the declared project root with a fresh output directory")
    generate.generate(False)
    generate.generate(True)
    if digest(root / "csrc/kernels/attention_cv_decode_bundle.cpp") != SOURCE_SHA:
        raise ValueError("production source drift")
    plan = output.with_name(output.name + ".plan.json")
    if plan.exists():
        raise ValueError("refuse repeated run plan")
    plan.parent.mkdir(parents=True, exist_ok=True)
    write(plan, {"case": CASE, "arm": args.arm, "source_sha256": SOURCE_SHA,
                 "source_dependencies": generate.PINS, "probe_sha256": digest(__file__),
                 "replayer_sha256": digest(replay.__file__), "library_sha256": digest(args.library),
                 "config_sha256": {p: digest(args.simulator_config_dir / p) for p in ("config.json", "config_stars.json")},
                 "all_sources_and_tasks": True, "task_filtering": False, "cores": 1,
                 "hard_limit_seconds": 600, "per_file_cap_bytes": CAP, "files": 6,
                 "maximum_raw_bytes": CAP * 6, "maximum_instrumentation_perturbation": .05,
                 "cpu_attention": "not_run_prepare_only", "npu": "not_run", "whole_service": "not_established"})
    argv = [replay.__file__, "--root", str(root), "--output", str(output), "--case", CASE,
            "--variant", "baseline" if args.arm == "normal" else "candidate",
            "--fixture-root", str(args.fixture_root.resolve()), "--export-root", str(args.export_root.resolve()),
            "--library", str(args.library.resolve()), "--simulator-config-dir", str(args.simulator_config_dir.resolve()),
            "--baseline-q4", "oscar_bundle_normal_q4_kernel", "--candidate-q4", "oscar_bundle_profile_q4_kernel",
            "--baseline-workspace-bytes-per-core", str(WORKSPACE), "--candidate-workspace-bytes-per-core", str(WORKSPACE),
            "--cpu-reference", "optional", "--instruction-budget-bytes", str(CAP)]
    if args.arm == "profile":
        argv += ["--profile-counters", "--require-phase-markers"]
    # The old 1 ms FD watcher remains unchanged. This child-only file limit
    # also bounds bytes between watcher polls; SIGXFSZ is a real failed arm,
    # never permission to keep a capped prefix as a complete instruction log.
    old_limit = resource.getrlimit(resource.RLIMIT_FSIZE)
    if old_limit[1] != resource.RLIM_INFINITY and old_limit[1] < CAP:
        raise ValueError("child hard file limit is below predeclared trace cap")
    resource.setrlimit(resource.RLIMIT_FSIZE, (CAP, old_limit[1]))
    previous = sys.argv
    try:
        sys.argv = argv
        rc = replay.main()
        report = json.loads((output / "report.json").read_text())
        if rc or report["status"] != "passed" or report["failures"]:
            raise ValueError("original full numerical/actor/trace gate failed")
        geometry(report)
        trace = capture(output, report, (args.export_root / CASE / "tasks.bin").read_bytes())
        report["bounded_trace_probe"] = {"status": "passed", "trace_report": "full_instructions.json",
                                         "plan_sha256": digest(plan), "total_raw_bytes": trace["total_raw_bytes"]}
        write(output / "report.json", report)
        print("DECODE_COMPLETION_TRACE", json.dumps(report["bounded_trace_probe"]), flush=True)
    except Exception as error:
        if output.exists():
            write(output / "bounded_trace_failure.json", {"status": "failed", "reason": repr(error)})
        raise
    finally:
        sys.argv = previous
        resource.setrlimit(resource.RLIMIT_FSIZE, old_limit)


def checked(path, control_path, arm):
    control = json.loads(control_path.read_text())
    if (control.get("status") != "completed" or control.get("child_returncode") != 0 or
            control.get("timed_out") or control.get("interrupt_signal") or
            control.get("supervisor_returncode") != 0 or control.get("cleanup_complete") is not True or
            control.get("timeout_seconds") != 600):
        raise ValueError("supervisor completion/deadline gate: " + arm)
    report = json.loads((path / "report.json").read_text())
    complete = json.loads((path / "completion.json").read_text())
    geometry(report)
    if (report["status"] != "passed" or report["failures"] or
            report.get("bounded_trace_probe", {}).get("status") != "passed" or
            complete["status"] != "device_complete_outputs_saved" or complete["failures"]):
        raise ValueError("incomplete numerical or bounded-trace evidence")
    if report["kernel"] != f"oscar_bundle_{arm}_q4_kernel" or report["profile_counters"] != (arm == "profile"):
        raise ValueError("wrong symbol/profile ABI")
    plan_path = path.with_name(path.name + ".plan.json")
    plan = json.loads(plan_path.read_text())
    if (digest(plan_path) != report["bounded_trace_probe"]["plan_sha256"] or
            plan["source_sha256"] != SOURCE_SHA or plan["library_sha256"] != report["library_sha256"] or
            plan["arm"] != arm or plan["case"] != CASE or plan["hard_limit_seconds"] != 600):
        raise ValueError("run plan/source/library binding")
    command = control.get("command", [])
    if "--arm" not in command or command[command.index("--arm") + 1] != arm or "--output" not in command:
        raise ValueError("supervisor command does not bind this arm")
    # Artifacts may have been copied from the VM; compare the original recorded
    # output to owner.json rather than requiring the host copy's absolute path.
    owner = json.loads((path / "owner.json").read_text())
    if command[command.index("--output") + 1] != owner["output"]:
        raise ValueError("supervisor supervised a different output directory")
    for name, item in report["outputs"].items():
        p = path / item["path"]
        if p.stat().st_size != item["bytes"] or digest(p) != item["sha256"] or complete["outputs"][name] != item:
            raise ValueError("saved output identity: " + name)
    if report["oracle"]["bad_output"] or report["oracle"]["bad_lse"] or report["task_ticks"] <= 0:
        raise ValueError("oracle/task timing gate")
    for kind, count in (("task", 1), ("aic", 1), ("aiv", 2)):
        rows = report["profiles"][kind]
        if len(rows) != count or any(r["duration"] <= 0 or r["end_cycle"] - r["start_cycle"] != r["duration"] for r in rows):
            raise ValueError("incomplete actors")
    trace = json.loads((path / "full_instructions.json").read_text())
    names = {f"core0.{engine}.{kind}.dump" for engine in ENGINES for kind in ("instr_log", "instr_popped_log")}
    if trace["status"] != "passed" or len(trace["files"]) != 6 or {r["source"] for r in trace["files"]} != names:
        raise ValueError("incomplete full instruction set")
    for row in trace["files"]:
        p = path / row["path"]
        if not row["complete"] or digest(p) != row["gzip_sha256"]:
            raise ValueError("full trace identity")
        verify_gzip(p, row["raw_bytes"], row["raw_sha256"])
    return report


def compare(args):
    normal = checked(args.normal, args.normal_control, "normal")
    profile = checked(args.profile, args.profile_control, "profile")
    for key in ("case", "mode", "cores", "attrs", "scale", "input_sha256", "oracle_sha256",
                "cpu_baseline_sha256", "library_sha256", "simulator_configuration", "cpu_reference",
                "cpu_reference_policy", "workspace_bytes_per_core", "trace_policy"):
        if normal[key] != profile[key]:
            raise ValueError("normal/profile " + key + " differs")
    for name in ("partial", "lse", "status"):
        if normal["outputs"][name]["sha256"] != profile["outputs"][name]["sha256"]:
            raise ValueError("normal/profile bits differ: " + name)
    fraction = profile["task_ticks"] / normal["task_ticks"] - 1
    result = {"status": "passed" if abs(fraction) <= .05 else "failed", "case": CASE,
              "normal_ticks": normal["task_ticks"], "profile_ticks": profile["task_ticks"],
              "instrumentation_perturbation": fraction, "predeclared_absolute_limit": .05,
              "all_partial_lse_status_bitwise": True, "independent_oracle": "passed_both",
              "all_sources_and_tasks": True, "actor_count": 3, "complete_trace_files_per_arm": 6,
              "scope": "small complete steady-pipeline diagnostic, not R32/HBM/whole-service throughput",
              "SYS_CNT": "not_used", "npu": "not_run", "graph": "not_run", "whole_service": "not_established"}
    write(args.output, result)
    print("DECODE_COMPLETION_COMPARISON", json.dumps(result), flush=True)
    return 0 if result["status"] == "passed" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    f = commands.add_parser("fixture")
    f.add_argument("--output", type=Path, required=True)
    r = commands.add_parser("run")
    for name in ("root", "fixture-root", "export-root", "library", "simulator-config-dir", "output"):
        r.add_argument("--" + name, type=Path, required=True)
    r.add_argument("--arm", choices=("normal", "profile"), required=True)
    c = commands.add_parser("compare")
    for name in ("normal", "profile", "normal-control", "profile-control", "output"):
        c.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fixture":
        fixture(args.output)
    elif args.command == "run":
        run(args)
    else:
        return compare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
