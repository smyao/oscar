"""Archive #94/#95/#129/#140/#145/#148/#154-157; D.4 whole-path evidence.

Reuse existing compiled q1 Normal/MDL symbols. Sequential bounded CPU cases,
then the complete same-library CAModel pair. Each child uses the existing
exclusive lock/deadline/cleanup supervisor; any failure stops qualification.
No production change, no retry, no tolerance override and no NPU claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from csrc.experiments.decode_mdl_load import generate, run
from csrc.experiments.decode_mdl_load.q1_fixture import CASES


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def bound(root, output, label, seconds, command, env=None):
    control = output / (label + "-control.json")
    work = output / (label + "-work")
    work.mkdir()
    args = [sys.executable, str(root / "tools/experiments/run_camodel_arm.py"),
            "--cwd", str(work), "--output", str(output / (label + "-logs")),
            "--control-path", str(control), "--timeout", str(seconds)]
    if env is not None:
        args += ["--env-json", str(env)]
    print("Q1_MDL_START", label, "deadline_s=" + str(seconds), flush=True)
    child = subprocess.run([*args, "--", *command], cwd=root, check=False)
    receipt = json.loads(control.read_text())
    if (child.returncode or receipt["status"] != "completed" or receipt["timed_out"] or
            receipt["child_returncode"] != 0 or not receipt["cleanup_complete"]):
        raise RuntimeError("bounded phase failed: " + str(control))
    return control, receipt


def qualify(args):
    root, output = args.root.resolve(), args.output.resolve()
    if root != ROOT or os.environ.get("LD_PRELOAD"):
        raise ValueError("run from declared stage, without supervisor LD_PRELOAD")
    output.mkdir(parents=True, exist_ok=False)
    source_check = subprocess.run([sys.executable, str(generate.HERE / "generate.py"), "--check"],
                                  cwd=root, check=False)
    if source_check.returncode:
        raise ValueError("source restoration gate failed")
    manifest = json.loads((generate.HERE / "manifest.json").read_text())
    binding = json.loads((args.library.resolve().parent.parent / "source_binding.json").read_text())
    cpu_binding = json.loads((args.cpu_executable.resolve().parent / "source_binding.json").read_text())
    if binding != cpu_binding or binding["manifest_sha256"] != digest(generate.HERE / "manifest.json"):
        raise ValueError("CPU/device built different Normal/MDL sources")
    if binding["generator_sha256"] != digest(generate.__file__):
        raise ValueError("compiled generator binding changed")
    for name, expected in binding["sources"].items():
        if digest(generate.HERE / name) != expected:
            raise ValueError("compiled source changed: " + name)
    fixture_root = args.fixture_root.resolve()
    fixture = json.loads((fixture_root / "cases.json").read_text())
    if [row["name"] for row in fixture["cases"]] != [case.name for case in CASES]:
        raise ValueError("missing full q1/error qualification cases")
    fixture_hashes = {str(p.relative_to(fixture_root)): digest(p) for p in fixture_root.rglob("*") if p.is_file()}
    save(output / "plan.json", {"source_sha256": generate.PINS[generate.SOURCE],
        "build_binding": binding, "cpu_executable_sha256": digest(args.cpu_executable),
        "library_sha256": digest(args.library), "fixture_sha256": fixture_hashes,
        "CPU_deadline_s_per_case": 120, "CAModel_deadline_s_per_arm": 600,
        "minimum_full_path_speedup": 1.15, "no_task_filtering": True,
        "fixture_generator_sha256": digest(generate.HERE / "q1_fixture.py"),
        "qualification_runner_sha256": digest(__file__)})
    exports = output / "exports"
    exports.mkdir()
    cpu = []
    for index, case in enumerate(CASES):
        destination = exports / case.name
        destination.mkdir()
        label = "cpu-" + str(index)
        control, receipt = bound(root, output, label, 120,
            [str(args.cpu_executable.resolve()), "q1", str(fixture_root / case.name), str(destination)])
        records = []
        for line in (output / (label + "-logs/stdout.log")).read_text().splitlines():
            if line.startswith('{"backend":"ascendc_cpu_debug"'):
                records.append(json.loads(line))
        if len(records) != 1:
            raise ValueError("CPU complete result missing: " + case.name)
        record = records[0]
        required = ("bitwise", "input_immutable", "gm_guards", "fault_order" if case.fault else "oracle")
        if record.get("status") != "passed" or any(record.get(key) is not True for key in required):
            raise ValueError("CPU q1 numeric/error domain failed: " + case.name)
        if record.get("op") != "mdl_load_q1" or record.get("cores") != 1:
            raise ValueError("CPU ran wrong symbol/geometry")
        for key in ("tasks", "partial", "lse", "status"):
            if not (destination / (key + ".bin")).is_file():
                raise ValueError("CPU complete bytes missing")
        save(destination / "report.json", record)
        cpu.append({"case": case.name, "control": str(control), "result": record,
                    "elapsed_seconds": receipt["elapsed_seconds"]})
    save(output / "cpu-summary.json", {"status": "passed", "cases": cpu})
    if fixture_hashes != {str(p.relative_to(fixture_root)): digest(p) for p in fixture_root.rglob("*") if p.is_file()}:
        raise ValueError("CPU changed fixture inputs")
    config = json.loads(args.env_json.read_text()).get("CAMODEL_CONFIG_PATH")
    if not config:
        raise ValueError("explicit simulator config required")
    controls = {}
    for arm in ("baseline", "candidate"):
        controls[arm], _ = bound(root, output, arm, 600,
            [sys.executable, str(generate.HERE / "run.py"), "run", "--root", str(root),
             "--case", run.Q1_CASE, "--arm", arm, "--fixture-root", str(fixture_root),
             "--export-root", str(exports), "--library", str(args.library.resolve()),
             "--simulator-config-dir", config, "--output", str(output / arm),
             "--cpu-reference", "required"], args.env_json.resolve())
    comparison = output / "comparison.json"
    run.compare(argparse.Namespace(baseline=output / "baseline", candidate=output / "candidate",
        baseline_control=controls["baseline"], candidate_control=controls["candidate"], output=comparison))
    result = json.loads(comparison.read_text())
    save(output / "summary.json", {"status": "passed" if result["local_speed_gate"] == "passed" else "insufficient_gain",
        "CPU": "complete_all_cases", "CAModel": result, "NPU": "not_run", "graph": "not_run",
        "production_enabled": False, "whole_service": "not_established"})
    return 0 if result["local_speed_gate"] == "passed" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "output", "fixture-root", "cpu-executable", "library", "env-json"):
        parser.add_argument("--" + name, type=Path, required=True)
    raise SystemExit(qualify(parser.parse_args()))
