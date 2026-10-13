"""Archive #94/#95/#126/#129/#145/#148/#154-157; complete D.4.

Qualify the full H16 to H32/member-pipeline composition, preserving all1008
tasks and the current source bindings. CPU120s/arm, CAModel900s/arm are fixed;
no partial time, retry, numerical relaxation, or production change.
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
from csrc.experiments.c4_composed_long import cv_compare, generate


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def bound(root, output, label, seconds, command, env=None):
    control = output / (label + "-control.json")
    # CPU-debug may put checker/GM temporaries beneath its cwd. Keep those
    # with this arm's /tmp output, not on the staged source tmpfs. All entry
    # points, inputs and outputs below are absolute; imports bind script ROOT.
    work = output / (label + "-work")
    work.mkdir(parents=True, exist_ok=False)
    args = [sys.executable, str(root / "tools/experiments/run_camodel_arm.py"),
            "--cwd", str(work), "--output", str(output / (label + "-logs")),
            "--control-path", str(control), "--timeout", str(seconds)]
    if env is not None:
        args += ["--env-json", str(env)]
    print("C4_COMPOSED_START", label, "deadline_s=" + str(seconds), flush=True)
    child = subprocess.run([*args, "--", *command], cwd=root, check=False)
    receipt = json.loads(control.read_text())
    if (child.returncode or receipt["status"] != "completed" or receipt["timed_out"] or
            receipt["child_returncode"] != 0 or not receipt["cleanup_complete"]):
        raise RuntimeError("bounded phase failed: " + str(control))
    return control


def qualify(args):
    root, output = args.root.resolve(), args.output.resolve()
    if root != ROOT or os.environ.get("LD_PRELOAD"):
        raise ValueError("run from declared stage, without supervisor LD_PRELOAD")
    output.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, str(generate.HERE / "generate.py"), "--check"],
                   cwd=root, check=True)
    expected = generate.build_binding()
    device_binding = json.loads((args.library.resolve().parent.parent / "source_binding.json").read_text())
    cpu_binding = json.loads((args.cpu_executable.resolve().parent / "source_binding.json").read_text())
    if device_binding != expected or cpu_binding != expected:
        raise ValueError("CPU/device source binding differs from declared full composition")
    fixture = args.fixture.resolve()
    fixture_hashes = {p.name: digest(p) for p in fixture.iterdir() if p.is_file()}
    save(output / "plan.json", {"source_binding": expected,
        "cpu_executable_sha256": digest(args.cpu_executable), "library_sha256": digest(args.library),
        "fixture_sha256": fixture_hashes, "CPU_deadline_s_per_arm": 120,
        "CAModel_deadline_s_per_arm": 900, "minimum_full_path_speedup": 1.15,
        "all_tasks": 1008, "qualification_runner_sha256": digest(__file__)})
    cpu_controls = {}
    for arm in ("baseline", "candidate"):
        label = "cpu-" + arm
        cpu_controls[arm] = bound(root, output, label, 120,
            [str(args.cpu_executable.resolve()), str(fixture), str(output / label), arm])
    cpu = cv_compare.compare(output / "cpu-baseline", output / "cpu-candidate",
                             cpu_controls["baseline"], cpu_controls["candidate"])
    save(output / "cpu-comparison.json", cpu)
    if fixture_hashes != {p.name: digest(p) for p in fixture.iterdir() if p.is_file()}:
        raise ValueError("CPU changed frozen fixture")
    controls = {}
    for arm in ("baseline", "candidate"):
        controls[arm] = bound(root, output, arm, 900,
            [sys.executable, str(generate.HERE / "cv_replay.py"), "--variant", arm,
             "--fixture", str(fixture), "--prepared", str(output / ("cpu-" + arm)),
             "--library", str(args.library.resolve()), "--output", str(output / arm)],
            args.env_json.resolve())
    result = cv_compare.compare(output / "baseline", output / "candidate",
                                controls["baseline"], controls["candidate"])
    save(output / "comparison.json", result)
    passed = result["combination_gate"] == "passed"
    save(output / "summary.json", {"status": "passed" if passed else "insufficient_gain",
        "CPU": "complete_bitwise_oracle_all1008tasks", "CAModel": result,
        "NPU": "not_run", "graph": "not_run", "production_enabled": False,
        "whole_service": "not_established"})
    print("C4_COMPOSED_RESULT", json.dumps(result), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "output", "fixture", "cpu-executable", "library", "env-json"):
        parser.add_argument("--" + name, type=Path, required=True)
    raise SystemExit(qualify(parser.parse_args()))
