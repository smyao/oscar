"""Archive #126/#145/#164: exercise a second history tile in the complete reader.

D.4: this is an offline bounded fixture, not production history restoration.
The existing independent writer/reference fixture is reused with only context
330 -> 850. With sink64 and S3 each history interval exceeds256, forcing the
PrefetchKeys/Values path. Both readers consume identical bytes and tasks.
No timeout increase, performance experiment, or numerical tolerance change.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "reports/long_context_20261027/decode"


def make(folder, family):
    import torch
    from csrc.experiments.official_math_20261021 import fixture
    torch.set_num_threads(1)
    original = inspect.getsource(fixture.make_decode)
    anchor = 'd, active, hk, context = 256, 4 if family == "q4" else 1, 2, 330'
    if original.count(anchor) != 1:
        raise RuntimeError("independent decode fixture context anchor drift")
    changed = original.replace(anchor, anchor.removesuffix("330") + "850")
    namespace = dict(vars(fixture))
    exec(compile(changed, "<independent-context850-fixture>", "exec"), namespace)
    namespace["make_decode"](folder, family, False)
    (folder / "context_override.json").write_text(json.dumps({
        "context": 850, "base_fixture_function_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "changed_fixture_function_sha256": hashlib.sha256(changed.encode()).hexdigest(),
        "only_change": "context 330 -> 850", "history_tiles_per_split": 2,
    }, indent=2) + "\n")


def run():
    from tools.phase import run_phase, atomic_json
    from csrc.experiments.long_context_20261027.validate_decode_pair import clean_completed_sdk_debug
    report_path = OUT / "long_validation.json"
    if report_path.exists():
        raise RuntimeError("preserve existing long validation")
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    report = {"status": "running", "scope": "complete q4/q1 mode0 two-history-tile CPU pair",
              "npu": "not_run", "graph": "not_run", "performance": "not_run", "phases": [], "cases": []}

    def phase(name, command):
        result = run_phase(name, command, cwd=ROOT, log_dir=OUT / "logs", timeout=180)
        report["phases"].append({"name": name, "returncode": result.returncode, "log": result.log,
            "timed_out": result.timed_out, "cleanup_complete": result.cleanup_complete})
        atomic_json(report_path, report)
        if result.returncode:
            raise RuntimeError(name + ": rc=" + str(result.returncode))
        report["phases"][-1]["completed_sdk_debug_removed_bytes"] = clean_completed_sdk_debug()
        atomic_json(report_path, report)

    try:
        for family in ("q4", "q1"):
            name = family + "_mode0_context850"
            fixture, golden = OUT / "fixtures" / name, OUT / "goldens" / name
            golden.mkdir(parents=True, exist_ok=False)
            phase(name + "-fixture", [sys.executable, "-m", "csrc.experiments.long_context_20261027.decode_long_pair", "make", family, str(fixture)])
            args = [family, str(fixture), str(golden), "0"]
            before = ROOT / "build/decode-pair-cpu/oscar_decode_before"
            after = ROOT / "build/decode-pair-cpu/oscar_decode_after"
            phase(name + "-prepare", [str(before), *args, "prepare-only"])
            phase(name + "-oracle", [sys.executable, "-m", "csrc.experiments.official_math_20261021.fixture", "golden", family, str(fixture), "--output", str(golden), "--mode", "0"])
            phase(name + "-before", [str(before), *args])
            files = ("partial.bin", "lse.bin", "status.bin", "output.bin", "output_lse.bin")
            old = {file: (golden / file).read_bytes() for file in files}
            phase(name + "-after", [str(after), *args])
            comparison = {file: {"bytes": len(data), "before_sha256": hashlib.sha256(data).hexdigest(),
                "after_sha256": hashlib.sha256((golden / file).read_bytes()).hexdigest(),
                "bitwise_equal": data == (golden / file).read_bytes()} for file, data in old.items()}
            if not all(row["bitwise_equal"] for row in comparison.values()):
                for file, data in old.items():
                    (golden / ("before_" + file)).write_bytes(data)
                atomic_json(OUT / (name + "_mismatch.json"), comparison)
                raise RuntimeError(name + ": paired bytes differ")
            report["cases"].append({"case": name, "status": "passed", "files": comparison,
                "independent_v4_oracle": True, "context": 850, "history_tiles_per_split": 2,
                "same_fixture_and_prepared_tasks": True})
            atomic_json(report_path, report)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        atomic_json(report_path, report)
    print(json.dumps({"status": report["status"], "paired_cases": len(report["cases"]), "report": str(report_path)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("make", "run"))
    parser.add_argument("family", nargs="?", choices=("q4", "q1"))
    parser.add_argument("folder", nargs="?", type=Path)
    args = parser.parse_args()
    if args.action == "make": make(args.folder, args.family)
    else: run()
