"""Archive #145/#164: fixed CPU functional matrix, owned bounded processes.

No NPU, performance measurement or tolerance adjustment. First failure stops
the run; source cases and earlier logs remain intact.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.phase import run_phase, atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists():
        raise ValueError("use a fresh output directory; failed evidence is preserved")
    destination.mkdir(parents=True)
    fixture = ROOT / "reports/c16_targeted_opt_20261020/cpu_fixture.py"
    env = dict(os.environ, OSCAR_ENABLED="0", TORCH_DEVICE_BACKEND_AUTOLOAD="0",
               LD_LIBRARY_PATH="/usr/local/Ascend/cann-9.1.0/lib64:/usr/local/Ascend/cann-9.1.0/tools/tikicpulib/lib")
    report = {"status": "running", "cases": [], "phases": [], "NPU": "not_run", "performance": "not_run",
              "per_phase_timeout_seconds": {"prepare": 120, "oracle": 120, "functional": 300}}

    def phase(label, command, timeout):
        result = run_phase(label, command, cwd=ROOT, log_dir=destination / "logs",
                           env=env, timeout=timeout, grace=5, heartbeat=60)
        report["phases"].append(asdict(result))
        atomic_json(destination / "validation.json", report)
        if result.returncode or not result.cleanup_complete:
            raise RuntimeError(f"{label} failed rc={result.returncode}; log={result.log}")
        return Path(result.log)

    try:
        phase("make-fixtures", [sys.executable, str(fixture), "make", str(destination / "fixtures"),
                               *[value for name in args.case for value in ("--case", name)]], 120)
        cases = json.loads((destination / "fixtures/cases.json").read_text())["cases"]
        for case in cases:
            name = case["name"]
            inputs = destination / "fixtures" / name
            outputs = destination / "outputs" / name
            outputs.mkdir(parents=True)
            command = [str(args.binary.resolve()), "c16", str(inputs), str(outputs), "0"]
            phase(name + "-prepare", [*command, "prepare-only"], 120)
            phase(name + "-oracle", [sys.executable, str(fixture), "golden", str(inputs), "--output", str(outputs)], 120)
            log = phase(name + "-functional", command, 300)
            result = next(json.loads(line) for line in log.read_text().splitlines()
                          if line.startswith('{"backend":"ascendc_cpu_debug"'))
            if result["status"] != "passed" or result.get("paired_bitwise") is not True:
                raise RuntimeError("functional process omitted passing paired numerical evidence")
            report["cases"].append({"name": name, "fixture": case, "result": result})
            atomic_json(destination / "validation.json", report)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        atomic_json(destination / "validation.json", report)
    print(json.dumps({"status": report["status"], "cases": len(report["cases"]),
                      "performance": "not_run", "NPU": "not_run"}))


if __name__ == "__main__":
    main()
