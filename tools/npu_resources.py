# Archive G21/G22/G31/#51/#52/#68: selected-device memory evidence from a
# short-lived process, bounded release wait; no npu-smi parsing/reset/global kill.
"""Observe NPU memory without keeping a probe context alive in the supervisor."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

from .phase import run_phase
from .target_cli import ROOT, target_env

MARKER = "NPU_RESOURCE_JSON="
DEFAULT_RELEASE_TOLERANCE = 256 * 1024 * 1024


def _validate_snapshot(snapshot, devices):
    if (not isinstance(devices, list) or len(devices) != 4 or len(set(devices)) != 4
            or any(type(x) is not int or x < 0 for x in devices)):
        raise ValueError("snapshot needs the four explicitly selected physical NPUs")
    if not isinstance(snapshot, dict) or snapshot.get("devices") != devices:
        raise ValueError("NPU resource snapshot has the wrong physical device selection")
    rows = snapshot.get("memory")
    if not isinstance(rows, list) or len(rows) != len(devices):
        raise ValueError("NPU resource snapshot must cover all selected devices")
    for logical, (physical, row) in enumerate(zip(devices, rows)):
        if not isinstance(row, dict):
            raise ValueError("NPU resource row must be an object")
        if row.get("physical_device") != physical or row.get("logical_device") != logical:
            raise ValueError("NPU resource snapshot device order changed")
        free, total = row.get("free_bytes"), row.get("total_bytes")
        if type(free) is not int or type(total) is not int or not 0 <= free <= total or total <= 0:
            raise ValueError("NPU resource snapshot contains invalid byte counts")
    return snapshot


def collect_resources(devices):
    # Enforce current-task physical selection before importing either backend.
    os.environ.update(target_env({"devices": devices}))
    os.environ["OSCAR_ENABLED"] = "0"
    import torch
    import torch_npu  # noqa: F401
    if not torch.npu.is_available() or torch.npu.device_count() != len(devices):
        raise RuntimeError("the four explicitly selected NPUs are not available")
    memory = []
    for logical, physical in enumerate(devices):
        torch.npu.set_device(logical)
        free, total = torch.npu.mem_get_info()
        memory.append({"physical_device": physical, "logical_device": logical,
                       "device_name": torch.npu.get_device_name(logical),
                       "free_bytes": int(free), "total_bytes": int(total)})
    return _validate_snapshot({"devices": devices, "memory": memory, "wall_time": time.time(),
                               "backend": "torch_npu.mem_get_info"}, devices)


def read_npu_resources(config, *, log_dir: Path, timeout: float = 30):
    if timeout <= 0 or not math.isfinite(timeout):
        raise ValueError("NPU resource observation timeout must be positive and finite")
    environment = target_env(config)
    environment["OSCAR_ENABLED"] = "0"  # This observer constructs no model/plugin.
    phase = f"npu-memory-{time.monotonic_ns()}"
    result = run_phase(phase, [sys.executable, "-m", "tools.npu_resources", "--collect",
        "--devices", *map(str, config["devices"])], cwd=ROOT,
        log_dir=Path(log_dir), timeout=timeout, env=environment, grace=min(3, timeout))
    text = Path(result.log).read_text()
    records = [line[len(MARKER):] for line in text.splitlines() if line.startswith(MARKER)]
    if result.returncode != 0 or not result.cleanup_complete or len(records) != 1:
        raise RuntimeError(f"NPU resource observation failed rc={result.returncode}; log={result.log}")
    snapshot = _validate_snapshot(json.loads(records[0]), config["devices"])
    snapshot["log"] = result.log
    return snapshot


def compare_release(before, after, *, tolerance_bytes=DEFAULT_RELEASE_TOLERANCE):
    if type(tolerance_bytes) is not int or tolerance_bytes < 0:
        raise ValueError("release tolerance must be a nonnegative integer byte count")
    devices = before.get("devices")
    _validate_snapshot(before, devices)
    _validate_snapshot(after, devices)
    checks = []
    for first, last in zip(before["memory"], after["memory"]):
        deficit = first["free_bytes"] - last["free_bytes"]
        checks.append({"physical_device": first["physical_device"],
            "before_free_bytes": first["free_bytes"], "after_free_bytes": last["free_bytes"],
            "free_deficit_bytes": deficit, "total_unchanged": first["total_bytes"] == last["total_bytes"],
            "passed": first["total_bytes"] == last["total_bytes"] and deficit <= tolerance_bytes})
    return {"status": "passed" if all(x["passed"] for x in checks) else "failed",
            "tolerance_bytes": tolerance_bytes, "devices": checks,
            "scope": "resource release observation, not a performance or ownership attribution"}


def wait_for_release(config, before, *, log_dir: Path, timeout=30.0,
                     tolerance_bytes=DEFAULT_RELEASE_TOLERANCE, reader=None):
    if timeout <= 0 or not math.isfinite(timeout):
        raise ValueError("release wait must have a finite positive bound")
    reader = read_npu_resources if reader is None else reader
    deadline = time.monotonic() + timeout
    attempts = []
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {"status": "failed", "reason": "NPU memory did not return within the bounded release wait",
                    "attempts": attempts, "tolerance_bytes": tolerance_bytes}
        try:
            after = reader(config, log_dir=log_dir, timeout=min(30.0, remaining))
            check = compare_release(before, after, tolerance_bytes=tolerance_bytes)
            attempts.append({"snapshot": after, "check": check})
            if check["status"] == "passed":
                return {"status": "passed", "attempts": attempts, "tolerance_bytes": tolerance_bytes}
        except Exception as error:
            # A broken observer is a failure, not permission to substitute
            # a zero reading or spin indefinitely on an unrecognized field.
            return {"status": "failed", "reason": str(error), "attempts": attempts,
                    "tolerance_bytes": tolerance_bytes}
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true", required=True)
    parser.add_argument("--devices", nargs=4, type=int, required=True)
    args = parser.parse_args()
    try:
        print(MARKER + json.dumps(collect_resources(args.devices), allow_nan=False), flush=True)
        return 0
    except Exception as error:
        print(f"NPU_RESOURCE_FAILED: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
