"""Archive #157: bounded real-NPU graph replay + original-Trap negative gate.

Each injected fault has its own disposable process/context. No model/AISBench,
CPU substitute, skipped guard, eager substitute or numerical acceptance claim.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

from .phase import atomic_json, run_phase
from .target_cli import ROOT, target_env

GROUPS = ("attention", "rotate", "merge", "store")


def child(group):
    import torch
    import torch_npu  # noqa: F401: register real NPU backend
    from oscar_ascend.ops.loader import require_capabilities
    from oscar_ascend.ops.meta import register_meta
    from oscar_ascend.status_diagnostics import status_guard
    if not torch.npu.is_available():
        raise RuntimeError("status diagnostic probe requires a real NPU")
    torch.npu.set_device(0)
    require_capabilities({"status_guard", "status_guard_diagnostic"})
    register_meta()
    groups = [torch.zeros(2049, dtype=torch.int32, device="npu") for _ in GROUPS]
    def step():
        status_guard(torch.ops.oscar_ascend_ops, *groups,
                     context={"layer": "synthetic", "phase": "injected_" + group})
    for _ in range(3):
        step()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        step()
    for _ in range(2):
        graph.replay()
        torch.npu.synchronize()
    print("OSCAR_DIAGNOSTIC_HEALTHY_REPLAYS=2", flush=True)
    groups[GROUPS.index(group)][1300] = 7
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()  # MUST fail after printing this replay's error packet.
    raise RuntimeError("original status_guard failed to Trap on injected nonzero status")


def evidence(text, result, group):
    records = []
    for line in text.splitlines():
        if line.startswith("OSCAR_FIRST_ERROR "):
            try:
                record = json.loads(line.removeprefix("OSCAR_FIRST_ERROR "))
            except json.JSONDecodeError:
                return False, records  # partial/interleaved evidence is NOT a passed negative gate
            if not isinstance(record, dict):
                return False, records
            records.append(record)
    matched = [r for r in records if r.get("group") == group and r.get("index") == 1300
               and r.get("code") == 7 and r.get("nonzero_count") == 1
               and r.get("size") == 2049 and r.get("callsite_invocation", 0) >= 3]
    return (result.returncode != 0 and not result.timed_out and not result.interrupted
            and result.cleanup_complete and bool(matched)
            and "OSCAR_DIAGNOSTIC_HEALTHY_REPLAYS=2" in text
            and "OSCAR_STATUS_DIAGNOSTICS_ACTIVE " in text
            and "original status_guard failed to Trap" not in text), records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/target.json")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs/status-diagnostic-probe")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/status_diagnostic_probe.json")
    parser.add_argument("--child", choices=GROUPS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        child(args.child)
        return 1
    config = json.loads(args.config.read_text())
    env = target_env(config)
    env.update(OSCAR_STATUS_DIAGNOSTICS="1", PYTHONUNBUFFERED="1")
    report = {"status": "running", "physical_device": config["devices"][0], "cases": [],
              "scope": "synthetic diagnostic graph/Trap only; NOT model or performance acceptance"}
    atomic_json(args.output, report)
    for group in GROUPS:
        result = run_phase("diagnostic-" + group,
            [sys.executable, "-m", "tools.probe_status_diagnostics", "--child", group],
            cwd=ROOT, log_dir=args.log_dir, env=env, timeout=90, grace=5, heartbeat=15)
        passed, records = evidence(Path(result.log).read_text(errors="replace"), result, group)
        report["cases"].append({"group": group, "passed": passed, "phase": asdict(result), "errors": records})
        report["status"] = "running" if passed else "failed"
        atomic_json(args.output, report)
        if not passed:
            print("FAILED status diagnostic gate; do not continue to service", flush=True)
            return 1
    report["status"] = "passed"
    atomic_json(args.output, report)
    print("STATUS_DIAGNOSTIC_PROBE passed (four expected Trap exits with live replay evidence)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
