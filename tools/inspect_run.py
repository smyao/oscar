# Archive #51/#68/#118/#122/#129: read only this run's owned processes, progress and CANN logs.
"""Collect stall evidence without starting a model, touching an NPU or sending signals."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from .phase import atomic_json
from .plog import OwnedProcessGroup, attach_plog
from .target_cli import ROOT


def tail(path, limit=65536):
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell()-limit))
        return stream.read(limit).decode("utf-8", errors="replace")


def inspect(run_dir, *, root=ROOT):
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_relative_to((Path(root)/"logs").resolve()) or not run_dir.is_dir():
        raise ValueError("run directory must be inside this project's logs directory")
    report = {"run_dir": str(run_dir), "read_only": True, "device_completion": "not_established",
              "processes": [], "requests": [], "worker_progress": [], "worker_trace_tail": []}
    lifecycle = run_dir/"service-probe/server_lifecycle.json"
    server_dir = lifecycle.parent
    if not lifecycle.is_file():
        lifecycle = run_dir/"serve/server_lifecycle.json"
        server_dir = lifecycle.parent
    state = json.loads(lifecycle.read_text())
    report["server"] = state
    ledger = OwnedProcessGroup(state["pid"])
    ledger.pids.update(p for p in state.get("owned_pids", []) if type(p) is int and p > 0)
    ledger.refresh(force=True)
    report["owned_pids"] = sorted(ledger.pids)
    processes = subprocess.run(["ps", "-o", "pid=,ppid=,pgid=,stat=,etime=,comm=", "-p",
                                ",".join(map(str, sorted(ledger.pids)))], capture_output=True, text=True, timeout=2)
    report["processes"] = processes.stdout.splitlines()
    for path in sorted((server_dir/"requests").glob("*/status.json"))[:16]:
        report["requests"].append(json.loads(path.read_text()))
    traces = sorted(server_dir.glob("trace-*"))[-1:]
    for directory in traces:
        for path in sorted(directory.glob("phase-*.json"))[:16]:
            report["worker_progress"].append(json.loads(path.read_text()))
        for path in sorted(directory.glob("worker-*.jsonl"))[:16]:
            report["worker_trace_tail"].append({"file": str(path), "tail": tail(path, 8192)})
    report["server_log_tail"] = tail(server_dir/"server.log")
    try:
        started = datetime.strptime(run_dir.name, "%Y%m%dT%H%M%S.%fZ").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        started = lifecycle.stat().st_mtime
    attach_plog(report, started_at=started, owned_pids=ledger.pids)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    report = inspect(args.run_dir)
    output = args.run_dir.resolve()/"stall-inspection.json"
    atomic_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[oscar] inspection saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
