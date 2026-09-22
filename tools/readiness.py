# 档案 #27/#98/#99/#103/#118：源码能力、二进制加载、设备完成和服务可用性分别记录。
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from .phase import atomic_json


def readiness_report() -> dict:
    from oscar_ascend.ops.contracts import SOURCE_CAPABILITIES, missing_capabilities
    missing = list(missing_capabilities(SOURCE_CAPABILITIES))
    return {"status": "incomplete", "implemented_operator_sources": sorted(SOURCE_CAPABILITIES),
            "missing_operator_capabilities": missing,
            "missing_integration": ["concrete_runtime_provider", "prefix_window_ownership", "mtp_commit_rollback",
                                    "fixed_graph_workspace", "tp4_service_probe", "npu_resource_release_probe"],
            "npu_build": "not_run", "device_completion": "not_run", "graph_capture": "not_run",
            "graph_replay": "not_run", "accuracy": "not_run", "performance": "not_run",
            "constraint_conflicts": ["H18 sublinear total history reads conflicts with exact dense attention"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reports/readiness.json"))
    args = parser.parse_args()
    report = readiness_report()
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 1  # An explicit incomplete report must not unlock the production service.


if __name__ == "__main__":
    sys.exit(main())
