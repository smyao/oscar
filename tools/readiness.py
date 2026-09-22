# 档案 #27/#98/#99/#103/#118：源码能力、二进制加载、设备完成和服务可用性分别记录。
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from .phase import atomic_json


def readiness_report(*, binary=False) -> dict:
    from oscar_ascend.ops.contracts import SOURCE_CAPABILITIES, missing_capabilities
    missing = list(missing_capabilities(SOURCE_CAPABILITIES))
    modules=("oscar_ascend.runtime","oscar_ascend.lifecycle","oscar_ascend.integration.impl",
             "oscar_ascend.integration.metadata","oscar_ascend.plugin")
    missing_modules=[name for name in modules if importlib.util.find_spec(name) is None]
    report={"status": "source_ready" if not missing and not missing_modules else "incomplete",
            "implemented_operator_sources": sorted(SOURCE_CAPABILITIES),
            "missing_operator_capabilities": missing,
            "missing_integration": missing_modules,
            "npu_build": "not_run", "device_completion": "not_run", "graph_capture": "not_run",
            "graph_replay": "not_run", "accuracy": "not_run", "performance": "not_run",
            "constraint_conflicts": ["H18 sublinear total history reads conflicts with exact dense attention"]}
    if binary:
        from oscar_ascend.ops.loader import validate_build_artifacts
        try:
            manifest=validate_build_artifacts()
        except (RuntimeError,OSError,ValueError) as error:
            report.update(status="binary_not_ready",binary_error=str(error))
        else:
            report.update(status="binary_ready_unverified",npu_build="passed",
                          build_signature=manifest["signature"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reports/readiness.json"))
    parser.add_argument("--source-only",action="store_true",
                        help="verify source availability before compiling; does not certify the NPU runtime")
    args = parser.parse_args()
    report = readiness_report(binary=not args.source_only)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] in {"source_ready","binary_ready_unverified"} else 1


if __name__ == "__main__":
    sys.exit(main())
