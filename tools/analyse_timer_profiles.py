"""Archive #133/#167: parse stopped workers' raw profiles outside daemon workers.

This is a normal Python child of the timer supervisor, after owned model services
and their resources are released. No model is loaded and no inference is run.
The SDK can log an error and return None; actual device exports decide success.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import multiprocessing
from pathlib import Path
import time

from oscar_ascend.diagnostic_timer import raw_capture_paths, refresh_run_summary
from tools.phase import atomic_json
from tools.summarize_timer import parse_csv, parse_trace


def _device_exports(directory):
    files = sorted(set(directory.rglob("kernel_details*.csv")) |
                   set(directory.rglob("op_summary*.csv")))
    if any(any(row["duration_ms"] > 0 for row in parse_csv(path)) for path in files):
        return [str(path.relative_to(directory)) for path in files if path.stat().st_size]
    traces = sorted(directory.rglob("trace_view*.json"))
    if any(any(row["duration_ms"] > 0 for row in parse_trace(path)) for path in traces):
        return [str(path.relative_to(directory)) for path in traces if path.stat().st_size]
    return []


class _CaptureFailure(RuntimeError):
    """An export cannot repair an unfinished or failed model/capture step."""


def _capture_failure(manifest):
    state = manifest.get("capture_status")
    if (manifest.get("status") == "collecting" or state == "collecting"
            or manifest.get("analysis_previous_status") == "collecting"):
        return "unfinished capture cannot be accepted by offline analysis"
    if manifest.get("error") or manifest.get("capture_error"):
        return str(manifest.get("error") or manifest["capture_error"])
    if state in {"not_started", "missing", "error", "failed"}:
        return f"capture status {state} cannot be accepted by offline analysis"
    if manifest.get("status") == "unsupported":
        return manifest.get("unsupported_reason", "capture is unsupported")
    if manifest.get("status") == "error" and not (
            state == "captured" and manifest.get("analysis_error")):
        return "capture failed before offline analysis; no isolated analysis error to recover"
    return None


def _remember_analysis(manifest):
    """Keep earlier failures when exports repair only the analysis stage."""
    previous = {key: manifest[key] for key in (
        "status", "capture_status", "error", "capture_error", "analysis_status", "analysis_error",
        "analysis_finished_unix", "analysis_previous_error") if key in manifest}
    if previous:
        history = manifest.setdefault("analysis_history", [])
        if not isinstance(history, list):
            raise ValueError("analysis_history must be a list")
        history.append(previous)


def analyse_windows(root, *, analyzer=None, workers=2, allow_analyse=True):
    root = Path(root).resolve()
    if multiprocessing.current_process().daemon:
        raise RuntimeError("offline profile analysis must run in a non-daemon process")
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("analysis workers must be in 1..8")
    if type(allow_analyse) is not bool:
        raise ValueError("allow_analyse must be boolean")
    report = {"schema_version": 1, "status": "missing", "daemon": False,
              "scope": "offline_export_only_pmu_verified_by_summary", "windows": [],
              "allow_analyse": allow_analyse,
              "windowstatus_summary": {}, "started_unix": time.time()}
    destination = root / "offline-analysis.json"
    atomic_json(destination, report)
    # Do not recursively parse raw profiler JSON/large traces as manifests.
    manifests = [p for p in sorted(root.rglob("step*.json"))
                 if p.parent.name.startswith("rank") and "windows" in p.parts]
    summaries = set()
    for path in manifests:
        manifest = json.loads(path.read_text())
        if manifest.get("t") != "oscar-diagnostic-window":
            continue
        profile_dir = Path(manifest["profile_dir"]).resolve()
        if not profile_dir.is_relative_to(root):
            raise ValueError("profile directory escapes this timer run")
        row = {key: manifest.get(key) for key in ("run_id", "window_id", "rank")}
        row.update(manifest=str(path), previous_status=manifest.get("status"),
                   status="missing", analysis_status="missing", capture_paths=[], profile_artifacts=[],
                   sdk_analysis_invoked=False, reused_existing_exports=False)
        _remember_analysis(manifest)
        try:
            captures = raw_capture_paths(profile_dir)
            row["capture_paths"] = captures
            failure = _capture_failure(manifest)
            if failure:
                raise _CaptureFailure(failure)
            exports = _device_exports(profile_dir)
            row["reused_existing_exports"] = bool(exports)
            if not exports and captures and allow_analyse:
                if analyzer is None:
                    # Import only in this ordinary analysis child, never a vLLM
                    # worker or the live service supervisor's request thread.
                    from torch_npu.profiler.profiler import analyse
                    analyzer = analyse
                manifest.setdefault("analysis_previous_status", manifest.get("status"))
                manifest["analysis_status"] = "running"
                atomic_json(path, manifest)
                row["sdk_analysis_invoked"] = True
                for relative in captures:
                    capture = (profile_dir / relative).resolve()
                    if not capture.is_relative_to(profile_dir):
                        raise ValueError("raw capture escapes its profile directory")
                    analyzer(str(capture), max_process_number=workers)
                exports = _device_exports(profile_dir)
            row["profile_artifacts"] = exports
            if not exports:
                row.update(error="offline_analysis_required" if not allow_analyse else
                           "offline analysis produced no nonempty device task export", error_domain="analysis")
            else:
                row.update(status="complete", analysis_status="complete")
        except _CaptureFailure as error:
            row.update(status="error", analysis_status="not_run", error_domain="capture", error=str(error))
        except Exception as error:
            row.update(status="error", analysis_status="error",
                       error_domain="analysis",
                       error=f"{type(error).__name__}: {error}")
        manifest.setdefault("analysis_previous_status", row["previous_status"])
        manifest.update(status=row["status"], analysis_status=row["analysis_status"],
                        analysis_finished_unix=time.time(), analysis_capture_paths=row["capture_paths"],
                        analysis_artifacts=row["profile_artifacts"])
        if row.get("error") and row.get("error_domain") == "analysis":
            manifest["analysis_error"] = row["error"]
        elif row.get("error"):
            manifest.setdefault("capture_error", row["error"])
        if row["status"] == "complete":
            if "analysis_error" in manifest:
                manifest["analysis_previous_error"] = manifest.pop("analysis_error")
            manifest["profile_artifacts"] = sorted(set(manifest.get("profile_artifacts", [])) |
                                                    set(row["profile_artifacts"]))
        atomic_json(path, manifest)
        summaries.add(path.parent / "run-summary.json")
        report["windows"].append(row)
        report["windowstatus_summary"] = dict(Counter(item["status"] for item in report["windows"]))
        atomic_json(destination, report)
        print(f"[timer-analysis] rank={row['rank']} window={row['window_id']} status={row['status']}", flush=True)
    for path in sorted(summaries):
        if path.is_file():
            refresh_run_summary(path)
    states = report["windowstatus_summary"]
    report["status"] = ("complete" if report["windows"] and set(states) == {"complete"}
                        else "partial" if states.get("complete") else "missing")
    report["finished_unix"] = time.time()
    atomic_json(destination, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--exports-only", action="store_true",
                        help="reuse existing CSV/trace only; never import or call the profiler SDK")
    args = parser.parse_args(argv)
    report = analyse_windows(args.root, workers=args.workers, allow_analyse=not args.exports_only)
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
