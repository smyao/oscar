# 档案 #70-73/#94/#95/#125/#133/#140-145：只伴随用户外部负载采证；
# 不发送推理请求、不启HTTP profiler、不把同步诊断或四rank和冒充速度。
"""One-command signed service and passive AISBench phase observation."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import traceback

from .deploy import _temporary_process_context
from .npu_resources import DEFAULT_RELEASE_TOLERANCE, read_npu_resources, wait_for_release
from .phase import atomic_json, live_log, run_phase, terminal_line
from .service_probe import managed_server
from .target_cli import target_env
from .serving_variants import variant_config, variant_features

ROOT = Path(__file__).resolve().parents[1]
FE0_KERNEL_SHA256 = {
    "attention_cv.cpp": "febd753f3bc1bb67639e3ab24cf7f4742608eaada3e835e52d04b92b235c0ec4",
    "attention_tasks.cpp": "1e6c7c4e4d5f71d8c8679573bcc3c4d523361f93ae4e16b2fa3606acb7bc3952",
    "store_int2.cpp": "5a6fa9619e399bb80bc0455c4426396d98510326bb899900b4b96c65d2b58506",
    "rotate_clip_store.cpp": "e18bfc08673b3f5a49d24029c9ad34797e0d5b9114e79639ac01310658982748",
    "merge_lse.cpp": "8b32602f0efa94e5e1ec23ec03471e17b53c94fb4b45b867aca7e4f1c9af1723",
    "status_guard.cpp": "01cbbcfdf99d41daf0ac1372e2ebaa8549380d543bc7c29b426814c96eb593ab",
    "oscar_common.h": "20574a83f919df5bd6450cd581d1dba5a5a6f169768e7e237ba62f51b7499964",
}
_DEBUG_ENV = ("OSCAR_DEBUG_SYNC", "OSCAR_TIMING", "OSCAR_PROFILER",
              "OSCAR_DEVICE_TIMING_CONTROL", "OSCAR_PROFILE_DIR")


def _terminal(line: str, *, error: bool = False) -> None:
    # redirect_stdout is active while supervising the service. These few
    # evidence lines still go to the foreground terminal, not a filtered log.
    descriptor = 2 if error else 1
    data = memoryview((line + "\n").encode(errors="replace"))
    while data:
        count = os.write(descriptor, data)
        if count <= 0:
            raise OSError("foreground write returned no bytes")
        data = data[count:]


def _source_identity(variant: str, config: dict) -> dict:
    actual = {name: hashlib.sha256((ROOT / "csrc/kernels" / name).read_bytes()).hexdigest()
              for name in FE0_KERNEL_SHA256}
    flag = config.get("experimental_history_reuse", False)
    fast = config.get("experimental_fast_unpack", False)
    mixed = config.get("experimental_mixed_cv", False)
    striped = config.get("experimental_striped_cache", False)
    if type(flag) is not bool:
        raise ValueError("experimental_history_reuse must be an explicit boolean")
    if type(fast) is not bool or (fast and not flag):
        raise ValueError("experimental_fast_unpack requires explicit candidate history configuration")
    if type(mixed) is not bool or (mixed and not fast):
        raise ValueError("experimental_mixed_cv requires explicit fast candidate configuration")
    if type(striped) is not bool or (striped and not mixed):
        raise ValueError("experimental_striped_cache requires explicit mixed candidate configuration")
    if variant in {"baseline", "candidate"} and actual != FE0_KERNEL_SHA256:
        raise RuntimeError("OSCAR observation requires byte-identical fe0 production kernels")
    if variant == "baseline" and flag:
        raise RuntimeError("baseline requires experimental_history_reuse=false")
    elif variant == "candidate" and not flag:
        raise RuntimeError("candidate requires experimental_history_reuse=true")
    cluster = ROOT / "csrc/kernels/attention_cv_cluster.cpp"
    q1 = ROOT / "csrc/kernels/attention_cv_q1.cpp"
    if variant == "candidate" and (not cluster.is_file() or not q1.is_file()):
        raise RuntimeError("candidate AscendC source is missing")
    fast_sources = {}
    if fast:
        for name in ("attention_cv_fast.cpp", "attention_cv_fast_q1.cpp",
                     "attention_cv_fast_cluster4.cpp", "attention_fast_unpack.h"):
            fast_sources[name] = hashlib.sha256((ROOT / "csrc/kernels" / name).read_bytes()).hexdigest()
    if mixed:
        for name in ("attention_cv_fast_balanced.cpp", "attention_cv_fast_cluster16.cpp"):
            fast_sources[name] = hashlib.sha256((ROOT / "csrc/kernels" / name).read_bytes()).hexdigest()
    striped_sources = {}
    if striped:
        for name in ("attention_striped_unpack.h", "attention_striped_unpack_simd.h",
                     "attention_cv_striped_decode.cpp", "attention_cv_striped_decode_simd.cpp",
                     "attention_cv_striped.cpp", "attention_cv_striped_balanced.cpp",
                     "attention_cv_striped_cluster4.cpp", "attention_cv_striped_cluster16.cpp",
                     "rotate_clip_store_striped.cpp"):
            striped_sources[name] = hashlib.sha256((ROOT / "csrc/kernels" / name).read_bytes()).hexdigest()
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                              capture_output=True, check=True).stdout.strip()
    return {"variant": variant, "fe0_production_kernels_match": actual == FE0_KERNEL_SHA256,
            "kernel_sha256": actual, "experimental_history_reuse": flag,
            "experimental_fast_unpack": fast, "experimental_mixed_cv": mixed,
            "experimental_striped_cache": striped,
            "cache_format": "striped_v1" if striped else "canonical_v1",
            "striped_source_sha256": striped_sources,
            "fast_source_sha256": fast_sources,
            "candidate_kernel_sha256": hashlib.sha256(cluster.read_bytes()).hexdigest()
                if variant == "candidate" else None,
            "candidate_q1_kernel_sha256": hashlib.sha256(q1.read_bytes()).hexdigest()
                if variant == "candidate" else None,
            "checkout_revision": revision,
            "reference_commit": "fe0e925e7ef78bfb64217a300031502fc4a7b7bc"}


def _step_summary(trace_dir: Path, output: Path, variant: str) -> dict:
    latest: dict[str, dict] = {}
    paths = sorted(trace_dir.glob("passive-step-*.jsonl"))
    for path in paths:
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record.get("t") == "oscar-passive-step" and isinstance(record.get("step_id"), str):
                latest[record["step_id"]] = record
    grouped: dict[str, dict[str, list[dict]]] = {}
    for record in latest.values():
        rank = str(record.get("rank"))
        bucket = record.get("bucket", "unknown")
        grouped.setdefault(rank, {}).setdefault(bucket, []).append(record)
    buckets = {}
    for rank, by_bucket in grouped.items():
        buckets[rank] = {}
        for bucket, rows in by_bucket.items():
            measured = [row for row in rows if row.get("status") == "measured"]
            graph_opaque = [row for row in rows
                if row.get("missing_reason") == "graph_replay_has_no_python_oscar_phase_breakdown"
                and type(row.get("step_device_ms")) in (int, float)
                and isinstance(row.get("scopes"), list)]
            timed = measured + graph_opaque
            def p50(field, source=measured):
                values = [row[field] for row in source if type(row.get(field)) in (int, float)]
                return statistics.median(values) if values else None
            scopes = {}
            for name in ("target_forward", "draft_forward", "draft_proposal",
                         "native_attention",
                         "graph_replay", "fia", "phase1_stores"):
                values = [scope["duration_ms"] for row in timed for scope in row.get("scopes", ())
                          if scope.get("phase") == name]
                scopes[name] = {"count": len(values), "p50_ms": statistics.median(values) if values else None}
            partial = {}
            for name in ("target_forward", "draft_proposal"):
                envelopes = [item for row in rows for item in row.get("partial_envelopes", ())
                             if item.get("phase") == name and item.get("residual_ms") is not None]
                partial[name] = {"count": len(envelopes),
                    "duration_p50_ms": statistics.median(item["duration_ms"] for item in envelopes)
                        if envelopes else None,
                    "attention_union_p50_ms": statistics.median(
                        item["attention_union_ms"] for item in envelopes)
                        if envelopes else None,
                    "residual_p50_ms": statistics.median(item["residual_ms"] for item in envelopes)
                        if envelopes else None,
                    "scope": "same_stream_partial_envelope"}
            buckets[rank][bucket] = {"sampled": len(rows), "measured": len(measured),
                "graph_opaque": len(graph_opaque), "timed_steps": len(timed),
                "coverage": ("partial" if measured and graph_opaque else
                             "graph_opaque" if graph_opaque else
                             "measured" if measured else "missing"),
                "missing": len(rows)-len(timed), "step_p50_ms": p50("step_device_ms", timed),
                "attention_union_p50_ms": p50("attention_union_ms"),
                "attention_backend": "native" if variant == "native" else "oscar",
                "oscar_union_p50_ms": p50("oscar_union_ms"),
                "residual_p50_ms": p50("residual_ms"), "scopes": scopes,
                "partial_envelopes": partial,
                "scheduled_prompt_tokens": sum(row.get("scheduled_prompt_tokens") or 0 for row in measured),
                "scheduled_decode_tokens": sum(row.get("scheduled_decode_tokens") or 0 for row in measured),
                "preempted_requests": sum(row.get("preempted_requests") or 0 for row in measured)}
    any_measured = any(row.get("status") == "measured" for row in latest.values())
    any_graph = any(row.get("missing_reason") == "graph_replay_has_no_python_oscar_phase_breakdown"
                    and type(row.get("step_device_ms")) in (int, float) for row in latest.values())
    report = {"status": "partial" if any_measured and any_graph else
              "observed" if any_measured else "graph_opaque" if any_graph else "missing",
              "variant": variant, "scope": "per_rank_same_step_async_npu_events",
              "event_files": [str(p) for p in paths], "samples": list(latest.values()),
              "by_rank_bucket": buckets,
              "rules": ["no four-rank duration sums", "graph replay OSCAR residual is missing",
                        "Running/Waiting/KV are independent one-second gauge samples",
                        "pending last steps remain missing without a later safe query"]}
    atomic_json(output, report)
    return report


def _headline(summary: dict) -> str:
    """Select one observed critical rank per bucket; never sum TP clocks."""
    def chosen(bucket):
        rows = [(rank, data[bucket]) for rank, data in summary["by_rank_bucket"].items()
                if bucket in data and data[bucket]["step_p50_ms"] is not None]
        return max(rows, key=lambda item: item[1]["step_p50_ms"]) if rows else None
    parts = []
    for bucket in ("prefill", "decode", "mixed"):
        item = chosen(bucket)
        if item is None:
            partial = [(rank, value) for rank, data in summary["by_rank_bucket"].items()
                       if bucket in data for value in data[bucket]["partial_envelopes"].values()
                       if value["duration_p50_ms"] is not None]
            if partial:
                rank, value = max(partial, key=lambda item: item[1]["duration_p50_ms"])
                parts.append(f"{bucket}=partial_rank{rank}:envelope{value['duration_p50_ms']:.2f}ms/"
                             f"residual{value['residual_p50_ms']:.2f}")
            else:
                parts.append(f"{bucket}=missing")
            continue
        rank, row = item
        graph = row["scopes"]["graph_replay"]["p50_ms"]
        parts.append(f"{bucket}=rank{rank}:step{row['step_p50_ms']:.2f}ms/"
                     f"attention{row['attention_union_p50_ms'] if row['attention_union_p50_ms'] is not None else 'missing'}/"
                     f"residual{row['residual_p50_ms'] if row['residual_p50_ms'] is not None else 'missing'}/"
                     f"graph{graph if graph is not None else 'missing'}")
    return " ".join(parts)


def _phase(name: str, command: list[str], *, config: dict, env: dict,
           log_dir: Path, status: dict) -> None:
    result = run_phase(name, command, cwd=ROOT, log_dir=log_dir, env=env,
                       timeout=float(config["phase_timeout_seconds"]),
                       grace=float(config["shutdown_timeout_seconds"]), heartbeat=60)
    status["phases"].append(asdict(result))
    atomic_json(log_dir / "status.json", status)
    if result.returncode != 0 or result.timed_out or not result.cleanup_complete:
        error = RuntimeError(f"{name} failed rc={result.returncode} log={result.log}")
        error.phase = name
        error.returncode = result.returncode or 1
        raise error


def _preflight(config_path: Path, config: dict, env: dict, log_dir: Path,
               status: dict, variant: str) -> None:
    python = sys.executable
    for name, command in (
        ("install-dependencies", [python, "-m", "pip", "install", "setuptools>=69", "wheel",
                                   "pybind11>=3", "cmake>=3.26", "ninja", "pytest"]),
        ("install-plugin", [python, "-m", "pip", "install", "--no-deps", "--no-build-isolation",
                            "-e", str(ROOT)]),
    ):
        _phase(name, command, config=config, env=env, log_dir=log_dir, status=status)
    if variant == "native":
        return
    from .paired_concurrency_probe import ensure_current_operators, ensure_native_current_attention
    acceptance = json.loads((ROOT / "configs/acceptance.json").read_text())
    if acceptance.get("frozen_before_measurement") is not True:
        raise RuntimeError("frozen operator acceptance policy is missing")
    evidence = ensure_current_operators(config_path, config, acceptance, log_dir / "accuracy")
    if evidence.get("status") != "passed":
        raise RuntimeError("signed build and real NPU CV/rotation evidence incomplete")
    status["operator_gate"] = evidence
    atomic_json(log_dir / "status.json", status)
    current = ensure_native_current_attention(config_path, config,
                ROOT / "configs/acceptance.json", log_dir / "accuracy")
    if current.get("status") != "passed" or current.get("resource_release") != "passed":
        raise RuntimeError("native-current partial/LSE gate incomplete")
    status["current_gate"] = current
    for name, command in (
        ("probe-ops", [python, "-m", "tools.probe_ops", "--output", str(log_dir / "operators.json")]),
        ("prepare-rotations", [python, "-m", "tools.prepare_rotations", "--config", str(config_path)]),
    ):
        if name == "probe-ops":
            (log_dir / "operators.json").unlink(missing_ok=True)
        _phase(name, command, config=config, env=env, log_dir=log_dir, status=status)
    primitive = json.loads((log_dir / "operators.json").read_text())
    if primitive.get("status") != "primitive_probe_passed":
        raise RuntimeError("real NPU store/merge primitive gate incomplete")


def _candidate_gate(config_path: Path, config: dict, env: dict, log_dir: Path,
                    status: dict) -> None:
    """Fail closed before routing experimental history reuse in a service."""
    from oscar_ascend.ops.loader import validate_build_artifacts
    output = log_dir / "history-reuse.json"
    output.unlink(missing_ok=True)
    before = read_npu_resources(config, log_dir=log_dir / "candidate-resources-before", timeout=30)
    result = None
    try:
        result = run_phase("history-reuse-npu",
            [sys.executable, "-m", "tools.probe_history_reuse", "--config", str(config_path),
             "--acceptance", str(ROOT / "configs/acceptance.json"), "--output", str(output)],
            cwd=ROOT, log_dir=log_dir, env=env,
            timeout=float(config["phase_timeout_seconds"]),
            grace=float(config["shutdown_timeout_seconds"]), heartbeat=60)
    finally:
        release = wait_for_release(config, before, log_dir=log_dir / "candidate-resources-after",
            timeout=float(config.get("resource_release_timeout_seconds", 30)),
            tolerance_bytes=config.get("resource_release_tolerance_bytes", DEFAULT_RELEASE_TOLERANCE))
        status["candidate_resource_release"] = release
        atomic_json(log_dir / "status.json", status)
    if result is not None:
        status["phases"].append(asdict(result))
        atomic_json(log_dir / "status.json", status)
    if result is None or result.returncode != 0 or result.timed_out or not result.cleanup_complete:
        code = (result.returncode if result is not None and result.returncode != 0
                else 124 if result is not None and result.timed_out
                else 125 if result is not None and not result.cleanup_complete else 1)
        failure = RuntimeError("candidate NPU/graph gate failed before service; "
                               f"rc={code} report={output}")
        failure.phase = "history-reuse-npu"
        failure.returncode = code
        raise failure
    if release.get("status") != "passed":
        raise RuntimeError("candidate NPU resources did not release after gate")
    report = json.loads(output.read_text())
    manifest = validate_build_artifacts(ROOT / "build/ascendc/build_manifest.json")
    if (report.get("status") != "passed" or report.get("candidate_evaluation_allowed") is not True
            or report.get("default_route") != "fe0"
            or report.get("reference_commit") != "fe0e925e7ef78bfb64217a300031502fc4a7b7bc"
            or report.get("production_promotion") !=
                "blocked_pending_full_model_quality_and_service_performance"
            or report.get("graph_capture") != "passed" or report.get("graph_replay") != "passed"
            or any(report.get(key) != "passed" for key in (
                "q1_schedule_gate", "q1_precision", "q1_performance", "q1_graph_capture", "q1_graph_replay"))
            or report.get("artifact_signature") != manifest.get("signature")
            or report.get("artifact_sha256") != manifest.get("sha256")
            or report.get("fe0_source_sha256") != {
                "csrc/kernels/attention_cv.cpp": FE0_KERNEL_SHA256["attention_cv.cpp"],
                "csrc/kernels/oscar_common.h": FE0_KERNEL_SHA256["oscar_common.h"]}):
        raise RuntimeError("candidate report lacks exact signed artifact, fe0 source, graph or precision evidence")
    status["candidate_gate"] = {"status": "passed", "report": str(output),
                                "artifact_signature": manifest["signature"],
                                "candidate_evaluation_allowed": True,
                                "production_promotion": "blocked"}
    atomic_json(log_dir / "status.json", status)


def _q4_diagnostic(config_path: Path, config: dict, env: dict, log_dir: Path,
                   status: dict) -> None:
    """#150 continuation: diagnose remaining q4 cost without a model/AISBench.

    Native-vs-INT2 differences are observations, not a performance pass. The
    diagnostic still must complete its own oracle, profile parity and cleanup.
    """
    from oscar_ascend.ops.loader import validate_build_artifacts
    # #151: run_phase owns <phase>.json; never let it overwrite probe data.
    output = log_dir / "q4-hotpath-report.json"
    output.unlink(missing_ok=True)
    before = read_npu_resources(config, log_dir=log_dir / "q4-resources-before", timeout=30)
    phase_error = None
    release = None
    try:
        _phase("q4-hotpath", [sys.executable, "-m", "tools.probe_decode_hotpath",
            "--config", str(config_path), "--acceptance", str(ROOT / "configs/acceptance.json"),
            "--output", str(output)], config=config, env=env, log_dir=log_dir, status=status)
    except BaseException as error:
        phase_error = error
        raise
    finally:
        try:
            release = wait_for_release(config, before, log_dir=log_dir / "q4-resources-after",
                timeout=float(config.get("resource_release_timeout_seconds", 30)),
                tolerance_bytes=config.get("resource_release_tolerance_bytes", DEFAULT_RELEASE_TOLERANCE))
        except BaseException as error:
            release = {"status": "failed", "reason": f"{type(error).__name__}: {error}"}
            if phase_error is None:
                raise
        finally:
            if release is not None:
                status["q4_resource_release"] = release
                atomic_json(log_dir / "status.json", status)
    if release.get("status") != "passed":
        raise RuntimeError("q4 diagnostic NPU resources did not release")
    report = json.loads(output.read_text())
    manifest = validate_build_artifacts(ROOT / "build/ascendc/build_manifest.json")
    expected = {"status": "observed", "native_oracle": "passed", "oscar_oracle": "passed",
                "profile_parity": "bitwise_passed", "profile_observed": True,
                "artifact_signature": manifest.get("signature"), "artifact_sha256": manifest.get("sha256")}
    mismatches = [key for key, value in expected.items()
                  if report.get(key) != value or (key == "profile_observed" and report.get(key) is not True)]
    if mismatches:
        error = RuntimeError("q4 diagnostic lacks signed native/OSCAR oracle or profile evidence; "
                             f"mismatched_fields={','.join(mismatches)} report={output}")
        error.phase = "q4-evidence"
        raise error
    status["q4_diagnostic"] = {"status": "observed", "report": str(output),
                               "performance_acceptance": "not_established"}
    atomic_json(log_dir / "status.json", status)


def _fast_unpack_gate(config_path: Path, config: dict, env: dict, log_dir: Path,
                      status: dict) -> None:
    """#151: a new unpack implementation needs its own signed real-NPU gate."""
    from oscar_ascend.ops.loader import validate_build_artifacts
    output = log_dir / "fast-unpack-report.json"
    output.unlink(missing_ok=True)
    before = read_npu_resources(config, log_dir=log_dir / "fast-unpack-resources-before", timeout=30)
    phase_error, release = None, None
    try:
        _phase("fast-unpack", [sys.executable, "-m", "tools.probe_fast_unpack",
            "--config", str(config_path), "--acceptance", str(ROOT / "configs/acceptance.json"),
            "--output", str(output)], config=config, env=env, log_dir=log_dir, status=status)
    except BaseException as error:
        phase_error = error
        raise
    finally:
        try:
            release = wait_for_release(config, before, log_dir=log_dir / "fast-unpack-resources-after",
                timeout=float(config.get("resource_release_timeout_seconds", 30)),
                tolerance_bytes=config.get("resource_release_tolerance_bytes", DEFAULT_RELEASE_TOLERANCE))
        except BaseException as error:
            release = {"status": "failed", "reason": f"{type(error).__name__}: {error}"}
            if phase_error is None:
                raise
        finally:
            if release is not None:
                status["fast_unpack_resource_release"] = release
                atomic_json(log_dir / "status.json", status)
    if release.get("status") != "passed":
        raise RuntimeError("fast unpack NPU resources did not release")
    report = json.loads(output.read_text())
    manifest = validate_build_artifacts(ROOT / "build/ascendc/build_manifest.json")
    expected = {key: "passed" for key in ("status", "precision", "graph_capture", "graph_replay", "performance")}
    expected.update(artifact_signature=manifest.get("signature"), artifact_sha256=manifest.get("sha256"))
    mismatches = [key for key, value in expected.items() if report.get(key) != value]
    if mismatches:
        error = RuntimeError(f"fast unpack gate lacks matching evidence: {','.join(mismatches)} report={output}")
        error.phase = "fast-unpack-evidence"
        raise error
    status["fast_unpack_gate"] = {"status": "passed", "report": str(output),
                                   "artifact_signature": manifest["signature"],
                                   "full_service_performance": "not_established"}
    atomic_json(log_dir / "status.json", status)


def _mixed_optimization_gate(config_path: Path, config: dict, env: dict, log_dir: Path,
                      status: dict) -> None:
    """#151 and 2026-09-29: balanced ownership/C16 require exact new NPU evidence."""
    from oscar_ascend.ops.loader import validate_build_artifacts
    output = log_dir / "mixed-optimization-report.json"
    output.unlink(missing_ok=True)
    before = read_npu_resources(config, log_dir=log_dir / "mixed-optimization-resources-before", timeout=30)
    phase_error, release = None, None
    try:
        _phase("mixed-optimization", [sys.executable, "-m", "tools.probe_mixed_optimization",
            "--config", str(config_path), "--acceptance", str(ROOT / "configs/acceptance.json"),
            "--output", str(output)], config=config, env=env, log_dir=log_dir, status=status)
    except BaseException as error:
        phase_error = error
        raise
    finally:
        try:
            release = wait_for_release(config, before, log_dir=log_dir / "mixed-optimization-resources-after",
                timeout=float(config.get("resource_release_timeout_seconds", 30)),
                tolerance_bytes=config.get("resource_release_tolerance_bytes", DEFAULT_RELEASE_TOLERANCE))
        except BaseException as error:
            release = {"status": "failed", "reason": f"{type(error).__name__}: {error}"}
            if phase_error is None:
                raise
        finally:
            if release is not None:
                status["mixed_optimization_resource_release"] = release
                atomic_json(log_dir / "status.json", status)
    if release.get("status") != "passed":
        raise RuntimeError("mixed CV NPU resources did not release")
    report = json.loads(output.read_text())
    manifest = validate_build_artifacts(ROOT / "build/ascendc/build_manifest.json")
    expected = {key: "passed" for key in ("status", "precision", "graph_capture", "graph_replay", "performance")}
    expected.update(artifact_signature=manifest.get("signature"), artifact_sha256=manifest.get("sha256"))
    mismatches = [key for key, value in expected.items() if report.get(key) != value]
    if mismatches:
        error = RuntimeError(f"mixed CV gate lacks matching evidence: {','.join(mismatches)} report={output}")
        error.phase = "mixed-optimization-evidence"
        raise error
    status["mixed_optimization_gate"] = {"status": "passed", "report": str(output),
                                   "artifact_signature": manifest["signature"],
                                   "full_service_performance": "not_established"}
    atomic_json(log_dir / "status.json", status)


def _striped_cache_gate(config_path: Path, config: dict, env: dict, log_dir: Path,
                      status: dict) -> None:
    """#151/#153/#154: signed paired cache writer/readers, graph and latency."""
    from oscar_ascend.ops.loader import validate_build_artifacts
    output = log_dir / "striped-cache-report.json"
    output.unlink(missing_ok=True)
    before = read_npu_resources(config, log_dir=log_dir / "striped-cache-resources-before", timeout=30)
    phase_error, release = None, None
    try:
        _phase("striped-cache", [sys.executable, "-m", "tools.probe_striped_cache",
            "--config", str(config_path), "--acceptance", str(ROOT / "configs/acceptance.json"),
            "--output", str(output)], config=config, env=env, log_dir=log_dir, status=status)
    except BaseException as error:
        phase_error = error
        raise
    finally:
        try:
            release = wait_for_release(config, before, log_dir=log_dir / "striped-cache-resources-after",
                timeout=float(config.get("resource_release_timeout_seconds", 30)),
                tolerance_bytes=config.get("resource_release_tolerance_bytes", DEFAULT_RELEASE_TOLERANCE))
        except BaseException as error:
            release = {"status": "failed", "reason": f"{type(error).__name__}: {error}"}
            if phase_error is None:
                raise
        finally:
            if release is not None:
                status["striped_cache_resource_release"] = release
                atomic_json(log_dir / "status.json", status)
    if release.get("status") != "passed":
        raise RuntimeError("striped cache NPU resources did not release")
    report = json.loads(output.read_text())
    manifest = validate_build_artifacts(ROOT / "build/ascendc/build_manifest.json")
    expected = {key: "passed" for key in ("status", "precision", "graph_capture", "graph_replay", "performance")}
    from .striped_fixture import FORMAT
    expected["format"] = FORMAT
    expected.update(artifact_signature=manifest.get("signature"), artifact_sha256=manifest.get("sha256"))
    mismatches = [key for key, value in expected.items() if report.get(key) != value]
    if report.get("writer", {}).get("status") != "passed":
        mismatches.append("writer")
    if mismatches:
        error = RuntimeError(f"striped cache gate lacks matching evidence: {','.join(mismatches)} report={output}")
        error.phase = "striped-cache-evidence"
        raise error
    status["striped_cache_gate"] = {"status": "passed", "report": str(output),
                                   "artifact_signature": manifest["signature"],
                                   "full_service_performance": "not_established"}
    atomic_json(log_dir / "status.json", status)


def _mixed_diagnostic(config_path: Path, config: dict, env: dict, log_dir: Path,
                      status: dict) -> None:
    """Measure the actual CV + current-FIA composition without loading a model."""
    from oscar_ascend.ops.loader import validate_build_artifacts
    output = log_dir / "mixed-attention-report.json"
    output.unlink(missing_ok=True)
    before = read_npu_resources(config, log_dir=log_dir / "mixed-resources-before", timeout=30)
    phase_error, release = None, None
    try:
        _phase("mixed-attention", [sys.executable, "-m", "tools.probe_mixed_attention",
            "--config", str(config_path), "--acceptance", str(ROOT / "configs/acceptance.json"),
            "--output", str(output)], config=config, env=env, log_dir=log_dir, status=status)
    except BaseException as error:
        phase_error = error
        raise
    finally:
        try:
            release = wait_for_release(config, before, log_dir=log_dir / "mixed-resources-after",
                timeout=float(config.get("resource_release_timeout_seconds", 30)),
                tolerance_bytes=config.get("resource_release_tolerance_bytes", DEFAULT_RELEASE_TOLERANCE))
        except BaseException as error:
            release = {"status": "failed", "reason": f"{type(error).__name__}: {error}"}
            if phase_error is None:
                raise
        finally:
            if release is not None:
                status["mixed_resource_release"] = release
                atomic_json(log_dir / "status.json", status)
    if release.get("status") != "passed":
        raise RuntimeError("mixed diagnostic NPU resources did not release")
    report = json.loads(output.read_text())
    manifest = validate_build_artifacts(ROOT / "build/ascendc/build_manifest.json")
    expected = {"status": "observed", "accuracy": "passed",
                "artifact_signature": manifest.get("signature"), "artifact_sha256": manifest.get("sha256")}
    mismatches = [key for key, value in expected.items() if report.get(key) != value]
    if mismatches:
        error = RuntimeError(f"mixed diagnostic lacks signed oracle evidence: {','.join(mismatches)} report={output}")
        error.phase = "mixed-evidence"
        raise error
    status["mixed_diagnostic"] = {"status": "observed", "report": str(output),
                                  "performance_acceptance": "not_established"}
    atomic_json(log_dir / "status.json", status)


def run(config_path: Path, log_dir: Path, variant: str, *, probe_only: bool = False,
        diagnose_q4: bool = False, diagnose_mixed: bool = False) -> int:
    config_path, log_dir = config_path.resolve(), log_dir.resolve()
    status = {"status": "preparing", "variant": variant,
              "measurement": "operator_microprobe" if probe_only else "passive_external_only",
              "performance_acceptance": "not_run", "phases": []}
    atomic_json(log_dir / "status.json", status)
    current_phase = "config"
    try:
        if probe_only and variant != "candidate":
            raise ValueError("--probe-only requires --variant candidate")
        if diagnose_q4 and variant != "candidate":
            raise ValueError("--diagnose-q4 requires --variant candidate")
        if diagnose_mixed and variant != "candidate":
            raise ValueError("--diagnose-mixed requires --variant candidate")
        original = config_path.read_bytes()
        config = variant_config(json.loads(original), variant)
        status["optimizations"] = variant_features(config)
        effective_path = log_dir / "effective-target.json"
        atomic_json(effective_path, config)
        status["target_config"] = {"original": str(config_path),
            "original_sha256": hashlib.sha256(original).hexdigest(),
            "effective": str(effective_path),
            "effective_sha256": hashlib.sha256(effective_path.read_bytes()).hexdigest()}
        if config.get("diagnostic_device_timing") is True or config.get("profiler_config") is not None:
            raise ValueError("passive observation requires unarmed debug/profiler configuration")
        identity = _source_identity(variant, config)
        status["source_identity"] = identity
        env = target_env(config)
        for key in _DEBUG_ENV:
            env.pop(key, None)
        env.update(OSCAR_TARGET_CONFIG=str(effective_path), OSCAR_TERMINAL_LOG_MODE="compact",
                   PYTHONUNBUFFERED="1", OSCAR_PASSIVE_VARIANT=variant)
        if variant == "native":
            env["OSCAR_ENABLED"] = "0"
        control = log_dir / "passive-control.json"
        control.unlink(missing_ok=True)
        env["OSCAR_PASSIVE_TIMING_CONTROL"] = str(control)
        _terminal(f"[oscar] OBSERVE_START variant={variant} devices={','.join(map(str,config['devices']))} "
                  f"port={config['port']} log={log_dir} user_load_only=true")
        with _temporary_process_context(env, sys.argv):
            for key in _DEBUG_ENV:
                os.environ.pop(key, None)
            current_phase = "preflight"
            with live_log(log_dir / "preflight-console.log", mode="compact") as preflight:
                with redirect_stdout(preflight), redirect_stderr(preflight):
                    _preflight(effective_path, config, env, log_dir, status, variant)
                    if variant == "candidate":
                        _candidate_gate(effective_path, config, env, log_dir, status)
                        _fast_unpack_gate(effective_path, config, env, log_dir, status)
                        if config.get("experimental_mixed_cv", False):
                            _mixed_optimization_gate(effective_path, config, env, log_dir, status)
                        if config.get("experimental_striped_cache", False):
                            _striped_cache_gate(effective_path, config, env, log_dir, status)
                        if diagnose_q4:
                            _q4_diagnostic(effective_path, config, env, log_dir, status)
                        if diagnose_mixed and not config.get("experimental_mixed_cv", False):
                            _mixed_diagnostic(effective_path, config, env, log_dir, status)
            if probe_only:
                status.update(status="operator_probes_passed", service_started=False,
                              performance_acceptance="operator_only_not_end_to_end")
                atomic_json(log_dir / "status.json", status)
                _terminal(f"[oscar] OBSERVE_PROBE_DONE variant={variant} service_started=false "
                          f"report={log_dir / 'status.json'}")
                return 0
            current_phase = "serve"
            before = read_npu_resources(config, log_dir=log_dir / "resources-before", timeout=30)
            try:
                with live_log(log_dir / "console.log", mode="compact") as console:
                    with redirect_stdout(console), redirect_stderr(console):
                        with managed_server(config, effective_path, log_dir=log_dir / "serve",
                                            lifecycle=status.setdefault("server", {}),
                                            native=variant == "native") as server:
                            atomic_json(control, {"enabled": True, "run_id": log_dir.name,
                                                  "variant": variant})
                            status["status"] = "observer_starting"
                            atomic_json(log_dir / "status.json", status)
                            from benchmarks.passive import observe
                            stop, observer_ready = threading.Event(), threading.Event()
                            observer_report: dict = {}
                            summary_path = log_dir / "summary.json"
                            def window_done(_index, window):
                                evidence = _step_summary(server.trace_dir, summary_path, variant)
                                observer_report["latest_step_status"] = evidence["status"]
                                mtp = window.get("mtp_delta", {})
                                if _index == 0:
                                    _terminal(f"[oscar] OBSERVE_WINDOW_DONE variant={variant} "
                                              f"running={window.get('peak_observed_running')} "
                                              f"waiting={window.get('peak_observed_waiting')} "
                                              f"kv_metric={window.get('peak_observed_kv_cache_usage_perc')} "
                                              f"preemptions={window.get('preemptions_delta')} "
                                              f"accepted={mtp.get('vllm:spec_decode_num_accepted_tokens')} "
                                              f"drafted={mtp.get('vllm:spec_decode_num_draft_tokens')} "
                                              f"step_status={evidence['status']} {_headline(evidence)} "
                                              f"summary={summary_path}")
                            def worker():
                                observer_report["metrics"] = observe(server.base_url,
                                    output=log_dir / "external-load.json", stop=stop,
                                    ready=observer_ready,
                                    sample_interval=1.0, idle_seconds=15.0,
                                    variant="native" if variant == "native" else "oscar",
                                    on_window_end=window_done, announce_ready=False)
                            thread = threading.Thread(target=worker, daemon=True,
                                                      name="oscar-passive-observer")
                            thread.start()
                            try:
                                if not observer_ready.wait(10) or not thread.is_alive():
                                    raise RuntimeError("passive /metrics observer did not establish an idle baseline")
                                status["status"] = "serving"
                                atomic_json(log_dir / "status.json", status)
                                _terminal(f"[oscar] OBSERVE_READY variant={variant} "
                                          f"url={server.base_url}/v1/chat/completions "
                                          f"model={config['served_model_name']} "
                                          f"observer={log_dir / 'external-load.json'}")
                                while True:
                                    server.check_alive()
                                    if not thread.is_alive():
                                        raise RuntimeError(f"passive observer exited: {observer_report}")
                                    time.sleep(0.25)
                            except KeyboardInterrupt:
                                status["status"] = "stopping"
                            finally:
                                stop.set(); thread.join(timeout=8)
                                atomic_json(control, {"enabled": False, "run_id": log_dir.name,
                                                      "variant": variant})
                                status["external_observation"] = observer_report.get("metrics", {
                                    "status": "missing", "reason": "observer did not finish"})
                                status["step_evidence"] = {"status": "cleanup_pending",
                                    "report": str(summary_path)}
                                atomic_json(log_dir / "status.json", status)
            finally:
                try:
                    release = wait_for_release(config, before, log_dir=log_dir / "resources-after",
                        timeout=float(config.get("resource_release_timeout_seconds", 30)),
                        tolerance_bytes=config.get("resource_release_tolerance_bytes", DEFAULT_RELEASE_TOLERANCE))
                except BaseException as resource_error:
                    # Do not replace a server failure/exit code with a second
                    # observer exception raised during cleanup.
                    release = {"status": "failed", "reason":
                        f"{type(resource_error).__name__}: {resource_error}"}
                status["resource_release"] = release
                trace_path = status.get("server", {}).get("trace_dir")
                if isinstance(trace_path, str):
                    try:
                        status["step_evidence"] = _step_summary(
                            Path(trace_path), log_dir / "summary.json", variant)
                    except BaseException as summary_error:
                        status["step_evidence"] = {"status": "missing", "reason":
                            f"{type(summary_error).__name__}: {summary_error}",
                            "report": str(log_dir / "summary.json")}
                else:
                    status["step_evidence"] = {"status": "missing", "reason":
                        "server_trace_dir_missing"}
                atomic_json(log_dir / "status.json", status)
            if status["server"].get("cleanup_complete") is not True or release.get("status") != "passed":
                raise RuntimeError("owned server cleanup or NPU resource release failed")
            status["status"] = "stopped"
            atomic_json(log_dir / "status.json", status)
            _terminal(f"[oscar] OBSERVE_STOPPED variant={variant} cleanup=passed report={log_dir / 'status.json'}")
            return 0
    except BaseException as error:
        rc = getattr(error, "returncode", 130 if isinstance(error, KeyboardInterrupt) else 1)
        status.update(status="failed", failed_phase=getattr(error, "phase", current_phase),
                      error=f"{type(error).__name__}: {error}", returncode=rc)
        atomic_json(log_dir / "status.json", status)
        with (log_dir / "failure.log").open("w") as stream:
            traceback.print_exception(error, file=stream)
        traceback.print_exception(error, file=sys.stderr)
        _terminal(f"[oscar] OBSERVE_FAILED phase={status['failed_phase']} rc={rc} "
                  f"report={log_dir / 'status.json'}", error=True)
        return rc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/target.json")
    parser.add_argument("--variant", choices=("baseline", "candidate", "native"), default="baseline")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--probe-only", action="store_true",
                        help="finish after candidate operator/graph/latency gates; do not start a model")
    parser.add_argument("--diagnose-q4", action="store_true",
                        help="also rerun the established native q4/profile diagnostic; normally reuse prior evidence")
    parser.add_argument("--diagnose-mixed", action="store_true",
                        help="mixed shapes are included in the candidate optimization gate; retains the legacy diagnostic for older presets")
    args = parser.parse_args(argv)
    log_dir = (args.log_dir or ROOT / "logs" / ("observe-" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S.%fZ"))).resolve()
    if args.plan:
        config = variant_config(json.loads(args.config.read_text()), args.variant)
        phases = ["install", "signed_operator_gate" if args.variant != "native" else "native_start"]
        if args.variant == "candidate":
            phases += ["candidate_operator_graph_latency_gates", "fast_unpack_gate"]
            if config.get("experimental_mixed_cv", False):
                phases.append("mixed_optimization_gate")
            if config.get("experimental_striped_cache", False):
                phases.append("striped_cache_gate")
            if args.diagnose_q4:
                phases.append("q4_native_comparison_and_profile")
            if args.diagnose_mixed and not config.get("experimental_mixed_cv", False):
                phases.append("mixed_cv_current_fia_merge_diagnostic")
        if not args.probe_only:
            phases += ["managed_service_health", "external_metrics_and_bounded_async_events"]
        print(json.dumps({"variant": args.variant, "devices": config["devices"],
            "optimizations": variant_features(config),
            "inference_requests_generated": 0,
            "measurement": "operator_microprobe" if args.probe_only else "passive_external_only",
            "probe_only": args.probe_only, "model_will_start": not args.probe_only,
            "phases": phases}, indent=2))
        return 0
    return run(args.config, log_dir, args.variant, probe_only=args.probe_only,
               diagnose_q4=args.diagnose_q4, diagnose_mixed=args.diagnose_mixed)


if __name__ == "__main__":
    raise SystemExit(main())
