"""Explicit, bounded worker-thread NPU profiles for paired service diagnosis.

Archive #133: create/start/step/export/stop on the actual runner thread, never
HTTP profiler RPC; finish RECORD_AND_SAVE before stop. #140/#143: host launch
intervals are not device time, TP ranks are not summed wall time, graph replay
stays native. No per-phase NPU events, synchronization, or tensor readbacks
are added; the requested profiler export owns its normal completion work.
"""
from __future__ import annotations

import atexit
from contextlib import contextmanager, nullcontext
import json
import math
import os
from pathlib import Path
import re
import threading
import time

from .integration.dummy_context import is_native_dummy_run
from .passive_timing import _metadata

_METRICS = ("PipeUtilization", "Memory")
_MAX_SCOPES = 4096
_NOOP = nullcontext()
_tls = threading.local()
_run = None
_serial = 0


class ProfilerUnsupported(RuntimeError):
    """Requested native profiling API or metric is unavailable."""


def configured() -> bool:
    path = os.environ.get("OSCAR_TIMER_CONTROL")
    return bool(path and Path(path).is_file())


def active() -> bool:
    return getattr(_tls, "step", None) is not None


def _atomic(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _rank():
    import torch
    dist = getattr(torch, "distributed", None)
    if dist is not None and dist.is_initialized():
        return int(dist.get_rank())
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def _capture():
    import torch
    npu = getattr(torch, "npu", None)
    return bool(npu is not None and npu.is_current_stream_capturing())


def _read_control():
    path = os.environ.get("OSCAR_TIMER_CONTROL")
    if not path:
        return None
    try:
        value = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None
    if not isinstance(value, dict) or type(value.get("enabled")) is not bool:
        raise ValueError("OSCAR_TIMER_CONTROL requires boolean enabled")
    if not value["enabled"]:
        return None
    if (not isinstance(value.get("run_id"), str)
            or not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", value["run_id"])
            or value["run_id"] in {".", ".."}):
        raise ValueError("invalid diagnostic timer run_id")
    if not isinstance(value.get("variant"), str) or not value["variant"]:
        raise ValueError("diagnostic timer requires variant")
    if not isinstance(value.get("output_dir"), str) or not Path(value["output_dir"]).is_absolute():
        raise ValueError("diagnostic timer output_dir must be absolute")
    if type(value.get("profile", True)) is not bool:
        raise ValueError("diagnostic timer profile must be boolean")
    for key in ("target_batch", "min_decode_requests", "min_early_history_tokens"):
        if key in value and (type(value[key]) is not int or value[key] < 1):
            raise ValueError(f"diagnostic timer {key} must be positive integer")
    return value


def _step_metadata(scheduler_output):
    result = _metadata(scheduler_output)
    counts = getattr(scheduler_output, "num_scheduled_tokens", {})
    if not isinstance(counts, dict):
        counts = {}
    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    history = getattr(cached, "num_computed_tokens", None)
    contexts, warm = [], []
    if isinstance(history, (list, tuple)):
        for req, computed, outputs in zip(cached.req_ids, history, cached.num_output_tokens):
            if req in counts and type(computed) is int:
                contexts.append(computed + counts[req])
                if outputs == 0:
                    warm.append(computed)
    for request in getattr(scheduler_output, "scheduled_new_reqs", ()):
        computed = getattr(request, "num_computed_tokens", None)
        if request.req_id in counts and type(computed) is int:
            contexts.append(computed + counts[request.req_id])
    complete = len(contexts) == len(counts) and bool(contexts)
    result.update(context_min=min(contexts) if complete else None,
                  context_max=max(contexts) if complete else None,
                  context_source="scheduler_cpu_num_computed_tokens" if complete else "missing_cpu_metadata",
                  warm_prefill_history_max=max(warm) if warm else None,
                  warm_prefill_requests=len(warm))
    return result


def _summary(final=False):
    if _run is None:
        return
    expected = {group: list(_METRICS) for group in ("early", "decode")}
    missing = {group: [metric for metric in metrics if metric not in _run["taken"][group]]
               for group, metrics in expected.items()}
    record = {key: _run[key] for key in ("run_id", "variant", "rank", "pid", "output_dir",
                                        "target_batch", "min_decode_requests")}
    record.update(schema_version=1, t="oscar-diagnostic-run", final=final,
                  expected_windows=expected, taken_windows=_run["taken"],
                  window_statuses=_run["window_statuses"],
                  missing_windows=missing, observed_steps=_run["observed"],
                  max_decode_requests=_run["max_decode_requests"],
                  min_early_history_tokens=_run["min_early_history_tokens"],
                  early_history_unknown_steps=_run["early_history_unknown_steps"],
                  skipped_cold_prefill_steps=_run["skipped_cold_prefill_steps"],
                  status="complete" if not any(missing.values()) and len(_run["window_statuses"]) == 4 and all(
                      value in {"complete", "host_only"} for value in _run["window_statuses"].values()) else "missing",
                  missing_reason="mature_decode_not_observed" if missing["decode"] else None,
                  device_acceptance="requires_profile_artifact_analysis",
                  profiled_steps_are_uninstrumented_performance=False)
    _atomic(Path(_run["directory"]) / "run-summary.json", record)


def _new_run(control):
    global _run
    rank = _rank()
    directory = Path(control["output_dir"]) / "windows" / control["run_id"] / f"rank{rank}-pid{os.getpid()}"
    _run = {**control, "rank": rank, "pid": os.getpid(), "directory": str(directory),
            "target_batch": control.get("target_batch"),
            "min_decode_requests": control.get("min_decode_requests", control.get("target_batch", 1)),
            "min_early_history_tokens": control.get("min_early_history_tokens", 16384),
            "taken": {"early": [], "decode": []}, "observed": 0, "max_decode_requests": 0,
            "early_history_unknown_steps": 0, "skipped_cold_prefill_steps": 0, "window_statuses": {}}
    _summary()


def _create_profiler(directory, metric, worker_name):
    """Native API only. A missing PMU metric never substitutes another metric."""
    try:
        import torch
        import torch_npu
    except (ImportError, OSError) as error:
        raise ProfilerUnsupported(f"torch_npu import unavailable: {error}") from error
    if not torch.npu.is_available():
        raise ProfilerUnsupported("NPU is unavailable")
    api = torch_npu.profiler
    metric_value = getattr(api.AiCMetrics, metric, None)
    if metric_value is None:
        raise ProfilerUnsupported(f"AiCMetrics.{metric} is unsupported by installed torch_npu")
    experimental = api._ExperimentalConfig(
        export_type=api.ExportType.Text, profiler_level=api.ProfilerLevel.Level1,
        aic_metrics=metric_value, data_simplification=False)
    # active=1 starts in RECORD_AND_SAVE; step() saves and moves to NONE.
    # Unlike HTTP start/stop (#133), no worker waits for an external stop RPC.
    return api.profile(
        activities=[api.ProfilerActivity.CPU, api.ProfilerActivity.NPU],
        schedule=api.schedule(wait=0, warmup=0, active=1, repeat=1),
        record_shapes=False, profile_memory=False, with_stack=False, with_modules=False,
        experimental_config=experimental,
        on_trace_ready=api.tensorboard_trace_handler(str(directory), worker_name=worker_name,
                                                    async_mode=False))


def begin_step(scheduler_output):
    """Arm at a real execute_model boundary; no import/device work when off."""
    global _serial, _run
    if active():
        error = RuntimeError("next_execute_before_sample_tokens")
        end_step(error)
        raise error
    control = _read_control()
    if control is None:
        if _run is not None:
            _summary(final=True)
            _run = None  # a later control batch must not rewrite a finished summary per step
        return
    if is_native_dummy_run() or _capture():
        return
    if _run is None or control["run_id"] != _run["run_id"]:
        _summary(final=True)
        _new_run(control)
    elif any(control.get(key) != _run.get(key) for key in ("variant", "output_dir")):
        raise ValueError("diagnostic run identity changed without a new run_id")
    metadata = _step_metadata(scheduler_output)
    _run["observed"] += 1
    _run["max_decode_requests"] = max(_run["max_decode_requests"],
                                      metadata.get("scheduled_decode_requests") or 0)
    bucket = metadata["bucket"]
    warm_history = metadata["warm_prefill_history_max"]
    warm_early = bucket in {"prefill", "mixed"} and warm_history is not None and warm_history >= _run["min_early_history_tokens"]
    if bucket in {"prefill", "mixed"} and not warm_early:
        _run["early_history_unknown_steps" if warm_history is None else "skipped_cold_prefill_steps"] += 1
    group = ("early" if warm_early else "decode" if bucket == "decode"
             and (metadata.get("scheduled_decode_requests") or 0) >= _run["min_decode_requests"] else None)
    if group is None or len(_run["taken"][group]) >= len(_METRICS):
        return
    metric = _METRICS[len(_run["taken"][group])]
    _run["taken"][group].append(metric)
    _serial += 1
    window_id = f"step{_serial:06d}-{bucket}-{metric}"
    profile_dir = Path(_run["directory"]) / window_id
    manifest_path = profile_dir.with_suffix(".json")
    manifest = {"schema_version": 1, "t": "oscar-diagnostic-window", "run_id": _run["run_id"],
                "variant": _run["variant"], "rank": _run["rank"], "pid": os.getpid(),
                "window_id": window_id, "bucket": bucket, "metric": metric,
                "steps": [_serial], "profiled": control.get("profile", True),
                "profile_dir": str(profile_dir), "thread_id": threading.get_ident(),
                "target_batch": _run["target_batch"], "min_decode_requests": _run["min_decode_requests"],
                "scheduled_requests": metadata["scheduled_requests"],
                "scheduled_prompt_tokens": metadata["scheduled_prompt_tokens"],
                "scheduled_decode_tokens": metadata["scheduled_decode_tokens"],
                "scheduled_decode_requests": metadata.get("scheduled_decode_requests"),
                "context_min": metadata["context_min"], "context_max": metadata["context_max"],
                "warm_prefill_history_max": warm_history,
                "status": "starting", "profile_artifacts": [],
                "graph_mode_modified": False, "profiled_performance_scope": "instrumented_only"}
    _atomic(manifest_path, manifest)
    profiler = None
    setup_start_ns = time.perf_counter_ns()
    if manifest["profiled"]:
        try:
            profiler = _create_profiler(profile_dir, metric, f"rank{_run['rank']}-step{_serial}")
        except (ProfilerUnsupported, AttributeError, TypeError) as error:
            profiler = None
            manifest.update(status="unsupported", unsupported_reason=str(error))
        except Exception as error:
            manifest.update(status="error", error=f"profiler_create: {type(error).__name__}: {error}")
            _atomic(manifest_path, manifest)
            raise
        if profiler is not None:
            try:
                profiler.start()
                manifest["status"] = "collecting"
            except Exception as error:
                manifest.update(status="error", error=f"profiler_start: {type(error).__name__}: {error}")
                _atomic(manifest_path, manifest)
                raise  # partially initialized profiler state must not be reused
    else:
        manifest["status"] = "host_only"
    manifest["profile_start_ms"] = (time.perf_counter_ns() - setup_start_ns) / 1e6
    _atomic(manifest_path, manifest)
    _summary()
    _tls.step = {"schema_version": 1, "t": "oscar-diagnostic-step", **metadata,
                 "run_id": _run["run_id"], "variant": _run["variant"], "rank": _run["rank"],
                 "pid": os.getpid(), "step_id": _serial, "window_id": window_id, "metric": metric,
                 "profiled": profiler is not None, "host_start_ns": time.perf_counter_ns(),
                 "profile_start_ms": manifest["profile_start_ms"],
                 "host_step_includes_profile_setup": False,
                 "thread_id": threading.get_ident(), "graph_replays": 0, "scopes": [],
                 "dropped_scopes": 0, "target_batch": _run["target_batch"], "context_length": None,
                 "profiler": profiler, "manifest": manifest, "manifest_path": manifest_path,
                 "step_path": Path(control["output_dir"]) / f"timer-steps-rank{_run['rank']}-pid{os.getpid()}.jsonl"}


def end_step(error=None):
    step = getattr(_tls, "step", None)
    if step is None:
        return
    if threading.get_ident() != step["thread_id"]:
        raise RuntimeError("diagnostic profiler cannot stop on a different runner thread")
    _tls.step = None
    step["host_end_ns"] = time.perf_counter_ns()
    step["host_ms"] = (step["host_end_ns"] - step["host_start_ns"]) / 1e6
    manifest = step.pop("manifest")
    manifest_path = step.pop("manifest_path")
    path = step.pop("step_path")
    profiler = step.pop("profiler")
    profile_error = None
    export_start_ns = time.perf_counter_ns()
    if profiler is not None:
        try:
            profiler.step()  # RECORD_AND_SAVE -> export -> NONE
            profiler.stop()
            files = sorted(str(item.relative_to(Path(manifest["profile_dir"])))
                           for item in Path(manifest["profile_dir"]).rglob("*")
                           if item.is_file() and item.stat().st_size > 0)
            exported = [name for name in files if Path(name).name in {"trace_view.json", "kernel_details.csv"}
                        or Path(name).name.startswith("op_summary") and name.endswith(".csv")
                        or name.endswith(".pt.trace.json")]
            if not exported:
                manifest.update(status="missing", missing_reason="profiler_export_has_no_artifacts")
            else:
                manifest.update(status="complete" if error is None else "error", profile_artifacts=files)
        except Exception as failure:
            profile_error = failure
            manifest.update(status="error", error=f"profiler_finish: {type(failure).__name__}: {failure}")
    if error is not None:
        manifest.update(status="error", error=f"runner: {type(error).__name__}: {error}")
    elif step.get("graph_capture_calls"):
        manifest.update(status="missing", missing_reason="unexpected_graph_capture_in_profiled_step")
    manifest["finished_unix"] = time.time()
    manifest["profile_export_ms"] = (time.perf_counter_ns() - export_start_ns) / 1e6
    _atomic(manifest_path, manifest)
    step.update(status="complete" if error is None and profile_error is None else "error",
                profile_status=manifest["status"], device_ms=None,
                profile_export_ms=manifest["profile_export_ms"],
                host_step_includes_profile_export=False,
                host_timing_scope="runner_execute_through_sample_submission",
                profiled_steps_are_uninstrumented_performance=False)
    if error is not None:
        step["error"] = f"{type(error).__name__}: {error}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(step, sort_keys=True, allow_nan=False) + "\n")
    _run["window_statuses"][step["window_id"]] = manifest["status"]
    _summary()
    if profile_error is not None and error is None:
        raise profile_error


def note_graph_capture():
    """A dynamic capture cannot masquerade as a measured replay window."""
    step = getattr(_tls, "step", None)
    if step is not None:
        step["graph_capture_calls"] = step.get("graph_capture_calls", 0) + 1


def phase(name, **fields):
    """Host scope plus profiler correlation marker; never a device duration."""
    if not active():
        return _NOOP
    if type(name) is not str or not re.fullmatch(r"[a-zA-Z0-9_]+", name):
        raise ValueError("invalid diagnostic phase name")
    for key, value in fields.items():
        if type(key) is not str or (value is not None and type(value) not in (str, bool, int, float)):
            raise TypeError("diagnostic phase fields must be host scalars")
        if type(value) is float and not math.isfinite(value):
            raise ValueError("diagnostic phase scalar must be finite")
    return _phase(name, fields)


@contextmanager
def _phase(name, fields):
    step = _tls.step
    if _capture() or is_native_dummy_run():
        yield
        return
    if len(step["scopes"]) >= _MAX_SCOPES:
        step["dropped_scopes"] += 1
        yield
        return
    if name == "graph_replay":
        step["graph_replays"] += 1
    item = {"phase": name, "start_ns": time.perf_counter_ns(), "fields": fields}
    step["scopes"].append(item)
    if step["profiled"]:
        from torch.profiler import record_function
        scope = record_function("oscar_timer::" + name)
    else:
        scope = _NOOP
    try:
        with scope:
            yield
    finally:
        item["end_ns"] = time.perf_counter_ns()
        item["host_ms"] = (item["end_ns"] - item["start_ns"]) / 1e6


def _finalize():
    # Never stop native profiling from atexit, another thread, or a dead runner.
    # An unfinished manifest remains visibly collecting and cannot pass export.
    _summary(final=True)


atexit.register(_finalize)
