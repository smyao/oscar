"""Explicit profiling scopes without production events or synchronization.

Archive #70-73 and startup D.4: distinguish 725ms host prepare from 6500ms
NPU dequant, 18.7ms FIA and 209ms stores. Host launch time is never device time.
Native precedent: references/vllm-ascend/vllm_ascend/profiler/torch_npu_profiler.py.
"""
from __future__ import annotations

from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import time

PHASES = frozenset({"prepare", "rotate", "fia", "history_window", "merge",
                    "phase1_stores", "stage_restore", "phase0_store",
                    "materialize", "dequant", "status_guard"})
_NOOP = nullcontext()


def _enabled(name: str) -> bool:
    value = os.environ.get(name, "0").lower()
    if value not in {"0", "1", "false", "true"}:
        raise ValueError(f"{name} must be 0/1/false/true")
    return value in {"1", "true"}


def enabled() -> bool:
    return _enabled("OSCAR_TIMING") or _enabled("OSCAR_PROFILER")


def _write(record):
    # Only explicit timing uses host IO. No tensor values are inspected.
    print(json.dumps(record, sort_keys=True, allow_nan=False), file=sys.stderr, flush=True)


class _Phase:
    def __init__(self, name, fields):
        self.name, self.fields = name, fields
        self.clock = _enabled("OSCAR_TIMING")
        self.scope = None
        self.started = None

    def __enter__(self):
        from torch.profiler import record_function
        label = "oscar::" + self.name + "::" + json.dumps(self.fields, sort_keys=True, separators=(",", ":"))
        self.scope = record_function(label)
        self.scope.__enter__()
        if self.clock:
            self.started = time.perf_counter_ns()
        return self

    def __exit__(self, kind, value, traceback):
        elapsed = (time.perf_counter_ns() - self.started) / 1e9 if self.clock else None
        self.scope.__exit__(kind, value, traceback)
        if self.clock:
            _write({"t": "oscar-timing", "phase_end": self.name, "host_s": elapsed,
                    "pid": os.getpid(), "ts_unix": time.time(), "device_ms": None,
                    "device_evidence": "requires_npu_trace", "failed": kind is not None,
                    **self.fields})
        return False


def phase(name: str, **fields):
    """Wrap a launch or native metadata phase; safe for ordinary graph mode.

    Disabled returns one reusable null context without importing torch, reading
    a clock, creating events, recording scopes or synchronizing any stream.
    Fields must be already available host scalars, never tensor values.
    """
    if not enabled():
        return _NOOP
    if name not in PHASES:
        raise ValueError(f"unknown OSCAR timing phase: {name}")
    for key, value in fields.items():
        if type(key) is not str or (value is not None and type(value) not in (str, int, float, bool)):
            raise TypeError("timing metadata must contain host scalars only")
    if set(fields) & {"t", "phase_end", "host_s", "pid", "ts_unix", "device_ms", "device_evidence", "failed"}:
        raise ValueError("timing metadata cannot replace evidence fields")
    return _Phase(name, fields)


class ProfilingUnavailable(RuntimeError):
    """No target NPU is available for the explicitly requested profiler."""


class ProfileSession:
    """Explicit probe session; service workers can use the native profiler.

    The caller keeps FULL_DECODE_ONLY and its native graph scheduler. This
    class changes no graph, MTP, launch, quantization or device configuration.
    Profiler stop/export may synchronize as part of the requested measurement;
    production phase scopes do not add NPU events or synchronization.
    """
    def __init__(self, directory=None, worker_name=None):
        self.directory = Path(directory or os.environ.get("OSCAR_PROFILE_DIR", "reports/npu_profile")).resolve()
        self.worker_name = worker_name or f"oscar-{os.getpid()}"
        if Path(self.worker_name).name != self.worker_name:
            raise ValueError("profiler worker_name cannot contain a path")
        self.profiler = None
        self.status_path = self.directory / f"{self.worker_name}-status.json"
        self.status = {"backend": "torch_npu.profiler", "state": "not_run",
                       "pid": os.getpid(), "device_measurements": None,
                       "graph_mode_modified": False, "worker_name": self.worker_name}

    def _save(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.status, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.status_path)

    def __enter__(self):
        if not _enabled("OSCAR_PROFILER"):
            raise ValueError("ProfileSession requires explicit OSCAR_PROFILER=1")
        self._save()
        config_path = Path(os.environ.get("OSCAR_TARGET_CONFIG", Path(__file__).resolve().parents[1] / "configs/target.json"))
        target = json.loads(config_path.read_text())
        devices = target.get("devices")
        expected = ",".join(map(str, devices)) if isinstance(devices, list) else None
        if not devices or os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != expected:
            self.status["reason"] = "current-task NPU device selection is missing or mismatched"
            self._save()
            raise ProfilingUnavailable(self.status["reason"])
        try:
            import torch
            import torch_npu
        except (ImportError, OSError) as exc:
            self.status["reason"] = f"NPU profiler import unavailable: {type(exc).__name__}: {exc}"
            self._save()
            raise ProfilingUnavailable(self.status["reason"]) from exc
        if not torch.npu.is_available():
            self.status["reason"] = "NPU profiler requested but no NPU is available"
            self._save()
            raise ProfilingUnavailable(self.status["reason"])
        try:
            experimental = torch_npu.profiler._ExperimentalConfig(
                export_type=torch_npu.profiler.ExportType.Text,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                data_simplification=False,
            )
            self.profiler = torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU,
                            torch_npu.profiler.ProfilerActivity.NPU],
                with_stack=False, with_modules=False, record_shapes=True,
                profile_memory=True, experimental_config=experimental,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    str(self.directory), worker_name=self.worker_name),
            )
            self.profiler.start()
        except Exception as exc:
            self.status.update(state="failed", reason=f"profiler startup failed: {exc}")
            self._save()
            raise
        self.status.update(state="collecting", started_unix=time.time())
        self._save()
        return self

    def step(self):
        if self.profiler is None:
            raise RuntimeError("profiler session has not started")
        self.profiler.step()

    def __exit__(self, kind, value, traceback):
        try:
            self.profiler.stop()
        except Exception as exc:
            self.status.update(state="failed", reason=f"profiler stop/export failed: {exc}")
            self._save()
            raise
        self.status.update(state="captured" if kind is None else "failed",
                           stopped_unix=time.time(),
                           device_measurements="must_parse_real_trace")
        if kind is not None:
            self.status["reason"] = f"profiled workload failed: {kind.__name__}: {value}"
        self._save()
        return False


def profile_session(directory=None, worker_name=None):
    """Opt-in profiling context; absent OSCAR_PROFILER returns a no-op."""
    return ProfileSession(directory, worker_name) if _enabled("OSCAR_PROFILER") else _NOOP
