"""Archive #70-73/#133/#142-143: isolated, synchronized NPU event diagnosis.

Only a separately labelled repeat can arm this collector. The normal batch
creates no events and performs no phase-level filesystem polling. Never use
the native HTTP profiler (#133), insert events in a graph, or call host wall
time kernel time. Event intervals include stream dependencies/dispatch gaps.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time

from .integration.dummy_context import is_native_dummy_run


@dataclass(frozen=True)
class _Run:
    run_id: str
    directory: Path


_run: _Run | None = None


def refresh() -> None:
    """Poll once at the model wrapper boundary, never inside each phase."""
    global _run
    control = os.environ.get("OSCAR_DEVICE_TIMING_CONTROL")
    if not control:
        _run = None
        return
    path = Path(control)
    try:
        record = json.loads(path.read_text())
    except FileNotFoundError:
        _run = None  # A missing marker explicitly means disarmed.
        return
    if not isinstance(record, dict) or type(record.get("enabled")) is not bool:
        raise ValueError("invalid device-timing control record")
    if not record["enabled"]:
        _run = None
        return
    run_id = record.get("run_id")
    if not isinstance(run_id, str) or not 1 <= len(run_id) <= 128:
        raise ValueError("device timing requires a bounded diagnostic run_id")
    _run = _Run(run_id, path.parent)


def active() -> bool:
    return _run is not None and not is_native_dummy_run()


class DevicePhase:
    def __init__(self, name: str, fields: dict):
        self.name, self.fields, self.run = name, fields, _run
        self.begin = self.end = None
        self.started = None
        self.torch = None

    def __enter__(self):
        if self.run is None or is_native_dummy_run():
            return self
        import torch
        if not hasattr(torch, "npu"):
            raise RuntimeError("device timing requires the real torch_npu event API")
        if torch.npu.is_current_stream_capturing():
            return self  # A replay wrapper is measured outside the graph.
        self.torch = torch
        self.begin = torch.npu.Event(enable_timing=True)
        self.end = torch.npu.Event(enable_timing=True)
        self.stream = torch.npu.current_stream()
        self.started = time.perf_counter_ns()
        self.begin.record(self.stream)
        return self

    def _write(self, *, device_ms, error=None):
        distributed = self.torch.distributed
        rank = distributed.get_rank() if distributed.is_initialized() else None
        record = {"t": "oscar-device-event", "phase": self.name,
                  "pid": os.getpid(), "rank": rank, "run_id": self.run.run_id,
                  "ts_unix": time.time(),
                  "device_ms": device_ms,
                  "host_ms": (time.perf_counter_ns() - self.started) / 1e6,
                  "device_evidence": "npu_event", "debug_synchronization": True,
                  "scope": "synchronized_npu_stream_interval",
                  "failed": error is not None, "error": error, **self.fields}
        path = self.run.directory / f"device-events-{os.getpid()}.jsonl"
        with path.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    def __exit__(self, kind, value, traceback):
        if self.begin is None:
            return False
        if kind is not None:
            self._write(device_ms=None, error=f"{kind.__name__}: {value}")
            return False
        try:
            self.end.record(self.stream)
            # Deliberate diagnostic-only synchronization ensures final phase
            # records are complete before HTTP completion and worker cleanup.
            self.end.synchronize()
            elapsed = float(self.begin.elapsed_time(self.end))
            if not math.isfinite(elapsed) or elapsed < 0:
                raise RuntimeError(f"invalid NPU event interval {elapsed}")
        except BaseException as error:
            self._write(device_ms=None, error=f"{type(error).__name__}: {error}")
            raise
        self._write(device_ms=elapsed)
        return False
