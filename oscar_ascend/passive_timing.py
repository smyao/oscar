"""Bounded asynchronous event samples of externally supplied service traffic.

Archive #70-73/#94/#95/#125/#133/#140-145: this collector never starts the
native HTTP profiler, synchronizes a hot phase, enters graph capture, or turns
four TP ranks into one wall time. It records NPU events only for sampled steps
after the passive observer arms an external traffic window.
"""
from __future__ import annotations

import atexit
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import threading
import time

from .integration.dummy_context import is_native_dummy_run

_MAX_PENDING = 16
_MAX_PHASES = 256
_QUOTAS = {"prefill": 4, "decode": 8, "mixed": 4, "unknown": 2}
_OSCAR_PHASES = frozenset({"prepare", "current_source_suppress", "rotate", "fia",
    "current_slot_guard", "current_native_fia", "merge", "phase1_stores", "status_guard"})
_NATIVE_PHASES = frozenset({"native_attention"})
_tls = threading.local()
_pending: list[dict] = []
_seen = {key: 0 for key in _QUOTAS}
_taken = {key: 0 for key in _QUOTAS}
_last_check = 0.0
_control: dict | None = None
_serial = 0
_current_run_id: str | None = None
_skipped_capacity = 0


def _control_record() -> dict | None:
    global _control, _last_check, _current_run_id, _skipped_capacity
    path = os.environ.get("OSCAR_PASSIVE_TIMING_CONTROL")
    if not path:
        return None
    now = time.monotonic()
    if _control is not None and now - _last_check < 1.0:
        return _control
    _last_check = now
    try:
        record = json.loads(Path(path).read_text())
    except FileNotFoundError:
        _control = None
        return None
    if (not isinstance(record, dict) or type(record.get("enabled")) is not bool
            or not isinstance(record.get("run_id"), str)):
        raise ValueError("invalid passive timing control record")
    _control = record if record["enabled"] else None
    if _control is not None and _control["run_id"] != _current_run_id:
        _current_run_id = _control["run_id"]
        _skipped_capacity = 0
        for key in _QUOTAS:
            _seen[key] = _taken[key] = 0
    return _control


def _metadata(scheduler_output) -> dict:
    counts = getattr(scheduler_output, "num_scheduled_tokens", None)
    if not isinstance(counts, dict):
        return {"bucket": "unknown", "scheduled_prompt_tokens": None,
                "scheduled_decode_tokens": None, "scheduled_requests": None,
                "preempted_requests": None, "scheduled_draft_tokens": None}
    new = {item.req_id for item in getattr(scheduler_output, "scheduled_new_reqs", ())}
    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    cached_outputs = (dict(zip(cached.req_ids, cached.num_output_tokens))
                      if cached is not None else {})
    prompt = decode = unknown = prompt_reqs = decode_reqs = 0
    for req_id, amount in counts.items():
        if type(amount) is not int or amount < 0:
            unknown += 1
        elif req_id in new or cached_outputs.get(req_id) == 0:
            prompt += amount; prompt_reqs += 1
        elif req_id in cached_outputs:
            decode += amount; decode_reqs += 1
        else:
            unknown += 1
    bucket = ("unknown" if unknown else "mixed" if prompt and decode else
              "prefill" if prompt else "decode" if decode else "unknown")
    spec = getattr(scheduler_output, "scheduled_spec_decode_tokens", None)
    preempted = getattr(scheduler_output, "preempted_req_ids", None)
    return {"bucket": bucket,
            "scheduled_prompt_tokens": prompt if not unknown else None,
            "scheduled_decode_tokens": decode if not unknown else None,
            "scheduled_prompt_requests": prompt_reqs if not unknown else None,
            "scheduled_decode_requests": decode_reqs if not unknown else None,
            "scheduled_requests": len(counts),
            "preempted_requests": len(preempted) if preempted is not None else None,
            "scheduled_draft_tokens": sum(map(len, spec.values())) if isinstance(spec, dict) else None,
            "accepted_tokens": None,
            "accepted_scope": "metrics_window_only"}


def active() -> bool:
    return getattr(_tls, "step", None) is not None


def _write(record: dict) -> None:
    path = os.environ.get("OSCAR_TRACE_DIR")
    if not path:
        return
    dest = Path(path) / f"passive-step-{os.getpid()}.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def _missing_record(step: dict, reason: str) -> dict:
    """Emit metadata only when event completion cannot be proven."""
    return {**{key: value for key, value in step.items()
                if key not in {"begin", "end", "stream", "phases", "missing_reported"}},
            "t": "oscar-passive-step", "status": "missing", "missing_reason": reason,
            "step_device_ms": None, "attention_union_ms": None,
            "attention_backend": "native" if step.get("variant") == "native" else "oscar",
            "oscar_union_ms": None, "residual_ms": None,
            "scopes": [], "partial_envelopes": []}


def _resolved(step: dict) -> dict:
    base = {key: value for key, value in step.items()
            if key not in {"begin", "end", "stream", "phases", "missing_reported"}}
    native = step["variant"] == "native"
    attention_phases = _NATIVE_PHASES if native else _OSCAR_PHASES
    base.update(t="oscar-passive-step", status="missing", step_device_ms=None,
                attention_union_ms=None, attention_backend="native" if native else "oscar",
                oscar_union_ms=None, residual_ms=None, scopes=[], partial_envelopes=[])
    if step.get("missing"):
        # A native execute/sample stream transition invalidates the full step,
        # yet already completed model/proposer envelopes remain useful. Only
        # same-stream child OSCAR intervals may be subtracted inside each one.
        for outer in step["phases"]:
            if (outer["phase"] not in {"target_forward", "draft_proposal"}
                    or outer.get("end") is None):
                continue
            try:
                duration = float(outer["begin"].elapsed_time(outer["end"]))
                children = []
                graph = False
                for inner in step["phases"]:
                    if inner.get("end") is None:
                        continue
                    a = float(outer["begin"].elapsed_time(inner["begin"]))
                    b = float(outer["begin"].elapsed_time(inner["end"]))
                    if 0 <= a <= b <= duration + 0.1:
                        graph |= inner["phase"] == "graph_replay"
                        if inner["phase"] in attention_phases:
                            children.append((a, b))
                children.sort()
                covered = 0.0
                if children:
                    left, right = children[0]
                    for a, b in children[1:]:
                        if a > right:
                            covered += right-left; left, right = a, b
                        else:
                            right = max(right, b)
                    covered += right-left
                valid = not graph and bool(children)
                base["partial_envelopes"].append({"phase": outer["phase"],
                    "duration_ms": duration, "attention_union_ms": covered if valid else None,
                    "attention_backend": base["attention_backend"],
                    "oscar_union_ms": covered if valid and not native else None,
                    "residual_ms": duration-covered if valid and covered <= duration + 0.1 else None,
                    "scope": "same_stream_partial_envelope" if valid else "missing_breakdown"})
            except Exception as error:
                base["partial_envelopes"].append({"phase": outer["phase"],
                    "scope": "missing", "reason": f"{type(error).__name__}: {error}"})
    end = step.get("end")
    if end is None or step.get("missing"):
        base["missing_reason"] = step.get("missing", "step_end_not_recorded")
        return base
    try:
        duration = float(step["begin"].elapsed_time(end))
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("invalid step event duration")
        scopes = []
        for item in step["phases"]:
            start_ms = float(step["begin"].elapsed_time(item["begin"]))
            finish_ms = float(step["begin"].elapsed_time(item["end"]))
            if not (0 <= start_ms <= finish_ms <= duration + 0.1):
                raise ValueError("phase event escapes its step envelope")
            scopes.append({"phase": item["phase"], "start_ms": start_ms,
                           "end_ms": finish_ms, "duration_ms": finish_ms-start_ms,
                           "fields": item["fields"]})
        base["step_device_ms"], base["scopes"] = duration, scopes
        if any(item["phase"] == "graph_replay" for item in scopes):
            base["missing_reason"] = "graph_replay_has_no_python_oscar_phase_breakdown"
            return base
        intervals = sorted((item["start_ms"], item["end_ms"]) for item in scopes
                           if item["phase"] in attention_phases)
        if not intervals:
            base["missing_reason"] = ("no_native_attention_phase_events" if native
                                      else "no_oscar_phase_events")
            return base
        covered = 0.0
        if intervals:
            left, right = intervals[0]
            for a, b in intervals[1:]:
                if a > right:
                    covered += right-left; left, right = a, b
                else:
                    right = max(right, b)
            covered += right-left
        if covered > duration + 0.1:
            raise ValueError("OSCAR interval union exceeds step event")
        base.update(status="measured", attention_union_ms=covered,
                    oscar_union_ms=None if native else covered,
                    residual_ms=max(0.0, duration-covered),
                    residual_scope="same_rank_same_step_same_stream_envelope")
        if native:
            base["native_attention_backends"] = sorted({item["fields"].get("backend")
                for item in scopes if item["phase"] == "native_attention"
                and isinstance(item["fields"].get("backend"), str)})
    except Exception as error:
        base["missing_reason"] = f"event_resolution_failed: {type(error).__name__}: {error}"
    return base


def _flush(*, final: bool = False) -> None:
    keep = []
    for step in _pending:
        try:
            complete = (bool(step["end"].query()) if step.get("end") is not None
                        else bool(step.get("missing")) and bool(step["begin"].query()) and all(
                            bool(item["begin"].query()) and
                            (item.get("end") is None or bool(item["end"].query()))
                            for item in step["phases"]))
        except Exception as error:
            step["missing"] = f"event_query_failed: {type(error).__name__}: {error}"
            if not step.get("missing_reported"):
                _write(_missing_record(step, step["missing"]))
                step["missing_reported"] = True
            keep.append(step)  # Retain unfinished device-event handles.
            continue
        if complete:
            _write(_resolved(step))
        elif final:
            step["missing"] = "step_event_incomplete_at_worker_exit"
            if not step.get("missing_reported"):
                _write(_missing_record(step, step["missing"]))
                step["missing_reported"] = True
            keep.append(step)  # Exit may reclaim memory; never do so in a hot step.
        else:
            keep.append(step)
    _pending[:] = keep


def begin_step(scheduler_output) -> None:
    """Called only at the outer native runner boundary, never inside CV."""
    global _serial, _skipped_capacity
    _flush()
    if active():
        previous = _tls.step
        previous["missing"] = "next_step_started_before_sample_tokens"
        _pending.append(previous)
        _tls.step = None
        _flush()
    control = _control_record()
    if control is None or is_native_dummy_run():
        return
    fields = _metadata(scheduler_output)
    bucket = fields["bucket"]
    _seen[bucket] += 1
    if (_taken[bucket] >= _QUOTAS[bucket] or
            not (_seen[bucket] == 1 or _seen[bucket] % 64 == 0)):
        return
    if len(_pending) >= _MAX_PENDING:
        _skipped_capacity += 1
        if _skipped_capacity & (_skipped_capacity-1) == 0:
            _write({"t": "oscar-passive-step", "status": "skipped",
                    "reason": "bounded_pending_capacity", "skipped_count": _skipped_capacity,
                    "run_id": control["run_id"], "bucket": bucket,
                    "pid": os.getpid(), "ts_unix": time.time()})
        return  # Never create or destroy another unfinished event.
    import torch
    if torch.npu.is_current_stream_capturing():
        return
    stream = torch.npu.current_stream()
    event = torch.npu.Event(enable_timing=True)
    event.record(stream)
    distributed = getattr(torch, "distributed", None)
    rank = (distributed.get_rank() if distributed is not None and distributed.is_initialized()
            else None)
    _serial += 1
    _taken[bucket] += 1
    _tls.step = {"run_id": control["run_id"], "variant": control.get("variant", "oscar"),
                 "step_id": f"{os.getpid()}-{_serial}", "pid": os.getpid(),
                 "rank": rank,
                 "ts_unix": time.time(), "bucket": bucket, **fields,
                 "begin": event, "end": None, "stream": stream, "phases": [],
                 "missing": None}


def end_step(error: BaseException | None = None) -> None:
    step = getattr(_tls, "step", None)
    _tls.step = None
    if step is None:
        return
    if error is not None:
        step["missing"] = f"runner_error: {type(error).__name__}: {error}"
        _pending.append(step)
        _flush()
        return
    import torch
    if torch.npu.is_current_stream_capturing() or torch.npu.current_stream() != step["stream"]:
        step["missing"] = "step_stream_mismatch_or_capture"
        _pending.append(step)
        _flush()
        return
    step["end"] = torch.npu.Event(enable_timing=True)
    step["end"].record(step["stream"])
    _write({"t": "oscar-passive-step", "status": "pending", "run_id": step["run_id"],
            "variant": step["variant"], "step_id": step["step_id"],
            "pid": step["pid"], "rank": step["rank"],
            "ts_unix": step["ts_unix"], "bucket": step["bucket"]})
    _pending.append(step)
    if len(_pending) > _MAX_PENDING:
        # A concurrent caller may race the begin_step capacity check. Keep
        # every unfinished event alive; the next begin skips sampling until
        # query() drains this queue. Never pop/destroy or resolve it here.
        step["missing"] = "bounded_pending_capacity_race"
        _write(_missing_record(step, step["missing"]))
        step["missing_reported"] = True
    _flush()


@contextmanager
def phase(name: str, **fields):
    step = getattr(_tls, "step", None)
    if step is None or is_native_dummy_run():
        yield
        return
    import torch
    if torch.npu.is_current_stream_capturing() or torch.npu.current_stream() != step["stream"]:
        step["missing"] = "phase_stream_mismatch_or_capture"
        yield
        return
    if len(step["phases"]) >= _MAX_PHASES:
        step["missing"] = "bounded_phase_event_budget_exceeded"
        yield
        return
    begin = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    begin.record(step["stream"])
    try:
        yield
    except BaseException as error:
        step["missing"] = f"phase_error: {type(error).__name__}: {error}"
        step["phases"].append({"phase": name, "fields": fields,
                               "begin": begin, "end": None})
        raise
    else:
        end.record(step["stream"])
        step["phases"].append({"phase": name, "fields": fields, "begin": begin, "end": end})


@atexit.register
def _finalize() -> None:
    step = getattr(_tls, "step", None)
    if step is not None:
        step["missing"] = "worker_exit_with_active_step"
        # No completion query exists for an interrupted active step. Preserve
        # metadata only; elapsed_time could implicitly wait during teardown.
        _write(_missing_record(step, step["missing"]))
    _flush(final=True)
