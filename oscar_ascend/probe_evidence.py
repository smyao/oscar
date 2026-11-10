"""Explicit short-probe host evidence, independent of native statistics.

Archive #27/#94/#125/#157/#163/#176: a completed HTTP request or a configured
MTP flag is not execution proof. Observe successful native scheduler methods;
never enable stats, alter their results, inspect tensors, or synchronize NPU.
Scheduled draft slots deliberately include native invalid/padded draft slots.
"""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path
import sys

_finder = None
_patches = []
_sequence = 0
_closed_runs = set()


def _window():
    directory = os.environ.get("OSCAR_PROBE_EVIDENCE_DIR")
    run_id = os.environ.get("OSCAR_PROBE_RUN_ID")
    if not directory or not run_id:
        return None
    identity = (directory, run_id)
    if identity in _closed_runs:
        return None
    try:
        window = json.loads((Path(directory) / "active.json").read_text())
    except FileNotFoundError:
        # The short gate can hand this process to formal serving. A terminal
        # close is cached locally: no per-step filesystem polling afterward.
        closed = Path(directory) / "closed.json"
        if closed.exists():
            if json.loads(closed.read_text()).get("run_id") != run_id:
                raise RuntimeError("invalid native probe terminal marker")
            _closed_runs.add(identity)
        return None
    if (window.get("run_id") != run_id or not isinstance(window.get("window_id"), str)
            or not window["window_id"]):
        raise RuntimeError("invalid native probe evidence window")
    return Path(directory), run_id, window["window_id"]


def _emit(window, event, request_id, **fields):
    global _sequence
    if window is None:
        return
    directory, run_id, window_id = window
    _sequence += 1
    record = dict(schema=1, run_id=run_id, window_id=window_id, pid=os.getpid(),
                  sequence=_sequence, event=event, request_id=request_id, **fields)
    # One scheduler writes each PID file. A successful write is host evidence,
    # not a device event; failures propagate instead of claiming acceptance.
    with (directory / f"scheduler-{os.getpid()}.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, separators=(",", ":")) + "\n")


def install(module):
    owner = module.Scheduler
    if any(entry[0] is owner for entry in _patches):
        return
    stats = owner.make_spec_decoding_stats
    preempt = owner._preempt_request

    @functools.wraps(stats)
    def observe_stats(self, spec_decoding_stats, num_draft_tokens, num_accepted_tokens,
                      num_invalid_spec_tokens, request_id):
        window = _window()
        result = stats(self, spec_decoding_stats, num_draft_tokens, num_accepted_tokens,
                       num_invalid_spec_tokens, request_id)
        if window is not None and num_draft_tokens:
            _emit(window, "mtp_verified", request_id,
                  scheduled_draft_slots=num_draft_tokens, num_accepted_tokens=num_accepted_tokens,
                  invalid_draft_slots=(num_invalid_spec_tokens or {}).get(request_id, 0))
        return result

    @functools.wraps(preempt)
    def observe_preempt(self, request, timestamp):
        window = _window()
        before = request.num_preemptions if window is not None else None
        result = preempt(self, request, timestamp)
        if window is not None:
            _emit(window, "preempted", request.request_id,
                  preemptions_before=before, preemptions_after=request.num_preemptions)
        return result

    for name, original, wrapped in (("make_spec_decoding_stats", stats, observe_stats),
                                     ("_preempt_request", preempt, observe_preempt)):
        setattr(owner, name, wrapped)
        _patches.append((owner, name, original, wrapped))


def register():
    """No native imports: compose with the plugin's generic deferred loader."""
    global _finder
    if not os.environ.get("OSCAR_PROBE_EVIDENCE_DIR"):
        return
    if not os.environ.get("OSCAR_PROBE_RUN_ID"):
        raise RuntimeError("explicit native probe evidence requires a run ID")
    if _finder is not None:
        return
    from .plugin import _NativeSeamFinder
    name = "vllm.v1.core.sched.scheduler"
    finder = _NativeSeamFinder({name: install}, lambda: _finder is finder,
                              lambda callback, module: callback(module) if _finder is finder else None)
    _finder = finder
    sys.meta_path.insert(0, _finder)
    try:
        module = sys.modules.get(name)
        if module is not None:
            if getattr(getattr(module, "__spec__", None), "_initializing", False):
                raise RuntimeError("probe registered while native scheduler is importing")
            install(module)
    except BaseException:
        unregister()
        raise


def unwrap_outer_hooks():
    """Allow either registration order with the production plugin's wrappers."""
    for entry in tuple(reversed(_patches)):
        owner, name, original, wrapped = entry
        if owner.__dict__.get(name) is wrapped:
            setattr(owner, name, original)
            _patches.remove(entry)


def unregister():
    global _finder
    for owner, name, original, wrapped in reversed(_patches):
        if owner.__dict__.get(name) is not wrapped:
            raise RuntimeError(f"native probe hook changed externally: {name}")
        setattr(owner, name, original)
    _patches.clear()
    if _finder in sys.meta_path:
        sys.meta_path.remove(_finder)
    _finder = None
    _closed_runs.clear()
