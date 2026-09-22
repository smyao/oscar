# Archive #27/#34/#50-52/#70-73: distinguish kernel dispatch, capture return, replay launch and device completion.
"""Optional process-local evidence. No tensor readback or implicit timing sync."""
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

_seen=set()
_last={}
_lock=threading.Lock()

DEFAULT_PROGRESS_INTERVAL_SECONDS = 5.0


def _record(event, fields):
    torch=sys.modules.get("torch")
    distributed=getattr(torch,"distributed",None)
    rank=None
    if distributed is not None and distributed.is_initialized():
        rank=distributed.get_rank()
    return {"event":event,"pid":os.getpid(),"rank":rank,"wall_time":time.time(),**fields}


def _append(directory, record):
    path=Path(directory);path.mkdir(parents=True,exist_ok=True)
    with (path/f"worker-{os.getpid()}.jsonl").open("a") as output:
        output.write(json.dumps(record,sort_keys=True,allow_nan=False)+"\n")


def emit_once(event, *, key=None, **fields):
    directory=os.environ.get("OSCAR_TRACE_DIR")
    if not directory:
        return
    identity=(event,key)
    with _lock:
        if identity in _seen:
            return
        _append(directory, _record(event, fields))
        _seen.add(identity)


def progress_interval_seconds():
    raw=os.environ.get("OSCAR_PROGRESS_INTERVAL_SECONDS")
    if raw is None or not raw.strip():
        return DEFAULT_PROGRESS_INTERVAL_SECONDS
    try:
        value=float(raw)
    except ValueError:
        raise ValueError("OSCAR_PROGRESS_INTERVAL_SECONDS must be finite and positive") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError("OSCAR_PROGRESS_INTERVAL_SECONDS must be finite and positive")
    return value


def emit_throttled(event, *, key=None, min_interval=None, **fields):
    """Liveness heartbeat: at most one record per (event, key) per interval.

    Unlike emit_once this repeats, so a long prefill or decode keeps a fresh
    wall_time in the worker trace instead of freezing at the first signature.
    """
    directory=os.environ.get("OSCAR_TRACE_DIR")
    if not directory:
        return
    interval=progress_interval_seconds() if min_interval is None else min_interval
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("progress throttle interval must be finite and positive")
    now=time.monotonic()
    identity=(event,key)
    with _lock:
        previous=_last.get(identity)
        if previous is not None and now-previous < interval:
            return
        _append(directory, _record(event, fields))
        _last[identity]=now
