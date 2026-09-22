# Archive #27/#34/#50-52/#70-73: distinguish kernel dispatch, capture return, replay launch and device completion.
"""Optional process-local evidence. No tensor readback or implicit timing sync."""
import json
import os
from pathlib import Path
import sys
import threading
import time

_seen=set()
_lock=threading.Lock()


def emit_once(event, *, key=None, **fields):
    directory=os.environ.get("OSCAR_TRACE_DIR")
    if not directory:
        return
    identity=(event,key)
    with _lock:
        if identity in _seen:
            return
        torch=sys.modules.get("torch")
        distributed=getattr(torch,"distributed",None)
        rank=None
        if distributed is not None and distributed.is_initialized():
            rank=distributed.get_rank()
        record={"event":event,"pid":os.getpid(),"rank":rank,"wall_time":time.time(),**fields}
        path=Path(directory);path.mkdir(parents=True,exist_ok=True)
        with (path/f"worker-{os.getpid()}.jsonl").open("a") as output:
            output.write(json.dumps(record,sort_keys=True,allow_nan=False)+"\n")
        _seen.add(identity)
