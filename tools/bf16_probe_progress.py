"""Diagnostic-only checkpoints (archive #174); never run inside event intervals."""
from contextlib import contextmanager
from contextvars import ContextVar
import json
import time

_sink = ContextVar('bf16_probe_progress_sink', default=None)


@contextmanager
def recording(report, persist):
    started = time.monotonic()
    def write(fields):
        report['progress'] = dict(fields, elapsed_seconds=round(time.monotonic()-started, 3))
        persist()
        print('[oscar] BF16_HISTORY_PROGRESS '+json.dumps(report['progress']), flush=True)
    token = _sink.set(write)
    try:
        yield
    finally:
        _sink.reset(token)


def checkpoint(stage, **fields):
    sink = _sink.get()
    if sink is not None:
        sink(dict(stage=stage, **fields))
