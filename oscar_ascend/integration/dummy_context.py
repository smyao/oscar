"""Archive #34/#36/#77/#78/#142: identify native synthetic warmup explicitly.

The native runner's MTP dummy runs use ChunkedPrefill even before graph
capture and fill slots with -1. Keep them on the complete CV padding path.
This stdlib-only scope neither inspects NPU data nor changes native inputs.
"""

from contextlib import contextmanager
from contextvars import ContextVar


_native_dummy_run = ContextVar("oscar_native_dummy_run", default=False)


def is_native_dummy_run() -> bool:
    return _native_dummy_run.get()


@contextmanager
def native_dummy_run():
    token = _native_dummy_run.set(True)
    try:
        yield
    finally:
        _native_dummy_run.reset(token)
