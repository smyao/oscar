"""Strict, synchronous byte transfers through the native memcache SDK.

Archive: #34/#36 (fixed graph buffers), #37--49/#127/#128 (physical
pages and complete window state), #157/#163/#171 (errors and joint state).
The native MemcacheBackend is used for initialization, never for its put/get
wrappers: those wrappers can log a failure and return apparent completion.

Only SDK calls present in references/vllm-ascend are used.  In particular this
module does not assume a transactional manifest API. Host put/get additionally
follow Ascend/memcache doc/memcache_python_api.md (official API, 2026-11-09).
Calls are synchronous, have no automatic retries, and use bounded batches.
A deadline watchdog terminates the worker if SDK/RPC hangs: Python cannot
cancel an in-flight DMA and then allow its registered buffer to be reused.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from numbers import Integral
import os
import signal
import subprocess
import sys
from threading import RLock
import time
from typing import Any, Callable, Iterable, Sequence


REGISTRATION_ALIGNMENT = 2 * 1024 * 1024
COPY_L2G = 0
COPY_G2L = 1
# Bounded application limit, not an SDK constraint. A complete target-model
# 262K/TP4 manifest with 16 target FULL layers plus MTP exceeds 1 MiB.
MAX_HOST_OBJECT_BYTES = 8 * 1024 * 1024


class TransferError(RuntimeError):
    """A storage/protocol failure; it must never be converted to a cache hit."""


class MissingObjectsError(TransferError):
    """An ordinary cache miss detected before any destination was modified."""

    def __init__(self, keys: Sequence[str]):
        self.keys = tuple(keys)
        super().__init__(f"memcache objects missing: {self.keys!r}")


class ObjectSizeError(TransferError):
    """The stored byte length does not match the immutable object schema."""


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _timeout(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    return float(value)


_WATCHDOG_PROGRAM = r'''
import json, os, select, signal, sys, time
parent = int(sys.argv[1])
descriptor = sys.stdin.fileno()
pending = b""
active = None
while os.getppid() == parent:
    timeout = None if active is None else max(0.0, active["deadline"] - time.monotonic())
    ready, _, _ = select.select([descriptor], [], [], timeout)
    if ready:
        data = os.read(descriptor, 65536)
        if not data:
            break
        pending += data
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            active = json.loads(line)
    if active is not None and time.monotonic() >= active["deadline"]:
        print("FATAL: OSCAR memcache " + active["operation"] +
              " exceeded its SDK deadline; killing worker PID " + str(parent) +
              " to prevent reuse of in-flight DMA buffers", file=sys.stderr, flush=True)
        if os.getppid() == parent:
            os.kill(parent, signal.SIGKILL)
        break
'''


class _DeadlineWatchdog:
    """A separate process can enforce a deadline even if SDK holds the GIL.

    No NPU state is forked: a fresh Python interpreter runs only stdlib code.
    Pipe EOF disarms/exits it when the owning worker exits.  A timeout kills
    the worker, rather than returning control with outstanding DMA accesses.
    """

    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = _timeout(timeout_seconds)
        self.owner_pid = os.getpid()
        self.process = subprocess.Popen(
            [sys.executable, "-u", "-c", _WATCHDOG_PROGRAM, str(os.getpid())],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, close_fds=True,
        )

    def _send(self, value: Any) -> None:
        if os.getpid() != self.owner_pid:
            raise TransferError("memcache transport cannot be reused after fork")
        if self.process.poll() is not None or self.process.stdin is None:
            raise TransferError("memcache deadline watchdog is unavailable")
        try:
            self.process.stdin.write(json.dumps(value, separators=(",", ":")).encode() + b"\n")
            self.process.stdin.flush()
        except (OSError, ValueError) as exc:
            raise TransferError("memcache deadline watchdog pipe failed") from exc

    def run(self, operation: str, function: Callable[[], Any]) -> Any:
        deadline = time.monotonic() + self.timeout_seconds
        self._send({"operation": operation, "deadline": deadline})
        try:
            return function()
        finally:
            if time.monotonic() >= deadline:
                print(f"FATAL: OSCAR memcache {operation} exceeded its SDK deadline; terminating worker", file=sys.stderr, flush=True)
                os.kill(os.getpid(), signal.SIGKILL)
            self._send(None)

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=1.0)

    def __del__(self) -> None:
        if not hasattr(self, "process"):
            return  # Process creation failed before any watchdog existed.
        try:
            self.close()
        except Exception as exc:
            print(f"OSCAR memcache watchdog cleanup failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)


@dataclass(frozen=True)
class ByteSegment:
    address: int
    length: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "address", _integer(self.address, "address", minimum=1))
        object.__setattr__(self, "length", _integer(self.length, "length", minimum=1))
        if self.address + self.length > 1 << 64:
            raise ValueError("byte range exceeds the address space")

    @property
    def ptr(self) -> int:
        return self.address

    @property
    def size(self) -> int:
        return self.length


def _segment(value: Any) -> ByteSegment:
    if isinstance(value, ByteSegment):
        return value
    if hasattr(value, "address") and hasattr(value, "length"):
        return ByteSegment(value.address, value.length)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return ByteSegment(value[0], value[1])
    raise TypeError("segments require address/length or a (pointer, size) pair")


def _keys(values: Iterable[str]) -> tuple[str, ...]:
    keys = tuple(values)
    if any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("object keys must be nonempty strings")
    if any(len(key.encode("utf-8")) >= 256 for key in keys):
        raise ValueError("memcache object keys must be shorter than 256 UTF-8 bytes")
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate object keys are not allowed")
    return keys


@dataclass(frozen=True)
class TransferObject:
    key: str
    segments: tuple[ByteSegment, ...]

    def __post_init__(self) -> None:
        _keys((self.key,))
        segments = tuple(_segment(item) for item in self.segments)
        if not segments:
            raise ValueError("a transfer object requires at least one byte segment")
        object.__setattr__(self, "segments", segments)

    @property
    def nbytes(self) -> int:
        return sum(segment.length for segment in self.segments)


@dataclass(frozen=True)
class TransferReceipt:
    keys: tuple[str, ...]
    byte_counts: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        return sum(self.byte_counts)


class StrictMemcacheTransport:
    """Registered-buffer transport with explicit success or an exception.

    A failed SDK call poisons this instance.  A caller must stop using its
    destinations rather than execute attention on possibly partial state.
    Pre-transfer cache misses do not poison the instance.  Owning raw tensors
    can be retained with ``register_buffers(..., owners=...)``; they must not
    be resized/replaced while this transport is alive.
    """

    def __init__(
        self,
        store: Any,
        *,
        backend: Any = None,
        scheduler: bool = False,
        registration_alignment: int = REGISTRATION_ALIGNMENT,
        max_batch_objects: int = 256,
        max_batch_bytes: int = 256 * 1024 * 1024,
        timeout_seconds: float = 60.0,
        _watchdog: _DeadlineWatchdog | None = None,
    ) -> None:
        if store is None:
            raise ValueError("an initialized memcache store is required")
        self.store = store
        self.backend = backend
        self.scheduler = bool(scheduler)
        self.registration_alignment = _integer(registration_alignment, "registration_alignment", minimum=1)
        self.max_batch_objects = _integer(max_batch_objects, "max_batch_objects", minimum=1)
        self.max_batch_bytes = _integer(max_batch_bytes, "max_batch_bytes", minimum=1)
        self.timeout_seconds = _timeout(timeout_seconds)
        self._watchdog = _watchdog
        self._regions: list[ByteSegment] = []
        self._registration_attempts: list[ByteSegment] = []
        self._owners: list[Any] = []
        self._failure: TransferError | None = None
        self._closed = False
        self._last_write_keys: frozenset[str] = frozenset()
        self._lock = RLock()

    @classmethod
    def from_backend(cls, backend: Any, *, scheduler: bool = False, **kwargs: Any) -> StrictMemcacheTransport:
        watchdog = kwargs.get("_watchdog")
        if watchdog is None:
            watchdog = _DeadlineWatchdog(kwargs.get("timeout_seconds", 60.0))
            kwargs["_watchdog"] = watchdog
        watchdog.run("ensure_initialized", backend.ensure_initialized)
        return cls(backend.store, backend=backend, scheduler=scheduler, **kwargs)

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: Any,
        *,
        scheduler: bool = False,
        local_rank: int | None = None,
        **kwargs: Any,
    ) -> StrictMemcacheTransport:
        # Lazy import keeps this module usable in host tests.  The scheduler
        # uses the native init_bm=False metadata client without NPU buffers.
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import (
            MemcacheBackend,
        )

        def initialize() -> Any:
            if scheduler:
                return MemcacheBackend.create_scheduler_client(vllm_config.parallel_config)
            return MemcacheBackend(vllm_config.parallel_config, local_rank=local_rank, lazy_init=False)

        watchdog = _DeadlineWatchdog(kwargs.get("timeout_seconds", 60.0))
        kwargs["_watchdog"] = watchdog
        backend = watchdog.run("backend initialization", initialize)
        return cls.from_backend(backend, scheduler=scheduler, **kwargs)

    @property
    def registered_regions(self) -> tuple[ByteSegment, ...]:
        return tuple(self._regions)

    @property
    def failed(self) -> bool:
        return self._failure is not None

    def _healthy(self) -> None:
        if self._closed:
            raise TransferError("memcache transport is closed")
        if self._failure is not None:
            raise TransferError("memcache transport is unusable after an earlier SDK failure") from self._failure

    def _fail(self, error: TransferError) -> None:
        self._failure = error
        raise error

    def _call(self, method: str, *args: Any) -> Any:
        self._healthy()
        try:
            if self._watchdog is None:
                self._watchdog = _DeadlineWatchdog(self.timeout_seconds)
            return self._watchdog.run(method, lambda: getattr(self.store, method)(*args))
        except Exception as exc:
            error = TransferError(f"memcache {method} raised {type(exc).__name__}: {exc}")
            self._failure = error
            raise error from exc

    def _statuses(self, result: Any, keys: Sequence[str], operation: str, *, existence: bool = False) -> tuple[int, ...]:
        try:
            values = tuple(result)
        except (TypeError, ValueError):
            self._fail(TransferError(f"{operation}: expected one explicit status per key, got {result!r}"))
        if len(values) != len(keys):
            self._fail(TransferError(f"{operation}: expected {len(keys)} statuses, got {len(values)}"))
        invalid = [
            (key, value)
            for key, value in zip(keys, values)
            if isinstance(value, bool)
            or not isinstance(value, Integral)
            or (int(value) not in (0, 1) if existence else int(value) != 0)
        ]
        if invalid:
            self._fail(TransferError(f"{operation}: invalid/failed object statuses {invalid!r}"))
        return tuple(int(value) for value in values)

    def register_buffers(self, regions: Iterable[Any], *, owners: Iterable[Any] = ()) -> None:
        regions = tuple(_segment(region) for region in regions)
        with self._lock:
            self._healthy()
            if self.scheduler:
                raise TransferError("scheduler metadata clients must not register NPU buffers")
            # Retain owners even if a later registration fails: a partial SDK
            # registration must never outlive the allocation it points into.
            self._owners.extend(owners)
            for region in regions:
                if region.address % self.registration_alignment:
                    raise ValueError(f"registered buffer base must be {self.registration_alignment}-byte aligned")
                if any(region.address >= old.address and region.address + region.length <= old.address + old.length for old in self._regions):
                    continue
                if any(region.address < old.address + old.length and old.address < region.address + region.length for old in self._regions):
                    raise ValueError("overlapping raw buffer registrations must use one owning storage region")
                self._registration_attempts.append(region)
                result = self._call("register_buffer", region.address, region.length)
                # Native code does not check this result.  Until a deployed
                # SDK is proven otherwise, None is NOT evidence of success.
                self._statuses((result,), (f"buffer@{region.address:#x}",), "register_buffer")
                self._regions.append(region)

    def exists(self, keys: Iterable[str]) -> tuple[bool, ...]:
        keys = _keys(keys)
        with self._lock:
            self._healthy()
            result: list[bool] = []
            for start in range(0, len(keys), self.max_batch_objects):
                batch = keys[start : start + self.max_batch_objects]
                codes = self._statuses(self._call("batch_is_exist", list(batch)), batch, "batch_is_exist", existence=True)
                result.extend(bool(code) for code in codes)
            return tuple(result)

    def host_put(self, key: str, payload: bytes, *, allow_duplicate: bool = False) -> bool:
        """Store bounded manifest bytes through the documented host API."""
        _keys((key,))
        if not isinstance(payload, bytes) or not 0 < len(payload) <= MAX_HOST_OBJECT_BYTES:
            raise ValueError("host payload must be nonempty bytes within the metadata limit")
        with self._lock:
            self._healthy()
            if self.scheduler:
                raise TransferError("init_bm=False scheduler clients cannot write host data")
            result = self._call("put", key, payload)
            if allow_duplicate and not isinstance(result, bool) and isinstance(result, Integral) and int(result) == -3006:
                return False
            self._statuses((result,), (key,), "host_put")
            return True

    def host_remove(self, key: str, *, missing_ok: bool = True) -> bool:
        _keys((key,))
        with self._lock:
            self._healthy()
            if self.scheduler:
                raise TransferError("init_bm=False scheduler clients cannot remove host data")
            result = self._call("remove", key)
            if missing_ok and not isinstance(result, bool) and isinstance(result, Integral) and int(result) == -3007:
                return False
            self._statuses((result,), (key,), "host_remove")
            return True

    def host_remove_batch(self, keys: Iterable[str], *, missing_ok: bool = True) -> tuple[bool, ...]:
        """Delete bounded batches; exact object-not-found is idempotent only."""
        keys = _keys(keys)
        with self._lock:
            self._healthy()
            if self.scheduler:
                raise TransferError("init_bm=False scheduler clients cannot remove host data")
            removed = []
            for start in range(0, len(keys), self.max_batch_objects):
                batch = keys[start : start + self.max_batch_objects]
                result = self._call("remove_batch", list(batch))
                try:
                    values = tuple(result)
                except TypeError:
                    self._fail(TransferError("host_remove_batch returned no explicit per-key statuses"))
                if len(values) != len(batch):
                    self._fail(TransferError("host_remove_batch returned an incomplete status list"))
                accepted = (0, -3007) if missing_ok else (0,)
                invalid = [(key, code) for key, code in zip(batch, values)
                           if isinstance(code, bool) or not isinstance(code, Integral) or int(code) not in accepted]
                if invalid:
                    self._fail(TransferError(f"host_remove_batch failed object statuses: {invalid!r}"))
                removed.extend(int(code) == 0 for code in values)
            return tuple(removed)

    def host_get(self, key: str) -> bytes:
        """Read manifest bytes; SDK empty-on-error must not become a hit."""
        _keys((key,))
        with self._lock:
            self._healthy()
            if self.scheduler:
                raise TransferError("init_bm=False scheduler clients cannot read host data")
            value = self._call("get", key)
            if not isinstance(value, bytes):
                self._fail(TransferError("host_get returned a non-bytes SDK result"))
            if not value:
                if not self.exists((key,))[0]:
                    raise MissingObjectsError((key,))
                self._fail(TransferError("host_get returned empty bytes for an existing object"))
            if len(value) > MAX_HOST_OBJECT_BYTES:
                self._fail(TransferError("host_get exceeded the bounded metadata size"))
            return value

    def object_sizes(self, keys: Iterable[str]) -> tuple[int, ...]:
        """Return zero for missing objects; reject malformed SDK metadata."""
        keys = _keys(keys)
        with self._lock:
            self._healthy()
            sizes: list[int] = []
            for start in range(0, len(keys), self.max_batch_objects):
                batch = keys[start : start + self.max_batch_objects]
                raw = self._call("batch_get_key_info", list(batch))
                try:
                    infos = tuple(raw)
                    if len(infos) != len(batch):
                        raise ValueError(f"expected {len(batch)} key infos, got {len(infos)}")
                    sizes.extend(_integer(info.size(), "key_info.size") for info in infos)
                except Exception as exc:
                    error = TransferError(f"batch_get_key_info: invalid metadata: {exc}")
                    self._failure = error
                    raise error from exc
            return tuple(sizes)

    def _objects(self, objects: Iterable[TransferObject]) -> tuple[TransferObject, ...]:
        objects = tuple(objects)
        if self.scheduler:
            raise TransferError("scheduler metadata clients must not transfer NPU buffers")
        if any(not isinstance(obj, TransferObject) for obj in objects):
            raise TypeError("objects must be TransferObject instances")
        _keys(obj.key for obj in objects)
        for obj in objects:
            if obj.nbytes > self.max_batch_bytes:
                raise ValueError(f"object {obj.key!r} exceeds max_batch_bytes; split at a schema object boundary")
            for segment in obj.segments:
                if not any(
                    segment.address >= region.address
                    and segment.address + segment.length <= region.address + region.length
                    for region in self._regions
                ):
                    raise ValueError(f"object {obj.key!r} references an unregistered byte range")
        return objects

    def _batches(self, objects: Sequence[TransferObject]) -> Iterable[tuple[TransferObject, ...]]:
        batch: list[TransferObject] = []
        byte_count = 0
        for obj in objects:
            if batch and (len(batch) == self.max_batch_objects or byte_count + obj.nbytes > self.max_batch_bytes):
                yield tuple(batch)
                batch, byte_count = [], 0
            batch.append(obj)
            byte_count += obj.nbytes
        if batch:
            yield tuple(batch)

    def _transfer(self, objects: Sequence[TransferObject], *, load: bool) -> TransferReceipt:
        method = "batch_get_into_layers" if load else "batch_put_from_layers"
        for batch in self._batches(objects):
            keys = [obj.key for obj in batch]
            addresses = [[segment.address for segment in obj.segments] for obj in batch]
            lengths = [[segment.length for segment in obj.segments] for obj in batch]
            result = self._call(method, keys, addresses, lengths, COPY_G2L if load else COPY_L2G)
            self._statuses(result, keys, method)
        return TransferReceipt(tuple(obj.key for obj in objects), tuple(obj.nbytes for obj in objects))

    def put(self, objects: Iterable[TransferObject]) -> TransferReceipt:
        """Save complete objects. Caller synchronizes producers before entry."""
        with self._lock:
            self._healthy()
            objects = self._objects(objects)
            self._last_write_keys = frozenset(obj.key for obj in objects)
            return self._transfer(objects, load=False)

    def rollback_private_write(self, keys: Iterable[str]) -> tuple[bool, ...]:
        """Cleanup only an unpublished private write, preserving its failure.

        The connector must call this immediately after a private save fails,
        before it could publish any completion. Only keys from the most recent
        validated put are allowed; arbitrary remote deletion is impossible.
        No data-read/write operation becomes usable after a poisoned transfer.
        """
        keys = _keys(keys)
        with self._lock:
            if self._closed or self.scheduler:
                raise TransferError("private rollback requires an open worker transport")
            if not set(keys).issubset(self._last_write_keys):
                raise TransferError("private rollback keys are outside the most recent write")
            if self._watchdog is None:
                self._watchdog = _DeadlineWatchdog(self.timeout_seconds)
            removed = []
            for start in range(0, len(keys), self.max_batch_objects):
                batch = keys[start : start + self.max_batch_objects]
                # Bypass _call/_healthy solely for cleanup. Its exception must
                # not replace the original poisoned transfer's _failure.
                try:
                    result = self._watchdog.run("rollback remove_batch", lambda: self.store.remove_batch(list(batch)))
                    values = tuple(result)
                except Exception as exc:
                    raise TransferError(f"private rollback remove_batch failed: {exc}") from exc
                if len(values) != len(batch):
                    raise TransferError("private rollback returned an incomplete status list")
                invalid = [(key, code) for key, code in zip(batch, values)
                           if isinstance(code, bool) or not isinstance(code, Integral) or int(code) not in (0, -3007)]
                if invalid:
                    raise TransferError(f"private rollback failed object statuses: {invalid!r}")
                removed.extend(int(code) == 0 for code in values)
            return tuple(removed)

    def get(self, objects: Iterable[TransferObject]) -> TransferReceipt:
        """Load exact-size objects, or fail before attention can consume them.

        Existence/size checks do not promise an eviction-proof lease.  If an
        object disappears during get, its nonzero SDK result is a hard error.
        No partially written destination may be submitted to a graph.
        """
        with self._lock:
            self._healthy()
            objects = self._objects(objects)
            keys = tuple(obj.key for obj in objects)
            present = self.exists(keys)
            missing = tuple(key for key, found in zip(keys, present) if not found)
            if missing:
                raise MissingObjectsError(missing)
            sizes = self.object_sizes(keys)
            missing = tuple(key for key, size in zip(keys, sizes) if not size)
            if missing:
                raise MissingObjectsError(missing)
            mismatch = [(obj.key, size, obj.nbytes) for obj, size in zip(objects, sizes) if size != obj.nbytes]
            if mismatch:
                self._fail(ObjectSizeError(f"stored/requested byte counts differ: {mismatch!r}"))
            return self._transfer(objects, load=True)

    def close(self) -> None:
        """Unregister while owners are alive, close the SDK, then its watchdog.

        Cleanup remains available after a failed SDK operation, but every
        cleanup error is reported. Failed unregistrations retain their owners
        until process teardown; closed transports can never be used again.
        """
        with self._lock:
            if self._closed:
                return
            errors = []
            failed_regions = []
            if self._watchdog is None:
                self._watchdog = _DeadlineWatchdog(self.timeout_seconds)
            def cleanup(operation: str, function: Callable[[], Any]) -> Any:
                return self._watchdog.run(operation, function)
            try:
                if self.backend is not None and not self.scheduler:
                    try:
                        cleanup("set_device for close", self.backend.set_device)
                    except Exception as exc:
                        errors.append(f"set_device: {exc}")
                for region in self._registration_attempts:
                    try:
                        code = cleanup("unregister_buffer", lambda region=region: self.store.unregister_buffer(region.address, region.length))
                        if isinstance(code, bool) or not isinstance(code, Integral) or int(code) != 0:
                            raise TransferError(f"unregister_buffer returned {code!r}")
                    except Exception as exc:
                        errors.append(f"buffer@{region.address:#x}: {exc}")
                        failed_regions.append(region)
                self._regions = list(failed_regions)
                self._registration_attempts = list(failed_regions)
                if not failed_regions:
                    self._owners.clear()
                try:
                    # Official store.close() has no status return value.
                    cleanup("store.close", self.store.close)
                except Exception as exc:
                    errors.append(f"store.close: {exc}")
            finally:
                self._closed = True
                self._watchdog.close()
                self._watchdog = None
            if errors:
                raise TransferError("memcache cleanup failed: " + "; ".join(errors))
