"""Pure-host ownership for prebudgeted exact request windows; no KV values.

Archive #26-49: identity survives batch reordering, and MTP resume creates a
new generation. #127-129: scheduler events and physical storage are separate.
#155: tags preserve exact ownership; host checks prove no device numerical,
asynchronous execution, graph or service acceptance.

Call RequestWindowLeases.update exactly once AFTER native _update_states has
completed condense/reorder. Keep its returned deferred callback unchanged.
Native evidence: scheduler.py:940-945,974,1943; gpu_model_runner.py:1135-1180,
1368-1379,1436-1447. A request absent from this batch still owns its lease.
This module allocates only Python metadata, never a data page or window tensor.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Any, Mapping, Sequence

MAX_EPOCH = (1 << 63) - 1
INVALID_SLOT = -1
INVALID_EPOCH = 0


class WindowLeaseError(ValueError):
    """Malformed ownership protocol or exhausted preallocated capacity."""


def _positive(name: str, value: int, *, maximum: int | None = None) -> int:
    if type(value) is not int or value <= 0 or (maximum is not None and value > maximum):
        raise WindowLeaseError(f"{name} must be a positive integer" +
                               (f" <= {maximum}" if maximum is not None else ""))
    return value


def _request_id(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise WindowLeaseError("request IDs must be nonempty strings")
    return value


def _event_ids(value: Any, name: str, *, optional: bool = False) -> set[str]:
    if value is None and optional:
        return set()
    if not isinstance(value, (set, frozenset)):
        raise WindowLeaseError(f"{name} must be a set of request IDs")
    return {_request_id(item) for item in value}


@dataclass(frozen=True)
class WindowLease:
    slot: int
    epoch: int


@dataclass(frozen=True)
class WindowBatch:
    """Final native batch order, followed by explicit invalid padding rows."""

    request_ids: tuple[str, ...]
    slots: tuple[int, ...]
    epochs: tuple[int, ...]

    @property
    def pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple(zip(self.slots, self.epochs))


class RequestWindowLeases:
    """Worker-local leases, with a transaction for each SchedulerOutput.

    Capacity must be the configured max_num_seqs, not an observed Running count
    or client concurrency. Epochs are globally monotonic within this tracker,
    so resuming onto a different free slot still obtains a strictly new epoch.
    If a device tag packs epoch into fewer bits, pass that smaller max_epoch.
    Never replace this tracker/reset its epoch counter while old tags survive.

    A failed update leaves every owner, free slot and epoch unchanged. The
    caller must stop on failure because native _update_states already ran.
    No stream ordering, GPU clears, graph mutation or tensor copies happen here.
    """

    def __init__(self, capacity: int, *, max_epoch: int = MAX_EPOCH):
        self._capacity = _positive("capacity", capacity)
        self._max_epoch = _positive("max_epoch", max_epoch, maximum=MAX_EPOCH)
        self._last_epoch = 0
        self._leases: dict[str, WindowLease] = {}
        self._free = list(range(capacity))

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, max_epoch: int = MAX_EPOCH):
        """Read this run's explicit server capacity; no hardcoded K32 default."""
        return cls(config["max_num_seqs"], max_epoch=max_epoch)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def active_count(self) -> int:
        return len(self._leases)

    @property
    def last_epoch(self) -> int:
        return self._last_epoch

    def snapshot(self) -> dict[str, WindowLease]:
        """An independent host diagnostic snapshot, not mutable tracker state."""
        return dict(self._leases)

    def update(self, scheduler_output: Any, batch_req_ids: Sequence[str], *,
               padded_rows: int | None = None) -> WindowBatch:
        """Consume native lifecycle events after the native batch is stable.

        finished/preempted IDs release owners. Resumed IDs invalidate an old
        owner even if its preemption notification was absent. New IDs that
        already have a lease are native streaming continuations unless a
        finish/preempt/resume event explicitly invalidates them.
        """
        if isinstance(batch_req_ids, (str, bytes)):
            raise WindowLeaseError("batch_req_ids must be a sequence of IDs")
        batch = tuple(_request_id(item) for item in batch_req_ids)
        if len(set(batch)) != len(batch):
            raise WindowLeaseError("native batch contains duplicate request IDs")
        if padded_rows is None:
            padded_rows = len(batch)
        if type(padded_rows) is not int or not len(batch) <= padded_rows <= self.capacity:
            raise WindowLeaseError("batch/padding rows exceed the configured lease capacity")
        finished = _event_ids(scheduler_output.finished_req_ids, "finished_req_ids")
        preempted = _event_ids(scheduler_output.preempted_req_ids, "preempted_req_ids", optional=True)
        resumed = _event_ids(scheduler_output.scheduled_cached_reqs.resumed_req_ids, "resumed_req_ids")
        new_list = [_request_id(item.req_id) for item in scheduler_output.scheduled_new_reqs]
        new = set(new_list)
        if len(new) != len(new_list):
            raise WindowLeaseError("scheduled_new_reqs contains duplicate request IDs")
        batch_set = set(batch)
        if not (new | resumed) <= batch_set:
            raise WindowLeaseError("new/resumed request is absent from the final native batch")
        if (finished & batch_set) - new:
            raise WindowLeaseError("finished request in batch lacks a new-request event")
        if (preempted & batch_set) - (new | resumed):
            raise WindowLeaseError("preempted request in batch lacks a new/resume event")

        # Transactional host metadata. Do not partly release leases before
        # discovering another row's missing identity or exhausted epoch.
        leases, free, epoch = dict(self._leases), list(self._free), self._last_epoch
        for request_id in finished | preempted | resumed:
            old = leases.pop(request_id, None)
            if old is not None:
                heapq.heappush(free, old.slot)
        for request_id in batch:
            if request_id in leases:
                continue
            if request_id not in new | resumed:
                raise WindowLeaseError(f"request {request_id!r} has no lease or admission event")
            if not free:
                raise WindowLeaseError("request-window lease capacity exhausted; unscheduled owners still retain their leases")
            if epoch == self._max_epoch:
                raise WindowLeaseError("request-window epoch exhausted; never wrap while old tags exist")
            epoch += 1
            leases[request_id] = WindowLease(heapq.heappop(free), epoch)
        padding = padded_rows - len(batch)
        result = WindowBatch(
            batch,
            tuple(leases[key].slot for key in batch) + (INVALID_SLOT,) * padding,
            tuple(leases[key].epoch for key in batch) + (INVALID_EPOCH,) * padding,
        )
        self._leases, self._free, self._last_epoch = leases, free, epoch
        return result


def note_preemption(scheduler, request_id: str) -> None:
    """Record only after the native preemption returned successfully."""
    pending = getattr(scheduler, "_oscar_window_preemptions", None)
    if pending is None:
        pending = scheduler._oscar_window_preemptions = set()
    pending.add(_request_id(request_id))


def publish_preemptions(scheduler, result):
    """Merge into the existing transport field after a successful schedule."""
    pending = getattr(scheduler, "_oscar_window_preemptions", None)
    if pending:
        existing = _event_ids(result.preempted_req_ids, "preempted_req_ids", optional=True)
        result.preempted_req_ids = existing | pending
        pending.clear()
    return result


class PreemptionNotificationMixin:
    """Cooperative native Scheduler mixin; no new transport fields.

    Place BEFORE the actual native Scheduler/AsyncScheduler class in the MRO.
    Its pinned signatures are _preempt_request(request, timestamp) and
    schedule(). Only successful native preemptions are queued. Native forced
    reset calls self._preempt_request, so it is covered without replacing
    reset_prefix_cache, GDN state management, or the scheduler's math.
    Native scheduling must remain serialized; this is not a concurrent queue.
    """

    def _preempt_request(self, request, timestamp: float) -> None:
        request_id = _request_id(request.request_id)
        result = super()._preempt_request(request, timestamp)
        note_preemption(self, request_id)
        return result

    def schedule(self):
        result = super().schedule()
        return publish_preemptions(self, result)
