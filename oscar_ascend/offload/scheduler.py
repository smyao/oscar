"""Request offload protocol around the unchanged native scheduler.

Archive #37-49/#171: capture physical ownership before native free; never
publish scheduled/speculative tokens as committed or retain freed HBM pages.
Archive #127/#128: preserve every native group and its physical block IDs.
Archive #157/#163: async MTP counts become authoritative only on the worker;
resume waits for that acknowledgement and all earlier native output frames.
Native evidence: scheduler._preempt_request, AsyncScheduler, and
NPUModelRunner.handle_preemptions before _update_states (pinned references).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import struct
from typing import Any


class OffloadLifecycleError(RuntimeError):
    """A transfer would restore an uncommitted or stale request state."""


def freeze_block_ids(blocks: Any) -> tuple[tuple[int, ...], ...]:
    if hasattr(blocks, "get_block_ids"):
        blocks = blocks.get_block_ids()
    result = tuple(tuple(int(b) for b in group) for group in blocks)
    if any(b < 0 for group in result for b in group):
        raise OffloadLifecycleError("physical offload block IDs must be nonnegative")
    return result


def token_digest(tokens) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        token = int(token)
        if token < 0:
            raise OffloadLifecycleError("unverified token placeholder in committed token history")
        digest.update(struct.pack("<Q", token))
    return digest.hexdigest()


@dataclass(frozen=True)
class RequestSnapshot:
    request_id: str
    generation: int
    version: int
    epoch: int
    block_ids: tuple[tuple[int, ...], ...]
    num_computed_hint: int
    prompt_tokens: int
    known_tokens: int
    known_prefix_digest: str
    mamba_mode: str

    @property
    def identity(self) -> tuple[str, int, int, int]:
        return self.request_id, self.generation, self.version, self.epoch


@dataclass(frozen=True)
class SnapshotAck:
    request_id: str
    generation: int
    version: int
    epoch: int
    committed_tokens: int
    rank: int
    payload_key: str

    @property
    def identity(self) -> tuple[str, int, int, int]:
        return self.request_id, self.generation, self.version, self.epoch


@dataclass(frozen=True)
class RestorePlan:
    snapshot: RequestSnapshot
    committed_tokens: int
    block_ids: tuple[tuple[int, ...], ...]
    payload_keys: tuple[str, ...]


@dataclass(frozen=True)
class BranchTransition:
    request_id: str
    old_generation: int
    new_generation: int
    epoch: int
    retained_tokens: int
    retained_prefix_digest: str
    retained_token_ids: tuple[int, ...]


@dataclass
class _RequestRecord:
    request: Any
    generation: int
    version: int = 0
    pending_steps: set[int] = field(default_factory=set)
    snapshot: RequestSnapshot | None = None
    acknowledgements: dict[int, SnapshotAck] = field(default_factory=dict)
    restore: RestorePlan | None = None
    branch_context: BranchTransition | None = None


class OffloadSchedulerState:
    """Metadata only: all admission, allocation, free and ordering stay native.

    A preemption can reuse its freed blocks in the same scheduler step. The
    worker stages the snapshot before native zeroing; this object does not
    pin those pages. Device-complete acknowledgements are generation fenced.
    Async output frames may arrive after preemption; they are still consumed
    by the original scheduler before this object makes the snapshot usable.
    """

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.epoch = 0
        self._serial = 0
        self._next_generation = 0
        self._requests: dict[str, _RequestRecord] = {}
        self._steps: dict[int, tuple[Any, int, dict[str, int]]] = {}
        self._preemptions: list[RequestSnapshot] = []
        self._branches: list[BranchTransition] = []
        # Connector-owned payload lifetime is longer than this request's
        # usable snapshot: cancelled saves and dispatched loads still finish.
        self.on_snapshot_released = None
        self.on_request_retired = None
        self.suppress_preemption = False
        parallel = scheduler.parallel_config
        self.world_size = (int(parallel.tensor_parallel_size)
                           * int(getattr(parallel, "pipeline_parallel_size", 1)))
        if self.world_size < 1:
            raise OffloadLifecycleError("offload requires a positive worker world size")

    def observe_request(self, request) -> int:
        rid = request.request_id
        previous = self._requests.get(rid)
        if previous is not None and previous.request is request:
            return previous.generation
        if previous is not None:
            self.finish(rid)
        self._next_generation += 1
        self._requests[rid] = _RequestRecord(request, self._next_generation)
        return self._next_generation

    def begin_step(self, scheduler_output) -> int:
        """Call from build_connector_meta, before native count advancement."""
        key = id(scheduler_output)
        old = self._steps.get(key)
        if old is not None and old[0] is scheduler_output:
            return old[1]
        self._serial += 1
        generations = {}
        for rid in scheduler_output.num_scheduled_tokens:
            request = self.scheduler.requests[rid]
            generations[rid] = self.observe_request(request)
            self._requests[rid].pending_steps.add(self._serial)
        self._steps[key] = (scheduler_output, self._serial, generations)
        return self._serial

    def step_generations(self, scheduler_output) -> dict[str, int]:
        self.begin_step(scheduler_output)
        return dict(self._steps[id(scheduler_output)][2])

    def before_preempt(self, request) -> RequestSnapshot | None:
        if self.suppress_preemption:
            return None
        self.observe_request(request)
        record = self._requests[request.request_id]
        if record.branch_context is not None:
            transition = record.branch_context
            if token_digest(request.all_token_ids[:transition.retained_tokens]) != transition.retained_prefix_digest:
                raise OffloadLifecycleError("streaming branch changed its retained committed prefix")
        blocks = freeze_block_ids(self.scheduler.kv_cache_manager.get_blocks(request.request_id))
        if not any(blocks):
            return None  # Native has never materialized a cache for this request.
        record.version += 1
        tokens = request.all_token_ids
        mode = self.scheduler.cache_config.mamba_cache_mode
        if mode not in ("none", "align"):
            raise OffloadLifecycleError(f"unsupported native Mamba mode {mode!r}")
        snapshot = RequestSnapshot(
            request.request_id, record.generation, record.version, self.epoch,
            blocks, int(request.num_computed_tokens), int(request.num_prompt_tokens),
            len(tokens), token_digest(tokens), mode,
        )
        self._release_snapshot(record)
        record.snapshot = snapshot
        record.acknowledgements.clear()
        record.restore = None
        self._preemptions.append(snapshot)
        return snapshot

    def drain_preemptions(self) -> tuple[RequestSnapshot, ...]:
        result = tuple(self._preemptions)
        self._preemptions.clear()
        return result

    def drain_branches(self) -> tuple[BranchTransition, ...]:
        result = tuple(self._branches)
        self._branches.clear()
        return result

    def _release_snapshot(self, record) -> None:
        if record.snapshot is not None and self.on_snapshot_released is not None:
            self.on_snapshot_released(record.snapshot.identity)

    def acknowledge(self, ack: SnapshotAck) -> bool:
        """Return False for a stale cancelled/reset/replaced transfer."""
        record = self._requests.get(ack.request_id)
        if record is None or record.snapshot is None or record.snapshot.identity != ack.identity:
            return False
        if not 0 <= ack.rank < self.world_size or ack.committed_tokens < 0 or not ack.payload_key:
            raise OffloadLifecycleError("invalid worker snapshot acknowledgement")
        previous = record.acknowledgements.get(ack.rank)
        if previous is not None and previous != ack:
            raise OffloadLifecycleError("conflicting duplicate snapshot acknowledgement")
        if any(item.committed_tokens != ack.committed_tokens
               for item in record.acknowledgements.values()):
            raise OffloadLifecycleError("TP workers disagree on the committed snapshot position")
        record.acknowledgements[ack.rank] = ack
        return True

    def after_output(self, scheduler_output) -> None:
        """Call after native update_from_output has accepted/rejected tokens."""
        step = self._steps.pop(id(scheduler_output), None)
        if step is None or step[0] is not scheduler_output:
            return
        _, serial, generations = step
        for rid, generation in generations.items():
            record = self._requests.get(rid)
            if record is not None and record.generation == generation:
                record.pending_steps.discard(serial)

    def match_resume(self, request, local_tokens: int):
        """None delegates to prefix lookup; (None, False) is native WAIT."""
        record = self._requests.get(request.request_id)
        if record is None or record.request is not request or record.snapshot is None:
            return None
        snapshot = record.snapshot
        if record.pending_steps or len(record.acknowledgements) != self.world_size:
            return None, False
        tokens = request.all_token_ids
        if (len(tokens) < snapshot.known_tokens
                or token_digest(tokens[:snapshot.known_tokens]) != snapshot.known_prefix_digest):
            raise OffloadLifecycleError("request token history changed while its snapshot was in flight")
        committed = record.acknowledgements[0].committed_tokens
        if committed >= len(tokens):
            # Generative requests retain the final sampled token without KV.
            # Subtracting one here would require an earlier recurrent state,
            # which cannot be manufactured from the saved state at committed.
            raise OffloadLifecycleError("snapshot has no remaining token to replay for logits")
        if int(local_tokens) >= committed:
            # Native APC already owns a complete joint checkpoint at least as
            # far ahead. It is authoritative for this same token prefix, and
            # restoring private remote bytes would overwrite shared hit pages.
            # This also permits another request's longer cached prompt to
            # supersede a request snapshot taken during chunked prefill.
            self._release_snapshot(record)
            record.snapshot = None
            record.restore = None
            record.acknowledgements.clear()
            return 0, False
        return committed - int(local_tokens), False

    def allocated(self, request, blocks) -> RestorePlan | None:
        record = self._requests.get(request.request_id)
        if record is None or record.snapshot is None:
            return None
        match = self.match_resume(request, 0)
        if match is None or match[0] is None:
            raise OffloadLifecycleError("native allocated a snapshot before its completion barrier")
        plan = RestorePlan(record.snapshot, int(match[0]), freeze_block_ids(blocks),
                           tuple(record.acknowledgements[r].payload_key for r in range(self.world_size)))
        if len(plan.block_ids) != len(plan.snapshot.block_ids):
            raise OffloadLifecycleError("restored native KV group count changed")
        record.restore = plan
        return plan

    def restored(self, request_id: str, generation: int, version: int) -> None:
        record = self._requests.get(request_id)
        if (record is not None and record.snapshot is not None
                and (record.generation, record.version) == (generation, version)):
            record.snapshot = None
            record.restore = None
            record.acknowledgements.clear()

    def finish(self, request_id: str) -> None:
        record = self._requests.pop(request_id, None)
        if record is not None and self.on_request_retired is not None:
            self.on_request_retired(request_id, record.generation)
        self._preemptions[:] = [s for s in self._preemptions if s.request_id != request_id]
        self._branches[:] = [s for s in self._branches if s.request_id != request_id]

    def branch(self, request) -> int:
        old = self.observe_request(request)
        retained = int(request.num_computed_tokens)
        if not 0 <= retained <= len(request.all_token_ids):
            raise OffloadLifecycleError("streaming branch does not contain its retained committed tokens")
        tokens = tuple(int(t) for t in request.all_token_ids[:retained])
        digest = token_digest(tokens)
        # Preserve earlier transitions if multiple input chunks are handled
        # before the next worker dispatch; apply them in the same order.
        earlier = [item for item in self._branches if item.request_id == request.request_id]
        self.finish(request.request_id)
        generation = self.observe_request(request)
        transition = BranchTransition(request.request_id, old, generation, self.epoch,
                                      retained, digest, tokens)
        self._requests[request.request_id].branch_context = transition
        self._branches.extend((*earlier, transition))
        return generation

    def reset(self) -> None:
        if self.on_request_retired is not None:
            for rid, record in tuple(self._requests.items()):
                self.on_request_retired(rid, record.generation)
        self.epoch += 1
        self._requests.clear()
        self._steps.clear()
        self._preemptions.clear()
        self._branches.clear()
