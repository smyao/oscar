"""Synchronous complete-checkpoint boundary for OSCAR KV offload.

Archive #34/#36/#37--49/#127/#128/#157/#163/#171: fixed graph addresses,
complete physical/window/GDN state, no speculative state as committed state,
and no page reuse or checkpoint publication before successful completion.
Scheduling, canonicalization, and collective communication belong to the
external connector; this module never changes native scheduler/attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Callable, Iterable, Mapping

from .transport import MissingObjectsError, StrictMemcacheTransport, TransferError, TransferObject


@dataclass(frozen=True)
class CheckpointDescriptor:
    namespace: str
    generation: str
    keys_by_participant: Mapping[int, tuple[str, ...]]

    def __post_init__(self) -> None:
        if not self.namespace or not self.generation:
            raise ValueError("checkpoint namespace and immutable generation are required")
        participants: dict[int, tuple[str, ...]] = {}
        all_keys: list[str] = []
        for rank, values in sorted(self.keys_by_participant.items()):
            if type(rank) is not int or rank < 0:
                raise ValueError("checkpoint participant IDs must be nonnegative integers")
            keys = tuple(values)
            if not keys or any(not isinstance(key, str) or not key for key in keys):
                raise ValueError("each participant requires explicit nonempty object keys")
            participants[rank] = keys
            all_keys.extend(keys)
        if not participants or len(set(all_keys)) != len(all_keys):
            raise ValueError("a checkpoint requires participants with disjoint unique object keys")
        object.__setattr__(self, "keys_by_participant", MappingProxyType(participants))

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(key for keys in self.keys_by_participant.values() for key in keys)

    @property
    def identity(self) -> str:
        value = [self.namespace, self.generation, list(self.keys_by_participant.items())]
        return sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class CheckpointReceipt:
    checkpoint_identity: str
    participant: int
    keys: tuple[str, ...]
    nbytes: int


def require_joint_receipts(
    descriptor: CheckpointDescriptor,
    receipts: Iterable[CheckpointReceipt],
) -> tuple[CheckpointReceipt, ...]:
    """Validate a real all-participant collective result before publication.

    This does not perform communication.  Passing only the local successful
    receipt cannot make a TP checkpoint complete.
    """
    receipts = tuple(receipts)
    by_rank: dict[int, CheckpointReceipt] = {}
    for receipt in receipts:
        if not isinstance(receipt, CheckpointReceipt):
            raise TransferError("checkpoint collective contains an invalid receipt")
        if receipt.participant in by_rank:
            raise TransferError("checkpoint collective contains a duplicate participant")
        expected = descriptor.keys_by_participant.get(receipt.participant)
        if receipt.checkpoint_identity != descriptor.identity or receipt.keys != expected or receipt.nbytes <= 0:
            raise TransferError("checkpoint collective contains a failed, stale, or incomplete participant")
        by_rank[receipt.participant] = receipt
    if set(by_rank) != set(descriptor.keys_by_participant):
        raise TransferError("checkpoint collective has not completed every TP/group participant")
    return tuple(by_rank[rank] for rank in descriptor.keys_by_participant)


class CheckpointWorker:
    """Hold caller-owned pages until a synchronous transfer has finished.

    ``synchronize`` must wait for all producers/consumers of the registered
    NPU buffers.  Save is called before native ``_update_states`` can reuse
    preempted pages; load is called before graph replay.  Any exception must
    abort that step, including errors in optional callbacks.
    """

    def __init__(
        self,
        transport: StrictMemcacheTransport,
        *,
        participant: int,
        synchronize: Callable[[], None],
    ) -> None:
        if type(participant) is not int or participant < 0:
            raise ValueError("participant must be a nonnegative integer")
        if not callable(synchronize):
            raise ValueError("a real producer/consumer synchronization callback is required")
        self.transport = transport
        self.participant = participant
        self.synchronize = synchronize

    def _local_objects(
        self, descriptor: CheckpointDescriptor, objects: Iterable[TransferObject]
    ) -> tuple[TransferObject, ...]:
        objects = tuple(objects)
        expected = descriptor.keys_by_participant.get(self.participant)
        if expected is None or tuple(obj.key for obj in objects) != expected:
            raise TransferError("local objects do not cover the checkpoint participant exactly")
        return objects

    def save_checkpoint(
        self,
        descriptor: CheckpointDescriptor,
        objects: Iterable[TransferObject],
    ) -> CheckpointReceipt:
        objects = self._local_objects(descriptor, objects)
        self.synchronize()
        receipt = self.transport.put(objects)
        # This is a local receipt only.  It is not a prefix-cache publication.
        return CheckpointReceipt(descriptor.identity, self.participant, receipt.keys, receipt.nbytes)

    def verify_joint_checkpoint(
        self,
        descriptor: CheckpointDescriptor,
        receipts: Iterable[CheckpointReceipt],
    ) -> None:
        require_joint_receipts(descriptor, receipts)
        present = self.transport.exists(descriptor.keys)
        missing = tuple(key for key, exists in zip(descriptor.keys, present) if not exists)
        if missing:
            raise MissingObjectsError(missing)

    def load_checkpoint(
        self,
        descriptor: CheckpointDescriptor,
        objects: Iterable[TransferObject],
        *,
        verify_publication: Callable[[CheckpointDescriptor], None],
        after_load: Callable[[], None] | None = None,
    ) -> CheckpointReceipt:
        """Restore only a published complete generation into fixed buffers.

        The connector must verify its committed generation/manifest.  Mere
        local key existence cannot distinguish an in-progress global save.
        ``after_load`` is for complete-state validation/rebasing, and runs
        before this function can return success to the graph caller.
        """
        objects = self._local_objects(descriptor, objects)
        if not callable(verify_publication):
            raise ValueError("a complete-generation publication verifier is required")
        verify_publication(descriptor)
        present = self.transport.exists(descriptor.keys)
        missing = tuple(key for key, exists in zip(descriptor.keys, present) if not exists)
        if missing:
            raise MissingObjectsError(missing)
        self.synchronize()
        receipt = self.transport.get(objects)
        if after_load is not None:
            after_load()
        return CheckpointReceipt(descriptor.identity, self.participant, receipt.keys, receipt.nbytes)
