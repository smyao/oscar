"""Host protocol experiment ONLY; deliberately not imported by serving.

Archive #26/#37-49/#127/#128: exact prefix checkpoints must outlive request
leases, retain physical-page generations, and agree with native GDN/MTP hits.
D.4: bounded exact-window copy replaces neither INT2 attention nor its math;
never dequantize full history (6499.8–6655.1 ms) or perform per-token host KV
work in production. This scalar byte model is a correctness/byte-ledger oracle,
not a kernel, stream-order implementation, NPU test or speedup claim.
"""
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from oscar_ascend.request_window_storage import RequestWindowLayout
from oscar_ascend.request_windows import WindowLease


@dataclass(frozen=True)
class PrefixCheckpointKey:
    scope: bytes  # frozen model + layer + rotation + representation identity
    block_hash: bytes  # original native prefix-chain hash, not request ID
    physical_page: int
    allocation_epoch: int
    logical_end: int

    def __post_init__(self):
        if not self.scope or not self.block_hash:
            raise ValueError("checkpoint requires native hash and representation scope")
        for name, lower in (("physical_page", 1), ("allocation_epoch", 1), ("logical_end", 1)):
            value = getattr(self, name)
            if type(value) is not int or value < lower:
                raise ValueError(f"invalid checkpoint {name}")


@dataclass(frozen=True)
class ExactPrefixCheckpoint:
    key: PrefixCheckpointKey
    positions: tuple[int, ...]
    keys: tuple[bytes, ...]
    values: tuple[bytes, ...]

    @classmethod
    def capture(cls, layout: RequestWindowLayout, key: PrefixCheckpointKey,
                raw_rows: Mapping[int, tuple[bytes, bytes]], *,
                committed_end: int, physical_block_tokens: int):
        if (type(physical_block_tokens) is not int or physical_block_tokens <= 0
                or key.logical_end % physical_block_tokens
                or key.logical_end > committed_end):
            raise ValueError("only complete committed physical prefix pages may publish")
        positions = tuple(p for span in layout.retained_ranges(key.logical_end) for p in span)
        # Exact BF16 bytes must already exist. A missing row is a failure;
        # compression/dequantization is intentionally absent from this API.
        rows = tuple(raw_rows[p] for p in positions)
        if any(len(row) != 2 or any(type(v) is not bytes or len(v) != layout.kv_row_bytes
                                   for v in row) for row in rows):
            raise ValueError("checkpoint row must contain original BF16 K and V bytes")
        return cls(key, positions, tuple(row[0] for row in rows), tuple(row[1] for row in rows))

    @property
    def payload_bytes(self):
        return sum(map(len, self.keys)) + sum(map(len, self.values)) + 8 * len(self.positions)


class ExactPrefixDirectory:
    """Bounded host model of page-owned records and in-flight read pins.

    Future production must attach these transitions to actual native page
    allocation/free/eviction/reset and device completion, not host return time.
    Retiring a hash while a reader is pinned cannot reuse that record's bytes.
    """
    def __init__(self, capacity: int):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("checkpoint capacity must be positive")
        self.capacity = capacity
        self.records = {}
        self.pins = {}

    def publish(self, checkpoint: ExactPrefixCheckpoint):
        key = checkpoint.key
        existing = self.records.get(key)
        if existing is not None:
            if existing != checkpoint:
                raise ValueError("published prefix checkpoint is immutable")
            return
        if any(k.physical_page == key.physical_page and k.scope == key.scope for k in self.records):
            raise ValueError("physical page must retire its old allocation before reuse")
        if len(self.records) >= self.capacity:
            raise ValueError("bounded checkpoint directory is full")
        self.records[key], self.pins[key] = checkpoint, 0

    def acquire(self, key: PrefixCheckpointKey):
        checkpoint = self.records[key]  # no hash/page/epoch substitution
        self.pins[key] += 1
        return checkpoint

    def release(self, key: PrefixCheckpointKey):
        if self.pins.get(key, 0) <= 0:
            raise ValueError("checkpoint release without matching acquisition")
        self.pins[key] -= 1

    def evict(self, key: PrefixCheckpointKey):
        if self.pins.get(key, 0):
            raise ValueError("checkpoint still owned by an in-flight prefix restore")
        self.records.pop(key)
        self.pins.pop(key)

    def reset(self):
        if any(self.pins.values()):
            raise ValueError("reset must drain/cancel prefix restores before byte reuse")
        self.records.clear()
        self.pins.clear()


def restore_exact_rows(checkpoint: ExactPrefixCheckpoint, layout: RequestWindowLayout,
                       lease: WindowLease):
    """Host transfer plan: destination row -> (BF16 key, BF16 value, new tag).

    The production equivalent must first validate every source generation/tag,
    then copy on the protected input-prep stream, then publish destination tags.
    No old lease epoch is copied into a newly admitted request.
    """
    layout.address(lease.slot, 0)
    return {layout.row_for_position(p): (k, v, layout.encode_tag(lease.epoch, p))
            for p, k, v in zip(checkpoint.positions, checkpoint.keys, checkpoint.values)}


def select_joint_checkpoint_hit(maximum: int, native_hit_at_most: Callable[[int], int],
                                available_ends_by_full_layer: Sequence[set[int]]) -> int:
    """Joint fixed point; FULL checkpoints are NOT downward-closed.

    The callback contract is a native FULL+GDN hit *after* MTP lookahead drop,
    bounded by its argument. It cannot be a naive raw coordinator invocation:
    an adapter must preserve the permitted MTP peek margin and return matching
    native computed block tables too. That adapter is NOT implemented here.
    """
    if type(maximum) is not int or maximum < 0 or not available_ends_by_full_layer:
        raise ValueError("invalid prefix bound or absent FULL checkpoint coverage")
    available = set.intersection(*(set(ends) for ends in available_ends_by_full_layer))
    if any(type(end) is not int or end <= 0 for end in available):
        raise ValueError("checkpoint ends must be positive positions")
    bound = maximum
    while bound:
        native = native_hit_at_most(bound)
        if type(native) is not int or not 0 <= native <= bound:
            raise ValueError("native hit must not grow beyond the requested bound")
        eligible = max((end for end in available if end <= native), default=0)
        if eligible == native:
            return eligible
        # Strict decrease guarantees termination. The native GDN hit at this
        # earlier boundary must be checked again, not inferred from a later one.
        bound = eligible
    return 0


def dense_checkpoint_ledger(*, page_bytes: int, native_block_tokens: int,
                            compressed_block_tokens: int,
                            layout: RequestWindowLayout) -> dict:
    for value in (page_bytes, native_block_tokens, compressed_block_tokens):
        if type(value) is not int or value <= 0:
            raise ValueError("page ledger geometry must be positive")
    checkpoint_bytes = layout.rows * (2 * layout.kv_row_bytes + 8)
    return {"page_bytes": page_bytes, "checkpoint_bytes_per_page": checkpoint_bytes,
            "compressed_tokens_per_page": compressed_block_tokens,
            "nominal_full_ratio_without_checkpoints": compressed_block_tokens / native_block_tokens,
            "full_ratio_dense_checkpoint_upper_bound":
                page_bytes * compressed_block_tokens / (native_block_tokens * (page_bytes + checkpoint_bytes)),
            "additional_costs_excluded": ["request sidecars", "checkpoint descriptors and pins",
                                          "GDN live/cached states", "MTP", "allocator fragmentation"],
            "evidence": "integer byte model; hypothetical align layout, not current DualRegion support"}
