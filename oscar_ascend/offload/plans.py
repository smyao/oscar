"""Serializable physical-page plans for the external AscendStore connector.

Archive #37-49/#127/#128/#157/#163/#171: addresses are worker-local; keys
identify a committed logical state across every physical group and TP shard.
No proportional truncation of windows or recurrent state is permitted.
Archive #182/#183: prefix identity queries keep native hashes and physical
alignment, without constructing unused per-layer page-hash vectors.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from math import ceil


@dataclass(frozen=True)
class PayloadPlan:
    request_id: str
    generation: int
    version: int
    epoch: int
    committed_tokens: int
    block_ids: tuple[tuple[int, ...], ...]
    base_key: str
    # Optional content-addressed FULL page keys for immutable prefix pages.
    page_hashes: tuple[tuple[str, tuple[str, ...]], ...] = ()
    local_tokens: int = 0


def object_key(namespace: str, rank: int, plan: PayloadPlan, layer_index: int,
               family: str, logical_page: int = 0, component: int = 0,
               layer_name: str = "") -> str:
    pages = dict(plan.page_hashes)
    if family == "oscar_full" and layer_name in pages:
        tail = f"prefix/page/{pages[layer_name][logical_page]}/l{layer_index}"
    elif family == "oscar_full":
        tail = f"{plan.base_key}/l{layer_index}/p{logical_page}"
    else:
        tail = f"{plan.base_key}/l{layer_index}/s{component}"
    # Memcache's public API limits keys to fewer than 256 bytes. Keep every
    # identity dimension in the digest instead of truncating request IDs.
    identity = f"{namespace}:tp{rank}:{tail}".encode()
    return "oscar-kv-v1:" + hashlib.sha256(identity).hexdigest()


def expected_keys(namespace: str, world_size: int, plan: PayloadPlan, geometry) -> tuple[str, ...]:
    result = []
    for rank in range(world_size):
        for index, layer in enumerate(geometry):
            family = layer["cache_family"]
            if family == "oscar_full":
                for page in range(ceil(plan.committed_tokens / layer["storage_block_size"])):
                    result.append(object_key(namespace, rank, plan, index, family, page,
                                             layer_name=layer["layer_name"]))
            elif family == "gdn":
                for component in range(len(layer["components"])):
                    result.append(object_key(namespace, rank, plan, index, family,
                                             component=component))
            else:
                raise ValueError(f"unrecognized offload cache family {family!r}")
    return tuple(result)


def snapshot_key(session: str, snapshot, committed: int) -> str:
    request = hashlib.sha256(snapshot.request_id.encode()).hexdigest()
    return (f"swap/{session}/{request}/g{snapshot.generation}/v{snapshot.version}"
            f"/e{snapshot.epoch}/n{committed}")


def hash_at(request, tokens: int, hash_block_size: int) -> str:
    """Use native salted/LoRA/multimodal hashes, never a bare token-only key."""
    if tokens <= 0 or tokens % hash_block_size:
        raise ValueError("prefix checkpoint must end at a native hash boundary")
    index = tokens // hash_block_size - 1
    if index >= len(request.block_hashes):
        raise ValueError("prefix checkpoint has not acquired its native block hash")
    value = request.block_hashes[index]
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, int):
        return hex(value)[2:]
    raise ValueError("native block hash encoding changed")


def prefix_identity(request, tokens: int, geometry, hash_block_size: int) -> str:
    """Return the existing stable index identity without materializing pages."""
    digest = hash_at(request, tokens, hash_block_size)
    for layer in geometry:
        if layer["cache_family"] != "oscar_full":
            continue
        if tokens % layer["storage_block_size"]:
            raise ValueError("prefix publication requires a complete physical FULL page")
    return f"prefix/state/{digest}"


def prefix_plan(request, generation: int, epoch: int, tokens: int, blocks,
                geometry, hash_block_size: int, *, local_tokens: int = 0,
                include_page_hashes: bool = True) -> PayloadPlan:
    from .scheduler import freeze_block_ids
    geometry = tuple(geometry)
    identity = prefix_identity(request, tokens, geometry, hash_block_size)
    page_hashes = []
    if include_page_hashes:
        for layer in geometry:
            if layer["cache_family"] == "oscar_full":
                block = layer["storage_block_size"]
                page_hashes.append((layer["layer_name"], tuple(
                    hash_at(request, end, hash_block_size) for end in range(block, tokens + 1, block))))
    return PayloadPlan(request.request_id, generation, 0, epoch, tokens,
                       freeze_block_ids(blocks), identity,
                       tuple(page_hashes), local_tokens)
