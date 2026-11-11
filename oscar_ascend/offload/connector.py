"""AscendStore/memcache adapter for complete OSCAR hybrid checkpoints.

Archive #27/#34/#36/#37-49/#127/#128/#157/#163/#171: external hooks retain
native scheduling, committed MTP/GDN semantics and captured pool addresses.
The native AscendStore SDK bootstrap is reused. Its lossy exception wrappers,
proportional partial-page slicing and align-only scheduler are not reused.
Saves retain synchronous ownership barriers. Optional background reads use
native WAITING_FOR_REMOTE_KVS and all-TP receive completion before admission.
Archive #182/#183: batch cold prefix-index probes while retaining descending
complete-checkpoint selection, explicit errors and native salted identities.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import os
import json
from pathlib import Path
import uuid

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorRole, KVConnectorMetadata,
    KVConnectorWorkerMetadata, KVConnectorHandshakeMetadata, SupportsHMA,
)

from .config import validate_transfer_config, allocate_aligned_bytes
from .layout import export_kv_caches, registration_regions, StorageRegion
from .identity import build_namespace_identity
from .plans import PayloadPlan, object_key, expected_keys, snapshot_key, prefix_plan, prefix_identity
from .scheduler import (OffloadSchedulerState, RequestSnapshot, SnapshotAck,
                        OffloadLifecycleError, freeze_block_ids, token_digest)
from .state import WorkerCheckpointTracker, reset_restored_cursor
from .transport import ByteSegment, TransferObject, StrictMemcacheTransport, TransferError
from .async_load import AsyncLoadQueue


@dataclass
class OscarOffloadMetadata(KVConnectorMetadata):
    session: str
    step_id: int
    generations: dict[str, int]
    preemptions: tuple = ()
    restores: tuple[PayloadPlan, ...] = ()
    prefixes: tuple[PayloadPlan, ...] = ()
    branches: tuple = ()
    prefix_view: str = "shared"
    retired_loads: tuple = ()


@dataclass
class OscarOffloadWorkerMetadata(KVConnectorWorkerMetadata):
    acknowledgements: tuple[SnapshotAck, ...] = ()
    prefixes: tuple = ()
    loads: tuple = ()

    def aggregate(self, other):
        if not isinstance(other, OscarOffloadWorkerMetadata):
            raise OffloadLifecycleError("incompatible offload worker metadata")
        return OscarOffloadWorkerMetadata(self.acknowledgements + other.acknowledgements,
                                         self.prefixes + other.prefixes,
                                         self.loads + other.loads)


@dataclass(frozen=True)
class PrefixReceipt:
    index_key: str
    plan: PayloadPlan
    rank: int
    prefix_view: str = "shared"


@dataclass(frozen=True)
class LoadReceipt:
    """Every TP reader has finished its synchronous SDK copy before receipt."""
    request_id: str
    generation: int
    version: int
    epoch: int
    committed_tokens: int
    rank: int
    payload_key: str

    @property
    def identity(self):
        return self.request_id, self.generation, self.version, self.epoch


@dataclass
class _SwapPayload:
    snapshot: RequestSnapshot
    saves: dict = field(default_factory=dict)
    loads: dict = field(default_factory=dict)
    plan: PayloadPlan | None = None
    discarded: bool = False
    restore_queued: bool = False
    restore_dispatched: bool = False


@dataclass
class OscarOffloadHandshake(KVConnectorHandshakeMetadata):
    namespace: str
    geometry: tuple[dict, ...]
    rank: int
    world_size: int


class OscarAscendStoreConnector(KVConnectorBase_V1, SupportsHMA):
    _oscar_offload = True

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config):
        return False

    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        validate_transfer_config(vllm_config.kv_transfer_config)
        self.config = vllm_config
        self.cache_config = kv_cache_config
        parallel = vllm_config.parallel_config
        # The current OSCAR target is local TP4, not a PP/CP cache layout.
        if any(getattr(parallel, name, 1) != 1 for name in (
                "pipeline_parallel_size", "data_parallel_size",
                "prefill_context_parallel_size", "decode_context_parallel_size")):
            raise OffloadLifecycleError("OSCAR offload requires the configured local TP topology")
        self.world_size = int(parallel.tensor_parallel_size)
        self.rank = int(getattr(parallel, "rank", 0)) % self.world_size
        self.session = uuid.uuid4().hex
        self.namespace = None
        self.geometry = ()
        self.exports = {}
        self.runner = None
        self.scheduler_state = None
        self._restores = []
        self._prefix_matches = {}
        self._acks = []
        self._prefix_acks = []
        self._prefix_publications = {}
        self._prefix_view = "shared"
        self._swap_payloads = {}
        self._load_acks = []
        self._transfer_session = None
        self._prefix_producer_event = None
        self.tracker = WorkerCheckpointTracker()
        self._group_for_layer = {}
        if kv_cache_config is None:
            raise OffloadLifecycleError("offload requires the native physical cache config")
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            for name in group.layer_names:
                if name in self._group_for_layer:
                    raise OffloadLifecycleError("cache layer appears in multiple physical groups")
                self._group_for_layer[name] = gid
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        self._init_load_state(extra["load_async"])
        self.transport = StrictMemcacheTransport.from_vllm_config(
            vllm_config, scheduler=role == KVConnectorRole.SCHEDULER,
            max_batch_objects=extra.get("max_batch_objects", 256),
            timeout_seconds=extra.get("timeout_seconds", 60))
        from .metadata import MetadataClient, MetadataServer, rpc_endpoint
        self.metadata_server = None
        self.metadata_client = None
        endpoint = rpc_endpoint(vllm_config)
        if role == KVConnectorRole.SCHEDULER:
            self.metadata_client = MetadataClient(endpoint, timeout_seconds=extra.get("timeout_seconds", 60))
        elif self.rank == 0:
            self.metadata_server = MetadataServer(self.transport, endpoint,
                timeout_seconds=extra.get("timeout_seconds", 60))
            self.metadata_server.start()

    def _init_load_state(self, load_async):
        self.load_async = load_async
        self._load_queue = None
        self._async_loads = {}
        self._loading_request_ids = set()
        self._cancelled_loads = set()
        self._restored_cursors = {}
        self._scheduled_loads = {}
        self._retired_loads = set()

    def _setup_load_device(self):
        # Same thread-local device selection as the native receiving thread.
        self.transport.backend.set_device()

    def bind_scheduler(self, scheduler):
        if self.scheduler_state is not None:
            if self.scheduler_state.scheduler is not scheduler:
                raise OffloadLifecycleError("offload connector rebound to a different scheduler")
            return
        self.scheduler_state = OffloadSchedulerState(scheduler)
        self.scheduler_state.on_snapshot_released = self._discard_swap
        self.scheduler_state.on_request_retired = self._retire_request

    def _discard_swap(self, identity):
        record = self._swap_payloads.get(identity)
        if record is not None:
            record.discarded = True
            self._collect_swap(identity)

    def _retire_request(self, request_id, generation):
        # A queued restore can be cancelled before dispatch. Once metadata
        # leaves the scheduler, every rank must report completion even if the
        # request is cancelled: another rank may still be checking all keys.
        self._retired_loads.update(key for key in self._scheduled_loads
                                   if key[:2] == (request_id, generation))
        # Native already holds async destination pages until finished_recving,
        # including cancellation before this metadata frame is dispatched.
        # Keep these loads queued so every rank emits that final receipt.
        if not self.load_async:
            self._restores[:] = [plan for plan in self._restores
                                if (plan.request_id, plan.generation) != (request_id, generation)]
        for identity, record in tuple(self._swap_payloads.items()):
            if identity[:2] == (request_id, generation):
                record.discarded = True
                if not self.load_async:
                    record.restore_queued = False
                self._collect_swap(identity)

    def _collect_swap(self, identity):
        record = self._swap_payloads.get(identity)
        if record is None or len(record.saves) != self.world_size:
            return
        loaded = record.restore_dispatched and len(record.loads) == self.world_size
        unused = record.discarded and not record.restore_queued and not record.restore_dispatched
        if not (loaded or unused):
            return
        if record.plan.committed_tokens:
            self.metadata_client.remove_swap(record.plan)
        del self._swap_payloads[identity]

    def _record_save(self, ack):
        record = self._swap_payloads.get(ack.identity)
        if record is None:
            # Known dispatched snapshots stay in the ledger through late
            # cancel/reset ACKs. Only an already-collected duplicate is absent.
            return
        if (not 0 <= ack.rank < self.world_size or ack.committed_tokens < 0
                or ack.payload_key != snapshot_key(self.session, record.snapshot, ack.committed_tokens)):
            raise OffloadLifecycleError("invalid private swap completion identity")
        previous = record.saves.get(ack.rank)
        if previous is not None and previous != ack:
            raise OffloadLifecycleError("conflicting private swap completion receipt")
        plan = PayloadPlan(ack.request_id, ack.generation, ack.version, ack.epoch,
                           ack.committed_tokens, (), ack.payload_key)
        if record.plan is not None and record.plan != plan:
            raise OffloadLifecycleError("TP private swap completions disagree")
        record.plan = plan
        record.saves[ack.rank] = ack
        if not self.scheduler_state.acknowledge(ack) and not (record.restore_queued or record.restore_dispatched):
            record.discarded = True
        self._collect_swap(ack.identity)

    def _record_load(self, receipt):
        record = self._swap_payloads.get(receipt.identity)
        if record is None:
            return  # Idempotent duplicate after all readers completed.
        plan = record.plan
        if (not record.restore_dispatched or plan is None
                or not 0 <= receipt.rank < self.world_size
                or (receipt.committed_tokens, receipt.payload_key) != (plan.committed_tokens, plan.base_key)):
            raise OffloadLifecycleError("invalid private swap load receipt")
        previous = record.loads.get(receipt.rank)
        if previous is not None and previous != receipt:
            raise OffloadLifecycleError("conflicting private swap load receipt")
        record.loads[receipt.rank] = receipt
        self._collect_swap(receipt.identity)

    def get_handshake_metadata(self):
        if self.namespace is None:
            raise OffloadLifecycleError("offload handshake preceded physical cache registration")
        return OscarOffloadHandshake(self.namespace, self.geometry, self.rank, self.world_size)

    def set_xfer_handshake_metadata_pp_aware(self, metadata):
        values = tuple(metadata.values())
        if (len(values) != self.world_size
                or any(not isinstance(v, OscarOffloadHandshake) for v in values)
                or {v.rank for v in values} != set(range(self.world_size))):
            raise OffloadLifecycleError("offload requires every TP worker's layout handshake")
        first = values[0]
        if any((v.namespace, v.geometry, v.world_size) !=
               (first.namespace, first.geometry, self.world_size) for v in values):
            raise OffloadLifecycleError("TP workers disagree on PT/math/physical cache identity")
        self.namespace, self.geometry = first.namespace, first.geometry

    def register_kv_caches(self, kv_caches):
        from oscar_ascend.integration.runtime_api import require_runtime
        import torch
        provider = require_runtime()
        self.runner = provider._offload_runner
        specs = self.runner._get_layer_kv_cache_specs(self.cache_config)
        self.exports = export_kv_caches(kv_caches, provider.layers, specs)
        identity = build_namespace_identity(
            provider.config, self.exports, tp_rank=self.rank, tp_size=self.world_size,
            loaded_rotation_identity=provider._offload_rotation_identity)
        self.namespace = identity.digest
        self.geometry = tuple(self.exports[name].geometry() for name in sorted(self.exports))
        self.transport.register_buffers(registration_regions(self.exports),
            owners=tuple(t for export in self.exports.values() for t in export.connector_cache))
        # One fixed zero tail completes a canonical conv slice; raw states and
        # model tensors never change during save. Size is independent of history.
        zero_size = max(component.page_bytes for export in self.exports.values()
                        for component in export.components)
        self._zero_tail = allocate_aligned_bytes(torch, zero_size, self.runner.device)
        self.transport.register_buffers(
            (StorageRegion(self._zero_tail.data_ptr(), self._zero_tail.numel()),),
            owners=(self._zero_tail,))
        self._synchronize()
        print(f"[oscar] KV_OFFLOAD_READY backend=memcache namespace={self.namespace} "
              f"rank={self.rank} groups={len(self.cache_config.kv_cache_groups)} "
              "layout=packed+windows+tags+gdn graph_addresses=fixed "
              f"load_async={str(self.load_async).lower()}", flush=True)
        groups = []
        for gid, group in enumerate(self.cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            export = self.exports[group.layer_names[0]]
            groups.append({"group_id": gid, "family": export.cache_family,
                "block_size": export.storage_block_size, "layer_count": len(group.layer_names),
                "num_speculative_blocks": int(getattr(spec, "num_speculative_blocks", 0))})
        print("[oscar] KV_OFFLOAD_GEOMETRY " + json.dumps({"rank": self.rank,
            "num_blocks": self.cache_config.num_blocks, "groups": groups,
            "mamba_cache_mode": self.runner.cache_config.mamba_cache_mode}, separators=(",", ":")), flush=True)

    def _synchronize(self):
        import torch
        torch.npu.current_stream(self.runner.device).synchronize()

    def _require_scheduler(self):
        if self.scheduler_state is None or self.namespace is None:
            raise OffloadLifecycleError("offload scheduler was not bound/handshaken")
        return self.scheduler_state.scheduler

    def _prefix_policy(self):
        scheduler = self._require_scheduler()
        return getattr(scheduler, "_oscar_prefix_checkpoint_policy", None)

    def _hash_block_size(self):
        scheduler = self._require_scheduler()
        return int(scheduler.kv_cache_manager.coordinator.hash_block_size)

    def _present_prefix_indices(self, index_keys):
        # Per-call only: a later publication must be visible after a cold miss.
        unique = tuple(dict.fromkeys(index_keys))
        present = set()
        for start in range(0, len(unique), 256):
            batch = unique[start:start + 256]
            for key, exists in zip(batch, self.metadata_client.exists(batch), strict=True):
                if exists:
                    present.add(key)
        return present

    def _read_complete_prefix(self, index_key, plan):
        manifest = self.metadata_client.get(index_key)
        if manifest is None:  # It may be evicted after the batch probe.
            return None
        if (manifest.get("namespace") != self.namespace
                or manifest.get("prefix_view") != self._prefix_view
                or manifest.get("index_key") != index_key
                or manifest.get("committed_tokens") != plan.committed_tokens
                or not isinstance(manifest.get("base_key"), str)
                or not manifest["base_key"].startswith("prefix/data/")):
            raise TransferError("published prefix manifest has an incompatible identity")
        plan = replace(plan, base_key=manifest["base_key"], page_hashes=())
        keys = expected_keys(self.namespace, self.world_size, plan, self.geometry)
        if manifest.get("keys") != list(keys):
            raise TransferError("published prefix manifest has incomplete/mixed payload keys")
        return plan if all(self.transport.exists(keys)) else None

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        self._require_scheduler()
        resume = self.scheduler_state.match_resume(request, num_computed_tokens)
        if resume is not None:
            count, _ = resume
            return count, self.load_async and count is not None and count > 0
        self._prefix_matches.pop(request.request_id, None)
        policy = self._prefix_policy()
        if policy is None:
            return 0, False
        limit = policy.terminal_position(request.num_prompt_tokens)
        generation = self.scheduler_state.observe_request(request)
        # All FULL pages, all GDN components, all TP ranks must exist for the
        # SAME checkpoint. A missing object is a miss; SDK failures propagate.
        counts = range(limit, num_computed_tokens, -policy.alignment_tokens)
        if not counts:
            return 0, False
        hash_block_size = self._hash_block_size()
        # Preserve the single-RPC warm path. host_get already probes SDK
        # existence before GET, so a cold first candidate still performs no
        # missing-object SDK GET. Only shorter candidates need batch probing.
        plan = prefix_plan(request, generation, self.scheduler_state.epoch,
            counts[0], (), self.geometry, hash_block_size,
            local_tokens=num_computed_tokens, include_page_hashes=False)
        plan = self._read_complete_prefix(self._prefix_index_key(plan.base_key), plan)
        if plan is not None:
            self._prefix_matches[request.request_id] = plan
            print(f"[oscar] KV_OFFLOAD_PREFIX_HIT request={request.request_id} tokens={plan.committed_tokens}", flush=True)
            return plan.committed_tokens - num_computed_tokens, self.load_async
        counts = counts[1:]
        for start in range(0, len(counts), 256):
            candidates = tuple((count, self._prefix_index_key(prefix_identity(
                request, count, self.geometry, hash_block_size)))
                for count in counts[start:start + 256])
            present = self._present_prefix_indices(key for _, key in candidates)
            for count, index_key in candidates:
                if index_key not in present:
                    continue
                plan = prefix_plan(request, generation, self.scheduler_state.epoch,
                    count, (), self.geometry, hash_block_size,
                    local_tokens=num_computed_tokens, include_page_hashes=False)
                plan = self._read_complete_prefix(index_key, plan)
                if plan is not None:
                    self._prefix_matches[request.request_id] = plan
                    print(f"[oscar] KV_OFFLOAD_PREFIX_HIT request={request.request_id} tokens={count}", flush=True)
                    return count - num_computed_tokens, self.load_async
        return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if not num_external_tokens:
            return
        restore = self.scheduler_state.allocated(request, blocks)
        if restore is not None:
            if len(set(restore.payload_keys)) != 1:
                raise OffloadLifecycleError("TP snapshot payload identities disagree")
            snap = restore.snapshot
            self._restores.append(PayloadPlan(
                request.request_id, snap.generation, snap.version, snap.epoch,
                restore.committed_tokens, restore.block_ids, restore.payload_keys[0],
                local_tokens=restore.committed_tokens - num_external_tokens))
            self._swap_payloads[snap.identity].restore_queued = True
            self.scheduler_state.restored(request.request_id, snap.generation, snap.version)
        else:
            plan = self._prefix_matches.pop(request.request_id)
            self._restores.append(replace(plan, block_ids=freeze_block_ids(blocks)))
        if self.load_async:
            plan = self._restores[-1]
            key = (plan.request_id, plan.generation, plan.version, plan.epoch)
            self._scheduled_loads[key] = plan

    def build_connector_meta(self, scheduler_output):
        scheduler = self._require_scheduler()
        state = self.scheduler_state
        step_id = state.begin_step(scheduler_output)
        generations = state.step_generations(scheduler_output)
        if os.environ.get("OSCAR_OFFLOAD_SERVICE_PROBE") == "1":
            null_id = scheduler.kv_cache_manager.block_pool.null_block.block_id
            for rid, count in scheduler_output.num_scheduled_tokens.items():
                blocks = freeze_block_ids(scheduler.kv_cache_manager.get_blocks(rid))
                base = int(scheduler.requests[rid].num_computed_tokens)
                live = {page for group in blocks for page in group if page != null_id}
                print("[oscar] KV_OFFLOAD_POOL_STATE " + json.dumps({"request": rid,
                    "base": base, "scheduled": int(count), "end": base + int(count),
                    "groups": blocks, "live_pages": len(live), "null_id": null_id},
                    separators=(",", ":")), flush=True)
        prefixes = []
        policy = self._prefix_policy()
        if policy is not None:
            for rid, count in scheduler_output.num_scheduled_tokens.items():
                request = scheduler.requests[rid]
                end = int(request.num_computed_tokens) + int(count)
                if (not scheduler_output.scheduled_spec_decode_tokens.get(rid)
                        and end > 0 and end == policy.terminal_position(request.num_prompt_tokens)):
                    prefixes.append(prefix_plan(
                        request, generations[rid], state.epoch, end,
                        scheduler.kv_cache_manager.get_blocks(rid), self.geometry,
                        self._hash_block_size(), include_page_hashes=False))
        if prefixes:
            indexed = [(self._prefix_index_key(plan.base_key), plan) for plan in prefixes]
            present = self._present_prefix_indices(key for key, _ in indexed)
            complete = {}
            for key, plan in indexed:
                identity = (key, plan.committed_tokens)
                if key in present and identity not in complete:
                    complete[identity] = self._read_complete_prefix(key, plan) is not None
            # This is cache-write deduplication, not a promised restore. A later
            # lookup revalidates every TP payload if the retained set is evicted.
            prefixes = [plan for key, plan in indexed
                        if not complete.get((key, plan.committed_tokens), False)]
        preemptions = state.drain_preemptions()
        for snapshot in preemptions:
            if snapshot.identity in self._swap_payloads:
                raise OffloadLifecycleError("duplicate private swap dispatch")
            self._swap_payloads[snapshot.identity] = _SwapPayload(snapshot)
        for plan in self._restores:
            if plan.version:
                record = self._swap_payloads[(plan.request_id, plan.generation, plan.version, plan.epoch)]
                record.restore_queued = False
                record.restore_dispatched = True
        result = OscarOffloadMetadata(self.session, step_id, generations,
                                     preemptions, tuple(self._restores),
                                     tuple(prefixes), state.drain_branches(), self._prefix_view,
                                     tuple(sorted(self._retired_loads)))
        self._restores.clear()
        self._retired_loads.clear()
        return result

    def update_connector_output(self, connector_output):
        received = getattr(connector_output, "finished_recving", None) or ()
        for key in tuple(self._scheduled_loads):
            if key[0] in received:
                del self._scheduled_loads[key]
        metadata = connector_output.kv_connector_worker_meta
        if metadata is not None:
            if not isinstance(metadata, OscarOffloadWorkerMetadata):
                raise OffloadLifecycleError("unexpected offload completion metadata")
            for ack in metadata.acknowledgements:
                self._record_save(ack)
            for receipt in metadata.loads:
                self._record_load(receipt)
            for receipt in metadata.prefixes:
                if receipt.prefix_view != self._prefix_view:
                    continue
                key = (receipt.index_key, receipt.plan.base_key)
                pending = self._prefix_publications.setdefault(key, {})
                previous = pending.get(receipt.rank)
                if previous is not None and previous != receipt:
                    raise OffloadLifecycleError("conflicting prefix completion receipt")
                if not 0 <= receipt.rank < self.world_size:
                    raise OffloadLifecycleError("prefix completion rank is outside TP world")
                if any(replace(item.plan, block_ids=()) != replace(receipt.plan, block_ids=())
                       for item in pending.values()):
                    raise OffloadLifecycleError("TP prefix completions disagree")
                pending[receipt.rank] = receipt
                if len(pending) == self.world_size:
                    plan = receipt.plan
                    keys = expected_keys(self.namespace, self.world_size, plan, self.geometry)
                    # MetadataServer.publish performs the complete payload
                    # check immediately before its checked host write. Repeating
                    # the same all-TP SDK query here adds no eviction lease.
                    self.metadata_client.put(receipt.index_key,
                        {"namespace": self.namespace, "committed_tokens": plan.committed_tokens,
                         "base_key": plan.base_key, "index_key": receipt.index_key,
                         "prefix_view": self._prefix_view,
                         "keys": list(keys)})
                    del self._prefix_publications[key]
                    print(f"[oscar] KV_OFFLOAD_PREFIX_PUBLISHED tokens={plan.committed_tokens}", flush=True)

    def _prefix_index_key(self, base_key, view=None):
        return f"oscar-kv-index-v2:{self.namespace}:{self._prefix_view if view is None else view}:{base_key}"

    def reset_cache(self):
        # Native external reset changes only this scheduler's future reusable
        # prefix view. Private active snapshots and already-selected restores
        # remain owned; immutable shared prefix payloads are never deleted.
        self._prefix_view = uuid.uuid4().hex
        self._prefix_matches.clear()
        self._prefix_publications.clear()
        return True

    def request_finished(self, request, block_ids):
        return self.request_finished_all_groups(request, (block_ids,))

    def request_finished_all_groups(self, request, block_ids):
        self.scheduler_state.finish(request.request_id)
        self._prefix_matches.pop(request.request_id, None)
        # Saves are complete here. Native WAITING_FOR_REMOTE_KVS cancellation
        # independently delays destination free until all receive receipts.
        return False, None

    def _objects(self, plan, *, loading=False, canonical=()):
        state = {(entry.layer_name, entry.state_index): entry.values for entry in canonical}
        objects = []
        for index, layer in enumerate(self.geometry):
            name, family = layer["layer_name"], layer["cache_family"]
            export = self.exports[name]
            group = self._group_for_layer[name]
            ids = plan.block_ids[group]
            if family == "oscar_full":
                count = math.ceil(plan.committed_tokens / export.storage_block_size)
                start = plan.local_tokens // export.storage_block_size if loading else 0
                if count > len(ids):
                    raise OffloadLifecycleError("FULL snapshot exceeds its physical block table")
                for logical in range(start, count):
                    key = object_key(self.namespace, self.rank, plan, index, family,
                                     logical, layer_name=name)
                    spans = export.page_spans((ids[logical],))
                    objects.append(TransferObject(key, tuple(ByteSegment(s.address, s.length) for s in spans)))
            else:
                mode = export.format_metadata["mamba_cache_mode"]
                logical = 0 if mode == "none" else (plan.committed_tokens - 1) // export.storage_block_size
                if loading and logical >= len(ids):
                    raise OffloadLifecycleError("GDN restore has no native destination page")
                for component_index, component in enumerate(export.components):
                    key = object_key(self.namespace, self.rank, plan, index, family,
                                     component=component_index)
                    if loading:
                        span = component.span(ids[logical])
                        segments = (ByteSegment(span.address, span.length),)
                    else:
                        values = state[(name, component_index)]
                        length = values.numel() * values.element_size()
                        if not 0 < length <= component.page_bytes:
                            raise OffloadLifecycleError("canonical GDN component exceeds physical state")
                        segments = (ByteSegment(values.data_ptr(), length),)
                        if length < component.page_bytes:
                            segments += (ByteSegment(self._zero_tail.data_ptr(), component.page_bytes - length),)
                    objects.append(TransferObject(key, segments))
        return tuple(objects)

    def _save_resolved(self, plan, resolved):
        if resolved.committed_tokens != plan.committed_tokens:
            raise OffloadLifecycleError("planned prefix does not equal the committed worker state")
        plan = replace(plan, block_ids=resolved.block_ids)
        objects = self._objects(plan, canonical=resolved.mamba)
        self._synchronize()
        if objects:
            try:
                self.transport.put(objects)
            except Exception as error:
                if plan.version:
                    # No save receipt was published; no reader can select
                    # these private keys. A partial SDK failure poisons the
                    # transport, but bounded deletion remains cleanup-only.
                    try:
                        self.transport.rollback_private_write(tuple(obj.key for obj in objects))
                    except Exception as cleanup_error:
                        note = f"private swap rollback also failed: {cleanup_error}"
                        if hasattr(error, "add_note"):
                            error.add_note(note)
                        else:  # Package still permits Python 3.10.
                            error.args = (*error.args, note)
                raise

    def handle_preemptions(self, metadata):
        if not isinstance(metadata, OscarOffloadMetadata):
            raise OffloadLifecycleError("invalid offload execution metadata")
        if self._transfer_session is not None and self._transfer_session != metadata.session:
            raise OffloadLifecycleError("private swap worker session cannot be rebound")
        if self._transfer_session is None:
            if self.metadata_server is not None:
                self.metadata_server.configure_swap_scope(
                    self.namespace, metadata.session, self.world_size, self.geometry,
                    max_tokens=self.config.model_config.max_model_len)
            self._transfer_session = metadata.session
        self.tracker.apply_branches(self.runner, metadata.branches)
        # Native executes this before _update_states zeros any recycled page.
        for snapshot in metadata.preemptions:
            resolved = self.tracker.resolve(self.runner, snapshot, clone=False)
            count = resolved.committed_tokens
            key = snapshot_key(metadata.session, snapshot, count)
            if count:
                plan = PayloadPlan(snapshot.request_id, snapshot.generation, snapshot.version,
                                   snapshot.epoch, count, resolved.block_ids, key)
                self._save_resolved(plan, resolved)
                print(f"[oscar] KV_OFFLOAD_SWAP_SAVE request={snapshot.request_id} "
                      f"tokens={count} rank={self.rank} generation={snapshot.generation} "
                      f"version={snapshot.version}", flush=True)
            self._acks.append(SnapshotAck(*snapshot.identity, count, self.rank, key))
            self.tracker.forget(snapshot.request_id)

    def after_native_update_states(self, runner, scheduler_output):
        metadata = scheduler_output.kv_connector_metadata
        if not self.load_async:
            for plan in metadata.restores:
                reset_restored_cursor(runner, plan.request_id)
            return
        scheduled = scheduler_output.num_scheduled_tokens
        if (self._loading_request_ids.intersection(scheduled)
                or any(plan.request_id in scheduled for plan in metadata.restores)):
            raise OffloadLifecycleError("native admitted a request before its async load receipt")
        # Async allocation has no input-batch row. Reset only when native has
        # received all TP completions and actually admits this request. This
        # must precede preprocess_mamba, not run in the later forward context.
        for rid in scheduler_output.finished_req_ids:
            self._restored_cursors.pop(rid, None)
        for rid in scheduled:
            plan = self._restored_cursors.pop(rid, None)
            if plan is not None:
                if metadata.generations.get(rid) != plan.generation:
                    raise OffloadLifecycleError("async restore admission generation changed")
                reset_restored_cursor(runner, rid)

    def _load_payload(self, plan, producer_event=None):
        # The plan and registered export geometry contain only immutable page
        # IDs/addresses. Build descriptors on the I/O thread while native page
        # initialization can still be running; never read runner/batch state.
        objects = self._objects(plan, loading=True)
        if producer_event is not None:
            # Recorded by the model thread AFTER native new-page zeroing.
            # Waiting on this event on the I/O thread avoids a model-thread
            # stream sync while still protecting every destination byte.
            producer_event.synchronize()
        if not self.load_async:
            # The explicit synchronous mode has no deferred all-TP admission.
            keys = expected_keys(self.namespace, self.world_size, plan, self.geometry)
            if not all(self.transport.exists(keys)):
                raise TransferError("advertised joint checkpoint was evicted before restore")
        # Async: every rank must size-check and finish its own required data;
        # native finished_recving joins ALL ranks before admission. Rechecking
        # every rank's full payload on each worker duplicates that protection
        # and unnecessarily requires external copies of already-local pages.
        self.transport.get(objects)
        return plan

    def _record_restored(self, plan):
        if plan.version:
            self._load_acks.append(LoadReceipt(plan.request_id, plan.generation,
                plan.version, plan.epoch, plan.committed_tokens, self.rank, plan.base_key))
        print(f"[oscar] KV_OFFLOAD_RESTORE request={plan.request_id} "
              f"tokens={plan.committed_tokens} rank={self.rank} "
              f"kind={'swap' if plan.version else 'prefix'} generation={plan.generation} "
              f"version={plan.version} load_async={str(self.load_async).lower()}", flush=True)

    def start_load_kv(self, forward_context, **kwargs):
        metadata = self._get_connector_metadata()
        if metadata.restores and not self.load_async:
            # SDK copy streams must not race the native new-page zeroing.
            self._synchronize()
        # All restores in this frame follow the same native page-zero stream
        # boundary. One recorded event can be waited by every queued reader.
        event = self._record_producer_event() if metadata.restores and self.load_async else None
        for plan in metadata.restores:
            if self.load_async:
                key = (plan.request_id, plan.generation, plan.version, plan.epoch)
                if (plan.request_id in self._loading_request_ids
                        or plan.request_id in self._restored_cursors):
                    raise OffloadLifecycleError("duplicate asynchronous restore dispatch")
                if self._load_queue is None:
                    self._load_queue = AsyncLoadQueue(self._setup_load_device)
                self._load_queue.submit(key, lambda p=plan, e=event: self._load_payload(p, e))
                self._async_loads[key] = plan
                self._loading_request_ids.add(plan.request_id)
            else:
                self._load_payload(plan)
                self._record_restored(plan)
        # This hook also runs before FULL graph replay, and before native
        # pending GDN block copies; no Python per-layer hook is required.

    def observe_and_save(self, runner, output_token_ids, scheduler_output):
        metadata = scheduler_output.kv_connector_metadata
        if not isinstance(metadata, OscarOffloadMetadata):
            raise OffloadLifecycleError("postprocess lost its offload metadata")
        self.tracker.record_step(runner, scheduler_output, generations=metadata.generations,
                                 step_id=metadata.step_id)
        if metadata.prefixes:
            completion = self._prefix_producer_event
            if completion is None or completion[0] != metadata.step_id:
                raise OffloadLifecycleError("prefix publication lost its target/draft producer completion")
            # Native postprocess runs on global_stream, which waits only for
            # sampling_done_event recorded BEFORE the drafter. The event below
            # was recorded by finalize_kv_connector AFTER draft writes. Wait
            # for both producer streams before reading their shared raw pool.
            completion[1].synchronize()
            self._prefix_producer_event = None
        for plan in metadata.prefixes:
            snapshot = RequestSnapshot(plan.request_id, plan.generation, 0, plan.epoch,
                plan.block_ids, plan.committed_tokens, plan.committed_tokens + 1,
                0, token_digest(()), runner.cache_config.mamba_cache_mode)
            resolved = self.tracker.resolve(runner, snapshot, clone=False)
            index_key = self._prefix_index_key(plan.base_key, metadata.prefix_view)
            # A writer never modifies another writer's payload, including
            # concurrent identical prefixes. Only the small checked manifest
            # may change; either version references a complete immutable set.
            generation = snapshot_key(metadata.session, snapshot, plan.committed_tokens)
            plan = replace(plan, base_key=f"prefix/data/{generation}/step{metadata.step_id}",
                           page_hashes=())
            self._save_resolved(plan, resolved)
            self._prefix_acks.append(PrefixReceipt(index_key, replace(plan, block_ids=()),
                                                   self.rank, metadata.prefix_view))

    def wait_for_layer_load(self, layer_name):
        return None

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        return None

    def wait_for_save(self):
        # Native defers this call until after MTP draft forward. Without MTP,
        # the target-forward exit calls it directly. The accepted GDN state
        # is still finalized later, so this is a producer fence, not a save.
        metadata = self._get_connector_metadata()
        if metadata.prefixes:
            self._prefix_producer_event = (metadata.step_id, self._record_producer_event())

    def _record_producer_event(self):
        import torch
        event = torch.npu.Event()
        event.record()
        return event

    def get_finished(self, finished_req_ids):
        for rid in finished_req_ids:
            self.tracker.forget(rid)
            self._restored_cursors.pop(rid, None)
        # finished_req_ids can refer to an old request ID resubmitted in this
        # same frame. Scheduler retirement identities distinguish its old DMA
        # from the new generation's read (including pre-dispatch cancellation).
        if self.load_async:
            self._cancelled_loads.update(key for key in self._get_connector_metadata().retired_loads
                                         if key in self._async_loads)
        received = set()
        if self._load_queue is not None:
            for key, plan in self._load_queue.poll():
                if self._async_loads.pop(key, None) != plan:
                    raise OffloadLifecycleError("unexpected async load completion identity")
                self._loading_request_ids.remove(plan.request_id)
                self._record_restored(plan)
                if key not in self._cancelled_loads:
                    self._restored_cursors[plan.request_id] = plan
                self._cancelled_loads.discard(key)
                # Exactly one receive receipt per rank, also after cancel.
                # Native frees cancelled destinations only after all TP ranks
                # finish; sending a second finished_sending would double free.
                received.add(plan.request_id)
        return set(), received

    def build_connector_worker_meta(self):
        result = OscarOffloadWorkerMetadata(tuple(self._acks), tuple(self._prefix_acks),
                                           tuple(self._load_acks))
        self._acks.clear()
        self._prefix_acks.clear()
        self._load_acks.clear()
        return result

    def shutdown(self):
        try:
            if self._load_queue is not None:
                self._load_queue.shutdown()
        finally:
            # SDK failure is raised after draining, but interruption of join
            # need not have drained. Keep owners in that case; a later close
            # can join again. Never unregister a still-running DMA's buffer.
            if self._load_queue is None or self._load_queue.drained:
                self.tracker.clear()
                if self.metadata_server is not None:
                    self.metadata_server.close()
                if self.metadata_client is not None:
                    self.metadata_client.close()
                self.transport.close()
