"""Worker-authoritative committed cursor and canonical GDN snapshot copies.

Archive #37-49: scheduled counts and stored/accepted state have different
owners. Archive #127/#128: index native physical groups, never metadata128.
Archive #157/#163: true warm prefill uses state column zero; MTP uses only
its accepted column. Archive #171: copy before native physical-page reuse.
Evidence: pinned model_runner_v1._prepare_inputs/_update_states_after_model_execute,
mamba_utils.preprocess_mamba/postprocess_mamba_align_gpu and copy functions.
All copies are offload-only; existing model arithmetic/CANN sources are unchanged.
Archive #172/#174: reuse allocation/immutable host metadata only, without
changing counter arithmetic, native state selection or input-length policy.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from .scheduler import OffloadLifecycleError, RequestSnapshot, freeze_block_ids, token_digest


@dataclass(frozen=True)
class CanonicalMambaTensor:
    layer_name: str
    group_id: int
    state_index: int
    # Contiguous native state span. With clone=False it aliases registered
    # storage and MUST be synchronously transferred before any page reuse.
    values: Any


@dataclass(frozen=True)
class ResolvedCheckpoint:
    snapshot: RequestSnapshot
    committed_tokens: int
    step_id: int
    # Last executed worker mapping, not a scheduler optimistic allocation.
    block_ids: tuple[tuple[int, ...], ...]
    mamba: tuple[CanonicalMambaTensor, ...]


@dataclass(frozen=True)
class _WorkerRecord:
    generation: int
    step_id: int
    block_ids: tuple[tuple[int, ...], ...]
    base: Any
    accepted: Any
    next_accepted: Any
    scheduled: int
    speculative: int
    running_state_index: int
    mode: str
    ready_event: Any
    counter_snapshot: Any = None
    block_cache: Any = None


@dataclass(eq=False)
class _CounterSnapshot:
    """One step's buffers; all records must release them before reuse."""

    signature: tuple
    buffers: tuple
    stream: Any
    ready_event: Any = None
    references: int = 0


def _freeze_worker_blocks(blocks, previous=None):
    """Reuse equal immutable groups without trusting mutable-list identity.

    Native lists are compared in full on every step. Only plain int/bool
    contents can take the equality shortcut: e.g. complex(1) compares equal
    to int(1) but the original int conversion must still reject it. Other
    int-like values retain freeze_block_ids' conversion/error semantics.
    """
    if hasattr(blocks, "get_block_ids"):
        blocks = blocks.get_block_ids()
    groups, comparisons, changed = [], [], []
    cached = previous.block_cache if previous is not None else None
    for index, group in enumerate(blocks):
        # map(type, ...) and sequence equality scan in C. This is still O(P)
        # content validation, not a version inferred from identity or length.
        plain = type(group) in (list, tuple) and set(map(type, group)) <= {int, bool}
        old = cached[index] if cached is not None and index < len(cached) else None
        if plain and old is not None and group == old:
            frozen = previous.block_ids[index]
            comparison = old
        else:
            frozen = None
            changed.append((index, group))
            comparison = group.copy() if plain and type(group) is list else group if plain else None
        groups.append(frozen)
        comparisons.append(comparison)
    # Preserve the original error order: convert ALL changed elements before
    # checking any negative ID. Equal cached groups already passed both steps.
    if changed:
        converted = freeze_block_ids(tuple(group for _, group in changed))
        for (index, _), frozen in zip(changed, converted):
            groups[index] = frozen
    if (previous is not None and len(groups) == len(previous.block_ids)
            and all(group is old for group, old in zip(groups, previous.block_ids))):
        return previous.block_ids, previous.block_cache
    return tuple(groups), tuple(comparisons)


def _record_counter_stream(buffers, stream):
    # Allocation-stream ordering already protects same-stream destruction.
    # Cross-stream reuse must also register its writer with the allocator, so
    # clear/free-cache eviction cannot recycle memory while a copy is pending.
    for buffer in buffers:
        if buffer.device.type != "cpu":
            buffer.record_stream(stream)


def committed_position(base: int, scheduled: int, speculative: int, accepted: int) -> int:
    """Mirror native rejection correction, including true chunked prefill."""
    if base < 0 or scheduled <= 0 or not 0 <= speculative < scheduled:
        raise OffloadLifecycleError("invalid native scheduled-token checkpoint")
    if speculative:
        if not 1 <= accepted <= speculative + 1:
            raise OffloadLifecycleError("no accepted GDN state for a speculative snapshot")
        return base + scheduled - speculative + accepted - 1
    return base + scheduled


def _integer(value) -> int:
    return int(value.item()) if hasattr(value, "item") else int(value)


def _flat_storage_view(tensor, start_addr: int, count: int):
    """Use the native pointer contract without flattening padded page strides."""
    delta = int(start_addr) - tensor.data_ptr()
    element_size = tensor.element_size()
    if delta < 0 or delta % element_size or count <= 0:
        raise OffloadLifecycleError("invalid native Mamba copy span")
    offset = tensor.storage_offset() + delta // element_size
    if offset + count > tensor.untyped_storage().nbytes() // element_size:
        raise OffloadLifecycleError("native Mamba copy exceeds the registered tensor storage")
    return tensor.as_strided((count,), (1,), storage_offset=offset)


def _mamba_groups(runner):
    # Lazy import: plugin registration must not import vLLM/torch/NPU (#77).
    from vllm.v1.kv_cache_interface import MambaSpec

    for group_id, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        if isinstance(group.kv_cache_spec, MambaSpec):
            yield group_id, group


def copy_canonical_mamba(runner, blocks, running_index: int, accepted: int,
                         *, groups=None, clone: bool = True) -> tuple[CanonicalMambaTensor, ...]:
    """Copy the exact state the next native forward would read to private scratch.

    Native copy functions select temporal block[running+accepted-1] and the
    conv slice [accepted-1:]. This works for none and align physical layouts.
    No source page is changed, including shared prefix checkpoint pages.
    clone=False borrows registered native ranges for a synchronous store at
    the before-reuse seam; the caller owns that lifetime through completion.
    """
    if accepted < 1 or running_index < 0:
        raise OffloadLifecycleError("invalid canonical Mamba state cursor")
    funcs = runner.model.get_mamba_state_copy_func()
    context = runner.compilation_config.static_forward_context
    result = []
    group_iter = _mamba_groups(runner) if groups is None else groups
    for gid, group in group_iter:
        ids = blocks[gid]
        if running_index + accepted - 1 >= len(ids):
            raise OffloadLifecycleError("accepted temporal state lies outside native group allocation")
        for layer_name in group.layer_names:
            states = context[layer_name].kv_cache
            if len(states) != len(funcs):
                raise OffloadLifecycleError("native GDN state/copy-function count changed")
            for state_index, (state, copy_func) in enumerate(zip(states, funcs)):
                spec = copy_func(state, list(ids), running_index, accepted)
                source = _flat_storage_view(state, spec.start_addr, spec.num_elements)
                values = source.detach().clone() if clone else source.detach()
                result.append(CanonicalMambaTensor(layer_name, gid, state_index, values))
    return tuple(result)


def restore_canonical_mamba(runner, tensors, blocks, committed_tokens: int,
                            mode: str, *, groups=None) -> None:
    """Install canonical state before native pending Mamba copies execute.

    The native align preprocessor uses (L-1)//B as the external state source;
    none uses block zero. Both subsequently read accepted count one. Restoring
    tensor bytes alone cannot replace the corresponding worker cursor reset.
    """
    if committed_tokens <= 0 or mode not in ("none", "align"):
        raise OffloadLifecycleError("invalid restored Mamba checkpoint position/mode")
    group_map = dict(_mamba_groups(runner) if groups is None else groups)
    context = runner.compilation_config.static_forward_context
    expected = {(name, gid, index)
                for gid, group in group_map.items()
                for name in group.layer_names
                for index in range(len(context[name].kv_cache))}
    seen = set()
    for entry in tensors:
        identity = entry.layer_name, entry.group_id, entry.state_index
        if identity not in expected or identity in seen:
            raise OffloadLifecycleError("unexpected/duplicate canonical GDN tensor")
        seen.add(identity)
        group = group_map[entry.group_id]
        logical = 0 if mode == "none" else (committed_tokens - 1) // group.kv_cache_spec.block_size
        ids = blocks[entry.group_id]
        if logical >= len(ids):
            raise OffloadLifecycleError("restored GDN checkpoint has no destination block")
        state = context[entry.layer_name].kv_cache[entry.state_index]
        destination = state[ids[logical]]
        if entry.values.dtype != state.dtype or entry.values.numel() > destination.numel():
            raise OffloadLifecycleError("canonical GDN tensor dtype/size mismatch")
        destination.zero_()
        flat = _flat_storage_view(state, destination.data_ptr(), entry.values.numel())
        flat.copy_(entry.values)
    if seen != expected:
        raise OffloadLifecycleError("canonical snapshot is missing native GDN tensors")


def reset_restored_cursor(runner, request_id: str) -> None:
    """Run after native _update_states has added a resumed request row."""
    index = runner.input_batch.req_id_to_index.get(request_id)
    if index is None:
        raise OffloadLifecycleError("restored request was not added to the native input batch")
    runner.input_batch.num_accepted_tokens_cpu[index] = 1
    runner.num_accepted_tokens.np[index] = 1
    # _prepare_inputs copies these CPU values; async remapping must treat the
    # resumed request as new, exactly as native remove_request/add_request do.
    previous = runner.input_batch.prev_req_id_to_index
    if previous is not None:
        previous.pop(request_id, None)
    request = runner.requests[request_id]
    request.prev_num_draft_len = 0
    runner.mamba_state_idx.pop(request_id, None)


class WorkerCheckpointTracker:
    """Retain small per-step device metadata, never entire request pages.

    record_step runs in the same native stream AFTER state postprocessing.
    Small owned tensor copies preserve corrected counters even if a request leaves
    the next persistent batch. Only an actual save reads them on the host.
    Live slabs are bounded by retained records (plus one under construction);
    the free cache holds at most two slabs across ALL layouts. Same-stream
    reuse is FIFO ordered; an unfinished other-stream producer is never reused.
    """

    def __init__(self):
        self._records: dict[str, _WorkerRecord] = {}
        self._free_counters: list[_CounterSnapshot] = []

    def _acquire_counters(self, sources, stream):
        signature = tuple((source.device, source.dtype, tuple(source.shape)) for source in sources)
        for index in range(len(self._free_counters) - 1, -1, -1):
            slot = self._free_counters[index]
            if slot.signature != signature:
                continue
            same_stream = slot.stream == stream
            if not same_stream and not slot.ready_event.query():
                continue
            self._free_counters.pop(index)
            if not same_stream:
                _record_counter_stream(slot.buffers, stream)
            slot.stream = stream
            return slot
        import torch
        return _CounterSnapshot(signature, tuple(torch.empty_like(source) for source in sources), stream)

    def _release_record(self, record):
        slot = record.counter_snapshot if record is not None else None
        if slot is None:
            return
        if slot.references <= 0:
            raise OffloadLifecycleError("counter snapshot has no remaining record owner")
        slot.references -= 1
        if slot.references == 0:
            self._free_counters.append(slot)
            # Every non-allocation writer was recorded with the allocator.
            # Dropping an unfinished free slab is therefore stream-safe too.
            del self._free_counters[:-2]

    def record_step(self, runner, scheduler_output, *, generations: Mapping[str, int],
                    step_id: int) -> None:
        req_ids = tuple(runner.input_batch.req_ids)
        if not req_ids:
            return
        mode = runner.cache_config.mamba_cache_mode
        if mode not in ("none", "align"):
            raise OffloadLifecycleError("offload worker requires native none or align mode")
        count = len(req_ids)
        pending = []
        for index, rid in enumerate(req_ids):
            if rid not in scheduler_output.num_scheduled_tokens:
                continue
            if rid not in generations:
                raise OffloadLifecycleError("scheduled offload request has no generation")
            previous = self._records.get(rid)
            if previous is not None and previous.generation != generations[rid]:
                previous = None
            blocks, comparison = _freeze_worker_blocks(runner.requests[rid].block_ids, previous)
            pending.append((index, rid, blocks, comparison,
                            int(scheduler_output.num_scheduled_tokens[rid]),
                            len(scheduler_output.scheduled_spec_decode_tokens.get(rid, ())),
                            0 if mode == "none" else int(runner.mamba_state_idx[rid])))
        if not pending:
            return
        sources = (runner.num_computed_tokens.detach(), runner.num_accepted_tokens.gpu.detach())
        # This is the raw sampler count used by native GDN postprocessing.
        # The drafter's discard-masked valid_sampled_token_count may be zero
        # for a true prefill row (#157); it is not this commit counter.
        if mode == "align":
            context = runner._get_mamba_bufs().postprocess_align
            sources += (context.num_accepted_tokens_out.detach(),)
        if any(source.ndim != 1 or source.numel() < count for source in sources):
            raise OffloadLifecycleError("native checkpoint counters do not cover the request rows")
        # The caller is inside native async_state_update's stream. A device
        # event establishes ownership before another stream reads these copies.
        import torch
        device = getattr(torch, runner.device.type)
        slot = self._acquire_counters(sources, device.current_stream())
        for destination, source in zip(slot.buffers, sources):
            destination[:count].copy_(source[:count])
        base, accepted = slot.buffers[:2]
        normalized = slot.buffers[2] if mode == "align" else accepted
        event = device.Event()
        event.record()
        slot.ready_event = event
        slot.references = len(pending)
        for index, rid, blocks, comparison, scheduled, speculative, running in pending:
            old = self._records.get(rid)
            self._records[rid] = _WorkerRecord(
                generations[rid], step_id, blocks,
                base[index], accepted[index], normalized[index],
                scheduled, speculative, running, mode, event, slot, comparison,
            )
            self._release_record(old)

    def resolve(self, runner, snapshot: RequestSnapshot, *, clone: bool = True) -> ResolvedCheckpoint:
        record = self._records.get(snapshot.request_id)
        if record is None:
            if snapshot.num_computed_hint == 0:
                return ResolvedCheckpoint(snapshot, 0, 0, snapshot.block_ids, ())
            raise OffloadLifecycleError("preempted request lacks an executed worker checkpoint")
        if record.generation != snapshot.generation or record.mode != snapshot.mamba_mode:
            raise OffloadLifecycleError("stale worker checkpoint generation or native layout")
        record.ready_event.synchronize()
        position = committed_position(_integer(record.base), record.scheduled,
                                      record.speculative, _integer(record.accepted))
        # True prefill stores only the final state at column zero (#163), even
        # for qLen == MTP width; an accepted count from another row is irrelevant.
        accepted = _integer(record.next_accepted) if record.speculative else 1
        canonical = copy_canonical_mamba(runner, record.block_ids,
                                          record.running_state_index, accepted, clone=clone)
        return ResolvedCheckpoint(snapshot, position, record.step_id,
                                  record.block_ids, canonical)

    def apply_branches(self, runner, transitions) -> None:
        """Carry a held native streaming checkpoint across an explicit branch.

        Native _handle_stopped_request retains the cache/worker request while
        waiting for more streaming input; it does not emit finished_req_ids.
        A new prompt chunk discards only the last, uncomputed sampled token.
        Relabeling is allowed only for that exact retained computed prefix.
        """
        for transition in transitions:
            if (transition.new_generation <= transition.old_generation
                    or len(transition.retained_token_ids) != transition.retained_tokens
                    or token_digest(transition.retained_token_ids) != transition.retained_prefix_digest):
                raise OffloadLifecycleError("invalid streaming branch checkpoint identity")
            record = self._records.get(transition.request_id)
            if record is None:
                # A genuinely new request may have no executed state yet.
                # If preemption needs existing bytes, resolve's normal guard
                # still requires an executed checkpoint for nonzero history.
                continue
            if record.generation >= transition.new_generation:
                continue  # duplicate, or a newer generation has already run
            if record.generation != transition.old_generation:
                raise OffloadLifecycleError("streaming branch skipped a worker checkpoint generation")
            record.ready_event.synchronize()
            committed = committed_position(_integer(record.base), record.scheduled,
                                           record.speculative, _integer(record.accepted))
            if committed != transition.retained_tokens:
                raise OffloadLifecycleError("streaming branch retained a different native GDN cursor")
            request = runner.requests.get(transition.request_id)
            if request is not None:
                known = list(getattr(request, "prompt_token_ids", ()) or ())
                known.extend(getattr(request, "output_token_ids", ()) or ())
                # Async native worker bookkeeping uses -1 placeholders until
                # the next input preparation. Compare every available token;
                # the scheduler's full prefix hash plus old generation and
                # device-derived cursor fence the unresolved async suffix.
                for before, retained in zip(known, transition.retained_token_ids):
                    if int(before) >= 0 and int(before) != retained:
                        raise OffloadLifecycleError("streaming branch changed the worker's known prefix")
            self._records[transition.request_id] = replace(record, generation=transition.new_generation)

    def forget(self, request_id: str) -> None:
        self._release_record(self._records.pop(request_id, None))

    def clear(self) -> None:
        for record in self._records.values():
            self._release_record(record)
        self._records.clear()
        self._free_counters.clear()
