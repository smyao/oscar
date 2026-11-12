"""Archive #34/#36/#37-49/#129/#142/#157: reuse one eager task description.

Native model_runner_v1.py shares one metadata object among an attention group's
layers. GraphWorkspace is shared by equal device/geometry and its tasks and
positions are read-only after source2 suppression. Physical KV, rotations,
outputs and status remain per-layer operations; none is cached here.

Reuse is confined to one registered main-model ForwardContext and stream.
Eager prefill retains its original qualification. Main decode may also share
the read-only descriptors, including one uniform FULL graph recording: the
first layer records the real prepare launch, so every replay rewrites these
fixed buffers. Standalone dummy/profile, draft and PIECEWISE are excluded.
Mixed request domains use two disjoint regions of the existing arenas; no
layer order establishes freshness. No device values or historical KV are
read by the host, no device allocation is added and no error task is discarded.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PrepareRegion:
    owner: object
    request_begin: int
    request_end: int
    token_begin: int
    token_end: int
    parent_tokens: int


@dataclass
class _Prepared:
    metadata: object
    geometry: tuple
    stream: object
    context: object
    layers: set[str]
    slots: object
    role: str
    token_span: tuple[int, int]
    task_span: tuple[int, int]
    boundaries: object
    phase: tuple
    descriptor: object


def invalidate_prepare_reuse(workspace):
    workspace._oscar_prepared_tasks = None


def can_reuse_prepare(metadata, *, enabled, native_current):
    if type(enabled) is not bool:
        raise ValueError("experimental_prepare_reuse must be an explicit boolean")
    return (enabled and native_current and not metadata.is_draft
            and not metadata.capture_origin and not metadata.dummy_origin
            and metadata.current_only_plan is None
            and (metadata.mixed_split_segment is None
                 or isinstance(metadata.prepare_region, PrepareRegion)))


def immutable_query_starts(metadata):
    """The builder's per-step tuple, never the mutable native CPU mirror."""
    ends = metadata.current_cumulative
    if (not isinstance(ends, tuple) or len(ends) != metadata.num_reqs or not ends
            or any(type(value) is not int for value in ends)
            or any(a >= b for a, b in zip((0, *ends[:-1]), ends))
            or ends[-1] != metadata.num_actual_tokens):
        return None
    return (0, *ends)


def registered_prepare_owner(metadata, layer_name, context):
    registered = getattr(context, "attn_metadata", None)
    return (isinstance(registered, dict) and registered.get(layer_name) is metadata
            and not getattr(context, "capturing", False)
            and not getattr(context, "in_profile_run", False))


def decode_prepare_candidate(metadata):
    """Cheap host-only filter before asking for the native forward context."""
    if (metadata.is_draft or metadata.current_only_plan is not None
            or metadata.mixed_split_segment is not None or metadata.prepare_region is not None):
        return False
    if metadata.capture_origin:
        return True
    state = metadata.attn_state
    return (not metadata.dummy_origin and type(state).__name__ == "AscendAttentionState"
            and state.name in {"DecodeOnly", "SpecDecoding"})


def registered_decode_prepare_owner(metadata, layer_name, context):
    """Prove eager main decode or one complete uniform FULL recording.

    #34/#36/#127/#128/#157/#163: never infer valid rows from dummy values or
    reuse MTP's accepted-position preparation. Native non-MLA MTP capture has
    ChunkedPrefill state (model_runner_v1.py:3483-3490); the uniform FULL batch
    descriptor, not a query-length guess, establishes its capture domain.
    ACLGraphWrapper.__call__ records the complete runnable once and later
    replays it without Python. Its first prepare must remain inside the graph.
    """
    registered = getattr(context, "attn_metadata", None)
    if (not decode_prepare_candidate(metadata) or not isinstance(registered, dict)
            or registered.get(layer_name) is not metadata
            or getattr(context, "is_draft_model", None) is not False
            or getattr(context, "in_profile_run", None) is not False
            or getattr(context, "ubatch_slices", None) is not None):
        return False
    mode = getattr(getattr(context, "cudagraph_runtime_mode", None), "name", None)
    capturing = getattr(context, "capturing", None)
    if metadata.capture_origin:
        return (capturing is True and mode == "FULL"
                and getattr(getattr(context, "batch_descriptor", None), "uniform", None) is True)
    return capturing is False and mode == "NONE"


def mixed_prepare_eligible(metadata, layer_name, *, enabled, native_current):
    """Require the actual main-model context to own the unsliced metadata."""
    if (not can_reuse_prepare(metadata, enabled=enabled, native_current=native_current)
            or not metadata.mixed_decode_split
            or metadata.mixed_split_segment is not None
            or type(metadata.num_input_tokens) is not int
            or immutable_query_starts(metadata) is None):
        return False
    from vllm.forward_context import get_forward_context
    context = get_forward_context()
    return registered_prepare_owner(metadata, layer_name, context)


def prepare_region_offset(metadata, tokens, layer_name, context):
    """Validate a trusted child span against its registered native parent."""
    region = getattr(metadata, "prepare_region", None)
    if region is None:
        return 0
    from .runtime_api import OscarReadinessError
    if not isinstance(region, PrepareRegion):
        raise OscarReadinessError("mixed prepare region has no declared ownership")
    owner = region.owner
    starts = immutable_query_starts(owner)
    rb, re, begin, end, capacity = (region.request_begin, region.request_end,
                                   region.token_begin, region.token_end, region.parent_tokens)
    if (not registered_prepare_owner(owner, layer_name, context)
            or starts is None or any(type(value) is not int for value in (rb,re,begin,end,capacity))
            or not 0 <= rb < re <= owner.num_reqs
            or not 0 <= begin < end <= owner.num_actual_tokens <= capacity
            or capacity != owner.num_input_tokens or capacity != metadata.cv_shape_tokens
            or begin != starts[rb] or end != starts[re] or tokens != end-begin
            or metadata.num_input_tokens != tokens or metadata.num_actual_tokens != tokens
            or metadata.num_reqs != re-rb
            or metadata.current_cumulative != tuple(value-begin for value in starts[rb+1:re+1])
            or metadata.mixed_split_segment not in {"short_prefix", "long_suffix"}
            or (metadata.mixed_split_segment == "short_prefix" and (rb != 0 or begin != 0))
            or (metadata.mixed_split_segment == "long_suffix" and (re != owner.num_reqs or end != owner.num_actual_tokens))):
        raise OscarReadinessError("mixed prepare span differs from its native ForwardContext parent")
    # Host address/shape facts only. Native physical table/slot contents and
    # discard semantics remain untouched; no .item/.cpu or table-value scan.
    for child, parent in ((metadata.slot_mapping, owner.slot_mapping[begin:end]),
                          (metadata.seq_lens, owner.seq_lens[rb:re]),
                          (metadata.block_tables, owner.block_tables[rb:re])):
        if (child.data_ptr() != parent.data_ptr() or child.shape != parent.shape
                or child.stride() != parent.stride() or child.dtype != parent.dtype):
            raise OscarReadinessError("mixed prepare child rebound native slot/length/table storage")
    return begin


def prepare_views(workspace, metadata, tokens, kv_heads, splits, layer_name, context):
    """Use token units for positions/slots and descriptor units for tasks."""
    from .runtime_api import OscarReadinessError
    begin = prepare_region_offset(metadata, tokens, layer_name, context)
    end = begin + tokens
    factor = kv_heads * 3 * splits
    if (end > workspace.positions.shape[0] or end > workspace.slots.shape[0]
            or end * factor > workspace.tasks.shape[0]):
        raise OscarReadinessError("mixed prepare region exceeds the fixed scratch arenas")
    return (workspace.tasks[begin*factor:end*factor], workspace.positions[begin:end],
            workspace.slots[begin:end])


def _identity(metadata, geometry):
    region = getattr(metadata, "prepare_region", None)
    begin = 0 if region is None else region.token_begin
    owner = metadata if region is None else region.owner
    role = "whole" if region is None else metadata.mixed_split_segment
    end = begin + geometry[0]
    factor = geometry[2] * 3 * geometry[-1]
    return owner, role, (begin, end), (begin*factor, end*factor)


def _same_scope(entries, owner, stream, context):
    return all(entry.metadata is owner and entry.context is context
               and entry.stream == stream
               and entry.boundaries is getattr(owner, "current_cumulative", None)
               and entry.phase == _phase(context)
               and entry.descriptor is getattr(context, "batch_descriptor", None)
               for entry in entries)


def _phase(context):
    # Explicitly separate warmup and capture even if a caller retains the same
    # context/metadata. Only identity and host flags are used, never tensor
    # values or inference tensor versions. PIECEWISE never enters this lease.
    return (getattr(context, "capturing", None),
            getattr(getattr(context, "cudagraph_runtime_mode", None), "name", None),
            getattr(context, "is_draft_model", None),
            getattr(context, "in_profile_run", None))


def _intersects(left, right):
    return left[0] < right[1] and right[0] < left[1]


def _before_write(entries, role, tokens, tasks):
    # At most two known mixed regions; this is never a request/history scan.
    if role == "whole":
        return []
    return [entry for entry in entries if entry.role not in {"whole", role}
            and not _intersects(entry.token_span, tokens)
            and not _intersects(entry.task_span, tasks)]


def reused_slots(workspace, metadata, layer_name, geometry, stream, context):
    """Return prepared slot storage only within one native model invocation.

    Native set_ascend_forward_context calls set_forward_context, whose
    create_forward_context creates a new object for every invocation. Identity,
    not tensor addresses or dataclass equality, identifies that invocation.
    Repeating any layer also protects repeated model calls within one context.
    """
    owner, role, token_span, task_span = _identity(metadata, geometry)
    entries = getattr(workspace, "_oscar_prepared_tasks", None) or []
    if not _same_scope(entries, owner, stream, context):
        entries = []
    for prepared in entries:
        if (prepared.role == role and prepared.token_span == token_span
                and prepared.task_span == task_span and prepared.geometry == geometry
                and layer_name not in prepared.layers):
            prepared.layers.add(layer_name)
            return prepared.slots
    remaining = _before_write(entries, role, token_span, task_span)
    workspace._oscar_prepared_tasks = remaining or None
    return None


def remember_prepared(workspace, metadata, layer_name, geometry, stream, slots, context):
    """Publish only after prepare and the existing current-source suppression."""
    owner, role, token_span, task_span = _identity(metadata, geometry)
    entries = getattr(workspace, "_oscar_prepared_tasks", None) or []
    if not _same_scope(entries, owner, stream, context):
        entries = []
    entries = _before_write(entries, role, token_span, task_span)
    entries.append(_Prepared(owner, geometry, stream, context, {layer_name}, slots,
                             role, token_span, task_span, getattr(owner, "current_cumulative", None),
                             _phase(context), getattr(context, "batch_descriptor", None)))
    workspace._oscar_prepared_tasks = entries
