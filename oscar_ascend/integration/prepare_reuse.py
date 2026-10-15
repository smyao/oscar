"""Archive #34/#36/#37-49/#129/#142/#157: reuse one eager task description.

Native model_runner_v1.py shares one metadata object among an attention group's
layers. GraphWorkspace is shared by equal device/geometry and its tasks and
positions are read-only after source2 suppression. Physical KV, rotations,
outputs and status remain per-layer operations; none is cached here.

Reuse is confined to a real main-model prefill in one native ForwardContext
on the same stream. A repeated layer, a new context/metadata, another geometry or any alternate workspace
writer ends the reuse interval. Capture, dummy, draft and partitioned paths
retain the original prepare launches. No device values or historical KV are
read by the host, no allocation is added and no error task is discarded.
"""

from dataclasses import dataclass


@dataclass
class _Prepared:
    metadata: object
    geometry: tuple
    stream: object
    context: object
    layers: set[str]
    slots: object


def invalidate_prepare_reuse(workspace):
    workspace._oscar_prepared_tasks = None


def can_reuse_prepare(metadata, *, enabled, native_current):
    if type(enabled) is not bool:
        raise ValueError("experimental_prepare_reuse must be an explicit boolean")
    return (enabled and native_current and not metadata.is_draft
            and not metadata.capture_origin and not metadata.dummy_origin
            and metadata.current_only_plan is None
            and metadata.mixed_split_segment is None)


def reused_slots(workspace, metadata, layer_name, geometry, stream, context):
    """Return prepared slot storage only within one unchanged layer sequence.

    Native set_ascend_forward_context calls set_forward_context, whose
    create_forward_context creates a new object for every invocation. Identity,
    not tensor addresses or dataclass equality, identifies that invocation.
    Repeating any layer also protects repeated model calls within one context.
    """
    prepared = getattr(workspace, "_oscar_prepared_tasks", None)
    if (prepared is not None and prepared.metadata is metadata and prepared.context is context
            and prepared.geometry == geometry and prepared.stream == stream
            and layer_name not in prepared.layers):
        prepared.layers.add(layer_name)
        return prepared.slots
    invalidate_prepare_reuse(workspace)
    return None


def remember_prepared(workspace, metadata, layer_name, geometry, stream, slots, context):
    """Publish only after prepare and the existing current-source suppression."""
    workspace._oscar_prepared_tasks = _Prepared(
        metadata, geometry, stream, context, {layer_name}, slots)
