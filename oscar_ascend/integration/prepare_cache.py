"""Archive #34/#36/#142/#148/#155: reuse immutable task metadata, never KV.

The native CommonAttentionMetadata supplies an epoch; tensor addresses alone
are not an epoch. Repeating a layer starts a new traversal even when native
warmup/capture reuses that common object. Partition writers invalidate the
shared arena. The first captured layer always records a prepare, so graph
replay refreshes tasks/positions/int64 slots before any layer reads them.
"""


def _tensor_key(tensor):
    return (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)


def prepare_key(metadata, *, tokens, heads, kv_heads, sink, recent, splits, native_current):
    epoch = getattr(metadata, "prepare_epoch", None)
    if epoch is None or getattr(metadata, "mixed_split_segment", None) is not None:
        return None
    return (epoch, tokens, heads, kv_heads, sink, recent, splits, native_current,
            metadata.num_reqs, metadata.draft_index, metadata.is_draft,
            metadata.capture_origin, metadata.dummy_origin,
            *(_tensor_key(t) for t in (metadata.query_start_loc, metadata.seq_lens,
                                       metadata.slot_mapping, metadata.block_tables)))


def invalidate_prepare(workspace):
    workspace._oscar_prepared = None


def reuse_prepare(workspace, key, layer):
    prior = getattr(workspace, "_oscar_prepared", None)
    if key is None or prior is None or prior[0] != key or layer in prior[1]:
        invalidate_prepare(workspace)
        return False
    prior[1].add(layer)
    return True


def commit_prepare(workspace, key, layer):
    workspace._oscar_prepared = (key, {layer}) if key is not None else None
