"""Explicit storage-format routing; archive #17-22/#37-49/#69/#127-129/#155.

No reader/writer may silently interpret a different layout. New request
windows are published after attention, then their device status is guarded
before any subsequent layer/step can consume the updated storage.
"""
from .runtime_api import OscarReadinessError
from ..ops.cv_dispatch import MEMBER_PIPELINE_CV_OP

REQUEST_FORMAT = "striped_request_window_v1"
REQUEST_MEMBER_PIPELINE_CV_OP = "attention_request_window_member_pipeline_cluster4_out"
REQUEST_READERS = {
    "attention_cv_bundle_decode_out": "attention_request_window_decode_out",
    "attention_cv_bundle_q1_out": "attention_request_window_decode_q1_out",
    "attention_cv_window_range_out": "attention_request_window_out",
    "attention_cv_window_range_balanced_out": "attention_request_window_balanced_out",
    "attention_cv_window_range_cluster4_out": "attention_request_window_cluster4_out",
    "attention_cv_window_range_cluster16_out": "attention_request_window_cluster16_out",
    MEMBER_PIPELINE_CV_OP: REQUEST_MEMBER_PIPELINE_CV_OP,
}
REQUEST_CLUSTER_READERS = frozenset({
    "attention_request_window_cluster4_out", "attention_request_window_cluster16_out",
    REQUEST_MEMBER_PIPELINE_CV_OP})


def storage_reader(state, original):
    if state.cache_format == "striped_rotated_fp32_v3":
        from ..ops.cv_dispatch import OFFICIAL_CV_OPS
        if original not in OFFICIAL_CV_OPS:
            raise OscarReadinessError("rotated-window cache requires its official-domain reader")
        return original
    if state.cache_format == REQUEST_FORMAT:
        try:
            return REQUEST_READERS[original]
        except KeyError as error:
            raise OscarReadinessError("request-window format requires its complete reader bundle") from error
    if state.cache_format == "striped_fp32_v2":
        from ..ops.cv_dispatch import ALIGNED_CV_OPS
        if original not in ALIGNED_CV_OPS:
            raise OscarReadinessError("144-byte aligned cache requires its versioned reader")
        return original
    if state.cache_format not in ("striped_v1", "canonical_v1"):
        raise OscarReadinessError("unknown OSCAR storage format")
    return original


def request_map(state, metadata, first=0, count=None):
    if state.cache_format != REQUEST_FORMAT:
        return None
    value = getattr(metadata, "window_map", None)
    count = metadata.num_reqs - first if count is None else count
    if (type(first) is not int or type(count) is not int or min(first, count) < 0 or
            value is None or value.ndim != 2 or value.shape[1] != 2 or
            first + count > metadata.num_reqs or value.shape[0] < metadata.num_reqs):
        raise OscarReadinessError("request-window metadata lacks its stable slot/epoch mapping")
    return value[first:first + count]


def store_current(ops, state, key, value, slots, positions, starts, window_map, config, *, rotated_kv=None):
    n = key.shape[0]
    if state.cache_format == "striped_rotated_fp32_v3":
        if not config.get("experimental_official_math", False):
            raise OscarReadinessError("rotated-window writer selected without official math")
        from .official_math import store_rotated
        return store_rotated(ops,state,key,value,slots,positions,config,rotated_kv=rotated_kv)
    if state.cache_format == "striped_fp32_v2":
        if not config.get("experimental_aligned_math",False):
            raise OscarReadinessError("aligned writer selected without matching math profile")
        from .aligned_math import store_aligned
        return store_aligned(ops,state,key,value,slots,positions,config)
    prefix = (key, value, state.rotation_k_transpose, state.rotation_v_transpose,
              slots, positions)
    storage = (state.raw, state.window_key, state.window_value, state.window_tags,
               state.workspace.store_status[:n])
    geometry = (state.spec.block_size, state.num_blocks,
                state.num_blocks * state.spec.conv_bytes, state.spec.ssm_bytes,
                state.snapshots.sink_tokens, state.snapshots.ring_tokens,
                float(config.get("k_clip_ratio", 0.0)), float(config.get("v_clip_ratio", 0.0)),
                state.hadamard)
    if state.cache_format == REQUEST_FORMAT:
        if window_map is None or starts is None:
            raise OscarReadinessError("request-window writer needs qstarts and stable slot/epoch mapping")
        ops.rotate_clip_store_request_window_out(
            *prefix, starts, window_map, *storage, state.window_status, *geometry,
            state.snapshots.max_num_seqs, state.snapshots.position_bits)
        empty = state.workspace.store_status[:0]
        from .guard_diag import run_status_guard
        run_status_guard(ops, "request_window.store",
                         [("window", state.window_status), ("rotate", empty),
                          ("merge", empty), ("store", empty)])
    elif state.cache_format in ("striped_v1", "canonical_v1"):
        name = ("rotate_clip_store_striped_out" if state.cache_format == "striped_v1"
                else "rotate_clip_store_out")
        getattr(ops, name)(*prefix, *storage, *geometry)
    else:
        raise OscarReadinessError("unknown OSCAR storage format")
