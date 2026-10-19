"""Archive #111/#126/#145/#164: native BF16 rotation, bounded graph buffers.

Only current query/KV tensors are transformed. The original-space BF16 exact
windows remain separate; rotated Q and K/V stay BF16 through the reader ABI.
"""
from .runtime_api import OscarReadinessError
from ..diagnostic_timer import phase as timer_phase


def dequant_mode(metadata):
    # Captured target graphs are FULL_DECODE_ONLY; dummy labels may be prefill.
    if getattr(metadata, "capture_origin", False):
        return 1
    state = getattr(getattr(metadata, "attn_state", None), "name", None)
    if state in {"DecodeOnly", "SpecDecoding"}:
        return 1
    if state in {"PrefillNoCache", "PrefillCacheHit", "ChunkedPrefill"}:
        return 0
    raise OscarReadinessError(f"aligned dequant requires native prefill/decode classification, got {state!r}")


def _matrix(state, name, dim, torch):
    matrix = getattr(state, name, None)
    if (matrix is None or matrix.dtype != torch.bfloat16 or
            tuple(matrix.shape) != (dim, dim) or not matrix.is_contiguous()):
        raise OscarReadinessError(f"aligned math needs pre-profile natural BF16 {name}")
    return matrix


def rotate_query(ops, state, query, slots, output, status):
    import torch
    n, h, dim = query.shape
    buffer = getattr(state.workspace, "query_rot_bf16", None)
    if buffer is None or buffer.shape[0] < n:
        raise OscarReadinessError("aligned BF16 query rotation was not budgeted before KV profiling")
    target = buffer[:n]
    with timer_phase("q_rotation", tokens=n, heads=h, head_dim=dim):
        torch.matmul(query, _matrix(state, "rotation_k_bf16", dim, torch), out=target)
    with timer_phase("query_validation", tokens=n, heads=h, head_dim=dim):
        ops.copy_validate_bf16_query_out(target, slots, output, status, state.workspace.geometry.cube_cores)


def store_aligned(ops, state, key, value, slots, positions, config):
    import torch
    n, hk, dim = key.shape
    kbuf = getattr(state.workspace, "key_rot_bf16", None)
    vbuf = getattr(state.workspace, "value_rot_bf16", None)
    if kbuf is None or vbuf is None or min(kbuf.shape[0], vbuf.shape[0]) < n:
        raise OscarReadinessError("aligned BF16 K/V rotation buffers were not profiled")
    with timer_phase("kv_rotation", tokens=n, heads=hk, head_dim=dim):
        torch.matmul(key, _matrix(state, "rotation_k_bf16", dim, torch), out=kbuf[:n])
        torch.matmul(value, _matrix(state, "rotation_v_bf16", dim, torch), out=vbuf[:n])
    with timer_phase("int2_quantize_store", tokens=n, heads=hk, head_dim=dim):
        ops.store_aligned_int2_out(kbuf[:n], vbuf[:n], key, value, slots, positions,
            state.raw, state.window_key, state.window_value, state.window_tags,
            state.workspace.store_status[:n], state.spec.block_size, state.num_blocks,
            state.num_blocks * state.spec.conv_bytes, state.spec.ssm_bytes,
            state.snapshots.sink_tokens, state.snapshots.ring_tokens,
            float(config.get("k_clip_ratio", 0.96)), float(config.get("v_clip_ratio", 0.92)))
