"""Archive #13-20/#126/#145/#148/#164: one finite-precision rotation domain.

V4 keeps all sources in the same BF16 rotated frame, normalized partials in
FP32, and applies Rv.T once after their merge. Only pre-profile current-step
buffers are used; no history materialization or implicit CPU path exists.
"""
from .aligned_math import _matrix
from .runtime_api import OscarReadinessError

CACHE_FORMAT = "striped_rotated_fp32_v3"
CONTRACT = "oscar_bf16_attention_v4"


def _center_key(state, key, torch):
    """Archive G19-24/#86: subtract only this TP head's BF16 mean.

    V1 has no mean. V2 is validated before profiling and currently scoped to
    one local KV head; never broadcast one target head's mean across others.
    """
    mean = getattr(state, "key_mean_bf16", None)
    if mean is None:
        return key
    n, heads, dim = key.shape
    if (heads != 1 or mean.dtype != torch.bfloat16 or tuple(mean.shape) != (dim,)
            or mean.device != key.device or not mean.is_contiguous()):
        raise OscarReadinessError("perhead-center requires one local KV head with its own contiguous BF16 mean")
    buffer = getattr(state.workspace, "centered_key_bf16", None)
    if buffer is None or buffer.shape[0] < n or tuple(buffer.shape[1:]) != (heads, dim):
        raise OscarReadinessError("centered current-key scratch must be reserved before KV profiling")
    centered = buffer[:n]
    # Preserve the official visible BF16 subtraction before BF16 rotation.
    torch.sub(key, mean.view(1, 1, dim), out=centered)
    return centered


def rotate_inputs(ops, state, query, key, value, slots, *, current_only=False):
    import torch
    workspace = state.workspace
    n, h, dim = query.shape
    qbuf, kbuf, vbuf = (workspace.query_rot_bf16, workspace.key_rot_bf16,
                        workspace.value_rot_bf16)
    if any(buffer is None or buffer.shape[0] < n for buffer in (qbuf, kbuf, vbuf)):
        raise OscarReadinessError("official rotation requires full current-step buffers before profiling")
    qr = workspace.query_input[:n] if current_only else workspace.query_rot[:n]
    if qr.shape != query.shape:
        raise OscarReadinessError("official rotated query capacity differs")
    torch.matmul(query, _matrix(state, "rotation_k_bf16", dim, torch), out=qbuf[:n])
    ops.copy_validate_bf16_query_out(qbuf[:n], slots, qr, workspace.rotate_status[:n],
                                     workspace.geometry.cube_cores)
    torch.matmul(_center_key(state, key, torch), _matrix(state, "rotation_k_bf16", dim, torch), out=kbuf[:n])
    torch.matmul(value, _matrix(state, "rotation_v_bf16", dim, torch), out=vbuf[:n])
    return qr, kbuf[:n], vbuf[:n]


def store_rotated(ops, state, key, value, slots, positions, config, *, rotated_kv=None):
    import torch
    n, _, dim = key.shape
    if state.cache_format != CACHE_FORMAT:
        raise OscarReadinessError("official writer requires rotated-window cache format")
    if rotated_kv is None:
        kr, vr = state.workspace.key_rot_bf16[:n], state.workspace.value_rot_bf16[:n]
        torch.matmul(_center_key(state, key, torch), _matrix(state, "rotation_k_bf16", dim, torch), out=kr)
        torch.matmul(value, _matrix(state, "rotation_v_bf16", dim, torch), out=vr)
    else:
        kr, vr = rotated_kv
    # The last K/V pair supplies the exact window, before clipping. Using
    # original-space inputs here would reintroduce the cross-source bias.
    # This immutable host scalar was selected from the authenticated global
    # head table before profiling; MTP/old recipes keep their config scalar.
    k_clip = getattr(state, "key_clip_ratio", None)
    if k_clip is None:
        k_clip = float(config.get("k_clip_ratio", .96))
    ops.store_aligned_int2_out(kr, vr, kr, vr, slots, positions,
        state.raw, state.window_key, state.window_value, state.window_tags,
        state.workspace.store_status[:n], state.spec.block_size, state.num_blocks,
        state.num_blocks * state.spec.conv_bytes, state.spec.ssm_bytes,
        state.snapshots.sink_tokens, state.snapshots.ring_tokens,
        k_clip, float(config.get("v_clip_ratio", .92)))


def finalize_output(ops, state, rotated_output, lse, output):
    import torch
    n, h, dim = rotated_output.shape
    workspace = state.workspace
    if workspace.final_status is None or workspace.query_rot_bf16.shape[0] < n:
        raise OscarReadinessError("official final inverse buffers were not budgeted")
    # Q and all attention reads have finished. Reuse the current Q-copy arena
    # for the sole post-merge BF16 round, without aliasing the FP32 input.
    rounded = workspace.query_input[:n]
    rounded.copy_(rotated_output)
    inverse = workspace.query_rot_bf16[:n]
    torch.matmul(rounded, _matrix(state, "rotation_v_bf16", dim, torch).T, out=inverse)
    ops.copy_validate_current_out(inverse, lse.view(n, h, 1), output.view(n, h, dim),
                                  workspace.final_lse[:n], workspace.final_status[:n])


def guard_final_output(ops, state, tokens, *, capture=False):
    # Preserve #157's original attention/rotation/merge/store error priority.
    # A propagated metadata NaN must not be relabelled as the final inverse's
    # first failure. The caller guards the original families before this one.
    from .guard_diag import run_status_guard
    status = state.workspace.final_status[:tokens]
    empty = status[:0]
    run_status_guard(ops, "official.final_inverse", [("final_inverse",status),
        ("empty",empty),("empty",empty),("empty",empty)], capture=capture)
