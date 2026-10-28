"""Archive #145/#164/#173: native arithmetic for an already-proven fresh suffix.

The dispatcher owns the existing CPU-bound/device-prepare context=0 proof.
This module never selects requests, changes cache math, or inspects device
values on the host. Raw Q retains the existing device finite check. Native
FIA output is validated without a rotation roundtrip; the unchanged writer
subsequently centers/rotates K/V into the official INT2/snapshot format.
D.4: bounded current-step buffers only, no historical reconstruction. Reuse
the existing Q/output checks and buffers; remove Q and inverse matmuls only.
"""
from .runtime_api import OscarReadinessError


def verified_query(ops, state, query, slots):
    """Copy raw BF16 Q into separate full-input storage and retain its status."""
    workspace = state.workspace
    n, heads, dim = query.shape
    buffer = workspace.query_rot_bf16
    if (buffer is None or buffer.shape[0] < n
            or tuple(buffer.shape[1:]) != (heads, dim)):
        raise OscarReadinessError("native exact current requires full-input query scratch")
    verified = buffer[:n]
    # query may already be query_input after a strided native projection copy.
    # A distinct destination keeps the raw source readable until FIA completes.
    ops.copy_validate_bf16_query_out(query, slots, verified,
        workspace.rotate_status[:n], workspace.geometry.cube_cores)
    return verified


def finalize_output(ops, state, current, lse, output):
    """Use the existing finite/LSE copy without an inverse or new allocation."""
    workspace = state.workspace
    n, heads, dim = current.shape
    if (workspace.final_status is None or workspace.final_lse is None
            or workspace.query_input.shape[0] < n):
        raise OscarReadinessError("native exact current output buffers were not budgeted")
    # Native FIA's Q reads precede this same-stream reuse of query_input,
    # while K/V projection storage remains intact for the unchanged writer.
    canonical = workspace.query_input[:n]
    canonical.copy_(current)
    widened_lse = workspace.lse[:n]
    widened_lse.copy_(lse.squeeze(-1))
    ops.copy_validate_current_out(canonical, widened_lse.view(n, heads, 1),
        output.view(n, heads, dim), workspace.final_lse[:n],
        workspace.final_status[:n])


def guard_output(ops, state, tokens):
    """Keep final-output error priority without calling it an inverse failure."""
    from .guard_diag import run_status_guard
    status = state.workspace.final_status[:tokens]
    empty = status[:0]
    run_status_guard(ops, "current_only.native_output", [("native_output", status),
        ("empty", empty), ("empty", empty), ("empty", empty)])
