"""Optional AscendC INT2 paged-attention ABI.

The binary operator is deliberately discovered at runtime.  Source installs
without a CANN-built operator retain the validated Triton/native-FIA routes.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import torch


_OP_NAMESPACE = "oscar_ascend"
_OP_NAME = "int2_paged_attention"


def ascendc_mode() -> str:
    mode = os.environ.get("OSCAR_ASCEND_USE_ASCENDC", "0").strip().lower()
    if mode not in ("0", "1", "required"):
        raise ValueError(
            "OSCAR_ASCEND_USE_ASCENDC must be 0, 1 or required"
        )
    return mode


def ascendc_required() -> bool:
    return ascendc_mode() == "required"


@lru_cache(maxsize=1)
def _load_library() -> None:
    path = os.environ.get("OSCAR_ASCEND_ASCENDC_LIBRARY", "").strip()
    if not path:
        return
    library = Path(path)
    if not library.is_file():
        if ascendc_mode() == "required":
            raise RuntimeError(f"AscendC OSCAR library does not exist: {library}")
        return
    torch.ops.load_library(str(library))


def _resolve_op():
    _load_library()
    namespace = getattr(torch.ops, _OP_NAMESPACE, None)
    return None if namespace is None else getattr(namespace, _OP_NAME, None)


def ascendc_available() -> bool:
    return _resolve_op() is not None


def ascendc_enabled() -> bool:
    mode = ascendc_mode()
    available = ascendc_available()
    if mode == "required" and not available:
        raise RuntimeError(
            "AscendC OSCAR operator is required but torch.ops.oscar_ascend."
            "int2_paged_attention is not registered"
        )
    return available and mode in ("1", "required")


def oscar_ascendc_attention(
    q_rot: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    q_starts: torch.Tensor,
    q_lens: torch.Tensor,
    prefixes: torch.Tensor,
    scale: float,
    *,
    stage: Sequence[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Invoke the fused INT2-unpack + Cube attention custom operator."""
    op = _resolve_op()
    if op is None:
        raise RuntimeError("AscendC OSCAR attention operator is unavailable")
    if q_rot.device.type != "npu" or q_rot.ndim != 3 or q_rot.shape[-1] != 256:
        raise ValueError("AscendC OSCAR attention requires NPU [N,Hq,256] Q")
    n, hq, d = q_rot.shape
    if (k_new.shape != v_new.shape or k_new.ndim != 3
            or k_new.shape[0] != n or k_new.shape[-1] != d):
        raise ValueError("Invalid AscendC OSCAR fresh K/V")
    hk = k_new.shape[1]
    if hk <= 0 or hq % hk:
        raise ValueError("AscendC OSCAR requires integral GQA groups")
    if (q_starts.dtype != torch.int32 or q_lens.dtype != torch.int32
            or prefixes.dtype != torch.int32
            or q_starts.shape != q_lens.shape
            or q_starts.shape != prefixes.shape):
        raise ValueError("Invalid AscendC OSCAR request metadata")
    if block_tables.dtype != torch.int32 or block_tables.shape[0] != q_lens.numel():
        raise ValueError("Invalid AscendC OSCAR block tables")
    tensors = (q_rot, k_new, v_new, k_cache, v_cache, block_tables,
               q_starts, q_lens, prefixes)
    if any(t.device != q_rot.device or not t.is_contiguous() for t in tensors):
        raise ValueError("AscendC OSCAR inputs must be contiguous on one NPU")
    if stage is None:
        stage_k = torch.empty(0, device=q_rot.device, dtype=torch.float32)
        stage_v = stage_k
        owner = torch.empty(0, device=q_rot.device, dtype=torch.int64)
    else:
        if len(stage) != 3:
            raise ValueError("AscendC OSCAR staging requires K/V/owner")
        stage_k, stage_v, owner = stage
        if (stage_k.shape != stage_v.shape or owner.shape != stage_k.shape[:2]
                or stage_k.device != q_rot.device or stage_v.device != q_rot.device
                or owner.device != q_rot.device):
            raise ValueError("Invalid AscendC OSCAR staging tensors")
    out = op(
        q_rot.contiguous(), k_new.contiguous(), v_new.contiguous(),
        k_cache, v_cache, block_tables, q_starts, q_lens, prefixes,
        stage_k, stage_v, owner, float(scale), int(hk), int(d),
    )
    if out.shape != q_rot.shape or out.device != q_rot.device:
        raise RuntimeError("AscendC OSCAR operator returned an invalid output")
    return out
