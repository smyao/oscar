"""Prepare native attention buffers without full-prefix splice/cast/cat copies."""

import torch

from ..format import K_IDX_OFF
from .decode_kernel import oscar_full_dequant_ref, tl, triton

if triton is not None:

    @triton.jit
    def _prepare_side(
        Cache8,
        Meta8,
        BlockTable,
        Stage,
        Owner,
        Out,
        length,
        stride_cb,
        stride_cp,
        stride_ch,
        stride_mb,
        stride_mp,
        stride_mh,
        BS: tl.constexpr,
        HK: tl.constexpr,
        D: tl.constexpr,
        BD: tl.constexpr,
        INDEX_OFFSET: tl.constexpr,
        META_OFFSET: tl.constexpr,
        HAS_STAGE: tl.constexpr,
        STAGE_ROWS: tl.constexpr,
        BT: tl.constexpr,
    ):
        pos = tl.program_id(0) * BT + tl.arange(0, BT)
        head = tl.program_id(1)
        valid = pos < length
        block = tl.load(BlockTable + pos // BS, mask=valid, other=0).to(tl.int64)
        offset = pos % BS
        slot = block * stride_cb + offset.to(tl.int64) * stride_cp + head * stride_ch
        meta = (
            block * stride_mb
            + offset.to(tl.int64) * stride_mp
            + head * stride_mh
            + META_OFFSET
        )
        dims = tl.arange(0, BD)
        mask = valid[:, None] & (dims[None, :] < D)
        byte = tl.load(
            Cache8 + slot[:, None] + INDEX_OFFSET + dims[None, :] // 4,
            mask=mask,
            other=0,
        ).to(tl.int32)
        quant = ((byte >> ((dims[None, :] % 4) * 2)) & 3).to(tl.float32)
        lo = tl.load(Meta8 + meta, mask=valid, other=0).to(tl.uint16)
        hi = tl.load(Meta8 + meta + 1, mask=valid, other=0).to(tl.uint16)
        scale = (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        lo = tl.load(Meta8 + meta + 2, mask=valid, other=0).to(tl.uint16)
        hi = tl.load(Meta8 + meta + 3, mask=valid, other=0).to(tl.uint16)
        zero = (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        values = quant * scale[:, None] + zero[:, None]
        if HAS_STAGE:
            seat = (block % STAGE_ROWS) * BS + offset
            owner = tl.load(Owner + seat, mask=valid, other=-1)
            hit = valid & (owner == block)
            stage_offset = (seat * HK + head) * D
            staged = tl.load(
                Stage + stage_offset[:, None] + dims[None, :],
                mask=hit[:, None] & (dims[None, :] < D),
                other=0.0,
            )
            values = tl.where(hit[:, None], staged, values)
        # Preserve the previous dequant/splice fp16 rounding before the native
        # bf16/fp16 cast. K and V are separate launches to bound live UB data.
        values = values.to(tl.float16).to(tl.float32)
        target = (pos[:, None] * HK + head) * D + dims[None, :]
        tl.store(Out + target, values, mask=mask)


def prepare_native_kv(
    kc, vc, bt, prefix, k_new, v_new, stage=None, *, use_triton=True, out=None
):
    """Return contiguous [prefix + new, Hk, D] buffers in new K/V dtype."""
    if prefix < 0 or k_new.shape != v_new.shape or k_new.dtype != v_new.dtype:
        raise ValueError("Invalid native KV preparation inputs")
    expected_shape = (prefix + k_new.shape[0], *k_new.shape[1:])
    if out is not None:
        k_out, v_out = out
        if (k_out.shape != expected_shape or v_out.shape != expected_shape
                or k_out.dtype != k_new.dtype or v_out.dtype != k_new.dtype
                or k_out.device != k_new.device or v_out.device != k_new.device
                or not k_out.is_contiguous() or not v_out.is_contiguous()):
            raise ValueError("Invalid native KV output buffers")
    if prefix == 0:
        if out is None:
            return k_new.contiguous(), v_new.contiguous()
        k_out, v_out = out
        k_out.copy_(k_new)
        v_out.copy_(v_new)
        return k_out, v_out
    n, hk, d = k_new.shape
    bs = kc.shape[1]
    if not use_triton:
        k, v = oscar_full_dequant_ref(kc, vc, bt, prefix, hk, d)
        if stage is not None:
            sk, sv, owner = stage
            pos = torch.arange(prefix, device=kc.device)
            blocks = bt[pos // bs].long()
            rows, offsets = blocks % owner.shape[0], pos % bs
            hit = (owner[rows, offsets] == blocks).view(-1, 1, 1)
            k = torch.where(hit, sk[rows, offsets], k)
            v = torch.where(hit, sv[rows, offsets], v)
        if out is None:
            k_out = torch.empty(prefix + n, hk, d, dtype=k_new.dtype, device=kc.device)
            v_out = torch.empty_like(k_out)
        else:
            k_out, v_out = out
        k_out[:prefix].copy_(k.half().to(k_new.dtype))
        v_out[:prefix].copy_(v.half().to(v_new.dtype))
        k_out[prefix:].copy_(k_new)
        v_out[prefix:].copy_(v_new)
        return k_out, v_out
    if triton is None:
        raise RuntimeError("Triton unavailable for fused KV preparation")
    if out is None:
        k_out = torch.empty(prefix + n, hk, d, dtype=k_new.dtype, device=kc.device)
        v_out = torch.empty_like(k_out)
    else:
        k_out, v_out = out
    k8, v8 = kc.view(torch.uint8), vc.view(torch.uint8)
    sk, sv, owner = (k_out, v_out, bt) if stage is None else stage
    rows = 1 if stage is None else owner.shape[0]
    for source, staged, out, index_offset, meta_offset in (
        (k8, sk, k_out, K_IDX_OFF, 0),
        (v8, sv, v_out, 0, 4),
    ):
        _prepare_side[(triton.cdiv(prefix, 4), hk)](
            source,
            k8,
            bt,
            staged,
            owner,
            out,
            prefix,
            source.stride(0),
            source.stride(1),
            source.stride(2),
            k8.stride(0),
            k8.stride(1),
            k8.stride(2),
            BS=bs,
            HK=hk,
            D=d,
            BD=triton.next_power_of_2(d),
            INDEX_OFFSET=index_offset,
            META_OFFSET=meta_offset,
            HAS_STAGE=stage is not None,
            STAGE_ROWS=rows,
            BT=4,
            num_warps=1,
            num_stages=1,
        )
    # Only copy the new chunk (usually 4 tokens), not the whole history.
    k_out[prefix:].copy_(k_new)
    v_out[prefix:].copy_(v_new)
    return k_out, v_out


def prepare_native_kv_batch(
    kc, vc, block_tables, prefixes, k_parts, v_parts, stage=None, *, use_triton=True
):
    """Prepare a packed TND KV buffer without per-request concat buffers.

    Each request owns one contiguous range in the returned tensors. Historical
    INT2 values are decoded directly into that final range and current K/V are
    copied only once into its tail. ``kv_ends`` are cumulative TND boundaries.
    """
    count = len(prefixes)
    if not (count and len(k_parts) == count and len(v_parts) == count):
        raise ValueError("Invalid OSCAR KV batch parts")
    if block_tables.shape[0] != count:
        raise ValueError("Block-table rows must match OSCAR KV batch size")
    first = k_parts[0]
    if first.ndim != 3:
        raise ValueError("OSCAR KV parts must have shape [tokens, heads, dim]")
    hk, d, dtype, device = first.shape[1], first.shape[2], first.dtype, first.device
    lengths, kv_ends = [], []
    for prefix, k_new, v_new in zip(prefixes, k_parts, v_parts):
        prefix = int(prefix)
        if prefix < 0 or k_new.shape != v_new.shape:
            raise ValueError("Invalid OSCAR KV batch request")
        if (k_new.ndim != 3 or k_new.shape[1:] != (hk, d)
                or k_new.dtype != dtype or k_new.device != device
                or v_new.dtype != dtype or v_new.device != device):
            raise ValueError("OSCAR KV batch parts must share shape, dtype and device")
        length = prefix + k_new.shape[0]
        lengths.append(length)
        kv_ends.append((kv_ends[-1] if kv_ends else 0) + length)
    k_all = torch.empty(kv_ends[-1], hk, d, dtype=dtype, device=device)
    v_all = torch.empty_like(k_all)
    start = 0
    for i, (prefix, k_new, v_new, length) in enumerate(
        zip(prefixes, k_parts, v_parts, lengths)
    ):
        # Slices are contiguous and point into the final FIA input allocation.
        k_dst = k_all[start:start + length]
        v_dst = v_all[start:start + length]
        if prefix:
            prepare_native_kv(
                kc, vc, block_tables[i], int(prefix), k_new, v_new, stage,
                use_triton=use_triton, out=(k_dst, v_dst),
            )
        else:
            k_dst.copy_(k_new)
            v_dst.copy_(v_new)
        start += length
    return k_all, v_all, kv_ends
