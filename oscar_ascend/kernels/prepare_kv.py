"""Prepare native attention buffers without full-prefix splice/cast/cat copies."""

import os
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

    @triton.jit
    def _prepare_side_batch(
        Cache8, Meta8, BlockTables, Prefixes, OutStarts,
        Stage, Owner, Out,
        stride_bt,
        stride_cb, stride_cp, stride_ch,
        stride_mb, stride_mp, stride_mh,
        BS: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
        BD: tl.constexpr, INDEX_OFFSET: tl.constexpr,
        META_OFFSET: tl.constexpr, HAS_STAGE: tl.constexpr,
        STAGE_ROWS: tl.constexpr, BT: tl.constexpr,
    ):
        request = tl.program_id(0)
        pos = tl.program_id(1) * BT + tl.arange(0, BT)
        head = tl.program_id(2)
        length = tl.load(Prefixes + request)
        valid = pos < length
        block = tl.load(
            BlockTables + request * stride_bt + pos // BS,
            mask=valid, other=0,
        ).to(tl.int64)
        offset = pos % BS
        slot = block * stride_cb + offset.to(tl.int64) * stride_cp + head * stride_ch
        meta = (
            block * stride_mb + offset.to(tl.int64) * stride_mp
            + head * stride_mh + META_OFFSET
        )
        dims = tl.arange(0, BD)
        mask = valid[:, None] & (dims[None, :] < D)
        byte = tl.load(
            Cache8 + slot[:, None] + INDEX_OFFSET + dims[None, :] // 4,
            mask=mask, other=0,
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
                mask=hit[:, None] & (dims[None, :] < D), other=0.0,
            )
            values = tl.where(hit[:, None], staged, values)
        values = values.to(tl.float16).to(tl.float32)
        out_start = tl.load(OutStarts + request)
        target = ((out_start + pos)[:, None] * HK + head) * D + dims[None, :]
        tl.store(Out + target, values, mask=mask)

    @triton.jit
    def _prepare_kv_batch(
        KCache8, VCache8, BlockTables, Prefixes, OutStarts,
        FreshK, FreshV, FreshStarts, StageK, StageV, Owner, KOut, VOut,
        stride_bt, stride_kb, stride_kp, stride_kh,
        stride_vb, stride_vp, stride_vh,
        BS: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
        BD: tl.constexpr, HAS_STAGE: tl.constexpr,
        HAS_FRESH_SOURCE: tl.constexpr, STAGE_ROWS: tl.constexpr,
        BT: tl.constexpr,
    ):
        request = tl.program_id(0)
        pos = tl.program_id(1) * BT + tl.arange(0, BT)
        head = tl.program_id(2)
        prefix = tl.load(Prefixes + request)
        out_start = tl.load(OutStarts + request)
        next_start = tl.load(OutStarts + request + 1)
        length = next_start - out_start
        valid = pos < length
        historical = valid & (pos < prefix)
        block = tl.load(
            BlockTables + request * stride_bt + pos // BS,
            mask=historical, other=0,
        ).to(tl.int64)
        offset = pos % BS
        kslot = block * stride_kb + offset.to(tl.int64) * stride_kp + head * stride_kh
        vslot = block * stride_vb + offset.to(tl.int64) * stride_vp + head * stride_vh
        dims = tl.arange(0, BD)
        dmask = dims < D
        cache_history = historical
        mask = cache_history[:, None] & dmask[None, :]
        byte_idx = dims[None, :] // 4
        shift = (dims[None, :] % 4) * 2
        kb = tl.load(KCache8 + kslot[:, None] + 32 + byte_idx,
                     mask=mask, other=0).to(tl.int32)
        vb = tl.load(VCache8 + vslot[:, None] + byte_idx,
                     mask=mask, other=0).to(tl.int32)
        kq = ((kb >> shift) & 3).to(tl.float32)
        vq = ((vb >> shift) & 3).to(tl.float32)

        ksl = tl.load(KCache8 + kslot, mask=cache_history, other=0).to(tl.uint16)
        ksh = tl.load(KCache8 + kslot + 1, mask=cache_history, other=0).to(tl.uint16)
        kzl = tl.load(KCache8 + kslot + 2, mask=cache_history, other=0).to(tl.uint16)
        kzh = tl.load(KCache8 + kslot + 3, mask=cache_history, other=0).to(tl.uint16)
        vsl = tl.load(KCache8 + kslot + 4, mask=cache_history, other=0).to(tl.uint16)
        vsh = tl.load(KCache8 + kslot + 5, mask=cache_history, other=0).to(tl.uint16)
        vzl = tl.load(KCache8 + kslot + 6, mask=cache_history, other=0).to(tl.uint16)
        vzh = tl.load(KCache8 + kslot + 7, mask=cache_history, other=0).to(tl.uint16)
        ks = (ksl | (ksh << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        kz = (kzl | (kzh << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        vs = (vsl | (vsh << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        vz = (vzl | (vzh << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        kval = kq * ks[:, None] + kz[:, None]
        vval = vq * vs[:, None] + vz[:, None]
        if HAS_STAGE:
            seat = (block % STAGE_ROWS) * BS + offset
            owner = tl.load(Owner + seat, mask=historical, other=-1)
            hit = historical & (owner == block)
            stage_base = (seat * HK + head) * D
            sk = tl.load(StageK + stage_base[:, None] + dims[None, :],
                         mask=hit[:, None] & dmask[None, :], other=0.0)
            sv = tl.load(StageV + stage_base[:, None] + dims[None, :],
                         mask=hit[:, None] & dmask[None, :], other=0.0)
            kval = tl.where(hit[:, None], sk, kval)
            vval = tl.where(hit[:, None], sv, vval)
        if HAS_FRESH_SOURCE:
            fresh = valid & ~historical
            fresh_start = tl.load(FreshStarts + request)
            fresh_base = ((fresh_start + pos - prefix) * HK + head) * D
            fk = tl.load(FreshK + fresh_base[:, None] + dims[None, :],
                         mask=fresh[:, None] & dmask[None, :], other=0.0)
            fv = tl.load(FreshV + fresh_base[:, None] + dims[None, :],
                         mask=fresh[:, None] & dmask[None, :], other=0.0)
            kval = tl.where(fresh[:, None], fk, kval)
            vval = tl.where(fresh[:, None], fv, vval)
        kval = kval.to(tl.float16).to(tl.float32)
        vval = vval.to(tl.float16).to(tl.float32)
        target = ((out_start + pos)[:, None] * HK + head) * D + dims[None, :]
        tl.store(KOut + target, kval, mask=valid[:, None] & dmask[None, :])
        tl.store(VOut + target, vval, mask=valid[:, None] & dmask[None, :])


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
    kc, vc, block_tables, prefixes, k_parts, v_parts, stage=None, *, use_triton=True,
    fresh_source=None, fresh_starts=None, prepared_metadata=None,
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
    if use_triton and triton is not None:
        starts = [0] + kv_ends[:-1]
        if prepared_metadata is None:
            prefix_tensor = torch.tensor(prefixes, dtype=torch.int32, device=device)
            boundary_tensor = torch.tensor(starts + [kv_ends[-1]], dtype=torch.int32, device=device)
        else:
            prefix_tensor, boundary_tensor, cached_fresh_tensor = prepared_metadata
        k8, v8 = kc.view(torch.uint8), vc.view(torch.uint8)
        sk, sv, owner = (k_all, v_all, prefix_tensor) if stage is None else stage
        stage_rows = 1 if stage is None else owner.shape[0]
        if fresh_source is None:
            fresh_k = torch.cat(k_parts, dim=0).contiguous()
            fresh_v = torch.cat(v_parts, dim=0).contiguous()
            fs, fresh_offset = [], 0
            for part in k_parts:
                fs.append(fresh_offset)
                fresh_offset += part.shape[0]
        else:
            fresh_k, fresh_v = fresh_source
            fs = fresh_starts
        fresh_tensor = (cached_fresh_tensor if prepared_metadata is not None else
                        torch.as_tensor(fs, dtype=torch.int32, device=device))
        bt = 32 if os.environ.get("OSCAR_ASCEND_PREP_BT", "16") == "32" else 16
        _prepare_kv_batch[(count, triton.cdiv(max(lengths), bt), hk)](
            k8, v8, block_tables, prefix_tensor, boundary_tensor,
            fresh_k, fresh_v, fresh_tensor, sk, sv, owner, k_all, v_all,
            block_tables.stride(0),
            k8.stride(0), k8.stride(1), k8.stride(2),
            v8.stride(0), v8.stride(1), v8.stride(2),
            BS=kc.shape[1], HK=hk, D=d, BD=triton.next_power_of_2(d),
            HAS_STAGE=stage is not None, HAS_FRESH_SOURCE=True,
            STAGE_ROWS=stage_rows, BT=bt, num_warps=1, num_stages=1,
        )
        return k_all, v_all, kv_ends
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
