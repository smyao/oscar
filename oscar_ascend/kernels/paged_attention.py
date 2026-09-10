"""Paged OSCAR attention prototype in rotated space.

One query/head/split per program. Reads packed history in registers and raw
current/staged K/V, applies a per-query causal bound, and shares the decode
split reducer. The production backend deliberately uses this only for q_len=1:
multi-query requests would reread history per row until a request/KV-head tiled
replacement is available. No dense historical K/V allocation is performed.
"""

from __future__ import annotations

import itertools
import os

import torch

from ..format import K_IDX_OFF
from .decode_kernel import oscar_prefill_ref, tl, triton
from .store_kernel import dequant_split_ref


def query_layout(qsl, seqs, device):
    lengths = [b - a for a, b in itertools.pairwise(qsl)]
    if len(seqs) != len(lengths) or any(n < 0 or s < n for n, s in zip(lengths, seqs)):
        raise ValueError("Invalid OSCAR query/sequence lengths")
    req = torch.tensor(
        [i for i, n in enumerate(lengths) for _ in range(n)],
        device=device,
        dtype=torch.int32,
    )
    prefix = torch.tensor(
        [s - n for s, n in zip(seqs, lengths)], device=device, dtype=torch.int32
    )
    starts = torch.tensor(qsl[:-1], device=device, dtype=torch.int32)
    ends = torch.tensor(
        [s - n + j + 1 for s, n in zip(seqs, lengths) for j in range(n)],
        device=device,
        dtype=torch.int32,
    )
    return req, prefix, starts, ends


def oscar_paged_attention_ref(q, k, v, kc, vc, bt, qsl, seqs, scale, stage=None):
    """CPU/NPU oracle: same causal/current/staging semantics as the fused path."""
    output = torch.zeros_like(q)
    hk, d = k.shape[1:]
    bs = kc.shape[1]
    for i, (a, b) in enumerate(itertools.pairwise(qsl)):
        if a == b:
            continue
        count = seqs[i] - (b - a)
        pos = torch.arange(count, device=q.device)
        blocks = bt[i, pos // bs].long()
        oldk, oldv = dequant_split_ref(
            kc.view(torch.uint8), vc.view(torch.uint8), blocks, pos % bs, hk, d
        )
        if stage is not None:
            sk, sv, owner = stage
            rows = blocks % owner.shape[0]
            mask = (owner[rows, pos % bs] == blocks).view(-1, 1, 1)
            oldk = torch.where(mask, sk[rows, pos % bs], oldk)
            oldv = torch.where(mask, sv[rows, pos % bs], oldv)
        output[a:b] = oscar_prefill_ref(
            q[a:b], k[a:b], v[a:b], oldk, oldv, scale, hk, d
        )
    return output


if triton is not None:

    @triton.jit
    def _oscar_grouped_mtp_stage1(
        QStarts, QLens, Prefixes, KNew, VNew,
        StageK, StageV, Owner, QRot, KCache8, VCache8, BlockTables, Mid,
        stride_qh, stride_bt, stride_kb, stride_kp, stride_kh,
        stride_vb, stride_vp, stride_vh,
        stride_mb, stride_mh, stride_ms,
        NUM_BLOCKS: tl.constexpr, NUM_BT_BLOCKS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr, NUM_SPLITS: tl.constexpr,
        KV_GROUP_SIZE: tl.constexpr, GROUP_TILES: tl.constexpr,
        BLOCK_G: tl.constexpr, BLOCK_Q: tl.constexpr,
        BLOCK_D: tl.constexpr, BLOCK_KV: tl.constexpr, ATTN_SCALE: tl.constexpr,
        K_IDX_OFF: tl.constexpr, HAS_STAGE: tl.constexpr,
        STAGE_ROWS: tl.constexpr,
    ):
        req = tl.program_id(0)
        grouped_head = tl.program_id(1)
        split = tl.program_id(2)
        kv_head = grouped_head // GROUP_TILES
        group_tile = grouped_head % GROUP_TILES
        q_start = tl.load(QStarts + req)
        q_len = tl.load(QLens + req)
        prefix = tl.load(Prefixes + req)
        seq_len = prefix + q_len
        split_len = tl.cdiv(seq_len, NUM_SPLITS)
        split_start = split * split_len
        split_end = tl.minimum(split_start + split_len, seq_len)
        if split_start >= split_end:
            return

        lanes = tl.arange(0, BLOCK_Q * BLOCK_G)
        qi = lanes // BLOCK_G
        gi = lanes % BLOCK_G
        group = group_tile * BLOCK_G + gi
        lane_valid = (qi < q_len) & (group < KV_GROUP_SIZE)
        q_head = kv_head * KV_GROUP_SIZE + group
        safe_qi = tl.where(lane_valid, qi, 0)
        safe_q_head = tl.where(lane_valid, q_head, kv_head * KV_GROUP_SIZE)
        dims = tl.arange(0, BLOCK_D)
        dmask = dims < HEAD_DIM
        q_base = (q_start + safe_qi)[:, None] * (NUM_KV_HEADS * KV_GROUP_SIZE * HEAD_DIM)
        q_base += safe_q_head[:, None] * stride_qh
        qv = tl.load(QRot + q_base + dims[None, :],
                     mask=lane_valid[:, None] & dmask[None, :], other=0.0).to(tl.float32)
        m = tl.full([BLOCK_Q * BLOCK_G], -float("inf"), tl.float32)
        denom = tl.zeros([BLOCK_Q * BLOCK_G], tl.float32)
        acc = tl.zeros([BLOCK_Q * BLOCK_G, BLOCK_D], tl.float32)
        byte_idx = dims // 4
        shift = (dims % 4) * 2
        bt_base = req * stride_bt

        kv_range = tl.arange(0, BLOCK_KV)
        for kv_start in range(split_start, split_end, BLOCK_KV):
            kv_pos = kv_start + kv_range
            kv_valid = kv_pos < split_end
            causal = lane_valid[:, None] & kv_valid[None, :] & (
                kv_pos[None, :] < prefix + qi[:, None] + 1
            )
            history = kv_valid & (kv_pos < prefix)
            page = kv_pos // BLOCK_SIZE
            page_valid = history & (page < NUM_BT_BLOCKS)
            safe_page = tl.where(page_valid, page, 0)
            block = tl.load(BlockTables + bt_base + safe_page,
                            mask=page_valid, other=0).to(tl.int64)
            cache_valid = page_valid & (block >= 0) & (block < NUM_BLOCKS)
            safe_block = tl.where(cache_valid, block, 0)
            off = tl.where(kv_valid, kv_pos % BLOCK_SIZE, 0)
            kslot = safe_block * stride_kb + off * stride_kp + kv_head * stride_kh
            vslot = safe_block * stride_vb + off * stride_vp + kv_head * stride_vh
            kb = tl.load(KCache8 + kslot[:, None] + K_IDX_OFF + byte_idx[None, :],
                         mask=cache_valid[:, None] & dmask[None, :], other=0).to(tl.int32)
            vb = tl.load(VCache8 + vslot[:, None] + byte_idx[None, :],
                         mask=cache_valid[:, None] & dmask[None, :], other=0).to(tl.int32)
            kq = ((kb >> shift[None, :]) & 3).to(tl.float32)
            vq = ((vb >> shift[None, :]) & 3).to(tl.float32)
            ksl = tl.load(KCache8 + kslot, mask=cache_valid, other=0).to(tl.uint16)
            ksh = tl.load(KCache8 + kslot + 1, mask=cache_valid, other=0).to(tl.uint16)
            kzl = tl.load(KCache8 + kslot + 2, mask=cache_valid, other=0).to(tl.uint16)
            kzh = tl.load(KCache8 + kslot + 3, mask=cache_valid, other=0).to(tl.uint16)
            vsl = tl.load(KCache8 + kslot + 4, mask=cache_valid, other=0).to(tl.uint16)
            vsh = tl.load(KCache8 + kslot + 5, mask=cache_valid, other=0).to(tl.uint16)
            vzl = tl.load(KCache8 + kslot + 6, mask=cache_valid, other=0).to(tl.uint16)
            vzh = tl.load(KCache8 + kslot + 7, mask=cache_valid, other=0).to(tl.uint16)
            ks = (ksl | (ksh << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            kz = (kzl | (kzh << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            vs = (vsl | (vsh << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            vz = (vzl | (vzh << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            kval = kq * ks[:, None] + kz[:, None]
            vval = vq * vs[:, None] + vz[:, None]
            if HAS_STAGE:
                seat = (safe_block % STAGE_ROWS) * BLOCK_SIZE + off
                tag = tl.load(Owner + seat, mask=cache_valid, other=-1)
                hit = cache_valid & (tag == safe_block)
                stage_base = (seat * NUM_KV_HEADS + kv_head) * HEAD_DIM
                sk = tl.load(StageK + stage_base[:, None] + dims[None, :],
                             mask=hit[:, None] & dmask[None, :], other=0.0)
                sv = tl.load(StageV + stage_base[:, None] + dims[None, :],
                             mask=hit[:, None] & dmask[None, :], other=0.0)
                kval = tl.where(hit[:, None], sk, kval)
                vval = tl.where(hit[:, None], sv, vval)
            fresh = kv_valid & (kv_pos >= prefix)
            fresh_idx = q_start + kv_pos - prefix
            safe_fresh = tl.where(fresh, fresh_idx, 0)
            fresh_base = (safe_fresh * NUM_KV_HEADS + kv_head) * HEAD_DIM
            fk = tl.load(KNew + fresh_base[:, None] + dims[None, :],
                         mask=fresh[:, None] & dmask[None, :], other=0.0)
            fv = tl.load(VNew + fresh_base[:, None] + dims[None, :],
                         mask=fresh[:, None] & dmask[None, :], other=0.0)
            kval = tl.where(fresh[:, None], fk, kval)
            vval = tl.where(fresh[:, None], fv, vval)

            score = tl.sum(
                qv[:, None, :] * kval[None, :, :], axis=2
            ) * ATTN_SCALE
            # `tl.where` is not lazy. Feeding -inf operands to expressions in
            # its unselected branch still creates -inf--inf NaNs on Ascend and
            # can contaminate neighbouring vector lanes. Use finite operands
            # for non-causal lanes before exp, then select the state update.
            has_causal = tl.sum(causal.to(tl.int32), axis=1) > 0
            block_max = tl.max(
                tl.where(causal, score, -float("inf")), axis=1
            )
            safe_m = tl.where(has_causal, m, 0.0)
            safe_block_max = tl.where(has_causal, block_max, 0.0)
            candidate_m = tl.maximum(safe_m, safe_block_max)
            candidate_old_scale = tl.exp(safe_m - candidate_m)
            prob = tl.where(
                causal, tl.exp(score - candidate_m[:, None]), 0.0
            )
            new_m = tl.where(has_causal, candidate_m, m)
            old_scale = tl.where(has_causal, candidate_old_scale, 1.0)
            acc = acc * old_scale[:, None] + tl.sum(
                prob[:, :, None] * vval[None, :, :], axis=1
            )
            denom = denom * old_scale + tl.sum(prob, axis=1)
            m = new_m

        safe_denom = tl.where(denom > 0.0, denom, 1.0)
        out_base = ((q_start + safe_qi) * stride_mb + safe_q_head * stride_mh
                    + split * stride_ms)
        tl.store(Mid + out_base[:, None] + dims[None, :],
                 acc / safe_denom[:, None],
                 mask=lane_valid[:, None] & dmask[None, :])
        tl.store(Mid + out_base + HEAD_DIM, m + tl.log(safe_denom), mask=lane_valid)

    @triton.jit
    def _oscar_paged_stage1(
        TokenReq_ptr,
        Prefix_ptr,
        QStart_ptr,
        KNew_ptr,
        VNew_ptr,
        StageK_ptr,
        StageV_ptr,
        Owner_ptr,
        STAGE_ROWS: tl.constexpr,
        HAS_STAGE: tl.constexpr,
        Q_rot_ptr,  # [B, Hq, D] fp32
        KCache8_ptr,
        VCache8_ptr,  # flat uint8
        BlockTable_ptr,  # [B, T] int32
        SeqLens_ptr,  # [B] int32
        Mid_o_ptr,  # [B, Hq, NUM_SPLITS, D+1] fp32
        stride_qb,
        stride_qh,
        stride_kb,
        stride_kp,
        stride_kh,
        stride_vb,
        stride_vp,
        stride_vh,
        stride_bt_b,
        stride_mid_b,
        stride_mid_h,
        stride_mid_s,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        NUM_KV_SPLITS: tl.constexpr,
        KV_GROUP_SIZE: tl.constexpr,
        DATA_BYTES: tl.constexpr,
        ATTN_SCALE: tl.constexpr,
        K_IDX_OFF: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_KV: tl.constexpr,
    ):
        bid = tl.program_id(0)
        hid = tl.program_id(1)
        sid = tl.program_id(2)
        kv_head = hid // KV_GROUP_SIZE
        req = tl.load(TokenReq_ptr + bid)
        prefix_len = tl.load(Prefix_ptr + req)
        query_start = tl.load(QStart_ptr + req)
        seq_len = prefix_len + bid - query_start + 1
        split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
        split_start = split_len * sid
        split_end = tl.minimum(split_start + split_len, seq_len)
        if split_start >= split_end:
            return
        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < HEAD_DIM
        kv_range = tl.arange(0, BLOCK_KV)
        byte_idx = d_offs // 4
        bit_shift = (d_offs % 4) * 2
        q_base = bid * stride_qb + hid * stride_qh
        q_rot = tl.load(Q_rot_ptr + q_base + d_offs, mask=d_mask, other=0.0).to(
            tl.float32
        )
        m_prev = -float("inf")
        l_prev = 0.0
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        bt_base = req * stride_bt_b
        for start_n in range(split_start, split_end, BLOCK_KV):
            kv_offs = start_n + kv_range
            kv_mask = kv_offs < split_end
            cache_mask = kv_mask & (kv_offs < prefix_len)
            page_idx = kv_offs // BLOCK_SIZE
            page_off = kv_offs % BLOCK_SIZE
            block_nums = tl.load(
                BlockTable_ptr + bt_base + page_idx, mask=cache_mask, other=0
            ).to(tl.int64)
            k_slot = (
                block_nums * stride_kb
                + page_off.to(tl.int64) * stride_kp
                + tl.cast(kv_head, tl.int64) * stride_kh
            )
            v_slot = (
                block_nums * stride_vb
                + page_off.to(tl.int64) * stride_vp
                + tl.cast(kv_head, tl.int64) * stride_vh
            )
            # ---- K: meta[0..7] + idx[32..32+D/4] ----
            k_byte = tl.load(
                KCache8_ptr + k_slot[:, None] + (K_IDX_OFF + byte_idx[None, :]),
                mask=cache_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            q_k = ((k_byte >> bit_shift[None, :]) & 3).to(tl.float32)
            k_meta_base = k_slot
            ksc_lo = tl.load(KCache8_ptr + k_meta_base, mask=cache_mask, other=0).to(
                tl.uint16
            )
            ksc_hi = tl.load(
                KCache8_ptr + k_meta_base + 1, mask=cache_mask, other=0
            ).to(tl.uint16)
            k_scale = (
                (ksc_lo | (ksc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            )
            kzr_lo = tl.load(
                KCache8_ptr + k_meta_base + 2, mask=cache_mask, other=0
            ).to(tl.uint16)
            kzr_hi = tl.load(
                KCache8_ptr + k_meta_base + 3, mask=cache_mask, other=0
            ).to(tl.uint16)
            k_zero = (
                (kzr_lo | (kzr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            )
            k_deq = q_k * k_scale[:, None] + k_zero[:, None]
            # ---- V: idx[0..D/4]（meta 在 K 槽 +4/+6）----
            v_byte = tl.load(
                VCache8_ptr + v_slot[:, None] + byte_idx[None, :],
                mask=cache_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            q_v = ((v_byte >> bit_shift[None, :]) & 3).to(tl.float32)
            vsc_lo = tl.load(
                KCache8_ptr + k_meta_base + 4, mask=cache_mask, other=0
            ).to(tl.uint16)
            vsc_hi = tl.load(
                KCache8_ptr + k_meta_base + 5, mask=cache_mask, other=0
            ).to(tl.uint16)
            v_scale = (
                (vsc_lo | (vsc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            )
            vzr_lo = tl.load(
                KCache8_ptr + k_meta_base + 6, mask=cache_mask, other=0
            ).to(tl.uint16)
            vzr_hi = tl.load(
                KCache8_ptr + k_meta_base + 7, mask=cache_mask, other=0
            ).to(tl.uint16)
            v_zero = (
                (vzr_lo | (vzr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            )
            values = q_v * v_scale[:, None] + v_zero[:, None]
            # Staging holds unquantized fp32 rotated K/V. Owner tags prevent
            # reading a different physical block after hash collisions.
            if HAS_STAGE:
                row = block_nums % STAGE_ROWS
                seat = row * BLOCK_SIZE + page_off
                tag = tl.load(Owner_ptr + seat, mask=cache_mask, other=-1)
                staged = cache_mask & (tag == block_nums)
                stage_base = (seat * NUM_KV_HEADS + kv_head) * HEAD_DIM
                sk = tl.load(
                    StageK_ptr + stage_base[:, None] + d_offs[None, :],
                    mask=staged[:, None] & d_mask[None, :],
                    other=0,
                )
                sv = tl.load(
                    StageV_ptr + stage_base[:, None] + d_offs[None, :],
                    mask=staged[:, None] & d_mask[None, :],
                    other=0,
                )
                k_deq = tl.where(staged[:, None], sk, k_deq)
                values = tl.where(staged[:, None], sv, values)
            # Current query chunk always uses unquantized K/V, even if it is
            # larger than the staging arena or hashes to the same staging row.
            fresh = kv_mask & (kv_offs >= prefix_len)
            fresh_base = (
                (query_start + kv_offs - prefix_len) * NUM_KV_HEADS + kv_head
            ) * HEAD_DIM
            nk = tl.load(
                KNew_ptr + fresh_base[:, None] + d_offs[None, :],
                mask=fresh[:, None] & d_mask[None, :],
                other=0,
            )
            nv = tl.load(
                VNew_ptr + fresh_base[:, None] + d_offs[None, :],
                mask=fresh[:, None] & d_mask[None, :],
                other=0,
            )
            k_deq = tl.where(fresh[:, None], nk, k_deq)
            values = tl.where(fresh[:, None], nv, values)
            scores = (
                tl.sum(tl.where(d_mask[None, :], q_rot[None, :] * k_deq, 0.0), axis=1)
                * ATTN_SCALE
            )
            scores = tl.where(kv_mask, scores, -float("inf"))
            n_e_max = tl.maximum(tl.max(scores, 0), m_prev)
            re_scale = tl.exp(m_prev - n_e_max)
            p = tl.exp(scores - n_e_max)
            acc = acc * re_scale + tl.sum(p[:, None] * values, 0)
            l_prev = l_prev * re_scale + tl.sum(p, 0)
            m_prev = n_e_max
        out_base = bid * stride_mid_b + hid * stride_mid_h + sid * stride_mid_s
        safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
        tl.store(Mid_o_ptr + out_base + d_offs, acc / safe_l, mask=d_mask)
        tl.store(Mid_o_ptr + out_base + HEAD_DIM, m_prev + tl.log(safe_l))


def paged_block_kv():
    # 32 overflows UB in the deployed Ascend910B4 compiler (PlanMemory).
    # Larger tiles remain opt-in experiments; 4 passed the target NPU probe.
    value = int(os.environ.get("OSCAR_ASCEND_PAGED_BLOCK_KV", "4"))
    if value not in (4, 16, 32, 64, 128):
        raise ValueError("OSCAR_ASCEND_PAGED_BLOCK_KV must be 4, 16, 32, 64 or 128")
    return value


def oscar_paged_attention_triton(
    q, k, v, kc, vc, bt, qsl, seqs, scale, stage=None, *, block_kv=None
):
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    block_kv = paged_block_kv() if block_kv is None else block_kv
    if block_kv not in (4, 16, 32, 64, 128):
        raise ValueError("Unsupported paged KV tile size")
    from .decode_kernel import _oscar_decode_stage2

    n, hq, d = q.shape
    if n == 0:
        return torch.empty_like(q)
    hk = k.shape[1]
    q, k, v = q.contiguous().float(), k.contiguous().float(), v.contiguous().float()
    bt = bt.to(device=q.device).contiguous()
    req, prefix, starts, ends = query_layout(qsl, seqs, q.device)
    if req.numel() != n:
        raise ValueError("Query layout does not match actual query tokens")
    k8, v8 = kc.view(torch.uint8), vc.view(torch.uint8)
    if stage is None:
        sk, sv, owner, rows = k, v, req, 1
    else:
        sk, sv, owner = stage
        rows = owner.shape[0]
    splits = 16
    mid = torch.empty(n, hq, splits, d + 1, device=q.device, dtype=torch.float32)
    _oscar_paged_stage1[(n, hq, splits)](
        req,
        prefix,
        starts,
        k,
        v,
        sk,
        sv,
        owner,
        rows,
        stage is not None,
        q,
        k8,
        v8,
        bt,
        ends,
        mid,
        q.stride(0),
        q.stride(1),
        k8.stride(0),
        k8.stride(1),
        k8.stride(2),
        v8.stride(0),
        v8.stride(1),
        v8.stride(2),
        bt.stride(0),
        mid.stride(0),
        mid.stride(1),
        mid.stride(2),
        NUM_KV_HEADS=hk,
        HEAD_DIM=d,
        BLOCK_SIZE=kc.shape[1],
        NUM_KV_SPLITS=splits,
        KV_GROUP_SIZE=hq // hk,
        DATA_BYTES=d // 4,
        ATTN_SCALE=scale,
        K_IDX_OFF=K_IDX_OFF,
        BLOCK_D=triton.next_power_of_2(d),
        BLOCK_KV=block_kv,
        num_warps=1,
        num_stages=1,
    )
    output = torch.empty_like(q)
    lse = torch.empty(n, hq, device=q.device, dtype=torch.float32)
    _oscar_decode_stage2[(n, hq)](
        mid,
        output,
        lse,
        ends,
        mid.stride(0),
        mid.stride(1),
        mid.stride(2),
        output.stride(0),
        output.stride(1),
        lse.stride(0),
        NUM_KV_SPLITS=splits,
        BLOCK_D=triton.next_power_of_2(d),
        HEAD_DIM=d,
        num_warps=4,
        num_stages=1,
    )
    return output


def oscar_grouped_mtp_attention_triton(
    q, k, v, kc, vc, bt, qsl, seqs, scale, stage=None, *, max_query_len=4,
    prepared_metadata=None,
):
    """Paged MTP attention with one historical KV read per request/GQA tile."""
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    lengths = [b - a for a, b in itertools.pairwise(qsl)]
    if (not lengths or len(lengths) != len(seqs) or max(lengths) > max_query_len
            or any(n <= 0 or s < n for n, s in zip(lengths, seqs))
            or sum(lengths) != q.shape[0]):
        raise ValueError("Grouped OSCAR MTP requires dense q_len in [1, 4]")
    n, hq, d = q.shape
    hk = k.shape[1]
    if hq % hk or k.shape != v.shape or k.shape != (n, hk, d):
        raise ValueError("Invalid grouped OSCAR MTP Q/K/V shapes")
    q = q.contiguous().float()
    k = k.contiguous().float()
    v = v.contiguous().float()
    bt = bt.to(device=q.device).contiguous()
    if prepared_metadata is None:
        q_starts = torch.tensor(qsl[:-1], dtype=torch.int32, device=q.device)
        q_lens = torch.tensor(lengths, dtype=torch.int32, device=q.device)
        prefixes = torch.tensor(
            [s - n_q for s, n_q in zip(seqs, lengths)],
            dtype=torch.int32, device=q.device,
        )
        ends = torch.tensor(
            [s - n_q + j + 1 for s, n_q in zip(seqs, lengths)
             for j in range(n_q)], dtype=torch.int32, device=q.device,
        )
    else:
        q_starts, q_lens, prefixes, ends = prepared_metadata
        expected = ((len(lengths),), (len(lengths),), (len(lengths),), (n,))
        for tensor, shape in zip(prepared_metadata, expected):
            if (tensor.shape != shape or tensor.device != q.device
                    or tensor.dtype != torch.int32 or not tensor.is_contiguous()):
                raise ValueError("Invalid grouped MTP prepared metadata")
    k8, v8 = kc.view(torch.uint8), vc.view(torch.uint8)
    if stage is None:
        sk, sv, owner, stage_rows = k, v, q_lens, 1
    else:
        sk, sv, owner = stage
        stage_rows = owner.shape[0]
    group_size = hq // hk
    tile = max(x for x in range(1, min(4, group_size) + 1)
               if group_size % x == 0)
    group_tiles = group_size // tile
    block_g = triton.next_power_of_2(tile)
    block_q = triton.next_power_of_2(max_query_len)
    block_kv = int(os.environ.get("OSCAR_ASCEND_GROUPED_MTP_BLOCK_KV", "4"))
    if block_kv not in (1, 2, 4):
        raise ValueError("OSCAR_ASCEND_GROUPED_MTP_BLOCK_KV must be 1, 2 or 4")
    longest = max(seqs)
    splits = 1 if longest <= 2048 else 4 if longest <= 8192 else 8 if longest <= 32768 else 16
    mid = torch.empty(n, hq, splits, d + 1, dtype=torch.float32, device=q.device)
    _oscar_grouped_mtp_stage1[(len(lengths), hk * group_tiles, splits)](
        q_starts, q_lens, prefixes, k, v, sk, sv, owner, q, k8, v8, bt, mid,
        q.stride(1), bt.stride(0),
        k8.stride(0), k8.stride(1), k8.stride(2),
        v8.stride(0), v8.stride(1), v8.stride(2),
        mid.stride(0), mid.stride(1), mid.stride(2),
        NUM_BLOCKS=kc.shape[0], NUM_BT_BLOCKS=bt.shape[1],
        NUM_KV_HEADS=hk, HEAD_DIM=d, BLOCK_SIZE=kc.shape[1],
        NUM_SPLITS=splits, KV_GROUP_SIZE=group_size, GROUP_TILES=group_tiles,
        BLOCK_G=block_g, BLOCK_Q=block_q, BLOCK_D=triton.next_power_of_2(d),
        BLOCK_KV=block_kv,
        ATTN_SCALE=scale, K_IDX_OFF=K_IDX_OFF,
        HAS_STAGE=stage is not None, STAGE_ROWS=stage_rows,
        num_warps=1, num_stages=1,
    )
    if splits == 1:
        return mid[:, :, 0, :d]
    from .decode_kernel import _oscar_decode_stage2
    out = torch.empty(n, hq, d, dtype=torch.float32, device=q.device)
    lse = torch.empty(n, hq, dtype=torch.float32, device=q.device)
    _oscar_decode_stage2[(n, hq)](
        mid, out, lse, ends,
        mid.stride(0), mid.stride(1), mid.stride(2),
        out.stride(0), out.stride(1), lse.stride(0),
        NUM_KV_SPLITS=splits, BLOCK_D=triton.next_power_of_2(d), HEAD_DIM=d,
        num_warps=4, num_stages=1,
    )
    return out
