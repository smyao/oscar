"""Offline numerical proof for bounded shared rotated-FP32 history tiles.

Archive #126/#129/#145/#148-155 and startup D.4: no whole-history *serving*
restore, no BF16 inverse route, no altered QK/PV dtype or KV256 online order.
This CPU experiment materializes history only to form an independent oracle;
it does not demonstrate an AscendC implementation or NPU speed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


SINK = 64
RECENT = 256
KV_TILE = 256
SHARED_TILE = 8192
D = 256
HEADS = 6
QUERY_TILE = 128 // HEADS


def _groups(query_tokens: int) -> tuple[int, ...]:
    choices = (0, 12 * QUERY_TILE, 16 * QUERY_TILE)
    return tuple(start for start in choices if start < query_tokens)


def _cv_source0(torch, qrot, packed, full_k, full_v, rv_t,
                context: int, start: int, *, shared: bool):
    from oscar_ascend.ops.reference import decode_kv

    count = min(QUERY_TILE, qrot.shape[0] - start)
    kvend = min(context, max(SINK, context + start + count - RECENT))
    queries = qrot[start:start+count]
    if kvend <= SINK:
        raise ValueError("test query group has no historical source0")
    rows = torch.arange(start, start+count)
    cuts = (context + rows + 1 - RECENT).clamp(min=SINK, max=context)
    m = torch.full((count, HEADS), -float("inf"), dtype=torch.float32)
    z = torch.zeros((count, HEADS), dtype=torch.float32)
    acc = torch.zeros((count, HEADS, D), dtype=torch.float32)
    scale = D ** -0.5
    if shared:
        # Simulates one producer decompression per 8K block. In the actual
        # kernel only a fixed ≤8K K/V buffer can exist at a time.
        cached: dict[int, tuple] = {}
        for block in range(SINK, kvend, SHARED_TILE):
            end = min(block+SHARED_TILE, context)
            cached[block] = decode_kv(packed[block:end], D)
    for block in range(SINK, kvend, SHARED_TILE):
        end_block = min(block+SHARED_TILE, kvend)
        for kv_start in range(block, end_block, KV_TILE):
            live = min(KV_TILE, kvend-kv_start)
            if shared:
                block_k, block_v = cached[block]
                source_k = block_k[kv_start-block:kv_start-block+live,0]
                source_v = block_v[kv_start-block:kv_start-block+live,0]
            else:
                source_k, source_v = decode_kv(packed[kv_start:kv_start+live], D)
                source_k, source_v = source_k[:,0], source_v[:,0]
            # Every original source0 task has its own kvend. Its last partial
            # 256-tile must be private and zero past kvend before *pre-mask*
            # finite checking; never feed a shared future NaN to its QK.
            k = torch.zeros((KV_TILE,D),dtype=torch.float32)
            v = torch.zeros_like(k)
            k[:live], v[:live] = source_k, source_v
            scores = torch.einsum("qhd,kd->qhk",queries,k).mul(scale)
            if not bool(torch.isfinite(scores).all()):
                raise ValueError("nonfinite score before original source0 mask")
            positions = kv_start + torch.arange(KV_TILE)
            scores.masked_fill_((positions[None,:] >= cuts[:,None])[:,None,:],-float("inf"))
            tile_m = scores.amax(-1)
            new_m = torch.maximum(m,tile_m)
            old_factor = torch.where(torch.isfinite(m),torch.exp(m-new_m),0.0)
            p = torch.exp(scores-new_m[...,None])
            p = torch.where(torch.isfinite(scores),p,0.0)
            acc = acc*old_factor[...,None] + torch.einsum("qhk,kd->qhd",p,v)
            z = z*old_factor + p.sum(-1)
            m = new_m
        # Actual consumer stores raw FP32 acc, max and sum across 8K blocks.
        # Copying all three exact bit patterns cannot change the recurrence.
        if shared and end_block < kvend:
            acc,z,m = acc.clone(),z.clone(),m.clone()
    out = acc / z[...,None]
    lse = m + z.log()
    return out @ rv_t,lse,kvend


def _dense_source0(torch, qrot, full_k, full_v, rv_t,
                   context: int, start: int):
    count = min(QUERY_TILE, qrot.shape[0]-start)
    result = torch.empty((count,HEADS,D),dtype=torch.float32)
    lse = torch.empty((count,HEADS),dtype=torch.float32)
    for local in range(count):
        cut = min(context,max(SINK,context+start+local+1-RECENT))
        scores = torch.einsum("hd,kd->hk",qrot[start+local],full_k[SINK:cut,0])*(D**-0.5)
        probs = scores.softmax(-1)
        result[local] = probs @ full_v[SINK:cut,0]
        lse[local] = scores.logsumexp(-1)
    return result @ rv_t,lse


def run_case(torch, query_tokens: int, context: int) -> dict:
    from oscar_ascend.ops.reference import encode_kv,decode_kv
    if query_tokens<=0 or context<=SINK or D!=256:
        raise ValueError("positive query and compressed history required")
    generator=torch.Generator().manual_seed(19009+query_tokens+context)
    q=torch.randn((query_tokens,HEADS,D),generator=generator).to(torch.bfloat16)
    k=torch.randn((context,1,D),generator=generator).to(torch.bfloat16)
    v=torch.randn((context,1,D),generator=generator).to(torch.bfloat16)
    # Orthogonal permutation/sign rotation avoids a cast while keeping the
    # same FP32 rotated-cache/rotated-query geometry as the serving kernels.
    signs=torch.where(torch.arange(D)%2==0,1.0,-1.0)
    qrot=q.float()*signs
    packed=encode_kv(k.float()*signs,v.float()*signs)
    full_k,full_v=decode_kv(packed,D)
    rv_t=torch.diag(signs)
    rows=[]
    for start in _groups(query_tokens):
        old_out,old_lse,old_end=_cv_source0(torch,qrot,packed,full_k,full_v,rv_t,
                                           context,start,shared=False)
        new_out,new_lse,new_end=_cv_source0(torch,qrot,packed,full_k,full_v,rv_t,
                                           context,start,shared=True)
        if old_end!=new_end or not torch.equal(old_out.view(torch.int32),new_out.view(torch.int32)) or \
                not torch.equal(old_lse.view(torch.int32),new_lse.view(torch.int32)):
            raise AssertionError("bounded shared tile changed original KV256 recurrence")
        dense_out,dense_lse=_dense_source0(torch,qrot,full_k,full_v,rv_t,context,start)
        max_out=(new_out-dense_out).abs().max().item()
        max_lse=(new_lse-dense_lse).abs().max().item()
        if not bool(torch.allclose(new_out,dense_out,atol=0.005,rtol=0.005)) or \
                not bool(torch.allclose(new_lse,dense_lse,atol=0.005,rtol=0.005)):
            raise AssertionError("source0 output/LSE exceeds frozen oracle")
        rows.append({"query_start":start,"kvend":new_end,"output_maxabs":max_out,
                     "lse_maxabs":max_lse,"old_new_bitwise":True})
    return {"query_tokens":query_tokens,"context":context,"groups":rows,
            "history_tiles":math.ceil((context-SINK)/SHARED_TILE)}


def main() -> int:
    import torch
    torch.set_num_threads(min(torch.get_num_threads(),4))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    cases=[run_case(torch,q,c) for q,c in ((21,511),(385,511),(21,20000),(385,20000))]
    report={"status":"cpu_mathematical_prototype_passed","scope":"source0_sampled_query_groups_only",
            "cases":cases,"cann_compiled":False,"target_npu_verified":False,
            "performance_measured":False,"promotion_allowed":False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":report["status"],"cases":len(cases),
                      "groups":sum(len(c["groups"]) for c in cases)}))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
