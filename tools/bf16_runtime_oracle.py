"""Archive #126/#145/#148/#157: TEST-ONLY sampled lifecycle BF16 oracle.

Use actual independent fixture/cache inputs, never candidate attention output.
Native current FIA is unchanged and has its own dense numerical reference;
CV sources follow the predeclared BF16 online-tile contract. D.4 permits this
offline bounded diagnostic reconstruction, not any serving materialization.
"""
from __future__ import annotations
import math
import torch
from tools.bf16_reference import attention_tiles,merge_parts,CONTRACT
from oscar_ascend.ops.reference import AttentionResult,attention,decode_kv

def request_samples(q,qr,ck,cv,context,rv,history,precise,samples,*,recent=256,
                    sink=64,splits=1,native_current=False):
    """Per-request producer ranges; query groups are request-relative128/GQA.

    history/precise are independent callbacks taking logical old positions.
    The query count of a CV task is its entire producer group, even when only
    one member is sampled; this preserves the KV tile/split rounding origins.
    """
    length,hq,dim=q.shape;hk=ck.shape[1];qtile=128//(hq//hk);expected={}
    for local in samples:
        begin=(local//qtile)*qtile;count=min(qtile,length-begin)
        position=context+local;first_position=context+begin
        cut=min(context,max(sink,position+1-recent))
        sink_count=min(sink,context);tail_begin=min(context,max(sink,first_position+1-recent))
        bounds=((min(sink,context),min(context,max(sink,first_position+count-recent))),
                (0,sink_count+context-tail_begin),(context,first_position+count))
        parts=[]
        for kind,(first,last) in enumerate(bounds):
            if kind==2 and native_current:
                # FIA's externally visible output has the query BF16 dtype.
                # Its internal P/reduction order is closed, not asserted here.
                part=attention(q[local:local+1],ck[:local+1],cv[:local+1],causal=False)
                parts.append(AttentionResult(part.output.bfloat16().float(),part.lse));continue
            width=(last-first+splits-1)//splits
            for split in range(splits):
                lo=min(last,first+split*width);hi=min(last,lo+width)
                candidates=torch.arange(lo,hi,dtype=torch.int64)
                if kind==0:
                    k,v=history(candidates);visible=(candidates>=sink)&(candidates<cut)&(candidates<=position)
                    query=qr[local:local+1];rotation=rv
                elif kind==1:
                    positions=torch.where(candidates<sink_count,candidates,tail_begin+candidates-sink_count)
                    k,v=precise(positions);visible=((positions<sink)|(positions>=cut))&(positions<=position)
                    query=q[local:local+1];rotation=None
                else:
                    k,v=ck[candidates-context],cv[candidates-context];visible=candidates<=position
                    query=q[local:local+1];rotation=None
                parts.append(attention_tiles(query,k,v,visible=visible[None,:],value_rotation=rotation))
        result=merge_parts(parts);expected[local]=(result.output[0],result.lse[0])
    return expected

def sampled_fixture_expected(fixture,*,native_current,indices=None):
    """All legacy sampled rows, using original canonical fixture storage."""
    from tools import probe_history_reuse as reuse
    data=fixture['cpu'];spec=fixture['spec'];dim=spec.dim;hk=spec.kv_heads
    if indices is None:indices=sorted(fixture['expected'])
    index_set=set(indices);out={};offset=0
    slot_bytes=dim//2+8
    raw=data['raw'][reuse.PREFIX:].view(fixture['blocks'],reuse.BLOCK_TOKENS,hk,slot_bytes)
    for request,(length,context,pages) in enumerate(zip(spec.qlens,spec.contexts,fixture['page_assignments'])):
        samples=[i-offset for i in indices if offset<=i<offset+length]
        if not samples:offset+=length;continue
        native=native_current[request] if isinstance(native_current,(tuple,list)) else native_current
        if context==0 and native:
            # Fresh current-only rows have no compressed/exact-old segments.
            # Reuse their already independent dense oracle, avoiding a second
            # quadratic diagnostic pass for the20K/30K system cases.
            out.update({offset+i:(fixture['expected'][offset+i][0].bfloat16().float(),
                                  fixture['expected'][offset+i][1]) for i in samples})
            offset+=length;continue
        if context:
            positions=torch.arange(context);mapping=torch.tensor(pages)[positions//reuse.BLOCK_TOKENS]
            decoded=decode_kv(raw[mapping,positions%reuse.BLOCK_TOKENS],dim)
        else:decoded=(torch.empty((0,hk,dim)),torch.empty((0,hk,dim)))
        def history(positions):return decoded[0][positions],decoded[1][positions]
        def precise(positions):
            physical=torch.tensor(pages)[positions//reuse.BLOCK_TOKENS];within=positions%reuse.BLOCK_TOKENS
            row=torch.where(positions<reuse.SINK,positions,reuse.SINK+within%(spec.recent_tokens+reuse.SPECULATIVE))
            if not torch.equal(data['tags'][physical,row],within):raise RuntimeError('independent lifecycle window tag mismatch')
            return data['wk'][physical,row],data['wv'][physical,row]
        result=request_samples(data['q'][offset:offset+length],data['qr'][offset:offset+length],
            data['ck'][offset:offset+length],data['cv'][offset:offset+length],context,data['rv'],history,precise,
            samples,recent=spec.recent_tokens,splits=spec.splits,
            native_current=native)
        out.update({offset+i:value for i,value in result.items()});offset+=length
    if set(out)!=index_set:raise RuntimeError('sampled BF16 oracle lost a requested row')
    return out
