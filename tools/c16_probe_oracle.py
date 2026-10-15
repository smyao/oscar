"""Archive #126/#145/#148/#154/#157 and startup D.4: shared C16 CPU oracle.

Shared only by independent stable-C16/pipeline and stable-C16/B1 probes.
All672 queries and original20 generic-oracle anchors are retained. Long-context
GQA uses zero-stride views with the original FP32 einsum order; monotonic exact
window cuts update only newly compressed rows. Production never imports this.
The function bodies were relocated unchanged from the bounded32K host check.
"""
import math
from . import probe_history_reuse as reuse
from .probe_striped_cache import _store_bits_equal

LONG_CONTEXTS={'long_context_32k':32768,'long_context_64k':65536}


def _shared_kv_attention(torch,query,key,value,scale,*,inputs_checked=False):
    """Independent FP32 dense oracle for the single-KV-head target.

    Zero-stride GQA views preserve the generic oracle's per-head M1 einsum,
    without materializing six history copies. All history tokens and all
    heads remain in the same FP32 softmax and value reductions.
    This is CPU diagnostic code, never a production route.
    """
    from oscar_ascend.ops.reference import AttentionResult,_finite
    if (query.ndim!=3 or key.ndim!=3 or key.shape[1]!=1 or value.shape!=key.shape
            or query.shape[-1]!=key.shape[-1] or query.device!=key.device or key.device!=value.device):
        raise RuntimeError('shared-KV oracle requires the single-head target')
    nq,heads,dim=query.shape
    if not math.isfinite(scale) or scale<=0:raise RuntimeError('invalid dense oracle scale')
    if not inputs_checked:
        for name,tensor in (('query',query),('key',key),('value',value)):_finite(name,tensor)
    if key.shape[0]==0:
        return AttentionResult(torch.zeros_like(query,dtype=torch.float32),
                               torch.full((nq,heads),-math.inf,device=query.device))
    k=key.float().expand(-1,heads,-1);v=value.float().expand(-1,heads,-1)
    scores=torch.einsum('qhd,khd->qhk',query.float(),k)*scale
    if not bool(torch.isfinite(scores).all()):raise RuntimeError('nonfinite dense oracle scores')
    maximum=scores.amax(-1);live=torch.isfinite(maximum)
    safe_max=torch.where(live,maximum,torch.zeros_like(maximum))
    weights=torch.exp(scores-safe_max.unsqueeze(-1));denominator=weights.sum(-1)
    safe_den=torch.where(live,denominator,torch.ones_like(denominator))
    output=torch.einsum('qhk,khd->qhd',weights,v)/safe_den.unsqueeze(-1)
    lse=torch.where(live,safe_max+torch.log(safe_den),torch.full_like(maximum,-math.inf))
    return AttentionResult(output,lse)


def _long_oracle(torch,fixture):
    from oscar_ascend.ops.reference import _finite
    spec=fixture['spec'];context=spec.contexts[0]
    k,v,dk,dv=fixture['logical_history'];q=fixture['cpu']['q']
    for name,tensor in (('q',q),('k',k),('v',v),('decoded_k',dk),('decoded_v',dv),
                        ('current_k',fixture['cpu']['ck']),('current_v',fixture['cpu']['cv'])):
        _finite(name,tensor)
    # The exact/quantized boundary increases monotonically with query row.
    # One selected K/V pair plus only the new boundary rows replaces672 full
    # history clones without changing any selected FP32 input value.
    kk,vv=k.float().clone(),v.float().clone();previous=reuse.SINK;expected={}
    anchors=fixture.get('generic_reference_anchors',{});checked=[];bit_differences=[]
    for row in range(q.shape[0]):
        cut=min(context,max(reuse.SINK,context+row+1-spec.recent_tokens))
        if cut>previous:
            kk[previous:cut]=dk[previous:cut];vv[previous:cut]=dv[previous:cut]
        previous=cut
        value=_shared_kv_attention(torch,q[row:row+1],kk,vv,fixture['scale'],inputs_checked=True)
        expected[row]=(value.output[0],value.lse[0])
        if row in anchors:
            # The generic fixture's old samples include source2 current K/V.
            # Match that domain explicitly for the cross-check; the main672
            # oracle above remains history+window because CV suppresses source2.
            all_k=torch.cat((kk,fixture['cpu']['ck'][:row+1].float()))
            all_v=torch.cat((vv,fixture['cpu']['cv'][:row+1].float()))
            complete=_shared_kv_attention(torch,q[row:row+1],all_k,all_v,
                                           fixture['scale'],inputs_checked=True)
            actual=(complete.output[0],complete.lse[0]);old=anchors[row]
            same=all(_store_bits_equal(torch,a,b) for a,b in zip(actual,old))
            if not same:
                bit_differences.append({'query':row,
                    'max_output_abs':float((actual[0]-old[0]).abs().max()),
                    'max_lse_abs':float((actual[1]-old[1]).abs().max())})
            # Frozen numeric gate is never relaxed. Even a within-tolerance
            # CPU bit difference is recorded and rejected by validate_evidence.
            for a,b in zip(actual,old):torch.testing.assert_close(a,b,atol=.005,rtol=.005)
            checked.append(row)
    fixture['expected']=expected
    if anchors:
        fixture['reference_anchor_check']={
            'status':'bitwise_passed' if not bit_differences else 'cpu_bit_differences',
            'domain':'history_window_current','queries':checked,
            'expected_count':len(anchors),'checked_count':len(checked),
            'bit_differences':bit_differences}


def _oracle(torch,fixture):
    from oscar_ascend.ops.reference import attention
    spec=fixture['spec'];context=spec.contexts[0]
    if context in LONG_CONTEXTS.values():
        _long_oracle(torch,fixture)
        return
    k,v,dk,dv=fixture['logical_history'];q=fixture['cpu']['q'];expected={}
    for row in range(672):
        cut=min(context,max(reuse.SINK,context+row+1-spec.recent_tokens))
        kk,vv=k.float().clone(),v.float().clone();kk[reuse.SINK:cut]=dk[reuse.SINK:cut];vv[reuse.SINK:cut]=dv[reuse.SINK:cut]
        value=attention(q[row:row+1],kk,vv,scale=fixture['scale'],causal=False)
        expected[row]=(value.output[0],value.lse[0])
    fixture['expected']=expected
