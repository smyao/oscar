"""CPU request-domain/control contracts; never NPU performance evidence."""
from dataclasses import replace
from types import SimpleNamespace
import sys

import pytest
import torch

from oscar_ascend.integration.metadata import OscarMetadata
from oscar_ascend.integration.mixed_decode_split import plan_mixed_decode,dispatch_mixed_decode
from oscar_ascend.ops.reference import attention


def metadata(lengths=(4,4,1024),padding=24,*,draft=False):
    starts=torch.tensor([0,*torch.tensor(lengths).cumsum(0).tolist()],dtype=torch.int32)
    contexts=(9,13,17)[:len(lengths)]
    n=int(starts[-1]);slots=torch.arange(n+padding,dtype=torch.int64);slots[n:]=-1
    return OscarMetadata(starts,torch.tensor([a+b for a,b in zip(contexts,lengths)],dtype=torch.int32),
        torch.arange(len(lengths),dtype=torch.int32)[:,None],slots,len(lengths),n,max(lengths),max(contexts)+max(lengths),
        num_input_tokens=n+padding,query_start_loc_cpu=starts.clone(),is_draft=draft,
        first_draft_current_fia=draft,mixed_decode_split=True)


@pytest.mark.parametrize('draft',[False,True])
def test_partition_preserves_every_request_current_and_history_domain(draft):
    torch.set_num_threads(min(4,torch.get_num_threads()))
    meta=metadata(draft=draft);n=meta.num_input_tokens;g=torch.Generator().manual_seed(47071)
    q=torch.randn(n,2,16,generator=g);k=torch.randn(n,1,16,generator=g);v=torch.randn(n,1,16,generator=g)
    histories={r:(torch.randn(c,1,16,generator=g),torch.randn(c,1,16,generator=g)) for r,c in enumerate((9,13,17))}
    def compute(q,k,v,m,out,cache):
        starts=m.query_start_loc.tolist()
        for r,(a,b) in enumerate(zip(starts,starts[1:])):
            ident=int(m.block_tables[r,0]);hk,hv=histories[ident]
            assert int(m.seq_lens[r])-(b-a)==len(hk)
            out[a:b]=attention(q[a:b],torch.cat((hk,k[a:b])),torch.cat((hv,v[a:b]))).output
            for j in range(a,b):cache[int(m.slot_mapping[j])]=(k[j].clone(),v[j].clone())
    baseline=torch.zeros_like(q);old_cache={}
    compute(q,k,v,meta,baseline,old_cache)
    records=[];new_cache={};validated=[]
    def guard(*statuses):
        if any(bool((s!=0).any()) for s in statuses):raise RuntimeError('mock device guard')
    workspace=SimpleNamespace(store_status=torch.empty(1,dtype=torch.int32),
        validate=lambda *shape:validated.append(shape))
    state=SimpleNamespace(workspace=workspace)
    impl=SimpleNamespace(num_heads=2,num_kv_heads=1,head_size=16,
        provider=SimpleNamespace(layer_state=lambda name:state,ops=SimpleNamespace(status_guard=guard)))
    def forward(layer,cq,ck,cv,cache,m,output):
        assert m.cv_shape_tokens==n and not m.mixed_decode_split
        assert m.current_cumulative==tuple(m.query_start_loc_cpu[1:].tolist())
        assert plan_mixed_decode(m,cq.shape[0]) is None
        records.append((m.num_reqs,m.num_actual_tokens,m.max_query_len,m.mixed_split_segment))
        compute(cq,ck,cv,m,output,new_cache)
    impl.forward=forward
    out=torch.full_like(q,float('nan'));layer=SimpleNamespace(layer_name='test.full')
    plan=plan_mixed_decode(meta,n)
    dispatch_mixed_decode(impl,layer,q,k,v,object(),meta,out,plan)
    assert records==[(2,8,4,'short_prefix'),(1,1024,1024,'long_suffix')]
    assert validated==[(n,2,1,16)]
    assert torch.equal(out,baseline)
    assert old_cache.keys()==new_cache.keys()
    assert all(torch.equal(a,b) for slot in old_cache for a,b in zip(old_cache[slot],new_cache[slot]))


@pytest.mark.parametrize('change',[
    {'capture_origin':True},{'dummy_origin':True},{'mixed_decode_split':False},
    {'is_draft':True,'draft_index':1},{'is_draft':True,'first_draft_current_fia':False},
    {'num_actual_tokens':1031},{'query_start_loc_cpu':None},
])
def test_ineligible_or_inconsistent_metadata_never_splits(change):
    m=replace(metadata(),**change)
    assert plan_mixed_decode(m,m.num_input_tokens) is None


@pytest.mark.parametrize('long_query,allowed',[(83,False),(84,True),(512,True)])
def test_uses_actual_c4_dispatch_boundary(long_query,allowed):
    m=metadata(lengths=(4,4,long_query))
    assert (plan_mixed_decode(m,m.num_input_tokens) is not None) is allowed


@pytest.mark.parametrize('which',['cut','trailing_padding'])
def test_device_boundaries_checked_before_any_subforward(which):
    m=metadata();plan=plan_mixed_decode(m,m.num_input_tokens)
    if which=='cut':m.query_start_loc[2]-=1
    else:m.slot_mapping[-1]=0
    calls=[]
    def guard(*statuses):
        if any(bool((s!=0).any()) for s in statuses):raise RuntimeError('mock device guard')
    ws=SimpleNamespace(validate=lambda *a:None,store_status=torch.empty(0,dtype=torch.int32))
    impl=SimpleNamespace(num_heads=2,num_kv_heads=1,head_size=16,
        provider=SimpleNamespace(layer_state=lambda name:SimpleNamespace(workspace=ws),
                                 ops=SimpleNamespace(status_guard=guard)),forward=lambda *a,**k:calls.append(True))
    q=torch.zeros(m.num_input_tokens,2,16);kv=q[:,:1]
    with pytest.raises(RuntimeError,match='device guard'):
        dispatch_mixed_decode(impl,SimpleNamespace(layer_name='x'),q,kv,kv,None,m,q.clone(),plan)
    assert not calls


@pytest.mark.parametrize('prefix_cache',[False,True,None])
def test_native_builder_requires_explicit_nonsharing_policy(monkeypatch,prefix_cache):
    from test_first_draft_current_only import builder,config,common
    builder(monkeypatch,enabled=False)
    backend=sys.modules['oscar_ascend.integration.backend']
    monkeypatch.setattr(backend,'require_runtime',lambda:SimpleNamespace(config={'experimental_mixed_decode_split':True}))
    cfg=config();cfg.cache_config=SimpleNamespace(enable_prefix_caching=prefix_cache)
    b=backend.OscarMetadataBuilder(object(),['x'],cfg,'cpu')
    m=b.build(0,common())
    assert m.mixed_decode_split is (prefix_cache is False)
    assert b.build_for_drafting(common(),1).mixed_decode_split is False
