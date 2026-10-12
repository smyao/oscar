"""Archive #26/#37-49/#148: request-domain contracts, never NPU performance evidence."""
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
@pytest.mark.parametrize('shared_prefix',[False,True])
def test_partition_preserves_every_request_current_and_history_domain(draft,shared_prefix):
    torch.set_num_threads(min(4,torch.get_num_threads()))
    meta=metadata(draft=draft);n=meta.num_input_tokens;g=torch.Generator().manual_seed(47071)
    q=torch.randn(n,2,16,generator=g);k=torch.randn(n,1,16,generator=g);v=torch.randn(n,1,16,generator=g)
    histories={r:(torch.randn(c,1,16,generator=g),torch.randn(c,1,16,generator=g)) for r,c in enumerate((9,13,17))}
    if shared_prefix:
        # APC fork: two decode requests and one long continuation all read
        # the same computed prefix. Their current suffixes remain separate.
        meta=replace(meta,block_tables=torch.zeros_like(meta.block_tables),
            seq_lens=torch.tensor([9+4,9+4,9+1024],dtype=torch.int32))
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
    assert meta.mixed_decode_split and meta.mixed_split_segment is None
    assert meta.is_draft is draft and meta.first_draft_current_fia is draft


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
def test_native_builder_keeps_request_partition_for_shared_immutable_prefix(monkeypatch,prefix_cache):
    from test_first_draft_current_only import builder,config,common
    builder(monkeypatch,enabled=False)
    backend=sys.modules['oscar_ascend.integration.backend']
    monkeypatch.setattr(backend,'require_runtime',lambda:SimpleNamespace(config={'experimental_mixed_decode_split':True}))
    cfg=config();cfg.cache_config=SimpleNamespace(enable_prefix_caching=prefix_cache)
    b=backend.OscarMetadataBuilder(object(),['x'],cfg,'cpu')
    m=b.build(0,common())
    assert m.mixed_decode_split is (type(prefix_cache) is bool)
    assert b.build_for_drafting(common(),1).mixed_decode_split is False


@pytest.mark.parametrize('draft',[False,True])
def test_real_prefix_hit_keeps_positive_context_and_private_suffix_metadata(monkeypatch,draft):
    from enum import Enum
    from test_first_draft_current_only import builder,config
    from oscar_ascend.integration.mixed_decode_split import _slice_metadata
    class AscendAttentionState(Enum):
        PrefillCacheHit=1
    builder(monkeypatch,enabled=True)
    backend=sys.modules['oscar_ascend.integration.backend']
    flags={'experimental_current_only':True,'experimental_mixed_decode_split':True,
           'experimental_first_mtp_current_fia':True}
    monkeypatch.setattr(backend,'require_runtime',lambda:SimpleNamespace(config=flags))
    cfg=config();cfg.cache_config=SimpleNamespace(enable_prefix_caching=True,mamba_cache_mode='align')
    b=backend.OscarMetadataBuilder(object(),['x'],cfg,'cpu')
    # Native FULL physical prefix page 1 becomes virtual IDs 18..35. The two
    # suffixes are distinct physical pages 2 and 3; 512 future/current tokens
    # do not become a fresh zero-context request simply because APC hit.
    starts=torch.tensor([0,4,516],dtype=torch.int32)
    prefix=torch.arange(18,36,dtype=torch.int32)
    tables=torch.stack((torch.cat((prefix,torch.arange(36,54,dtype=torch.int32))),
                        torch.cat((prefix,torch.arange(54,72,dtype=torch.int32)))))
    slots=torch.cat((torch.arange(4608,4612),torch.arange(6912,7424))).to(torch.int64)
    common=SimpleNamespace(causal=True,query_start_loc=starts,query_start_loc_cpu=starts.clone(),
        seq_lens=torch.tensor([2308,2816],dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([2308,2816],dtype=torch.int32),
        block_table_tensor=tables,slot_mapping=slots,num_reqs=2,num_actual_tokens=516,
        num_input_tokens=516,max_query_len=512,max_seq_len=2816,attn_state=AscendAttentionState.PrefillCacheHit)
    m=b.build(2304,common,fast_build=object() if draft else False)
    assert m.current_only_plan is None
    assert m.mixed_decode_split and m.is_draft is draft
    plan=plan_mixed_decode(m,516)
    assert plan.prefix_tokens==4
    for rb,re,tb,te,segment in [(0,1,0,4,'short_prefix'),(1,2,4,516,'long_suffix')]:
        child=_slice_metadata(m,rb,re,tb,te,516,segment)
        assert child.block_tables.data_ptr()==m.block_tables[rb:].data_ptr()
        assert child.slot_mapping.data_ptr()==slots[tb:].data_ptr()
        assert torch.equal(child.slot_mapping,slots[tb:te])
        assert int(child.seq_lens[0])-int(child.query_start_loc[-1])==2304
        # These are the main/first-MTP positions prepared from context+row;
        # later-MTP's accepted slot-derived prepare is intentionally unsplit.
        assert list(range(int(child.seq_lens[0])-(te-tb),int(child.seq_lens[0])))==list(range(2304,2304+te-tb))
        assert child.is_draft is draft and child.draft_index==0
    assert torch.equal(m.query_start_loc,starts) and m.block_tables is tables
