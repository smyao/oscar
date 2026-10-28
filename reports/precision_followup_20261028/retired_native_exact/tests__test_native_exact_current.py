"""Archive #145/#164/#173: fresh-only arithmetic, cache identity and scratch lifetime.

Host operator doubles verify production Python routing and buffer ownership.
They do not establish NPU/graph/LongBench acceptance. Expected cache rotations
are calculated independently from the supplied raw inputs and BF16 matrices.
"""
from types import SimpleNamespace as NS

import pytest
import torch

from oscar_ascend.integration import current_only_dispatch as dispatch
from oscar_ascend.integration.current_only_plan import plan_current_suffix
from oscar_ascend.integration.metadata import OscarMetadata
from oscar_ascend.runtime import GraphWorkspace, WorkspaceGeometry


def fixture(*, draft=False, strided=False, mixed=False, fresh=3,
            lse_dtype=torch.float32, fault=None, policy="native_exact"):
    cut=2 if mixed else 0
    active=cut+fresh; n=active+2; h,hk,d=6,1,256
    w=object.__new__(GraphWorkspace)
    w.geometry=WorkspaceGeometry(n,h,hk,d,aligned_math=True,official_math=True)
    for name,heads,dtype in (
        ('query_input',h,torch.bfloat16),('key_input',hk,torch.bfloat16),
        ('value_input',hk,torch.bfloat16),('query_rot_bf16',h,torch.bfloat16),
        ('key_rot_bf16',hk,torch.bfloat16),('value_rot_bf16',hk,torch.bfloat16),
        ('centered_key_bf16',hk,torch.bfloat16)):
        setattr(w,name,torch.full((n,heads,d),-19,dtype=dtype))
    # Each status starts poisoned: every guarded live status must be written.
    for name,heads in (('rotate_status',h),('merge_status',h),('final_status',h),('store_status',hk)):
        setattr(w,name,torch.full((n,heads),173,dtype=torch.int32))
    w.final_lse=torch.full((n,h),float('nan'))
    w.lse=torch.full((n,h),float('nan'))
    w.slots=torch.empty(n,dtype=torch.int64)
    w.positions=torch.full((n,),-999,dtype=torch.int64)
    w.tasks=torch.empty(n*3,16,dtype=torch.int64)
    w.attention_status=torch.full((n*3,2),173,dtype=torch.int32)
    rotation=torch.eye(d,dtype=torch.float32)
    rotation[:2,:2]=torch.tensor([[.6,-.8],[.8,.6]])
    rotation=rotation.bfloat16()
    mean=torch.full((d,),.125,dtype=torch.bfloat16)
    state=NS(workspace=w,cache_format='striped_rotated_fp32_v3',
        rotation_k_bf16=rotation,rotation_v_bf16=rotation,key_mean_bf16=mean,
        raw=object(),window_key=object(),window_value=object(),window_tags=object(),
        spec=NS(block_size=2304,conv_bytes=30720,ssm_bytes=393216),num_blocks=2,
        snapshots=NS(sink_tokens=64,recent_tokens=256,ring_tokens=259))
    storage=(torch.arange(n*(h+2)*d).reshape(n,(h+2)*d)%29-14).float().mul_(.03125).bfloat16()
    q,k,v=storage.split((h*d,d,d),dim=1)
    v[:, :2]=1
    if not strided:q,k,v=(x.contiguous() for x in (q,k,v))
    originals=tuple(x.clone() for x in (q,k,v))
    slots=torch.tensor(list(range(active))+[-1,-1],dtype=torch.int32)
    if fault=='slot':slots[cut]=-1
    if fault=='padding':slots[-1]=active
    starts=(0,cut,active) if mixed else (0,active)
    lens=(100,fresh) if mixed else (fresh,)
    metadata=OscarMetadata(torch.tensor(starts,dtype=torch.int32),torch.tensor(lens,dtype=torch.int32),
        torch.zeros(len(lens),4,dtype=torch.int32),slots,len(lens),active,max(fresh,cut),max(lens),
        num_input_tokens=n,query_start_loc_cpu=torch.tensor(starts),current_cumulative=starts[1:],
        is_draft=draft,first_draft_current_only=draft)
    plan=plan_current_suffix(starts,lens,actual_tokens=active,padded_tokens=n)
    events=[]; stores=[]; guards=[]
    class Ops:
        def prepare_official_attention_tasks_out(self,starts,lengths,slot,tasks,positions,*args):
            events.append('prepare')
            assert starts.tolist()==[0,fresh] and lengths.tolist()==[fresh]
            tasks.zero_();tasks[:,1]=1
            if fault=='context':tasks[0,8]=1
            positions.copy_(torch.arange(fresh))
        def copy_validate_bf16_query_out(self,source,slot,destination,status,cores):
            events.append('verify_q')
            assert source.data_ptr()!=destination.data_ptr()
            assert destination.data_ptr()==w.query_rot_bf16.data_ptr()
            assert torch.equal(source,originals[0][cut:active].view(fresh,h,d))
            destination.copy_(source);status.zero_()
            if fault=='query':status[0,0]=2
        def copy_validate_current_out(self,current,lse,out,out_lse,status):
            events.append('validate_output')
            assert current.data_ptr()==w.query_input.data_ptr()
            assert lse.dtype==torch.float32 and lse.data_ptr()==w.lse.data_ptr()
            assert out_lse.data_ptr()==w.final_lse.data_ptr()
            assert status.data_ptr()==w.final_status.data_ptr()
            assert out.data_ptr()!=current.data_ptr() and out_lse.data_ptr()!=lse.data_ptr()
            out.copy_(current);out_lse.copy_(lse.squeeze(-1));status.zero_()
            if fault=='output':status[0,0]=2
        def store_aligned_int2_out(self,kr,vr,wk,wv,slot,positions,raw,window_k,window_v,tags,status,*geometry):
            events.append('store')
            assert torch.equal(kr,torch.matmul((originals[1][cut:active].view(fresh,hk,d)-mean).bfloat16(),rotation))
            assert torch.equal(vr,torch.matmul(originals[2][cut:active].view(fresh,hk,d),rotation))
            assert kr.data_ptr()==wk.data_ptr() and vr.data_ptr()==wv.data_ptr()
            assert slot.tolist()==list(range(cut,active))
            assert positions.tolist()==list(range(fresh))
            status.zero_();stores.append((kr.clone(),vr.clone()))
            if fault=='store':status[0,0]=3
        def status_guard(self,*statuses):
            if any(bool(x.ne(0).any()) for x in statuses):raise RuntimeError('device status trap')
    ops=Ops()
    config={'experimental_official_math':True,'oscar_current_only_policy':policy}
    class Impl:
        num_heads=h;num_kv_heads=hk;head_size=d;scale=d**-.5
        provider=NS(ops=ops,config=config,layer_state=lambda _:state)
        def forward(self,layer,query,key,value,cache,meta,output):
            events.append('prefix')
            assert mixed and meta.num_actual_tokens==cut and meta.current_only_plan is None
            assert meta.seq_lens.tolist()==[100] and not meta.first_draft_current_only
            assert meta.is_draft==draft
            # Prefix finishes before suffix reuses the global scratch arena.
            w.query_input.fill_(77);w.key_input.fill_(78);w.value_input.fill_(79)
            output.fill_(7)
    def fia(query,key,value,cumulative,**kwargs):
        events.append('native_fia')
        assert cumulative==(fresh,)
        assert query.data_ptr()==w.query_rot_bf16.data_ptr()
        assert torch.equal(query,originals[0][cut:active].view(fresh,h,d))
        assert torch.equal(key,originals[1][cut:active].view(fresh,hk,d))
        assert torch.equal(value,originals[2][cut:active].view(fresh,hk,d))
        return value.expand(fresh,h,d).contiguous(),torch.full((fresh,h,1),1.125,dtype=lse_dtype)
    def slot_guard(ops,status,slot,count):
        status[:count].copy_((slot[:count]<0).int()[:,None])
        ops.status_guard(status[:count])
    output=torch.full_like(q,99)
    return NS(w=w,state=state,impl=Impl(),metadata=metadata,plan=plan,
        inputs=(q,k,v),originals=originals,output=output,fia=fia,slot_guard=slot_guard,
        events=events,stores=stores,cut=cut,active=active,fresh=fresh,guards=guards)


def invoke(case,monkeypatch):
    monkeypatch.setattr(dispatch,'native_current_partial',case.fia)
    monkeypatch.setattr(dispatch,'guard_current_slots',case.slot_guard)
    return dispatch.dispatch_current_suffix(case.impl,NS(layer_name='test'),
        *case.inputs,object(),case.metadata,case.output,case.plan)


@pytest.mark.parametrize('draft',[False,True])
@pytest.mark.parametrize('strided',[False,True])
@pytest.mark.parametrize('mixed',[False,True])
@pytest.mark.parametrize('lse_dtype',[torch.float16,torch.bfloat16,torch.float32])
def test_native_exact_suffix_keeps_raw_fia_and_identical_rotated_writer(monkeypatch,draft,strided,mixed,lse_dtype):
    case=fixture(draft=draft,strided=strided,mixed=mixed,lse_dtype=lse_dtype)
    addresses={name:getattr(case.w,name).data_ptr() for name in
        ('query_input','query_rot_bf16','key_input','value_input','final_status','final_lse')}
    assert invoke(case,monkeypatch) is case.output
    assert case.events==(['prefix'] if mixed else [])+['prepare','verify_q','native_fia','validate_output','store']
    for actual,original in zip(case.inputs,case.originals):assert torch.equal(actual,original)
    assert all(getattr(case.w,name).data_ptr()==pointer for name,pointer in addresses.items())
    expected=case.originals[2][case.cut:case.active].view(case.fresh,1,256).expand(case.fresh,6,256)
    assert torch.equal(case.output[case.cut:case.active].view_as(expected),expected)
    assert bool(case.output[case.active:].eq(0).all())
    if mixed:assert bool(case.output[:case.cut].eq(7).all())
    assert bool(case.w.final_lse[:case.fresh].eq(1.125).all())
    assert bool(case.w.final_status[:case.fresh].eq(0).all())
    assert bool(case.w.final_status[case.fresh:].eq(173).all())
    assert len(case.stores)==1


@pytest.mark.parametrize('fault',['slot','padding','context','query','output','store'])
def test_native_exact_guards_are_live_and_do_not_consume_stale_status(monkeypatch,fault):
    case=fixture(mixed=True,strided=True,fault=fault)
    with pytest.raises(RuntimeError,match='device status trap'):invoke(case,monkeypatch)
    if fault in ('slot','padding','context'):assert 'native_fia' not in case.events
    else:assert 'native_fia' in case.events


def test_single_token_native_exact_avoids_known_bf16_inverse_error(monkeypatch):
    case=fixture(fresh=1)
    invoke(case,monkeypatch)
    assert torch.equal(case.output[0].view(6,256)[:,:2],torch.ones((6,2),dtype=torch.bfloat16))
    raw=case.originals[2][0].view(1,1,256)
    rotated=torch.matmul(raw,case.state.rotation_v_bf16)
    old=torch.matmul(rotated,case.state.rotation_v_bf16.T)
    assert not torch.equal(old,raw)


def test_native_exact_removes_two_matmuls_without_adding_launches_or_buffers(monkeypatch):
    case=fixture(strided=True)
    matmul=torch.matmul
    writes=[]
    def counted(left,right,*,out=None):
        if out is not None:
            writes.append(out.data_ptr())
            return matmul(left,right,out=out)
        return matmul(left,right)
    monkeypatch.setattr(torch,'matmul',counted)
    invoke(case,monkeypatch)
    assert writes==[case.w.key_rot_bf16.data_ptr(),case.w.value_rot_bf16.data_ptr()]
    assert case.events==['prepare','verify_q','native_fia','validate_output','store']


def test_native_exact_uses_full_input_capacity_not_small_history_arena(monkeypatch):
    from dataclasses import replace
    case=fixture(fresh=9,strided=True)
    case.w.geometry=replace(case.w.geometry,history_token_capacity=1)
    # No query_rot history arena exists in this fixture: only the full-size
    # query_rot_bf16 can serve the exact suffix's verified raw Q.
    assert not hasattr(case.w,'query_rot')
    invoke(case,monkeypatch)
    assert bool(case.w.final_status[:case.fresh].eq(0).all())


@pytest.mark.parametrize('capture,dummy,later',[(True,False,False),(False,True,False),(False,False,True)])
def test_native_exact_does_not_expand_existing_draft_or_capture_qualification(monkeypatch,capture,dummy,later):
    case=fixture(draft=later)
    object.__setattr__(case.metadata,'capture_origin',capture)
    object.__setattr__(case.metadata,'dummy_origin',dummy)
    if later:object.__setattr__(case.metadata,'draft_index',1)
    with pytest.raises(Exception,match='fresh|eager main prefill|qualified first MTP'):
        invoke(case,monkeypatch)
    assert not case.events
