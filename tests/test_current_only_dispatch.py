"""Archive #126/#129/#155: host ordering/metadata proof, not NPU precision."""
from types import SimpleNamespace

import pytest
import torch

from oscar_ascend.integration import current_only_dispatch as module
from oscar_ascend.integration.current_only_plan import plan_current_suffix
from oscar_ascend.integration.metadata import OscarMetadata
from oscar_ascend.integration.current_attention import _populate_slot_error_status, use_native_current
from oscar_ascend.runtime import GraphWorkspace, WorkspaceGeometry


@pytest.mark.parametrize('draft', [False, True])
@pytest.mark.parametrize('bad_context,bad_padding,bad_slot,lse_dtype',[
    (False,False,False,torch.float32),(True,False,False,torch.float32),
    (False,True,False,torch.float32),(False,False,True,torch.float32),
    (False,False,False,torch.float16),(False,False,False,torch.bfloat16)])
def test_current_suffix_keeps_prefix_and_checks_gpu_context_before_fia(monkeypatch,bad_context,bad_padding,bad_slot,lse_dtype,draft):
    n,active,cut,h,hk,d=16,14,4,6,1,256
    workspace=object.__new__(GraphWorkspace)
    workspace.geometry=WorkspaceGeometry(n,h,hk,d,cube_cores=1)
    workspace.query_input=torch.empty(n,h,d,dtype=torch.bfloat16)
    workspace.key_input=torch.empty(n,hk,d,dtype=torch.bfloat16)
    workspace.value_input=torch.empty_like(workspace.key_input)
    workspace.slots=torch.empty(n,dtype=torch.int64)
    workspace.tasks=torch.empty(n*3,16,dtype=torch.int64)
    workspace.attention_status=torch.empty(n*3,2,dtype=torch.int32)
    workspace.positions=torch.empty(n,dtype=torch.int64)
    workspace.merge_status=torch.empty(n,h,dtype=torch.int32)
    workspace.rotate_status=torch.empty_like(workspace.merge_status)
    workspace.store_status=torch.empty(n,hk,dtype=torch.int32)
    workspace.lse=torch.empty(n,h)
    snapshots=SimpleNamespace(sink_tokens=64,recent_tokens=256,ring_tokens=259)
    state=SimpleNamespace(workspace=workspace,snapshots=snapshots,cache_format='striped_v1',
        rotation_k_transpose=torch.eye(d),rotation_v_transpose=torch.eye(d),
        raw=torch.zeros(1,dtype=torch.uint8),window_key=torch.empty(1),window_value=torch.empty(1),
        window_tags=torch.empty(1),spec=SimpleNamespace(block_size=512,conv_bytes=0,ssm_bytes=136*512),
        num_blocks=1,hadamard=True)
    order=[]
    slots=torch.tensor(list(range(active))+[-1,-1],dtype=torch.int32)
    if bad_padding:slots[-1]=20
    if bad_slot:slots[cut]=-1
    metadata=OscarMetadata(torch.tensor([0,cut,active],dtype=torch.int32),
        torch.tensor([100,10],dtype=torch.int32),torch.zeros(2,2,dtype=torch.int32),slots,
        2,active,10,100,num_input_tokens=n,query_start_loc_cpu=torch.tensor([0,cut,active]),
        current_cumulative=(cut,active),is_draft=draft,first_draft_current_only=draft)
    plan=plan_current_suffix((0,cut,active),(100,10),actual_tokens=active,padded_tokens=n)
    class Ops:
        def prepare_attention_tasks_out(self,starts,lengths,slots,tasks,positions,*attrs):
            order.append('prepare_fresh')
            assert starts.tolist()==[0,10] and lengths.tolist()==[10]
            expected=list(range(cut,active))
            if bad_slot:expected[0]=-1
            assert slots.dtype==torch.int64 and slots.tolist()==expected
            tasks.zero_();tasks[:,1]=1
            if bad_context:tasks[0,8]=1
            positions.copy_(torch.arange(10))
        def status_guard(self,*items):
            if any(bool(item.ne(0).any()) for item in items):raise RuntimeError('device metadata trap')
        def copy_validate_current_out(self,values,lse,out,out_lse,status):
            order.append('validate_copy')
            assert lse.dtype == torch.float32
            assert values.data_ptr()==workspace.query_input.data_ptr()
            assert lse.data_ptr()==workspace.rotate_status.data_ptr()
            out.copy_(values);out_lse.copy_(lse.squeeze(-1));status.zero_()
        def rotate_clip_store_striped_out(self,k,v,rk,rv,slots,positions,raw,wk,wv,tags,status,*attrs):
            order.append('store_fresh')
            assert slots.tolist()==list(range(cut,active))
            assert positions.tolist()==list(range(10))
            status.zero_()
    ops=Ops()
    class Impl:
        num_heads=h;num_kv_heads=hk;head_size=d;scale=d**-.5
        provider=SimpleNamespace(ops=ops,config={},layer_state=lambda _name:state)
        def forward(self,layer,q,k,v,cache,meta,output):
            order.append('prefix_full_cv')
            assert meta.num_reqs==1 and meta.num_actual_tokens==cut
            assert meta.query_start_loc.tolist()==[0,cut] and meta.current_only_plan is None
            assert meta.seq_lens.tolist()==[100]
            assert meta.is_draft==draft and meta.draft_index==0
            assert not meta.first_draft_current_only
            if draft:assert not use_native_current(meta)
            output.fill_(7)
    def slot_guard(ops,status,slots,active):
        checked=_populate_slot_error_status(status,slots,active)
        ops.status_guard(checked)
    monkeypatch.setattr(module,'guard_current_slots',slot_guard)
    def current(q,k,v,cumulative,**kwargs):
        order.append('native_current')
        assert cumulative==(10,) and q.shape==(10,h,d)
        return torch.full_like(q,3),torch.full((10,h,1),1.125,dtype=lse_dtype)
    monkeypatch.setattr(module,'native_current_partial',current)
    q=torch.randn(n,h*d).to(torch.bfloat16);k=torch.randn(n,hk*d).to(torch.bfloat16);v=k.clone()
    q_before=q.clone();output=torch.full_like(q,99)
    invoke=lambda:module.dispatch_current_suffix(Impl(),SimpleNamespace(layer_name='full0'),
        q,k,v,torch.empty(1),metadata,output,plan)
    if bad_context or bad_padding or bad_slot:
        with pytest.raises(RuntimeError,match='device metadata trap'):invoke()
        assert 'native_current' not in order
    else:
        assert invoke() is output
        assert order==['prefix_full_cv','prepare_fresh','native_current','validate_copy','store_fresh']
        assert torch.equal(q,q_before)
        assert bool(output[:cut].eq(7).all())
        assert bool(output[cut:active].eq(3).all())
        assert bool(output[active:].eq(0).all())
        assert bool(workspace.lse[:10].eq(1.125).all())
