"""Archive #13-20/#126/#148/#164: V4 routing and finite-precision frame.

Host ops doubles verify actual Python integration; they are not device gates.
Independent scalar cases exercise native BF16 matmul and final rounding.
"""
from types import SimpleNamespace as NS

import pytest
import torch

from oscar_ascend.integration.official_math import rotate_inputs, store_rotated, finalize_output, guard_final_output
from oscar_ascend.runtime import GraphWorkspace, WorkspaceGeometry


@pytest.mark.parametrize('n,later,name',[
    (8,False,'attention_cv_official_decode_out'),
    (8,True,'attention_cv_official_decode_q1_out'),
    (256,False,'attention_cv_official_cluster4_out'),
    (8192,False,'attention_cv_official_cluster16_out'),
])
def test_actual_candidate_forward_uses_rotated_sources_and_one_final_inverse(n,later,name):
    from test_runtime import test_forward_dispatches_native_draft_strides_and_int32_slots_in_order
    test_forward_dispatches_native_draft_strides_and_int32_slots_in_order(
        n,1,1,n,16,later,True,True,True,_member_pipeline=True,_c16_l1=True,
        _bf16_expected=name,_aligned=True,_official=True)


def workspace(n=2,h=6,hk=1,d=256):
    w=object.__new__(GraphWorkspace)
    w.geometry=WorkspaceGeometry(n,h,hk,d,aligned_math=True,official_math=True)
    w.query_input=torch.empty(n,h,d,dtype=torch.bfloat16)
    w.key_input=torch.empty(n,hk,d,dtype=torch.bfloat16)
    w.value_input=torch.empty_like(w.key_input)
    w.query_rot=torch.empty_like(w.query_input)
    w.query_rot_bf16=torch.empty_like(w.query_input)
    w.key_rot_bf16=torch.empty_like(w.key_input)
    w.value_rot_bf16=torch.empty_like(w.key_input)
    w.rotate_status=torch.empty(n,h,dtype=torch.int32)
    w.final_status=torch.empty(n,h,dtype=torch.int32)
    w.final_lse=torch.empty(n,h)
    w.store_status=torch.empty(n,hk,dtype=torch.int32)
    return w


class Ops:
    def copy_validate_bf16_query_out(self,x,slots,out,status,cores):
        out.copy_(x);status.zero_()
    def copy_validate_current_out(self,x,lse,out,out_lse,status):
        out.copy_(x);out_lse.copy_(lse.squeeze(-1));status.zero_()
        if not torch.isfinite(x).all():status.fill_(2)
    def status_guard(self,*statuses):
        if any(bool(s.ne(0).any()) for s in statuses):raise RuntimeError('final inverse trap')


def state(w):
    d=w.geometry.head_dim
    r=torch.eye(d,dtype=torch.float32)
    r[:2,:2]=torch.tensor([[.6,-.8],[.8,.6]])
    return NS(workspace=w,rotation_k_bf16=r.bfloat16(),rotation_v_bf16=r.bfloat16(),
        cache_format='striped_rotated_fp32_v3',raw=object(),window_key=object(),
        window_value=object(),window_tags=object(),num_blocks=1,
        spec=NS(block_size=2688,conv_bytes=30720,ssm_bytes=393216),
        snapshots=NS(sink_tokens=64,ring_tokens=259))


def test_rotated_window_writer_reuses_current_rotations_without_reapplying_them():
    w=workspace();s=state(w);ops=Ops()
    q=torch.ones_like(w.query_input);k=torch.ones_like(w.key_input);v=k*2
    qr,kr,vr=rotate_inputs(ops,s,q,k,v,torch.arange(2))
    expected_k=torch.matmul(k,s.rotation_k_bf16)
    expected_v=torch.matmul(v,s.rotation_v_bf16)
    assert torch.equal(qr,torch.matmul(q,s.rotation_k_bf16))
    def store(rk,rv,wk,wv,*args):
        assert rk.data_ptr()==wk.data_ptr() and rv.data_ptr()==wv.data_ptr()
        assert torch.equal(rk,expected_k) and torch.equal(rv,expected_v)
        assert not torch.equal(wk,k) and args[-2:]==(.96,.92)
    ops.store_aligned_int2_out=store
    store_rotated(ops,s,k,v,torch.arange(2),torch.arange(2),{},rotated_kv=(kr,vr))


def test_final_inverse_rounds_only_after_fp32_merge_and_preserves_fixed_buffers():
    w=workspace(n=1);s=state(w);s.rotation_v_bf16=torch.eye(256,dtype=torch.bfloat16)
    # Independent analytical 3+2 split case: FP32 weighted merge rounds to1,
    # while premature per-segment BF16 rounding yields1.0078125.
    merged=torch.full((1,6,256),(3*(1+1+1.015625)/3+2)/5,dtype=torch.float32)
    output=torch.empty_like(w.query_input)
    addresses=[b.data_ptr() for b in (w.query_input,w.query_rot_bf16,w.final_status,w.final_lse)]
    for value in (merged,torch.zeros_like(merged)):
        finalize_output(Ops(),s,value,torch.zeros(1,6),output)
        guard_final_output(Ops(),s,1,capture=True)
        assert torch.equal(output,value.bfloat16())
        assert addresses==[b.data_ptr() for b in (w.query_input,w.query_rot_bf16,w.final_status,w.final_lse)]
    s.rotation_v_bf16.fill_(float('nan'))
    finalize_output(Ops(),s,merged,torch.zeros(1,6),output)
    assert bool(w.final_status.ne(0).all())
    with pytest.raises(RuntimeError,match='final inverse trap'):
        guard_final_output(Ops(),s,1,capture=True)


def test_official_current_buffers_cover_full_input_even_with_bounded_history_capacity():
    geometry=WorkspaceGeometry(16384,6,1,256,history_token_capacity=128,
        aligned_math=True,official_math=True)
    assert geometry.buffer_bytes['query_rot_bf16']==16384*6*256*2
    assert geometry.buffer_bytes['query_rot']==128*6*256*2
    assert geometry.buffer_bytes['final_status']==geometry.buffer_bytes['final_lse']==16384*6*4


@pytest.mark.parametrize('draft',[False,True])
def test_current_only_actual_dispatch_rotates_and_inverts_the_single_exact_token(monkeypatch,draft):
    from oscar_ascend.integration import current_only_dispatch as dispatch
    from oscar_ascend.integration.current_only_plan import plan_current_suffix
    from oscar_ascend.integration.metadata import OscarMetadata
    w=workspace(n=1);s=state(w);h,d=6,256
    w.slots=torch.empty(1,dtype=torch.int64)
    w.tasks=torch.empty(3,16,dtype=torch.int64)
    w.attention_status=torch.empty(3,2,dtype=torch.int32)
    w.positions=torch.empty(1,dtype=torch.int64)
    w.merge_status=torch.empty(1,h,dtype=torch.int32)
    w.lse=torch.empty(1,h)
    s.snapshots.recent_tokens=256
    events=[]
    q=torch.zeros(1,h*d,dtype=torch.bfloat16)
    k=torch.zeros(1,d,dtype=torch.bfloat16)
    v=torch.zeros_like(k);v[0,:2]=1
    class CurrentOps(Ops):
        def prepare_official_attention_tasks_out(self,starts,lengths,slots,tasks,positions,*geometry):
            tasks.zero_();tasks[:,1]=1;positions.zero_();events.append('prepare')
        def store_aligned_int2_out(self,rk,rv,wk,wv,*args):
            assert rk.data_ptr()==wk.data_ptr() and rv.data_ptr()==wv.data_ptr()
            assert torch.equal(rv,torch.matmul(v.view(1,1,d),s.rotation_v_bf16))
            args[6].zero_();events.append('store')
    ops=CurrentOps()
    impl=NS(num_heads=h,num_kv_heads=1,head_size=d,scale=d**-.5,
        provider=NS(ops=ops,config={'experimental_official_math':True},layer_state=lambda _:s))
    def native_current(qr,kr,vr,cumulative,**kwargs):
        assert qr.data_ptr()==w.query_input.data_ptr()
        assert vr.data_ptr()==w.value_rot_bf16.data_ptr()
        assert cumulative==(1,)
        events.append('fia_rotated')
        return vr.expand(1,h,d).contiguous(),torch.zeros(1,h,1)
    monkeypatch.setattr(dispatch,'native_current_partial',native_current)
    monkeypatch.setattr(dispatch,'guard_current_slots',lambda ops,status,slots,n:status[:n].zero_())
    meta=OscarMetadata(torch.tensor([0,1],dtype=torch.int32),torch.tensor([1],dtype=torch.int32),
        torch.zeros(1,1,dtype=torch.int32),torch.tensor([0],dtype=torch.int32),1,1,1,1,
        num_input_tokens=1,query_start_loc_cpu=torch.tensor([0,1]),current_cumulative=(1,),
        is_draft=draft,first_draft_current_only=draft)
    plan=plan_current_suffix((0,1),(1,),actual_tokens=1,padded_tokens=1)
    output=torch.empty_like(q)
    dispatch.dispatch_current_suffix(impl,NS(layer_name='test'),q,k,v,object(),meta,output,plan)
    assert events==['prepare','fia_rotated','store']
    # Exact 2x2 rotation-roundtrip from the pinned official Torch functions.
    assert torch.equal(output.view(1,h,d)[0,:,:2],torch.full((h,2),1.0078125,dtype=torch.bfloat16))
    assert torch.count_nonzero(output.view(1,h,d)[0,:,2:])==0


@pytest.mark.parametrize('missing',[
    'prepare_official_attention_tasks_out','attention_cv_official_decode_out',
    'attention_cv_official_decode_q1_out','attention_cv_official_cluster4_out',
    'attention_cv_official_cluster16_out','copy_validate_current_out'])
def test_missing_v4_operator_fails_before_npu_availability(monkeypatch,missing):
    import json
    from pathlib import Path
    from oscar_ascend.runtime import AscendRuntimeProvider
    from oscar_ascend.ops import loader
    from tools.serving_variants import variant_config
    config=variant_config(json.loads(Path('configs/target.json').read_text()),'candidate')
    config['experimental_current_only']=False
    provider=AscendRuntimeProvider(config)
    monkeypatch.setenv('ASCEND_RT_VISIBLE_DEVICES','0,1,2,3')
    monkeypatch.setattr(loader,'require_production_ops',lambda:None)
    def require(names):
        if missing in names:raise loader.OperatorUnavailable(missing)
    monkeypatch.setattr(loader,'require_capabilities',require)
    monkeypatch.setattr(torch,'npu',NS(is_available=lambda:pytest.fail('must reject missing capability first')),raising=False)
    with pytest.raises(loader.OperatorUnavailable,match=missing):provider.assert_ready()
