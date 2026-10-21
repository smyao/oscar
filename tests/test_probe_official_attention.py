"""Archive #37-49/#126/#145/#164: v4 probe routing/domain checks on the host.

CPU spies verify wiring only. The real gate always requires NPU and its graph
replay checks use the independent v4 oracle, never an old mixed-domain golden.
"""
import copy
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tools import probe_official_attention as probe
from tools.official_fixture import make_fixture
from tools.official_reference import fixture_tasks_reference, pack_kv, rotate_bf16
from tools.probe_history_reuse import Shape


class FinalCopyOps:
    """CPU wiring spy for the frozen copy primitive, never a device gate."""
    def copy_validate_current_out(self,value,lse,out,out_lse,status):
        if hasattr(self,'events'):self.events.append('final_copy')
        source_lse=lse.squeeze(-1)
        live=torch.isfinite(source_lse)
        invalid=(~live)&(~torch.isneginf(source_lse))
        invalid|=live&(~torch.isfinite(value.float().sum(-1)))
        out.copy_(torch.where(live[...,None],value,torch.zeros_like(value)))
        out_lse.copy_(torch.where((~live)&(~torch.isneginf(source_lse)),
                                  -torch.inf,source_lse+0.0))
        status.copy_(invalid.int()*2)

    def status_guard(self,*statuses):
        assert len(statuses)==4
        if hasattr(self,'events'):self.events.append('guard')
        if hasattr(self,'guard_states'):
            self.guard_states.append([tensor.clone() for tensor in statuses])
        if any(bool(value.ne(0).any()) for value in statuses):
            raise RuntimeError('device status guard trap (host wiring spy)')


def fixture_and_tasks(length=2,context=65,recent=1,wide_history=False):
    fixture = make_fixture(torch,Shape('official_host',(length,),(context,),256,1,1,False,
                                      recent_tokens=recent,padded_tokens=length+1))
    tasks = torch.zeros((fixture['tokens']*3,16),dtype=torch.int64)
    sink=min(64,context);cut=fixture['cached_quant_ends'][0]
    for token in range(length):
        for kind,(low,high) in enumerate(((sink,context if wide_history else cut),
                                        (0,sink+context-cut),(context,context+length))):
            tasks[token*3+kind,:11] = torch.tensor([token,1,0,low,high,0,kind,kind,context,0,0])
    for kind in range(3):
        tasks[length*3+kind,:11] = torch.tensor([length,-1,0,0,0,0,kind,kind,0,0,0])
    return fixture,tasks


@pytest.mark.parametrize('family',('q4','q1','c4','c16'))
@pytest.mark.parametrize('mode',(0,1))
def test_all_four_v4_reader_abis_receive_rotated_inputs_and_explicit_mode(family,mode):
    calls=[]
    class Ops:
        def __getattr__(self,name):return lambda *args:calls.append((name,args))
    tensors={name:torch.empty(1) for name in ('q','qr','ck','cv','raw','table','wk','wv','tags')}
    tensors['rv']=torch.eye(256,dtype=torch.bfloat16)
    buffers={name:torch.empty(1) for name in ('tasks','partial','lse','status','workspace','cluster_stats')}
    fixture={'spec':SimpleNamespace(recent_tokens=256,splits=1),'blocks':2,'stride':512*144,'scale':1/16}
    probe.launch(Ops(),family,tensors,fixture,buffers,7,dequant_mode=mode,merge=False)
    name,args=calls[0]
    assert name==probe.OPERATORS[family] and '_official_' in name
    assert args[0] is tensors['q'] and args[1] is tensors['qr'] and args[4] is tensors['rv']
    assert args[-3:]==(1/16,mode,7)
    assert len(args)==(27 if probe.CLUSTER[family]>1 else 26)


@pytest.mark.parametrize('mode',(0,1))
def test_fixture_all_sources_share_rotated_frame_and_preserve_original_runtime_inputs(mode):
    fixture,tasks=fixture_and_tasks()
    data,original=fixture['cpu'],fixture['original_inputs']
    assert fixture['cache_format']==probe.CACHE_FORMAT=='striped_rotated_fp32_v3'
    assert fixture['cached_quant_ends']==[65] and fixture['window_ring_tokens']==4
    assert torch.equal(data['q'],data['qr'])
    assert torch.equal(data['q'],rotate_bf16(original['q'],fixture['rotation_k']))
    assert torch.equal(data['ck'],rotate_bf16(original['ck'],fixture['rotation_k']))
    assert torch.equal(data['cv'],rotate_bf16(original['cv'],data['rv']))
    keys,values=fixture['original_histories'][0]
    page=fixture['page_assignments'][0][0]
    torch.testing.assert_close(data['wk'][page,:64],rotate_bf16(keys[:64],fixture['rotation_k']),atol=0,rtol=0)
    torch.testing.assert_close(data['wv'][page,:64],rotate_bf16(values[:64],data['rv']),atol=0,rtol=0)
    result=fixture_tasks_reference(fixture,tasks,mode)
    assert result['contract']==probe.CONTRACT
    assert result['merged_rotated'].dtype==torch.float32 and result['output'].dtype==torch.bfloat16
    assert result['output'][-1].eq(0).all()
    assert not torch.equal(result['pre_inverse_bf16'][0],result['output'][0])


def test_prepare_uses_new_op_and_checks_complete_fixed_map_without_old_prepare():
    fixture,tasks=fixture_and_tasks()
    calls=[]
    class Ops:
        def prepare_official_attention_tasks_out(self,starts,lens,slots,out,positions,*args):
            calls.append(probe.PREPARE);out.copy_(tasks);positions.copy_(torch.tensor([65,66,-1]))
        def prepare_attention_tasks_out(self,*args):pytest.fail('old prepare cannot qualify v4')
    class TorchProxy:
        npu=SimpleNamespace(synchronize=lambda:None)
        def __getattr__(self,name):return getattr(torch,name)
    buffers={'tasks':torch.empty_like(tasks),'positions':torch.empty(3,dtype=torch.int64)}
    result=probe.prepare(TorchProxy(),Ops(),fixture['cpu'],fixture,buffers)
    assert calls==[probe.PREPARE] and result['fixed_cached_map']=='exact'
    broken=tasks.clone();broken[3,4]=64
    with pytest.raises(RuntimeError,match='omitted'):
        probe.check_task_coverage(fixture,broken)


def test_wide_history_tasks_are_checked_after_fixed_cut_mask_but_window_coverage_is_exact():
    fixture,tasks=fixture_and_tasks(context=80,recent=8,wide_history=True)
    assert fixture['cached_quant_ends']==[73]
    assert tasks[0,4]==80 and tasks[1,4]==71
    probe.check_task_coverage(fixture,tasks)
    result=fixture_tasks_reference(fixture,tasks,0)
    assert torch.isfinite(result['output'][:2]).all()
    broken=tasks.clone();broken[0,4]=72
    with pytest.raises(RuntimeError,match='omitted'):
        probe.check_task_coverage(fixture,broken)
    broken=tasks.clone();broken[1,4]=70
    with pytest.raises(RuntimeError,match='omitted'):
        probe.check_task_coverage(fixture,broken)


def test_launch_merges_fp32_rotated_partials_then_casts_once_before_inverse():
    fixture,tasks=fixture_and_tasks()
    golden=fixture_tasks_reference(fixture,tasks,0)
    buffers=probe.allocate(torch,fixture,torch.device('cpu'),1,'q4')
    calls=[]
    class Ops(FinalCopyOps):
        events=calls
        def attention_cv_official_decode_out(self,*args):
            calls.append('cv');buffers['partial'].copy_(golden['partial']);buffers['lse'].copy_(golden['lse'])
            buffers['status'].zero_()
        def merge_lse_out(self,partial,lse,out,out_lse,status):
            calls.append('merge');out.copy_(golden['merged_rotated'].view_as(out))
            out_lse.copy_(golden['output_lse'].view_as(out_lse));status.zero_()
    probe.launch(Ops(),'q4',fixture['cpu'],fixture,buffers,1,dequant_mode=0)
    assert calls==['cv','merge','final_copy','guard','guard']
    assert torch.equal(buffers['pre_inverse'],golden['pre_inverse_bf16'].reshape_as(buffers['pre_inverse']))
    assert torch.equal(buffers['bf16'],golden['output'].reshape_as(buffers['bf16']))
    assert not torch.equal(buffers['pre_inverse'],buffers['bf16'])
    assert len({buffers[name].data_ptr() for name in
                ('output','pre_inverse','inverse_result','bf16','output_lse','final_lse','final_status')})==7


def test_output_gate_checks_fp32_merge_and_final_inverse_independently():
    fixture,tasks=fixture_and_tasks()
    golden=fixture_tasks_reference(fixture,tasks,0)
    buffers=probe.allocate(torch,fixture,torch.device('cpu'),1,'q4')
    buffers['tasks'].copy_(tasks);buffers['status'].zero_();buffers['merge_status'].zero_();buffers['final_status'].zero_()
    for name,field in (('partial','partial'),('lse','lse'),('output','merged_rotated'),
                       ('output_lse','output_lse'),('pre_inverse','pre_inverse_bf16'),
                       ('inverse_result','output'),('final_lse','output_lse'),('bf16','output')):
        buffers[name].copy_(golden[field].reshape_as(buffers[name]))
    assert probe.check_outputs(torch,fixture,buffers,golden,tasks,'q4')['status']=='passed'
    buffers['bf16'].copy_(buffers['pre_inverse'])
    with pytest.raises(RuntimeError,match='bf16'):
        probe.check_outputs(torch,fixture,buffers,golden,tasks,'q4')


@pytest.mark.parametrize('failure',('inverse_nan','copy_lse_nan','source_error_priority'))
def test_final_copy_fault_is_guarded_after_original_source_and_merge_status(failure):
    fixture,tasks=fixture_and_tasks()
    golden=fixture_tasks_reference(fixture,tasks,0)
    buffers=probe.allocate(torch,fixture,torch.device('cpu'),1,'q4')
    events=[]
    class Ops(FinalCopyOps):
        guard_states=[]
        def attention_cv_official_decode_out(self,*args):
            buffers['partial'].copy_(golden['partial']);buffers['lse'].copy_(golden['lse'])
            buffers['status'].zero_()
            if failure=='source_error_priority':buffers['status'][0,0]=3
        def merge_lse_out(self,partial,lse,out,out_lse,status):
            out.copy_(golden['merged_rotated'].view_as(out))
            out_lse.copy_(golden['output_lse'].view_as(out_lse));status.zero_()
            if failure=='copy_lse_nan':out_lse[0]=float('nan')
    ops=Ops();ops.events=events
    if failure in ('inverse_nan','source_error_priority'):
        fixture['cpu']['rv'].fill_(float('nan'))
    with pytest.raises(RuntimeError,match='status guard trap'):
        probe.launch(ops,'q4',fixture['cpu'],fixture,buffers,1,dequant_mode=0)
    assert bool(buffers['final_status'].ne(0).any())
    if failure=='source_error_priority':
        assert events==['final_copy','guard']
        assert ops.guard_states[0][0][0,0]==3
    else:
        assert events==['final_copy','guard','guard']
        assert all(not bool(value.ne(0).any()) for value in ops.guard_states[0])
        assert bool(ops.guard_states[1][0].ne(0).any())


def test_no_npu_cannot_produce_a_cpu_success_receipt(tmp_path,monkeypatch):
    # The CLI owns its process environment. This host double runs in pytest,
    # so keep _active_device's writes out of later readiness tests.
    import os
    monkeypatch.setattr(os,'environ',dict(os.environ))
    config=json.loads((probe.ROOT/'configs/target.json').read_text())
    config.update(experimental_official_math=True,experimental_aligned_math=True,experimental_bf16_attention=True)
    path=tmp_path/'config.json';path.write_text(json.dumps(config));out=tmp_path/'report.json'
    def unavailable():raise RuntimeError('requires real NPU; no CPU substitution')
    monkeypatch.setattr(probe,'require_real_npu',unavailable)
    monkeypatch.delenv('ASCEND_RT_VISIBLE_DEVICES',raising=False)
    assert probe.main(['npu','--config',str(path),'--output',str(out)])==2
    report=json.loads(out.read_text())
    assert report['status']=='failed' and report['NPU']=='not_passed'
    assert report['contract']==probe.CONTRACT


def test_v3_configuration_is_rejected_before_device_access(tmp_path,monkeypatch):
    path=tmp_path/'config.json'
    path.write_text(json.dumps({'experimental_aligned_math':True,'experimental_bf16_attention':True}))
    monkeypatch.setattr(probe,'require_real_npu',lambda:pytest.fail('old configuration reached the device'))
    with pytest.raises(RuntimeError,match='experimental_official_math=True'):
        probe.run_npu(path)


def complete_receipt():
    rows=[]
    for case in probe.case_plan():
        shape=case['shape'];graph=None
        if case['graph']:
            graph={'capture':'passed','replay':'passed','same_addresses':True,'final_copy_and_guard':'passed','states':[
                {'state':name,'v4_oracle':'passed','eager_replay_bitwise':'passed'}
                for name in ('original','changed_query','changed_context')]}
        rows.append({'case':shape.name,'family':case['family'],'operator':probe.OPERATORS[case['family']],
            'prepare_operator':probe.PREPARE,'dequant_mode':case['dequant_mode'],'status':'passed',
            'all_tasks':(shape.padded_tokens or sum(shape.qlens))*shape.kv_heads*3*shape.splits,
            'check':{'status':'passed'},'timing':None,'graph':graph})
    return {'status':'passed','contract':probe.CONTRACT,'cache_format':probe.CACHE_FORMAT,'slot_bytes':144,
        **{key:'passed' for key in ('device_completion','precision','error_domain','graph_capture','graph_replay')},
        'performance':'not_run_user_requested','acceptance_sha256':probe.digest(probe.ROOT/'configs/acceptance_official_bf16.json'),
        'oracle_sha256':probe.digest(probe.ROOT/'tools/official_reference.py'),'probe_sha256':probe.digest(probe.__file__),
        'source_binding':probe.source_binding(),'operator_families':probe.OPERATORS,'cases':rows,
        'writer_rotation':{'status':'passed','writer':probe.WRITER,'query_copy':probe.QUERY_COPY,
            'codec_bytes':'exact','rotated_windows':'exact','canaries':'exact','query_status':'exact','writer_status':'exact',
            'graph_capture':'passed','graph_replay':'passed','graph_same_addresses':True}}


@pytest.mark.parametrize('failure',('v3','old_prepare','old_reader','old_window','missing_case','timing','old_graph_oracle','missing_copy_graph'))
def test_v3_or_incomplete_success_receipts_cannot_certify_v4(failure):
    report=complete_receipt();assert probe.validate_evidence(report)==[]
    if failure=='v3':report['contract']='oscar_bf16_attention_v3'
    elif failure=='old_prepare':report['cases'][0]['prepare_operator']='prepare_attention_tasks_out'
    elif failure=='old_reader':report['cases'][0]['operator']='attention_cv_aligned_decode_out'
    elif failure=='old_window':report['writer_rotation'].pop('rotated_windows')
    elif failure=='missing_case':report['cases'].pop()
    elif failure=='timing':report['cases'][0]['timing']={'passed':True}
    elif failure=='old_graph_oracle':report['cases'][0]['graph']['states'][0].pop('v4_oracle')
    elif failure=='missing_copy_graph':report['cases'][0]['graph'].pop('final_copy_and_guard')
    assert probe.validate_evidence(report)


def test_writer_gate_stores_rotated_exact_windows_and_replays_fresh_inputs():
    class Ops:
        def store_aligned_int2_out(self,rk,rv,k,v,slots,positions,raw,wk,wv,tags,status,block,blocks,prefix,stride,sink,recent,kc,vc):
            self.last=(rk,rv,k,v,slots,positions,raw,wk,wv,tags,status,block,blocks,prefix,stride,sink,recent,kc,vc)
            status.zero_()
            for token in range(rk.shape[0]):
                slot=int(slots[token]);position=int(positions[token])
                for head in range(rk.shape[1]):
                    if slot<0:continue
                    if slot>=blocks*block:status[token,head]=1;continue
                    if position<0:status[token,head]=4;continue
                    if not bool(torch.isfinite(rk[token,head]).all() and torch.isfinite(rv[token,head]).all()):
                        status[token,head]=2;continue
                    assert torch.equal(rk[token,head],k[token,head]) and torch.equal(rv[token,head],v[token,head])
                    encoded=pack_kv(rk[token,head],rv[token,head],kc,vc)
                    page,within=divmod(slot,block);offset=prefix+page*stride+(within*rk.shape[1]+head)*144
                    raw[offset:offset+144]=encoded
                    row=position if position<sink else sink+within%recent
                    wk[page,row,head]=k[token,head];wv[page,row,head]=v[token,head];tags[page,row]=within
        def copy_validate_bf16_query_out(self,x,slots,out,status,cores):
            out.copy_(x);out[slots<0]=0
            status.copy_((~torch.isfinite(x).all(-1)).int()*2);status[slots<0]=0
    ops=Ops()
    class Graph:
        def replay(self):ops.store_aligned_int2_out(*ops.last)
    class TorchProxy:
        npu=SimpleNamespace(synchronize=lambda:None,NPUGraph=Graph,graph=lambda *a,**k:nullcontext())
        def __getattr__(self,name):return getattr(torch,name)
    report=probe.writer_rotation_gate(TorchProxy(),ops,torch.device('cpu'),1)
    assert report['rotated_windows']==report['codec_bytes']=='exact'
    assert report['graph_same_addresses'] is True
