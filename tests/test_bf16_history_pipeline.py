"""Archive #126/#145/#148/#153/#154/#157: host mechanics, no NPU evidence."""
from dataclasses import asdict,replace
from enum import Enum
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from oscar_ascend.ops import reference
from tools import bf16_history_pipeline as full,probe_bf16_history as probe
from test_probe_bf16_history import CONFIG,EXECUTION,ACCEPTANCE,synthetic_evidence,task_fixture


@pytest.fixture(autouse=True)
def single_thread():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('index,splits,padded,native',[(0,20,12,False),(1,20,8,False),(2,2,352,True),
    (3,1,8208,True),(4,10,24,True),(5,1,512,True)])
def test_full_fixed_geometry_matches_runtime_selector_and_native_predicate(index,splits,padded,native):
    # True full-path shapes, without allocating NPU or claiming device execution.
    original=probe.case_plan()[index];case=full.production_case(original,EXECUTION)
    assert case.shape.heads==6 and case.shape.kv_heads==1
    assert case.shape.splits==splits and case.shape.padded_tokens==padded
    policy=full.case_policy(case)
    state=Enum('AscendAttentionState',['ChunkedPrefill','SpecDecoding'])
    metadata=SimpleNamespace(capture_origin=False,dummy_origin=False,is_draft=index==1,draft_index=1,
        attn_state=state.ChunkedPrefill if native else state.SpecDecoding)
    assert full.current.use_native_current(metadata)==policy['native_current']==native
    assert original.shape.splits in (1,3)  # independent operator diagnostics retained
    assert EXECUTION['cv_arena_bytes']['candidate']==EXECUTION['cv_arena_bytes']['baseline']*3//2


@pytest.mark.parametrize('mutation',['native_budget','missing_budget','boolean_cores','invalid_capacity','wrong_rotation'])
def test_complete_pipeline_rejects_unknown_or_invalid_runtime_capacity(mutation):
    config=copy.deepcopy(CONFIG);cores=20
    if mutation=='native_budget':config['use_native_token_budget']=True
    elif mutation=='missing_budget':config.pop('max_num_batched_tokens')
    elif mutation=='boolean_cores':cores=True
    elif mutation=='invalid_capacity':config['max_num_batched_tokens']=True
    else:config['rotation_method']='qr'
    with pytest.raises(ValueError):full.execution_contract(config,cores)


@pytest.mark.parametrize('mutation',['geometry','config','settings','diagnostic','physical','oracle','missing_native','scope','samples'])
def test_validator_binds_full_scope_geometry_settings_and_numerics(mutation):
    report=synthetic_evidence();row=report['cases'][0];receipt=row['full_attention']
    if mutation=='geometry':receipt['shape']['splits']=3
    elif mutation=='config':receipt['execution']['effective_config_sha256']='stale'
    elif mutation=='settings':receipt['settings']['k_clip_ratio']=.5
    elif mutation=='diagnostic':receipt['settings']['guard_host_sync']=True
    elif mutation=='physical':receipt['physical_page_stride_bytes']=512
    elif mutation=='oracle':receipt['numerical']['precision']['comparisons']['candidate']['partial']['passed']=False
    elif mutation=='missing_native':receipt['policy']['native_current']=True
    elif mutation=='scope':receipt['timing']['scope']='cv_only'
    else:next(iter(receipt['timing']['samples_ms'].values())).pop()
    assert full.validate_receipt(probe.case_plan()[0],row,timed=True,execution=EXECUTION)


def test_blocked_oracles_match_all_small_queries_and_bound_native_score_rows(monkeypatch):
    # Deliberately small host mechanics copy, not the production selector case.
    case=replace(probe.case_plan()[4],shape=replace(probe.case_plan()[4].shape,
        qlens=(67,),contexts=(80,),kv_heads=1,padded_tokens=72,splits=2))
    fixture=probe.make_fixture(torch,case);expected=fixture['expected'];calls=[]
    original=reference.attention
    def attention(q,k,v,**kwargs):
        calls.append((len(q),len(k)))
        return original(q,k,v,**kwargs)
    monkeypatch.setattr(reference,'attention',attention)
    full.bounded_dense_oracle(torch,fixture)
    assert len(fixture['expected'])==67 and max(n for n,_ in calls)<=32
    for index,(out,lse) in expected.items():
        torch.testing.assert_close(fixture['expected'][index][0],out)
        torch.testing.assert_close(fixture['expected'][index][1],lse)
    tasks=task_fixture(fixture)
    # CPU-only descriptor fixture: production helper correctly rejects CPU.
    current_rows=(tasks[:,7]==2)&(tasks[:,10]==0)
    tasks[current_rows,4]=tasks[current_rows,3]
    calls.clear();gold=full.full_oracle(torch,fixture,tasks,{'native_current':True})
    assert max(n for n,_ in calls)<=32
    for begin,q,k,v,ck,cv,dk,dv in fixture['logical'].values():
        dense=original(q,ck,cv,scale=fixture['scale'],causal=True)
        torch.testing.assert_close(gold['partial'][begin:begin+len(q),:,4],dense.output)
        torch.testing.assert_close(gold['lse'][begin:begin+len(q),:,4],dense.lse)
    assert bool((gold['partial'][:,:,5]==0).all()) and bool(torch.isneginf(gold['lse'][:,:,5]).all())


@pytest.mark.parametrize('native,slot_context',[(True,False),(False,True)])
def test_launch_order_uses_current_guard_source2_and_read_before_write(monkeypatch,native,slot_context):
    events=[];n=4;h=6;s=2;spec=probe.reuse.Shape('host_mechanics',(2,),(80,),256,1,s,False,padded_tokens=n,slot_context=slot_context)
    fixture={'spec':spec,'tokens':n,'actual_tokens':2,'current_cumulative':(2,), 'blocks':1,'stride':512*136,'scale':1/16}
    tensors={key:torch.empty(1) for key in ('raw','wk','wv','tags','rv','table','starts','lens','rk_transpose','rv_transpose')}
    tensors.update({key:torch.zeros((n,1 if key in ('ck','cv') else h,256),dtype=torch.bfloat16) for key in ('q','ck','cv')})
    tensors.update(qr=torch.zeros(n,h,256),slots=torch.tensor([0,1,-1,-1]))
    buffers={'tasks':torch.zeros(n*3*s,16,dtype=torch.int64),'positions':torch.zeros(n,dtype=torch.int64),
        'partial':torch.zeros(n,h,3*s,256),'lse':torch.full((n,h,3*s),-float('inf')),
        'output':torch.empty(n*h,256),'output_lse':torch.empty(n*h),'bf16':torch.empty(n*h,256,dtype=torch.bfloat16),
        'merge_status':torch.zeros(n*h,dtype=torch.int32),'status':torch.zeros(n*3*s,2,dtype=torch.int32),
        'rotate_status':torch.zeros(n,h,dtype=torch.int32),'store_status':torch.zeros(n,1,dtype=torch.int32)}
    def prepare(*args):
        events.append('prepare');assert args[-1] is slot_context
        assert args[-2] is (tensors['table'] if slot_context else None)
    def cv(*args,**kwargs):events.append('cv');assert kwargs=={'merge':False}
    def guard(ops,status,slots,active):
        events.append('current_guard');assert status.shape==(n,h) and active==2
    def native_fia(*args,**kwargs):events.append('native');return torch.ones(2,h,256,dtype=torch.bfloat16),torch.zeros(2,h,1)
    def write(partial,lse,out,native_lse,active,splits):
        # Host mechanics stub; actual NPU helper remains unchanged/device-only.
        events.append('write_current');partial[:active,:,2*splits].copy_(out);lse[:active,:,2*splits].copy_(native_lse.squeeze(-1))
    def merge(partial,lse,out,*args):
        events.append('merge')
        if native:assert bool((partial.view(n,h,3*s,256)[:2,:,4]==1).all())
        assert bool((partial.view(n,h,3*s,256)[:,:,5]==0).all());out.fill_(2)
    def store(*args):
        events.append('store');assert args[6] is tensors['raw'] and args[7] is tensors['wk']
        assert bool((buffers['bf16']==2).all())  # final copy preceded publication
    ops=SimpleNamespace(prepare_attention_tasks_out=prepare,rotate_vector_init_out=lambda *a:events.append('rotate'),
        merge_lse_out=merge,rotate_clip_store_striped_out=store)
    monkeypatch.setattr(probe,'launch',cv)
    monkeypatch.setattr(full.current,'suppress_current_source_tasks',lambda *a:events.append('suppress'))
    monkeypatch.setattr(full.current,'guard_current_slots',guard)
    monkeypatch.setattr(full.current,'native_current_partial',native_fia)
    monkeypatch.setattr(full.current,'write_current_partial',write)
    monkeypatch.setattr(full,'run_status_guard',lambda *a,**k:events.append('status_guard'))
    full.launch_pipeline(ops,fixture,tensors,buffers,20,name='diagnostic',cluster=1,
        policy={'native_current':native},settings=EXECUTION['settings'])
    assert events==(['prepare','suppress','rotate','cv','current_guard','native','write_current','merge','store','status_guard']
        if native else ['prepare','rotate','cv','merge','store','status_guard'])


def test_full_path_failure_blocks_every_timing_scope(monkeypatch):
    report=synthetic_evidence();rows=report['cases'];calls=[]
    for row in rows:row['timing']=None
    def numerical(t,o,case,d,c,config,a,row):
        receipt=copy.deepcopy(row['full_attention']);receipt['timing']=None
        if case.route=='balanced':receipt['numerical']['precision']['passed']=False
        return receipt
    monkeypatch.setattr(full,'numerical_case',numerical)
    monkeypatch.setattr(probe,'time_case',lambda *a,**k:calls.append('cv'))
    monkeypatch.setattr(full,'timing_case',lambda *a,**k:calls.append('full'))
    result=probe.performance_phase(None,None,rows,report['cpu_screen'],None,20,ACCEPTANCE,CONFIG)
    assert result['status']=='not_run_numerical_gate_failed' and calls==[]


def test_diagnostic_guard_environment_blocks_every_timing_scope(monkeypatch):
    report=synthetic_evidence();calls=[]
    def numerical(t,o,case,d,c,config,a,row):
        receipt=copy.deepcopy(row['full_attention']);receipt['settings']['guard_host_sync']=True;return receipt
    monkeypatch.setattr(full,'numerical_case',numerical)
    monkeypatch.setattr(probe,'time_case',lambda *a,**k:calls.append('cv'))
    monkeypatch.setattr(full,'timing_case',lambda *a,**k:calls.append('full'))
    result=probe.performance_phase(None,None,report['cases'],report['cpu_screen'],None,20,ACCEPTANCE,CONFIG)
    assert result['status']=='not_run_numerical_gate_failed' and calls==[]


def test_full_case_rejects_active_rows_outside_explicit_history_capacity():
    execution=full.execution_contract(dict(CONFIG,history_token_capacity=512),20)
    with pytest.raises(ValueError,match='history arena'):
        full.production_case(probe.case_plan()[3],execution)


def test_full_timing_restores_same_input_before_every_single_step_event(monkeypatch):
    # Fake clock/launch only verifies timing orchestration, not device speed.
    report=synthetic_evidence();row=report['cases'][0];case=probe.case_plan()[0]
    row['full_attention']['numerical']['task_sha256']='tasks'
    events=[];pair=[{'name':name,'tensors':{},'buffers':{}} for name in probe.ROUTES[case.route][:2]]
    fixture={'execution':EXECUTION};policy=row['full_attention']['policy'];settings=row['full_attention']['settings']
    monkeypatch.setattr(full,'build_pair',lambda *a:(fixture,pair,policy,settings))
    monkeypatch.setattr(full,'check_pair',lambda *a:{'passed':True,'task_sha256':'tasks'})
    monkeypatch.setattr(full,'reset_arm',lambda t,arm:events.append(('reset',arm['name'])))
    monkeypatch.setattr(full,'snapshot_arm',lambda arm:{})
    monkeypatch.setattr(full,'same_arm',lambda t,r,arm:events.append(('compare',arm['name'])))
    active={'name':None,'count':0}
    def launch(*args,**kwargs):
        active['name']=kwargs['name'];events.append(('launch',kwargs['name']))
    monkeypatch.setattr(full,'launch_pipeline',launch)
    class Event:
        def __init__(self,**kw):self.is_start=active['count']%2==0;active['count']+=1
        def record(self):events.append(('start' if self.is_start else 'end',None))
        def synchronize(self):events.append(('sync',None))
        def elapsed_time(self,end):return 12. if active['name']==pair[0]['name'] else 10.
    backend=SimpleNamespace(npu=SimpleNamespace(Event=Event,synchronize=lambda:None))
    timing=full.timing_case(backend,None,case,None,20,CONFIG,ACCEPTANCE,row)
    assert timing['gate']=='passed' and all(len(v)==5 for v in timing['samples_ms'].values())
    starts=[i for i,event in enumerate(events) if event[0]=='start'];assert len(starts)==14
    for index in starts:
        assert [event[0] for event in events[index-1:index+5]]==['reset','start','launch','end','sync','compare']
        assert events[index-1][1]==events[index+1][1]==events[index+4][1]


def test_configuration_binding_changes_with_model_bytes_and_effective_budget(tmp_path):
    model=tmp_path/'model';model.mkdir();source=model/'config.json';source.write_text('{"model":"first"}')
    config=dict(CONFIG,model=str(model));first=probe.configuration_binding(config,20)
    source.write_text('{"model":"second"}')
    second=probe.configuration_binding(config,20)
    assert first['model_config_sha256']!=second['model_config_sha256']
    third=probe.configuration_binding(dict(config,max_num_batched_tokens=32768),20)
    assert third['execution']['effective_config_sha256']!=second['execution']['effective_config_sha256']
    assert third['execution']['workspace_geometry']['tokens']==32768
    assert third['probe_physical_device']==CONFIG['devices'][0]
