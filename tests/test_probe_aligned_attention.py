"""Archive #126/#145/#148/#164: v3 probe wiring and rejection on the host."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from tools import probe_aligned_attention as probe
from tools import probe_current_only as shared


@pytest.mark.parametrize('family', ('q4','q1','c4','c16'))
@pytest.mark.parametrize('mode', (0,1))
def test_all_four_new_abis_bind_explicit_dequant_mode_before_cores(family,mode):
    calls=[]
    class Ops:
        def __getattr__(self,name):return lambda *args:calls.append((name,args))
    tensors={k:torch.empty(1) for k in ('q','qr','ck','cv','raw','table','wk','wv','tags')}
    tensors['rv']=torch.eye(256,dtype=torch.bfloat16)
    tensors['qr']=torch.empty(1,dtype=torch.bfloat16)
    buffers={k:torch.empty(1) for k in ('tasks','partial','lse','status','workspace','cluster_stats')}
    fixture={'spec':SimpleNamespace(recent_tokens=256,splits=1),'blocks':2,'stride':512*144,'scale':1/16}
    probe.launch(Ops(),family,tensors,fixture,buffers,7,dequant_mode=mode,merge=False)
    name,args=calls[0]
    assert name==probe.OPERATORS[family] and args[4] is tensors['rv'] and args[1].dtype==torch.bfloat16
    assert len(args)==(27 if probe.CLUSTER[family]>1 else 26)
    assert args[-3:]==(1/16,mode,7)
    if probe.CLUSTER[family]>1:assert args[15] is buffers['cluster_stats']


def test_fault_domain_retains_h16_lanes_inside_cluster_h32_unpack():
    tasks=torch.zeros(4,16,dtype=torch.int64)
    tasks[0,:11]=torch.tensor([0,21,0,64,385,0,0,0,641,0,0])
    tasks[1,:11]=torch.tensor([21,21,0,64,406,0,0,0,641,0,0])
    spec=SimpleNamespace(heads=6,kv_heads=1,corrupt_metadata=True,corrupt_position=80,corrupt_qr=False)
    fixture={'spec':spec}
    assert probe.expected_fault_status(torch,fixture,tasks,'c16').tolist()==[[0,3],[0,3],[0,0],[0,0]]
    assert probe.expected_fault_status(torch,fixture,tasks,'q4').tolist()==[[3,0],[3,0],[0,0],[0,0]]
    spec.corrupt_metadata=False;spec.corrupt_qr=True;spec.corrupt_qr_token=22
    assert probe.expected_fault_status(torch,fixture,tasks,'c4').tolist()==[[0,0],[2,0],[0,0],[0,0]]


@pytest.mark.parametrize('mode',(0,1))
def test_new_fixture_uses_144b_original_input_codec_and_all_source_oracle(mode):
    from tools.aligned_fixture import make_fixture
    from tools.aligned_reference import fixture_tasks_reference,rotate_bf16
    from tools.probe_history_reuse import Shape,PREFIX,BLOCK_TOKENS
    shape=Shape('aligned_small',(1,),(65,),256,1,1,False,recent_tokens=1,padded_tokens=2)
    fixture=make_fixture(torch,shape)
    assert fixture['stride']==BLOCK_TOKENS*144
    assert fixture['cpu']['raw'].numel()==PREFIX+fixture['blocks']*fixture['stride']
    assert fixture['cache_format']=='striped_fp32_v2' and fixture['cpu']['rv'].dtype==torch.bfloat16
    assert fixture['cpu']['qr'].dtype==torch.bfloat16
    assert torch.equal(fixture['cpu']['qr'],rotate_bf16(fixture['cpu']['q'],fixture['rotation_k']))
    tasks=torch.zeros(6,16,dtype=torch.int64)
    for kind,(lo,hi) in enumerate(((64,65),(0,64),(65,66))):
        tasks[kind,:11]=torch.tensor([0,1,0,lo,hi,0,kind,kind,65,0,0])
        tasks[3+kind,:11]=torch.tensor([1,-1,0,0,0,0,kind,kind,0,0,0])
    result=fixture_tasks_reference(fixture,tasks,mode)
    assert result['contract']==probe.CONTRACT and result['all_task_slots']==6
    assert torch.isfinite(result['output'][0]).all() and result['output'][1].eq(0).all()
    sampled=shared._expected(fixture,{'experimental_bf16_attention':True,'experimental_aligned_math':True},native_current=False)
    # Initial lifecycle metadata is explicit prefill/mode0.
    if mode==0:
        torch.testing.assert_close(sampled[0][0],result['output'][0],atol=0,rtol=0)


def test_no_npu_never_becomes_cpu_substitute(tmp_path,monkeypatch):
    config=json.loads((probe.ROOT/'configs/target.json').read_text())
    config.update(experimental_aligned_math=True,experimental_bf16_attention=True)
    path=tmp_path/'config.json';path.write_text(json.dumps(config));out=tmp_path/'report.json'
    def unavailable():raise RuntimeError('requires real NPU; no CPU substitution')
    monkeypatch.setattr(probe,'require_real_npu',unavailable)
    monkeypatch.delenv('ASCEND_RT_VISIBLE_DEVICES',raising=False)
    assert probe.main(['npu','--config',str(path),'--output',str(out)])==2
    report=json.loads(out.read_text())
    assert report['status']=='failed' and report['NPU']=='not_passed'
    assert report['contract']==probe.CONTRACT and report['performance']=='not_run_user_requested'


def test_old_config_rejected_before_device_access(tmp_path,monkeypatch):
    path=tmp_path/'config.json';path.write_text(json.dumps({'experimental_bf16_attention':True}))
    monkeypatch.setattr(probe,'require_real_npu',lambda:pytest.fail('must not access device'))
    with pytest.raises(RuntimeError,match='experimental_aligned_math=True'):probe.run_npu(path)


def test_source_binding_includes_real_abi_layout_and_new_oracle():
    binding=probe.source_binding()
    assert {'csrc/aligned_attention_bindings.cpp','csrc/aligned_store_bindings.cpp',
        'oscar_ascend/integration/aligned_math.py','oscar_ascend/integration/cache_storage_ops.py',
        'oscar_ascend/integration/specs.py','oscar_ascend/layout.py',
        'tools/aligned_reference.py','tools/aligned_runtime_oracle.py','configs/acceptance_aligned_bf16.json'}<=binding.keys()
    assert all(len(value)==64 for value in binding.values())


def complete_receipt():
    rows=[]
    for case in probe.case_plan():
        shape=case['shape'];graph=None
        if case['graph']:
            graph={'capture':'passed','replay':'passed','same_addresses':True,'states':[
                {'state':name,'v3_oracle':'passed','eager_replay_bitwise':'passed'}
                for name in ('original','changed_query','changed_context')]}
        rows.append({'case':shape.name,'family':case['family'],'operator':probe.OPERATORS[case['family']],
            'dequant_mode':case['dequant_mode'],'status':'passed','all_tasks':(shape.padded_tokens or sum(shape.qlens))*shape.kv_heads*3*shape.splits,
            'check':{'status':'passed'},'timing':None,'graph':graph})
    return {'status':'passed','contract':probe.CONTRACT,'cache_format':probe.CACHE_FORMAT,'slot_bytes':144,
        **{key:'passed' for key in ('device_completion','precision','error_domain','graph_capture','graph_replay')},
        'performance':'not_run_user_requested','acceptance_sha256':probe.digest(probe.ROOT/'configs/acceptance_aligned_bf16.json'),
        'oracle_sha256':probe.digest(probe.ROOT/'tools/aligned_reference.py'),'probe_sha256':probe.digest(probe.__file__),
        'source_binding':probe.source_binding(),'operator_families':probe.OPERATORS,'cases':rows,
        'writer_rotation':{'status':'passed','writer':probe.WRITER,'query_copy':probe.QUERY_COPY,
            'codec_bytes':'exact','original_windows':'exact','canaries':'exact','query_status':'exact','writer_status':'exact',
            'graph_capture':'passed','graph_replay':'passed','graph_same_addresses':True}}


@pytest.mark.parametrize('failure',('contract','format','old_writer','missing_case','wrong_mode','timing','graph','source'))
def test_success_words_or_v2_receipt_do_not_certify_v3(failure):
    report=complete_receipt();assert probe.validate_evidence(report)==[]
    if failure=='contract':report['contract']='oscar_bf16_attention_v2'
    elif failure=='format':report['slot_bytes']=136
    elif failure=='old_writer':report['writer_rotation']['writer']='rotate_clip_store_striped_out'
    elif failure=='missing_case':report['cases'].pop()
    elif failure=='wrong_mode':report['cases'][0]['dequant_mode']=1
    elif failure=='timing':report['cases'][0]['timing']={'passed':True}
    elif failure=='graph':report['cases'][0]['graph']['states'].pop()
    elif failure=='source':report['source_binding']={}
    assert probe.validate_evidence(report)


def test_writer_probe_geometry_status_and_fixed_address_replay_on_host():
    # CPU implementations below are wiring spies, not device substitutes:
    # run_npu still refuses to use them and no device receipt is emitted.
    from contextlib import nullcontext
    from tools.aligned_reference import pack_kv
    class Ops:
        def store_aligned_int2_out(self,rk,rv,k,v,slots,positions,raw,wk,wv,tags,status,block,blocks,prefix,stride,sink,recent,kc,vc):
            self.last=(rk,rv,k,v,slots,positions,raw,wk,wv,tags,status,block,blocks,prefix,stride,sink,recent,kc,vc)
            assert status.shape==rk.shape[:2] and recent<=block
            assert rk.dtype==rv.dtype==k.dtype==v.dtype==torch.bfloat16
            assert stride>=block*rk.shape[1]*144
            status.zero_()
            for token in range(rk.shape[0]):
                slot=int(slots[token]);position=int(positions[token])
                for head in range(rk.shape[1]):
                    if slot<0:continue
                    if slot>=blocks*block:status[token,head]=1;continue
                    if position<0:status[token,head]=4;continue
                    if not bool(torch.isfinite(rk[token,head]).all() and torch.isfinite(rv[token,head]).all()):
                        status[token,head]=2;continue
                    packed=pack_kv(rk[token,head],rv[token,head],kc,vc)
                    page,within=divmod(slot,block);offset=prefix+page*stride+(within*rk.shape[1]+head)*144
                    raw[offset:offset+144]=packed
                    if position<sink:
                        wk[page,position,head]=k[token,head];wv[page,position,head]=v[token,head]
                        tags[page,position]=within
        def copy_validate_bf16_query_out(self,x,slots,out,status,cores):
            assert out.shape==x.shape and status.shape==x.shape[:2]
            assert out.dtype==x.dtype==torch.bfloat16
            out.copy_(x);out[slots<0]=0
            status.copy_((~torch.isfinite(x).all(-1)).int()*2);status[slots<0]=0
    ops=Ops()
    class Graph:
        def replay(self):ops.store_aligned_int2_out(*ops.last)
    class TorchProxy:
        npu=SimpleNamespace(synchronize=lambda:None,NPUGraph=Graph,graph=lambda *a,**k:nullcontext())
        def __getattr__(self,name):return getattr(torch,name)
    report=probe.writer_rotation_gate(TorchProxy(),ops,torch.device('cpu'),1)
    assert report['writer_status']==report['codec_bytes']=='exact'
    assert report['graph_same_addresses'] is True


def test_lifecycle_state_allocates_new_rotations_and_codec_without_real_device(monkeypatch):
    import sys
    from types import ModuleType
    module=ModuleType('oscar_ascend.integration.impl')
    module.OscarAttentionImpl=type('HostShapeOnlyImpl',(),{})
    monkeypatch.setitem(sys.modules,module.__name__,module)
    from oscar_ascend import runtime
    def workspace(geometry,device):
        assert geometry.aligned_math is True
        return SimpleNamespace(geometry=geometry,
            query_rot_bf16=torch.empty(geometry.tokens,geometry.query_heads,geometry.head_dim,dtype=torch.bfloat16),
            key_rot_bf16=torch.empty(geometry.tokens,geometry.kv_heads,geometry.head_dim,dtype=torch.bfloat16))
    monkeypatch.setattr(runtime,'GraphWorkspace',workspace)
    from tools.aligned_fixture import make_fixture
    from tools.probe_history_reuse import Shape
    from tools.serving_variants import variant_config
    config=variant_config(json.loads((probe.ROOT/'configs/target.json').read_text()),'candidate')
    fixture=make_fixture(torch,Shape('aligned_lifecycle_state',(1,),(65,),256,1,1,False,recent_tokens=1))
    impl,state=shared._runtime(torch,fixture,fixture['cpu'],torch.device('cpu'),1,config,False)
    assert state.cache_format=='striped_fp32_v2'
    assert state.rotation_k_bf16.dtype==state.rotation_v_bf16.dtype==torch.bfloat16
    assert state.workspace.query_rot_bf16 is not None and state.workspace.key_rot_bf16 is not None
    assert state.raw.numel()==fixture['blocks']*fixture['stride']
    assert impl.provider.config['experimental_aligned_math'] is True


def test_aligned_prefix_layout_and_reader_use_new_metadata_width():
    from tools import probe_prefix_cache as prefix
    from tools.serving_variants import variant_config
    cfg=variant_config(json.loads((probe.ROOT/'configs/target.json').read_text()),'candidate')
    cfg['enable_prefix_caching']=True
    layout=prefix.fixture_layout(aligned=True)
    assert layout.page.slot_layout.metadata_bytes==4 and layout.page.slot_layout.slot_bytes==144
    assert layout.page.block_size==2304
    assert prefix.selected_reader(cfg)==probe.OPERATORS['q4']
    state=prefix.allocate(torch,torch.device('cpu'),1,bf16=True,aligned=True)
    assert state['store_status'].shape==(8,1)
    generator=torch.Generator().manual_seed(99)
    logical=tuple(tuple(torch.randn(320,1,256,generator=generator).bfloat16() for _ in range(2)) for _ in range(2))
    inputs=prefix.make_inputs(torch,logical,((3,),(2,)),18)
    output,lse=prefix.oracle(torch,logical,inputs,bf16=True,aligned=True)
    assert torch.isfinite(output).all() and torch.isfinite(lse).all()


@pytest.mark.parametrize('field,value', [('math_contract','oscar_bf16_attention_v2'),('aligned_math',False),('cache_format','striped_v1')])
def test_lifecycle_parent_rejects_legacy_receipt_for_aligned_config(field,value):
    from tools.observe_serve import _lifecycle_precision_mismatches
    config={'experimental_bf16_attention':True,'experimental_aligned_math':True}
    report=shared.bind_aligned_evidence(shared.precision_fields(config),config)
    assert _lifecycle_precision_mismatches(report,config)==[]
    report[field]=value
    assert field in _lifecycle_precision_mismatches(report,config)


def test_normal_bf16_start_removes_inherited_npu_event_control():
    from tools.target_cli import target_env
    base={'OSCAR_PASSIVE_TIMING_CONTROL':'/tmp/old-control.json'}
    config={'devices':[0,1,2,3],'experimental_bf16_attention':True,'experimental_aligned_math':True}
    assert 'OSCAR_PASSIVE_TIMING_CONTROL' not in target_env(config,base)
    assert target_env({**config,'diagnose_guard':True},base)['OSCAR_PASSIVE_TIMING_CONTROL']==base['OSCAR_PASSIVE_TIMING_CONTROL']
    assert target_env({'devices':[0,1,2,3]},base)['OSCAR_PASSIVE_TIMING_CONTROL']==base['OSCAR_PASSIVE_TIMING_CONTROL']


def test_aligned_default_probe_only_uses_new_gate_and_no_legacy_writer_or_timing(tmp_path,monkeypatch):
    import os
    from tools import observe_serve as observe
    seen=[]
    for name in dir(observe):
        if name.startswith('_') and name.endswith('_gate'):
            monkeypatch.setattr(observe,name,lambda *a,_name=name,**k:seen.append(_name))
    def preflight(*args):
        assert 'OSCAR_PASSIVE_TIMING_CONTROL' not in os.environ
        assert 'OSCAR_PASSIVE_TIMING_CONTROL' not in args[2]
    monkeypatch.setattr(observe,'_preflight',preflight)
    monkeypatch.setattr(observe,'_terminal',lambda *a,**k:None)
    monkeypatch.setenv('OSCAR_PASSIVE_TIMING_CONTROL','/tmp/inherited-do-not-arm')
    monkeypatch.setattr(observe,'managed_server',lambda *a,**k:pytest.fail('probe-only cannot serve'))
    log=tmp_path/'run'
    assert observe.run(probe.ROOT/'configs/target.json',log,'candidate',probe_only=True)==0
    assert '_aligned_attention_gate' in seen
    assert '_bf16_attention_gate' not in seen and '_striped_cache_gate' not in seen
    assert not (log/'passive-control.json').exists()
    report=json.loads((log/'status.json').read_text())
    assert report['performance_acceptance']=='not_run_user_requested'
    assert os.environ['OSCAR_PASSIVE_TIMING_CONTROL']=='/tmp/inherited-do-not-arm'


@pytest.mark.parametrize('failure',('none','old_contract','artifact','source','release','child'))
def test_new_parent_gate_requires_v3_current_source_artifact_and_cleanup(tmp_path,monkeypatch,failure):
    from tools import observe_serve as observe
    from oscar_ascend.ops import loader
    config={'experimental_bf16_attention':True,'experimental_aligned_math':True}
    path=tmp_path/'config.json';path.write_text(json.dumps(config));status={'phases':[]}
    manifest={'signature':'signed','sha256':{'extension':'sha'}}
    monkeypatch.setattr(loader,'validate_build_artifacts',lambda *a:manifest)
    monkeypatch.setattr(observe,'read_npu_resources',lambda *a,**k:{})
    releases=[]
    def release(*a,**k):releases.append(True);return {'status':'failed' if failure=='release' else 'passed'}
    monkeypatch.setattr(observe,'wait_for_release',release)
    def phase(name,command,**kwargs):
        assert name=='aligned-attention' and command[2]=='tools.probe_aligned_attention' and 'npu' in command
        if failure=='child':raise RuntimeError('child rejected device')
        report=complete_receipt();report.update(artifact_signature=manifest['signature'],artifact_sha256=manifest['sha256'],effective_config_sha256=probe.digest(path))
        if failure=='old_contract':report['contract']='oscar_bf16_attention_v2'
        if failure=='artifact':report['artifact_signature']='old'
        if failure=='source':report['source_binding']={}
        (tmp_path/'aligned-attention-report.json').write_text(json.dumps(report))
    monkeypatch.setattr(observe,'_phase',phase)
    if failure=='none':
        observe._aligned_attention_gate(path,config,{},tmp_path,status)
        assert status['aligned_attention_gate']['status']=='passed'
    else:
        with pytest.raises(RuntimeError):observe._aligned_attention_gate(path,config,{},tmp_path,status)
        assert 'aligned_attention_gate' not in status
    assert releases==[True]


def test_aligned_lifecycle_never_hides_legacy_reader_or_writer_calls():
    config={'experimental_aligned_math':True}
    calls={'store_aligned_int2_out':2,probe.OPERATORS['q4']:1,probe.QUERY_COPY:1}
    shared.validate_aligned_calls(calls,config)
    for legacy in ('attention_cv_bf16_decode_out','rotate_clip_store_striped_out','rotate_out'):
        with pytest.raises(RuntimeError,match='legacy'):
            shared.validate_aligned_calls({**calls,legacy:1},config)
    with pytest.raises(RuntimeError,match='writer'):
        shared.validate_aligned_calls({probe.OPERATORS['q4']:1},config)


def test_lifecycle_source_drift_cannot_be_relabelled_after_execution(monkeypatch):
    config={'experimental_aligned_math':True}
    before=shared.aligned_source_snapshot(config)
    monkeypatch.setattr(probe,'source_binding',lambda:{'changed':'source'})
    with pytest.raises(RuntimeError,match='changed during'):
        shared.bind_aligned_evidence({},config,before)


def test_v3_current_only_validator_demands_new_full_call_witnesses():
    from test_probe_current_only import complete_report,target
    report=complete_report();config={**target(),'experimental_bf16_attention':True,'experimental_aligned_math':True}
    for row in report['cases']:
        row['performance']='not_run_user_requested';calls=row['real_operator_calls']
        calls['store_aligned_int2_out']=calls.pop('rotate_clip_store_striped_out')
        if 'ab_output' in row:
            row['ab_output']='separate_BF16_native_contracts_passed'
            calls[probe.OPERATORS['q4']]=calls.pop('attention_cv_striped_decode_out')
            calls[probe.OPERATORS['q1']]=calls.pop('attention_cv_striped_q1_out')
        else:calls['copy_validate_current_out']=calls['store_aligned_int2_out']=1
    acceptance={'performance':{'warmup':2,'repeats':5}}
    shared.validate_case_evidence(report,config,acceptance)
    report['cases'][0]['real_operator_calls']['attention_cv_bf16_decode_out']=1
    with pytest.raises(RuntimeError,match='legacy'):shared.validate_case_evidence(report,config,acceptance)


def test_v3_first_mtp_validator_does_not_accept_v2_calls():
    from test_probe_first_mtp_current import evidence
    from tools import probe_first_mtp_current as first
    report,policy=evidence();report.update(bf16_attention=True,aligned_math=True)
    for row in report['cases']:
        row.update(ab_output='separate_BF16_native_contracts_passed',ab_lse='separate_BF16_native_contracts_passed',
            performance='not_run_user_requested',native_current_calls={'baseline':0,'candidate':1},
            bf16_cv_calls={side:{probe.OPERATORS['c16']:1} for side in ('baseline','candidate')},
            real_operator_calls={side:{probe.WRITER:1,probe.OPERATORS['c16']:1} for side in ('baseline','candidate')})
        row['decode_after_store']['operator']=probe.OPERATORS['q1']
    first.validate_case_evidence(report,policy)
    report['cases'][0]['real_operator_calls']['candidate']['rotate_clip_store_striped_out']=1
    with pytest.raises(RuntimeError,match='legacy'):first.validate_case_evidence(report,policy)


def test_v3_mixed_split_validator_does_not_accept_v2_calls():
    from test_probe_mixed_decode_split import evidence
    from tools import probe_mixed_decode_split as mixed
    report,policy=evidence();report.update(bf16_attention=True,aligned_math=True)
    for row in report['cases']:
        cv={'old':{probe.OPERATORS['c16']:1},'new':{probe.OPERATORS['c16']:1,probe.OPERATORS['q4']:1}}
        row.update(precision='BF16_independent_runtime_oracle_passed',performance='not_run_user_requested',timings=None,
            bf16_cv_calls=cv,real_operator_calls={side:{probe.WRITER:1,**calls} for side,calls in cv.items()})
    mixed.validate_case_evidence(report,policy)
    report['cases'][0]['real_operator_calls']['new']['attention_cv_bf16_decode_out']=1
    with pytest.raises(RuntimeError,match='legacy'):mixed.validate_case_evidence(report,policy)
