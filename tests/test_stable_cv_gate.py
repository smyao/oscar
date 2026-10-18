"""Archive #94/#125/#151: new gate failures must block formal serving.

Mocked evidence below tests host orchestration only; each probe separately
rejects incomplete shape/precision/graph/performance evidence.
"""
import importlib
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tools import observe_serve as observe


@pytest.mark.parametrize('family', ['hybrid_l1_decode', 'c16_member_pipeline', 'c16_l1_reuse', 'bf16_attention'])
@pytest.mark.parametrize('failure', [None, 'child', 'signature', 'source', 'release'])
def test_new_gate_requires_matching_artifact_and_cleans_up(monkeypatch, tmp_path, family, failure):
    from oscar_ascend.ops import loader
    probe = importlib.import_module('tools.probe_'+family)
    monkeypatch.setattr(probe, 'validate_evidence', lambda _: [])
    monkeypatch.setattr(probe, 'source_binding', lambda: {'kernel': 'source'})
    monkeypatch.setattr(loader, 'validate_build_artifacts', lambda _: {'signature': 'build', 'sha256': 'binary'})
    monkeypatch.setattr(observe, 'read_npu_resources', lambda *a, **k: {})
    config_path=tmp_path/'config.json';config_path.write_text(json.dumps({'experimental_bf16_attention':family=='bf16_attention'}))
    released = []

    def release(*args, **kwargs):
        released.append(True)
        return {'status': 'failed' if failure == 'release' else 'passed'}

    monkeypatch.setattr(observe, 'wait_for_release', release)

    def phase(name, command, **kwargs):
        if failure == 'child':
            raise RuntimeError('original child rc=7')
        report = {'artifact_signature': 'other' if failure == 'signature' else 'build',
                  'artifact_sha256': 'binary', 'source_binding': {'kernel': 'old' if failure == 'source' else 'source'}}
        if family=='bf16_attention':
            assert command[command.index('tools.probe_bf16_attention')+1]=='npu'
            assert '--acceptance' not in command and command[command.index('--config')+1]==str(config_path)
            report.update(contract='oscar_bf16_attention_v2',performance='not_run_user_requested',
                          effective_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest())
        path = tmp_path / (family.replace('_', '-')+'-report.json')
        assert command[command.index('--output')+1] == str(path)
        path.write_text(json.dumps(report))
        assert path != tmp_path / (name+'.json')

    monkeypatch.setattr(observe, '_phase', phase)
    state = {}
    if failure:
        with pytest.raises(RuntimeError):
            observe._stable_cv_gate(tmp_path/'config.json', {}, {}, tmp_path, state, family=family)
        assert family+'_gate' not in state
    else:
        observe._stable_cv_gate(tmp_path/'config.json', {}, {}, tmp_path, state, family=family)
        assert state[family+'_gate']['status'] == 'passed'
        assert state[family+'_gate']['full_service_performance'] == 'not_established'
        if family=='bf16_attention':assert state[family+'_gate']['performance']=='not_run_user_requested'
    assert released == [True]


LIFECYCLE=(
    ('current_only','experimental_current_only'),
    ('first_mtp_current','experimental_first_mtp_current_fia'),
    ('mixed_decode_split','experimental_mixed_decode_split'),
    ('prefix_cache','enable_prefix_caching'),
)


def lifecycle_parent_harness(monkeypatch, tmp_path, family, bf16, failure=None):
    """Host-only child receipts isolate the parent's mode/artifact/failure checks.

    Probe-specific full numerical schemas have their own suites. Here their
    validator is represented by one explicit rejection marker, not NPU data.
    """
    from oscar_ascend.ops import loader
    probe=importlib.import_module('tools.probe_'+family)
    manifest={'signature':'current-build','sha256':{'binary':'current-digest'}}
    monkeypatch.setattr(loader,'validate_build_artifacts',lambda _:manifest)
    monkeypatch.setattr(observe,'read_npu_resources',lambda *a,**k:{})
    released=[]
    def release(*a,**k):
        released.append(True)
        if failure=='child_and_release':raise RuntimeError('secondary resource error')
        return {'status':'failed' if failure=='release' else 'passed'}
    monkeypatch.setattr(observe,'wait_for_release',release)
    def validate(report,*_):
        if not report.get('complete_child_schema'):raise RuntimeError('child schema incomplete')
    if family=='prefix_cache':
        monkeypatch.setattr(probe,'validate_evidence',lambda report:
            ['precision'] if report.get('precision')!='passed' else
            ([] if report.get('complete_child_schema') else ['child_schema']))
        monkeypatch.setattr(probe,'source_binding',lambda:{'source':'current'})
        monkeypatch.setattr(probe,'selected_reader',lambda config:
            'attention_cv_bf16_decode_out' if config.get('experimental_bf16_attention') else 'attention_cv_mdl_decode_out')
    else:monkeypatch.setattr(probe,'validate_case_evidence',validate)
    def phase(name,command,**kwargs):
        assert name==family.replace('_','-')+'-npu'
        effective=Path(command[command.index('--config')+1])
        assert json.loads(effective.read_text())==kwargs['config']
        if failure in ('child','child_and_release'):
            error=RuntimeError('original child failure');error.returncode=37;error.phase=name;raise error
        report={k:'passed' for k in ('status','precision','store_bitwise','decode_after_store','device_completion')}
        report.update(artifact_signature=manifest['signature'],artifact_sha256=manifest['sha256'],complete_child_schema=True)
        if family=='mixed_decode_split':report['performance']='passed'
        if bf16:report.update(bf16_attention=True,math_contract='oscar_bf16_attention_v2',performance='not_run_user_requested')
        if family=='prefix_cache':report.update(source_binding={'source':'current'},reader=probe.selected_reader(kwargs['config']))
        if failure=='artifact':report['artifact_signature']='stale-build'
        if failure=='precision':report['precision']='failed'
        if failure=='performance':report['performance']='failed'
        if failure=='schema':report['complete_child_schema']=False
        if failure=='wrong_mode':
            if bf16:
                for key in ('bf16_attention','math_contract','performance'):report.pop(key,None)
                if family=='mixed_decode_split':report['performance']='passed'
            else:report.update(bf16_attention=True,math_contract='oscar_bf16_attention_v2',performance='not_run_user_requested')
        Path(command[command.index('--output')+1]).write_text(json.dumps(report))
    monkeypatch.setattr(observe,'_phase',phase)
    config={flag:True for _,flag in LIFECYCLE};config['experimental_bf16_attention']=bf16
    path=tmp_path/'effective.json';path.write_text(json.dumps(config))
    return getattr(observe,'_'+family+'_gate'),config,path,released


@pytest.mark.parametrize('family,flag',LIFECYCLE)
@pytest.mark.parametrize('bf16',[False,True])
def test_lifecycle_parent_accepts_only_the_effective_math_scope(monkeypatch,tmp_path,family,flag,bf16):
    gate,config,path,released=lifecycle_parent_harness(monkeypatch,tmp_path,family,bf16)
    state={};gate(path,config,{},tmp_path,state)
    assert state[family+'_gate']['status']=='passed' and released==[True]
    if bf16:assert state[family+'_gate']['performance']=='not_run_user_requested'
    gate,config,path,released=lifecycle_parent_harness(monkeypatch,tmp_path,family,bf16,'wrong_mode')
    with pytest.raises(RuntimeError,match='evidence'):
        gate(path,config,{},tmp_path,{})
    assert released==[True]


def test_legacy_mixed_parent_still_requires_passed_performance(monkeypatch,tmp_path):
    gate,config,path,released=lifecycle_parent_harness(monkeypatch,tmp_path,'mixed_decode_split',False,'performance')
    with pytest.raises(RuntimeError,match='performance'):gate(path,config,{},tmp_path,{})
    assert released==[True]


def test_legacy_scope_rejects_correctness_only_marker_even_without_bf16_flag():
    assert observe._lifecycle_precision_mismatches({'performance':'not_run_user_requested'},
        {'experimental_bf16_attention':False})==['legacy_fp32_scope']


@pytest.mark.parametrize('family,flag',LIFECYCLE)
@pytest.mark.parametrize('failure',['child','child_and_release','artifact','precision','schema','release','wrong_mode'])
def test_each_bf16_lifecycle_parent_failure_blocks_service(monkeypatch,tmp_path,family,flag,failure):
    real_gate,_,_,released=lifecycle_parent_harness(monkeypatch,tmp_path,family,True,failure)
    # All unrelated gates are stubs; the selected parent is real and receives
    # a failed child/control/schema/cleanup record through normal run().
    for name in dir(observe):
        if name.startswith('_') and name.endswith('_gate') and name!='_'+family+'_gate':
            monkeypatch.setattr(observe,name,lambda *a,**k:None)
    monkeypatch.setattr(observe,'_preflight',lambda *a,**k:None)
    monkeypatch.setattr(observe,'managed_server',lambda *a,**k:pytest.fail('failed lifecycle gate started service'))
    monkeypatch.setattr(observe,'_terminal',lambda *a,**k:None)
    config=json.loads((observe.ROOT/'configs/target.json').read_text())
    config.update({key:True for _,key in LIFECYCLE});config['experimental_bf16_attention']=True
    model=tmp_path/'model';model.mkdir()
    (model/'config.json').write_text(json.dumps({'architectures':['Qwen3_5ForConditionalGeneration']}))
    config['model']=str(model)
    path=tmp_path/'target.json';path.write_text(json.dumps(config));logs=tmp_path/'run'
    rc=observe.run(path,logs,'candidate')
    assert rc==(37 if failure in ('child','child_and_release') else 1)
    state=json.loads((logs/'status.json').read_text())
    assert state['status']=='failed' and state.get('service_started') is not True
    assert family+'_gate' not in state and released==[True]
    if failure in ('child','child_and_release'):
        assert state['failed_phase']==family.replace('_','-')+'-npu'
    if failure=='child_and_release':assert state[family+'_resource_release']['status']=='failed'
