"""Only validate evidence rejection. Mock reports are not NPU results."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from tools import probe_first_mtp_current as probe


def evidence():
    acceptance=json.loads(Path('configs/acceptance.json').read_text())
    policy=acceptance['performance'];rows=[]
    for case in probe.case_plan():
        qlens,slot=case
        rows.append({'case':probe.case_name(case),'status':'passed','qlens':list(qlens),
            'contexts':[511,769],'slot_dtype':slot,'rotation':'draft_identity',
            'ab_output':'frozen_tolerance_passed','ab_lse':'frozen_tolerance_passed',
            'independent_oracle':'passed','store_bitwise':'passed','current_only_calls':0,
            'cache_reset_before_each_forward':True,
            'native_current_calls':{'baseline':0,'candidate':1+policy['warmup']+policy['repeats']},
            'device_event_ms':{side:[1.] * policy['repeats'] for side in ('baseline','candidate')},
            'device_event_median_ms':{'baseline':1.,'candidate':1.},
            'decode_after_store':{'status':'passed','reader':'production_subsequent_q1',
                'output_bitwise':'passed','independent_oracle':'passed',
                'subsequent_store_bitwise':'passed','sampled_queries':2}})
    return {'cases':rows},acceptance


def test_complete_mock_evidence_contract():
    report,policy=evidence();probe.validate_case_evidence(report,policy)


@pytest.mark.parametrize('mutation',[
    lambda r:r.update(cases=[]),
    lambda r:r['cases'].pop(),
    lambda r:r['cases'].__setitem__(1,deepcopy(r['cases'][0])),
    lambda r:r['cases'][0].update(contexts=[0,0]),
    lambda r:r['cases'][0].update(current_only_calls=1),
    lambda r:r['cases'][0].update(cache_reset_before_each_forward=False),
    lambda r:r['cases'][0]['native_current_calls'].update(candidate=0),
    lambda r:r['cases'][0]['native_current_calls'].update(baseline=1),
    lambda r:r['cases'][0]['device_event_ms']['candidate'].__setitem__(0,float('nan')),
    lambda r:r['cases'][0]['device_event_median_ms'].update(candidate=2.),
    lambda r:r['cases'][0]['decode_after_store'].update(independent_oracle='not_run'),
    lambda r:r['cases'][0].update(rotation='target_hadamard'),
])
def test_partial_or_wrong_route_cannot_pass(mutation):
    report,policy=evidence();mutation(report)
    with pytest.raises(RuntimeError):probe.validate_case_evidence(report,policy)


def test_no_explicit_first_mtp_flag_fails_before_device_import(tmp_path):
    target=tmp_path/'target.json';target.write_text('{}')
    with pytest.raises(RuntimeError,match='explicit'):
        probe.probe(target,Path('configs/acceptance.json'))


@pytest.mark.parametrize('failure',[None,'phase','release','incomplete'])
def test_observer_requires_complete_evidence_and_releases_after_failure(monkeypatch,tmp_path,failure):
    from tools import observe_serve
    from oscar_ascend.ops import loader
    report,_=evidence()
    report.update({key:'passed' for key in ('status','precision','store_bitwise',
                                          'decode_after_store','device_completion')})
    report.update(artifact_signature='mock-signature',artifact_sha256='mock-sha')
    if failure=='incomplete':report['cases']=[]
    released=[]
    monkeypatch.setattr(observe_serve,'read_npu_resources',lambda *a,**k:{'mock':True})
    def release(*a,**k):
        released.append(True)
        return {'status':'failed' if failure=='release' else 'passed'}
    monkeypatch.setattr(observe_serve,'wait_for_release',release)
    monkeypatch.setattr(loader,'validate_build_artifacts',lambda path:{'signature':'mock-signature','sha256':'mock-sha'})
    def phase(*a,**k):
        if failure=='phase':raise RuntimeError('mock device process failed')
        (tmp_path/'first-mtp-current-report.json').write_text(json.dumps(report))
    monkeypatch.setattr(observe_serve,'_phase',phase)
    status={};config={'experimental_first_mtp_current_fia':True}
    if failure:
        with pytest.raises(RuntimeError):
            observe_serve._first_mtp_current_gate(tmp_path/'config.json',config,{},tmp_path,status)
        assert 'first_mtp_current_gate' not in status
    else:
        observe_serve._first_mtp_current_gate(tmp_path/'config.json',config,{},tmp_path,status)
        assert status['first_mtp_current_gate']['full_service_performance']=='not_established'
    assert released==[True]
