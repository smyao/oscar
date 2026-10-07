"""Evidence validation only; all fabricated rows below are explicit mocks."""
from copy import deepcopy
import json
from pathlib import Path
import pytest

from tools import probe_decode_bundle as probe


def evidence():
    acceptance=json.loads(Path('configs/acceptance.json').read_text());rows=[]
    for case in probe.case_plan():
        invalid=case.shape.corrupt_metadata or case.shape.corrupt_qr
        rows.append({'case':case.shape.name,'old_operator':case.old,'new_operator':case.new,
            'rotation':'identity' if 'q1_' in case.shape.name else 'hadamard',
            'same_striped_inputs':True,'status':'passed',
            'precision':'error_contract_passed' if invalid else 'bitwise_and_frozen_oracle_passed',
            'invalid_evidence':{'error_task_parity':'passed','nan_classification':'passed'} if invalid else None,
            'timings':{'old_ms':1.,'new_ms':.5,'ratio':.5,'samples_ms':{'old':[1.]*5,'new':[.5]*5},'gate':'passed'} if case.timed else None,
            'graph':{'capture':'passed','replay':'passed','same_address_changed_key_value_alone':True,
                'same_address_changed_query_key_value_task':True} if case.graph else None})
    return {'cases':rows},acceptance


def test_mock_complete_evidence_contract():
    report,acceptance=evidence();probe.validate_case_evidence(report,acceptance)


@pytest.mark.parametrize('mutate',[
    lambda r:r.update(cases=[]),
    lambda r:r['cases'].pop(),
    lambda r:r['cases'].__setitem__(1,deepcopy(r['cases'][0])),
    lambda r:r['cases'][0].update(new_operator=r['cases'][0]['old_operator']),
    lambda r:r['cases'][0].update(same_striped_inputs=False),
    lambda r:r['cases'][0]['graph'].update(same_address_changed_key_value_alone=False),
    lambda r:r['cases'][0]['timings']['samples_ms']['new'].pop(),
    lambda r:r['cases'][0]['timings'].update(new_ms=.3),
    lambda r:r['cases'][0]['timings'].update(ratio=.3),
    lambda r:r['cases'][0]['timings']['samples_ms']['new'].__setitem__(0,float('nan')),
    lambda r:r['cases'][-2].update(invalid_evidence={}),
])
def test_incomplete_or_wrong_route_evidence_fails(mutate):
    report,acceptance=evidence();mutate(report)
    with pytest.raises(RuntimeError):probe.validate_case_evidence(report,acceptance)


def test_even_small_true_regression_does_not_relax_frozen_gate():
    report,acceptance=evidence()
    timing=report['cases'][0]['timings']
    timing.update(new_ms=1.001,ratio=1.001)
    timing['samples_ms']['new']=[1.001]*5
    with pytest.raises(RuntimeError,match='latency'):
        probe.validate_case_evidence(report,acceptance)


@pytest.mark.parametrize('failure',[None,'phase','release','empty'])
def test_observer_blocks_bad_evidence_and_always_releases(monkeypatch,tmp_path,failure):
    from tools import observe_serve
    from oscar_ascend.ops import loader
    report,_=evidence()
    report.update({k:'passed' for k in ('status','precision','performance','graph_capture','graph_replay','device_completion')})
    report.update(artifact_signature='mock-signature',artifact_sha256='mock-sha')
    if failure=='empty':report['cases']=[]
    calls=[]
    monkeypatch.setattr(loader,'validate_build_artifacts',lambda p:{'signature':'mock-signature','sha256':'mock-sha'})
    monkeypatch.setattr(observe_serve,'read_npu_resources',lambda *a,**k:{})
    def release(*a,**k):
        calls.append(True);return {'status':'failed' if failure=='release' else 'passed'}
    monkeypatch.setattr(observe_serve,'wait_for_release',release)
    def phase(*a,**k):
        if failure=='phase':raise RuntimeError('mock graph failure')
        (tmp_path/'decode-bundle-report.json').write_text(json.dumps(report))
    monkeypatch.setattr(observe_serve,'_phase',phase)
    status={};target={'experimental_decode_bundle':True}
    if failure:
        with pytest.raises(RuntimeError):observe_serve._decode_bundle_gate(tmp_path/'config',target,{},tmp_path,status)
        assert 'decode_bundle_gate' not in status
    else:
        observe_serve._decode_bundle_gate(tmp_path/'config',target,{},tmp_path,status)
        assert status['decode_bundle_gate']['full_service_performance']=='not_established'
    assert calls==[True]
