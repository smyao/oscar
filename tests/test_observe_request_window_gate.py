"""Archive #94/#95/#125/#151/#153/#155: fail-closed phase and evidence flow.

Host fixtures only test rejection and cleanup; no NPU success is simulated.
"""
import json
import hashlib

import pytest

from tools import observe_serve as observe
from oscar_ascend.ops import loader


def hooks(monkeypatch):
    calls=[]
    monkeypatch.setattr(observe,'read_npu_resources',lambda *a,**k:{'host_fixture':True})
    def release(*args,**kwargs):
        calls.append('release')
        return {'status':'passed','scope':'host_control_flow_fixture'}
    monkeypatch.setattr(observe,'wait_for_release',release)
    monkeypatch.setattr(loader,'validate_build_artifacts',lambda *a,**k:{'signature':'fixture','sha256':{}})
    return calls


def test_child_failure_releases_resources_and_propagates_original_error(monkeypatch,tmp_path):
    calls=hooks(monkeypatch);status={'phases':[]}
    error=RuntimeError('child numerical rejection')
    def phase(*args,**kwargs):raise error
    monkeypatch.setattr(observe,'_phase',phase)
    with pytest.raises(RuntimeError) as raised:
        observe._request_window_gate(tmp_path/'config.json',{'experimental_request_windows':True},
                                     {},tmp_path,status)
    assert raised.value is error and calls==['release']
    assert 'request_window_gate' not in status


def test_missing_numerical_evidence_cannot_be_promoted_by_child_success(monkeypatch,tmp_path):
    calls=hooks(monkeypatch);status={'phases':[]}
    output=tmp_path/'request-window-report.json'
    output.write_text('stale evidence must be removed')
    def phase(name,argv,**kwargs):
        assert not output.exists()
        assert name=='request-window-npu'
        assert argv[1:3]==['-m','tools.probe_request_windows']
        assert argv[argv.index('--output')+1]==str(output)
        output.write_text(json.dumps({'status':'passed','artifact_signature':'fixture','artifact_sha256':{}}))
    monkeypatch.setattr(observe,'_phase',phase)
    with pytest.raises(RuntimeError,match='complete matching evidence'):
        observe._request_window_gate(tmp_path/'config.json',{'experimental_request_windows':True},
                                     {},tmp_path,status)
    assert calls==['release'] and 'request_window_gate' not in status


def test_old_storage_does_not_run_or_inherit_new_gate(monkeypatch,tmp_path):
    monkeypatch.setattr(observe,'read_npu_resources',lambda *a,**k:pytest.fail('old format must not run new gate'))
    status={'phases':[]}
    observe._request_window_gate(tmp_path/'config.json',{}, {},tmp_path,status)
    assert status=={'phases':[]}


@pytest.mark.parametrize('mismatch',['acceptance','physical_selection','missing_core','configured_core'])
def test_parent_rejects_wrong_run_identity_after_resource_cleanup(monkeypatch,tmp_path,mismatch):
    from tools import probe_request_windows
    calls=hooks(monkeypatch);status={'phases':[]}
    config={'experimental_request_windows':True,'devices':[4,5,6,7]}
    report={'status':'passed','artifact_signature':'fixture','artifact_sha256':{},
        'acceptance_sha256':hashlib.sha256((observe.ROOT/'configs/acceptance.json').read_bytes()).hexdigest(),
        'device':{'physical_selection':[4,5,6,7],'cube_cores':20}}
    if mismatch=='acceptance':report['acceptance_sha256']='0'*64
    elif mismatch=='physical_selection':report['device']['physical_selection']=[0,1,2,3]
    elif mismatch=='missing_core':report['device'].pop('cube_cores')
    else:config['cube_cores']=16
    # Isolate parent identity rejection; never synthesize a passing NPU gate.
    monkeypatch.setattr(probe_request_windows,'final_status',lambda _report:'passed')
    def phase(*args,**kwargs):(tmp_path/'request-window-report.json').write_text(json.dumps(report))
    monkeypatch.setattr(observe,'_phase',phase)
    with pytest.raises(RuntimeError,match='complete matching evidence'):
        observe._request_window_gate(tmp_path/'config.json',config,{},tmp_path,status)
    assert calls==['release'] and 'request_window_gate' not in status


def test_parent_requires_pipeline_reader_evidence_for_the_enabled_combination(monkeypatch,tmp_path):
    from tools import probe_request_windows
    from tests.test_request_window_reader_gate import complete_schema
    calls=hooks(monkeypatch);status={'phases':[]}
    config={'experimental_request_windows':True,'experimental_c4_member_pipeline':True,
            'devices':[4,5,6,7]}
    old_readers=complete_schema()
    report={'status':'passed','artifact_signature':'fixture','artifact_sha256':{},
        'acceptance_sha256':hashlib.sha256((observe.ROOT/'configs/acceptance.json').read_bytes()).hexdigest(),
        'device':{'physical_selection':[4,5,6,7],'cube_cores':20},
        'accuracy':{'reader':old_readers['accuracy']},'graph':{'reader':old_readers['graph']}}
    # Isolate a rejection by the real parent reader validator. These are
    # synthetic contract dictionaries, not a fabricated successful NPU run.
    monkeypatch.setattr(probe_request_windows,'final_status',lambda _report:'passed')
    def phase(*args,**kwargs):
        (tmp_path/'request-window-report.json').write_text(json.dumps(report))
    monkeypatch.setattr(observe,'_phase',phase)
    with pytest.raises(RuntimeError):
        observe._request_window_gate(tmp_path/'config.json',config,{},tmp_path,status)
    assert calls==['release'] and 'request_window_gate' not in status
