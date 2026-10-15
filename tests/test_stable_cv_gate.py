"""Archive #94/#125/#151: new gate failures must block formal serving.

Mocked evidence below tests host orchestration only; each probe separately
rejects incomplete shape/precision/graph/performance evidence.
"""
import importlib
import json
from types import SimpleNamespace

import pytest
from tools import observe_serve as observe


@pytest.mark.parametrize('family', ['hybrid_l1_decode', 'c16_member_pipeline', 'c16_l1_reuse'])
@pytest.mark.parametrize('failure', [None, 'child', 'signature', 'source', 'release'])
def test_new_gate_requires_matching_artifact_and_cleans_up(monkeypatch, tmp_path, family, failure):
    from oscar_ascend.ops import loader
    probe = importlib.import_module('tools.probe_'+family)
    monkeypatch.setattr(probe, 'validate_evidence', lambda _: [])
    monkeypatch.setattr(probe, 'source_binding', lambda: {'kernel': 'source'})
    monkeypatch.setattr(loader, 'validate_build_artifacts', lambda _: {'signature': 'build', 'sha256': 'binary'})
    monkeypatch.setattr(observe, 'read_npu_resources', lambda *a, **k: {})
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
    assert released == [True]
