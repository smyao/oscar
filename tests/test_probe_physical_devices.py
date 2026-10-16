"""Archive #123/#125: probes use the explicit current-task physical cards."""
import json
import os

import pytest

from tools import probe_cv_hotshape as hot, probe_history_reuse as reuse


@pytest.mark.parametrize('select', [hot._select_device, reuse._active_device])
@pytest.mark.parametrize('devices', [[4,5,6,7], [0,1,2,3], [8,9,10,11]])
def test_explicit_legal_selection_is_used_without_inherited_defaults(monkeypatch,select,devices):
    monkeypatch.delenv('ASCEND_RT_VISIBLE_DEVICES',raising=False)
    target={'devices':devices,'soc_version':'ascend910b4'}
    select(target)
    assert os.environ['ASCEND_RT_VISIBLE_DEVICES'] == ','.join(map(str,devices))
    select(target)  # Same selection is idempotent.


@pytest.mark.parametrize('select', [hot._select_device, reuse._active_device])
@pytest.mark.parametrize('devices', [None, [], [4,5,6], [4,5,6,7,8], [4,5,6,6],
    [-1,5,6,7], [True,5,6,7], ['4',5,6,7], [4.0,5,6,7], [{},5,6,7]])
def test_invalid_explicit_selection_fails_without_changing_environment(monkeypatch,select,devices):
    monkeypatch.delenv('ASCEND_RT_VISIBLE_DEVICES',raising=False)
    with pytest.raises(RuntimeError,match='four unique nonnegative integer'):
        select({'devices':devices,'soc_version':'ascend910b4'})
    assert 'ASCEND_RT_VISIBLE_DEVICES' not in os.environ


@pytest.mark.parametrize('select', [hot._select_device, reuse._active_device])
def test_conflicting_environment_and_wrong_soc_are_rejected(monkeypatch,select):
    monkeypatch.setenv('ASCEND_RT_VISIBLE_DEVICES','0,1,2,3')
    with pytest.raises(RuntimeError,match='differs from target'):
        select({'devices':[4,5,6,7],'soc_version':'ascend910b4'})
    assert os.environ['ASCEND_RT_VISIBLE_DEVICES']=='0,1,2,3'
    monkeypatch.delenv('ASCEND_RT_VISIBLE_DEVICES')
    with pytest.raises(RuntimeError,match='SOC'):
        select({'devices':[4,5,6,7],'soc_version':'ascend910b'})
    assert 'ASCEND_RT_VISIBLE_DEVICES' not in os.environ


def test_baseline_availability_accepts_rear_cards_and_remains_read_only(tmp_path,monkeypatch):
    target=tmp_path/'target.json';target.write_text(json.dumps({'devices':[4,5,6,7],'soc_version':'ascend910b4'}))
    monkeypatch.delenv('ASCEND_RT_VISIBLE_DEVICES',raising=False)
    seen=[]
    def artifact(variant,path,config):
        seen.append((variant,config['devices']))
        return {'signature':'signed','sha256':{}},{},'source'
    monkeypatch.setattr(hot,'_verify_artifact',artifact)
    assert hot.baseline_availability(tmp_path/'manifest.json',target)['status']=='available'
    assert seen==[('baseline',[4,5,6,7])]
    assert 'ASCEND_RT_VISIBLE_DEVICES' not in os.environ
    monkeypatch.setenv('ASCEND_RT_VISIBLE_DEVICES','0,1,2,3')
    assert hot.baseline_availability(tmp_path/'manifest.json',target)['status']=='not_available'
    assert len(seen)==1
