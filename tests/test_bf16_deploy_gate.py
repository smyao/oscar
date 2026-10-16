"""Archive #94/#125/#148/#154: default deploy cannot omit the new numerical gate.

Mocks test orchestration only; no hardware or acceptance evidence is invented.
"""
import hashlib
import json
from pathlib import Path

import pytest

from tools import bf16_attention_gate, deploy, observe_serve
from tools.serving_variants import variant_config

ROOT = Path(__file__).resolve().parents[1]


def test_deploy_plan_names_real_probe_wrapper_and_effective_config(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr('sys.argv', ['deploy', '--plan', '--log-dir', str(tmp_path/'logs')])
    assert deploy.main() == 0
    plan = json.loads(capsys.readouterr().out)
    stages = dict(plan['stages'])
    command = stages['bf16-attention-gate']
    assert command[1:3] == ['-m', 'tools.bf16_attention_gate']
    effective = str(tmp_path/'logs'/'effective-target.json')
    for name in ('prepare-rotations','bf16-attention-gate','native-synthetic','service-probe'):
        child = stages[name]
        assert child[child.index('--config')+1] == effective
    assert plan['serve'][-1] == effective
    assert not (tmp_path/'logs').exists()


@pytest.mark.parametrize('code', [0, 23])
def test_wrapper_uses_same_observe_gate_and_preserves_exit_code(tmp_path, monkeypatch, code):
    config = variant_config(json.loads((ROOT/'configs/target.json').read_text()), 'candidate')
    path = tmp_path/'effective-target.json'; path.write_text(json.dumps(config))
    logs = tmp_path/'gate'
    seen = []
    def gate(given_path, given_config, env, log_dir, status):
        seen.append(given_path)
        assert given_path == path and given_config == config and log_dir == logs
        assert env['OSCAR_TARGET_CONFIG'] == str(path)
        assert env['ASCEND_RT_VISIBLE_DEVICES'] == '4,5,6,7'
        if code:
            error = RuntimeError('real subprocess failure')
            error.returncode = code; error.phase = 'bf16-history-npu'
            raise error
        status['bf16_history_gate'] = {'status':'passed'}
        status['bf16_history_resource_release'] = {'status':'passed'}
    monkeypatch.setattr(observe_serve, '_bf16_history_gate', gate)
    assert bf16_attention_gate.run(path, logs) == code
    assert seen == [path]
    status = json.loads((logs/'status.json').read_text())
    assert status['status'] == ('passed' if code == 0 else 'failed')
    assert status['effective_config_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    if code:
        assert status['returncode'] == code and status['failed_phase'] == 'bf16-history-npu'


def test_wrapper_rejects_raw_unbound_configuration_before_gate(tmp_path, monkeypatch):
    path = tmp_path/'raw.json'; path.write_text((ROOT/'configs/target.json').read_text())
    monkeypatch.setattr(observe_serve,'_bf16_history_gate',lambda *args:pytest.fail('raw config reached NPU gate'))
    assert bf16_attention_gate.run(path,tmp_path/'gate') == 1


@pytest.mark.parametrize('receipt', [None, {}, {'status':'passed','bf16_history_gate':[]}])
def test_zero_exit_and_stale_or_malformed_receipt_cannot_start_a_service(tmp_path,monkeypatch,receipt):
    from tools.phase import PhaseResult
    from tools import service_probe
    logs=tmp_path/'logs';child=logs/'bf16-attention';child.mkdir(parents=True)
    (child/'status.json').write_text(json.dumps({'status':'passed','old_run':True}))
    path=tmp_path/'target.json';path.write_text((ROOT/'configs/target.json').read_text())
    monkeypatch.setattr('sys.argv',['deploy','--config',str(path),'--log-dir',str(logs)])
    seen=[]
    def phase(name,command,**kwargs):
        seen.append(name)
        if name=='bf16-attention-gate':
            assert not (child/'status.json').exists()
            if receipt is not None:(child/'status.json').write_text(json.dumps(receipt))
        return PhaseResult(name,command,0,0.0,False,False,str(logs/name),True)
    monkeypatch.setattr(deploy,'run_phase',phase)
    monkeypatch.setattr(service_probe,'main',lambda:pytest.fail('service ran after an invalid gate receipt'))
    assert deploy.main()==1
    assert seen[-1]=='bf16-attention-gate'
    status=json.loads((logs/'status.json').read_text())
    assert status['status']=='failed' and status['failed_phase']=='bf16-attention-gate'
