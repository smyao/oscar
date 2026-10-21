import json
import pytest
from tools.bf16_probe_progress import recording, checkpoint
from tools.summarize_bf16 import summarize


def test_progress_persists_before_exception_and_resets(tmp_path, capsys):
    path=tmp_path/'result.json'
    report={'status':'running','cases':[{'status':'passed'}]}
    def persist():path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError):
        with recording(report,persist):
            checkpoint('cpu_oracle_rows',completed_rows=256,total_rows=8192)
            raise RuntimeError('simulated interruption')
    saved=json.loads(path.read_text())
    assert saved['status']=='running'
    assert saved['progress']['completed_rows']==256
    assert summarize(saved)['progress']==saved['progress']
    assert 'BF16_HISTORY_PROGRESS' in capsys.readouterr().out
    checkpoint('outside')
    assert json.loads(path.read_text())==saved


def test_complete_checkpoint_contains_new_receipt():
    saved=[];report={'cases':[]}
    with recording(report,lambda:saved.append(json.loads(json.dumps(report)))):
        checkpoint('full_numerical_begin',case='mdl_q4')
        report['cases'].append({'full_attention':{'numerical':{'passed':True}}})
        checkpoint('full_numerical_complete',case='mdl_q4')
    assert saved[0]['cases']==[]
    assert saved[1]['cases'][0]['full_attention']['numerical']['passed']
