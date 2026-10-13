"""Archive #140/#154: postprocess repair cannot bless failed device evidence."""
import hashlib
import json

import pytest

from csrc.experiments.decode_mdl_load import recover_q1_postprocess as recovery


def evidence(tmp_path):
    arm = tmp_path / 'baseline'
    arm.mkdir()
    data = b'complete output'
    (arm / 'partial.bin').write_bytes(data)
    outputs = {'partial': {'path': 'partial.bin', 'bytes': len(data),
                          'sha256': hashlib.sha256(data).hexdigest()}}
    report = {'case': recovery.run.Q1_CASE, 'mode': 'q1', 'variant': 'baseline',
              'status': 'passed', 'failures': [], 'all_sources_and_tasks': True,
              'task_filtering': False, 'oracle': {'bad_output': 0, 'bad_lse': 0},
              'task_ticks': 100, 'cpu_reference': 'available_checked',
              'outputs': outputs, 'profiles': {
                  kind: [{'duration': 100, 'end_cycle': 120, 'start_cycle': 20}] * count
                  for kind, count in (('task', 1), ('aic', 1), ('aiv', 2))}}
    done = {'status': 'device_complete_outputs_saved', 'failures': [], 'outputs': outputs}
    failure = {'status': 'failed', 'reason': "ValueError('q1 padding task became live')"}
    control = {'status': 'failed', 'timed_out': False, 'child_returncode': 1,
               'supervisor_returncode': 1, 'cleanup_complete': True,
               'interrupt_signal': None, 'timeout_seconds': 600}
    for name, obj in (('report', report), ('completion', done), ('mdl_failure', failure)):
        (arm / (name + '.json')).write_text(json.dumps(obj))
    receipt = tmp_path / 'baseline-control.json'
    receipt.write_text(json.dumps(control))
    return arm, receipt


def test_only_postprocess_error_recovers_and_original_rc_is_preserved(tmp_path):
    arm, receipt = evidence(tmp_path)
    before = {p: p.read_bytes() for p in [receipt, *arm.iterdir()]}
    assert recovery.original_complete(arm, receipt)['task_ticks'] == 100
    assert all(p.read_bytes() == data for p, data in before.items())
    assert json.loads(receipt.read_text())['child_returncode'] == 1


@pytest.mark.parametrize('change', ['timeout', 'cleanup', 'signal', 'numeric',
                                  'other_error', 'partial_actor', 'changed_bytes'])
def test_failed_or_incomplete_device_evidence_cannot_be_reclassified(tmp_path, change):
    arm, receipt = evidence(tmp_path)
    path = receipt if change in ('timeout', 'cleanup', 'signal') else arm / 'report.json'
    obj = json.loads(path.read_text())
    if change == 'timeout': obj['timed_out'] = True
    elif change == 'cleanup': obj['cleanup_complete'] = False
    elif change == 'signal': obj['interrupt_signal'] = 15
    elif change == 'numeric': obj['oracle']['bad_output'] = 1
    elif change == 'partial_actor': obj['profiles']['aiv'].pop()
    elif change == 'other_error':
        path = arm / 'mdl_failure.json'
        obj = {'status': 'failed', 'reason': "ValueError('full numeric/task/actor gate failed')"}
    else: (arm / 'partial.bin').write_bytes(b'changed output')
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError):
        recovery.original_complete(arm, receipt)
