"""Archive #129/#140/#145/#148/#154: correct only a proven postprocess error.

The original exit1/failure/report stay immutable. This offline step can finish
trace collection only after full numerical/device/profile completion preceded
the exact erroneous q1 padding assertion. No kernel is rerun, no partial tick
or unknown error is accepted, and the correction retains the original rc1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from csrc.experiments.decode_mdl_load import run


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_complete(path, control):
    c = json.loads(control.read_text())
    # SDK, signal, timeout, nonzero numerical code, or failed cleanup cannot
    # use this recovery. Only the postprocessing ValueError has exit1 here.
    if (c['status'] != 'failed' or c['timed_out'] or c['child_returncode'] != 1 or
            c['supervisor_returncode'] != 1 or not c['cleanup_complete'] or
            c['interrupt_signal'] is not None or c['timeout_seconds'] != 600):
        raise ValueError('not the completed q1 postprocess failure')
    failure = json.loads((path / 'mdl_failure.json').read_text())
    if failure != {'status': 'failed', 'reason': "ValueError('q1 padding task became live')"}:
        raise ValueError('unexpected original error; recovery prohibited')
    r = json.loads((path / 'report.json').read_text())
    done = json.loads((path / 'completion.json').read_text())
    if (r['case'] != run.Q1_CASE or r['mode'] != 'q1' or r['variant'] != 'baseline' or
            r['status'] != 'passed' or r['failures'] or done['failures'] or
            done['status'] != 'device_complete_outputs_saved' or
            not r['all_sources_and_tasks'] or r['task_filtering'] or
            r['oracle']['bad_output'] or r['oracle']['bad_lse'] or
            type(r['task_ticks']) is not int or r['task_ticks'] <= 0 or
            r['cpu_reference'] != 'available_checked'):
        raise ValueError('full CPU/numeric/task completion not established')
    for key, meta in r['outputs'].items():
        file = path / meta['path']
        if (file.stat().st_size != meta['bytes'] or digest(file) != meta['sha256'] or
                done['outputs'][key] != meta):
            raise ValueError('saved output changed: ' + key)
    for kind, count in (('task', 1), ('aic', 1), ('aiv', 2)):
        rows = r['profiles'][kind]
        if len(rows) != count or any(x['duration'] <= 0 or
                x['end_cycle'] - x['start_cycle'] != x['duration'] for x in rows):
            raise ValueError('incomplete SDK actor profile')
    return r


def originals(path, control):
    return {'control': digest(control),
            **{name: digest(path / name) for name in
               ('report.json', 'completion.json', 'mdl_failure.json')},
            'plan': digest(path.with_name(path.name + '.plan.json'))}


def validate_correction(path, control):
    original = original_complete(path, control)
    correction = json.loads((path / 'postprocess-correction.json').read_text())
    qualified = json.loads((path / 'postprocess-qualified.json').read_text())
    if (correction['status'] != 'postprocess_corrected_original_exit1_preserved' or
            correction['original_sha256'] != originals(path, control) or
            correction['qualified_sha256'] != digest(path / 'postprocess-qualified.json') or
            correction['recovery_source_sha256'] != digest(__file__) or
            correction['runner_source_sha256'] != digest(run.__file__)):
        raise ValueError('postprocess correction binding changed')
    if qualified != {**original, 'mdl_complete_trace': correction['mdl_complete_trace']}:
        raise ValueError('correction changed mathematical/profile evidence')
    run.geometry(qualified, (path / 'task_witness.bin').read_bytes())
    return qualified


def recover(path, control, tasks):
    if (path / 'postprocess-correction.json').exists():
        raise ValueError('refuse repeated postprocess correction')
    original = original_complete(path, control)
    before = originals(path, control)
    witness = tasks.read_bytes()
    run.geometry(original, witness)
    (path / 'task_witness.bin').write_bytes(witness)
    trace = run.collector.capture(path, original, witness)
    details = {'status': 'passed', 'plan_sha256': before['plan'],
               'trace_report': 'full_instructions.json', 'total_raw_bytes': trace['total_raw_bytes']}
    run.save(path / 'postprocess-qualified.json', {**original, 'mdl_complete_trace': details})
    if originals(path, control) != before:
        raise ValueError('original failed-run evidence was changed')
    run.save(path / 'postprocess-correction.json', {
        'status': 'postprocess_corrected_original_exit1_preserved',
        'original_sha256': before, 'original_child_returncode': 1,
        'reason': 'prepare padding is count=-1/error=0, not nonleader count=0',
        'qualified_sha256': digest(path / 'postprocess-qualified.json'),
        'recovery_source_sha256': digest(__file__), 'runner_source_sha256': digest(run.__file__),
        'mdl_complete_trace': details, 'kernel_rerun': False,
        'NPU': 'not_run', 'whole_service': 'not_established'})
    validate_correction(path, control)
    run.checked(path, control, 'baseline')
    print('Q1_MDL_POSTPROCESS_CORRECTED', json.dumps({'task_ticks': original['task_ticks'],
          'original_rc': 1, 'kernel_rerun': False, 'full_trace_files': 6}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('path', 'control', 'tasks'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    recover(args.path.resolve(), args.control.resolve(), args.tasks.resolve())
