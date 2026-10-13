"""Archive #140/#145/#154: reject incomplete q1 performance witnesses."""
import hashlib
import json
import struct
from types import SimpleNamespace

import pytest

from csrc.experiments.decode_mdl_load import extract_trace, qualify_q1, run


def witness():
    rows = []
    for token in range(8):
        for source in range(3):
            context = (1344, 1376)[token] if token < 2 else 0
            bounds = ((64, context - 255), (0, 319), (context, context + 1))[source] if token < 2 else (0, 0)
            rows.append([token, 1 if token < 2 else -1, 0, *bounds, token if token < 2 else 0,
                         source, source, context, token if token < 2 else 0, 0, 0, 0, 0, 0, 0])
    return rows


def report(rows):
    data = b"".join(struct.pack("<16q", *row) for row in rows)
    attrs = [8, 6, 1, 256, 2, 11, 24, 128, 22, 8192, 17408, 82688, 323, 64, 256, 3, 1]
    return {"case": run.Q1_CASE, "mode": "q1", "cores": 1, "attrs": attrs,
            "all_sources_and_tasks": True, "task_filtering": False,
            "input_sha256": {"tasks": hashlib.sha256(data).hexdigest()}}, data


def test_complete_q1_source_tail_and_padding_witness():
    data, tasks = report(witness())
    run.geometry(data, tasks)


def test_q1_complete_trace_plan_uses_actual_one_query_owners_and_all_sources():
    data, tasks = report(witness())
    plan = extract_trace.task_plan(data, tasks)
    assert len(plan) == 34  # history10*2 + rotation2 + window4*2 + current2*2
    assert {p['task_id'] for p in plan} == set(range(6))
    assert {p['M'] for p in plan} == {6}
    tails = [p['live_KV'] for p in plan if p['source'] == 0 and p['phase'] == 'qk']
    assert tails == [256, 256, 256, 256, 1, 256, 256, 256, 256, 33]


@pytest.mark.parametrize("change", ["drop_tail", "drop_current", "live_padding", "padding_as_follower", "swap_request", "error"])
def test_cannot_claim_full_path_after_altering_actual_tasks(change):
    rows = witness()
    if change == "drop_tail":
        rows[0][4] -= 1
    elif change == "drop_current":
        rows[2][1] = 0
    elif change == "live_padding":
        rows[6][1] = 1
    elif change == "padding_as_follower":
        rows[6][1] = 0
    elif change == "swap_request":
        rows[3][5] = 0
    else:
        rows[0][10] = 5
    data, tasks = report(rows)  # Rehash tampered bytes: geometry must still reject.
    with pytest.raises(ValueError):
        run.geometry(data, tasks)


def test_old_q4_geometry_remains_supported():
    data, _ = report(witness())
    data.update(case=run.collector.CASE, mode="q4")
    run.geometry(data)


def test_cpu_checker_work_files_follow_owned_output_not_source_stage(tmp_path, monkeypatch):
    """Archive #129: checker logs must not exhaust the shared source tmpfs."""
    stage, output = tmp_path / 'source-stage', tmp_path / 'disk-output'
    stage.mkdir(); output.mkdir()
    monkeypatch.setenv('LD_LIBRARY_PATH', '/official/toolkit/lib64')

    def child(argv, *, cwd, check):
        assert cwd == stage
        assert argv[argv.index('--cwd') + 1] == str(output / 'cpu-0-work')
        assert (output / 'cpu-0-work').is_dir()
        assert '--env-json' not in argv  # official parent environment inherited
        receipt = {'status': 'completed', 'timed_out': False,
                   'child_returncode': 0, 'cleanup_complete': True}
        (output / 'cpu-0-control.json').write_text(json.dumps(receipt))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(qualify_q1.subprocess, 'run', child)
    control, receipt = qualify_q1.bound(stage, output, 'cpu-0', 120,
                                       ['/absolute/qualified-cpu', 'q1'])
    assert control == output / 'cpu-0-control.json'
    assert receipt['cleanup_complete'] is True
