"""#167: real process ownership plus offline export-contract host tests only."""
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.analyse_timer_profiles import analyse_windows
from tools import timer32k
from tools.phase import PhaseResult


def fixture(root, status="captured_pending_analysis"):
    rank = root / "profile/windows/run/rank0-pid123"
    profile = rank / "step000001-prefill-Memory"
    raw = profile / "rank0_ascend_pt"
    (raw / "FRAMEWORK").mkdir(parents=True)
    (raw / "profiler_info_0.json").write_text('{}')
    (raw / "FRAMEWORK/torch.op_range").write_bytes(b'raw host test payload')
    path = profile.with_suffix('.json')
    path.write_text(json.dumps({"t": "oscar-diagnostic-window", "run_id": "run", "rank": 0,
        "window_id": profile.name, "profile_dir": str(profile), "status": status,
        "analysis_status": "pending", "capture_paths": [raw.name], "profiled": True}))
    (rank / 'run-summary.json').write_text(json.dumps({"run_id": "run", "rank": 0,
        "missing_windows": {"early": ["PipeUtilization"], "decode": ["PipeUtilization", "Memory"]},
        "observed_steps": 7, "window_statuses": {profile.name: status}}))
    return path, raw


def _export_in_child(raw):
    path = Path(raw) / 'ASCEND_PROFILER_OUTPUT'
    path.mkdir(exist_ok=True)
    (path / 'kernel_details.csv').write_text('Name,Task Duration(us),Start Time(us)\n'
        'oscar_attention_cv_aligned_decode_kernel,8,100\n')


def test_ordinary_process_can_spawn_parser_and_verify_real_exports(tmp_path):
    manifest, raw = fixture(tmp_path)
    calls = []
    def analyzer(path, *, max_process_number):
        assert not multiprocessing.current_process().daemon
        assert max_process_number == 2 and Path(path) == raw
        calls.append(path)
        child = multiprocessing.get_context('spawn').Process(target=_export_in_child, args=(path,))
        child.start()
        try:
            child.join(10)
            assert child.exitcode == 0
        finally:
            if child.is_alive():
                child.kill(); child.join(2)
    report = analyse_windows(tmp_path, analyzer=analyzer)
    assert report['status'] == 'complete' and len(calls) == 1
    result = json.loads(manifest.read_text())
    assert result['status'] == result['analysis_status'] == 'complete'
    assert result['analysis_previous_status'] == 'captured_pending_analysis'
    saved = json.loads((manifest.parent/'run-summary.json').read_text())
    assert saved['observed_steps'] == 7 and saved['status'] == 'missing'  # missing other windows is preserved
    assert saved['window_statuses'][result['window_id']] == 'complete'
    # Valid existing export reuses without invoking the SDK again.
    analyse_windows(tmp_path, analyzer=lambda *_a, **_k: pytest.fail('reparsed existing export'))


def test_old_worker_parse_failure_is_recoverable_without_erasing_previous_status(tmp_path):
    path, raw = fixture(tmp_path, 'missing')
    report = analyse_windows(tmp_path, analyzer=lambda path, **_: _export_in_child(path))
    assert report['status'] == 'complete'
    assert json.loads(path.read_text())['analysis_previous_status'] == 'missing'


@pytest.mark.parametrize('mode', ['silent_return', 'empty_csv', 'exception', 'unfinished'])
def test_sdk_return_does_not_prove_device_export(tmp_path, mode):
    manifest, raw = fixture(tmp_path, 'collecting' if mode == 'unfinished' else 'captured_pending_analysis')
    def analyzer(path, **_):
        if mode == 'exception':
            raise RuntimeError('parser failed')
        if mode == 'empty_csv':
            (Path(path)/'kernel_details.csv').write_text('Name,Task Duration(us)\n')
        return None
    report = analyse_windows(tmp_path, analyzer=analyzer)
    assert report['status'] == 'missing'
    assert json.loads(manifest.read_text())['status'] != 'complete'


def test_analysis_refuses_daemon_before_sdk_import(tmp_path, monkeypatch):
    monkeypatch.setattr(multiprocessing, 'current_process', lambda: SimpleNamespace(daemon=True))
    with pytest.raises(RuntimeError, match='non-daemon'):
        analyse_windows(tmp_path, analyzer=lambda *_a, **_k: pytest.fail('SDK called'))


def test_supervisor_analysis_requires_cleanup_and_uses_separate_bounded_python(tmp_path, monkeypatch):
    config = {'shutdown_timeout_seconds': 2}
    manifest = {'services': [{'lifecycle': {'cleanup_complete': False}, 'resource_release': {'status': 'passed'}}]}
    assert timer32k.analyse_stopped_profiles(tmp_path, config, manifest)['status'] == 'missing'
    manifest['services'][0]['lifecycle']['cleanup_complete'] = True
    seen = []
    def run(name, command, **kwargs):
        seen.append(command)
        assert command[:3] == [timer32k.sys.executable, '-m', 'tools.analyse_timer_profiles']
        assert kwargs['timeout'] == 600 and 'OSCAR_TIMER_CONTROL' not in kwargs['env']
        assert kwargs['env']['OSCAR_ENABLED'] == '0'
        (tmp_path/'offline-analysis.json').write_text('{"status":"complete","daemon":false}')
        return PhaseResult(name, command, 0, .1, False, False, 'offline.log', True)
    monkeypatch.setattr(timer32k, 'run_phase', run)
    assert timer32k.analyse_stopped_profiles(tmp_path, config, manifest)['status'] == 'complete'
    assert len(seen) == 1


def test_supervisor_rc_zero_without_published_analysis_is_missing(tmp_path, monkeypatch):
    manifest = {'services': [{'lifecycle': {'cleanup_complete': True}, 'resource_release': {'status': 'passed'}}]}
    monkeypatch.setattr(timer32k, 'run_phase', lambda name, command, **_: PhaseResult(
        name, command, 0, .1, False, False, 'absent.log', True))
    result = timer32k.analyse_stopped_profiles(tmp_path, {'shutdown_timeout_seconds': 2}, manifest)
    assert result['status'] == 'missing'
