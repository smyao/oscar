"""#168: complete saved-export recovery with official string timestamp schema.

Synthetic host fixtures only. No model, SDK parser, HTTP request or NPU work.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools import timer32k


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def saved_run(root):
    run = {"kind": "timer32k", "run_id": "saved-host-fixture", "status": "completed",
           "returncode": 1, "summary_error": 'TypeError: can only concatenate str (not "float") to str',
           "pair_hash": "same-pair", "effective_config": {"tensor_parallel_size": 4}, "services": []}
    for variant, sizes in [('native', [4]), ('candidate', [4, 8])]:
        service = {"variant": variant, "lifecycle": {"cleanup_complete": True},
                   "resource_release": {"status": "passed"}, "batches": []}
        run['services'].append(service)
        for size in sizes:
            for mode in ('control', 'profile'):
                ident = f'{variant}-{size}-{mode}'
                service['batches'].append({"run_id": ident, "variant": variant, "concurrency": size,
                    "mode": mode, "status": "completed", "pair_hash": "same-pair",
                    "prompt_manifest_sha256": f'prompts{size}',
                    "mtp_counter_delta": {"status": "observed", "mtp_counter_delta": {
                        "vllm:spec_decode_num_drafts": 100,
                        "vllm:spec_decode_num_draft_tokens": 300,
                        "vllm:spec_decode_num_accepted_tokens": 200}},
                    "request_completion_check": {"status": "passed"},
                    "sample": {"batch_elapsed_seconds": 10, "completed_requests": size,
                        "failed_requests": 0, "timeouts": 0,
                        "requests": [{"usage": {"completion_tokens": 256}} for _ in range(size)]}})
                if mode != 'profile':
                    continue
                base = root / variant / f'b{size}-profile' / 'profile'
                for rank in range(4):
                    rank_dir = base / 'windows' / ident / f'rank{rank}-pid123'
                    states, steps = {}, []
                    for step, (bucket, metric) in enumerate([(b, m) for b in ('prefill', 'decode')
                            for m in ('PipeUtilization', 'Memory')], 1):
                        name = f'step{step:06d}-{bucket}-{metric}'
                        profile = rank_dir / name
                        export = profile / 'rank_ascend_pt/ASCEND_PROFILER_OUTPUT'
                        export.mkdir(parents=True)
                        kernel = 'FusedInferAttentionScore' if variant == 'native' else 'oscar_attention_cv_aligned_decode_kernel'
                        counters = {'aic_mac_ratio': .4, 'aic_vec_ratio': .2, 'aic_mte2_ratio': .8} if metric == 'PipeUtilization' else {'aic_main_mem_read_bw(GB/s)': 80.5}
                        rows = [{'Name': 'MatMul', 'Duration(us)': 20, 'Start Time(us)': '1004.0', **counters},
                                {'Name': kernel, 'Duration(us)': 10, 'Start Time(us)': '1030.0', **counters}]
                        with (export / 'kernel_details.csv').open('w', newline='') as handle:
                            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
                        # This is the actual upstream timestamp TYPE contract:
                        # ts = convert_ns2us_str, dur = convert_ns2us_float.
                        write_json(export / 'trace_view.json', {'traceEvents': [
                            {'name': 'oscar_timer::q_rotation', 'ph': 'X', 'pid': 1, 'tid': 1, 'ts': '1000.0\t', 'dur': 25.0},
                            {'name': 'aten::matmul', 'ph': 'X', 'pid': 1, 'tid': 1, 'ts': '1001', 'dur': '3', 'args': {'correlation_id': 7}},
                            {'name': 'MatMul', 'ph': 'X', 'pid': 2, 'tid': 2, 'ts': '1004.0', 'dur': 20.0,
                             'args': {'Task Type': 'AI_CORE', 'correlation_id': 7}},
                            {'name': kernel, 'ph': 'X', 'pid': 2, 'tid': 2, 'ts': '1030', 'dur': 10.0, 'args': {'Task Type': 'AI_CORE'}},
                        ]})
                        # Previously the analyser could encounter the same type
                        # bug; preserve that history while accepting real exports.
                        state = 'error' if step == 1 else 'complete'
                        manifest = {'t': 'oscar-diagnostic-window', 'run_id': ident, 'rank': rank,
                            'window_id': name, 'bucket': bucket, 'metric': metric, 'steps': [step],
                            'profile_dir': str(profile), 'status': state, 'capture_status': 'captured',
                            'analysis_status': state}
                        if state == 'error':
                            manifest['analysis_error'] = run['summary_error']
                        write_json(profile.with_suffix('.json'), manifest)
                        states[name] = state
                        steps.append({'run_id': ident, 'rank': rank, 'step_id': step, 'bucket': bucket,
                            'profiled': True, 'host_ms': 5, 'graph_replays': int(bucket == 'decode'),
                            'scheduled_requests': size if bucket == 'decode' else 1,
                            'scheduled_prompt_tokens': 16384 if bucket == 'prefill' else 0,
                            'scheduled_decode_tokens': size * 4 if bucket == 'decode' else 0,
                            'context_min': 32768, 'context_max': 32768,
                            'scopes': [{'phase': 'q_rotation', 'start_ns': '0', 'end_ns': 1000.0}]})
                    write_json(rank_dir / 'run-summary.json', {'t': 'oscar-diagnostic-run', 'run_id': ident,
                        'rank': rank, 'window_statuses': states, 'missing_windows': {'early': [], 'decode': []}})
                    (base / f'timer-steps-rank{rank}.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in steps))
    write_json(root / 'run.json', run)
    write_json(root / 'timer-control.json', {'enabled': False})
    write_json(root / 'offline-analysis.json', {'status': 'partial', 'previous_failure': 'type error'})
    return run


def test_full_48_window_recovery_prints_real_device_breakdown_without_loading_model(tmp_path, monkeypatch, capsys):
    before = saved_run(tmp_path)
    def forbidden(*_a, **_k):
        pytest.fail('summary recovery must not load a model, send traffic or run SDK analysis')
    for name in ('apply_model_options', 'run_phase', 'managed_server', 'run_batch', 'load_tokenizer'):
        monkeypatch.setattr(timer32k, name, forbidden)
    # No torch_npu even if somebody accidentally tries its lazy import.
    monkeypatch.setitem(sys.modules, 'torch_npu', None)
    protected = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob('*')
                 if p.is_file() and (p.suffix in ('.csv', '.jsonl') or p.name == 'trace_view.json')}
    assert timer32k.main(['--summarize-only', str(tmp_path), '--config', '/missing/model/config']) == 0
    report = json.loads((tmp_path/'timer-summary.json').read_text())
    assert report['evidence_status'] == 'complete'
    windows = [w for run in report['runs'] for w in run['device_windows']]
    assert len(windows) == 48 and all(w['task_count'] == 2 for w in windows)
    assert all('q_rotation' in w['phases'] for w in windows)
    text = (tmp_path/'timer-summary.txt').read_text()
    assert len(text.splitlines()) <= 200 and len(text.encode()) <= 50 * 1024
    assert all(marker in text for marker in ('DEVICE_PHASES', 'PHASE_PMU', 'HOST_EXCLUSIVE', 'DEVICE_COMPARE', 'COMPARE', 'TOP'))
    assert text.endswith('=== OSCAR_TIMER_SUMMARY_END ===\n')
    assert json.loads((tmp_path/'run.json.before-summary-recovery').read_text()) == before
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in protected.items())
    recovery = json.loads((tmp_path/'summary-recovery.json').read_text())
    assert recovery['status'] == 'complete' and recovery['requests_sent'] == 0
    assert not recovery['model_started'] and not recovery['sdk_analysis_started']
    assert 'DEVICE_PHASES' in capsys.readouterr().out
    assert timer32k.main(['--summarize-only', str(tmp_path)]) == 0
    assert json.loads((tmp_path/'run.json.before-summary-recovery').read_text()) == before


def test_existing_directory_recovery_never_treats_absent_exports_as_success(tmp_path, monkeypatch, capsys):
    saved_run(tmp_path)
    for path in tmp_path.rglob('*.csv'):
        path.unlink()
    for path in tmp_path.rglob('trace_view.json'):
        path.unlink()
    monkeypatch.setitem(sys.modules, 'torch_npu', None)
    assert timer32k.main(['--summarize-only', str(tmp_path)]) == 2
    assert '=== OSCAR_TIMER_SUMMARY_END ===' in capsys.readouterr().out


def test_real_cli_prints_all_three_arms_without_sdk_or_model_loading(tmp_path):
    saved_run(tmp_path)
    forbidden = tmp_path / 'forbidden-sdk'
    forbidden.mkdir()
    (forbidden/'torch_npu.py').write_text('raise RuntimeError("recovery must not import torch_npu")\n')
    env = dict(os.environ, PYTHONPATH=str(forbidden) + os.pathsep + str(timer32k.ROOT))
    result = subprocess.run([sys.executable, '-m', 'tools.timer32k', '--summarize-only', str(tmp_path)],
        cwd=timer32k.ROOT, env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr + result.stdout
    assert result.stdout.count('=== OSCAR_TIMER_SUMMARY_BEGIN ===') == 1
    assert result.stdout.count('=== OSCAR_TIMER_SUMMARY_END ===') == 1
    assert all(ident in result.stdout for ident in ('native-4-profile', 'candidate-4-profile', 'candidate-8-profile'))
    assert 'DEVICE_PHASES' in result.stdout and 'PHASE_PMU' in result.stdout
    assert not (tmp_path/'effective-target.json').exists()  # no normal-run setup


@pytest.mark.parametrize('omitted', ['candidate', 'native_control'])
def test_recovery_keeps_available_breakpoints_but_never_certifies_incomplete_comparison(tmp_path, capsys, omitted):
    run = saved_run(tmp_path)
    if omitted == 'candidate':
        run['services'] = run['services'][:1]
    else:
        run['services'][0]['batches'] = run['services'][0]['batches'][1:]
    write_json(tmp_path/'run.json', run)
    assert timer32k.main(['--summarize-only', str(tmp_path)]) == 2
    report = json.loads((tmp_path/'timer-summary.json').read_text())
    assert report['evidence_status'] != 'complete'
    assert any('required_completed_batch_missing:' in item for item in report['missing'])
    text = capsys.readouterr().out
    assert 'DEVICE_PHASES' in text and '=== OSCAR_TIMER_SUMMARY_END ===' in text
