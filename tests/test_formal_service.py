"""Archive G21/G22/#94/#95/#120/#174: real host lifecycle, never NPU evidence."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from tools import formal_service as service
from tools.phase import PhaseResult, cleanup_group, group_exists


ROOT = Path(__file__).resolve().parents[1]
MEMORY_ERROR = ('ValueError: Free memory on device (1.19/29.49 GiB) on startup is less than '
                'desired GPU memory utilization (0.9, 26.54 GiB).')
WORKER = r'''
import os,signal,sys,time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(sys.argv[1]).write_text(str(os.getpid()))
while True: time.sleep(1)
'''
API = r'''
import subprocess,sys,time
from pathlib import Path
p = subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2]])
deadline=time.monotonic()+5
while not Path(sys.argv[2]).exists():
    if time.monotonic()>deadline: raise RuntimeError("worker fixture startup timed out")
    time.sleep(.01)
print("host fixture worker ready", flush=True)
if sys.argv[3] != "wait": sys.exit(int(sys.argv[3]))
while True: time.sleep(1)
'''


def api_command(tmp_path, code):
    worker, api = tmp_path / 'worker.py', tmp_path / 'api.py'
    worker.write_text(WORKER)
    api.write_text(API)
    return [sys.executable, str(api), str(worker), str(tmp_path / 'worker.pid'), str(code)]


def wait_file(path, process):
    deadline = time.monotonic() + 5
    while not path.exists():
        assert process.poll() is None
        assert time.monotonic() < deadline
        time.sleep(.01)
    return path.read_text()


def test_old_exec_style_leaves_worker_after_api_failure(tmp_path):
    # The old direct launcher becomes the API process: no owner remains to
    # run cleanup after it exits. Reproduce that missing lifecycle boundary.
    process = subprocess.Popen(api_command(tmp_path, 7), start_new_session=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert process.wait(timeout=5) == 7
        worker = int((tmp_path / 'worker.pid').read_text())
        os.kill(worker, 0)
        assert group_exists(process.pid)
    finally:
        assert cleanup_group(process, .1)


@pytest.mark.parametrize('code', [0, 7])
def test_supervisor_reaps_leftover_worker_preserves_api_status_and_outsider(tmp_path, code):
    outsider = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'], start_new_session=True)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        result = service.run_service(api_command(tmp_path, code), cwd=ROOT,
            log_dir=tmp_path/'logs', env=dict(os.environ), grace=.1)
        lifecycle = json.loads((tmp_path/'logs/serve-lifecycle.json').read_text())
        assert result.returncode == code and result.cleanup_complete
        assert not result.timed_out and not result.interrupted
        assert not group_exists(lifecycle['pgid'])
        assert outsider.poll() is None
        assert 'host fixture worker ready' in Path(result.log).read_text()
        assert lifecycle['result']['returncode'] == code and lifecycle['timeout'] is None
        assert {sig: signal.getsignal(sig) for sig in previous} == previous
    finally:
        assert cleanup_group(outsider, .1)


@pytest.mark.parametrize('signum', [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_supervisor_signal_cleans_api_and_stubborn_worker(tmp_path, signum):
    command = api_command(tmp_path, 'wait')
    runner = tmp_path/'supervisor.py'
    runner.write_text('import json,os,sys\nfrom pathlib import Path\n'
                     f'sys.path.insert(0,{str(ROOT)!r})\n'
                     'from tools.formal_service import run_service\n'
                     f'r=run_service({command!r},cwd=Path({str(ROOT)!r}),'
                     f'log_dir=Path({str(tmp_path / "logs")!r}),env=dict(os.environ),grace=.3)\n'
                     'sys.exit(r.returncode)\n')
    supervisor = subprocess.Popen([sys.executable, str(runner)], start_new_session=True,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_file(tmp_path/'worker.pid', supervisor)
        supervisor.send_signal(signum)
        # The supervisor owns shutdown: another signal must not abort cleanup.
        time.sleep(.08)
        if supervisor.poll() is None:
            supervisor.send_signal(signum)
        assert supervisor.wait(timeout=6) == 128 + signum
        lifecycle = json.loads((tmp_path/'logs/serve-lifecycle.json').read_text())
        assert lifecycle['result']['cleanup_complete']
        assert lifecycle['result']['interrupted'] and lifecycle['signal'] == signum
        assert not group_exists(lifecycle['pgid'])
    finally:
        cleanup_group(supervisor, .1)


def test_device_leases_are_per_card_noninherited_and_released(tmp_path):
    script = tmp_path/'lease.py'
    script.write_text('import sys\nfrom pathlib import Path\n'
                     f'sys.path.insert(0,{str(ROOT)!r})\n'
                     'from tools.formal_service import device_lease\n'
                     'with device_lease(Path(sys.argv[1]),[1,2,3],Path(sys.argv[1])/"old"):\n'
                     ' print("leased",flush=True)\n sys.stdin.readline()\n')
    holder = subprocess.Popen([sys.executable, str(script), str(tmp_path)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == 'leased'
        # Acquire card 0 first then fail on card 3: partial acquisitions release.
        with pytest.raises(RuntimeError, match='physical NPU 3 already has an active direct launcher'):
            with service.device_lease(tmp_path, [0, 3, 4], tmp_path/'new'):
                pytest.fail('must not launch overlapping model')
        with service.device_lease(tmp_path, [0], tmp_path/'partial-released'):
            pass
        with service.device_lease(tmp_path, [4, 5, 6, 7], tmp_path/'other'):
            pass
        assert holder.poll() is None  # No live owner was killed.
        holder.communicate('\n', timeout=5)
        assert holder.returncode == 0
        with service.device_lease(tmp_path, [0, 1, 2, 3], tmp_path/'next'):
            # exec children must not retain the parent's locks.
            child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(.3)'], close_fds=False)
        with service.device_lease(tmp_path, [0, 1, 2, 3], tmp_path/'again'):
            assert child.poll() is None
        assert child.wait(timeout=5) == 0
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout=5)


def test_failed_cleanup_cannot_report_success(tmp_path, monkeypatch):
    monkeypatch.setattr(service, 'cleanup_group', lambda process, grace: False)
    result = service.run_service([sys.executable, '-c', 'pass'], cwd=ROOT,
                                 log_dir=tmp_path, env=dict(os.environ), grace=0)
    assert result.returncode == 125 and not result.cleanup_complete


def test_child_signal_exit_retains_raw_status_and_conventional_shell_code(tmp_path):
    result = service.run_service([sys.executable, '-c', 'import os,signal;os.kill(os.getpid(),signal.SIGKILL)'],
                                 cwd=ROOT, log_dir=tmp_path, env=dict(os.environ), grace=.1)
    assert result.returncode == 137 and result.cleanup_complete
    assert json.loads((tmp_path/'serve-lifecycle.json').read_text())['child_returncode'] == -9


def test_signal_before_popen_returns_does_not_lose_child_handle(tmp_path, monkeypatch):
    original = subprocess.Popen
    original_signal = signal.signal
    spawned = []
    def interrupted_popen(*args, **kwargs):
        child = original(*args, **kwargs)
        spawned.append(child)
        signal.raise_signal(signal.SIGTERM)
        return child
    def repeated_during_cleanup(signum, handler):
        if signum == signal.SIGINT and handler == signal.SIG_IGN:
            # TERM has not been set to IGN yet: the handler must coalesce it.
            signal.raise_signal(signal.SIGTERM)
        return original_signal(signum, handler)
    monkeypatch.setattr(service.subprocess, 'Popen', interrupted_popen)
    monkeypatch.setattr(service.signal, 'signal', repeated_during_cleanup)
    result = service.run_service([sys.executable, '-c', 'import time;time.sleep(30)'], cwd=ROOT,
                                 log_dir=tmp_path, env=dict(os.environ), grace=.1)
    assert result.returncode == 143 and result.interrupted and result.cleanup_complete
    assert len(spawned) == 1 and spawned[0].poll() is not None
    assert not group_exists(spawned[0].pid)


def test_first_signal_during_normal_exit_cleanup_cannot_skip_cleanup(tmp_path, monkeypatch):
    original_signal = signal.signal
    received = []
    def during_cleanup(signum, handler):
        if signum == signal.SIGINT and handler == signal.SIG_IGN:
            received.append(signal.SIGTERM)
            signal.raise_signal(signal.SIGTERM)
        return original_signal(signum, handler)
    monkeypatch.setattr(service.signal, 'signal', during_cleanup)
    result = service.run_service(api_command(tmp_path, 0), cwd=ROOT, log_dir=tmp_path/'logs',
                                 env=dict(os.environ), grace=.1)
    assert received == [signal.SIGTERM]
    assert result.returncode == 0 and result.cleanup_complete


@pytest.mark.parametrize('available', [True, False])
def test_native_memory_failure_keeps_budget_and_records_readonly_evidence(tmp_path, monkeypatch, available):
    config = {'devices': [0,1,2,3], 'gpu_memory_utilization': .9}
    before = dict(config)
    log = tmp_path/'serve.log'; log.write_text(MEMORY_ERROR)
    result = PhaseResult('serve', ['service'], 1, .1, False, False, str(log), True)
    calls = []
    def observe(name, command, **kwargs):
        calls.append(command)
        assert command == ['/fake/npu-smi', 'info']
        assert kwargs['timeout'] == 10 and kwargs['grace'] == 1
        return PhaseResult(name, command, 124, .1, True, False, 'occupancy.log', True)
    monkeypatch.setattr(service.shutil, 'which', lambda *args, **kwargs: '/fake/npu-smi' if available else None)
    monkeypatch.setattr(service, 'run_phase', observe)
    service.memory_failure_diagnostics(result, config, cwd=ROOT, log_dir=tmp_path, env=dict(os.environ))
    report = json.loads((tmp_path/'serve-memory-failure.json').read_text())
    assert report['native_errors'] == [MEMORY_ERROR]
    assert report['service_returncode'] == 1 and config == before
    assert report['ownership'] == 'not_established_by_memory_error'
    assert report['processes_terminated_by_diagnostic'] == []
    assert bool(calls) == available
    if available:
        assert report['occupancy_evidence']['returncode'] == 124


def test_success_and_unrelated_errors_do_not_query_npus(tmp_path, monkeypatch):
    monkeypatch.setattr(service.shutil, 'which', lambda *args, **kwargs: pytest.fail('no occupancy query'))
    log = tmp_path/'serve.log'; log.write_text('different startup failure')
    for rc in (0, 7):
        service.memory_failure_diagnostics(PhaseResult('serve', [], rc, 0., False, False, str(log), True),
                                           {}, cwd=ROOT, log_dir=tmp_path, env={})
    assert not (tmp_path/'serve-memory-failure.json').exists()
