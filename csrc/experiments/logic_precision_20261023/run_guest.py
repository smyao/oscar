"""Archive #79-85/#126/#145/#148/#164: one bounded local CPU case family.

Reuses source-bound completed SDK executors. No compilation, NPU calls or
performance sampling. The shared local CANN lock covers all child processes.
"""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BUILT = Path('/tmp/oscar-official-math-20261021')
SDK = Path('/usr/local/Ascend/cann-9.1.0')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = HERE / 'evidence'
    root.mkdir(exist_ok=False)
    manifest = json.loads((HERE / 'source_hashes.json').read_text())
    report = {'status': 'running', 'npu': 'not_run', 'model_quality': 'not_run',
              'performance': 'not_run', 'phases': [], 'source_sha256': {},
              'baseline_commit': manifest['baseline_commit'], 'binary_sha256': {}}
    def save():
        (root / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    with open('/tmp/oscar-cann-validation.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report['lock_acquired'] = True
        processes = subprocess.check_output(['ps', '-eo', 'pid,ppid,stat,comm,args'], text=True)
        active = [line for line in processes.splitlines() if any(
            name in line for name in ('oscar_official_decode_cpu', 'oscar_official_clusters_cpu', 'camodel', 'msprof'))]
        if active:
            raise RuntimeError('another relevant VM process is active: ' + str(active))
        report['relevant_processes_before'] = active
        for family, target in [('q4', 'oscar_official_decode_cpu'), ('c4', 'oscar_official_clusters_cpu')]:
            directory = BUILT / 'build/official-cpu/CMakeFiles' / (target + '.dir')
            dependencies = list(directory.rglob('*.o.d'))
            if not dependencies:
                raise RuntimeError('completed CPU source dependency records missing: ' + str(directory))
            for dependency in dependencies:
                for word in dependency.read_text().replace('\\\n', ' ').split():
                    path = Path(word)
                    if not path.is_absolute() or not path.is_relative_to(BUILT):
                        continue
                    path = path.resolve()
                    relative = str(path.relative_to(BUILT))
                    if relative.startswith('build/'):
                        continue
                    expected = manifest['sha256'].get(relative)
                    if expected is None or sha(path) != expected:
                        raise RuntimeError('completed CPU build source differs: ' + relative)
                    report['source_sha256'][relative] = expected
            binary = BUILT / 'build/official-cpu' / target
            report['binary_sha256'][target] = sha(binary)
        env = os.environ.copy()
        env['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
        env['LD_LIBRARY_PATH'] = ':'.join([str(SDK / 'lib64'), str(SDK / 'tools/tikicpulib/lib'),
            str(SDK / 'tools/tikicpulib/lib/Ascend910B1'), str(SDK / 'tools/simulator/Ascend910B1/lib'), env.get('LD_LIBRARY_PATH', '')])
        def phase(name, command, timeout):
            log = root / (name + '.log')
            print(json.dumps({'phase': name, 'state': 'started', 'deadline_seconds': timeout}), flush=True)
            with log.open('w') as stream:
                child = subprocess.Popen(command, cwd=HERE, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                timed_out = False
                try:
                    rc = child.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=5)
                    rc = 124
            report['phases'].append({'phase': name, 'returncode': rc, 'timed_out': timed_out, 'log': log.name})
            save()
            print(json.dumps(report['phases'][-1]), flush=True)
            if rc:
                report.update(status='failed', failed_phase=name)
                save()
                print(log.read_text()[-5000:], flush=True)
                raise RuntimeError(f'{name}: rc={rc}')
        save()
        for family in ('q4', 'c4', 'c16'):
            phase(family + '-fixture', [sys.executable, str(HERE / 'analytic_case.py'), 'make', str(root), family], 60)
            binary = BUILT / 'build/official-cpu' / ('oscar_official_decode_cpu' if family == 'q4' else 'oscar_official_clusters_cpu')
            args = [str(binary), family, str(root / family / 'fixture'), str(root / family / 'output'), '1']
            phase(family + '-prepare', args + ['prepare-only'], 60)
            phase(family + '-execute', args, 180)
            phase(family + '-analytic-check', [sys.executable, str(HERE / 'analytic_case.py'), 'check', str(root), family], 30)
        report.update(status='passed', cases=3, lock_released_on_exit=True)
        save()
        print(json.dumps({'status': 'passed', 'cases': 3, 'source_dependencies': len(report['source_sha256'])}), flush=True)


if __name__ == '__main__':
    main()
