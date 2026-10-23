"""Archive #13-20/#126/#145/#148/#164: inspect saved real CPU outputs.

Continue only the isolated harness checks/real merge, without rerunning C16.
Preserve the initial rc1, byte-hash the copied evidence, and retain receipts.
"""
from pathlib import Path
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys

from run_guest import BUILT, SDK, sha


def main(source):
    here = Path(__file__).resolve().parent
    root = here / 'evidence'
    root.mkdir(exist_ok=False)
    report = {'scope': 'post-only on preserved real C16 outputs; no C16 replay',
              'source': str(source), 'source_cv_run_status': 'rc1 preserved',
              'status': 'running', 'phases': [], 'copied_sha256': {}}
    def save():
        (root / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    with open('/tmp/oscar-cann-validation.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = json.loads((source / 'summary.json').read_text())
        for path, expected in previous['unchanged_production_object_sha256'].items():
            if sha(Path(path)) != expected:
                raise RuntimeError('completed production object changed')
        report['unchanged_production_object_sha256'] = previous['unchanged_production_object_sha256']
        for part in ('fixture', 'output'):
            destination = root / 'c16' / part
            destination.mkdir(parents=True)
            for path in (source / 'c16' / part).iterdir():
                if path.is_file():
                    target = destination / path.name
                    target.write_bytes(path.read_bytes())
                    assert sha(target) == sha(path)
                    report['copied_sha256'][str(target.relative_to(root))] = sha(path)
        (root / 'c16/definition.json').write_bytes((source / 'c16/definition.json').read_bytes())
        env = os.environ.copy(); env['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
        env['LD_LIBRARY_PATH'] = str(SDK / 'lib64') + ':' + env.get('LD_LIBRARY_PATH', '')
        def phase(name, command, timeout):
            log = root / (name + '.log')
            print(json.dumps({'phase': name, 'state': 'started'}), flush=True)
            with log.open('w') as stream:
                child = subprocess.Popen(command, cwd=here, env=env, stdout=stream,
                                         stderr=subprocess.STDOUT, start_new_session=True)
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
            if rc:
                report.update(status='failed', failed_phase=name)
            save(); print(json.dumps(report['phases'][-1]), flush=True)
            if rc:
                raise RuntimeError(f'{name}: rc={rc}')
        save()
        phase('configure-harness', ['cmake', '-S', str(here), '-B', str(here / 'build-history'),
              '-DASCEND_HOME_PATH=' + str(SDK), '-DOSCAR_COMPLETED_BUILD=' + str(BUILT / 'build/official-cpu')], 60)
        phase('compile-harness', ['cmake', '--build', str(here / 'build-history'), '--parallel', '2'], 60)
        report['harness_binary_sha256'] = sha(here / 'build-history/oscar_history_tail_cpu')
        phase('post-only-real-merge', [str(here / 'build-history/oscar_history_tail_cpu'), 'c16',
              str(root / 'c16/fixture'), str(root / 'c16/output'), '1', 'post-only'], 60)
        phase('analytic-check', [sys.executable, str(here / 'extended_case.py'), 'check', str(root),
                                'c16', '--history-only'], 30)
        report.update(status='passed', lock_released_on_exit=True)
        save(); print(json.dumps({'status': 'passed', 'scope': report['scope']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('source', type=Path)
    main(parser.parse_args().source)
