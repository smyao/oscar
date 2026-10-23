"""Archive #13-20/#148/#164 and D.4: bounded paired decoder proof.

Two tiny functional CPU calls use identical real writer inputs and unchanged
source-bound kernels. No NPU, model-quality or performance measurement.
"""
from pathlib import Path
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys

from run_guest import BUILT, SDK, sha


def main():
    here = Path(__file__).resolve().parent
    root = here / 'evidence'; root.mkdir(exist_ok=False)
    report = {'status': 'running', 'scope': 'same real writer input, mode0/mode1 reader and merge',
              'npu': 'not_run', 'model_quality': 'not_run', 'performance': 'not_run',
              'phases': [], 'source_sha256': {}}
    def save():
        (root / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    with open('/tmp/oscar-cann-validation.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report['free_disk_before'] = shutil.disk_usage(here).free
        if report['free_disk_before'] < 1024 ** 3:
            raise RuntimeError('less than 1 GiB scratch space; do not start CPU executors')
        manifest = json.loads((here / 'source_hashes.json').read_text())
        directory = BUILT / 'build/official-cpu/CMakeFiles/oscar_official_decode_cpu.dir'
        for dependency in directory.rglob('*.o.d'):
            for word in dependency.read_text().replace('\\\n', ' ').split():
                path = Path(word)
                if not path.is_absolute() or not path.is_relative_to(BUILT):
                    continue
                path = path.resolve(); relative = str(path.relative_to(BUILT))
                if relative.startswith('build/'):
                    continue
                expected = manifest['sha256'].get(relative)
                if expected is None or sha(path) != expected:
                    raise RuntimeError('compiled CPU source differs: ' + relative)
                report['source_sha256'][relative] = expected
        if not report['source_sha256']:
            raise RuntimeError('completed CPU source bindings unavailable')
        binary = BUILT / 'build/official-cpu/oscar_official_decode_cpu'
        report['binary_sha256'] = sha(binary)
        report['fixture_program_sha256'] = sha(here / 'dequant_case.py')
        env = os.environ.copy(); env['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
        env['LD_LIBRARY_PATH'] = str(SDK / 'lib64') + ':' + env.get('LD_LIBRARY_PATH', '')
        def phase(name, command, timeout):
            print(json.dumps({'phase': name, 'state': 'started'}), flush=True)
            log = root / (name + '.log')
            with log.open('w') as stream:
                child = subprocess.Popen(command, cwd=here, env=env, stdout=stream,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                timed_out = False
                try:
                    rc = child.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True; os.killpg(child.pid, signal.SIGTERM)
                    try: child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=5)
                    rc = 124
            report['phases'].append({'phase': name, 'returncode': rc, 'timed_out': timed_out, 'log': log.name})
            if rc: report.update(status='failed', failed_phase=name)
            save(); print(json.dumps(report['phases'][-1]), flush=True)
            if rc:
                print(log.read_text()[-4000:], flush=True)
                raise RuntimeError(f'{name}: rc={rc}')
        save()
        for mode in (0, 1):
            phase(f'mode{mode}-fixture', [sys.executable, str(here / 'dequant_case.py'), 'make', str(root), str(mode)], 30)
        fingerprints = [{p.name: sha(p) for p in (root / f'mode{mode}/fixture').iterdir()} for mode in (0, 1)]
        if fingerprints[0] != fingerprints[1]:
            raise RuntimeError('paired inputs differ')
        report['paired_input_sha256'] = fingerprints[0]
        save()
        for mode in (0, 1):
            args = [str(binary), 'q4', str(root / f'mode{mode}/fixture'), str(root / f'mode{mode}/output'), str(mode)]
            phase(f'mode{mode}-prepare', args + ['prepare-only'], 60)
            phase(f'mode{mode}-execute', args, 60)
            phase(f'mode{mode}-analytic-check', [sys.executable, str(here / 'dequant_case.py'), 'check', str(root), str(mode)], 30)
        a, b = [json.loads((root / f'mode{mode}/analytic_check.json').read_text()) for mode in (0, 1)]
        assert a['absolute_error_to_unquantized_attention'] < 1e-6
        assert b['absolute_error_to_unquantized_attention'] > .02
        report.update(status='passed', cases=2, paired_input_bytes_identical=True,
                      mode0_output=a['actual_merged_output'], mode1_output=b['actual_merged_output'],
                      target_unquantized_output=.6, lock_released_on_exit=True)
        save(); print(json.dumps({'status': 'passed', 'mode0': report['mode0_output'], 'mode1': report['mode1_output']}), flush=True)


if __name__ == '__main__':
    main()
