"""Archive #126/#145/#148/#164 and D.4: two bounded CPU boundary cases.

Reuse completed executors under the shared local CANN lock. No kernel build,
device connection, timing experiment, or change to previous evidence.
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

HERE = Path(__file__).resolve().parent


def main(families=('c16', 'q4'), history_only=False):
    if history_only and families != ('c16',):
        raise ValueError('history-only isolation is scoped to one C16 case')
    root = HERE / 'evidence'
    root.mkdir(exist_ok=False)
    report = {'status': 'running', 'case_scope': 'R256 masked tail and empty source split',
              'npu': 'not_run', 'model_quality': 'not_run', 'performance': 'not_run',
              'phases': [], 'source_sha256': {}, 'binary_sha256': {}}
    report['isolated_history_only'] = history_only
    def save():
        (root / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    with open('/tmp/oscar-cann-validation.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report['lock_acquired'] = True
        ps = subprocess.check_output(['ps', '-eo', 'pid,ppid,stat,comm,args'], text=True)
        active = [line for line in ps.splitlines() if any(name in line for name in
            ('oscar_official_decode_cpu', 'oscar_official_clusters_cpu', 'camodel', 'msprof'))]
        if active:
            raise RuntimeError('relevant VM processes already active: ' + str(active))
        report['relevant_processes_before'] = active
        manifest = json.loads((HERE / 'source_hashes.json').read_text())
        report['source_baseline_commit'] = manifest['baseline_commit']
        for target in ('oscar_official_decode_cpu', 'oscar_official_clusters_cpu'):
            directory = BUILT / 'build/official-cpu/CMakeFiles' / (target + '.dir')
            dependencies = list(directory.rglob('*.o.d'))
            if not dependencies:
                raise RuntimeError('source dependency records missing')
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
                        raise RuntimeError('compiled source differs: ' + relative)
                    report['source_sha256'][relative] = expected
            report['binary_sha256'][target] = sha(BUILT / 'build/official-cpu' / target)
        report['fixture_program_sha256'] = {p.name: sha(p) for p in HERE.glob('*.py')}
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
            item = {'phase': name, 'returncode': rc, 'timed_out': timed_out, 'log': log.name}
            report['phases'].append(item)
            if rc:
                report.update(status='failed', failed_phase=name)
            save(); print(json.dumps(item), flush=True)
            if rc:
                print(log.read_text()[-5000:], flush=True)
                raise RuntimeError(f'{name}: rc={rc}')
        save()
        if history_only:
            phase('configure-history-harness', ['cmake', '-S', str(HERE), '-B', str(HERE / 'build-history'),
                  '-DASCEND_HOME_PATH=' + str(SDK), '-DOSCAR_COMPLETED_BUILD=' + str(BUILT / 'build/official-cpu')], 60)
            phase('compile-history-harness', ['cmake', '--build', str(HERE / 'build-history'), '--parallel', '2'], 60)
            report['binary_sha256']['oscar_history_tail_cpu'] = sha(HERE / 'build-history/oscar_history_tail_cpu')
            objects = (BUILT / 'build/official-cpu/CMakeFiles/oscar_official_clusters_cpu.dir').rglob('*.o')
            report['unchanged_production_object_sha256'] = {str(p): sha(p) for p in objects
                if p.name in {'attention_cv_official_cluster4.cpp.o', 'attention_cv_official_cluster16.cpp.o',
                              'attention_official_tasks.cpp.o', 'merge_lse.cpp.o'}}
            save()
        for family in families:
            extra = ['--history-only'] if history_only else []
            phase(family + '-fixture', [sys.executable, str(HERE / 'extended_case.py'), 'make', str(root), family] + extra, 60)
            binary = BUILT / 'build/official-cpu' / ('oscar_official_decode_cpu' if family == 'q4' else 'oscar_official_clusters_cpu')
            if history_only:
                binary = HERE / 'build-history/oscar_history_tail_cpu'
            args = [str(binary), family, str(root / family / 'fixture'), str(root / family / 'output'), '1']
            phase(family + '-prepare', args + ['prepare-only'], 60)
            phase(family + '-execute', args, 180)
            phase(family + '-analytic-check', [sys.executable, str(HERE / 'extended_case.py'), 'check', str(root), family] + extra, 30)
        report.update(status='passed', cases=len(families), lock_released_on_exit=True)
        save(); print(json.dumps({'status': 'passed', 'cases': len(families)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=('c16', 'q4'), action='append')
    parser.add_argument('--history-only', action='store_true')
    args = parser.parse_args()
    main(tuple(args.family) if args.family else ('c16', 'q4'), args.history_only)
