"""Export real prepared tasks without claiming CPU attention completion.

The original long-shape CPU timeout is preserved. These tasks feed complete
CAModel baseline/candidate runs with an independent mathematical oracle.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.phase import atomic_json, run_phase
from csrc.experiments.window_batch_decode.run_cpu import _save_trace_summary,validation_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--export-dir', type=Path, required=True)
    parser.add_argument('--log-dir', type=Path, required=True)
    args = parser.parse_args()
    report = {'status': 'running', 'attention_numerics': 'not_run', 'cases': []}
    for case in json.loads(args.cases.read_text())['cases']:
        if case['expected_error']:
            raise ValueError('This exporter requires valid complete long cases')
        out = args.export_dir / case['name']
        out.mkdir(parents=True, exist_ok=False)
        with tempfile.TemporaryDirectory(prefix='oscar-long-prepare-', dir='/dev/shm') as work:
            result = run_phase(case['name'], [str(args.executable.resolve()), case['mode'],
                str((args.cases.parent / case['name']).resolve()), str(out.resolve()), 'prepare-only'],
                cwd=Path(work), log_dir=args.log_dir, timeout=120, grace=5, heartbeat=30,
                env={**os.environ, 'OSCAR_TERMINAL_LOG_MODE': 'compact'})
            _save_trace_summary(Path(work), args.log_dir / (case['name'] + '-trace'))
        text = Path(result.log).read_text()
        markers = [json.loads(line) for line in text.splitlines()
                   if line.startswith('{"backend":"ascendc_cpu_debug"')]
        ok = (result.returncode == 0 and result.cleanup_complete and not result.timed_out
              and len(markers) == 1 and markers[0].get('status') == 'prepared_only'
              and markers[0].get('positions') == 'passed' and markers[0].get('gm_guards') is True
              and markers[0].get('all_tasks_exported') is True
              and '[ERROR]' not in text and 'error happened!' not in text)
        hashes = {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                  for name in ('tasks.bin', 'attrs.txt')} if ok else {}
        report['cases'].append({'name': case['name'], 'status': 'prepared_only' if ok else 'failed',
                                'evidence': markers, 'sha256': hashes, 'log': result.log})
        report['status'] = 'prepared_only' if ok else 'failed'
        atomic_json(args.export_dir / 'preparation.json', report)
        if not ok:
            return 1
    return 0


if __name__ == '__main__':
    with validation_lock():
        raise SystemExit(main())
