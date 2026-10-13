"""Archive #126/#129/#145/#148/#154-157: reuse genuine C337 CPU evidence.

Never changes the incomplete CPU1341 result. Rechecks archived raw outputs,
real completion controls and exact current-source inheritance on this host.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from csrc.experiments.c4_composed_long import generate
from csrc.experiments.c4_mdl_load import generate as mdl
from tools.package_c4_member_pipeline import package


def sha(value): return hashlib.sha256(value).hexdigest()


def audit():
    archive = ROOT / 'reports/c4_mdl_load_20261012/cpu-baseline-complete.tar.gz'
    candidate = ROOT / 'reports/c4_member_pipeline_20261010/reports/c4-pipeline-cpu337'
    prefix = 'reports/c4-mdl-cpu/'
    with tarfile.open(archive) as bundle:
        def old(name): return bundle.extractfile(prefix + name).read()
        reports = {'baseline': json.loads(old('baseline/report.json')),
                   'candidate': json.loads((candidate / 'candidate/report.json').read_text())}
        controls = {'baseline': json.loads(old('baseline-control.json')),
                    'candidate': json.loads((candidate / 'candidate-control.json').read_text())}
        for arm in generate.SOURCES:
            report, control = reports[arm], controls[arm]
            if any(report.get(k) != v for k, v in {'status': 'passed', 'arm': arm,
                'context': 337, 'bad_position': -1, 'clusters': 1, 'guards': True,
                'input_immutable': True, 'all_status_lanes': True, 'oracle': 'numeric'}.items()):
                raise ValueError('historical complete CPU337 receipt differs: ' + arm)
            if any(control.get(k) != v for k, v in {'status': 'completed', 'timeout_seconds': 120,
                'child_returncode': 0, 'supervisor_returncode': 0, 'cleanup_complete': True,
                'timed_out': False, 'interrupt_signal': None}.items()):
                raise ValueError('historical CPU337 did not complete: ' + arm)
        body, identity = mdl.generate(False)
        binding = json.loads(old('baseline/source-binding.json'))
        if binding != {'production_H16_sha256': generate.SOURCES['baseline'][1],
                       'kernel_sha256': sha(body.encode()), 'arm': 'baseline'}:
            raise ValueError('archived H16 CPU source is not the current H16 mother')
        packaging = package(check=True)
        if packaging['target_sha256'] != generate.SOURCES['candidate'][1]:
            raise ValueError('archived pipeline inheritance differs from current candidate')
        hashes = {}
        for name, size in {'partial.bin': 6193152, 'lse.bin': 24192, 'status.bin': 8064,
                           'cluster_stats.bin': 1280, 'tasks.bin': 129024,
                           'prepared/positions.bin': 2688}.items():
            a = old('baseline/' + name); b = (candidate / 'candidate' / name).read_bytes()
            if len(a) != size or a != b: raise ValueError('historical H16/pipeline C337 full bytes differ: ' + name)
            hashes[name] = sha(a)
        receipts = {'baseline_control': sha(old('baseline-control.json')),
                    'baseline_report': sha(old('baseline/report.json')),
                    'baseline_source_binding': sha(old('baseline/source-binding.json')),
                    'candidate_control': sha((candidate / 'candidate-control.json').read_bytes()),
                    'candidate_report': sha((candidate / 'candidate/report.json').read_bytes())}
    return {'status': 'revalidated_historical_complete_CPU337', 'context': 337, 'tokens': 336,
        'task_rows': 1008, 'cores': 20, 'oracle': 'numeric_all_queries_frozen_0p005',
        'pair_sources': generate.pair_sources(), 'equal_full_outputs_sha256': hashes,
        'H16_source_binding': binding, 'pipeline_packaging': packaging,
        'controls': controls, 'reports': reports, 'receipt_sha256': receipts,
        'baseline_archive': str(archive.relative_to(ROOT)), 'baseline_archive_sha256': sha(archive.read_bytes()),
        'candidate_evidence': str(candidate.relative_to(ROOT)),
        'CPU1341': 'timed_out_not_numerically_qualified', 'CAModel1341': 'pending_full_oracle_all_actors',
        'NPU': 'not_run', 'whole_service': 'not_established'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); result = audit(); a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'pair_sources': result['pair_sources'],
                      'CPU1341': result['CPU1341'], 'report_sha256': sha(a.output.read_bytes())}))
