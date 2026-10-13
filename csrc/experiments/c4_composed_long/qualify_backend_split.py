"""Archive #129/#145/#148/#154-157: genuine alternate backend qualification.

No CPU1341 rerun or timeout extension. Keep its failed control and use only
its completed prepare checkpoint. Both complete CAModel arms retain900s,
same-library bitwise, independent oracle, all task/actor and>=1.15 gates.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from csrc.experiments.c4_composed_long import cv_compare, generate, generate_backend_split
from csrc.experiments.c4_composed_long.backend_split_contract import validate
from csrc.experiments.c4_composed_long.qualify import bound, digest, save


def qualify(args):
    if args.root.resolve() != ROOT or os.environ.get('LD_PRELOAD'):
        raise ValueError('declare stage ROOT and keep supervisor free of LD_PRELOAD')
    for field in ('output', 'fixture', 'prepared', 'library', 'small_cpu_evidence', 'cpu_timeout_control', 'env_json'):
        setattr(args, field, getattr(args, field).resolve())
    if (generate.HERE / 'cv_replay_backend_split.py').read_text() != generate_backend_split.generate_replay():
        raise ValueError('alternate replay source drift')
    meta = json.loads((args.fixture / 'fixture.json').read_text())
    attrs = list(map(int, (args.fixture / 'shape.txt').read_text().split()))
    contract = validate(args, meta, attrs)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / 'plan.json', {'source_binding': generate.build_binding(),
        'qualification': contract, 'CAModel_deadline_seconds_per_arm': 900,
        'minimum_complete_CV_speedup': 1.15, 'CPU1341_rerun': False,
        'library_sha256': digest(args.library), 'runner_sha256': digest(__file__)})
    controls = {}
    for arm in ('baseline', 'candidate'):
        controls[arm] = bound(ROOT, args.output, arm, 900,
            [sys.executable, str(generate.HERE / 'cv_replay_backend_split.py'), '--variant', arm,
             '--fixture', str(args.fixture), '--prepared', str(args.prepared),
             '--library', str(args.library), '--output', str(args.output / arm),
             '--small-cpu-evidence', str(args.small_cpu_evidence),
             '--cpu-timeout-control', str(args.cpu_timeout_control)], args.env_json)
    result = cv_compare.compare(args.output / 'baseline', args.output / 'candidate',
                                controls['baseline'], controls['candidate'])
    for arm in controls:
        report = json.loads((args.output / arm / 'report.json').read_text())
        if report.get('qualification') != contract: raise ValueError('backend qualification changed ' + arm)
    result['qualification'] = contract
    save(args.output / 'comparison.json', result)
    passed = result['combination_gate'] == 'passed'
    save(args.output / 'summary.json', {'status': 'passed' if passed else 'insufficient_gain',
        'CPU337': contract['CPU337'], 'CPU1341': contract['CPU1341'], 'CAModel1341': result,
        'NPU': 'not_run', 'graph': 'not_run', 'production_enabled': False,
        'whole_service': 'not_established'})
    print('C4_COMPOSED_RESULT', json.dumps(result), flush=True)
    return 0 if passed else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'output', 'fixture', 'prepared', 'library', 'small-cpu-evidence', 'cpu-timeout-control', 'env-json'):
        parser.add_argument('--' + name, type=Path, required=True)
    raise SystemExit(qualify(parser.parse_args()))
