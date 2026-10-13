"""Archive #126/#129/#145/#148/#154-157: explicit CPU-small/CAModel-long gate.

CPU1341 remains timed out. Its completed prepare checkpoint is metadata only;
partial attention output is never a golden tensor. CAModel must independently
finish all1008 tasks, raw-space oracle, guards and all60 actors at fixed900s.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import struct

from csrc.experiments.c4_composed_long import generate

SMALL_EVIDENCE_SHA256 = '315deb43c59a59b420e8f39aa0fcca4adf53ff933e2ab709c235daaec1bd2336'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(args, meta, attrs):
    evidence = args.small_cpu_evidence.resolve()
    if sha(evidence) != SMALL_EVIDENCE_SHA256:
        raise ValueError('historical CPU337 evidence differs from raw-output revalidation')
    small = json.loads(evidence.read_text())
    if (small['status'] != 'revalidated_historical_complete_CPU337' or
            small['pair_sources'] != generate.pair_sources() or small['task_rows'] != 1008 or
            small['context'] != 337 or small['CPU1341'] != 'timed_out_not_numerically_qualified'):
        raise ValueError('small CPU source/scope does not bind current composition')
    control = json.loads(args.cpu_timeout_control.read_text())
    required = {'status': 'failed', 'timeout_seconds': 120, 'timed_out': True,
                'child_returncode': -15, 'supervisor_returncode': 124,
                'cleanup_complete': True, 'interrupt_signal': None}
    if any(control.get(k) != v for k, v in required.items()):
        raise ValueError('original CPU1341 timeout/cleanup receipt changed')
    command = control['command']
    if (len(command) != 4 or command[3] != 'baseline' or
            Path(command[1]).resolve() != args.fixture.resolve() or
            Path(command[2]).resolve() != args.prepared.resolve().parent):
        raise ValueError('prepare checkpoint does not belong to the original CPU1341 attempt')
    plan = json.loads((args.cpu_timeout_control.parent / 'plan.json').read_text())
    if (plan['source_binding'] != generate.build_binding() or
            plan['cpu_executable_sha256'] != sha(command[0]) or
            plan['library_sha256'] != sha(args.library)):
        raise ValueError('original CPU1341 attempt/build no longer binds current sources')
    actual = {p.name: sha(p) for p in args.fixture.iterdir() if p.is_file()}
    if actual != plan['fixture_sha256']:
        raise ValueError('original full fixture changed since CPU1341 prepare')
    receipt = json.loads((args.prepared / 'metadata.json').read_text())
    wanted = {'status': 'metadata_only', 'backend': 'real_ascendc_cpu_prepare',
        'tokens': 336, 'context': 1341, 'task_rows': 1008, 'source2_suppressed': True,
        'positions_checked': True, 'guards': True, 'input_immutable': True,
        'attention_execution': 'not_established'}
    if any(receipt.get(k) != v for k, v in wanted.items()):
        raise ValueError('actual prepare metadata receipt is incomplete')
    if (attrs != meta['attrs'] or len(attrs) != 17 or attrs[:5] != [336, 6, 1, 256, 1] or
            attrs[6] != 1008 or attrs[13:] != [64, 256, 3, 1] or
            meta['context'] != 1341 or meta['bad_position'] != -1):
        raise ValueError('undeclared complete CAModel shape')
    tasks = (args.prepared / 'tasks.bin').read_bytes()
    if (len(tasks) != 1008 * 128 or tasks != (args.fixture / 'expected_tasks.bin').read_bytes() or
            hashlib.sha256(tasks).hexdigest() != meta['input_sha256']['expected_tasks']):
        raise ValueError('real prepared tasks differ from full declared task witness')
    positions = (args.prepared / 'positions.bin').read_bytes()
    if len(positions) != 336 * 8 or struct.unpack('<336q', positions) != tuple(range(1341, 1677)):
        raise ValueError('real prepare position witness differs')
    return {'contract': 'explicit_CPU337_complete_and_CPU1341_metadata_then_full_CAModel1341',
        'CPU337': small['status'], 'CPU337_evidence_sha256': sha(evidence),
        'CPU1341': 'timed_out_not_numerically_qualified',
        'CPU1341_control_sha256': sha(args.cpu_timeout_control),
        'prepared_tasks_sha256': hashlib.sha256(tasks).hexdigest(),
        'prepared_positions_sha256': hashlib.sha256(positions).hexdigest(),
        'CAModel1341_required': 'full_same_library_all1008_tasks_336_query_oracle_and60actors',
        'same_shape_CPU_attention_output_used': False}
