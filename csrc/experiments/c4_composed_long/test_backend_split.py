"""Archive #129/#145/#148/#154-157: host evidence-contract tests only."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from csrc.experiments.c4_composed_long import backend_split_contract as contract
from csrc.experiments.c4_composed_long import generate, generate_backend_split
from csrc.experiments.c4_composed_long.fixture import prepared_formula
import struct


def test_alternate_backend_keeps_complete_device_math_and_oracle_body():
    alternate = generate_backend_split.generate_replay()
    assert alternate == (generate.HERE / 'cv_replay_backend_split.py').read_text()
    original = (generate.HERE / 'cv_replay.py').read_text()
    assert "receipt['status']!='passed'" in original
    assert "qualification=validate(a,meta,attrs)" in alternate
    assert "same_shape_CPU_attention_output_used" not in original


def test_only_completed_prepare_can_feed_long_backend(tmp_path):
    fixture = tmp_path / 'fixture'; fixture.mkdir()
    attempt = tmp_path / 'cpu-attempt'; attempt.mkdir()
    prepared = attempt / 'cpu-baseline' / 'prepared'; prepared.mkdir(parents=True)
    tasks = b''.join(struct.pack('<16q', *row) for row in prepared_formula(336, 1341))
    positions = struct.pack('<336q', *range(1341, 1677))
    (fixture / 'expected_tasks.bin').write_bytes(tasks)
    (prepared / 'tasks.bin').write_bytes(tasks); (prepared / 'positions.bin').write_bytes(positions)
    metadata = {'status': 'metadata_only', 'backend': 'real_ascendc_cpu_prepare',
        'tokens': 336, 'context': 1341, 'task_rows': 1008, 'source2_suppressed': True,
        'positions_checked': True, 'guards': True, 'input_immutable': True,
        'attention_execution': 'not_established'}
    (prepared / 'metadata.json').write_text(json.dumps(metadata))
    executable = tmp_path / 'test-only-cpu'; executable.write_bytes(b'host contract fixture')
    library = tmp_path / 'test-only-lib'; library.write_bytes(b'host contract fixture library')
    control = {'status': 'failed', 'timeout_seconds': 120, 'timed_out': True,
        'child_returncode': -15, 'supervisor_returncode': 124, 'cleanup_complete': True,
        'interrupt_signal': None, 'command': [str(executable), str(fixture), str(prepared.parent), 'baseline']}
    control_path = attempt / 'cpu-baseline-control.json'; control_path.write_text(json.dumps(control))
    (attempt / 'plan.json').write_text(json.dumps({'source_binding': generate.build_binding(),
        'cpu_executable_sha256': contract.sha(executable), 'library_sha256': contract.sha(library),
        'fixture_sha256': {p.name: contract.sha(p) for p in fixture.iterdir()}}))
    evidence = generate.ROOT / 'reports/c4_composed_long_20261013/small_cpu_evidence.json'
    args = SimpleNamespace(small_cpu_evidence=evidence, cpu_timeout_control=control_path,
                           prepared=prepared, fixture=fixture, library=library)
    attrs = [336, 6, 1, 256, 1, 14, 1008, 512, 4, 64, 69632, 82688, 323, 64, 256, 3, 1]
    meta = {'attrs': attrs, 'context': 1341, 'bad_position': -1,
            'input_sha256': {'expected_tasks': contract.sha(fixture / 'expected_tasks.bin')}}
    result = contract.validate(args, meta, attrs)
    assert result['CPU1341'] == 'timed_out_not_numerically_qualified'
    assert result['same_shape_CPU_attention_output_used'] is False
    control['cleanup_complete'] = False; control_path.write_text(json.dumps(control))
    with pytest.raises(ValueError, match='timeout/cleanup'): contract.validate(args, meta, attrs)
    control['cleanup_complete'] = True; control_path.write_text(json.dumps(control))
    (prepared / 'tasks.bin').write_bytes(tasks[:-128])
    with pytest.raises(ValueError, match='prepared tasks'): contract.validate(args, meta, attrs)
