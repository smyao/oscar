"""Archive #133/#145/#164: output ownership, bounded diagnosis and source identity.

Host contracts for the explicit diagnostic seam, not Ascend execution.
"""
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from oscar_ascend.integration import c16_breakdown as hook
from oscar_ascend.integration.dummy_context import native_dummy_run


def metadata(**changes):
    return NS(**({'max_query_len': 16384, 'max_seq_len': 32768,
                  'num_reqs': 1, 'is_draft': False, 'draft_index': 0,
                  'capture_origin': False, 'dummy_origin': False} | changes))


@pytest.fixture
def selected(tmp_path, monkeypatch):
    monkeypatch.setenv('OSCAR_C16_BREAKDOWN_DIR', str(tmp_path))
    monkeypatch.setattr(hook, '_seen', set())
    monkeypatch.setattr(hook, '_serial', 0)
    monkeypatch.setattr(hook, '_loaded', None)
    return tmp_path


def test_off_unselected_dummy_and_graph_leave_original_callable(monkeypatch, selected):
    original = lambda *args: 'original'
    monkeypatch.delenv('OSCAR_C16_BREAKDOWN_DIR')
    assert hook.wrap(original, layer='layer', metadata=metadata(), tokens=16384) is original
    monkeypatch.setenv('OSCAR_C16_BREAKDOWN_DIR', str(selected))
    for changes, tokens in [({}, 128), ({'max_seq_len': 16384}, 16384),
                           ({'max_query_len': 4}, 16384), ({'capture_origin': True}, 16384),
                           ({'dummy_origin': True}, 16384)]:
        assert hook.wrap(original, layer='layer', metadata=metadata(**changes), tokens=tokens) is original
    with native_dummy_run():
        assert hook.wrap(original, layer='layer', metadata=metadata(), tokens=16384) is original


def test_actual_capture_never_loads_or_allocates_diagnostic(selected, monkeypatch):
    monkeypatch.setattr(torch, 'npu', NS(is_current_stream_capturing=lambda: True), raising=False)
    def forbidden_load(_):
        pytest.fail('capture must not load the timer library')
    monkeypatch.setattr(hook, '_load', forbidden_load)
    original = lambda *args: args
    wrapped = hook.wrap(original, layer='layer', metadata=metadata(), tokens=16384)
    assert wrapped('same input') == ('same input',)
    assert not hook._seen and not list(selected.iterdir())


def _args():
    # Tiny D stands in for NPU buffers. The real binding enforces D256/GQA.
    n = 16384
    tensors = [torch.empty(n, 1, 1, dtype=torch.bfloat16) for _ in range(4)]
    tensors += [torch.ones(1, 1, dtype=torch.bfloat16), torch.empty(16, dtype=torch.uint8),
                torch.zeros(1, 1, dtype=torch.int32), torch.zeros(1), torch.zeros(1),
                torch.zeros(1, dtype=torch.int64), torch.zeros(1, dtype=torch.int64),
                torch.empty(n, 1, 3, 1), torch.empty(n, 1, 3),
                torch.zeros(n * 3, 2, dtype=torch.int32), torch.zeros(16, dtype=torch.uint8),
                torch.zeros(1, 8, dtype=torch.int64)]
    return (*tensors, 2688, 1, 0, 1, 64, 256, 3, 1, .0625, 0, 1)


@pytest.mark.parametrize('mismatch', [False, True])
def test_same_inputs_separate_outputs_original_results_retained_and_failure_saved(selected, monkeypatch, mismatch):
    calls = []
    class Event:
        def __init__(self, **kwargs):
            assert kwargs == {'enable_timing': True}
        def record(self, stream):
            calls.append('event')
        def synchronize(self):
            calls.append('complete')
        def elapsed_time(self, end):
            return 7.0
    monkeypatch.setattr(torch, 'npu', NS(Event=Event, current_stream=lambda: 'stream',
                                       is_current_stream_capturing=lambda: False), raising=False)
    monkeypatch.setattr(hook, '_load', lambda _: {'phase_schema_id': 'host-fixture'})
    schema = types.ModuleType('oscar_ascend.c16_timer_schema')
    schema.SCHEMA_ID = 'host-fixture'
    schema.CLOCK_HZ = 50000000
    schema.CLOCK_SOURCE = 'host-fixture'
    schema.counter_shape = lambda cores: (3 * cores, 4, 2)
    schema.decode_ticks = lambda rows, cores: {'actors': [{'total_cycles': 50000}], 'valid': True, 'issues': []}
    monkeypatch.setitem(sys.modules, schema.__name__, schema)
    args = _args()
    def fill(values):
        for index, value in ((11, 7), (12, 8), (13, 0), (15, 9)):
            values[index].fill_(value)
    def original(*values):
        assert values[11] is args[11]
        calls.append('production'); fill(values)
    def diagnostic(*values):
        assert calls.index('production') < calls.index('complete')
        assert all(values[i] is args[i] for i in (*range(11), 14))
        assert all(values[i].data_ptr() != args[i].data_ptr() for i in hook._OUTPUTS)
        calls.append('diagnostic'); fill(values); values[-1].zero_()
        if mismatch:
            values[11].view(-1)[0] += 1
    monkeypatch.setattr(torch.ops, 'oscar_c16_timer', NS(attention_out=diagnostic))
    wrapped = hook.wrap(original, layer='model.layers.3.self_attn.attn', metadata=metadata(), tokens=16384)
    if mismatch:
        with pytest.raises(RuntimeError, match='validation failed'):
            wrapped(*args)
    else:
        wrapped(*args)
    assert torch.all(args[11] == 7) and torch.all(args[12] == 8) and torch.all(args[13] == 0)
    records = [p for p in selected.glob('call-*.json') if not p.name.endswith('-ticks.json')]
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record['status'] == ('failed' if mismatch else 'complete')
    assert record['validation']['status'] == ('failed' if mismatch else 'passed')
    assert record['baseline_ms'] == record['diagnostic_ms'] == 7
    assert record['inference_uses'] == 'original_production_outputs'
    assert hook.wrap(original, layer='model.layers.3.self_attn.attn', metadata=metadata(), tokens=16384) is original


def test_bitwise_comparison_detects_signed_zero():
    a = list(_args()); b = list(a)
    for index in hook._OUTPUTS:
        a[index] = torch.zeros_like(a[index]); b[index] = a[index].clone()
    b[11].view(-1)[0] = -0.0
    result = hook._compare(torch, a, b)
    assert result['status'] == 'failed' and result['bitwise']['partial'] is False


def test_library_receipt_source_and_binary_are_verified_before_loading(selected, monkeypatch):
    library = selected / 'diagnostic.so'; library.write_bytes(b'host-test-placeholder')
    kernel = selected / 'kernel.so'; kernel.write_bytes(b'host-test-kernel-placeholder')
    root = Path(hook.__file__).resolve().parents[2]
    source = 'csrc/kernels/attention_cv_aligned_cluster16.cpp'
    manifest = {'status': 'built', 'library': str(library), 'library_sha256': hook._digest(library),
                'kernel_library': str(kernel), 'kernel_library_sha256': hook._digest(kernel),
                'source_sha256': {source: hook._digest(root / source)}, 'phase_schema_id': 'fixture'}
    path = selected / 'build.json'; path.write_text(json.dumps(manifest))
    monkeypatch.setenv('OSCAR_C16_TIMER_LIBRARY', str(library))
    monkeypatch.setenv('OSCAR_C16_TIMER_MANIFEST', str(path))
    seen = []
    fake = NS(ops=NS(load_library=lambda p: seen.append(p)))
    assert hook._load(fake) == manifest and seen == [str(kernel), str(library)]
    monkeypatch.setattr(hook, '_loaded', None)
    manifest['source_sha256'][source] = 'stale'; path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match='source changed'):
        hook._load(fake)
    assert len(seen) == 2
    manifest['source_sha256'][source] = hook._digest(root / source)
    path.write_text(json.dumps(manifest))
    kernel.write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='kernel identity'):
        hook._load(fake)
    assert len(seen) == 2
