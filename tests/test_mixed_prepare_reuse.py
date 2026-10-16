"""Archive #34/#36/#129/#142/#157: host lease and real-device gate contracts.

The CPU operator below is an orchestration double, not NPU accuracy evidence.
Pinned native ForwardContext code is executed unchanged from references.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from enum import Enum
import json
from types import SimpleNamespace
import sys

import pytest
import torch

from oscar_ascend.integration.current_attention import _empty_current_source_ranges
from oscar_ascend.integration.mixed_decode_split import _slice_metadata, plan_mixed_decode
from oscar_ascend.integration.prepare_reuse import (
    invalidate_prepare_reuse, prepare_views, remember_prepared, reused_slots,
)
from oscar_ascend.integration.runtime_api import OscarReadinessError
from tools import probe_mixed_prepare_reuse as probe


def pinned_context(monkeypatch):
    from test_runtime_native import definitions
    class CUDAGraphMode(Enum):
        NONE = 0
        def is_valid_runtime_mode(self): return True
    native = definitions('references/vllm/vllm/forward_context.py', 'vllm.forward_context',
        {'ForwardContext', 'override_forward_context', 'get_forward_context'},
        {'dataclass': dataclass, 'field': field, 'contextmanager': contextmanager,
         'CUDAGraphMode': CUDAGraphMode,
         '_forward_context': None}, monkeypatch)
    stream = object()
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(current_stream=lambda: stream), raising=False)
    return native, stream


class HostOps:
    def prepare_attention_tasks_out(self, starts, lengths, slots, tasks, positions, h, hk,
                                    sink, recent, splits, *unused):
        n = positions.numel()
        requests = torch.bucketize(torch.arange(n), starts[1:].to(torch.int64), right=True)
        context = lengths[requests] - (starts[requests+1] - starts[requests])
        positions.copy_(context + torch.arange(n) - starts[requests])
        positions[context < 0] = -1
        tasks.copy_(torch.arange(tasks.numel()).view_as(tasks))
        view = tasks.view(n, hk, 3, splits, 16)
        view[..., 3] = context[:, None, None, None]
        view[..., 4] = lengths[requests, None, None, None]
        view[..., 7] = torch.arange(3)[None, None, :, None]
        view[..., 10] = (context < 0)[:, None, None, None]

    def status_guard(self, *statuses):
        assert all(not torch.count_nonzero(t) for t in statuses)


def children(metadata):
    return [
        _slice_metadata(metadata, 0, probe.SHORT_REQUESTS, 0, probe.SHORT_TOKENS,
                        probe.TOKENS, 'short_prefix', prepare_reuse=True),
        _slice_metadata(metadata, probe.SHORT_REQUESTS, probe.SHORT_REQUESTS+1,
                        probe.SHORT_TOKENS, probe.TOKENS, probe.TOKENS,
                        'long_suffix', prepare_reuse=True),
    ]


def test_complete_metadata_16_layers_on_off_same_addresses_and_new_context(monkeypatch):
    pinned_context(monkeypatch)
    workspace, audit = probe.allocate(torch, 'cpu')
    metadata, inputs = probe.parent_metadata(torch, probe.input_values(torch, 32768), 'cpu')
    pointers = [t.data_ptr() for t in (*inputs.values(), workspace.tasks, workspace.positions, workspace.slots)]
    outputs = {}; receipts = {}
    for enabled in (False, True):
        for index, case in enumerate(probe.CASE_NAMES):
            values = probe.input_values(torch, 32768 if index == 0 else 65536,
                                        invalid_length=index == 2)
            for key, tensor in inputs.items(): tensor.copy_(values[key])
            receipt = probe.execute_epoch(torch, HostOps(), workspace, audit, metadata,
                enabled=enabled, epoch=(enabled, index), suppress=_empty_current_source_ranges)
            assert probe._counts_valid(receipt, 'on' if enabled else 'off')
            snapshot = probe._snapshot(audit)
            probe._check(torch, snapshot, values, invalid_length=index == 2)
            if enabled:
                assert all(torch.equal(snapshot[key], outputs[case][key]) for key in snapshot)
            else: outputs[case] = snapshot
            receipts[enabled, index] = receipt
    assert all(receipts[True, i]['native_context_id'] != receipts[True, i+1]['native_context_id'] for i in range(2))
    assert pointers == [t.data_ptr() for t in (*inputs.values(), workspace.tasks, workspace.positions, workspace.slots)]
    assert not torch.equal(outputs[probe.CASE_NAMES[0]]['positions'], outputs[probe.CASE_NAMES[1]]['positions'])
    assert receipts[False, 0]['task_bytes_written'] == 96 * 1024**2
    assert receipts[True, 0]['task_bytes_written'] == 6 * 1024**2


def test_region_order_and_layer_order_are_independent_and_cpu_mirror_is_not_owner(monkeypatch):
    native, stream = pinned_context(monkeypatch)
    workspace, _ = probe.allocate(torch, 'cpu')
    metadata, _ = probe.parent_metadata(torch, probe.input_values(torch, 32768), 'cpu')
    names = ['full7', 'full3', 'full15']
    with probe.native_context(metadata, names) as context:
        short, long = children(metadata)
        metadata.query_start_loc_cpu.fill_(-999)  # mutable mirror is not cached or trusted
        plan = plan_mixed_decode(metadata, probe.TOKENS, immutable_starts=True)
        assert plan.prefix_tokens == probe.SHORT_TOKENS
        for child in (long, short):
            n = child.num_actual_tokens; shape = (n, 6, 1, 64, 256, 1)
            tasks, positions, slots = prepare_views(workspace, child, n, 1, 1, names[0], context)
            assert reused_slots(workspace, child, names[0], shape, stream, context) is None
            tasks.fill_(n); positions.fill_(n); slots.fill_(n)
            remember_prepared(workspace, child, names[0], shape, stream, slots, context)
        for name in names[1:]:
            for child in (short, long):
                shape = (child.num_actual_tokens, 6, 1, 64, 256, 1)
                slots = reused_slots(workspace, child, name, shape, stream, context)
                assert slots is not None and bool((slots == child.num_actual_tokens).all())
        # Even replacing the immutable tuple expires both regions; a mutable
        # CPU field's assumed stability never establishes the lease.
        # OscarMetadata is frozen; bypass it only as an explicit fault injection.
        object.__setattr__(metadata, 'current_cumulative', tuple(list(metadata.current_cumulative)))
        assert reused_slots(workspace, long, 'new', shape, stream, context) is None
        assert workspace._oscar_prepared_tasks is None


@pytest.mark.parametrize('fault', ['owner', 'span', 'slot_alias', 'length_alias', 'table_alias',
                                  'tasks_capacity', 'positions_capacity', 'slots_capacity', 'capture'])
def test_invalid_region_or_arena_fails_before_write(monkeypatch, fault):
    pinned_context(monkeypatch)
    workspace, _ = probe.allocate(torch, 'cpu')
    parent, _ = probe.parent_metadata(torch, probe.input_values(torch, 32768), 'cpu')
    with probe.native_context(parent, ['full0']) as context:
        child = children(parent)[1]
        if fault == 'owner': context.attn_metadata['full0'] = replace(parent)
        elif fault == 'span': child = replace(child, prepare_region=replace(child.prepare_region, token_begin=0))
        elif fault.endswith('_alias'):
            name = {'slot_alias': 'slot_mapping', 'length_alias': 'seq_lens', 'table_alias': 'block_tables'}[fault]
            child = replace(child, **{name: getattr(child, name).clone()})
        elif fault.endswith('_capacity'):
            name = fault.removesuffix('_capacity'); setattr(workspace, name, getattr(workspace, name)[:-1])
        elif fault == 'capture': context.capturing = True
        with pytest.raises(OscarReadinessError):
            prepare_views(workspace, child, child.num_actual_tokens, 1, 1, 'full0', context)


def report_fixture():
    cases = []
    for index, case in enumerate(probe.CASE_NAMES):
        receipts = {mode: {'prepare': 2 if mode == 'on' else 32,
            'suppress': 2 if mode == 'on' else 32, 'boundary_guards': 16,
            'consumer_copies': 64, 'task_bytes_written': (6 if mode == 'on' else 96)*1024**2,
            'native_context_class': 'vllm.forward_context.ForwardContext',
            'native_context_id': index+1, 'extra_production_hbm_bytes': 0} for mode in probe.MODES}
        cases.append({'case': case, 'status': 'passed', 'bitwise': True,
            'task_error': 'invalid_long_length_code1_preserved' if index == 2 else 'all_zero',
            'complete_task_fields': True, 'complete_positions': True, **receipts})
    return {'status': 'passed', 'device_completion': 'passed',
        'validation_mode': 'metadata_latency', 'performance': 'passed',
        'precision': 'all_task_fields_and_positions_bitwise', 'scope': 'mixed_prepare_only_no_CV_no_model',
        'same_addresses': True, 'source_inputs_unchanged': True, 'tokens': probe.TOKENS,
        'layers': 16, 'split_tokens': [124, 16260], 'cases': cases,
        'timing': {'samples_ms': {'off': [2.0]*5, 'on': [1.0]*5}, 'off_ms': 2.0, 'on_ms': 1.0,
                   'ratio': .5, 'speedup': 2.0, 'minimum_speedup': 1.0,
                   'scope': probe.TIMING_SCOPE, 'gate': 'passed'}}


@pytest.mark.parametrize('fault', ['none', 'stale_context', 'missing_error', 'extra_launch',
                                  'missing_consumer', 'regression', 'no_time', 'source_mutated'])
def test_device_gate_rejects_missing_correctness_or_complete_path_timing(fault):
    report = report_fixture()
    if fault == 'stale_context': report['cases'][1]['on']['native_context_id'] = 1
    elif fault == 'missing_error': report['cases'][2]['task_error'] = 'all_zero'
    elif fault == 'extra_launch': report['cases'][0]['on']['prepare'] = 32
    elif fault == 'missing_consumer': report['cases'][0]['on']['consumer_copies'] = 0
    elif fault == 'regression': report['timing']['samples_ms']['on'] = [2.01]*5
    elif fault == 'no_time': report['timing']['samples_ms']['on'] = []
    elif fault == 'source_mutated': report['source_inputs_unchanged'] = False
    assert bool(probe.validate_evidence(report)) is (fault != 'none')


def test_probe_requires_real_npu_before_loading_native_or_ops(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, 'torch_npu', SimpleNamespace())
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(is_available=lambda: False), raising=False)
    monkeypatch.setenv('ASCEND_RT_VISIBLE_DEVICES', '0,1,2,3')
    config = json.loads((probe.ROOT/'configs/target.json').read_text())
    config['experimental_prepare_reuse'] = True
    path = tmp_path/'config.json'; path.write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match='target NPU required; no CPU substitution'):
        probe.probe(path, probe.ROOT/'configs/acceptance.json')


@pytest.mark.parametrize('context', [32768, 65536])
def test_real_forward_and_mixed_dispatch_reuse_without_skipping_layer_work(monkeypatch, context):
    from test_prepare_reuse import _fixture
    runs = []
    for enabled in (False, True):
        f = _fixture(monkeypatch, enabled, context=context, tokens=516)
        f.impl.provider.config['experimental_mixed_decode_split'] = True
        f.metadata = replace(f.metadata, mixed_decode_split=True, num_reqs=2,
            query_start_loc=torch.tensor([0, 4, 516], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 4, 516], dtype=torch.int32),
            seq_lens=torch.tensor([context+4, context+512], dtype=torch.int32),
            block_tables=f.metadata.block_tables.expand(2, -1).contiguous(),
            current_cumulative=(4, 516), max_query_len=512)
        outputs = f.run(f.metadata)
        runs.append((f, outputs))
    base, candidate = (entry[0] for entry in runs)
    assert base.counts['prepare'] == base.counts['suppress'] == 32
    assert candidate.counts['prepare'] == candidate.counts['suppress'] == 2
    for name in ('rotate', 'cv', 'merge', 'store', 'slot_guard', 'current_fia'):
        assert base.counts[name] == candidate.counts[name] == 32
    assert base.counts['guard'] == candidate.counts['guard'] == 48
    assert all(torch.equal(a, b) for a, b in zip(runs[0][1], runs[1][1]))
    assert all(torch.equal(a[0], b[0]) for a, b in zip(base.seen, candidate.seen))
    assert all(torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])
               for a, b in zip(base.stored, candidate.stored))
    assert len(candidate.workspace._oscar_prepared_tasks) == 2


def test_default_16k_mixed_forward_selects_bf16_decode_and_c16_with_prepare_reuse(monkeypatch):
    """Real dispatcher and numeric metadata, shape-only QKV to avoid model math."""
    from test_prepare_reuse import _fixture, AscendAttentionState
    from tools.serving_variants import variant_config
    from oscar_ascend.runtime import WorkspaceGeometry
    f = _fixture(monkeypatch, True, tokens=probe.TOKENS, dim=256, data_device='meta')
    config = variant_config(json.loads((probe.ROOT/'configs/target.json').read_text()), 'candidate')
    f.impl.provider.config = config
    f.workspace.geometry = WorkspaceGeometry(probe.TOKENS, 6, 1, 256, cube_cores=20,
                                             history_cluster_size=16, c16_l1_reuse=True)
    f.workspace.cluster_stats = torch.empty(20, 8, dtype=torch.int64)
    metadata, _ = probe.parent_metadata(torch, probe.input_values(torch, 32768), 'cpu')
    metadata = replace(metadata, attn_state=AscendAttentionState.ChunkedPrefill)
    ops = f.impl.provider.ops
    ops.rotate_vector_init_out = ops.rotate_out
    ops.rotate_clip_store_striped_out = ops.rotate_clip_store_out
    selected = []
    for name in ('attention_cv_bf16_decode_out', 'attention_cv_bf16_cluster16_out'):
        def cv(*args, name=name):
            assert args[4].dtype == torch.bfloat16
            selected.append(name); ops.attention_cv_out(*args)
        setattr(ops, name, cv)
    for state in f.states.values():
        state.cache_format = 'striped_v1'
        state.rotation_v_bf16 = torch.empty(256, 256, dtype=torch.bfloat16, device='meta')
        state.rotation_v_nz = None
    f.run(metadata)
    assert selected == ['attention_cv_bf16_decode_out', 'attention_cv_bf16_cluster16_out']*16
    assert f.counts['prepare'] == f.counts['suppress'] == 2
    assert f.counts['cv'] == f.counts['current_fia'] == f.counts['store'] == f.counts['merge'] == 32
    assert f.counts['guard'] == 48


@pytest.mark.parametrize('kv_heads,splits', [(1, 3), (2, 1)])
def test_region_capacity_uses_descriptor_units_and_overlapping_writer_expires(monkeypatch, kv_heads, splits):
    _, stream = pinned_context(monkeypatch)
    metadata, _ = probe.parent_metadata(torch, probe.input_values(torch, 32768), 'cpu')
    factor = kv_heads*3*splits
    workspace = SimpleNamespace(tasks=torch.empty(probe.TOKENS*factor, 16, dtype=torch.int64),
        positions=torch.empty(probe.TOKENS, dtype=torch.int64), slots=torch.empty(probe.TOKENS, dtype=torch.int64))
    with probe.native_context(metadata, ['full0', 'full1']) as context:
        short, long = children(metadata)
        for child in (short, long):
            n = child.num_actual_tokens; shape = (n, 6, kv_heads, 64, 256, splits)
            tasks, positions, slots = prepare_views(workspace, child, n, kv_heads, splits, 'full0', context)
            assert tasks.storage_offset() == child.prepare_region.token_begin*factor*16
            assert positions.storage_offset() == slots.storage_offset() == child.prepare_region.token_begin
            remember_prepared(workspace, child, 'full0', shape, stream, slots, context)
        # Any whole-arena writer invalidates both regions before its write.
        whole_shape = (probe.TOKENS, 6, kv_heads, 64, 256, splits)
        assert reused_slots(workspace, metadata, 'full1', whole_shape, stream, context) is None
        assert workspace._oscar_prepared_tasks is None
        remember_prepared(workspace, long, 'full0', shape, stream, slots, context)
        invalidate_prepare_reuse(workspace)
        assert reused_slots(workspace, long, 'full1', shape, stream, context) is None


@pytest.mark.parametrize('fault', ['none', 'stale_source', 'stale_binary', 'incomplete', 'release', 'wrong_mode'])
def test_observe_gate_binds_sources_and_artifact_keeps_reports_separate(monkeypatch, tmp_path, fault):
    from tools import observe_serve
    from oscar_ascend.ops import loader
    manifest = {'signature': 'current-build', 'sha256': {'extension': 'current-lib'}}
    monkeypatch.setattr(loader, 'validate_build_artifacts', lambda *args: manifest)
    monkeypatch.setattr(observe_serve, 'read_npu_resources', lambda *args, **kwargs: {})
    monkeypatch.setattr(observe_serve, 'wait_for_release', lambda *args, **kwargs:
                        {'status': 'failed' if fault == 'release' else 'passed'})
    def phase(name, command, **kwargs):
        assert name == 'mixed-prepare-reuse' and command[2] == 'tools.probe_mixed_prepare_reuse'
        output = probe.Path(command[command.index('--output')+1])
        assert output.name == name+'-report.json'
        report = report_fixture()
        report.update(artifact_signature=manifest['signature'], artifact_sha256=manifest['sha256'],
                      source_binding=probe.source_binding())
        if fault == 'stale_source': report['source_binding'] = {}
        elif fault == 'stale_binary': report['artifact_sha256'] = {'extension': 'old'}
        elif fault == 'incomplete': report['cases'].pop()
        elif fault == 'wrong_mode':
            report.update(validation_mode='correctness_only_user_requested', timing=None, performance='not_run_user_requested')
        output.write_text(json.dumps(report))
        (tmp_path/(name+'.json')).write_text(json.dumps({'returncode': 0}))
    monkeypatch.setattr(observe_serve, '_phase', phase)
    config = json.loads((probe.ROOT/'configs/target.json').read_text())
    status = {}
    call = lambda: observe_serve._mixed_prepare_reuse_gate(probe.ROOT/'configs/target.json',
                                                          config, {}, tmp_path, status)
    if fault == 'none':
        call()
        assert status['mixed_prepare_reuse_gate']['status'] == 'passed'
        assert json.loads((tmp_path/'mixed-prepare-reuse.json').read_text()) == {'returncode': 0}
    else:
        with pytest.raises(RuntimeError): call()
        assert 'mixed_prepare_reuse_gate' not in status


def test_prepare_gate_failure_blocks_managed_service_and_preserves_rc(monkeypatch, tmp_path):
    from tools import observe_serve
    for name in ('_preflight', '_bf16_attention_gate', '_striped_cache_gate'):
        monkeypatch.setattr(observe_serve, name, lambda *args: None)
    def failure(*args):
        error = RuntimeError('metadata gate rejected'); error.returncode = 2
        error.phase = 'mixed-prepare-reuse'; raise error
    monkeypatch.setattr(observe_serve, '_mixed_prepare_reuse_gate', failure)
    monkeypatch.setattr(observe_serve, 'managed_server', lambda *args, **kwargs: pytest.fail('gate must block service'))
    monkeypatch.setattr(observe_serve, '_terminal', lambda *args, **kwargs: None)
    assert observe_serve.run(probe.ROOT/'configs/target.json', tmp_path, 'candidate') == 2
    status = json.loads((tmp_path/'status.json').read_text())
    assert status['failed_phase'] == 'mixed-prepare-reuse' and status['returncode'] == 2
    effective = json.loads((tmp_path/'effective-target.json').read_text())
    assert effective['experimental_prepare_reuse'] is True
    assert effective['max_num_batched_tokens'] == 16384 and effective['enable_prefix_caching'] is False


@pytest.mark.parametrize('fault', ['none', 'wrong_mode', 'timing_present', 'missing_error', 'missing_consumer'])
def test_bf16_metadata_correctness_gate_needs_no_timing_but_keeps_semantics(fault):
    report = report_fixture()
    report.update(validation_mode='correctness_only_user_requested', timing=None,
                  performance='not_run_user_requested')
    if fault == 'wrong_mode': report['validation_mode'] = 'metadata_latency'
    elif fault == 'timing_present': report['timing'] = {'off_ms': 1.0}
    elif fault == 'missing_error': report['cases'][2]['task_error'] = 'all_zero'
    elif fault == 'missing_consumer': report['cases'][0]['on']['consumer_copies'] = 0
    if fault == 'wrong_mode': report['timing'] = {}
    assert bool(probe.validate_evidence(report)) is (fault != 'none')
