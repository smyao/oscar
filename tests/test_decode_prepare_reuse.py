"""Native graph orchestration with CPU command recording, not NPU graph evidence.

Archive #34/#36/#127/#128/#157/#163/#172/#174: real ForwardContext and
ACLGraphWrapper definitions drive the production attention implementation.
The graph double records our operator doubles' calls, not Torch/CANN kernels;
it establishes metadata rewrite order and consumers, not device numerics/time.
"""
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass, field, replace
import dataclasses
from enum import Enum
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch

from oscar_ascend.integration.prepare_reuse import (
    registered_decode_prepare_owner, remember_prepared, reused_slots,
)
from test_mixed_prepare_reuse import HostOps
from test_prepare_reuse import _fixture, AscendAttentionState
from test_runtime_native import definitions


class Mode(Enum):
    NONE = 0
    FULL = 1
    PIECEWISE = 2

    def is_valid_runtime_mode(self):
        return True


def setup(monkeypatch, enabled=True, capacity=112):
    f = _fixture(monkeypatch, enabled, tokens=capacity)
    native = definitions('references/vllm/vllm/forward_context.py', 'vllm.forward_context',
        {'BatchDescriptor', 'ForwardContext', 'override_forward_context', 'get_forward_context'},
        {'dataclass': dataclass, 'field': field, 'contextmanager': contextmanager,
         'CUDAGraphMode': Mode, '_forward_context': None}, monkeypatch)
    f.native = native
    f.names = list(f.states)
    f.starts = torch.arange(0, capacity + 1, 4, dtype=torch.int32)
    f.lengths = torch.full((capacity // 4,), 32772, dtype=torch.int32)
    f.tables = f.metadata.block_tables.expand(capacity // 4, -1).contiguous()
    f.slots = f.metadata.slot_mapping
    f.stream = SimpleNamespace(synchronize=lambda: None)
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(current_stream=lambda: f.stream), raising=False)
    host = HostOps()

    def prepare(starts, lengths, slots, tasks, positions, *args):
        f.counts['prepare'] += 1
        host.prepare_attention_tasks_out(starts, lengths, slots, tasks, positions, *args)
        positions.masked_fill_(slots < 0, -1)
        tasks[:, 10].view(positions.numel(), -1)[slots < 0] = 0

    previous_cv = f.impl.provider.ops.attention_cv_out
    def cv(*args):
        previous_cv(*args)
        args[13][:, 0].copy_(args[10][:, 10])

    f.impl.provider.ops.prepare_attention_tasks_out = prepare
    f.impl.provider.ops.attention_cv_out = cv
    return f


def metadata(f, n, *, capture=False, draft=False, index=0):
    return replace(f.metadata, query_start_loc=f.starts[:n // 4 + 1],
        query_start_loc_cpu=None, seq_lens=f.lengths[:n // 4], block_tables=f.tables[:n // 4],
        slot_mapping=f.slots[:n], num_reqs=n // 4, num_actual_tokens=n,
        num_input_tokens=n, max_query_len=4, current_cumulative=None,
        # Native non-MLA MTP uniform capture really uses ChunkedPrefill.
        attn_state=AscendAttentionState.ChunkedPrefill if capture else AscendAttentionState.DecodeOnly,
        capture_origin=capture, dummy_origin=capture, is_draft=draft, draft_index=index)


def context(f, meta, *, mode=Mode.NONE, capturing=False, names=None):
    names = f.names if names is None else names
    result = f.native.ForwardContext({}, {name: meta for name in names}, {},
        cudagraph_runtime_mode=mode,
        batch_descriptor=f.native.BatchDescriptor(meta.num_input_tokens, meta.num_reqs, uniform=True))
    result.capturing = capturing
    result.in_profile_run = False
    result.is_draft_model = meta.is_draft
    return result


def forward(f, meta, *, names=None):
    n = meta.num_input_tokens
    outputs = []
    for index, name in enumerate(f.names if names is None else names):
        q = torch.full((n, 6 * 64), index + 1, dtype=torch.bfloat16)
        kv = torch.zeros(n, 64, dtype=torch.bfloat16)
        output = torch.empty_like(q)
        f.impl.forward(SimpleNamespace(layer_name=name), q, kv, kv, f.states[name].packed,
                       meta, output=output)
        outputs.append(output)
    return outputs


@pytest.mark.parametrize('n', [64, 112])
def test_main_decode_changes_only_prepare_count_and_keeps_all_layer_bytes(monkeypatch, n):
    records = []
    for enabled in (False, True):
        f = setup(monkeypatch, enabled)
        meta = metadata(f, n)
        f.workspace.positions.fill_(-9)
        with f.native.override_forward_context(context(f, meta)):
            outputs = forward(f, meta)
        records.append((f, outputs))
    old, new = (record[0] for record in records)
    assert old.counts['prepare'] == 16 and new.counts['prepare'] == 1
    for name in ('rotate', 'cv', 'merge', 'store', 'guard'):
        assert old.counts[name] == new.counts[name] == 16
    assert old.counts['suppress'] == new.counts['suppress'] == 0
    assert all(torch.equal(a, b) for a, b in zip(records[0][1], records[1][1]))
    assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
               for a, b in zip(old.seen, new.seen))
    assert all(torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])
               for a, b in zip(old.stored, new.stored))


@pytest.mark.parametrize('fault', ['draft', 'profile', 'dummy', 'piecewise', 'unregistered',
                                  'ubatch', 'missing_flags', 'nonuniform_capture', 'warmup'])
def test_nonqualified_execution_does_not_share_or_leave_a_lease(monkeypatch, fault):
    f = setup(monkeypatch)
    meta = metadata(f, 64, capture=fault in {'nonuniform_capture', 'warmup'})
    ctx = context(f, meta)
    if fault == 'draft':
        meta = replace(meta, is_draft=True)
        ctx = context(f, meta)
    elif fault == 'profile': ctx.in_profile_run = True
    elif fault == 'dummy':
        meta = replace(meta, dummy_origin=True)
        ctx = context(f, meta)
    elif fault == 'piecewise': ctx.cudagraph_runtime_mode = Mode.PIECEWISE
    elif fault == 'unregistered': ctx.attn_metadata = {name: replace(meta) for name in f.names}
    elif fault == 'ubatch': ctx.ubatch_slices = object()
    elif fault == 'missing_flags': del ctx.is_draft_model
    elif fault == 'nonuniform_capture':
        ctx.capturing = True
        ctx.cudagraph_runtime_mode = Mode.FULL
        ctx.batch_descriptor = replace(ctx.batch_descriptor, uniform=False)
    with f.native.override_forward_context(ctx):
        forward(f, meta, names=f.names[:2])
    assert f.counts['prepare'] == 2
    assert f.workspace._oscar_prepared_tasks is None


def test_descriptor_identity_phase_group_and_repeated_layer_invalidate(monkeypatch):
    f = setup(monkeypatch)
    meta = metadata(f, 64)
    ctx = context(f, meta)
    shape = (64, 6, 1, 64, 256, False, 1)
    slots = f.slots[:64]
    for mutate in (
        lambda: setattr(ctx, 'capturing', not ctx.capturing),
        lambda: setattr(ctx, 'batch_descriptor', replace(ctx.batch_descriptor)),
        lambda: setattr(ctx, 'cudagraph_runtime_mode', Mode.FULL),
    ):
        remember_prepared(f.workspace, meta, 'full0', shape, f.stream, slots, ctx)
        mutate()
        assert reused_slots(f.workspace, meta, 'full1', shape, f.stream, ctx) is None
    remember_prepared(f.workspace, meta, 'full0', shape, f.stream, slots, ctx)
    assert reused_slots(f.workspace, replace(meta), 'full1', shape, f.stream, ctx) is None
    remember_prepared(f.workspace, meta, 'full0', shape, f.stream, slots, ctx)
    assert reused_slots(f.workspace, meta, 'full0', shape, f.stream, ctx) is None


def test_shared_workspace_interleaved_native_metadata_groups_reprepare(monkeypatch):
    f = setup(monkeypatch)
    first = metadata(f, 64)
    second = replace(first, seq_lens=first.seq_lens + 32768)
    ctx = context(f, first)
    ctx.attn_metadata.update(full2=second, full3=second)
    with f.native.override_forward_context(ctx):
        forward(f, first, names=['full0', 'full1'])
        forward(f, second, names=['full2', 'full3'])
        forward(f, first, names=['full4', 'full5'])
    assert f.counts['prepare'] == 3 and f.counts['cv'] == 6
    for index, (_, _, positions) in enumerate(f.stored):
        start = 65536 if index in (2, 3) else 32768
        assert torch.equal(positions, torch.arange(64) % 4 + start)


def native_graph_wrapper(monkeypatch, f, runnable):
    recording = SimpleNamespace(current=None)

    class Graph:
        def __init__(self): self.commands = []
        def replay(self):
            for _, operation in self.commands: operation()

    @contextmanager
    def capture(graph, **unused):
        recording.current = graph
        try: yield
        finally: recording.current = None

    for name in ('prepare_attention_tasks_out', 'rotate_out', 'attention_cv_out',
                 'merge_lse_out', 'rotate_clip_store_out', 'status_guard'):
        original = getattr(f.impl.provider.ops, name)
        def record(*args, _name=name, _original=original, **kwargs):
            if recording.current is not None:
                recording.current.commands.append((_name, lambda: _original(*args, **kwargs)))
            return _original(*args, **kwargs)
        monkeypatch.setattr(f.impl.provider.ops, name, record)
    torch.npu.NPUGraph = Graph
    torch.npu.graph = capture
    offloader = ModuleType('vllm.model_executor.offloader.base')
    offloader.get_offloader = lambda: SimpleNamespace(sync_prev_onload=lambda: None,
                                                     join_after_forward=lambda: None)
    monkeypatch.setitem(sys.modules, offloader.__name__, offloader)
    graph_module = definitions('references/vllm-ascend/vllm_ascend/compilation/acl_graph.py',
        'native_decode_reuse_graph', {'ACLGraphEntry', 'ACLGraphWrapper'},
        {'dataclasses': dataclasses, 'torch': torch, 'CUDAGraphMode': Mode,
         'get_forward_context': f.native.get_forward_context, 'ExitStack': ExitStack,
         'validate_cudagraph_capturing_enabled': lambda: None,
         'envs': SimpleNamespace(VLLM_LOGGING_LEVEL='INFO'),
         'current_platform': SimpleNamespace(get_global_graph_pool=lambda: None),
         'CUDAGraphOptions': lambda: SimpleNamespace(debug_log_enable=False, gc_disable=False,
                                                    weak_ref_output=False),
         '_acl_graph_wrappers': set(), '_EXTRA_CTX': SimpleNamespace(is_draft_model=False),
         'logger': SimpleNamespace(info_once=lambda *args: None),
         'weak_ref_workspaces': lambda value: None, 'weak_ref_tensors': lambda value: value,
         '_graph_params': None, '_draft_graph_params': None, '_draft_graph_prefill_params': None,
         'compilation_counter': SimpleNamespace(num_cudagraph_captured=0),
         '_is_stream_resource_capture_error': lambda exc: False}, monkeypatch)
    return graph_module.ACLGraphWrapper(runnable, SimpleNamespace(compilation_config=object()), Mode.FULL)


def test_real_native_wrapper_records_first_prepare_and_replays_changed_fixed_buffers(monkeypatch):
    f = setup(monkeypatch)
    model_calls = []
    def model():
        meta = f.native.get_forward_context().attn_metadata['full0']
        model_calls.append(meta.num_input_tokens)
        return forward(f, meta)
    wrapper = native_graph_wrapper(monkeypatch, f, model)
    pointers = tuple(getattr(f.workspace, name).data_ptr() for name in ('tasks', 'positions', 'slots'))
    for n in (64, 112):
        f.slots.fill_(-1)
        meta = metadata(f, n, capture=True)
        ctx = context(f, meta, mode=Mode.FULL)
        before = f.counts['prepare']
        with f.native.override_forward_context(ctx): wrapper()
        graph = wrapper.concrete_aclgraph_entries[ctx.batch_descriptor].aclgraph
        assert graph.commands[0][0] == 'prepare_attention_tasks_out'
        assert sum(name == 'prepare_attention_tasks_out' for name, _ in graph.commands) == 1
        assert f.counts['prepare'] - before == 1
    assert model_calls == [64, 112]
    # Eager -> captured replay overwrites scratch -> eager, even when the
    # original eager metadata/tensor identities are deliberately retained.
    eager = metadata(f, 64)
    f.slots.copy_(torch.arange(f.slots.numel()))
    f.lengths.fill_(32772)
    with f.native.override_forward_context(context(f, eager)):
        forward(f, eager)
    for index, n in enumerate((112, 64, 112)):
        # MTP's three invocations are separate writers and remain unshared.
        f.lengths.fill_(65540)
        before = f.counts['prepare']
        for draft_index in range(3):
            draft = metadata(f, n, draft=True, index=draft_index)
            with f.native.override_forward_context(context(f, draft)):
                forward(f, draft, names=['full0'])
            assert f.workspace._oscar_prepared_tasks is None
        assert f.counts['prepare'] - before == 3
        f.workspace.tasks.fill_(-777)
        f.workspace.positions.fill_(-777)
        f.lengths.fill_(32772 + index * 128)
        before = f.counts.copy()
        meta = metadata(f, n)
        with f.native.override_forward_context(context(f, meta, mode=Mode.FULL)):
            wrapper()
        assert model_calls == [64, 112]  # replay never calls Python model/lease
        assert f.counts['prepare'] - before['prepare'] == 1
        for name in ('rotate', 'cv', 'merge', 'store', 'guard'):
            assert f.counts[name] - before[name] == 16
        expected = torch.arange(n) % 4 + 32768 + index * 128
        assert all(torch.equal(positions[:n], expected) for _, positions in f.seen[-16:])
        assert all(torch.equal(positions, expected) for _, _, positions in f.stored[-16:])
    before = f.counts['prepare']
    f.lengths.fill_(131076)
    with f.native.override_forward_context(context(f, eager)):
        forward(f, eager)
    assert f.counts['prepare'] - before == 1
    assert torch.equal(f.stored[-1][2], torch.arange(64) % 4 + 131072)
    assert pointers == tuple(getattr(f.workspace, name).data_ptr() for name in ('tasks', 'positions', 'slots'))
    # Error bytes must be regenerated from the current device metadata too;
    # capturing padding must never permanently record an all-healthy result.
    f.lengths[0] = 1
    before = f.counts.copy()
    with f.native.override_forward_context(context(f, eager, mode=Mode.FULL)):
        with pytest.raises(AssertionError): wrapper()
    assert f.counts['prepare'] - before['prepare'] == 1
    assert f.counts['guard'] - before['guard'] == 1
    assert torch.count_nonzero(f.seen[-1][0][:, 10]) > 0


def test_each_layer_keeps_metadata_error_bytes_and_guard(monkeypatch):
    f = setup(monkeypatch)
    meta = metadata(f, 64)
    f.lengths[0] = 1
    with f.native.override_forward_context(context(f, meta)):
        with pytest.raises(AssertionError): forward(f, meta)
    assert f.counts['prepare'] == f.counts['cv'] == f.counts['guard'] == 1
    assert torch.count_nonzero(f.seen[0][0][:, 10]) > 0
