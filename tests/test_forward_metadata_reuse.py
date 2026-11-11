"""Same native invocation derivation reuse; host evidence, never NPU timing.

Archive #34/#36/#127/#128/#157/#163/#171/#172/#174. Execute the production
forward/dispatcher with native ForwardContext definitions and existing probe
metadata. Operator doubles preserve the established per-layer work receipts.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from oscar_ascend.integration import mixed_decode_split as mixed
from test_mixed_prepare_reuse import pinned_context, probe
from test_prepare_reuse import _fixture


def fixture(monkeypatch, enabled=True):
    f = _fixture(monkeypatch, enabled, tokens=516)
    f.impl.provider.config['experimental_mixed_decode_split'] = True
    f.metadata = replace(f.metadata, mixed_decode_split=True, num_reqs=2,
        query_start_loc=torch.tensor([0, 4, 516], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 4, 516], dtype=torch.int32),
        seq_lens=torch.tensor([32772, 33280], dtype=torch.int32),
        block_tables=f.metadata.block_tables.expand(2, -1).contiguous(),
        current_cumulative=(4, 516), max_query_len=512,
        window_map=torch.tensor([[0, 7], [1, 9]], dtype=torch.int64))
    native, stream = pinned_context(monkeypatch)
    return f, native, stream


def derivation_counts(monkeypatch):
    counts = {'plan': 0, 'slice': 0}
    original_plan, original_slice = mixed.plan_mixed_decode, mixed._slice_metadata

    def plan(metadata, *args, **kwargs):
        if metadata.mixed_decode_split:
            counts['plan'] += 1
        return original_plan(metadata, *args, **kwargs)

    def slice_metadata(*args, **kwargs):
        counts['slice'] += 1
        return original_slice(*args, **kwargs)

    monkeypatch.setattr(mixed, 'plan_mixed_decode', plan)
    monkeypatch.setattr(mixed, '_slice_metadata', slice_metadata)
    return counts


def test_real_forward_all_layers_derive_once_with_identical_work_and_outputs(monkeypatch):
    counts = derivation_counts(monkeypatch)
    baseline, _, _ = fixture(monkeypatch)
    baseline.workspace.positions.fill_(-7)
    # Preserve the already-existing task/position reuse on both sides. Only
    # remove the new host derivation lease from this reference execution.
    def uncached(metadata, tokens, *, query_heads, kv_heads, prepare_reuse, **unused):
        return mixed.plan_mixed_decode(metadata, tokens, query_heads=query_heads,
            kv_heads=kv_heads, immutable_starts=prepare_reuse), None
    with monkeypatch.context() as before:
        before.setattr(mixed, 'plan_mixed_decode_for_forward', uncached)
        with probe.native_context(baseline.metadata, list(baseline.states)):
            expected = baseline.run(baseline.metadata)
    assert counts == {'plan': 16, 'slice': 32}
    counts.update(plan=0, slice=0)
    candidate, _, _ = fixture(monkeypatch)
    candidate.workspace.positions.fill_(-7)
    with probe.native_context(candidate.metadata, list(candidate.states)) as context:
        actual = candidate.run(candidate.metadata)
        lease, = context._oscar_mixed_metadata
        assert len(lease.children) == 2
        short, long = lease.children
        assert short.prepare_region.owner is long.prepare_region.owner is candidate.metadata
        assert short.window_map.data_ptr() == candidate.metadata.window_map.data_ptr()
        assert long.window_map.data_ptr() == candidate.metadata.window_map[1:].data_ptr()
        assert tuple(short.current_cumulative) == (4,)
        assert tuple(long.current_cumulative) == (512,)
    assert counts == {'plan': 1, 'slice': 2}
    for name in ('rotate', 'cv', 'merge', 'store', 'slot_guard', 'current_fia', 'guard'):
        assert candidate.counts[name] == baseline.counts[name]
    assert candidate.counts['guard'] == 48
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
               for a, b in zip(baseline.seen, candidate.seen))
    assert all(torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])
               for a, b in zip(baseline.stored, candidate.stored))


def test_same_buffers_next_native_step_rederive_and_consume_changed_values(monkeypatch):
    f, _, _ = fixture(monkeypatch)
    counts = derivation_counts(monkeypatch)
    parent = f.metadata
    addresses = tuple(t.data_ptr() for t in (parent.query_start_loc, parent.seq_lens,
                                            parent.slot_mapping, parent.block_tables))
    with probe.native_context(parent, ['full0', 'full1']) as first_context:
        f.run(parent, ['full0', 'full1'])
        first_children = first_context._oscar_mixed_metadata[0].children
    # The native runner reuses physical buffers but builds a new per-step
    # frozen parent. Keep token capacity constant and change real boundaries,
    # lengths, pages and slots; a pointer-keyed cache would retain stale starts.
    parent.query_start_loc[1] = 3
    parent.query_start_loc_cpu[1] = 3
    parent.seq_lens.copy_(torch.tensor([65539, 66049], dtype=torch.int32))
    parent.slot_mapping.add_(32768)
    parent.block_tables.add_(256)
    following = replace(parent, current_cumulative=(3, 516), max_query_len=513,
                        max_seq_len=66049)
    with probe.native_context(following, ['full0', 'full1']) as next_context:
        f.run(following, ['full0', 'full1'])
        next_children = next_context._oscar_mixed_metadata[0].children
    assert counts == {'plan': 2, 'slice': 4}
    assert next_children[0] is not first_children[0]
    assert next_children[0].query_start_loc.tolist() == [0, 3]
    assert next_children[1].query_start_loc.tolist() == [0, 513]
    assert next_children[1].seq_lens.tolist() == [66049]
    assert next_children[1].block_tables.data_ptr() == following.block_tables[1:].data_ptr()
    assert next_children[1].slot_mapping.data_ptr() == following.slot_mapping[3:].data_ptr()
    assert addresses == tuple(t.data_ptr() for t in (following.query_start_loc, following.seq_lens,
                                                    following.slot_mapping, following.block_tables))
    assert torch.equal(f.stored[-1][1], following.slot_mapping[3:])


def test_unchanged_parent_in_new_context_and_repeated_layer_both_renew(monkeypatch):
    f, _, _ = fixture(monkeypatch)
    counts = derivation_counts(monkeypatch)
    with probe.native_context(f.metadata, ['full0', 'full1']):
        f.run(f.metadata, ['full0', 'full1'])
        f.run(f.metadata, ['full0', 'full1'])
    with probe.native_context(f.metadata, ['full0', 'full1']):
        f.run(f.metadata, ['full0', 'full1'])
    assert counts == {'plan': 3, 'slice': 6}


def test_replaced_immutable_boundaries_expire_the_parent_lease(monkeypatch):
    f, _, _ = fixture(monkeypatch)
    counts = derivation_counts(monkeypatch)
    with probe.native_context(f.metadata, ['full0', 'full1']) as context:
        f.run(f.metadata, ['full0'])
        previous = context._oscar_mixed_metadata[0]
        # Frozen production metadata cannot be rebound this way. Inject the
        # same explicit replacement protected by the existing prepare lease.
        object.__setattr__(f.metadata, 'current_cumulative', tuple(list(f.metadata.current_cumulative)))
        f.run(f.metadata, ['full1'])
        assert context._oscar_mixed_metadata[0] is not previous
    assert counts == {'plan': 2, 'slice': 4}


def test_failed_subforward_cannot_publish_derived_children(monkeypatch):
    f, _, _ = fixture(monkeypatch)
    counts = derivation_counts(monkeypatch)
    forward = f.impl.forward

    def fail_child(layer, query, key, value, kv_cache, metadata, output=None, **kwargs):
        if metadata.mixed_split_segment == 'short_prefix':
            raise RuntimeError('subforward failed')
        return forward(layer, query, key, value, kv_cache, metadata, output=output, **kwargs)

    with probe.native_context(f.metadata, ['full0', 'full1']) as context:
        monkeypatch.setattr(f.impl, 'forward', fail_child)
        with pytest.raises(RuntimeError, match='subforward failed'):
            f.run(f.metadata, ['full0'])
        assert context._oscar_mixed_metadata[0].children is None
        assert f.workspace._oscar_prepared_tasks is None
        monkeypatch.setattr(f.impl, 'forward', forward)
        f.run(f.metadata, ['full1'])
        assert len(context._oscar_mixed_metadata[0].children) == 2
    assert counts == {'plan': 1, 'slice': 3}


@pytest.mark.parametrize('fault', ['device_boundary', 'padding'])
def test_reused_children_do_not_skip_each_layers_device_guard(monkeypatch, fault):
    f, _, _ = fixture(monkeypatch)
    if fault == 'padding':
        f.metadata = replace(f.metadata, num_actual_tokens=515, current_cumulative=(4, 515),
            query_start_loc=torch.tensor([0, 4, 515], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 4, 515], dtype=torch.int32), max_query_len=511)
        f.metadata.slot_mapping[-1] = -1
    counts = derivation_counts(monkeypatch)
    with probe.native_context(f.metadata, ['full0', 'full1']):
        f.run(f.metadata, ['full0'])
        completed = f.counts['cv']
        if fault == 'device_boundary':
            f.metadata.query_start_loc[1] += 1
        else:
            f.metadata.slot_mapping[-1] = 0
        with pytest.raises(AssertionError):
            f.run(f.metadata, ['full1'])
        assert f.counts['cv'] == completed
    assert counts == {'plan': 1, 'slice': 2}


@pytest.mark.parametrize('origin', ['capture', 'dummy', 'first_draft', 'unregistered'])
def test_excluded_origins_keep_original_uncached_path(monkeypatch, origin):
    f, _, _ = fixture(monkeypatch)
    if origin == 'capture': f.metadata = replace(f.metadata, capture_origin=True)
    elif origin == 'dummy': f.metadata = replace(f.metadata, dummy_origin=True)
    elif origin == 'first_draft':
        f.metadata = replace(f.metadata, is_draft=True, first_draft_current_fia=True)
    counts = derivation_counts(monkeypatch)
    registered = replace(f.metadata) if origin == 'unregistered' else f.metadata
    with probe.native_context(registered, ['full0', 'full1']) as context:
        f.run(f.metadata, ['full0', 'full1'])
        assert not getattr(context, '_oscar_mixed_metadata', [])
    assert counts['plan'] == 2
    assert counts['slice'] == (4 if origin in {'first_draft', 'unregistered'} else 0)


@pytest.mark.parametrize('changed', ['parent', 'geometry', 'layout', 'flags', 'stream'])
def test_native_probe_metadata_lease_requires_the_entire_scope(monkeypatch, changed):
    native, stream = pinned_context(monkeypatch)
    parent, _ = probe.parent_metadata(torch, probe.input_values(torch, 32768), 'cpu')
    counts = derivation_counts(monkeypatch)
    fields = dict(query_heads=6, kv_heads=1, head_size=64, geometry=(probe.TOKENS, 6, 1, 64),
                  layout=('canonical_v1', 2304), flags=(True, False), prepare_reuse=True)
    with probe.native_context(parent, ['full0', 'full1']) as context:
        first, first_lease = mixed.plan_mixed_decode_for_forward(
            parent, probe.TOKENS, layer_name='full0', **fields)
        following = parent
        if changed == 'parent':
            following = replace(parent)
            context.attn_metadata['full1'] = following
        elif changed == 'geometry': fields['geometry'] = (probe.TOKENS, 6, 1, 64, 2)
        elif changed == 'layout': fields['layout'] = ('canonical_v1', 2816)
        elif changed == 'flags': fields['flags'] = (True, True)
        elif changed == 'stream':
            monkeypatch.setattr(torch, 'npu', SimpleNamespace(current_stream=lambda: object()))
        second, second_lease = mixed.plan_mixed_decode_for_forward(
            following, probe.TOKENS, layer_name='full1', **fields)
        assert second_lease is not first_lease
        assert first == second
    assert counts['plan'] == 2
