"""Archive #34/#36/#37-49/#129/#142/#157: same-step task reuse boundaries.

CPU host execution establishes byte preservation, launch counts and lifetime
only. The operator doubles below do not claim NPU numerical or speed evidence.
"""

from collections import Counter
from contextlib import contextmanager, nullcontext
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from oscar_ascend.integration.current_attention import _empty_current_source_ranges
from oscar_ascend.integration.metadata import OscarMetadata
from oscar_ascend.integration.prepare_reuse import (
    can_reuse_prepare, invalidate_prepare_reuse, remember_prepared, reused_slots,
)
from oscar_ascend.runtime import GraphWorkspace, WorkspaceGeometry
from test_current_attention import AscendAttentionState


def _fixture(monkeypatch, enabled, *, context=32768, slots_dtype=torch.int32, dim=64, tokens=32,
             data_device='cpu'):
    backend = ModuleType("vllm.v1.attention.backend")
    backend.AttentionImpl = type("AttentionImpl", (), {})
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    forward_context = ModuleType("vllm.forward_context")
    invocation = SimpleNamespace(attn_metadata={})
    forward_context.get_forward_context = lambda: invocation
    monkeypatch.setitem(sys.modules, forward_context.__name__, forward_context)
    path = Path(__file__).resolve().parents[1] / "oscar_ascend/integration/impl.py"
    spec = importlib.util.spec_from_file_location("oscar_ascend.integration._reuse_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    n, h, hk, d = tokens, 6, 1, dim
    w = object.__new__(GraphWorkspace)
    w.geometry = WorkspaceGeometry(n, h, hk, d, cube_cores=1)
    for name, shape, dtype in (
        ("query_input", (n, h, d), torch.bfloat16),
        ("key_input", (n, hk, d), torch.bfloat16),
        ("value_input", (n, hk, d), torch.bfloat16),
        ("query_rot", (n, h, d), torch.float32),
        ("partial", (n, h, 3, d), torch.float32),
        ("partial_lse", (n, h, 3), torch.float32),
        ("output", (n, h, d), torch.float32),
        ("lse", (n, h), torch.float32),
        ("rotate_status", (n, h), torch.int32),
        ("merge_status", (n, h), torch.int32),
        ("store_status", (n, hk), torch.int32),
        ("tasks", (n * 3, 16), torch.int64),
        ("attention_status", (n * 3, 2), torch.int32),
        ("positions", (n,), torch.int64),
        ("slots", (n,), torch.int64),
        ("cv", (1,), torch.uint8),
    ):
        device = data_device if dtype in (torch.bfloat16, torch.float32) else 'cpu'
        setattr(w, name, torch.empty(shape, dtype=dtype, device=device))
    w.cluster_stats = None
    counts = Counter()
    seen = []
    stored = []

    class Ops:
        def prepare_attention_tasks_out(self, starts, lens, slots, tasks, positions, *attrs):
            counts["prepare"] += 1
            n = positions.numel()
            # Distinct source ranges and one retained malformed task exercise
            # the existing source2 rewrite; the CV double records every byte.
            tasks.copy_(torch.arange(tasks.numel()).view_as(tasks))
            view = tasks.view(n, hk, 3, 1, 16)
            view[..., 3] = lens[0] - n
            view[..., 4] = lens[0]
            view[..., 10] = 0
            view[-1, :, 2, :, 10] = 5
            positions.copy_(torch.arange(n) + lens[0] - n)

        def rotate_out(self, q, rotation, out, status, hadamard, slots):
            counts["rotate"] += 1
            out.copy_(q)
            status.zero_()

        def attention_cv_out(self, *args):
            counts["cv"] += 1
            seen.append((args[10].clone(), w.positions.clone()))
            args[11].copy_(args[0].float()[:, :, None, :].expand_as(args[11]))
            args[12].zero_()
            args[13].zero_()

        def merge_lse_out(self, partial, lse, output, output_lse, status):
            counts["merge"] += 1
            output.copy_(partial[:, 0])
            output_lse.zero_()
            status.zero_()

        def rotate_clip_store_out(self, *args):
            counts["store"] += 1
            stored.append((args[6].data_ptr(), args[4].clone(), args[5].clone()))
            args[10].zero_()

        def status_guard(self, *statuses):
            counts["guard"] += 1
            assert all(bool((status == 0).all()) for status in statuses)

    def suppress(tasks, tokens, kv_heads, splits):
        counts["suppress"] += 1
        _empty_current_source_ranges(tasks, tokens, kv_heads, splits)

    def native(q, k, v, cumulative, **kwargs):
        counts["current_fia"] += 1
        return q.clone(), torch.zeros(q.shape[0], h, 1)

    def write(partial, lse, current, current_lse, active, splits):
        partial[:active, :, 2 * splits].copy_(current)
        lse[:active, :, 2 * splits].copy_(current_lse.squeeze(-1))

    stream = object()
    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_stream=lambda: stream), raising=False)
    monkeypatch.setattr(module, "suppress_current_source_tasks", suppress)
    monkeypatch.setattr(module, "native_current_partial", native)
    monkeypatch.setattr(module, "write_current_partial", write)
    monkeypatch.setattr(module, "guard_current_slots", lambda *args: counts.update(["slot_guard"]))
    monkeypatch.setattr(module, "phase", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(module, "run_status_guard", lambda ops, site, segments, **kwargs:
                        ops.status_guard(*(tensor for _, tensor in segments)))
    states = {}
    for i in range(16):
        packed = torch.empty(128, dtype=torch.uint8)
        states[f"full{i}"] = SimpleNamespace(
            packed=packed, raw=packed, workspace=w, cache_format="canonical_v1",
            rotation_k_transpose=torch.eye(d), rotation_v_transpose=torch.eye(d),
            rotation_v=torch.eye(d), hadamard=False, num_blocks=1,
            window_key=torch.empty(1), window_value=torch.empty(1), window_tags=torch.empty(1),
            spec=SimpleNamespace(block_size=2304, conv_bytes=15360, ssm_bytes=393216),
            snapshots=SimpleNamespace(sink_tokens=64, recent_tokens=256,
                                      ring_tokens=259, speculative_tokens=3))
    impl = object.__new__(module.OscarAttentionImpl)
    impl.num_heads, impl.num_kv_heads, impl.head_size, impl.scale = h, hk, d, d ** -.5
    impl.provider = SimpleNamespace(layer_state=states.__getitem__, ops=Ops(),
                                    config={"experimental_prepare_reuse": enabled})
    metadata = OscarMetadata(
        torch.tensor([0, n], dtype=torch.int32),
        torch.tensor([context + n], dtype=torch.int32),
        torch.arange((context + n + 127) // 128, dtype=torch.int32)[None, :],
        torch.arange(context, context + n, dtype=slots_dtype),
        1, n, n, context + n, num_input_tokens=n,
        current_cumulative=(n,), attn_state=AscendAttentionState.ChunkedPrefill)

    def run(meta=metadata, names=None):
        invocation.attn_metadata={name:meta for name in states}
        outputs = []
        for i, name in enumerate(states if names is None else names):
            q = torch.full((n, h * d), i + 1, dtype=torch.bfloat16, device=data_device)
            kv = torch.zeros(n, hk * d, dtype=torch.bfloat16, device=data_device)
            output = torch.empty_like(q)
            impl.forward(SimpleNamespace(layer_name=name), q, kv, kv,
                         states[name].packed, meta, output)
            outputs.append(output)
        return outputs

    return SimpleNamespace(impl=impl, workspace=w, metadata=metadata, counts=counts,
                           seen=seen, stored=stored, run=run, stream=stream, states=states,
                           module=module, invocation=invocation)


@pytest.mark.parametrize("context", [32768, 65536])
@pytest.mark.parametrize("slots_dtype", [torch.int32, torch.int64])
def test_all_layers_keep_bytes_and_work_while_prepare_runs_once(monkeypatch, context, slots_dtype):
    base = _fixture(monkeypatch, False, context=context, slots_dtype=slots_dtype)
    base_out = base.run()
    candidate = _fixture(monkeypatch, True, context=context, slots_dtype=slots_dtype)
    addresses = tuple(getattr(candidate.workspace, name).data_ptr()
                      for name in ("tasks", "positions", "slots"))
    candidate_out = candidate.run()
    assert base.counts["prepare"] == base.counts["suppress"] == 16
    assert candidate.counts["prepare"] == candidate.counts["suppress"] == 1
    for name in ("rotate", "cv", "merge", "store", "guard", "slot_guard", "current_fia"):
        assert base.counts[name] == candidate.counts[name] == 16
    for before, after, baseline_output, candidate_output in zip(
            base.seen, candidate.seen, base_out, candidate_out):
        assert torch.equal(before[0], after[0]) and torch.equal(before[1], after[1])
        assert torch.equal(baseline_output, candidate_output)
    assert len({entry[0] for entry in candidate.stored}) == 16
    assert all(torch.equal(entry[1], candidate.metadata.slot_mapping) for entry in candidate.stored)
    assert tuple(getattr(candidate.workspace, name).data_ptr()
                 for name in ("tasks", "positions", "slots")) == addresses
    # The existing source2 suppression keeps malformed task/error bytes.
    assert candidate.seen[-1][0].view(32, 1, 3, 1, 16)[-1, 0, 2, 0, 10] == 5


def test_repeated_layer_and_new_native_step_rebuild_even_at_same_addresses(monkeypatch):
    f = _fixture(monkeypatch, True)
    f.run()
    f.metadata.seq_lens.add_(128)
    f.run()  # Same object deliberately reused: repeated full0 invalidates.
    assert f.counts["prepare"] == 2
    assert f.seen[16][1][0] == 32768 + 128
    newer = replace(f.metadata)
    f.run(newer, names=["full1"])
    assert f.counts["prepare"] == 3


@pytest.mark.parametrize("field,value", [
    ("capture_origin", True), ("dummy_origin", True), ("is_draft", True),
    ("mixed_split_segment", "long_suffix"),
])
def test_excluded_execution_never_consumes_or_leaves_reuse(monkeypatch, field, value):
    f = _fixture(monkeypatch, True)
    f.run(names=["full0"])
    f.run(replace(f.metadata, **{field: value}), names=["full1"])
    assert f.counts["prepare"] == 2
    assert f.workspace._oscar_prepared_tasks is None


def test_geometry_stream_and_other_writers_expire_lease():
    w, meta, slots, stream, context = SimpleNamespace(), object(), object(), object(), object()
    shape = (32, 6, 1, 64, 256, 1)
    remember_prepared(w, meta, "full0", shape, stream, slots, context)
    assert reused_slots(w, meta, "full1", shape, stream, context) is slots
    assert reused_slots(w, meta, "full2", (*shape[:-1], 2), stream, context) is None
    remember_prepared(w, meta, "full0", shape, stream, slots, context)
    assert reused_slots(w, meta, "full1", shape, object(), context) is None
    remember_prepared(w, meta, "full0", shape, stream, slots, context)
    assert reused_slots(w, meta, "full1", shape, stream, object()) is None
    remember_prepared(w, meta, "full0", shape, stream, slots, context)
    invalidate_prepare_reuse(w)
    assert reused_slots(w, meta, "full1", shape, stream, context) is None


@pytest.mark.parametrize("route", ["current_only", "mixed"])
def test_alternate_dispatch_invalidates_before_and_after_writer(monkeypatch, route):
    f = _fixture(monkeypatch, True)
    f.run(names=["full0"])
    def dispatch(*args, **kwargs):
        assert f.workspace._oscar_prepared_tasks is None
        remember_prepared(f.workspace, f.metadata, "inner", (32,6,1,64,256,1), f.stream, object(), object())
        return args[7]
    if route == "current_only":
        from oscar_ascend.integration import current_only_dispatch
        f.impl.provider.config["experimental_current_only"] = True
        monkeypatch.setattr(current_only_dispatch, "dispatch_current_suffix", dispatch)
        meta = replace(f.metadata, current_only_plan=object())
    else:
        from oscar_ascend.integration import mixed_decode_split
        f.impl.provider.config["experimental_mixed_decode_split"] = True
        monkeypatch.setattr(mixed_decode_split, "plan_mixed_decode", lambda *args, **kwargs: object())
        monkeypatch.setattr(mixed_decode_split, "dispatch_mixed_decode", dispatch)
        meta = f.metadata
    f.run(meta, names=["full1"])
    assert f.workspace._oscar_prepared_tasks is None


def test_nonboolean_flag_rejected():
    with pytest.raises(ValueError, match="explicit boolean"):
        can_reuse_prepare(None, enabled=1, native_current=True)


def test_pinned_native_context_identity_separates_reentry_with_same_metadata(monkeypatch):
    from test_runtime_native import definitions
    native = definitions("references/vllm/vllm/forward_context.py", "native_reuse_context",
        {"create_forward_context", "override_forward_context", "get_forward_context"},
        {"ForwardContext": SimpleNamespace, "CUDAGraphMode": SimpleNamespace(NONE=object()),
         "contextmanager": contextmanager, "_forward_context": None}, monkeypatch)
    config = SimpleNamespace(compilation_config=SimpleNamespace(
        fast_moe_cold_start=False, static_forward_context={}))
    meta = object()
    first = native.create_forward_context(meta, config)
    second = native.create_forward_context(meta, config)
    assert first is not second and first.attn_metadata is second.attn_metadata is meta
    w, stream, slots, geometry = SimpleNamespace(), object(), object(), (32, 6, 1, 64, 256, 1)
    with native.override_forward_context(first):
        remember_prepared(w, meta, "full0", geometry, stream, slots, native.get_forward_context())
        with native.override_forward_context(second):
            # Inner invocation starts at a different layer: object identity,
            # not repeated-layer detection, must expire the outer tasks.
            assert reused_slots(w, meta, "full7", geometry, stream, native.get_forward_context()) is None
            remember_prepared(w, meta, "full7", geometry, stream, slots, native.get_forward_context())
        assert native.get_forward_context() is first
        assert reused_slots(w, meta, "full1", geometry, stream, native.get_forward_context()) is None


@pytest.mark.parametrize("has_nz", [False, True])
def test_actual_q4_forward_uses_required_hybrid_rotation_storage(monkeypatch, has_nz):
    from oscar_ascend.integration.runtime_api import OscarReadinessError
    f = _fixture(monkeypatch, False, dim=256)
    f.workspace.geometry = WorkspaceGeometry(32, 6, 1, 256, cube_cores=1,
                                             history_cluster_size=16)
    f.workspace.cluster_stats = torch.empty(1, 8, dtype=torch.int64)
    for state in f.states.values():
        state.cache_format = "striped_v1"
        if has_nz:
            state.rotation_v_nz = torch.empty(256, 256)
    f.impl.provider.config.update({"experimental_" + name: True for name in
        ("fast_unpack", "mixed_cv", "striped_cache", "decode_bundle", "mdl_decode",
         "hybrid_l1_decode", "c16_member_pipeline")})
    class Chosen(Exception):
        pass
    def hybrid(*args):
        assert args[4] is f.states["full0"].rotation_v_nz
        assert args[4] is not f.states["full0"].rotation_v
        raise Chosen()
    f.impl.provider.ops.attention_cv_hybrid_l1_decode_out = hybrid
    metadata = replace(f.metadata, attn_state=AscendAttentionState.DecodeOnly,
        max_query_len=4, num_reqs=8, query_start_loc=torch.arange(0, 33, 4, dtype=torch.int32),
        seq_lens=torch.full((8,), 32772, dtype=torch.int32),
        block_tables=f.metadata.block_tables.expand(8, -1).contiguous())
    if has_nz:
        with pytest.raises(Chosen):
            f.run(metadata, names=["full0"])
    else:
        with pytest.raises(OscarReadinessError, match="preallocated NZ V rotation"):
            f.run(metadata, names=["full0"])
