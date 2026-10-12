"""Archive #34/#36/#127-129/#140/#142/#145/#155: capacity and ownership gates.

Meta allocations validate the complete production allocation ledger without
allocating model-sized CPU tensors. Small host mocks validate dispatch ordering
and fail-closed metadata checks only; these are not NPU or precision evidence.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from oscar_ascend import runtime
from oscar_ascend.integration import current_only_dispatch as current
from oscar_ascend.integration.current_only_plan import plan_current_suffix
from oscar_ascend.integration.metadata import OscarMetadata
from oscar_ascend.integration.runtime_api import OscarReadinessError
from oscar_ascend.runtime import GraphWorkspace, WorkspaceGeometry


def _workspace(monkeypatch, geometry, allocation_device="meta"):
    """Exercise the constructor with only the allocator/device seam replaced."""
    device, empty = torch.device, torch.empty
    with monkeypatch.context() as patch:
        patch.setattr(torch, "device", lambda value: (
            SimpleNamespace(type="npu") if value == "npu:0" else device(value)))
        patch.setattr(torch, "empty", lambda *args, **kwargs: empty(
            *args, **{**kwargs, "device": allocation_device}))
        return GraphWorkspace(geometry, "npu:0")


@pytest.mark.parametrize("cluster", [1, 4, 16])
@pytest.mark.parametrize("splits", [1, 3, 32])
def test_default_allocation_is_byte_identical_to_original_formula(monkeypatch, cluster, splits):
    n, h, k, d = 32768, 6, 1, 256
    g = WorkspaceGeometry(n, h, k, d, splits, 20, cluster)
    p = 3 * splits
    # Original pre-change allocation formula, independently retained here.
    expected = (n*h*d*4*2 + n*(h+2*k)*d*2 + n*h*p*d*4 + n*h*p*4
                + n*h*8 + n*k*p*(128+8) + n*k*4 + n*h*4 + n*16
                + g.cv_bytes + (20*8*8 if cluster in (4, 16) else 0))
    w = _workspace(monkeypatch, g)
    actual = {name: 0 if getattr(w, name) is None else
              getattr(w, name).numel()*getattr(w, name).element_size()
              for name in g.buffer_bytes}
    assert actual == g.buffer_bytes
    assert sum(actual.values()) == g.total_bytes == expected
    assert g.history_tokens == n
    assert replace(g, history_token_capacity=n).buffer_bytes == actual


def test_large_current_only_keeps_every_shared_buffer_and_bounds_cv(monkeypatch):
    g = WorkspaceGeometry(66048, 6, 1, 256, cube_cores=20,
                          history_cluster_size=16, history_token_capacity=16896)
    w = _workspace(monkeypatch, g)
    expected_shapes = {
        "query_rot": (16896, 6, 256), "partial": (16896, 6, 3, 256),
        "partial_lse": (16896, 6, 3), "output": (16896, 6, 256),
        "query_input": (66048, 6, 256), "key_input": (66048, 1, 256),
        "value_input": (66048, 1, 256), "lse": (66048, 6),
        "merge_status": (66048, 6), "rotate_status": (66048, 6),
        "store_status": (66048, 1), "positions": (66048,), "slots": (66048,),
        "tasks": (198144, 16), "attention_status": (198144, 2),
        "cv": (99942400,), "cluster_stats": (20, 8),
    }
    assert {name: tuple(getattr(w, name).shape) for name in expected_shapes} == expected_shapes
    assert sum(getattr(w, name).numel()*getattr(w, name).element_size()
               for name in expected_shapes) == g.total_bytes
    old = WorkspaceGeometry(32768, 6, 1, 256, cube_cores=20, history_cluster_size=16)
    assert old.total_bytes == 1259537664
    assert g.total_bytes == 923761920
    assert g.total_bytes - old.total_bytes == -335775744
    w.validate_current_only(66048, 6, 1, 256)
    w.validate(16896, 6, 1, 256)
    for rows in (16897, 66048):
        with pytest.raises(OscarReadinessError, match="fixed workspace mismatch"):
            w.validate(rows, 6, 1, 256)
        with pytest.raises(OscarReadinessError, match="preallocated workspace"):
            w.partial_views(rows, 1)
    with pytest.raises(OscarReadinessError, match="fixed workspace mismatch"):
        w.validate_current_only(66049, 6, 1, 256)
    # The original padded shape remains valid for compact/mixed S selection;
    # only its smaller live CV rows may be viewed or launched.
    assert g.splits_for_tokens(66048) == 1
    assert g.splits_for_tokens(128) == 3
    assert w.partial_views(128, 3)[0].shape == (128, 6, 9, 256)
    with pytest.raises(OscarReadinessError, match="preallocated workspace"):
        w.partial_views(16896, 2)


@pytest.mark.parametrize("bad", [0, -1, 17, True, 1.5, "8"])
def test_history_capacity_is_explicit_and_cannot_exceed_input(bad):
    with pytest.raises(ValueError, match="history token capacity"):
        WorkspaceGeometry(16, 6, 1, 256, history_token_capacity=bad)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "8"])
def test_runtime_rejects_implicit_history_capacity_before_allocation(bad):
    with pytest.raises(OscarReadinessError, match="explicit positive integer"):
        runtime.AscendRuntimeProvider({"history_token_capacity": bad})


def test_split_task_capacity_covers_both_current_and_history_without_aliasing(monkeypatch):
    g = WorkspaceGeometry(16, 2, 1, 64, splits=3, history_token_capacity=8)
    w = _workspace(monkeypatch, g)
    assert w.tasks.shape == (8*3*3, 16)  # Larger than current-only's 16*3.
    assert w.attention_status.shape == (72, 2)
    partial, lse = w.partial_views(8, 3)
    assert partial.shape == (8, 2, 9, 64) and lse.shape == (8, 2, 9)
    with pytest.raises(OscarReadinessError, match="preallocated workspace"):
        w.partial_views(8, 4)


def test_provider_reserves_optional_capacity_in_reused_loading_workspace(monkeypatch):
    seen = []
    class MetaWorkspace:
        def __init__(self, geometry, device):
            seen.append((geometry, device))
            self.geometry = geometry
    monkeypatch.setattr(runtime, "GraphWorkspace", MetaWorkspace)
    provider = runtime.AscendRuntimeProvider({
        "max_num_batched_tokens": 66048, "history_token_capacity": 16896,
        "experimental_current_only": True,
        "compilation_config": {"cudagraph_capture_sizes": [128, 512]}})
    monkeypatch.setattr(provider, "_device_cube_cores", lambda _device: 20)
    monkeypatch.setattr(provider, "_prepare_rotations", lambda *_args: None)
    first = provider.ensure_workspace(6, 1, 256, "npu:0")
    assert provider.ensure_workspace(6, 1, 256, "npu:0") is first
    assert len(seen) == 1 and first.geometry.history_tokens == 16896
    # The new knob cannot add an unprofiled reservation after construction.
    provider.config["history_token_capacity"] = 8192
    with pytest.raises(OscarReadinessError, match="changed after runtime construction"):
        provider.ensure_workspace(6, 1, 256, "npu:0")
    assert len(seen) == 1 and first.geometry.history_tokens == 16896


@pytest.mark.parametrize("origin", ["capture_origin", "dummy_origin"])
def test_current_only_does_not_admit_large_dummy_or_capture(origin):
    metadata = SimpleNamespace(is_draft=False, draft_index=0,
        capture_origin=origin == "capture_origin", dummy_origin=origin == "dummy_origin")
    with pytest.raises(OscarReadinessError, match="real eager"):
        current.dispatch_current_suffix(None, None, None, None, None, None, metadata, None, None)


@pytest.mark.parametrize("history_capacity,bad_context", [(2, False), (1, False), (2, True)])
def test_proven_current_suffix_exceeds_history_but_prefix_and_context_still_guarded(
        monkeypatch, history_capacity, bad_context):
    n, h, hk, d, cut = 8, 2, 1, 64, 2
    w = _workspace(monkeypatch, WorkspaceGeometry(n, h, hk, d,
                   history_token_capacity=history_capacity), "cpu")
    order = []
    state = SimpleNamespace(workspace=w, cache_format="striped_v1", hadamard=True,
        snapshots=SimpleNamespace(sink_tokens=64, recent_tokens=256, ring_tokens=259),
        rotation_k_transpose=None, rotation_v_transpose=None, raw=None,
        window_key=None, window_value=None, window_tags=None, num_blocks=1,
        spec=SimpleNamespace(block_size=512, conv_bytes=0, ssm_bytes=136*512))
    class Ops:
        def prepare_attention_tasks_out(self, starts, lengths, slots, tasks, positions, *_attrs):
            order.append("prepare")
            assert tasks.shape == (18, 16) and positions.shape == (6,)
            tasks.zero_(); tasks[:, 1] = 1
            if bad_context: tasks[0, 8] = 1
            positions.copy_(torch.arange(6))
        def status_guard(self, *statuses):
            if any(bool(status.ne(0).any()) for status in statuses):
                raise RuntimeError("device context trap")
        def copy_validate_current_out(self, values, lse, out, out_lse, status):
            order.append("copy")
            out.copy_(values); out_lse.copy_(lse.squeeze(-1)); status.zero_()
        def rotate_clip_store_striped_out(self, k, v, rk, rv, slots, positions,
                                         raw, wk, wv, tags, status, *_attrs):
            order.append("store")
            assert k.shape == v.shape == (6, 1, 64)
            assert slots.tolist() == list(range(2, 8))
            status.zero_()
    ops = Ops()
    class Impl:
        num_heads=h; num_kv_heads=hk; head_size=d; scale=d**-.5
        provider = SimpleNamespace(ops=ops, config={}, layer_state=lambda _name: state)
        def forward(self, layer, q, k, v, cache, metadata, output):
            order.append("history")
            w.validate(q.shape[0], h, hk, d)
            assert metadata.current_only_plan is None and q.shape[0] == cut
            output.fill_(7)
    def native(q, k, v, cumulative, **_kwargs):
        order.append("native")
        assert cumulative == (6,)
        return torch.full_like(q, 3), torch.full((6, h, 1), 1.25)
    monkeypatch.setattr(current, "native_current_partial", native)
    monkeypatch.setattr(current, "guard_current_slots", lambda *_args: None)
    metadata = OscarMetadata(torch.tensor([0, 2, 8], dtype=torch.int32),
        torch.tensor([100, 6], dtype=torch.int32), torch.zeros((2, 2), dtype=torch.int32),
        torch.arange(n, dtype=torch.int32), 2, n, 6, 100, num_input_tokens=n,
        query_start_loc_cpu=torch.tensor([0, 2, 8]), current_cumulative=(2, 8))
    plan = plan_current_suffix((0, 2, 8), (100, 6), actual_tokens=n, padded_tokens=n)
    q = torch.zeros((n, h*d), dtype=torch.bfloat16)
    k = torch.zeros((n, hk*d), dtype=torch.bfloat16)
    out = torch.full_like(q, 99)
    call = lambda: current.dispatch_current_suffix(Impl(), SimpleNamespace(layer_name="full"),
        q, k, k, None, metadata, out, plan)
    if history_capacity < cut:
        with pytest.raises(OscarReadinessError, match="fixed workspace mismatch"): call()
        assert order == [] and bool(out.eq(99).all())
    elif bad_context:
        with pytest.raises(RuntimeError, match="device context trap"): call()
        assert order == ["history", "prepare"]
    else:
        assert call() is out
        assert order == ["history", "prepare", "native", "copy", "store"]
        assert bool(out[:2].eq(7).all()) and bool(out[2:].eq(3).all())
        assert bool(w.lse[:6].eq(1.25).all())
