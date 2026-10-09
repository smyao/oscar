"""Archive #27/#31-36/#77/#78/#86/#140: concrete host/runtime contracts.

No fake NPU results are reported by these tests. CPU tensors are used only
for shape/address metadata and independent native-lifetime verification.
"""
import importlib
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import types
from unittest.mock import patch

import pytest
import torch

from oscar_ascend.integration.metadata import GraphMetadataBindings, OscarMetadataError, from_common
from oscar_ascend.integration import runtime_api
from oscar_ascend.runtime import AscendRuntimeProvider, GraphWorkspace, WorkspaceGeometry, canonical_rotation_name


def _common():
    return SimpleNamespace(
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        seq_lens=torch.tensor([32768, 8], dtype=torch.int32),
        slot_mapping=torch.arange(8, dtype=torch.int64),
        block_table_tensor=torch.zeros((2, 2052), dtype=torch.int32),
        num_reqs=2, num_actual_tokens=8, num_input_tokens=8, max_query_len=4,
        max_seq_len=32768, causal=True)


def test_native_graph_metadata_mutates_values_at_the_same_captured_addresses():
    common = _common()
    bindings = GraphMetadataBindings()
    bindings.bind(from_common(common, capture_origin=True))
    common.seq_lens.copy_(torch.tensor([50000, 9], dtype=torch.int32))
    common.block_table_tensor[0, 0] = 91
    updated = bindings.bind(from_common(common))
    assert updated.seq_lens is common.seq_lens
    assert updated.block_tables is common.block_table_tensor
    assert updated.block_tables.shape[1] == 2052


def test_graph_metadata_cannot_silently_rebind_a_captured_device_pointer():
    common = _common()
    bindings = GraphMetadataBindings()
    bindings.bind(from_common(common, capture_origin=True))
    common.seq_lens = common.seq_lens.clone()
    with pytest.raises(OscarMetadataError, match="captured device addresses"):
        bindings.bind(from_common(common))


def test_workspace_byte_account_includes_all_live_buffers():
    geometry = WorkspaceGeometry(16384, 6, 1, 256, cube_cores=24)
    n, h, k, d, s = 16384, 6, 1, 256, 3
    elements = [
        (n * h * d, 4), (n * h * d, 2), (n * k * d, 2), (n * k * d, 2),
        (n * h * s * d, 4), (n * h * s, 4), (n * h * d, 4),
        (n * h, 4), (n * h, 4), (n * h, 4), (n * k, 4),
        (n * k * s * 16, 8), (n * k * s * 2, 4), (n, 8), (n, 8),
        (24 * (768 * d + 32768) * 4, 1)]
    assert geometry.total_bytes == sum(count * size for count, size in elements)
    # History capacity 262144 is absent from the scratch shape. Only current
    # query rows and fixed split count contribute to the partial-output arena.
    assert geometry.total_bytes < 600 * 1024**2


def test_cluster4_state_is_explicit_bounded_and_includes_counter_buffer():
    baseline = WorkspaceGeometry(16384, 6, 1, 256, cube_cores=20)
    candidate = replace(baseline, history_cluster_size=4)
    mixed = replace(baseline, history_cluster_size=16)
    assert baseline.history_cluster_size == 1
    assert baseline.cv_bytes == 20 * 917504
    assert candidate.cv_bytes == 20 * 1839104
    assert mixed.cv_bytes == 20 * 4997120
    assert candidate.total_bytes - baseline.total_bytes == 20 * (1839104 - 917504 + 64)
    assert mixed.total_bytes - baseline.total_bytes == 20 * (4997120 - 917504 + 64)
    assert replace(candidate, tokens=128).cv_bytes == candidate.cv_bytes
    assert replace(mixed, tokens=128).cv_bytes == mixed.cv_bytes
    with pytest.raises(ValueError, match="cluster size"):
        replace(baseline, history_cluster_size=2)


@pytest.mark.parametrize("value", [1, "true", None])
def test_candidate_route_requires_explicit_boolean(value):
    with pytest.raises(runtime_api.OscarReadinessError, match="explicit boolean"):
        AscendRuntimeProvider({"experimental_history_reuse": value})
    assert AscendRuntimeProvider({}).config.get("experimental_history_reuse", False) is False


def test_mixed_cv_requires_explicit_fast_history_and_reserves_c16(monkeypatch):
    from oscar_ascend import runtime
    for bad in (1, "true", None):
        with pytest.raises(runtime_api.OscarReadinessError, match="explicit boolean"):
            AscendRuntimeProvider({"experimental_mixed_cv": bad})
    with pytest.raises(runtime_api.OscarReadinessError, match="requires explicit"):
        AscendRuntimeProvider({"experimental_mixed_cv": True,
                               "experimental_history_reuse": True})
    provider = AscendRuntimeProvider({"experimental_mixed_cv": True,
                                     "experimental_fast_unpack": True,
                                     "experimental_history_reuse": True,
                                     "max_num_batched_tokens": 16384,
                                     "compilation_config": {"cudagraph_capture_sizes": [128]}})
    monkeypatch.setattr(provider, "_device_cube_cores", lambda _device: 20)
    monkeypatch.setattr(provider, "_prepare_rotations", lambda *_: None)
    class FakeWorkspace:
        def __init__(self, geometry, device):
            self.geometry, self.device = geometry, device
    monkeypatch.setattr(runtime, "GraphWorkspace", FakeWorkspace)
    workspace = provider.ensure_workspace(6, 1, 256, "npu:0")
    assert workspace.geometry.history_cluster_size == 16
    assert workspace.geometry.cv_bytes == 20 * 4997120
    assert provider.ensure_workspace(6, 1, 256, "npu:0") is workspace


def test_mixed_cv_selector_uses_known_host_query_bound_without_padded_false_eligibility():
    from oscar_ascend.ops.cv_dispatch import (
        FAST_BALANCED_CV_OP, FAST_CLUSTER16_CV_OP, FAST_CLUSTER4_CV_OP, FAST_CV_OP,
        FAST_Q1_CV_OP, select_cv_op)
    def choose(tokens, max_query_len, *, draft=False):
        return select_cv_op(16, 6, 1, tokens, max_query_len,
                            q1_draft=draft, fast_unpack=True, mixed_cv=True)
    assert choose(16384, 1, draft=True) == FAST_Q1_CV_OP
    assert choose(16384, 1) == FAST_BALANCED_CV_OP
    assert choose(128, 4) == FAST_CV_OP
    assert choose(8, 4) == FAST_CV_OP
    assert choose(16384, 4) == FAST_BALANCED_CV_OP
    assert choose(390, 385) == FAST_CLUSTER4_CV_OP
    assert choose(8191, 8191) == FAST_CLUSTER4_CV_OP
    assert choose(16384, None) == FAST_CLUSTER4_CV_OP
    assert choose(8192, 8192) == FAST_CLUSTER16_CV_OP
    assert choose(16384, 16260) == FAST_CLUSTER16_CV_OP
    with pytest.raises(ValueError, match="mixed CV requires"):
        select_cv_op(4, 6, 1, 16384, 16260, fast_unpack=True, mixed_cv=True)
    with pytest.raises(ValueError, match="C16 workspace requires"):
        select_cv_op(16, 6, 1, 16384, 16260, fast_unpack=True)


def test_striped_dispatch_keeps_every_history_reader_in_one_format():
    from oscar_ascend.ops.cv_dispatch import (STRIPED_CV_OPS, STRIPED_CV_OP,
        STRIPED_Q1_CV_OP, STRIPED_CLUSTER4_CV_OP, STRIPED_BALANCED_CV_OP,
        STRIPED_CLUSTER16_CV_OP, STRIPED_DECODE_CV_OP, select_cv_op)
    flags = dict(fast_unpack=True, mixed_cv=True, striped_cache=True)
    cases = [
        (128, 4, False, STRIPED_DECODE_CV_OP),
        (16384, 1, True, STRIPED_Q1_CV_OP),
        (16384, 4, False, STRIPED_DECODE_CV_OP),
        (128, 5, False, STRIPED_CV_OP),
        (16384, 5, False, STRIPED_BALANCED_CV_OP),
        (512, 385, False, STRIPED_CLUSTER4_CV_OP),
        (16384, 16260, False, STRIPED_CLUSTER16_CV_OP),
        (16384, None, False, STRIPED_CLUSTER4_CV_OP),
    ]
    for tokens, max_q, draft, expected in cases:
        chosen = select_cv_op(16, 6, 1, tokens, max_q, q1_draft=draft, **flags)
        assert chosen == expected and chosen in STRIPED_CV_OPS
    # GQA16 cannot use the M32 decode kernel. It still uses a striped reader.
    assert select_cv_op(16, 16, 1, 128, 4, **flags) == STRIPED_CV_OP
    assert select_cv_op(16, 16, 1, 16384, 1, q1_draft=True,
                        **flags) == STRIPED_BALANCED_CV_OP
    assert select_cv_op(16, 16, 1, 128, 1, q1_draft=True,
                        **flags) == STRIPED_CV_OP
    with pytest.raises(ValueError, match="striped cache requires"):
        select_cv_op(4, 6, 1, 128, 4, striped_cache=True)


def test_striped_runtime_requires_d256_and_all_signed_reader_writer_symbols(monkeypatch):
    from oscar_ascend.ops import loader
    from oscar_ascend.ops.cv_dispatch import STRIPED_CV_OPS
    config = {"devices": [4, 5, 6, 7], "experimental_history_reuse": True,
              "experimental_fast_unpack": True, "experimental_mixed_cv": True,
              "experimental_striped_cache": True,
              "max_num_batched_tokens": 16384,
              "compilation_config": {"cudagraph_capture_sizes": [128]}}
    for bad in (None, "true", 1):
        with pytest.raises(runtime_api.OscarReadinessError, match="explicit boolean"):
            AscendRuntimeProvider({**config, "experimental_striped_cache": bad})
    with pytest.raises(runtime_api.OscarReadinessError, match="striped cache requires"):
        AscendRuntimeProvider({**config, "experimental_mixed_cv": False})
    provider = AscendRuntimeProvider(config)
    with pytest.raises(runtime_api.OscarReadinessError, match="D256"):
        provider.ensure_workspace(6, 1, 64, "npu:0")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4,5,6,7")
    monkeypatch.setattr(loader, "require_production_ops", lambda: None)
    seen = []
    def require(required, *_args):
        seen.append(set(required))
        if set(required) == set(STRIPED_CV_OPS) | {"rotate_clip_store_striped_out"}:
            raise RuntimeError("striped reader/writer symbols missing")
    monkeypatch.setattr(loader, "require_capabilities", require)
    with pytest.raises(RuntimeError, match="symbols missing"):
        provider.assert_ready()
    assert set(STRIPED_CV_OPS) | {"rotate_clip_store_striped_out"} in seen
    assert provider._ready is False
    provider.config["experimental_striped_cache"] = False
    with pytest.raises(runtime_api.OscarReadinessError, match="format changed"):
        provider.assert_ready()


def test_mixed_cv_readiness_requires_both_new_signed_ops(monkeypatch):
    from oscar_ascend.ops import loader
    from oscar_ascend.ops.cv_dispatch import MIXED_CV_OPS
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4,5,6,7")
    monkeypatch.setattr(loader, "require_production_ops", lambda: None)
    seen = []
    def require(required, *_args):
        seen.append(set(required))
        if set(required) == set(MIXED_CV_OPS):
            raise RuntimeError("signed balanced/C16 symbols missing")
    monkeypatch.setattr(loader, "require_capabilities", require)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_available=lambda: True),
                        raising=False)
    provider = AscendRuntimeProvider({"devices": [4, 5, 6, 7],
                                     "experimental_history_reuse": True,
                                     "experimental_fast_unpack": True,
                                     "experimental_mixed_cv": True})
    with pytest.raises(RuntimeError, match="symbols missing"):
        provider.assert_ready()
    assert set(MIXED_CV_OPS) in seen
    assert provider._ready is False


@pytest.mark.parametrize("tokens,expected", [(1, 20), (4, 20), (8, 20), (16, 20),
                                            (64, 5), (128, 3), (512, 1), (16384, 1)])
def test_shape_splits_use_measured_cube_parallelism_without_more_workspace(tokens, expected):
    geometry = WorkspaceGeometry(16384, 6, 1, 256, cube_cores=20)
    budget_before = geometry.total_bytes
    assert geometry.splits_for_tokens(tokens) == expected
    assert geometry.splits_for_tokens(tokens) == expected  # unchanged capture shape
    assert tokens * expected <= geometry.tokens * geometry.splits
    assert geometry.total_bytes == budget_before


def test_split_selection_bounds_every_shape_and_multiple_kv_heads():
    geometry = WorkspaceGeometry(16384, 12, 2, 256, cube_cores=20)
    assert geometry.splits_for_tokens(4) == 10  # two KV groups share 20 Cubes
    for tokens in range(1, geometry.tokens + 1):
        splits = geometry.splits_for_tokens(tokens)
        assert 1 <= splits <= 32
        assert tokens * splits <= geometry.tokens * geometry.splits
        assert tokens * geometry.kv_heads * 3 * splits <= geometry.task_count
    assert WorkspaceGeometry(16, 6, 1, 64, cube_cores=20).splits_for_tokens(4) == 4
    assert WorkspaceGeometry(16, 6, 1, 64, splits=2, cube_cores=20).splits_for_tokens(4) == 8
    with pytest.raises(ValueError, match="active tokens"):
        geometry.splits_for_tokens(0)
    with pytest.raises(ValueError, match="active tokens"):
        geometry.splits_for_tokens(geometry.tokens + 1)


def test_adaptive_partial_views_share_fixed_storage_and_preserve_unused_capacity():
    workspace = object.__new__(GraphWorkspace)
    g = WorkspaceGeometry(256, 6, 1, 64, cube_cores=20)
    workspace.geometry = g
    workspace.partial = torch.full((g.tokens, g.query_heads, 3, g.head_dim), 91.0)
    workspace.partial_lse = torch.full((g.tokens, g.query_heads, 3), 91.0)
    n, splits = 4, g.splits_for_tokens(4)
    partial, lse = workspace.partial_views(n, splits)
    assert partial.shape == (4, 6, 60, 64) and lse.shape == (4, 6, 60)
    assert partial.is_contiguous() and lse.is_contiguous()
    assert partial.data_ptr() == workspace.partial.data_ptr()
    assert lse.data_ptr() == workspace.partial_lse.data_ptr()
    partial.fill_(7)
    lse.fill_(8)
    assert bool(torch.all(workspace.partial.view(-1)[partial.numel():] == 91))
    assert bool(torch.all(workspace.partial_lse.view(-1)[lse.numel():] == 91))
    original_signature = (partial.data_ptr(), partial.shape, partial.stride(), lse.data_ptr(), lse.shape)
    workspace.partial_views(128, g.splits_for_tokens(128))[0].zero_()
    replay_partial, replay_lse = workspace.partial_views(n, splits)
    assert (replay_partial.data_ptr(), replay_partial.shape, replay_partial.stride(),
            replay_lse.data_ptr(), replay_lse.shape) == original_signature
    with pytest.raises(runtime_api.OscarReadinessError, match="preallocated workspace"):
        workspace.partial_views(128, 3)


def test_production_workspace_rejects_cpu():
    with pytest.raises(runtime_api.OscarReadinessError, match="requires NPU"):
        GraphWorkspace(WorkspaceGeometry(8, 4, 1, 64), "cpu")


def test_current_task_devices_are_required_before_any_operator_import():
    provider = AscendRuntimeProvider({"devices": None})
    with pytest.raises(runtime_api.OscarReadinessError, match="current-task devices"):
        provider.assert_ready()


def test_multimodal_and_draft_layer_names_follow_native_qwen_prefixes():
    assert canonical_rotation_name("language_model.model.layers.3.self_attn.attn") == "model.layers.3.self_attn.attn"
    assert canonical_rotation_name("mtp.layers.0.self_attn.attn") is None
    with pytest.raises(runtime_api.OscarReadinessError, match="unrecognized"):
        canonical_rotation_name("layers.3.attn")


def test_default_factory_is_lazy_and_does_not_override_explicit_provider():
    runtime_api.clear_runtime()
    sentinel = object()
    factory = lambda: sentinel
    runtime_api.install_runtime_factory(factory)
    runtime_api.ensure_default_runtime_factory()
    assert runtime_api._factory is factory
    runtime_api.clear_runtime()
    source = """
import sys,json
from oscar_ascend.integration.runtime_api import ensure_default_runtime_factory
ensure_default_runtime_factory()
print(json.dumps([name for name in sys.modules if name.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend')]))
"""
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == []


def test_online_forward_contains_no_host_request_loop_or_tensor_readback():
    import ast
    path = Path(__file__).resolve().parents[1] / "oscar_ascend/integration/impl.py"
    source = ast.parse(path.read_text())
    forward = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name == "forward")
    assert not any(isinstance(n, (ast.For, ast.While, ast.ListComp, ast.DictComp)) for n in ast.walk(forward))
    for node in ast.walk(forward):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"item", "tolist", "numpy", "cpu", "synchronize"}


@pytest.mark.parametrize("capacity,cube_cores,expected_splits,n", [(8, 1, 1, 8), (64, 20, 8, 8), (256, 20, 1, 256), (8192, 20, 1, 8192)])
@pytest.mark.parametrize("cluster_size,fast_unpack,mixed_cv,striped_cache", [
    (1, False, False, False), (4, False, False, False),
    (4, True, False, False), (16, True, True, False),
    (16, True, True, True)])
@pytest.mark.parametrize("later_draft", [False, True])
def test_forward_dispatches_native_draft_strides_and_int32_slots_in_order(capacity, cube_cores, expected_splits, n, cluster_size, later_draft, fast_unpack, mixed_cv, striped_cache):
    """Host ABI exercise only: fake ops establish ordering, never accuracy."""
    if n == 8192 and (cluster_size != 16 or later_draft):
        pytest.skip("large FULL host route is needed once for C16")
    # patch.dict restores sys.modules after the fake impl import. Preload the
    # real current_attention module first so the package attribute and the
    # restored absolute-import entry remain the same object for later tests.
    canonical_current = importlib.import_module("oscar_ascend.integration.current_attention")
    importlib.import_module("oscar_ascend.integration").current_attention = canonical_current
    backend = types.ModuleType("vllm.v1.attention.backend")
    backend.AttentionImpl = type("AttentionImpl", (), {})
    path = Path(__file__).resolve().parents[1] / "oscar_ascend/integration/impl.py"
    with patch.dict(sys.modules, {backend.__name__: backend}):
        spec = importlib.util.spec_from_file_location("oscar_ascend.integration._dispatch_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    h, hk, d, parts = 6 if striped_cache else 4, 1, 256 if striped_cache else 64, 3
    w = object.__new__(GraphWorkspace)
    w.geometry = WorkspaceGeometry(capacity, h, hk, d, cube_cores=cube_cores,
                                   history_cluster_size=cluster_size)
    w.cluster_stats = torch.empty((cube_cores, 8), dtype=torch.int64) if cluster_size in (4, 16) else None
    w.query_input = torch.empty(capacity, h, d, dtype=torch.bfloat16)
    w.key_input = torch.empty(capacity, hk, d, dtype=torch.bfloat16)
    w.value_input = torch.empty_like(w.key_input)
    w.query_rot = torch.empty(capacity, h, d)
    w.partial = torch.empty(capacity, h, parts, d)
    w.partial_lse = torch.empty(capacity, h, parts)
    w.output = torch.empty(capacity, h, d)
    w.lse = torch.empty(capacity, h)
    w.rotate_status = torch.empty(capacity, h, dtype=torch.int32)
    w.merge_status = torch.empty_like(w.rotate_status)
    w.store_status = torch.empty(capacity, hk, dtype=torch.int32)
    w.tasks = torch.empty(capacity * hk * parts, 16, dtype=torch.int64)
    w.attention_status = torch.empty(capacity * hk * parts, 2, dtype=torch.int32)
    w.positions = torch.empty(capacity, dtype=torch.int64)
    w.slots = torch.empty(capacity, dtype=torch.int64)
    w.cv = torch.empty(32, dtype=torch.uint8)
    calls = []

    class Ops:
        def prepare_attention_tasks_out(self, starts, lengths, slots, tasks, positions, *geometry):
            calls.append("prepare")
            assert slots.dtype == torch.int64 and slots.numel() == n
            assert geometry[4] == expected_splits
            assert tasks.shape == (n * hk * 3 * expected_splits, 16)
            positions.copy_(torch.arange(4, n + 4))
            tasks.zero_()

        def rotate_out(self, q, rotation, out, status, hadamard, slots):
            calls.append("rotate")
            assert q.is_contiguous()
            assert slots.dtype == torch.int64 and slots.shape == (n,)
            assert slots.data_ptr() == w.slots.data_ptr()
            out.copy_(q)
            status.zero_()

        def attention_cv_out(self, *args):
            calls.append("cv")
            assert args[3].is_contiguous()
            assert args[11].shape == (n, h, 3 * expected_splits, d)
            assert args[22] == expected_splits
            args[11].fill_(7)
            args[12].zero_()
            args[13].zero_()

        def attention_cv_cluster4_out(self, *args):
            assert cluster_size in (4, 16)
            assert args[15] is w.cluster_stats
            self.attention_cv_out(*(args[:15] + args[16:]))
            calls[-1] = "cluster4"

        def attention_cv_q1_out(self, *args):
            assert cluster_size in (4, 16) and later_draft
            self.attention_cv_out(*args)  # Identical fe0 ABI; no stats argument.
            calls[-1] = "q1"

        def attention_cv_fast_out(self, *args):
            assert fast_unpack
            self.attention_cv_out(*args)
            calls[-1] = "fast_cv"

        def attention_cv_fast_q1_out(self, *args):
            assert fast_unpack
            self.attention_cv_q1_out(*args)
            calls[-1] = "fast_q1"

        def attention_cv_fast_cluster4_out(self, *args):
            assert fast_unpack
            self.attention_cv_cluster4_out(*args)
            calls[-1] = "fast_cluster4"

        def attention_cv_fast_balanced_out(self, *args):
            assert fast_unpack and mixed_cv and cluster_size == 16
            self.attention_cv_out(*args)
            calls[-1] = "fast_balanced"

        def attention_cv_fast_cluster16_out(self, *args):
            assert fast_unpack and mixed_cv and cluster_size == 16
            assert args[15] is w.cluster_stats
            self.attention_cv_cluster4_out(*args)
            calls[-1] = "fast_cluster16"

        def attention_cv_striped_out(self, *args):
            assert striped_cache
            self.attention_cv_out(*args)
            calls[-1] = "striped_base"

        def attention_cv_striped_q1_out(self, *args):
            assert striped_cache and later_draft
            self.attention_cv_out(*args)
            calls[-1] = "striped_q1"

        def attention_cv_striped_decode_out(self, *args):
            assert striped_cache and not later_draft
            self.attention_cv_out(*args)
            calls[-1] = "striped_decode"

        def attention_cv_striped_cluster4_out(self, *args):
            assert striped_cache and args[15] is w.cluster_stats
            self.attention_cv_cluster4_out(*args)
            calls[-1] = "striped_cluster4"

        def attention_cv_striped_cluster16_out(self, *args):
            assert striped_cache and args[15] is w.cluster_stats
            self.attention_cv_cluster4_out(*args)
            calls[-1] = "striped_cluster16"

        def attention_cv_striped_balanced_out(self, *args):
            assert striped_cache
            self.attention_cv_out(*args)
            calls[-1] = "striped_balanced"

        def merge_lse_out(self, partial, lse, output, output_lse, status):
            calls.append("merge")
            assert partial.shape == (n * h, 3 * expected_splits, d)
            output.copy_(partial[:, 0])
            output_lse.zero_()
            status.zero_()

        def rotate_clip_store_out(self, *args):
            calls.append("store")
            assert torch.equal(args[5], torch.arange(4, n + 4))
            assert args[0].is_contiguous() and args[1].is_contiguous()
            args[10].zero_()

        def rotate_clip_store_striped_out(self, *args):
            assert striped_cache
            self.rotate_clip_store_out(*args)
            calls[-1] = "striped_store"

        def status_guard(self, *statuses):
            calls.append("guard")
            assert all(bool(torch.all(status == 0)) for status in statuses)

    packed = torch.empty(128, dtype=torch.uint8)
    state = SimpleNamespace(
        packed=packed, raw=packed, workspace=w,
        rotation_k_transpose=torch.eye(d), rotation_v_transpose=torch.eye(d),
        rotation_v=torch.eye(d), hadamard=False, num_blocks=1,
        window_key=torch.empty(1), window_value=torch.empty(1), window_tags=torch.empty(1),
        cache_format="striped_v1" if striped_cache else "canonical_v1",
        spec=SimpleNamespace(block_size=2304, conv_bytes=15360, ssm_bytes=393216),
        snapshots=SimpleNamespace(sink_tokens=64, recent_tokens=256, ring_tokens=259, speculative_tokens=3))
    impl = object.__new__(module.OscarAttentionImpl)
    impl.num_heads, impl.num_kv_heads, impl.head_size, impl.scale = h, hk, d, d**-0.5
    impl.provider = SimpleNamespace(layer_state=lambda _name: state, ops=Ops(),
                                    config={"experimental_fast_unpack": fast_unpack,
                                            "experimental_mixed_cv": mixed_cv,
                                            "experimental_striped_cache": striped_cache})
    q = torch.randn(n, h * d, dtype=torch.bfloat16)
    k = torch.randn(n, hk * d, dtype=torch.bfloat16)
    value = torch.randn(n, hk * d * 3, dtype=torch.bfloat16)[:, d:2*d]
    assert not value.is_contiguous()
    common = _common()
    common.query_start_loc = torch.tensor([0, n // 2, n], dtype=torch.int32)
    common.num_actual_tokens = common.num_input_tokens = n
    common.max_query_len = n // 2
    if n == 8192:
        common.query_start_loc = torch.tensor([0, n], dtype=torch.int32)
        common.num_reqs = 1
        common.max_query_len = n
    elif mixed_cv and n == 256 and not later_draft:
        if striped_cache:
            common.query_start_loc = torch.tensor([0, 128, 256], dtype=torch.int32)
            common.seq_lens = torch.tensor([32768, 128], dtype=torch.int32)
            common.num_actual_tokens = 256
            common.max_query_len = 128
        else:
            common.query_start_loc = torch.tensor([0, 4, 8], dtype=torch.int32)
            common.num_actual_tokens = 8
            common.max_query_len = 4
    if later_draft:
        common.max_query_len = 1
        common.num_actual_tokens = 2
        common.query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    common.slot_mapping = torch.arange(n + 4, dtype=torch.int32)
    output = torch.empty_like(q)
    result = impl.forward(SimpleNamespace(layer_name="mtp.layers.0.self_attn.attn"),
                          q, k, value, packed, replace(from_common(common), is_draft=True,
                          draft_index=1 if later_draft else 0), output)
    assert result is output and bool(torch.all(result == 7))
    # #150: short per-request query lengths keep the original OSCAR op even
    # with candidate workspace; long requests keep the C4 ABI/extra buffer.
    if striped_cache:
        expected_op = ("striped_q1" if later_draft else
                       "striped_cluster16" if n == 8192 else
                       "striped_cluster4" if n == 256 else "striped_decode")
    elif mixed_cv:
        expected_op = ("fast_q1" if later_draft else
                       "fast_cluster16" if n == 8192 else
                       "fast_balanced" if n == 256 else "fast_cv")
    else:
        expected_op = ("q1" if cluster_size == 4 and later_draft else
                       "cluster4" if cluster_size == 4 and n == 256 else "cv")
        if fast_unpack:
            expected_op = "fast_" + expected_op
    assert calls == ["prepare", "rotate", expected_op, "merge",
                     "striped_store" if striped_cache else "store", "guard"]
