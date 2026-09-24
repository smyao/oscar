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


@pytest.mark.parametrize("capacity,cube_cores,expected_splits", [(8, 1, 1), (64, 20, 8)])
def test_forward_dispatches_native_draft_strides_and_int32_slots_in_order(capacity, cube_cores, expected_splits):
    """Host ABI exercise only: fake ops establish ordering, never accuracy."""
    backend = types.ModuleType("vllm.v1.attention.backend")
    backend.AttentionImpl = type("AttentionImpl", (), {})
    path = Path(__file__).resolve().parents[1] / "oscar_ascend/integration/impl.py"
    with patch.dict(sys.modules, {backend.__name__: backend}):
        spec = importlib.util.spec_from_file_location("oscar_ascend.integration._dispatch_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    n, h, hk, d, parts = 8, 4, 1, 64, 3
    w = object.__new__(GraphWorkspace)
    w.geometry = WorkspaceGeometry(capacity, h, hk, d, cube_cores=cube_cores)
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
            positions.copy_(torch.arange(4, 12))
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

        def merge_lse_out(self, partial, lse, output, output_lse, status):
            calls.append("merge")
            assert partial.shape == (n * h, 3 * expected_splits, d)
            output.copy_(partial[:, 0])
            output_lse.zero_()
            status.zero_()

        def rotate_clip_store_out(self, *args):
            calls.append("store")
            assert torch.equal(args[5], torch.arange(4, 12))
            assert args[0].is_contiguous() and args[1].is_contiguous()
            args[10].zero_()

        def status_guard(self, *statuses):
            calls.append("guard")
            assert all(bool(torch.all(status == 0)) for status in statuses)

    packed = torch.empty(128, dtype=torch.uint8)
    state = SimpleNamespace(
        packed=packed, raw=packed, workspace=w,
        rotation_k_transpose=torch.eye(d), rotation_v_transpose=torch.eye(d),
        rotation_v=torch.eye(d), hadamard=False, num_blocks=1,
        window_key=torch.empty(1), window_value=torch.empty(1), window_tags=torch.empty(1),
        spec=SimpleNamespace(block_size=2304, conv_bytes=15360, ssm_bytes=393216),
        snapshots=SimpleNamespace(sink_tokens=64, recent_tokens=256, ring_tokens=259, speculative_tokens=3))
    impl = object.__new__(module.OscarAttentionImpl)
    impl.num_heads, impl.num_kv_heads, impl.head_size, impl.scale = h, hk, d, d**-0.5
    impl.provider = SimpleNamespace(layer_state=lambda _name: state, ops=Ops(), config={})
    q = torch.randn(n, h * d, dtype=torch.bfloat16)
    k = torch.randn(n, hk * d, dtype=torch.bfloat16)
    value = torch.randn(n, hk * d * 3, dtype=torch.bfloat16)[:, d:2*d]
    assert not value.is_contiguous()
    common = _common()
    common.slot_mapping = torch.arange(n + 4, dtype=torch.int32)
    output = torch.empty_like(q)
    result = impl.forward(SimpleNamespace(layer_name="mtp.layers.0.self_attn.attn"),
                          q, k, value, packed, replace(from_common(common), is_draft=True), output)
    assert result is output and bool(torch.all(result == 7))
    assert calls == ["prepare", "rotate", "cv", "merge", "store", "guard"]
