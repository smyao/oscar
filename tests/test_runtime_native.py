"""Archive #27/#28/#34/#36/#37-49/#78: execute pinned native contracts.

Native class/method AST nodes are compiled unchanged from the read-only
reference tree. Dependencies outside each tested contract are supplied by
the harness; this is host compatibility evidence, not an NPU/vLLM launch.
"""
import ast
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Sequence, Mapping
import copy
from dataclasses import dataclass, fields, replace
from enum import Enum, IntEnum
import importlib.util
import math
import itertools
import logging
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace, MethodType
from typing import Any, ClassVar, Generic, TypeVar, overload
from unittest.mock import patch

import pytest
import torch

from oscar_ascend.runtime import AscendRuntimeProvider
from oscar_ascend.integration.runtime_api import OscarReadinessError

ROOT = Path(__file__).resolve().parents[1]


def definitions(path, name, names, namespace, monkeypatch, *, methods=None):
    tree = ast.parse((ROOT / path).read_text())
    chosen = [node for node in tree.body
              if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    for cls, method in methods or ():
        native_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls)
        chosen.append(next(node for node in native_class.body if isinstance(node, ast.FunctionDef) and node.name == method))
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    code = ast.fix_missing_locations(ast.Module(body=[future, *chosen], type_ignores=[]))
    module = ModuleType(name)
    module.__dict__.update(namespace)
    monkeypatch.setitem(sys.modules, name, module)
    exec(compile(code, str(ROOT / path), "exec"), module.__dict__)
    return module


def load_our_module(name, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name.replace(".", "/") + ".py"))
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def native_specs(monkeypatch):
    dtype_size = lambda dtype: torch.empty((), dtype=dtype).element_size()
    registry = definitions(
        "references/vllm/vllm/v1/kv_cache_spec_registry.py", "vllm.v1.kv_cache_spec_registry",
        {"KVCacheSpecMetadata", "KVCacheSpecRegistry"},
        {"dataclass": dataclass, "_REGISTRY_KVCACHESPEC_LIST": {}}, monkeypatch)
    interface = definitions(
        "references/vllm/vllm/v1/kv_cache_interface.py", "vllm.v1.kv_cache_interface",
        {"KVQuantMode", "get_kv_quant_mode", "KVCacheSpec", "AttentionSpec", "FullAttentionSpec", "MambaSpec",
         "MLAAttentionSpec", "SlidingWindowMLASpec", "HiddenStateCacheSpec", "SlidingWindowSpec",
         "ChunkedLocalAttentionSpec", "TQFullAttentionSpec",
         "UniformTypeKVCacheSpecs", "KVCacheTensor", "KVCacheGroupSpec", "KVCacheConfig"},
        {"dataclass": dataclass, "fields": fields, "replace": replace, "Enum": Enum, "IntEnum": IntEnum,
         "torch": torch, "copy": copy, "prod": math.prod, "Counter": Counter, "get_dtype_size": dtype_size,
         "MambaAttentionBackendEnum": SimpleNamespace(MAMBA2="mamba2"),
         "KVCacheSpecRegistry": registry.KVCacheSpecRegistry}, monkeypatch)
    utils = ModuleType("vllm.utils.torch_utils")
    utils.get_dtype_size = dtype_size
    managers = definitions(
        "references/vllm/vllm/v1/core/single_type_kv_cache_manager.py", "vllm.v1.core.single_type_kv_cache_manager",
        {"SingleTypeKVCacheManager", "FullAttentionManager", "MambaManager", "SlidingWindowManager"},
        {**interface.__dict__, "__name__": "vllm.v1.core.single_type_kv_cache_manager",
         "ABC": ABC, "abstractmethod": abstractmethod, "defaultdict": defaultdict,
         "cdiv": lambda x, y: (x + y - 1) // y, "itertools": itertools}, monkeypatch)
    ascend_mamba = definitions(
        "references/vllm-ascend/vllm_ascend/patch/platform/patch_mamba_manager.py", "native_mamba_manager_contract",
        {"AscendMambaManager"}, {"MambaManager": managers.MambaManager, "MambaSpec": interface.MambaSpec}, monkeypatch)
    managers.MambaManager = ascend_mamba.AscendMambaManager
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    monkeypatch.setitem(sys.modules, managers.__name__, managers)
    ours = load_our_module("oscar_ascend.integration.specs", monkeypatch)
    ours.register_oscar_spec()
    registry.KVCacheSpecRegistry.register(interface.MambaSpec, managers.MambaManager, interface.MambaSpec)
    return interface, ours, registry


def test_real_native_spec_conversion_and_registry_preserve_gdn(native_specs):
    interface, ours, registry = native_specs
    full = interface.FullAttentionSpec(block_size=768, num_kv_heads=1, head_size=256,
                                       dtype=torch.bfloat16, page_size_padded=801792)
    gdn = interface.MambaSpec(block_size=768, shapes=((3, 2560), (12, 128, 128)),
                              dtypes=(torch.bfloat16, torch.bfloat16), page_size_padded=801792,
                              num_speculative_blocks=3, mamba_cache_mode="align")
    converted = ours.transform_native_specs({"full": full, "gdn": gdn})
    assert converted["gdn"] is gdn
    spec = converted["full"]
    assert isinstance(spec, interface.FullAttentionSpec)
    assert (spec.block_size, spec.page_size_bytes, spec.real_page_size_bytes) == (2304, 801792, 313344)
    assert registry.KVCacheSpecRegistry.get_uniform_type_base_spec(spec) is ours.OscarFullAttentionSpec
    assert registry.KVCacheSpecRegistry.get_manager_class(spec).__name__ == "FullAttentionManager"
    # Native dataclass operations used by grouping/config transfer retain all
    # external layout fields; no local reimplementation of FullAttentionSpec.
    cloned = copy.deepcopy(spec)
    assert cloned == spec and cloned.layout == spec.layout
    assert ours.OscarFullAttentionSpec.merge([spec, cloned]) == spec


@pytest.fixture
def native_groups(native_specs, monkeypatch):
    interface, ours, registry = native_specs
    functions = {
        "resolve_kv_cache_block_sizes", "get_kv_cache_groups", "is_kv_cache_type_attention_free",
        "is_kv_cache_spec_uniform", "_get_kv_cache_groups_uniform_spec", "_get_kv_cache_groups_uniform_type",
        "group_and_unify_kv_cache_specs", "get_uniform_page_size", "is_kv_cache_page_size_uniform",
        "unify_kv_cache_spec_page_size", "_get_kv_cache_groups_uniform_page_size", "create_kv_cache_group_specs",
        "_pool_bytes_per_block", "get_num_blocks", "get_kv_cache_config_from_groups", "may_override_num_blocks",
        "KVCacheBlock", "FreeKVCacheBlockQueue", "BlockHashListWithBlockSize"}
    utils = definitions(
        "references/vllm/vllm/v1/core/kv_cache_utils.py", "native_grouping_contract", functions,
        {**interface.__dict__, "__name__": "native_grouping_contract", "math": math,
         "defaultdict": defaultdict, "Sequence": Sequence, "BlockHash": bytes, "overload": overload,
         "cdiv": lambda x, y: (x + y - 1) // y, "logger": logging.getLogger("native-group-test")}, monkeypatch)
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        cache_config=SimpleNamespace(block_size=768, enable_prefix_caching=True, hash_block_size=None,
                                     num_gpu_blocks_override=None, mamba_cache_mode="align"),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1, prefill_context_parallel_size=1),
        kv_transfer_config=None)
    full = interface.FullAttentionSpec(block_size=768, num_kv_heads=1, head_size=256,
                                       dtype=torch.bfloat16, page_size_padded=801792)
    gdn = interface.MambaSpec(block_size=768, shapes=((3, 2560), (12, 128, 128)),
                              dtypes=(torch.bfloat16, torch.bfloat16), page_size_padded=801792,
                              num_speculative_blocks=3, mamba_cache_mode="align")
    # Match the actual runner's FULL-before-Mamba insertion order. This is
    # essential: uniform detection dispatches merge() on the first spec.
    specs = {f"full.{i}": full for i in range(17)} | {f"gdn.{i}": gdn for i in range(48)}
    converted = ours.transform_native_specs(specs)
    groups = utils.get_kv_cache_groups(config, converted)
    return interface, ours, registry, utils, config, groups


def test_actual_native_grouping_lcm_hashing_and_shared_pool(native_groups):
    interface, ours, registry, utils, config, groups = native_groups
    assert sorted(len(group.layer_names) for group in groups) == [16, 16, 16, 17]
    assert [group.kv_cache_spec.block_size for group in groups] == [2304, 768, 768, 768]
    memory = 17 * 801792 * 100 + 17
    pool_config = utils.get_kv_cache_config_from_groups(config, groups, memory)
    assert pool_config.num_blocks == 100 and len(pool_config.kv_cache_tensors) == 17
    assert utils._pool_bytes_per_block(groups) == 17 * 801792
    assert sum(tensor.size for tensor in pool_config.kv_cache_tensors) == 17 * 801792 * 100
    assert utils.resolve_kv_cache_block_sizes(pool_config, config) == (2304, 768)
    hashes = utils.BlockHashListWithBlockSize([bytes([i]) for i in range(9)], 768, 2304)
    assert list(hashes) == [bytes([0, 1, 2]), bytes([3, 4, 5]), bytes([6, 7, 8])]


def test_real_qwen_shape_and_ascend_config_account_for_speculative_conv_width(native_specs, monkeypatch):
    interface, ours, registry = native_specs
    state = definitions(
        "references/vllm/vllm/model_executor/layers/mamba/mamba_utils.py", "native_mamba_shape_contract",
        {"MambaStateShapeCalculator"},
        {"divide": lambda a, b: a // b, "is_conv_state_dim_first": lambda: False}, monkeypatch)
    qwen = definitions(
        "references/vllm/vllm/model_executor/models/qwen3_5.py", "native_qwen_shape_contract",
        set(), {"MambaStateShapeCalculator": state.MambaStateShapeCalculator}, monkeypatch,
        methods=[("Qwen3_5ForConditionalGeneration", "get_mamba_state_shape_from_config")])
    model_cls = type("QwenConfiguredShape", (), {
        "get_mamba_state_shape_from_config": qwen.get_mamba_state_shape_from_config,
        "get_mamba_state_dtype_from_config": classmethod(lambda cls, cfg: (torch.bfloat16, torch.bfloat16))})
    base = definitions(
        "references/vllm/vllm/model_executor/models/config.py", "native_mamba_config_contract",
        {"VerifyAndUpdateConfig", "MambaModelConfig"},
        {"logger": logging.getLogger("native-config-test")}, monkeypatch)
    ascend = definitions(
        "references/vllm-ascend/vllm_ascend/patch/platform/patch_mamba_config.py", "native_hybrid_config_contract",
        {"_using_kv_store", "verify_and_update_config"},
        {"math": math, "MambaModelConfig": base.MambaModelConfig,
         "ModelRegistry": SimpleNamespace(resolve_model_cls=lambda *_args, **_kwargs: (model_cls, "fixture")),
         "logger": logging.getLogger("native-config-test"), "cdiv": lambda a, b: (a + b - 1) // b,
         "get_dtype_size": lambda dtype: torch.empty((), dtype=dtype).element_size()}, monkeypatch)
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False, enable_chunked_prefill=True),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16, architecture="Qwen3_5ForConditionalGeneration", use_mla=False,
            supports_mamba_prefix_caching=False, max_model_len=262144,
            get_num_kv_heads=lambda _parallel: 1, get_head_size=lambda: 256,
            hf_text_config=SimpleNamespace(linear_num_key_heads=16, linear_num_value_heads=48,
                                           linear_key_head_dim=128, linear_value_head_dim=128,
                                           linear_conv_kernel_dim=4)),
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=None, mamba_page_size_padded=None,
                                     mamba_block_size=None, enable_prefix_caching=True, mamba_cache_mode="none"),
        parallel_config=SimpleNamespace(tensor_parallel_size=4),
        speculative_config=SimpleNamespace(num_speculative_tokens=3, method="mtp"), kv_transfer_config=None)
    ascend.verify_and_update_config.__func__(None, config)
    shapes = model_cls.get_mamba_state_shape_from_config(config)
    assert shapes == ((6, 2560), (12, 128, 128))
    assert config.cache_config.block_size == config.cache_config.mamba_block_size == 768
    assert config.cache_config.mamba_cache_mode == "align"
    assert config.cache_config.mamba_page_size_padded == 817152
    gdn = interface.MambaSpec(block_size=768, shapes=shapes, dtypes=(torch.bfloat16, torch.bfloat16),
                              page_size_padded=config.cache_config.mamba_page_size_padded)
    full = interface.FullAttentionSpec(block_size=768, num_kv_heads=1, head_size=256, dtype=torch.bfloat16,
                                       page_size_padded=config.cache_config.mamba_page_size_padded)
    spec = ours.transform_native_specs({"full": full, "gdn": gdn})["full"]
    assert (spec.conv_bytes, spec.ssm_bytes, spec.page_size_bytes, spec.block_size) == (30720, 393216, 817152, 2304)
    assert spec.layout.native_padding_bytes == 393216


def test_actual_ascend_coordinator_initializes_native_managers_for_different_block_sizes(native_groups, monkeypatch):
    interface, ours, registry, utils, config, groups = native_groups
    managers = sys.modules["vllm.v1.core.single_type_kv_cache_manager"]
    pool_types = definitions(
        "references/vllm/vllm/v1/core/block_pool.py", "native_block_pool_contract",
        {"BlockHashToBlockMap", "BlockPool"},
        {"KVCacheBlock": utils.KVCacheBlock, "FreeKVCacheBlockQueue": utils.FreeKVCacheBlockQueue}, monkeypatch)
    ascend_interface = ModuleType("vllm_ascend.core.kv_cache_interface")
    ascend_interface.AscendMLAAttentionSpec = interface.MLAAttentionSpec
    monkeypatch.setitem(sys.modules, ascend_interface.__name__, ascend_interface)
    factory = definitions(
        "references/vllm-ascend/vllm_ascend/core/single_type_kv_cache_manager.py", "native_manager_factory_contract",
        {"get_manager_for_kv_cache_spec"},
        {**interface.__dict__, "__name__": "native_manager_factory_contract"}, monkeypatch)
    coordinator = definitions(
        "references/vllm-ascend/vllm_ascend/patch/platform/patch_kv_cache_coordinator.py", "native_coordinator_contract",
        {"AscendHybridKVCacheCoordinator"},
        {**interface.__dict__, "__name__": "native_coordinator_contract", "lcm": math.lcm,
         "HybridKVCacheCoordinator": object, "BlockPool": pool_types.BlockPool,
         "get_manager_for_kv_cache_spec": factory.get_manager_for_kv_cache_spec,
         "SlidingWindowManager": managers.SlidingWindowManager, "envs_vllm": SimpleNamespace(),
         "vllm_kv_cache_coordinator": SimpleNamespace()}, monkeypatch)
    pool_config = utils.get_kv_cache_config_from_groups(config, groups, 17 * 801792 * 100)
    instance = coordinator.AscendHybridKVCacheCoordinator(
        pool_config, max_model_len=262144, use_eagle=True, enable_caching=True,
        enable_kv_cache_events=False, dcp_world_size=1, pcp_world_size=1,
        hash_block_size=768, max_num_batched_tokens=16384, scheduler_block_size=2304)
    assert instance.lcm_block_size == 2304 and instance.hash_block_size == 768
    assert [manager.block_size for manager in instance.single_type_managers] == [2304, 768, 768, 768]
    full_manager, *gdn_managers = instance.single_type_managers
    assert type(full_manager) is managers.FullAttentionManager
    assert all(type(manager).__name__ == "AscendMambaManager" for manager in gdn_managers)
    assert all(manager.scheduler_block_size == 2304 for manager in instance.single_type_managers)
    assert all(manager.block_pool is instance.block_pool for manager in instance.single_type_managers)
    allocations = [manager.allocate_new_blocks("request", 4611, 4608) for manager in instance.single_type_managers]
    ids = [[block.block_id for block in blocks] for blocks in allocations]
    assert len(ids[0]) == 3  # ceil((4608 + 3 lookahead)/2304)
    # Actual MambaManager.allocate_new_blocks:1165-1198 inserts null metadata
    # placeholders for historical positions and allocates only the current
    # state plus three speculative states. Null IDs do not own physical bytes.
    active = [[block_id for block_id in group if block_id != 0] for group in ids]
    assert list(map(len, active)) == [3, 4, 4, 4]
    assert len(set(itertools.chain.from_iterable(active))) == sum(map(len, active))
    assert instance.block_pool.null_block.is_null


def test_real_native_gdn_reshape_runs_with_original_objects_and_byte_ranges(native_specs, monkeypatch):
    interface, ours, registry = native_specs
    full_name = "language_model.model.layers.3.self_attn.attn"
    gdn_name = "language_model.model.layers.0.linear_attn"
    native_gdn = interface.MambaSpec(block_size=768, shapes=((3, 2560), (12, 128, 128)),
                                   dtypes=(torch.bfloat16, torch.bfloat16), page_size_padded=801792)
    native_full = interface.FullAttentionSpec(block_size=768, num_kv_heads=1, head_size=256,
                                             dtype=torch.bfloat16, page_size_padded=801792)
    specs = ours.transform_native_specs({full_name: native_full, gdn_name: native_gdn})
    groups = [interface.KVCacheGroupSpec([name], spec) for name, spec in specs.items()]
    config = interface.KVCacheConfig(3, [interface.KVCacheTensor(3 * 801792, [full_name, gdn_name])], groups)
    native = definitions(
        "references/vllm-ascend/vllm_ascend/worker/model_runner_v1.py", "native_runner_contract",
        set(), {"torch": torch, "math": math, **interface.__dict__}, monkeypatch,
        methods=[("NPUModelRunner", "_get_layer_kv_cache_specs"), ("NPUModelRunner", "_reshape_kv_cache_tensors")])
    runner = SimpleNamespace(
        runner_only_attn_layers=set(), use_compress=False, device="cpu",
        parallel_config=SimpleNamespace(tensor_parallel_size=4),
        model_config=SimpleNamespace(get_num_attention_heads=lambda _parallel: 6),
        vllm_config=SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=16384),
                                   compilation_config=SimpleNamespace(cudagraph_capture_sizes=[512])))
    runner._get_layer_kv_cache_specs = MethodType(native._get_layer_kv_cache_specs, runner)
    runner._kv_cache_spec_attn_group_iterator = lambda: iter(
        SimpleNamespace(backend=object(), layer_names=g.layer_names, kv_cache_spec=g.kv_cache_spec) for g in groups)
    allocation = torch.full((3 * 801792,), 37, dtype=torch.int8)
    raw = {full_name: allocation, gdn_name: allocation}
    gdn_original_bytes = allocation[:3 * (15360 + 393216)].clone()
    provider = AscendRuntimeProvider({"sink_tokens": 64, "recent_tokens": 256,
                                     "speculative_config": {"num_speculative_tokens": 3}})
    rotation = SimpleNamespace(key_transposed=torch.eye(256), value_transposed=torch.eye(256),
                               inverse_value_transposed=torch.eye(256), hadamard=False)
    monkeypatch.setattr(provider, "_rotations", lambda *_args: {full_name: rotation})
    monkeypatch.setattr(provider, "ensure_workspace", lambda *_args: SimpleNamespace(
        geometry=SimpleNamespace(tokens=16384, total_bytes=1)))
    previous_skip_set = runner.runner_only_attn_layers
    views = provider.reshape_kv_cache_tensors(
        runner, config, raw, lambda cfg, tensors: native._reshape_kv_cache_tensors(runner, cfg, tensors))
    assert runner.runner_only_attn_layers is previous_skip_set
    conv, ssm = views[gdn_name]
    assert conv.shape == (3, 3, 2560) and ssm.shape == (3, 12, 128, 128)
    assert conv.data_ptr() == allocation.data_ptr()
    assert ssm.data_ptr() == allocation.data_ptr() + 3 * 15360
    assert conv.dtype == ssm.dtype == torch.bfloat16
    assert views[full_name].shape == (3, 2304, 1, 136)
    assert provider.layers[full_name].window_key.data_ptr() >= allocation.data_ptr() + 3 * (15360 + 393216)
    torch.testing.assert_close(allocation[:gdn_original_bytes.numel()], gdn_original_bytes, rtol=0, atol=0)


def test_driver_core_discovery_executes_native_property_functions(monkeypatch):
    queried = []
    driver = SimpleNamespace(get_device_properties=lambda index: queried.append(index) or
                             {"num_aicore": 20, "num_vectorcore": 40})
    native = definitions(
        "references/vllm-ascend/vllm_ascend/ops/triton/triton_utils.py",
        "vllm_ascend.ops.triton.triton_utils",
        {"init_device_properties_triton", "get_aicore_num", "get_vectorcore_num"},
        {"torch": torch, "HAS_TRITON": True, "_NUM_AICORE": -1, "_NUM_VECTORCORE": -1,
         "triton": SimpleNamespace(runtime=SimpleNamespace(driver=SimpleNamespace(active=SimpleNamespace(utils=driver))))},
        monkeypatch)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_device=lambda: 0), raising=False)
    monkeypatch.setattr(torch, "device", lambda _device: SimpleNamespace(type="npu", index=0))
    provider = AscendRuntimeProvider({})
    assert provider._device_cube_cores("npu:0") == 20
    assert provider._device_cube_cores("npu:0") == 20
    assert queried == [0]
    assert native.get_aicore_num() == 20
    with pytest.raises(OscarReadinessError, match="exceeds measured"):
        AscendRuntimeProvider({"cube_cores": 24})._device_cube_cores("npu:0")


def test_real_native_metadata_base_builds_capture_then_replay(monkeypatch):
    base = definitions(
        "references/vllm/vllm/v1/attention/backend.py", "vllm.v1.attention.backend",
        {"AttentionType", "MultipleOf", "AttentionBackend", "AttentionCGSupport", "AttentionMetadataBuilder"},
        {"ABC": ABC, "abstractmethod": abstractmethod, "Enum": Enum, "Generic": Generic,
         "ClassVar": ClassVar, "M": TypeVar("M"), "torch": torch}, monkeypatch)
    backend = load_our_module("oscar_ascend.integration.backend", monkeypatch)
    builder = backend.OscarMetadataBuilder(object(), ["layer"], object(), torch.device("cpu"))
    common = SimpleNamespace(
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        seq_lens=torch.tensor([4, 4], dtype=torch.int32),
        slot_mapping=torch.zeros(8, dtype=torch.int64),
        block_table_tensor=torch.zeros((2, 2052), dtype=torch.int32),
        num_reqs=2, num_actual_tokens=8, num_input_tokens=8, max_query_len=4,
        max_seq_len=262144, causal=True)
    capture = builder.build_for_cudagraph_capture(common)
    assert capture.capture_origin and bool(torch.all(capture.slot_mapping == -1))
    common.slot_mapping.copy_(torch.arange(8))
    common.seq_lens.copy_(torch.tensor([2308, 4612], dtype=torch.int32))
    live = builder.build(0, common)
    assert not live.capture_origin and bool(torch.equal(live.slot_mapping, torch.arange(8)))
    assert builder.get_cudagraph_support(None, None) is base.AttentionCGSupport.UNIFORM_BATCH
    assert builder.kv_cache_spec is not None and builder.layer_names == ["layer"]
    # The first native draft build passes the model as argument three; later
    # draft steps call the native base build_for_drafting method. Both retain
    # device tensors and work with eager draft metadata whose max_seq_len=0.
    draft = copy.copy(common)
    draft.max_seq_len = 0
    draft.slot_mapping = torch.arange(12, dtype=torch.int32)
    draft.seq_lens = common.seq_lens.clone()
    draft.query_start_loc = common.query_start_loc.clone()
    draft.block_table_tensor = common.block_table_tensor.clone()
    draft_builder = backend.OscarMetadataBuilder(object(), ["mtp.layers.0.self_attn.attn"], object(), "cpu")
    first = draft_builder.build(0, draft, object())
    later = draft_builder.build_for_drafting(draft, 1)
    assert first.max_seq_len == later.max_seq_len == 0
    assert first.slot_mapping.dtype == later.slot_mapping.dtype == torch.int32
    assert backend.OscarAttentionBackend.supports_dtype(torch.bfloat16)
    assert not backend.OscarAttentionBackend.supports_dtype(torch.float16)


def test_native_profile_selects_metadata_none_before_kv_cache_initialization(monkeypatch):
    modes = SimpleNamespace(NONE=object(), FULL=object())
    native = definitions(
        "references/vllm-ascend/vllm_ascend/worker/model_runner_v1.py", "native_profile_contract",
        set(), {"CUDAGraphMode": modes}, monkeypatch,
        methods=[("NPUModelRunner", "_should_build_dummy_attn_metadata")])
    predicate = native._should_build_dummy_attn_metadata
    assert not predicate(object(), force_attention=False, is_profile=True, cudagraph_runtime_mode=modes.NONE)
    assert predicate(object(), force_attention=False, is_profile=False, cudagraph_runtime_mode=modes.FULL)


def test_native_extra_fia_dummy_does_not_require_a_new_physical_request_row():
    from oscar_ascend.integration.metadata import from_common, OscarMetadataError
    common = SimpleNamespace(
        query_start_loc=torch.tensor([0, 4, 7, 8], dtype=torch.int32),
        seq_lens=torch.tensor([100, 200], dtype=torch.int32),
        slot_mapping=torch.tensor([128, 129, 130, 131, 256, 257, 258, -1], dtype=torch.int64),
        block_table_tensor=torch.tensor([[1, 3, 0], [2, 4, 0]], dtype=torch.int32),
        num_reqs=3, num_actual_tokens=7, num_input_tokens=8, max_query_len=4,
        max_seq_len=200, causal=True)
    metadata = from_common(common)
    assert metadata.query_start_loc.data_ptr() == common.query_start_loc.data_ptr()
    assert metadata.query_start_loc.shape == (3,)
    assert torch.equal(metadata.query_start_loc, torch.tensor([0, 4, 7], dtype=torch.int32))
    assert metadata.seq_lens is common.seq_lens and metadata.block_tables is common.block_table_tensor
    assert metadata.slot_mapping is common.slot_mapping and metadata.num_reqs == 2
    common.num_reqs = 4
    with pytest.raises(OscarMetadataError, match="capacity exceeded"):
        from_common(common)


@pytest.mark.parametrize("base", [100, 2302, 2303])
@pytest.mark.parametrize("accepted_proposals", [0, 1, 2, 3])
def test_native_padded_mtp_requires_slot_derived_draft_position(monkeypatch, base, accepted_proposals):
    # Execute the actual native step-update method, including its seq_lens
    # increment and its separate accepted-position slot computation. This
    # reproduces the disagreement instead of inventing a corrected mock.
    native = definitions(
        "references/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py", "native_mtp_update_contract",
        set(), {"torch": torch, "CUDAGraphMode": SimpleNamespace(FULL="full"),
                "AscendAttentionState": SimpleNamespace(SpecDecoding="spec", ChunkedPrefill="chunk"),
                "PADDING_SLOT_ID": -1}, monkeypatch,
        methods=[("AscendSpecDecodeBaseProposer", "attn_update_stack_num_spec_norm")])
    # NumPy is not needed for the numerical contract under test. Replace only
    # the host arange conversion dependency with the identical int32 values.
    monkeypatch.setattr(torch, "from_numpy", lambda values: values.clone())
    from oscar_ascend.integration.metadata import from_common
    updates = []

    class Builder:
        def build_for_drafting(self, common, draft_index):
            updates.append(draft_index)
            return replace(from_common(common), draft_index=draft_index)

    proposer = SimpleNamespace(
        shallow_copy_metadata=copy.copy, arange=torch.arange(5, dtype=torch.int32),
        token_arange_np=torch.arange(5, dtype=torch.int32), method="mtp", uses_mrope=False,
        max_model_len=262144, runner=SimpleNamespace(), has_gdn=True, kernel_block_size=128,
        block_size=2304, pcp_size=1, use_compress=False,
        slot_mapping_group=[torch.full((8,), -1, dtype=torch.int32) for _ in range(3)],
        seq_lens_group=[torch.zeros(4, dtype=torch.int32) for _ in range(3)],
        query_start_loc_group=[torch.zeros(5, dtype=torch.int32) for _ in range(3)])
    # The second physical page has a smaller ID: physical IDs are not sorted
    # by logical position, so slot-to-logical recovery cannot binary-search it.
    table = torch.tensor([[7 * 18 + i for i in range(18)] + [18 + i for i in range(18)]], dtype=torch.int32)
    common = SimpleNamespace(
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 4], dtype=torch.int32),
        seq_lens=torch.tensor([base + 4], dtype=torch.int32), seq_lens_cpu=None, _seq_lens_cpu=None,
        num_computed_tokens_cpu=None, positions=torch.tensor([base], dtype=torch.int32),
        block_table_tensor=table, slot_mapping=torch.zeros(4, dtype=torch.int32),
        num_reqs=1, num_actual_tokens=4, num_input_tokens=4, max_query_len=4,
        max_seq_len=0, causal=True)
    used_positions = torch.tensor([base + accepted_proposals], dtype=torch.int32)
    group = SimpleNamespace(get_metadata_builder=lambda: Builder())
    updated, metadata = native.attn_update_stack_num_spec_norm(
        proposer, 1, None, common, 1, 1, used_positions, "none", attn_group=group)
    slot = int(updated.slot_mapping[0])
    matches = (table[0] == slot // 128).nonzero().flatten()
    assert matches.numel() == 1
    logical_position = int(matches[0]) * 128 + slot % 128
    assert logical_position == base + accepted_proposals + 1
    assert int(updated.seq_lens[0]) - 1 - logical_position == 3 - accepted_proposals
    assert metadata.draft_index == 1 and updates == [1]
    assert int(updated.query_start_loc[-1]) == 1
