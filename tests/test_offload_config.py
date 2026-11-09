"""Archive #34/#36/#77/#78/#127/#128/#157/#163/#171: offload launch seams.

The explicitly requested KV-transfer argument is the only extension to the
qualified native argv. Native none/align, model/PT/math, placement and timer
selection are preserved. These are host checks, not NPU/service acceptance.
"""
from copy import deepcopy
import gc
import json
import logging
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace
import weakref

import pytest
import torch

from test_runtime_native import definitions, native_specs  # noqa: F401
from test_stable_launch_frozen import FROZEN, AUTHORIZED_KV_TRANSFER, authorized_model_arguments
from oscar_ascend.offload.config import (
    ALIGNMENT, CONNECTOR, MODULE, allocate_aligned_bytes, offload_options,
    transfer_config, validate_transfer_config,
)
from oscar_ascend.rotations import calibration_cache_identity, rotation_artifact_path
from tools import deploy, install_serve, observe_serve, target_cli
from tools.serving_variants import variant_config, variant_features

ROOT = Path(__file__).resolve().parents[1]


def enabled_target():
    target = json.loads((ROOT / "configs/target.json").read_text())
    target["kv_offload"] = {"enabled": True, "backend": "memcache"}
    return target


def remove_transfer(argv):
    result = list(argv)
    assert result.count("--kv-transfer-config") == 1
    index = result.index("--kv-transfer-config")
    value = json.loads(result[index + 1])
    del result[index:index + 2]
    return result, value


def test_enabled_registration_and_config_import_remain_lightweight(tmp_path):
    path = tmp_path / "target.json"
    path.write_text(json.dumps(enabled_target()))
    code = """
import json, sys
from oscar_ascend.offload.config import transfer_config
from oscar_ascend.plugin import register, unregister
register()
from oscar_ascend.plugin import hook_status
assert hook_status()['enabled']
print(json.dumps(sorted(name for name in sys.modules
    if name.split('.')[0] in {'torch', 'torch_npu', 'vllm', 'vllm_ascend', 'memcache_hybrid'})))
unregister()
"""
    env = dict(os.environ, OSCAR_TARGET_CONFIG=str(path), OSCAR_ENABLED="1")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                             text=True, capture_output=True, check=True)
    assert json.loads(result.stdout) == []


def test_connector_config_is_dedicated_complete_and_does_not_mutate_input():
    target = enabled_target()
    previous = deepcopy(target)
    transfer = transfer_config(target)
    assert target == previous
    assert transfer["kv_connector"] == CONNECTOR
    assert transfer["kv_connector_module_path"] == MODULE
    assert transfer["kv_role"] == "kv_both"
    assert transfer["kv_load_failure_policy"] == "fail"
    assert transfer["kv_connector_extra_config"]["use_layerwise"] is False
    assert transfer == AUTHORIZED_KV_TRANSFER
    validate_transfer_config(SimpleNamespace(**transfer))
    for field, value in (("kv_connector", "AscendStoreConnector"),
                         ("kv_connector_module_path", "unrelated.connector"),
                         ("kv_role", "kv_producer"),
                         ("kv_load_failure_policy", "recompute")):
        invalid = dict(transfer, **{field: value})
        with pytest.raises(ValueError):
            validate_transfer_config(SimpleNamespace(**invalid))
    for extra in (None, {}, {"backend": "other"}, {"backend": "memcache", "use_layerwise": True},
                  dict(transfer["kv_connector_extra_config"], enabled=False),
                  dict(transfer["kv_connector_extra_config"], max_batch_objects=0),
                  dict(transfer["kv_connector_extra_config"], timeout_seconds=float("nan"))):
        with pytest.raises(ValueError):
            validate_transfer_config(SimpleNamespace(**dict(transfer, kv_connector_extra_config=extra)))


@pytest.mark.parametrize("options", [
    "memcache", {"enabled": "true"}, {"enabled": True, "backend": "memcached"},
    {"enabled": True, "max_batch_objects": 0}, {"enabled": True, "max_batch_objects": True},
    {"enabled": True, "max_batch_objects": 4097}, {"enabled": True, "timeout_seconds": 0},
    {"enabled": True, "timeout_seconds": float("nan")},
    {"enabled": True, "timeout_seconds": float("inf")},
    {"enabled": True, "timeout_seconds": True}, {"enabled": True, "silent_fallback": True},
])
def test_invalid_offload_configuration_fails_before_model_import(options):
    with pytest.raises(ValueError):
        offload_options({"kv_offload": options})


@pytest.mark.parametrize("prefix", [False, True])
def test_real_native_config_keeps_none_or_align_without_forcing_store_policy(monkeypatch, prefix):
    state = definitions(
        "references/vllm/vllm/model_executor/layers/mamba/mamba_utils.py", "offload_native_shape",
        {"MambaStateShapeCalculator"},
        {"divide": lambda a, b: a // b, "is_conv_state_dim_first": lambda: False}, monkeypatch)
    qwen = definitions(
        "references/vllm/vllm/model_executor/models/qwen3_5.py", "offload_native_qwen_shape",
        set(), {"MambaStateShapeCalculator": state.MambaStateShapeCalculator}, monkeypatch,
        methods=[("Qwen3_5ForConditionalGeneration", "get_mamba_state_shape_from_config")])
    model_cls = type("QwenConfiguredShape", (), {
        "get_mamba_state_shape_from_config": qwen.get_mamba_state_shape_from_config,
        "get_mamba_state_dtype_from_config": classmethod(lambda cls, cfg: (torch.bfloat16, torch.bfloat16))})
    base = definitions(
        "references/vllm/vllm/model_executor/models/config.py", "offload_native_mamba_config",
        {"VerifyAndUpdateConfig", "MambaModelConfig"},
        {"logger": logging.getLogger("offload-native-config")}, monkeypatch)
    ascend = definitions(
        "references/vllm-ascend/vllm_ascend/patch/platform/patch_mamba_config.py", "offload_native_hybrid_config",
        {"_using_kv_store", "verify_and_update_config"},
        {"math": math, "MambaModelConfig": base.MambaModelConfig,
         "ModelRegistry": SimpleNamespace(resolve_model_cls=lambda *a, **k: (model_cls, "fixture")),
         "logger": logging.getLogger("offload-native-config"), "cdiv": lambda a, b: (a + b - 1) // b,
         "get_dtype_size": lambda dtype: torch.empty((), dtype=dtype).element_size()}, monkeypatch)
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False, enable_chunked_prefill=True),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16, architecture="Qwen3_5ForConditionalGeneration", use_mla=False,
            supports_mamba_prefix_caching=False, max_model_len=262144,
            get_num_kv_heads=lambda parallel: 1, get_head_size=lambda: 256,
            hf_text_config=SimpleNamespace(linear_num_key_heads=16, linear_num_value_heads=48,
                                           linear_key_head_dim=128, linear_value_head_dim=128,
                                           linear_conv_kernel_dim=4)),
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=None, mamba_page_size_padded=None,
                                     mamba_block_size=None, enable_prefix_caching=prefix, mamba_cache_mode="none"),
        parallel_config=SimpleNamespace(tensor_parallel_size=4),
        speculative_config=SimpleNamespace(num_speculative_tokens=3, method="mtp"),
        kv_transfer_config=SimpleNamespace(**transfer_config(enabled_target())))
    # The external adapter has a complete none-mode state protocol; bypassing
    # native AscendStore's prefix-only align policy is therefore intentional.
    assert ascend._using_kv_store(config) is False
    ascend.verify_and_update_config.__func__(None, config)
    assert config.cache_config.mamba_cache_mode == ("align" if prefix else "none")
    assert config.cache_config.mamba_block_size == (768 if prefix else 262144)
    assert config.cache_config.block_size == 768
    assert config.cache_config.mamba_page_size_padded == 817152
    assert model_cls.get_mamba_state_shape_from_config(config) == ((6, 2560), (12, 128, 128))


def test_real_aligned_allocator_retains_owner_through_native_views():
    allocations = []

    class TorchSpy:
        int8 = torch.int8

        @staticmethod
        def zeros(*args, **kwargs):
            value = torch.zeros(*args, **kwargs)
            allocations.append(weakref.ref(value))
            return value

    size = 817152 * 3
    raw = allocate_aligned_bytes(TorchSpy, size, "cpu")
    assert raw.data_ptr() % ALIGNMENT == 0
    assert raw.shape == (size,) and raw.dtype == torch.int8 and raw.stride() == (1,)
    assert raw.untyped_storage().nbytes() == size + ALIGNMENT
    assert raw.storage_offset() < ALIGNMENT
    assert allocations[0]() is not None
    storage_pointer = raw.untyped_storage().data_ptr()
    raw.fill_(37)
    native_view = raw.view(torch.uint8).as_strided((3, 64), (128, 1), raw.storage_offset())
    pointer = native_view.data_ptr()
    del raw
    gc.collect()
    # A dtype reinterpretation can release the Python owner object while
    # retaining its C++ storage. Storage identity/liveness is the contract.
    assert native_view.untyped_storage().data_ptr() == storage_pointer
    assert native_view.untyped_storage().nbytes() == size + ALIGNMENT
    assert native_view.data_ptr() == pointer and torch.all(native_view == 37)


def test_runtime_allocates_one_aligned_owner_and_rejects_config_drift(native_specs, monkeypatch):
    from oscar_ascend.runtime import AscendRuntimeProvider
    from oscar_ascend.integration.runtime_api import OscarReadinessError

    interface, ours, _ = native_specs
    full = interface.FullAttentionSpec(block_size=768, num_kv_heads=1, head_size=256,
                                      dtype=torch.bfloat16, page_size_padded=817152)
    gdn = interface.MambaSpec(block_size=262144, shapes=((6, 2560), (12, 128, 128)),
        dtypes=(torch.bfloat16, torch.bfloat16), page_size_padded=817152,
        num_speculative_blocks=3, mamba_cache_mode="none")
    specs = ours.transform_native_specs({"full": full, "gdn": gdn}, metadata_bytes=4)
    allocation_calls = []

    def zeros(size, *, dtype, device):
        allocation_calls.append((size, dtype, device))
        return torch.zeros(size, dtype=dtype, device="cpu")

    torch_metadata_spy = SimpleNamespace(int8=torch.int8, device=lambda device: SimpleNamespace(type="npu"),
                                         zeros=zeros)
    config = enabled_target()
    provider = AscendRuntimeProvider.__new__(AscendRuntimeProvider)
    provider.config = config
    transfer = SimpleNamespace(**transfer_config(config))
    runner = SimpleNamespace(device="npu:0", vllm_config=SimpleNamespace(kv_transfer_config=transfer),
                              _get_layer_kv_cache_specs=lambda cache: specs)
    size = 3 * specs["full"].page_size_bytes
    cache = SimpleNamespace(kv_cache_tensors=[SimpleNamespace(size=size, shared_by=("full", "gdn"))])
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "torch", torch_metadata_spy)
        raw = provider.allocate_kv_cache_tensors(runner, cache, lambda cache: pytest.fail("native allocation"))
    assert len(allocation_calls) == 1
    provider.config = config
    runner.vllm_config.kv_transfer_config = None
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "torch", torch_metadata_spy)
        with pytest.raises(OscarReadinessError, match="configuration is missing"):
            provider.allocate_kv_cache_tensors(runner, cache, None)
    assert len(allocation_calls) == 1
    assert raw["full"] is raw["gdn"]
    assert raw["full"].numel() == size and raw["full"].data_ptr() % ALIGNMENT == 0
    assert provider._offload_runner is runner and runner.hybrid_with_attn_and_mamba
    provider.config = dict(config, kv_offload={"enabled": False})
    runner.vllm_config.kv_transfer_config = transfer
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "torch", torch_metadata_spy)
        with pytest.raises(OscarReadinessError, match="disagree"):
            provider.allocate_kv_cache_tensors(runner, cache, None)
    assert len(allocation_calls) == 1


@pytest.mark.parametrize("entry,name", [(install_serve.main, "direct"), (observe_serve.main, "observe")])
@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("rear", [False, True])
def test_only_explicit_transfer_extends_frozen_argv_and_placement(tmp_path, capsys, entry, name, prefix, rear):
    path = tmp_path / "target.json"
    target = enabled_target()
    if rear and name == "observe":
        # Only direct install_serve exposes --rear-cards; other entry points
        # continue to honor explicitly supplied placement in their config.
        target.update(devices=[4, 5, 6, 7], port=7878)
    path.write_text(json.dumps(target))
    options = ["--config", str(path), "--variant", "candidate", "--plan"]
    if prefix:
        options.append("--enable-prefix-caching")
    if rear and name == "direct":
        options.append("--rear-cards")
    assert entry(options) == 0
    report = json.loads(capsys.readouterr().out)
    argv, transfer = remove_transfer(report["vllm_argv"])
    expected = authorized_model_arguments(
        FROZEN["cases"][f"{name}_prefix_{str(prefix).lower()}"]["argv"], offload=False)
    if rear:
        expected[expected.index("--port") + 1] = "7878"
    assert argv == expected
    assert transfer == transfer_config(enabled_target())
    assert report.get("target_devices", report.get("devices")) == ([4, 5, 6, 7] if rear else [0, 1, 2, 3])


def test_offload_selection_cannot_change_pt_math_or_timer_configuration():
    original = enabled_target()
    local = deepcopy(original)
    local.pop("kv_offload")
    enabled = variant_config(original, "candidate")
    disabled = variant_config(local, "candidate")
    assert {key: value for key, value in enabled.items() if key != "kv_offload"} == disabled
    assert variant_features(enabled) == variant_features(disabled)
    fingerprint = "a" * 64
    assert calibration_cache_identity(enabled, fingerprint) == calibration_cache_identity(disabled, fingerprint)
    assert rotation_artifact_path(enabled, fingerprint) == rotation_artifact_path(disabled, fingerprint)
    base_env = {"OSCAR_TIMER_CONTROL": "/tmp/explicit-timer", "OSCAR_PASSIVE_TIMING_CONTROL": "/tmp/stale"}
    enabled_env = target_cli.target_env(enabled, base_env)
    assert enabled_env.pop("PYTHONHASHSEED") == "0"
    assert enabled_env == target_cli.target_env(disabled, base_env)


@pytest.mark.parametrize("variant", ["baseline", "native"])
def test_native_and_historical_baseline_remove_external_offload(variant):
    original = enabled_target()
    result = variant_config(original, variant)
    assert "kv_offload" not in result
    assert target_cli.serve_argv(result) == authorized_model_arguments(
        FROZEN["cases"]["direct_prefix_false"]["argv"], offload=False)
    assert original["kv_offload"]["enabled"] is True


def test_direct_native_cli_strips_offload_before_building_argv(tmp_path, capsys):
    path = tmp_path / "target.json"
    path.write_text(json.dumps(enabled_target()))
    assert target_cli.main(["--config", str(path), "--native", "--print-command"]) == 0
    argv = shlex.split(capsys.readouterr().out)
    assert "--kv-transfer-config" not in argv


@pytest.mark.parametrize("entry", [install_serve, deploy])
@pytest.mark.parametrize("config_name, port", [(None, 8989), ("benchmark_node93.json", 7878)])
def test_fixed_default_candidate_command_enables_offload(capsys, entry, config_name, port):
    args = ["--plan"]  # No variant/offload flags: exercise the shipped defaults.
    if config_name:
        args += ["--config", str(ROOT / "configs" / config_name)]
    assert entry.main(args) == 0
    report = json.loads(capsys.readouterr().out)
    _, transfer = remove_transfer(report["vllm_argv"])
    assert transfer["kv_connector"] == CONNECTOR
    assert transfer["kv_connector_extra_config"]["backend"] == "memcache"
    assert transfer["kv_load_failure_policy"] == "fail"
    assert report["target_devices"] == [0, 1, 2, 3]
    argv = report["vllm_argv"]
    assert argv[argv.index("--port") + 1] == str(port)
    assert argv[argv.index("--max-num-batched-tokens") + 1] == "16384"


def test_explicit_disabled_config_stays_disabled():
    target = enabled_target()
    target["kv_offload"]["enabled"] = False
    assert transfer_config(variant_config(target, "candidate")) is None


def test_probe_capacity_override_requires_explicit_probe_environment(monkeypatch):
    config = variant_config(enabled_target(), "candidate")
    config["offload_probe_num_gpu_blocks"] = 17
    monkeypatch.delenv("OSCAR_OFFLOAD_SERVICE_PROBE", raising=False)
    with pytest.raises(ValueError, match="explicit service probe"):
        target_cli.serve_argv(config)
    monkeypatch.setenv("OSCAR_OFFLOAD_SERVICE_PROBE", "1")
    argv = target_cli.serve_argv(config)
    assert argv[-2:] == ["--num-gpu-blocks-override", "17"]
    for value in (0, -1, True, "17"):
        with pytest.raises(ValueError, match="explicit service probe"):
            target_cli.serve_argv(dict(config, offload_probe_num_gpu_blocks=value))


@pytest.mark.parametrize("enabled", [False, True])
def test_chunked_prefill_override_is_explicit_and_boolean(enabled):
    config = variant_config(enabled_target(), "candidate")
    base = target_cli.serve_argv(config)
    argv = target_cli.serve_argv(dict(config, enable_chunked_prefill=enabled))
    assert argv == base + ["--enable-chunked-prefill" if enabled else "--no-enable-chunked-prefill"]
    with pytest.raises(ValueError, match="explicit boolean"):
        target_cli.serve_argv(dict(config, enable_chunked_prefill="false"))


@pytest.mark.parametrize("seed", [None, "0", "4231", "random"])
def test_offload_hashseed_defaults_only_when_unset_and_preserves_native_control(seed):
    config = enabled_target()
    base = {} if seed is None else {"PYTHONHASHSEED": seed}
    prior = dict(base)
    enabled = target_cli.target_env(config, base)
    assert enabled["PYTHONHASHSEED"] == ("0" if seed is None else seed)
    config["kv_offload"]["enabled"] = False
    disabled = target_cli.target_env(config, base)
    assert disabled.get("PYTHONHASHSEED") == seed
    assert base == prior


def test_seeded_services_use_the_unchanged_native_none_hash_initializer(monkeypatch):
    import hashlib
    from typing import Any, Callable

    def random_would_break_reuse(size):
        raise AssertionError("seeded service must not request a random native root hash")

    for base in ({}, {"PYTHONHASHSEED": "4231"}):
        env = target_cli.target_env(enabled_target(), base)
        namespace = definitions("references/vllm/vllm/v1/core/kv_cache_utils.py",
            "offload_native_hash_seed", {"init_none_hash"},
            {"os": SimpleNamespace(getenv=env.get, urandom=random_would_break_reuse),
             "Callable": Callable, "Any": Any, "BlockHash": bytes,
             "_CBOR_HASH_FUNCTIONS": frozenset()}, monkeypatch)
        hash_fn = lambda value: hashlib.sha256(value.encode()).digest()
        namespace.init_none_hash(hash_fn)
        first = namespace.NONE_HASH
        namespace.init_none_hash(hash_fn)
        assert namespace.NONE_HASH == first == hash_fn(env["PYTHONHASHSEED"])
