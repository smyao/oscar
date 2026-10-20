"""Archive #28/#77/#86/#97/#127: bind PT identity before native model loading.

Host fakes expose the pinned runner's model_config/vllm_config fields. They
prove mismatch rejection and hook ordering, not model/NPU/graph acceptance.
"""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from oscar_ascend import plugin
from oscar_ascend.integration.model_identity import validate_runner_model_identity
from oscar_ascend.integration.runtime_api import (
    OscarReadinessError, clear_runtime, install_runtime,
)


def model_config(tmp_path, *, quantized=False, name="model"):
    directory = tmp_path / name
    directory.mkdir()
    metadata = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5_text",
            "mamba_ssm_dtype": "float32",
        },
    }
    (directory / "config.json").write_text(json.dumps(metadata))
    if quantized:
        (directory / "quant_model_description.json").write_text(json.dumps(
            {"model.layers.0.self_attn.q_proj.weight": "W8A8"}
        ))
    return {
        "model": str(directory),
        "served_model_name": "qwen3.8",
        "dtype": "bfloat16",
        "quantization": "ascend" if quantized else None,
        "quantization_mode": "ascend" if quantized else "none",
        "hf_overrides": {},
    }


def native_runner(config, *, direct_model_config=True):
    model = SimpleNamespace(
        model=config["model"], quantization=config["quantization"],
        dtype="bfloat16", revision=config.get("revision"),
        hf_overrides=deepcopy(config.get("hf_overrides", {})),
    )
    cache = SimpleNamespace(mamba_cache_dtype="bfloat16",
                            mamba_ssm_cache_dtype="bfloat16")
    runner = SimpleNamespace(vllm_config=SimpleNamespace(
        model_config=model, cache_config=cache,
    ))
    if direct_model_config:
        runner.model_config = model
    return runner


@pytest.mark.parametrize("quantized", [False, True], ids=["original", "w8a8"])
@pytest.mark.parametrize("direct", [False, True], ids=["vllm_config", "runner_model_config"])
def test_matching_actual_model_execution_accepts_without_rewriting_config(tmp_path, quantized, direct):
    config = model_config(tmp_path, quantized=quantized)
    before = deepcopy(config)
    runner = native_runner(config, direct_model_config=direct)
    actual_before = deepcopy(vars(runner.vllm_config.model_config))
    result = validate_runner_model_identity(config, runner)
    assert result == {
        "model": str(Path(config["model"]).resolve()),
        "quantization": "ascend" if quantized else None,
        "dtype": "bfloat16",
        "mamba_cache_dtype": "bfloat16",
        "mamba_ssm_cache_dtype": "bfloat16",
    }
    assert config == before
    assert vars(runner.vllm_config.model_config) == actual_before


def test_manual_actual_model_cannot_use_another_checkpoints_oscar_config(tmp_path):
    config = model_config(tmp_path, name="configured")
    other = model_config(tmp_path, name="manual")
    with pytest.raises(OscarReadinessError, match="model identity mismatch.*'model'"):
        validate_runner_model_identity(config, native_runner(other))


@pytest.mark.parametrize("quantized", [False, True], ids=["added_ascend", "removed_ascend"])
def test_manual_quantization_flag_drift_cannot_borrow_the_configured_rotations(tmp_path, quantized):
    config = model_config(tmp_path, quantized=quantized)
    runner = native_runner(config)
    runner.model_config.quantization = None if quantized else "ascend"
    with pytest.raises(OscarReadinessError, match="model identity mismatch.*'quantization'"):
        validate_runner_model_identity(config, runner)


def test_symlink_paths_to_the_same_checkpoint_have_the_same_identity(tmp_path):
    config = model_config(tmp_path)
    alias = tmp_path / "checkpoint_alias"
    alias.symlink_to(config["model"], target_is_directory=True)
    runner = native_runner(config)
    runner.model_config.model = str(alias)
    assert validate_runner_model_identity(config, runner)["model"] == str(Path(config["model"]).resolve())


@pytest.mark.parametrize("field,actual", [
    ("dtype", "float32"),
    ("revision", "other-revision"),
    ("hf_overrides", {"text_config": {"rope_theta": 10000}}),
])
def test_actual_execution_metadata_must_match_explicit_config(tmp_path, field, actual):
    config = model_config(tmp_path)
    config["revision"] = "selected-revision"
    config["hf_overrides"] = {"text_config": {"rope_theta": 1000000}}
    runner = native_runner(config)
    setattr(runner.model_config, field, actual)
    with pytest.raises(OscarReadinessError, match=f"model identity mismatch.*'{field}'"):
        validate_runner_model_identity(config, runner)


@pytest.mark.parametrize("conv", ["auto", "bfloat16"])
@pytest.mark.parametrize("ssm", ["auto", "bfloat16"])
def test_native_auto_resolves_to_bf16_state_even_when_hf_declares_fp32(tmp_path, conv, ssm):
    config = model_config(tmp_path)
    runner = native_runner(config)
    runner.vllm_config.cache_config.mamba_cache_dtype = conv
    runner.vllm_config.cache_config.mamba_ssm_cache_dtype = ssm
    result = validate_runner_model_identity(config, runner)
    assert result["mamba_cache_dtype"] == result["mamba_ssm_cache_dtype"] == "bfloat16"


@pytest.mark.parametrize("field", ["mamba_cache_dtype", "mamba_ssm_cache_dtype"])
def test_actual_non_native_state_precision_is_rejected(tmp_path, field):
    config = model_config(tmp_path)
    runner = native_runner(config)
    setattr(runner.vllm_config.cache_config, field, "float32")
    with pytest.raises(OscarReadinessError, match=f"model identity mismatch.*'{field}'"):
        validate_runner_model_identity(config, runner)


@pytest.mark.parametrize("missing", ["model_config", "cache_config"])
def test_actual_configuration_must_exist_before_loading_weights(tmp_path, missing):
    config = model_config(tmp_path)
    runner = native_runner(config, direct_model_config=False)
    delattr(runner.vllm_config, missing)
    with pytest.raises(OscarReadinessError, match="requires the native model/cache configuration"):
        validate_runner_model_identity(config, runner)


def test_identity_validation_in_a_fresh_interpreter_imports_no_torch_or_native_engine(tmp_path):
    config = model_config(tmp_path)
    source = """
import json, sys
from types import SimpleNamespace
baseline = set(sys.modules)
from oscar_ascend.integration.model_identity import validate_runner_model_identity
config = json.loads(sys.argv[1])
model = SimpleNamespace(model=config['model'], quantization=None,
                        dtype='bfloat16', revision=None, hf_overrides={})
cache = SimpleNamespace(mamba_cache_dtype='auto', mamba_ssm_cache_dtype='auto')
runner = SimpleNamespace(model_config=model,
    vllm_config=SimpleNamespace(model_config=model, cache_config=cache))
validate_runner_model_identity(config, runner)
heavy = {'torch', 'torch_npu', 'vllm', 'vllm_ascend'}
print(json.dumps(sorted(name for name in set(sys.modules) - baseline
                        if name.split('.')[0] in heavy)))
"""
    result = subprocess.run(
        [sys.executable, "-c", source, json.dumps(config)],
        cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, check=True,
    )
    assert json.loads(result.stdout) == []


class Provider:
    request_windows_enabled = True

    def __init__(self, config, events):
        self.config = config
        self.events = events

    def assert_ready(self):
        return None

    def get_impl_cls(self):
        return Provider

    def transform_kv_cache_specs(self, runner, native_specs):
        return native_specs

    def allocate_kv_cache_tensors(self, runner, config, native_allocate):
        return native_allocate(config)

    def reshape_kv_cache_tensors(self, runner, config, raw, native_reshape):
        return native_reshape(config, raw)

    def seal_request_windows(self, runner):
        self.events.append("seal")


@pytest.fixture
def patched_runner(tmp_path, monkeypatch):
    # Preserve the real plugin wrapper/require_runtime sequence. Only the
    # native class and provider are host fakes; validation is not mocked.
    monkeypatch.delenv("OSCAR_PASSIVE_TIMING_CONTROL", raising=False)
    monkeypatch.delenv("OSCAR_TIMER_CONTROL", raising=False)
    plugin.unregister()
    clear_runtime()
    config = model_config(tmp_path, quantized=True)
    events = []
    module = ModuleType("vllm_ascend.worker.model_runner_v1")

    class NPUModelRunner:
        def __init__(self):
            actual = native_runner(config)
            self.model_config = actual.model_config
            self.vllm_config = actual.vllm_config

        def get_kv_cache_spec(self):
            return {}

        def _allocate_kv_cache_tensors(self, config):
            return {}

        def _reshape_kv_cache_tensors(self, config, raw):
            return raw

        def _dummy_run(self):
            return None

        def load_model(self, marker):
            events.append("native_load")
            return marker

    module.NPUModelRunner = NPUModelRunner
    original = NPUModelRunner.load_model
    install_runtime(Provider(config, events))
    plugin._patch_runner(module)
    try:
        yield SimpleNamespace(runner=NPUModelRunner(), events=events, config=config)
    finally:
        plugin.unregister()
        clear_runtime()
        assert NPUModelRunner.load_model is original


def test_removed_quantization_fails_before_native_load_or_window_sealing(patched_runner):
    runner = patched_runner.runner
    runner.model_config.quantization = None
    with pytest.raises(OscarReadinessError, match="identity mismatch before load_model.*quantization"):
        runner.load_model("native-result")
    assert patched_runner.events == []


def test_matching_identity_preserves_native_load_result_and_window_lifecycle(patched_runner):
    assert patched_runner.runner.load_model("native-result") == "native-result"
    assert patched_runner.events == ["native_load", "seal"]
