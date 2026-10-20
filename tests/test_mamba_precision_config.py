"""Archive #86/#97/#127/#128: supported native GDN state precision at launch.

Native cache.py MambaDType and MambaStateDtypeCalculator define explicit auto:
SSM follows conv. The 910B4 chunk operator and frozen launch use BF16 state;
the HF model's mamba_ssm_dtype must not silently change native cache storage.
Host JSON, launch plans and a fake native LLM only; no NPU/quality acceptance.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tools import deploy, install_serve, observe_serve, target_cli
from tools.model_config import (
    add_model_arguments,
    apply_model_options,
    resolve_mamba_cache_dtypes,
)
from tools.serving_variants import variant_config, variant_features


_OMITTED = object()


def checkpoint(tmp_path, name="model", *, state="float32", quantized=False, nested=True):
    path = tmp_path / name
    path.mkdir()
    text = {
        "model_type": "qwen3_5_text",
        "num_hidden_layers": 4,
        "full_attention_interval": 4,
        "head_dim": 256,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
    }
    if state is not _OMITTED:
        text["mamba_ssm_dtype"] = state
    metadata = ({"model_type": "qwen3_5", "text_config": text} if nested else text)
    metadata["architectures"] = ["Qwen3_5ForConditionalGeneration"]
    (path / "config.json").write_text(json.dumps(metadata))
    if quantized:
        (path / "quant_model_description.json").write_text(
            json.dumps({"model.layers.0.weight": "W8A8"}))
    return path


def config_for(model, **overrides):
    config = json.loads((install_serve.ROOT / "configs/target.json").read_text())
    config.update(model=str(model), quantization_mode="auto")
    config.update(overrides)
    return config


def model_options(*args):
    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    return parser.parse_args(args)


def argv_option(argv, name):
    return argv[argv.index("--" + name.replace("_", "-")) + 1]


@pytest.mark.parametrize("quantized", [False, True])
def test_checkpoint_ssm_precision_is_independent_of_weight_quantization(tmp_path, quantized):
    model = checkpoint(tmp_path, quantized=quantized)
    config = config_for(model)
    before = deepcopy(config)
    selected = apply_model_options(config, require_model=True)
    expected = {"mamba_cache_dtype": "bfloat16", "mamba_ssm_cache_dtype": "bfloat16"}
    assert selected["model_resolution"]["mamba_cache_dtypes"] == expected
    assert resolve_mamba_cache_dtypes(selected, require_model=True) == expected
    assert selected["quantization"] == ("ascend" if quantized else None)
    assert config == before
    assert "mamba_cache_dtype" not in selected
    assert "mamba_ssm_cache_dtype" not in selected
    argv = target_cli.serve_argv(selected)
    assert argv_option(argv, "dtype") == "bfloat16"
    assert argv_option(argv, "mamba_cache_dtype") == "bfloat16"
    assert argv_option(argv, "mamba_ssm_cache_dtype") == "bfloat16"


@pytest.mark.parametrize("overrides,expected", [
    ({"mamba_ssm_cache_dtype": "bfloat16"}, ("bfloat16", "bfloat16")),
    ({"mamba_ssm_cache_dtype": "auto"}, ("bfloat16", "bfloat16")),
    ({"mamba_cache_dtype": "bfloat16", "mamba_ssm_cache_dtype": "auto"},
     ("bfloat16", "bfloat16")),
    ({"mamba_cache_dtype": "auto"}, ("bfloat16", "bfloat16")),
])
def test_explicit_state_override_and_native_auto_semantics(tmp_path, overrides, expected):
    config = config_for(checkpoint(tmp_path), **overrides)
    resolved = resolve_mamba_cache_dtypes(config, require_model=True)
    assert resolved == dict(zip(("mamba_cache_dtype", "mamba_ssm_cache_dtype"), expected))
    selected = apply_model_options(config, require_model=True)
    assert all(selected[key] == value for key, value in overrides.items())


@pytest.mark.parametrize("state", [_OMITTED, "auto", "float16", "float32", "bfloat16"])
def test_hf_model_state_declaration_does_not_override_native_cache_storage(tmp_path, state):
    config = config_for(checkpoint(tmp_path, state=state))
    assert resolve_mamba_cache_dtypes(config)["mamba_ssm_cache_dtype"] == "bfloat16"


def test_missing_model_is_deferred_only_for_read_only_plans(tmp_path):
    config = config_for(tmp_path / "missing")
    assert resolve_mamba_cache_dtypes(config) == {
        "mamba_cache_dtype": "bfloat16", "mamba_ssm_cache_dtype": "bfloat16"}
    with pytest.raises(ValueError, match="config.json is unavailable"):
        resolve_mamba_cache_dtypes(config, require_model=True)
    selected = apply_model_options(config)
    assert selected["model_resolution"]["status"] == "deferred_model_unavailable"
    assert argv_option(target_cli.serve_argv(selected), "mamba_ssm_cache_dtype") == "bfloat16"
    assert resolve_mamba_cache_dtypes({})["mamba_ssm_cache_dtype"] == "bfloat16"


@pytest.mark.parametrize("nested", [False, True])
def test_hf_state_override_does_not_change_native_cache_precision(tmp_path, nested):
    model = checkpoint(tmp_path, nested=nested)
    text_override = {"mamba_ssm_dtype": "float16"}
    overrides = {"text_config": text_override} if nested else text_override
    config = config_for(model, hf_overrides=overrides)
    before = (model / "config.json").read_bytes()
    assert resolve_mamba_cache_dtypes(config)["mamba_ssm_cache_dtype"] == "bfloat16"
    config["mamba_ssm_cache_dtype"] = "auto"
    assert resolve_mamba_cache_dtypes(config)["mamba_ssm_cache_dtype"] == "bfloat16"
    assert (model / "config.json").read_bytes() == before


def test_nested_model_does_not_apply_top_level_text_dtype_override(tmp_path):
    config = config_for(checkpoint(tmp_path), hf_overrides={"mamba_ssm_dtype": "float16"})
    assert resolve_mamba_cache_dtypes(config)["mamba_ssm_cache_dtype"] == "bfloat16"


@pytest.mark.parametrize("key", ["mamba_cache_dtype", "mamba_ssm_cache_dtype"])
@pytest.mark.parametrize("invalid", [None, True, 32, "float64", "bf16", "float16", "float32"])
def test_invalid_explicit_state_precision_fails_closed(tmp_path, key, invalid):
    config = config_for(checkpoint(tmp_path), **{key: invalid})
    with pytest.raises(ValueError, match=key):
        apply_model_options(config, require_model=True)


def test_explicit_fp32_state_is_rejected_before_install_or_model_execution(tmp_path, monkeypatch):
    config = config_for(checkpoint(tmp_path), mamba_ssm_cache_dtype="float32")
    config_path = tmp_path / "unsupported.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(install_serve, "run_phase",
                        lambda *_a, **_k: pytest.fail("unsupported state must fail before build"))
    log_dir = tmp_path / "failure"
    with pytest.raises(ValueError, match="mamba_ssm_cache_dtype"):
        install_serve.main(["--config", str(config_path), "--log-dir", str(log_dir)])
    assert json.loads((log_dir / "status.json").read_text())["failed_phase"] == "config"


def test_pinned_native_chunk_tiling_rejects_fp32_state_on_non_950_soc():
    """Source evidence only: recurrent FP32 support is not a prefill contract."""
    source = (install_serve.ROOT / "references/vllm-ascend/csrc/attention/"
              "chunk_gated_delta_rule/op_host/chunk_gated_delta_rule_tiling.cpp").read_text()
    method = source.split("ChunkGatedDeltaRuleTiling::AnalyzeDtype()", 1)[1]
    method = method.split("ChunkGatedDeltaRuleTiling::CheckDim(", 1)[0]
    rejection = re.search(
        r"if\s*\(\s*stateDtype\s*==\s*ge::DT_FLOAT\s*&&\s*"
        r"socVersion_\s*!=\s*platform_ascendc::SocVersion::ASCEND950\s*\)"
        r"\s*\{([^}]+)\}", method)
    assert rejection is not None, "the native GDN state contract changed; re-audit launch precision"
    assert "return ge::GRAPH_FAILED;" in rejection.group(1)
    config = config_for("model-unavailable-for-host-plan")
    assert config["soc_version"] == "ascend910b4"
    with pytest.raises(ValueError, match="mamba_ssm_cache_dtype"):
        resolve_mamba_cache_dtypes({**config, "mamba_ssm_cache_dtype": "float32"})


def test_changing_model_recomputes_derived_state_dtype_but_preserves_explicit_choice(tmp_path):
    first = checkpoint(tmp_path, "first", state="float32")
    second = checkpoint(tmp_path, "second", state=_OMITTED, quantized=True)
    selected = apply_model_options(config_for(first), require_model=True)
    # A stale derived report is descriptive, never a launch override.
    selected["model_resolution"]["mamba_cache_dtypes"]["mamba_ssm_cache_dtype"] = "float32"
    changed = apply_model_options(selected, model_options("--model", str(second)), require_model=True)
    assert changed["model_resolution"]["mamba_cache_dtypes"]["mamba_ssm_cache_dtype"] == "bfloat16"
    assert changed["quantization"] == "ascend"
    assert selected["model_resolution"]["mamba_cache_dtypes"]["mamba_ssm_cache_dtype"] == "float32"
    explicit = apply_model_options(
        {**selected, "mamba_ssm_cache_dtype": "auto"},
        model_options("--model", str(second)), require_model=True)
    assert explicit["model_resolution"]["mamba_cache_dtypes"]["mamba_ssm_cache_dtype"] == "bfloat16"
    assert explicit["mamba_ssm_cache_dtype"] == "auto"


@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("entrypoint", [install_serve, observe_serve, deploy])
def test_all_one_key_plans_use_model_state_precision_without_changing_launch_policy(
        tmp_path, capsys, prefix, entrypoint):
    config = config_for(checkpoint(tmp_path), enable_prefix_caching=prefix)
    config_path = tmp_path / "target.json"
    config_path.write_text(json.dumps(config))
    log_dir = tmp_path / "plan-must-not-create-logs"
    before = config_path.read_bytes()
    assert entrypoint.main(["--config", str(config_path), "--variant", "candidate",
                            "--plan", "--log-dir", str(log_dir)]) == 0
    plan = json.loads(capsys.readouterr().out)
    argv = plan["vllm_argv"]
    assert argv_option(argv, "mamba_cache_dtype") == "bfloat16"
    assert argv_option(argv, "mamba_ssm_cache_dtype") == "bfloat16"
    assert plan["model_resolution"]["mamba_cache_dtypes"]["mamba_ssm_cache_dtype"] == "bfloat16"
    assert argv_option(argv, "max_num_batched_tokens") == "16384"
    assert argv_option(argv, "tensor_parallel_size") == "4"
    assert argv_option(argv, "port") == "8989"
    assert json.loads(argv[argv.index("--speculative_config") + 1]) == config["speculative_config"]
    assert json.loads(argv_option(argv, "compilation_config")) == config["compilation_config"]
    assert ("--enable-prefix-caching" if prefix else "--no-enable-prefix-caching") in argv
    assert "--scheduler-cls" not in argv and "--enforce-eager" not in argv
    assert not log_dir.exists() and config_path.read_bytes() == before


@pytest.mark.parametrize("prefix", [False, True])
def test_native_and_candidate_keep_identical_model_state_precision(tmp_path, prefix):
    config = config_for(checkpoint(tmp_path), enable_prefix_caching=prefix)
    candidate = variant_config(config, "candidate")
    native = variant_config(candidate, "native")
    assert not any(variant_features(native).values())
    assert target_cli.serve_argv(candidate) == target_cli.serve_argv(native)
    assert argv_option(target_cli.serve_argv(native), "mamba_ssm_cache_dtype") == "bfloat16"


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("raw_quantization", [None, "auto"])
def test_standalone_rotation_entry_resolves_auto_before_profile_without_rewriting_input(
        tmp_path, monkeypatch, capsys, quantized, raw_quantization):
    from oscar_ascend.rotations import calibration_cache_identity, checkpoint_identity
    from tools import prepare_rotations

    model = checkpoint(tmp_path, quantized=quantized)
    config = config_for(model, quantization=raw_quantization)
    config_path = tmp_path / "raw-target.json"
    config_path.write_text(json.dumps(config))
    before = config_path.read_bytes()
    fingerprint = checkpoint_identity(model)[2]
    expected_quantization = "ascend" if quantized else None
    seen = []
    monkeypatch.setattr(prepare_rotations.os, "environ", dict(prepare_rotations.os.environ))

    def prepare_double(selected, *, config_path, output, import_pair):
        # This is the same raw file that the isolated collect() entry parses.
        capture_config = apply_model_options(
            json.loads(config_path.read_text()), require_model=True)
        assert selected["quantization"] == capture_config["quantization"] == expected_quantization
        assert calibration_cache_identity(selected, fingerprint) == calibration_cache_identity(
            capture_config, fingerprint)
        assert selected["model_resolution"]["mamba_cache_dtypes"] == {
            "mamba_cache_dtype": "bfloat16", "mamba_ssm_cache_dtype": "bfloat16"}
        assert output is None and import_pair is None
        seen.append(selected)
        return {"host_only": True}

    monkeypatch.setattr(prepare_rotations, "prepare", prepare_double)
    assert prepare_rotations.main(["--config", str(config_path)]) == 0
    assert json.loads(capsys.readouterr().out) == {"host_only": True}
    assert len(seen) == 1 and config_path.read_bytes() == before


@pytest.mark.parametrize("explicit", [None, "bfloat16", "auto"])
def test_real_capture_entrypoint_and_service_share_state_kwargs(tmp_path, monkeypatch, explicit):
    """Execute the collector orchestration with host LLM/tokenizer doubles."""
    from tools import calibrate_model_rotations as calibration

    # collect normally runs in a disposable interpreter. Keep its calibration
    # environment changes local to this host double as well.
    monkeypatch.setattr(calibration.os, "environ", dict(calibration.os.environ))
    config = config_for(checkpoint(tmp_path, quantized=True))
    if explicit is not None:
        config["mamba_ssm_cache_dtype"] = explicit
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('"promptA"\n"promptB"\n')
    config["rotation_calibration_prompts"] = str(prompts)
    path = tmp_path / "target.json"
    path.write_text(json.dumps(config))
    service_argv = target_cli.serve_argv(config)
    tokenizer_module = ModuleType("transformers")
    tokenizer_module.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *_a, **_kw: SimpleNamespace(
            apply_chat_template=lambda *_a, **_kw: [7, 8, 9]))
    monkeypatch.setitem(sys.modules, "transformers", tokenizer_module)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    options, spec = {}, {}

    class NativeLLMDouble:
        def __init__(self, **kwargs):
            options.update(kwargs)
            self.llm_engine = SimpleNamespace(
                engine_core=SimpleNamespace(shutdown=lambda **_kw: None))

        def collective_rpc(self, name, **kwargs):
            if name.endswith("_start"):
                spec.update(kwargs["args"][0])
                return [{"rank": rank, "layers": spec["layer_ids"], "state": "armed_after_warmup"}
                        for rank in range(spec["tp_size"])]
            if name.endswith("_finish"):
                result = []
                for rank in range(spec["tp_size"]):
                    output = Path(spec["output_dir"]) / f"rank{rank}.pt"
                    output.write_bytes(b"host-double-not-a-model-sample")
                    result.append({"rank": rank, "layers": spec["layer_ids"],
                                   "tokens": spec["token_count"], "status": "complete",
                                   "path": str(output)})
                return result
            return [{"status": "not_armed"}]

        def generate(self, *_args, **_kwargs):
            return [SimpleNamespace(finished=True)]

    vllm = ModuleType("vllm")
    vllm.LLM = NativeLLMDouble
    vllm.SamplingParams = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    output_dir = tmp_path / "capture"
    calibration.collect(path, output_dir)
    report = json.loads((output_dir / "capture-manifest.json").read_text())
    for key, expected in (("mamba_cache_dtype", "bfloat16"),
                          ("mamba_ssm_cache_dtype", "bfloat16")):
        assert options[key] == argv_option(service_argv, key) == expected
        assert report["mamba_cache_dtypes"][key] == expected
        assert report["calibration_profile"]["model_execution"][key] == expected
    assert options["quantization"] == "ascend"
    assert report["weights_quantization"] == "ascend"
    assert options["dtype"] == "bfloat16" and "speculative_config" not in options
    assert config["speculative_config"]["num_speculative_tokens"] == 3
    assert not options["enable_prefix_caching"] and options["enforce_eager"]
