"""Archive #26/#28-30/#84/#86/#96/#97: model selection, host JSON only."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools import install_serve, target_cli
from tools.model_config import add_model_arguments, apply_model_options


def checkpoint(tmp_path, name, *, quantized=False, architecture="Qwen3_5ForConditionalGeneration"):
    path = tmp_path / name
    path.mkdir()
    (path / "config.json").write_text(json.dumps({
        "architectures": [architecture], "model_type": "qwen3_5",
        "text_config": {"model_type": "qwen3_5_text", "num_hidden_layers": 64},
    }))
    if quantized:
        (path / "quant_model_description.json").write_text(json.dumps({"model.layers.0.weight": "W8A8"}))
    return path


def target():
    return json.loads((target_cli.ROOT / "configs/target.json").read_text())


def options(*args):
    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    return parser.parse_args(args)


@pytest.mark.parametrize("version", ["3.5", "3.8"])
@pytest.mark.parametrize("quantized", [False, True])
def test_actual_metadata_selects_all_four_models(tmp_path, version, quantized):
    model = checkpoint(tmp_path, f"Qwen{version}-27B" + ("-w8a8" if quantized else ""), quantized=quantized)
    original = target()
    before = deepcopy(original)
    selected = apply_model_options(original, options("--model", str(model)), require_model=True)
    assert original == before
    assert selected["quantization_mode"] == "auto"
    assert selected["quantization"] == ("ascend" if quantized else None)
    argv = target_cli.serve_argv(selected)
    assert ("--quantization" in argv) is quantized
    assert argv[argv.index("--dtype") + 1] == "bfloat16"
    assert argv[argv.index("--max-num-batched-tokens") + 1] == "16384"
    assert "--no-enable-prefix-caching" in argv and "--scheduler-cls" not in argv
    assert selected["speculative_config"] == original["speculative_config"]
    assert selected["devices"] == original["devices"] and selected["port"] == original["port"]


@pytest.mark.parametrize("name,quantized", [("misleading-w8a8-name", False), ("original-looking-name", True)])
def test_name_does_not_determine_quantization(tmp_path, name, quantized):
    path = checkpoint(tmp_path, name, quantized=quantized)
    selected = apply_model_options(target(), options("--model", str(path)), require_model=True)
    assert selected["quantization"] == ("ascend" if quantized else None)


def test_default_is_original_qwen38_and_no_quantization_flag():
    cfg = target()
    assert cfg["model"] == "/softwarePlatform/c00879303/Qwen3.8-27B"
    assert cfg["served_model_name"] == "qwen3.8"
    assert cfg["quantization"] is None and cfg["quantization_mode"] == "none"
    assert cfg["rotation_method"] == "qqt_sst"
    assert "--quantization" not in target_cli.serve_argv(cfg)


def test_plan_missing_model_is_deferred_and_actual_launch_is_blocked(tmp_path, monkeypatch):
    args = options("--model", str(tmp_path / "absent"))
    cfg = apply_model_options(target(), args)
    assert cfg["model_resolution"]["status"] == "deferred_model_unavailable"
    assert cfg["quantization"] == "auto"
    monkeypatch.setattr(target_cli, "_run_service", lambda *a, **k: pytest.fail("missing metadata must not launch"))
    path = tmp_path / "target.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="config.json is unavailable"):
        target_cli.main(["--config", str(path)])


def test_explicit_none_rejects_metadata_instead_of_native_auto_quantization(tmp_path):
    path = checkpoint(tmp_path, "checkpoint", quantized=True)
    with pytest.raises(ValueError, match="conflicts with quantized"):
        apply_model_options(target(), options("--model", str(path), "--quantization", "none"), require_model=True)


def test_explicit_ascend_and_served_name_override(tmp_path):
    path = checkpoint(tmp_path, "checkpoint")
    cfg = apply_model_options(target(), options("--model", str(path), "--served-name", "chosen",
                                              "--quantization", "ascend"), require_model=True)
    assert cfg["quantization"] == "ascend" and cfg["served_model_name"] == "chosen"
    assert cfg["model_resolution"]["detected_quantization"] is None


@pytest.mark.parametrize("quant", [{"quant_method": "ascend"}, {"weight_quant": "W8A8"}, {}])
def test_config_quantization_metadata_is_used(tmp_path, quant):
    path = checkpoint(tmp_path, "checkpoint")
    metadata = json.loads((path / "config.json").read_text())
    metadata["text_config"]["quantization_config"] = quant
    (path / "config.json").write_text(json.dumps(metadata))
    cfg = apply_model_options(target(), options("--model", str(path)), require_model=True)
    assert cfg["quantization"] == "ascend"


@pytest.mark.parametrize("failure,match", [
    ("architecture", "unsupported model architectures"),
    ("conflict", "conflicting checkpoint"),
    ("unsupported", "unsupported checkpoint quantization"),
    ("invalid_json", "Expecting"), ("empty_sidecar", "empty ModelSlim"),
])
def test_metadata_failures_are_not_name_based_fallbacks(tmp_path, failure, match):
    path = checkpoint(tmp_path, "Qwen3.8-27B", quantized=failure in ("conflict", "empty_sidecar"))
    config_path = path / "config.json"
    metadata = json.loads(config_path.read_text())
    if failure == "architecture":
        metadata["architectures"] = ["Qwen3_8InventedClass"]
    elif failure in ("conflict", "unsupported"):
        metadata["quantization_config"] = {"quant_method": "fp8"}
    config_path.write_text("{" if failure == "invalid_json" else json.dumps(metadata))
    if failure == "empty_sidecar":
        (path / "quant_model_description.json").write_text("{}")
    with pytest.raises(ValueError, match=match):
        apply_model_options(target(), options("--model", str(path)), require_model=True)


def test_direct_install_plan_binds_all_phases_to_one_selected_model(tmp_path, capsys):
    model = checkpoint(tmp_path, "weights", quantized=True)
    logs = tmp_path / "logs"
    assert install_serve.main(["--model", str(model), "--served-model-name", "test-model",
                               "--variant", "candidate", "--log-dir", str(logs), "--plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["vllm_argv"][:2] == ["serve", str(model)]
    assert plan["vllm_argv"][plan["vllm_argv"].index("--quantization") + 1] == "ascend"
    effective = plan["serve"][-1]
    for _, command in plan["stages"]:
        if "--config" in command:
            assert command[command.index("--config") + 1] == effective
    assert not logs.exists()


def test_target_cli_overrides_use_same_effective_config_and_cleanup(tmp_path, monkeypatch):
    model = checkpoint(tmp_path, "weights")
    seen = []
    def run(config, config_path, command, *, native):
        seen.append(config_path)
        assert json.loads(config_path.read_text()) == config
        assert config["model"] == str(model) and config["served_model_name"] == "same"
        assert command[:2] == ["serve", str(model)]
        assert "--quantization" not in command and native
        return 19
    monkeypatch.setattr(target_cli, "_run_service", run)
    assert target_cli.main(["--model", str(model), "--served-name", "same", "--native"]) == 19
    assert len(seen) == 1 and not seen[0].exists()


def test_actual_install_missing_model_records_config_failure_and_never_builds(tmp_path, monkeypatch):
    logs = tmp_path / "logs"
    monkeypatch.setattr(install_serve, "run_phase", lambda *a, **k: pytest.fail("must resolve before build"))
    with pytest.raises(ValueError, match="config.json is unavailable"):
        install_serve.main(["--model", str(tmp_path / "missing"), "--log-dir", str(logs)])
    status = json.loads((logs / "status.json").read_text())
    assert status["status"] == "failed" and status["failed_phase"] == "config"


def test_existing_non_bf16_activation_config_is_rejected():
    with pytest.raises(ValueError, match="dtype=bfloat16"):
        apply_model_options({**target(), "dtype": "float16"})
