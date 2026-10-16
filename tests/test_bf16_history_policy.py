"""Archive #148/#153/#154/#77: default precision is not acceptance.

User 2026-10-16 requires all-source BF16; incompatible explicit requests must
fail before installation. Host tests never claim NPU or numerical acceptance.
"""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from oscar_ascend.bf16_history_policy import (
    BF16_HISTORY_FLAG, BF16_HISTORY_DEPENDENCIES, validate_bf16_history,
)
from oscar_ascend.integration.runtime_api import OscarReadinessError
from oscar_ascend.runtime import AscendRuntimeProvider
from tools import deploy, install_serve, observe_serve, target_cli
from tools.serving_variants import research_variant_config, variant_config, variant_features


ROOT = Path(__file__).resolve().parents[1]
UNAVAILABLE = "retired FP32 production attention"


def target():
    return json.loads((ROOT / "configs/target.json").read_text())


def research_target():
    return research_variant_config({**target(), BF16_HISTORY_FLAG: True}, "candidate")


def write_target(tmp_path, value):
    path = tmp_path / "target.json"
    path.write_text(json.dumps({**research_target(), BF16_HISTORY_FLAG: value}))
    return path


@pytest.mark.parametrize("variant", [None, "candidate"])
def test_default_and_explicit_true_use_the_only_production_precision(variant):
    original = target()
    ordinary = variant_config(original, variant)
    explicit = variant_config({**original, BF16_HISTORY_FLAG: True}, variant)
    assert ordinary == explicit
    assert ordinary[BF16_HISTORY_FLAG] is True
    assert ordinary["attention_compute_dtype"] == "bfloat16"
    assert all(ordinary[key] is True for key in BF16_HISTORY_DEPENDENCIES)
    assert BF16_HISTORY_FLAG not in original
    with pytest.raises(ValueError, match=UNAVAILABLE):
        variant_config({**original, BF16_HISTORY_FLAG: False}, variant)


def test_retired_baseline_is_not_silently_reinterpreted_as_native():
    with pytest.raises(ValueError, match="baseline FP32 OSCAR production attention is retired"):
        variant_config(target(), "baseline")
    assert research_variant_config(target(), "baseline")["experimental_striped_cache"] is False


@pytest.mark.parametrize("variant", [None, "candidate", "baseline", "native"])
@pytest.mark.parametrize("value", [None, 0, 1, "false", "true", [], {}])
def test_type_errors_are_rejected_before_variant_normalization(variant, value):
    with pytest.raises(ValueError, match="must be an explicit boolean"):
        variant_config({**target(), BF16_HISTORY_FLAG: value}, variant)


def test_native_clears_boolean_alias_without_mutating_input():
    original = {**target(), BF16_HISTORY_FLAG: True}
    selected = variant_config(original, "native")
    assert selected[BF16_HISTORY_FLAG] is False
    assert original[BF16_HISTORY_FLAG] is True
    assert target_cli.target_env(selected, base={}, native=True)["OSCAR_ENABLED"] == "0"
    assert "--no-enable-prefix-caching" in target_cli.serve_argv(selected, native=True)
    with pytest.raises(OscarReadinessError, match=UNAVAILABLE):
        AscendRuntimeProvider(selected)


@pytest.mark.parametrize("value", [False, "false", 1])
def test_runtime_constructor_cannot_bypass_policy(value):
    with pytest.raises(OscarReadinessError, match=BF16_HISTORY_FLAG):
        AscendRuntimeProvider({**target(), BF16_HISTORY_FLAG: value})


def test_runtime_uses_the_same_default_bundle_as_both_entrypoints():
    provider = AscendRuntimeProvider(target())
    assert provider.config == variant_config(target(), "candidate")
    assert provider.ops is None  # Configuration is not device acceptance.


def test_runtime_file_configuration_cannot_bypass_policy(tmp_path, monkeypatch):
    path = write_target(tmp_path, False)
    monkeypatch.setenv("OSCAR_TARGET_CONFIG", str(path))
    with pytest.raises(OscarReadinessError, match=UNAVAILABLE):
        AscendRuntimeProvider()


@pytest.mark.parametrize("prepare", [target_cli.serve_argv, target_cli.target_env])
def test_direct_command_and_environment_reject_retired_route_before_device_loading(prepare):
    with pytest.raises(ValueError, match=UNAVAILABLE):
        prepare({**research_target(), BF16_HISTORY_FLAG: False})


@pytest.mark.parametrize("dtype", ["float32", "fp32", "bf16", None, 1, True])
def test_precision_is_an_explicit_supported_dtype(dtype):
    with pytest.raises(ValueError, match="attention_compute_dtype"):
        variant_config({**target(), "attention_compute_dtype": dtype}, "candidate")


@pytest.mark.parametrize("flag", BF16_HISTORY_DEPENDENCIES)
@pytest.mark.parametrize("value", [False, 1, "true"])
def test_research_route_requires_each_strict_boolean_dependency(flag, value):
    config = research_target()
    config[flag] = value
    with pytest.raises(ValueError, match=flag):
        validate_bf16_history(config)


@pytest.mark.parametrize("updates,geometry,expected", [
    ({"experimental_request_windows": True}, {}, "physical-page snapshots"),
    ({"experimental_request_windows": 0}, {}, "explicit boolean"),
    ({}, {"head_dim": 128}, "head_dim=256"),
    ({}, {"head_dim": 256.0}, "head_dim=256"),
    ({}, {"cache_format": "striped_request_window_v1"}, "striped_v1"),
    ({}, {"cache_format": "canonical_v1"}, "striped_v1"),
])
def test_research_shape_and_storage_boundaries_are_explicit(updates, geometry, expected):
    with pytest.raises(ValueError, match=expected):
        validate_bf16_history({**research_target(), **updates}, **geometry)


@pytest.mark.parametrize("entry", ["assert_ready", "layer_state", "ensure_workspace"])
def test_mutating_an_existing_provider_cannot_restore_fp32_math(entry):
    provider = AscendRuntimeProvider(target())
    provider.config[BF16_HISTORY_FLAG] = False
    with pytest.raises(OscarReadinessError, match=UNAVAILABLE):
        if entry == "layer_state":
            provider.layer_state("model.layers.0.self_attn.attn")
        elif entry == "ensure_workspace":
            provider.ensure_workspace(6, 1, 256, "npu:0")
        else:
            provider.assert_ready()


def test_target_cli_native_normalizes_true_but_still_rejects_wrong_type(tmp_path, monkeypatch, capsys):
    path = write_target(tmp_path, True)
    monkeypatch.setattr(sys, "argv", ["target_cli", "--config", str(path), "--native", "--print-command"])
    assert target_cli.main() == 0
    command = capsys.readouterr().out
    assert command.startswith("vllm serve ")
    assert json.loads(path.read_text())[BF16_HISTORY_FLAG] is True
    path.write_text(json.dumps({**target(), BF16_HISTORY_FLAG: "false"}))
    with pytest.raises(ValueError, match="must be an explicit boolean"):
        target_cli.main()


@pytest.mark.parametrize("entry", ["install", "observe", "deploy"])
def test_retired_config_stops_before_any_installation_phase(entry, tmp_path, monkeypatch):
    path = write_target(tmp_path, False)
    logs = tmp_path / "logs"

    def forbidden_phase(*args, **kwargs):
        pytest.fail("an installation/probe phase ran before the config was rejected")

    module = {"install": install_serve, "observe": observe_serve, "deploy": deploy}[entry]
    monkeypatch.setattr(module, "run_phase", forbidden_phase)
    args = ["--config", str(path), "--log-dir", str(logs)]
    if entry == "observe":
        assert observe_serve.main(args + ["--variant", "candidate"]) == 1
    elif entry == "install":
        with pytest.raises(ValueError, match=UNAVAILABLE):
            install_serve.main(args + ["--variant", "candidate"])
    else:
        monkeypatch.setattr(sys, "argv", ["deploy", *args])
        with pytest.raises(ValueError, match=UNAVAILABLE):
            deploy.main()
    status = json.loads((logs / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["failed_phase"] == "config"
    assert status["phases"] == []
    assert UNAVAILABLE in status["error"]
    assert not (logs / "effective-target.json").exists()


def test_policy_import_is_lightweight_and_has_no_device_dependency():
    result = subprocess.run([sys.executable, "-c", (
        "import sys; before=set(sys.modules); "
        "from oscar_ascend.bf16_history_policy import validate_bf16_history; "
        "assert validate_bf16_history({}) is False; "
        "assert not {'torch', 'torch_npu', 'vllm', 'vllm_ascend'} & (set(sys.modules)-before)"
    )], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
