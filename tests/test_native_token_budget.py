"""Archive #34/#36/#94/#125/#148: omitted CLI budget, native-sized arena, same bundle.

Host contracts only. Native initialize_model provides get_current_vllm_config;
these tests verify the value reaches allocation before serving, not NPU speed.
"""
import json
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tools import install_serve
from tools.phase import PhaseResult
from tools.serving_variants import variant_config, variant_features
from tools.target_cli import serve_argv
from oscar_ascend.integration import runtime_api
from oscar_ascend.integration.native_token_budget import NativeTokenBudgetRuntime


def target():
    return json.loads((install_serve.ROOT / "configs/target.json").read_text())


def selected():
    return variant_config({**target(), "use_native_token_budget": True}, "candidate")


def test_native_mode_omits_argument_preserves_bundle_and_is_idempotent():
    original = target()
    legacy = variant_config(original, "candidate")
    config = variant_config({**legacy, "use_native_token_budget": True}, "candidate")
    assert original["max_num_batched_tokens"] == 16384
    assert legacy["max_num_batched_tokens"] == 32768
    assert "max_num_batched_tokens" not in config
    assert "scheduler_cls" not in config and "_oscar_whole_prefill_base" not in config
    assert variant_config(config, "candidate") == config
    assert variant_features(config) == {
        **variant_features(legacy), "whole_prefill": False}
    args = serve_argv(config)
    assert "--max-num-batched-tokens" not in args
    assert "--scheduler-cls" not in args
    for flag in ("--async-scheduling", "--speculative_config", "--compilation-config"):
        assert flag in args
    assert serve_argv(legacy)[serve_argv(legacy).index("--max-num-batched-tokens") + 1] == "32768"


@pytest.mark.parametrize("override", [
    {"max_num_batched_tokens": 2048}, {"scheduler_cls": "user.Scheduler"},
    {"experimental_whole_prefill": True}, {"use_native_token_budget": "true"},
])
def test_conflicting_cli_config_fails_instead_of_hiding_an_override(override):
    with pytest.raises(ValueError, match="native token budget|explicit boolean"):
        serve_argv({**selected(), **override})


@pytest.mark.parametrize("rear", [False, True])
def test_every_install_phase_and_service_receive_budgetless_config(monkeypatch, tmp_path, rear):
    path = tmp_path / "target.json"
    path.write_text(json.dumps(target()))
    before = path.read_bytes()
    logs = tmp_path / "run"
    seen = []

    def check(env):
        config = json.loads(open(env["OSCAR_TARGET_CONFIG"]).read())
        assert "max_num_batched_tokens" not in config
        assert config["use_native_token_budget"] is True
        assert config["experimental_decode_bundle"] is True
        assert config["experimental_whole_prefill"] is False
        assert config["devices"] == ([4, 5, 6, 7] if rear else [0, 1, 2, 3])
        assert config["port"] == (7878 if rear else 8989)
        return config

    def phase(name, command, *, env, **kwargs):
        check(env)
        if "--config" in command:
            assert command[command.index("--config") + 1] == env["OSCAR_TARGET_CONFIG"]
        seen.append(name)
        return PhaseResult(name, command, 0, 0.0, False, False, str(logs / name), True)

    class Started(Exception):
        pass

    def launch(executable, command, env):
        config = check(env)
        assert seen == list(install_serve.BUILD_PHASES)
        assert command[-1] == env["OSCAR_TARGET_CONFIG"]
        assert "--max-num-batched-tokens" not in serve_argv(config)
        raise Started

    monkeypatch.setattr(install_serve, "run_phase", phase)
    monkeypatch.setattr(install_serve.os, "execvpe", launch)
    with pytest.raises(Started):
        install_serve.main(["--config", str(path), "--log-dir", str(logs),
                            "--variant", "candidate", "--native-token-budget"] +
                           (["--rear-cards"] if rear else []))
    assert path.read_bytes() == before


def native_config(budget, captures):
    return SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=budget),
                           compilation_config=SimpleNamespace(cudagraph_capture_sizes=captures))


@pytest.mark.parametrize("budget,captures,expected", [
    (2048, [128, 512], 2048), (4096, [512], 4096), (1024, [2048], 2048),
])
def test_native_budget_reaches_pre_profile_workspace_without_hardcoding(
        monkeypatch, budget, captures, expected):
    import oscar_ascend.runtime as runtime
    config = selected()
    snapshot = json.dumps(config, sort_keys=True)
    native = native_config(budget, captures)
    module = ModuleType("vllm.config")
    module.get_current_vllm_config = lambda: native
    monkeypatch.setitem(sys.modules, "vllm.config", module)
    provider = NativeTokenBudgetRuntime(config)
    assert not provider.workspaces and "max_num_batched_tokens" not in provider.config
    monkeypatch.setattr(provider, "_device_cube_cores", lambda _: 20)
    monkeypatch.setattr(provider, "_prepare_rotations", lambda *_: None)
    monkeypatch.setattr(runtime, "GraphWorkspace",
                        lambda geometry, device: SimpleNamespace(geometry=geometry, device=device))
    workspace = provider.ensure_workspace(6, 1, 256, "npu:0")
    assert workspace.geometry.tokens == expected
    assert provider.config["max_num_batched_tokens"] == budget
    assert provider.ensure_workspace(6, 1, 256, "npu:0") is workspace
    assert json.dumps(config, sort_keys=True) == snapshot
    assert native.scheduler_config.max_num_batched_tokens == budget
    assert native.compilation_config.cudagraph_capture_sizes == captures
    provider._bind_native_capacity(native_config(budget, []))  # eager MTP has fewer captures
    with pytest.raises(runtime_api.OscarReadinessError, match="changed after"):
        provider._bind_native_capacity(native_config(budget + 128, captures))
    with pytest.raises(runtime_api.OscarReadinessError, match="changed after"):
        provider._bind_native_capacity(native_config(budget, [expected + 128]))


@pytest.mark.parametrize("budget", [None, 0, -1, "2048", True])
def test_invalid_native_budget_is_not_replaced_with_a_guessed_default(budget):
    with pytest.raises(runtime_api.OscarReadinessError, match="positive integers"):
        NativeTokenBudgetRuntime(selected())._bind_native_capacity(native_config(budget, []))


def test_worker_factory_uses_adapter_only_for_explicit_native_mode(monkeypatch, tmp_path):
    from oscar_ascend.runtime import AscendRuntimeProvider
    path = tmp_path / "target.json"
    monkeypatch.setenv("OSCAR_TARGET_CONFIG", str(path))
    for config, expected in ((selected(), NativeTokenBudgetRuntime),
                             (target(), AscendRuntimeProvider)):
        path.write_text(json.dumps(config))
        provider = runtime_api._create_default_runtime()
        assert type(provider) is expected
        assert provider.workspaces == {}
        result = subprocess.run([sys.executable, "-c",
            "from oscar_ascend.integration.runtime_api import _create_default_runtime; "
            "print(type(_create_default_runtime()).__name__)"],
            check=True, capture_output=True, text=True)
        assert result.stdout.strip() == expected.__name__


def test_candidate_q4_q1_and_small_prefill_keep_the_same_operators():
    from oscar_ascend.ops.cv_dispatch import select_cv_op
    config = selected()
    def choose(tokens, query, draft=False):
        return select_cv_op(16, 6, 1, tokens, query, q1_draft=draft,
                            fast_unpack=config["experimental_fast_unpack"],
                            mixed_cv=config["experimental_mixed_cv"],
                            striped_cache=config["experimental_striped_cache"],
                            decode_bundle=config["experimental_decode_bundle"])
    assert choose(128, 4) == "attention_cv_bundle_decode_out"
    assert choose(128, 1, True) == "attention_cv_bundle_q1_out"
    assert choose(2048, 2048) == "attention_cv_window_range_cluster4_out"
