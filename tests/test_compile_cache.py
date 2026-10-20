# Archive #28/#77/#96: no heavy imports; #34/#36: native model options unchanged.
# #85: stale source/config identity; #94/#95: persisted failure provenance.
import builtins
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

from tools import compile_cache, target_cli


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    source = tmp_path / "checkout/oscar_ascend"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("# lightweight package\n")
    (source / "plugin.py").write_text("MODE = 'BF16'\n")
    monkeypatch.setattr(compile_cache, "_versions", lambda: {
        "torch": "2.10", "torch-npu": "2.10", "vllm": "0.23", "vllm-ascend": "0.23"})
    config = {"model": "/models/qwen", "dtype": "bfloat16", "max_num_batched_tokens": 16384,
              "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"},
              "speculative_config": {"num_speculative_tokens": 3},
              "experimental_bf16_attention": True, "phase_timeout_seconds": 1800}
    return source, config, {"VLLM_CACHE_ROOT": str(tmp_path / "user-cache")}


def select(fixture, *, native=False):
    source, config, env = fixture
    return compile_cache.select_compile_cache(config, native=native, env=env, source_root=source)


def test_stable_reuse_preserves_old_artifacts_and_child_does_not_nest(fixture):
    source, config, env = fixture
    original = deepcopy(config)
    old = Path(env["VLLM_CACHE_ROOT"]) / "torch_compile_cache/previous/model"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"original cached graph")
    first = select(fixture)
    second = select((source, config, dict(env)))
    assert first["key"] == second["key"] and first["root"] == second["root"]
    assert not first["reused"] and second["reused"]
    assert Path(second["root"]).parent == Path(second["base"]) / compile_cache.SCHEMA
    assert old.read_bytes() == b"original cached graph"
    assert config == original
    identity = json.loads(Path(second["identity_file"]).read_text())
    assert identity["config"]["compilation_config"] == original["compilation_config"]
    assert identity["config"]["speculative_config"] == original["speculative_config"]


@pytest.mark.parametrize("change", ["source", "config", "native", "c16", "version", "build"])
def test_graph_relevant_changes_select_separate_partition(fixture, monkeypatch, change):
    source, config, env = fixture
    first = select(fixture)
    sentinel = Path(first["root"]) / "old-artifact"
    sentinel.write_text("keep")
    if change == "source":
        (source / "plugin.py").write_text("MODE = 'new BF16 interface'\n")
    elif change == "config":
        config["max_num_batched_tokens"] = 8192
    elif change == "c16":
        env["OSCAR_C16_BREAKDOWN_DIR"] = "/logs/run1/device"
    elif change == "version":
        monkeypatch.setattr(compile_cache, "_versions", lambda: {"torch": "2.11"})
    elif change == "build":
        build = source.parent / "build/ascendc/build_manifest.json"
        build.parent.mkdir(parents=True)
        build.write_text(json.dumps({"signature": "new-build", "build": "passed"}))
    second = select(fixture, native=change == "native")
    assert first["key"] != second["key"]
    assert Path(first["root"]).parent == Path(second["root"]).parent
    assert sentinel.read_text() == "keep"


def test_transient_paths_and_timeouts_do_not_change_identity(fixture):
    source, config, env = fixture
    env["OSCAR_C16_BREAKDOWN_DIR"] = "/logs/first/device"
    env["OSCAR_TARGET_CONFIG"] = "/logs/first/effective-target.json"
    config["profiler_config"] = {"profiler": "torch", "torch_profiler_dir": "/logs/first/profile"}
    first = select(fixture)
    env["OSCAR_C16_BREAKDOWN_DIR"] = "/logs/second/device"
    env["OSCAR_TARGET_CONFIG"] = "/logs/second/effective-target.json"
    config["phase_timeout_seconds"] = 3600
    config["service_startup_timeout_seconds"] = 950
    config["profiler_config"]["torch_profiler_dir"] = "/logs/second/profile"
    assert select(fixture)["key"] == first["key"]


def test_new_explicit_base_overrides_inherited_owned_base(fixture, tmp_path):
    first = select(fixture)
    fixture[2]["VLLM_CACHE_ROOT"] = str(tmp_path / "new-user-base")
    second = select(fixture)
    assert second["key"] == first["key"]
    assert second["base"] == str(tmp_path / "new-user-base")
    assert Path(first["root"]).is_dir()


def test_native_default_base_honors_xdg_cache_home(fixture, tmp_path):
    source, config, _ = fixture
    env = {"XDG_CACHE_HOME": str(tmp_path / "xdg")}
    result = select((source, config, env), native=True)
    assert result["base"] == str(tmp_path / "xdg/vllm")
    assert result["mode"] == "native"


@pytest.mark.parametrize("field", ["cache_dir", "local_cache_dir"])
def test_explicit_inner_cache_override_is_not_silently_ignored(fixture, field):
    fixture[1]["compilation_config"][field] = "/user/specific-cache"
    with pytest.raises(ValueError, match=f"explicit compilation_config.{field}"):
        select(fixture)
    assert not Path(fixture[2]["VLLM_CACHE_ROOT"]).exists()


def test_receipt_corruption_is_not_silently_overwritten(fixture):
    first = select(fixture)
    path = Path(first["identity_file"])
    path.write_text('{"tampered": true}')
    with pytest.raises(RuntimeError, match="identity mismatch"):
        select(fixture)
    assert json.loads(path.read_text()) == {"tampered": True}


def test_explicit_missing_build_manifest_is_observable(fixture):
    fixture[2]["OSCAR_BUILD_MANIFEST"] = "/missing/build_manifest.json"
    with pytest.raises(ValueError, match="explicit OSCAR_BUILD_MANIFEST is unavailable"):
        select(fixture)


def test_build_installation_paths_do_not_randomize_signature(fixture):
    source, config, env = fixture
    path = source.parent / "build/ascendc/build_manifest.json"
    path.parent.mkdir(parents=True)
    def manifest(prefix):
        return {"build": "passed", "signature": "fixed", "extension": prefix + "/ext.so",
                "kernel_library": prefix + "/kernel.so", "sha256": {prefix + "/ext.so": "abc",
                prefix + "/kernel.so": "def"}}
    path.write_text(json.dumps(manifest("/build1")))
    first = select(fixture)
    path.write_text(json.dumps(manifest("/build2")))
    assert select(fixture)["key"] == first["key"]
    changed = manifest("/build2")
    changed["sha256"]["/build2/ext.so"] = "new-binary"
    path.write_text(json.dumps(changed))
    assert select(fixture)["key"] != first["key"]


def test_module_import_and_identity_resolution_do_not_import_device_stack(tmp_path):
    # A fresh interpreter import hook tests the real source package resolution;
    # no monkeypatch of _package_root and no NPU availability assumption.
    script = """
import builtins, os
old = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torch_npu', 'vllm', 'vllm_ascend'}:
        raise AssertionError('heavy import: ' + name)
    return old(name, *args, **kwargs)
builtins.__import__ = guarded
from tools.compile_cache import select_compile_cache
receipt = select_compile_cache({'model': '/unavailable/plan/model'}, native=False,
                              env={'VLLM_CACHE_ROOT': os.environ['TEST_CACHE_ROOT']})
assert receipt['mode'] == 'oscar'
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
                            env={**os.environ, "TEST_CACHE_ROOT": str(tmp_path / "cache")},
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_target_cli_selects_before_native_import_and_keeps_argv(monkeypatch, tmp_path):
    config = json.loads((target_cli.ROOT / "configs/target.json").read_text())
    command = target_cli.serve_argv(config)
    observed = []
    def select_fake(config, *, native, env):
        assert native
        env["VLLM_CACHE_ROOT"] = str(tmp_path / "bound")
        observed.append("selected")
        return {"key": "k", "root": env["VLLM_CACHE_ROOT"]}
    monkeypatch.setattr(compile_cache, "select_compile_cache", select_fake)
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for name in ("OSCAR_PASSIVE_TIMING_CONTROL", "OSCAR_TIMER_CONTROL"):
        os.environ.pop(name, None)
    torch = types.ModuleType("torch")
    torch.set_num_threads = lambda n: observed.append("threads")
    ascend = types.ModuleType("vllm_ascend")
    ascend.ops = types.ModuleType("vllm_ascend.ops")
    main = types.ModuleType("vllm.entrypoints.cli.main")
    def native_main():
        assert os.environ["VLLM_CACHE_ROOT"] == str(tmp_path / "bound")
        assert sys.argv == ["vllm", *command]
        observed.append("main")
    main.main = native_main
    for name, module in {"torch": torch, "vllm_ascend": ascend,
            "vllm_ascend.ops": ascend.ops, "vllm.entrypoints.cli.main": main}.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "argv", [])
    assert target_cli._run_service(config, tmp_path / "config.json", command, native=True) == 0
    assert observed == ["selected", "threads", "main"]
