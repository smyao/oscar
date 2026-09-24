"""Archive #27/#28/#31–36/#74–78: plugin load, routing and restoration.

Fake host classes verify Python binding and import contracts only. This is
not native worker, NPU, graph replay, or complete service acceptance.
"""

import importlib
import importlib.abc
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

from oscar_ascend import plugin
from oscar_ascend.integration.runtime_api import (
    OscarReadinessError, clear_runtime, install_runtime, install_runtime_factory, require_runtime)


class Provider:
    def assert_ready(self):
        return None

    def get_impl_cls(self):
        return Provider

    def transform_kv_cache_specs(self, runner, native_specs):
        return {"oscar": native_specs}

    def allocate_kv_cache_tensors(self, runner, config, native_allocate):
        return {"oscar": native_allocate(config)}

    def reshape_kv_cache_tensors(self, runner, config, raw, native_reshape):
        return {"oscar": native_reshape(config, raw)}


def platform_module():
    module = types.ModuleType("vllm_ascend.platform")

    class NPUPlatform:
        @classmethod
        def get_attn_backend_cls(cls, selected_backend, attn_selector_config, num_heads=None):
            return ("vllm_ascend.attention.mla_v1.AscendMLABackend" if attn_selector_config.use_mla
                    else "vllm_ascend.attention.attention_v1.AscendAttentionBackend")

        @classmethod
        def register_custom_kv_cache_specs(cls, vllm_config):
            return None

    module.NPUPlatform = NPUPlatform
    return module


def runner_module():
    module = types.ModuleType("vllm_ascend.worker.model_runner_v1")

    class NPUModelRunner:
        def get_kv_cache_spec(self):
            return {"native": "spec"}

        def _allocate_kv_cache_tensors(self, kv_cache_config):
            return {"native": kv_cache_config}

        def _reshape_kv_cache_tensors(self, kv_cache_config, kv_cache_raw_tensors):
            return {"native": (kv_cache_config, kv_cache_raw_tensors)}

        def _dummy_run(self, *args, probe=None, **kwargs):
            return probe(self, *args, **kwargs) if probe is not None else "native dummy"

    module.NPUModelRunner = NPUModelRunner
    return module


class PluginTests(unittest.TestCase):
    def setUp(self):
        clear_runtime()
        plugin.unregister()

    def tearDown(self):
        plugin.unregister()
        clear_runtime()

    def test_fresh_interpreter_register_imports_no_heavy_modules(self):
        source = """
import sys, os, json
baseline = set(sys.modules)
os.environ['OSCAR_ENABLED'] = '1'
from oscar_ascend.plugin import register, unregister
register()
added = set(sys.modules) - baseline
print(json.dumps(sorted(m for m in added if m.split('.')[0] in {'torch','torch_npu','vllm','vllm_ascend'})))
unregister()
"""
        output = subprocess.check_output([sys.executable, "-c", source], text=True)
        self.assertEqual(json.loads(output), [])

    def test_disabled_registration_does_not_change_host(self):
        module = platform_module()
        original = module.NPUPlatform.__dict__["get_attn_backend_cls"]
        with patch.dict(sys.modules, {module.__name__: module}), patch.dict(os.environ, {"OSCAR_ENABLED": "0"}):
            plugin.register()
            self.assertIs(module.NPUPlatform.__dict__["get_attn_backend_cls"], original)
            self.assertFalse(plugin.hook_status()["enabled"])

    def test_classmethod_signature_idempotence_and_exact_restore(self):
        module = platform_module()
        cls = module.NPUPlatform
        original = cls.__dict__["get_attn_backend_cls"]
        signature = inspect.signature(cls.get_attn_backend_cls)
        with patch.dict(sys.modules, {module.__name__: module}), patch.dict(os.environ, {"OSCAR_ENABLED": "1"}):
            install_runtime(Provider())
            plugin.register()
            first = cls.__dict__["get_attn_backend_cls"]
            plugin.register()
            self.assertIs(cls.__dict__["get_attn_backend_cls"], first)
            self.assertEqual(inspect.signature(cls.get_attn_backend_cls), signature)
            self.assertEqual(cls.get_attn_backend_cls(None, types.SimpleNamespace(use_mla=False)),
                             "oscar_ascend.integration.backend.OscarAttentionBackend")
            plugin.unregister()
            self.assertIs(cls.__dict__["get_attn_backend_cls"], original)

    def test_enabled_dense_route_cannot_succeed_without_runtime(self):
        module = platform_module()
        with patch.dict(sys.modules, {module.__name__: module}), patch.dict(os.environ, {"OSCAR_ENABLED": "1"}):
            plugin.register()
            with self.assertRaises(OscarReadinessError):
                module.NPUPlatform.get_attn_backend_cls(None, types.SimpleNamespace(use_mla=False))
            self.assertIn("AscendMLABackend", module.NPUPlatform.get_attn_backend_cls(None, types.SimpleNamespace(use_mla=True)))

    def test_encoder_route_is_outside_full_decoder_scope(self):
        module = platform_module()
        with patch.dict(sys.modules, {module.__name__: module}), patch.dict(os.environ, {"OSCAR_ENABLED": "1"}):
            plugin.register()
            config = types.SimpleNamespace(use_mla=False, attn_type="encoder_only")
            self.assertEqual(module.NPUPlatform.get_attn_backend_cls(None, config),
                             "vllm_ascend.attention.attention_v1.AscendAttentionBackend")

    def test_alternative_native_dense_backend_does_not_bypass_readiness(self):
        module = platform_module()

        @classmethod
        def fa3(cls, selected_backend, attn_selector_config, num_heads=None):
            return "vllm_ascend.attention.fa3_v1.AscendFABackend"

        module.NPUPlatform.get_attn_backend_cls = fa3
        with patch.dict(sys.modules, {module.__name__: module}), patch.dict(os.environ, {"OSCAR_ENABLED": "1"}):
            plugin.register()
            with self.assertRaises(OscarReadinessError):
                module.NPUPlatform.get_attn_backend_cls(None, types.SimpleNamespace(use_mla=False))

    def test_runner_hooks_delegate_to_concrete_provider(self):
        module = runner_module()
        cls = module.NPUModelRunner
        originals = {name: cls.__dict__[name] for name in (
            "get_kv_cache_spec", "_allocate_kv_cache_tensors", "_reshape_kv_cache_tensors", "_dummy_run")}
        with patch.dict(sys.modules, {module.__name__: module}), patch.dict(os.environ, {"OSCAR_ENABLED": "1"}):
            install_runtime(Provider())
            plugin.register()
            runner = cls()
            self.assertEqual(runner.get_kv_cache_spec(), {"oscar": {"native": "spec"}})
            self.assertEqual(runner._allocate_kv_cache_tensors(42), {"oscar": {"native": 42}})
            self.assertEqual(runner._reshape_kv_cache_tensors(42, "raw"), {"oscar": {"native": (42, "raw")}})
            plugin.unregister()
            for name, original in originals.items():
                self.assertIs(cls.__dict__[name], original)

    def test_dummy_run_scope_resets_on_success_and_failure(self):
        from oscar_ascend.integration.dummy_context import is_native_dummy_run
        module = runner_module()
        cls = module.NPUModelRunner
        original = cls.__dict__["_dummy_run"]
        with patch.dict(sys.modules, {module.__name__: module}), patch.dict(os.environ, {"OSCAR_ENABLED": "1"}):
            install_runtime(Provider())
            plugin.register()
            runner = cls()
            self.assertFalse(is_native_dummy_run())
            self.assertEqual(runner._dummy_run(17, probe=lambda _runner, count: (is_native_dummy_run(), count)),
                             (True, 17))
            self.assertFalse(is_native_dummy_run())

            def fail(_runner):
                self.assertTrue(is_native_dummy_run())
                raise RuntimeError("dummy warmup failed")

            with self.assertRaisesRegex(RuntimeError, "dummy warmup failed"):
                runner._dummy_run(probe=fail)
            self.assertFalse(is_native_dummy_run())
            plugin.unregister()
            self.assertIs(cls.__dict__["_dummy_run"], original)
            self.assertFalse(is_native_dummy_run())

    def test_dummy_run_nested_scope_and_idempotent_reversible_patch(self):
        from oscar_ascend.integration.dummy_context import is_native_dummy_run
        module = runner_module()
        cls = module.NPUModelRunner
        original = cls.__dict__["_dummy_run"]
        signature = inspect.signature(cls._dummy_run)
        with patch.dict(sys.modules, {module.__name__: module}), patch.dict(os.environ, {"OSCAR_ENABLED": "1"}):
            install_runtime(Provider())
            plugin.register()
            replacement = cls.__dict__["_dummy_run"]
            plugin.register()
            self.assertIs(cls.__dict__["_dummy_run"], replacement)
            self.assertEqual(inspect.signature(cls._dummy_run), signature)
            runner = cls()

            def outer(instance):
                before = is_native_dummy_run()
                nested = instance._dummy_run(probe=lambda _runner: is_native_dummy_run())
                return before, nested, is_native_dummy_run()

            self.assertEqual(runner._dummy_run(probe=outer), (True, True, True))
            self.assertFalse(is_native_dummy_run())
            plugin.unregister()
            self.assertIs(cls.__dict__["_dummy_run"], original)

    def test_future_import_runs_the_existing_loader_before_patching(self):
        target = "oscar_fake_seam"
        observations = []

        class Loader(importlib.abc.Loader):
            def create_module(self, spec):
                return None

            def exec_module(self, module):
                module.finished = True
                observations.append("exec")

        class Finder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "oscar_fake_seam":
                    return importlib.util.spec_from_loader(fullname, Loader())
                return None

        def callback(module):
            self.assertTrue(module.finished)
            observations.append("patch")

        finder = Finder()
        sys.meta_path.insert(0, finder)
        try:
            with patch.dict(plugin._callbacks, {target: callback}, clear=True), patch.dict(os.environ, {"OSCAR_ENABLED": "1"}):
                plugin.register()
                importlib.import_module(target)
                self.assertEqual(observations, ["exec", "patch"])
        finally:
            sys.modules.pop(target, None)
            sys.meta_path.remove(finder)

    def test_partial_installation_rolls_back(self):
        module = platform_module()
        original = module.NPUPlatform.__dict__["get_attn_backend_cls"]
        delattr(module.NPUPlatform, "register_custom_kv_cache_specs")
        with patch.dict(sys.modules, {module.__name__: module}), patch.dict(os.environ, {"OSCAR_ENABLED": "1"}):
            with self.assertRaises(plugin.OscarHookConflictError):
                plugin.register()
            self.assertIs(module.NPUPlatform.__dict__["get_attn_backend_cls"], original)
            self.assertFalse(plugin.hook_status()["enabled"])

    def test_runtime_factory_is_lazy_and_once(self):
        calls = []
        provider = Provider()

        def factory():
            calls.append(1)
            return provider

        install_runtime_factory(factory)
        self.assertEqual(calls, [])
        self.assertIs(require_runtime(), provider)
        self.assertIs(require_runtime(), provider)
        self.assertEqual(calls, [1])

    def test_missing_provider_methods_are_explicit(self):
        with self.assertRaisesRegex(OscarReadinessError, "required methods"):
            install_runtime(object())


if __name__ == "__main__":
    unittest.main()
