"""Archive #27/#28/#34/#36/#74–78/#94/#95/#117: cross-module contracts.

The target command is compared against the supplied Appendix A, not against
a second copy of the implementation. Deployment failures are real subprocess
executions. Fake host classes exercise only Python hook/cache semantics.
"""

import functools
import inspect
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from oscar_ascend import plugin
from oscar_ascend.integration.runtime_api import clear_runtime
from tools import deploy, prepare_rotations, target_cli


ROOT = Path(__file__).resolve().parents[1]


def _options(argv):
    result = {"positional": []}
    i = 0
    while i < len(argv):
        token = argv[i]
        if token.startswith("--"):
            key = token.replace("_", "-")
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                value = argv[i + 1]
                try:
                    result[key] = json.loads(value)
                except json.JSONDecodeError:
                    result[key] = value
                i += 2
            else:
                result[key] = True
                i += 1
        else:
            result["positional"].append(token)
            i += 1
    return result


class TargetContracts(unittest.TestCase):
    def test_h14_command_matches_supplied_appendix_a(self):
        doc = (ROOT / "oscar_ascend_agent_start.md").read_text()
        appendix = doc.split("## 附录 A：", 1)[1]
        command = appendix.split("```bash", 1)[1].split("```", 1)[0]
        expected = shlex.split(command.replace("\\\n", " "))[1:]
        config = json.loads((ROOT / "configs/target.json").read_text())
        self.assertEqual(_options(target_cli.serve_argv(config)), _options(expected))

    def test_existing_plugin_allowlist_cannot_exclude_ascend(self):
        result = target_cli.target_env({"devices": [4, 5, 6, 7]}, {"VLLM_PLUGINS": "other_plugin"})
        self.assertEqual(set(result["VLLM_PLUGINS"].split(",")),
                         {"ascend", "oscar_ascend", "other_plugin"})

    def test_configuration_failure_writes_deployment_status(self):
        config = json.loads((ROOT / "configs/target.json").read_text())
        config["devices"] = None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "target.json"
            config_path.write_text(json.dumps(config))
            result = subprocess.run(
                [sys.executable, "-m", "tools.deploy", "--config", str(config_path),
                 "--only", "prepare-rotations", "--log-dir", str(root / "logs")],
                cwd=ROOT, capture_output=True, text=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            status_path = root / "logs/status.json"
            self.assertTrue(status_path.exists(), result.stdout + result.stderr)
            status = json.loads(status_path.read_text())
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["failed_phase"], "config")

    def test_prepare_rotations_public_api_binding(self):
        # Binding is pure Python and catches stale keyword names before any
        # model loading or NPU initialization. Numerical behavior is elsewhere.
        from oscar_ascend.rotations import build_hadamard_artifact, load_artifact, save_artifact
        values = {"layer_names": ["model.layers.3.self_attn.attn"], "head_dim": 256,
                  "model_fingerprint": "verified-source", "device": "npu:0"}
        inspect.signature(build_hadamard_artifact).bind(**values)
        inspect.signature(load_artifact).bind("artifact.pt", **values)
        inspect.signature(save_artifact).bind({}, "artifact.pt")

    def test_geometry_inconsistent_layer_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps({"text_config": {
                "num_hidden_layers": 8, "head_dim": 256, "layer_types": ["full_attention"]}}))
            with self.assertRaises(ValueError):
                prepare_rotations.model_geometry(path)

    def test_failed_probe_preserves_code_and_stops_later_phases(self):
        config = json.loads((ROOT / "configs/target.json").read_text())
        config["devices"] = [4, 5, 6, 7]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "target.json"
            config_path.write_text(json.dumps(config))
            logs = root / "logs"

            def plan(_config, log_dir):
                marker = "from pathlib import Path; Path(" + repr(str(log_dir / "must-not-start")) + ").write_text('started')"
                return [
                    ("probe-failed", [sys.executable, "-c", "raise SystemExit(7)"]),
                    ("service-probe", [sys.executable, "-c", marker]),
                ]

            with patch.object(deploy, "plan", plan), patch.object(sys, "argv", ["deploy", "--config", str(config_path), "--log-dir", str(logs)]):
                code = deploy.main()
            self.assertEqual(code, 7)
            state = json.loads((logs / "status.json").read_text())
            self.assertEqual(state["failed_phase"], "probe-failed")
            self.assertEqual([(x["phase"], x["returncode"]) for x in state["phases"]],
                             [("probe-failed", 7)])
            self.assertFalse((logs / "must-not-start").exists())

    def test_missing_full_service_probe_cannot_leave_running_status(self):
        config = json.loads((ROOT / "configs/target.json").read_text())
        config["devices"] = [4, 5, 6, 7]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "target.json"
            config_path.write_text(json.dumps(config))
            with patch.object(deploy, "plan", lambda *_args: []), patch.object(sys, "argv", ["deploy", "--config", str(config_path), "--log-dir", str(root / "logs")]):
                code = deploy.main()
            self.assertEqual(code, 1)
            state = json.loads((root / "logs/status.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["failed_phase"], "full-service-probe")


class NativeHookCacheContracts(unittest.TestCase):
    def tearDown(self):
        plugin.unregister()
        clear_runtime()

    def test_selector_cache_is_invalidated_on_install_and_restore(self):
        platform = types.ModuleType("vllm_ascend.platform")

        class NPUPlatform:
            @classmethod
            def get_attn_backend_cls(cls, selected_backend, attn_selector_config, num_heads=None):
                return "vllm_ascend.attention.attention_v1.AscendAttentionBackend"

            @classmethod
            def register_custom_kv_cache_specs(cls, vllm_config):
                return None

        platform.NPUPlatform = NPUPlatform
        selector = types.ModuleType("vllm.v1.attention.selector")

        @functools.cache
        def cached(key):
            return object()

        selector._cached_get_attn_backend = cached
        with patch.dict(sys.modules, {platform.__name__: platform, selector.__name__: selector}), patch.dict(os.environ, {"OSCAR_ENABLED": "1"}):
            cached("native")
            self.assertEqual(cached.cache_info().currsize, 1)
            plugin.register()
            self.assertEqual(cached.cache_info().currsize, 0)
            cached("oscar")
            plugin.unregister()
            self.assertEqual(cached.cache_info().currsize, 0)

    def test_inherited_descriptor_is_removed_on_restore(self):
        class Parent:
            @classmethod
            def action(cls, argument):
                return cls, argument

        class Child(Parent):
            pass

        def builder(original):
            def wrapper(cls, argument):
                return original(cls, argument)
            return wrapper

        original = Parent.__dict__["action"]
        plugin._patch(Child, "action", builder)
        self.assertEqual(Child.action(7), (Child, 7))
        plugin.unregister()
        self.assertNotIn("action", Child.__dict__)
        self.assertIs(Parent.__dict__["action"], original)

    def test_failed_future_callback_restores_its_partial_patch(self):
        class Owner:
            def action(self):
                return 1

        original = Owner.__dict__["action"]

        def bad_callback(_module):
            plugin._patch(Owner, "action", lambda original: lambda self: original(self))
            raise RuntimeError("missing next native seam")

        with self.assertRaisesRegex(RuntimeError, "missing next"):
            plugin._apply_callback(bad_callback, types.ModuleType("test"))
        self.assertIs(Owner.__dict__["action"], original)
        self.assertEqual(plugin.hook_status()["patched"], [])


if __name__ == "__main__":
    unittest.main()
