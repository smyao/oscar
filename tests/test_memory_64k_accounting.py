"""Archive #34/#36/#127/#128/#152: audit byte accounting against native code.

These host tests execute only shape/config arithmetic from the read-only native
reference. They do not execute kernels or claim device/performance acceptance.
"""
import ast
from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

from tools.analyze_memory_64k import analyze, geometry_from_model, grouping, reference_geometry

ROOT = Path(__file__).resolve().parents[1]


def native_functions(path, names, namespace):
    """Execute named native arithmetic unchanged, without importing vLLM/NPU."""
    tree = ast.parse((ROOT / path).read_text())
    selected = [node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise AssertionError("native function selection drifted")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


class MemoryLedgerTests(unittest.TestCase):
    def setUp(self):
        self.target = json.loads((ROOT / "configs/target.json").read_text())
        # User 2026-10-13: stable defaults to none; APC/align remains an
        # explicitly selected compatibility case below.
        self.assertFalse(self.target["enable_prefix_caching"])

    def test_native_single_k_alignment_is_not_combined_state_roundup(self):
        shapes = native_functions(
            "references/vllm/vllm/model_executor/layers/mamba/mamba_utils.py",
            {"gated_delta_net_state_shape"}, {"divide": lambda x, y: x // y})
        shape = shapes["gated_delta_net_state_shape"].__func__(
            NS(_orient_conv_shape=lambda width, length: (length, width)),
            4, 16, 48, 128, 128, 4, 3)
        native_model = NS(get_mamba_state_shape_from_config=lambda config: shape,
                          get_mamba_state_dtype_from_config=lambda config: (2, 2))
        env = native_functions(
            "references/vllm-ascend/vllm_ascend/patch/platform/patch_mamba_config.py",
            {"_using_kv_store", "verify_and_update_config"},
            {"math": math, "cdiv": lambda x, y: (x + y - 1) // y,
             "get_dtype_size": lambda dtype: dtype,
             "MambaModelConfig": NS(verify_and_update_config=lambda config: None),
             "ModelRegistry": NS(resolve_model_cls=lambda *args, **kwargs: (native_model, "fixture")),
             "logger": NS(debug=lambda *args: None, info=lambda *args: None)})
        config = NS(scheduler_config=NS(disable_hybrid_kv_cache_manager=False),
                    cache_config=NS(cache_dtype="auto", block_size=None, mamba_page_size_padded=None,
                                    enable_prefix_caching=False, mamba_cache_mode="none"),
                    model_config=NS(dtype=2, architecture="fixture", use_mla=False, max_model_len=262144,
                                    get_num_kv_heads=lambda parallel: 1, get_head_size=lambda: 256),
                    parallel_config=NS(tensor_parallel_size=4), kv_transfer_config=None,
                    speculative_config=NS(method="qwen3_5_mtp"))
        env["verify_and_update_config"].__func__(None, config)
        ledger = analyze(self.target, reference_geometry())
        self.assertEqual(shape, ((6, 2560), (12, 128, 128)))
        self.assertEqual(config.cache_config.block_size, ledger["layout"]["native_full_block_tokens"])
        self.assertEqual(config.cache_config.mamba_page_size_padded, ledger["layout"]["native_padded_page_bytes"])
        self.assertEqual(config.cache_config.block_size, 768)
        self.assertEqual(config.cache_config.mamba_page_size_padded, 817152)
        # model_runner_v1.py:5031-5034 makes native FULL and GDN P identical.
        native_specs = {"full": NS(page_size_bytes=817152), "gdn": NS(page_size_bytes=817152)}
        unify = native_functions("references/vllm/vllm/v1/core/kv_cache_utils.py",
                                 {"unify_kv_cache_spec_page_size"}, {"replace": replace})
        self.assertIs(unify["unify_kv_cache_spec_page_size"](native_specs), native_specs)

    def test_physical_pool_counts_null_once_and_snapshots_without_double_charge(self):
        result = analyze(self.target, reference_geometry())
        layout = result["layout"]
        self.assertEqual((layout["shared_pool_width"], layout["full_groups"], layout["gdn_groups"]), (17, 1, 3))
        self.assertEqual(layout["snapshot_bytes_per_physical_page"], 323 * (1024 + 8))
        prompt, end, reserved = result["per_request"]
        self.assertEqual((prompt["native_pool_blocks_per_request"], prompt["oscar_pool_blocks_per_request"]), (97, 36))
        self.assertEqual((end["native_pool_blocks_per_request"], end["oscar_pool_blocks_per_request"]), (98, 36))
        self.assertEqual(end["oscar_total_kv_bytes"], 36 * 17 * 817152)
        self.assertEqual(sum(end["oscar_full_page_components_bytes"].values()), end["oscar_full_bytes"])
        unit = layout["pool_block_bytes"]
        # A net profiled KV budget must not pay external scratch a second time.
        result = analyze(self.target, reference_geometry(), cube_cores=20,
                         oscar_kv_budget_bytes=(26 * 36 + 1) * unit)
        self.assertEqual(result["capacity_from_supplied_profiled_budgets"]["oscar"]["equal_full_length_requests_upper_bound"], 26)
        self.assertEqual(result["cohort_kv_upper_all_at_full_length"]["26"]["oscar_bytes_per_rank"], (26 * 36 + 1) * unit)

    def test_model_config_changes_geometry_and_rejects_unsupported_scope(self):
        model = {"text_config": {"num_attention_heads": 24, "num_key_value_heads": 4,
            "head_dim": 256, "num_hidden_layers": 64, "full_attention_interval": 4,
            "linear_num_key_heads": 16, "linear_num_value_heads": 48,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4}}
        self.assertEqual(geometry_from_model(model, self.target), reference_geometry())
        model["text_config"]["mtp_num_hidden_layers"] = 2
        self.assertEqual(geometry_from_model(model, self.target).draft_full_layers, 2)
        with self.assertRaisesRegex(ValueError, "not all mode"):
            analyze(self.target | {"mamba_cache_mode": "all"}, reference_geometry())
        with self.assertRaisesRegex(ValueError, "exceeds max_model_len"):
            analyze(self.target, reference_geometry(), prompt_tokens=262144)
        self.assertEqual(grouping(12, 13), (13, 1, 1))

    def test_unknown_device_costs_remain_unknown_and_proposals_keep_gdn_states(self):
        result = analyze(self.target, reference_geometry())
        fixed = result["fixed_external_allocations"]
        self.assertIsNone(fixed["variants"]["candidate"]["workspace_per_rank"])
        measured_cores_only = analyze(self.target, reference_geometry(), cube_cores=20)
        candidate = measured_cores_only["fixed_external_allocations"]["variants"]["candidate"]
        self.assertEqual(candidate["token_capacity"], 16384)
        self.assertIsNotNone(candidate["workspace_per_rank"])
        self.assertEqual(candidate["token_capacity_source"], "explicit")
        native_budget = dict(self.target, use_native_token_budget=True)
        native_budget.pop("max_num_batched_tokens")
        unresolved = analyze(native_budget, reference_geometry(), cube_cores=20)
        unresolved_candidate = unresolved["fixed_external_allocations"]["variants"]["candidate"]
        self.assertIsNone(unresolved_candidate["token_capacity"])
        self.assertIsNone(unresolved_candidate["workspace_per_rank"])
        self.assertEqual(unresolved_candidate["token_capacity_source"], "native_unresolved")
        self.assertIsNone(fixed["whole_hbm_compression_ratio"])
        self.assertIsNone(result["capacity_from_supplied_profiled_budgets"]["native"])
        proposal = result["per_request"][1]["proposed_unimplemented"]
        self.assertEqual(proposal["tight_gdn_bytes_preserving_all_native_states"], 48 * 4 * (30720 + 393216))
        self.assertLess(proposal["native_all_kv_over_proposed_tight_gdn_ratio"], 6)
        self.assertGreater(proposal["full_bf16_logical_over_proposed_ratio"], 7)

    def test_explicit_candidate_workspace_accounts_for_bf16_c16_and_member_pipeline(self):
        # #34/#36/#148/#164: the offline arena must cover the selected reader,
        # not the retired FP32 scratch formula. This is byte arithmetic only.
        from oscar_ascend.runtime import WorkspaceGeometry
        result = analyze(self.target, reference_geometry(), cube_cores=20)
        candidate = result["fixed_external_allocations"]["variants"]["candidate"]
        current = WorkspaceGeometry(16384, 6, 1, 256, 1, 20, 16,
                                    c4_member_pipeline=True, bf16_history=True)
        retired_same_cluster = WorkspaceGeometry(16384, 6, 1, 256, 1, 20, 16)
        self.assertEqual(current.cv_bytes, 149913600)
        self.assertEqual(retired_same_cluster.cv_bytes, 99942400)
        self.assertEqual(candidate["workspace_per_rank"],
                         {"cv_bytes": current.cv_bytes, "total_bytes": current.total_bytes})
        self.assertNotEqual(candidate["workspace_per_rank"]["cv_bytes"],
                            retired_same_cluster.cv_bytes)
        # The historical baseline remains diagnostic-only, never a serving
        # variant, and still uses its own C1 geometry rather than C16.
        diagnostic = result["fixed_external_allocations"]["variants"]["baseline"]
        self.assertEqual(diagnostic["history_cluster_size"], 1)
        self.assertEqual(diagnostic["workspace_per_rank"]["cv_bytes"],
                         WorkspaceGeometry(16384, 6, 1, 256, 1, 20, 1).cv_bytes)

    def test_explicit_prefix_align_counts_native_previous_state_and_snapshots(self):
        target = json.loads((ROOT / "configs/target.json").read_text())
        target["enable_prefix_caching"] = True
        result = analyze(target, reference_geometry())
        layout = result["layout"]
        self.assertTrue(layout["prefix_caching"])
        self.assertEqual(layout["native_mamba_cache_mode"], "align")
        self.assertEqual((layout["native_mamba_block_tokens"], layout["oscar_full_block_tokens"]), (768, 2304))
        self.assertEqual(layout["gdn_states_per_layer_per_request"], 5)
        self.assertEqual(layout["gdn_conv_ssm_payload_per_request_bytes"], 48 * 5 * (30720 + 393216))
        prompt, end, reserved = result["per_request"]
        self.assertEqual((prompt["native_pool_blocks_per_request"], prompt["oscar_pool_blocks_per_request"]), (100, 44))
        self.assertEqual((reserved["native_pool_blocks_per_request"], reserved["oscar_pool_blocks_per_request"]), (101, 44))
        self.assertEqual(reserved["native_over_oscar_full_physical_ratio"], 86 / 29)
        self.assertEqual(reserved["native_over_oscar_all_kv_physical_ratio"], 101 / 44)
        self.assertFalse(reserved["full_target_gap"]["target_met_by_current_layout"])
        self.assertEqual(sum(reserved["oscar_full_page_components_bytes"].values()), reserved["oscar_full_bytes"])
        self.assertEqual(result["prefix_sharing_and_retention"]["sharing_savings_bytes_claimed"], 0)
        self.assertIsNone(result["prefix_sharing_and_retention"]["actual_gdn_cached_checkpoint_blocks"])
        research = reserved["research_request_windows_none_only"]
        self.assertFalse(research["serving_available_under_current_default"])
        self.assertFalse(research["required_prefix_caching"])
        self.assertEqual(research["sidecar_reserved_requests"], 128)
        self.assertEqual(research["physical_tokens_per_page"], 5632)

    def test_gdn_reservation_matches_unchanged_native_spec_math_for_both_modes(self):
        path = ROOT / "references/vllm/vllm/v1/kv_cache_interface.py"
        tree = ast.parse(path.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MambaSpec")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "max_memory_usage_bytes")
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
        namespace = {"cdiv": lambda a, b: (a + b - 1) // b}
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
        for prefix in (False, True):
            result = analyze(self.target | {"enable_prefix_caching": prefix}, reference_geometry())
            layout = result["layout"]
            spec = NS(page_size_bytes=layout["native_padded_page_bytes"],
                      block_size=layout["native_mamba_block_tokens"], num_speculative_blocks=3)
            cfg = NS(cache_config=NS(mamba_cache_mode=layout["native_mamba_cache_mode"]))
            actual = namespace["max_memory_usage_bytes"](spec, cfg)
            self.assertEqual(actual, layout["gdn_states_per_layer_per_request"] * spec.page_size_bytes)

    def test_request_window_research_is_not_reported_as_current_prefix_route(self):
        target = self.target | {"enable_prefix_caching": True, "experimental_request_windows": True}
        with self.assertRaisesRegex(ValueError, "research layout"):
            analyze(target, reference_geometry())
        with self.assertRaisesRegex(ValueError, "explicit boolean"):
            analyze(self.target | {"enable_prefix_caching": "false"}, reference_geometry())


if __name__ == "__main__":
    unittest.main()
