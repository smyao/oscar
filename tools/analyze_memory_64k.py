"""Offline byte ledger for the 65100-input/512-output, 120-request K32 task.

Archive #26/#37-49/#69: exact windows and MTP retain their full byte cost.
Archive #34/#36/#127/#128: physical pages are not metadata128 blocks.
Archive #123-125: read this checkout's target; no device/VM/deployment action.
Archive #152/#155: Running ratios and short probes are not memory acceptance.
This is integer accounting, never a serving path or performance simulation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from oscar_ascend.layout import HybridPageLayout, SlotLayout, ceil_div
from oscar_ascend.lifecycle import SnapshotLayout
from oscar_ascend.dual_region_layout import DualRegionPageLayout
from oscar_ascend.request_window_storage import RequestWindowLayout
from oscar_ascend.runtime import WorkspaceGeometry
from tools.serving_variants import variant_config, research_variant_config

ROOT = Path(__file__).resolve().parents[1]
MIB = 1 << 20


def positive(name: str, value: int) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class ModelGeometry:
    target_full_layers: int
    gdn_layers: int
    draft_full_layers: int
    local_query_heads: int
    local_kv_heads: int
    head_dim: int
    linear_key_heads: int
    linear_value_heads: int
    linear_key_dim: int
    linear_value_dim: int
    conv_kernel: int

    def __post_init__(self):
        for key, value in asdict(self).items():
            positive(key, value)


def geometry_from_model(model: dict, target: dict) -> ModelGeometry:
    """Derive shapes from config, not a model name or Running count.

    This target uses one uniform FULL geometry, native BF16 GDN and TP.
    Heterogeneous attention or Mamba all mode requires a separate ledger.
    """
    text = model.get("text_config", model)
    tp = positive("tensor_parallel_size", target["tensor_parallel_size"])
    heads = positive("num_attention_heads", text["num_attention_heads"])
    kv_heads = positive("num_key_value_heads", text["num_key_value_heads"])
    count = positive("num_hidden_layers", text["num_hidden_layers"])
    types = text.get("layer_types")
    if types is None:
        interval = positive("full_attention_interval", text["full_attention_interval"])
        types = ["full_attention" if (i + 1) % interval == 0 else "linear_attention"
                 for i in range(count)]
    if len(types) != count or set(types) - {"full_attention", "linear_attention"}:
        raise ValueError("unsupported layer_types; exact FULL/GDN identities required")
    if heads % tp or (kv_heads >= tp and kv_heads % tp) or (kv_heads < tp and tp % kv_heads):
        raise ValueError("heads must support native TP partition or KV replication")
    dim = text.get("head_dim")
    if dim is None:
        hidden = positive("hidden_size", text["hidden_size"])
        if hidden % heads:
            raise ValueError("hidden_size is not divisible by attention heads")
        dim = hidden // heads
    return ModelGeometry(
        types.count("full_attention"), types.count("linear_attention"),
        text.get("mtp_num_hidden_layers", 1), heads // tp, max(1, kv_heads // tp),
        dim, text["linear_num_key_heads"], text["linear_num_value_heads"],
        text["linear_key_head_dim"], text["linear_value_head_dim"],
        text["linear_conv_kernel_dim"],
    )


def reference_geometry() -> ModelGeometry:
    # docs/design.md:57 and tests/test_runtime_native.py:282 are conditional
    # geometry evidence, NOT the absent current target model config.json.
    return ModelGeometry(16, 48, 1, 6, 1, 256, 16, 48, 128, 128, 4)


def grouping(full_layers: int, gdn_layers: int) -> tuple[int, int, int]:
    """Mirror native kv_cache_utils._get_kv_cache_groups_uniform_page_size.

    Native's 1.5 heuristic is integer-exact here, including the equality edge.
    """
    lo, hi = sorted((positive("full_layers", full_layers), positive("gdn_layers", gdn_layers)))
    width = hi if 2 * hi < 3 * lo else lo
    return width, ceil_div(full_layers, width), ceil_div(gdn_layers, width)


def analyze(target: dict, geometry: ModelGeometry, *, prompt_tokens: int = 65100,
            output_tokens: int = 512, requests: int = 120, client_concurrency: int = 32,
            cube_cores: int | None = None, native_kv_budget_bytes: int | None = None,
            oscar_kv_budget_bytes: int | None = None) -> dict:
    for name, value in (("prompt_tokens", prompt_tokens), ("output_tokens", output_tokens),
                        ("requests", requests), ("client_concurrency", client_concurrency)):
        positive(name, value)
    tp = positive("tensor_parallel_size", target["tensor_parallel_size"])
    max_len = positive("max_model_len", target["max_model_len"])
    spec = target["speculative_config"]
    if spec.get("method") != "qwen3_5_mtp":
        raise ValueError("this ledger requires the configured qwen3_5_mtp method")
    mtp = positive("num_speculative_tokens", spec["num_speculative_tokens"])
    prefix = target.get("enable_prefix_caching", True)
    if type(prefix) is not bool:
        raise ValueError("enable_prefix_caching must be an explicit boolean")
    requested_mode = target.get("mamba_cache_mode", "none")
    if requested_mode not in {"none", "align"}:
        raise ValueError("this Qwen hybrid ledger supports native none/align, not all mode")
    # Pinned MambaModelConfig: APC turns Qwen's none into align; disabling APC
    # turns align into none. Do not carry the historical none assumption into
    # the new default, and do not change the real GDN spec to suit accounting.
    mode = "align" if prefix else "none"
    if target.get("experimental_request_windows", False):
        raise ValueError("request_windows is a research layout, not the current page-snapshot route; see the conditional projection")
    if prompt_tokens + output_tokens + mtp > max_len:
        raise ValueError("prompt + output + MTP lookahead exceeds max_model_len")
    conv_width = 2 * geometry.linear_key_heads * geometry.linear_key_dim + geometry.linear_value_heads * geometry.linear_value_dim
    if conv_width % tp or geometry.linear_value_heads % tp:
        raise ValueError("native GDN state dimensions must divide TP")
    # Native gated_delta_net_state_shape includes MTP in conv length AND
    # MambaManager retains (1+mtp) state blocks. They are distinct costs.
    conv_bytes = conv_width // tp * (geometry.conv_kernel - 1 + mtp) * 2
    ssm_bytes = geometry.linear_value_heads // tp * geometry.linear_key_dim * geometry.linear_value_dim * 2
    slot = SlotLayout(geometry.head_dim, num_kv_heads=geometry.local_kv_heads)
    single_k_token_bytes = geometry.local_kv_heads * geometry.head_dim * 2
    native_block = ceil_div(ssm_bytes, 128 * single_k_token_bytes) * 128
    if native_block * single_k_token_bytes != ssm_bytes:
        raise ValueError("native Ascend single-K/SSM exact alignment is not satisfied")
    page_bytes = native_block * slot.bf16_token_bytes + conv_bytes
    page = HybridPageLayout(slot, conv_bytes, ssm_bytes, native_block if prefix else max_len,
                            native_page_size_bytes=page_bytes, native_mamba_cache_mode=mode)
    window = SnapshotLayout(page, target["sink_tokens"], target["recent_tokens"], mtp)
    full_layers = geometry.target_full_layers + geometry.draft_full_layers
    width, full_groups, gdn_groups = grouping(full_layers, geometry.gdn_layers)
    pool_block_bytes = width * page_bytes
    # Pinned MambaSpec.max_memory_usage_bytes: align keeps one previous state
    # during handoff in addition to the running state and MTP reserve. This is
    # a live reservation bound, not a measurement of exact instantaneous use.
    gdn_state_blocks = (2 if prefix else 1) + mtp
    gdn_pool_blocks = gdn_groups * gdn_state_blocks
    gdn_physical = gdn_pool_blocks * pool_block_bytes
    gdn_tight = geometry.gdn_layers * gdn_state_blocks * (conv_bytes + ssm_bytes)
    none_page = HybridPageLayout(slot, conv_bytes, ssm_bytes, max_len,
        native_page_size_bytes=page_bytes, native_mamba_cache_mode="none")
    research_page = DualRegionPageLayout(none_page) if geometry.head_dim == 256 else None
    research_window = RequestWindowLayout.from_config(target, num_kv_heads=geometry.local_kv_heads,
                                                       head_dim=geometry.head_dim)
    rows = []
    for stage, length in (("prompt", prompt_tokens), ("output_end", prompt_tokens + output_tokens),
                          ("output_end_plus_mtp_reserve", prompt_tokens + output_tokens + mtp)):
        native_pages = ceil_div(length, native_block)
        oscar_pages = ceil_div(length, page.block_size)
        native_full = native_pages * full_groups * pool_block_bytes
        oscar_full = oscar_pages * full_groups * pool_block_bytes
        # Proposal: exact INT2 format in 128-token standalone pages, plus one
        # retained per-request BF16 sink/recent/MTP ring. Not implemented.
        tight_packed = ceil_div(length, 128) * 128 * slot.token_bytes * full_layers
        tight_window = window.required_bytes * full_layers
        tight_full = tight_packed + tight_window
        current_kv = oscar_full + gdn_physical
        native_kv = native_full + gdn_physical
        native_semantic = length * slot.bf16_token_bytes * full_layers + gdn_tight
        row = {
            "stage": stage, "tokens": length,
            "native_full_pages_per_group": native_pages,
            "oscar_full_pages_per_group": oscar_pages,
            "gdn_pages_all_groups": gdn_pool_blocks,
            "native_pool_blocks_per_request": native_pages * full_groups + gdn_pool_blocks,
            "oscar_pool_blocks_per_request": oscar_pages * full_groups + gdn_pool_blocks,
            "native_full_bytes": native_full, "oscar_full_bytes": oscar_full,
            "gdn_physical_pool_bytes": gdn_physical,
            "native_total_kv_bytes": native_kv, "oscar_total_kv_bytes": current_kv,
            "native_over_oscar_full_physical_ratio": native_full / oscar_full,
            "native_over_oscar_all_kv_physical_ratio": native_kv / current_kv,
            "full_target_gap": {
                "minimum_ratio": 6, "stretch_ratio": 7,
                "current_ratio": native_full / oscar_full,
                "target_met_by_current_layout": native_full / oscar_full >= 6,
                "additional_full_bytes_reduction_factor_to_6x": oscar_full / (native_full / 6),
                "additional_full_bytes_reduction_factor_to_7x": oscar_full / (native_full / 7),
            },
            "oscar_full_page_components_bytes": {
                "packed_capacity": oscar_pages * full_groups * width * page.payload_bytes,
                "precise_snapshot_capacity": oscar_pages * full_groups * width * window.required_bytes,
                "unused_conv_region": oscar_pages * full_groups * width * conv_bytes,
                "unused_ssm_tail": oscar_pages * full_groups * width * page.unused_ssm_bytes,
                "unused_snapshot_padding": oscar_pages * full_groups * width * (page.native_padding_bytes - window.required_bytes),
            },
            "proposed_unimplemented": {
                "prefix_caching_compatible": False,
                "serving_available": False,
                "packed_128_page_bytes": tight_packed,
                "one_exact_window_per_request_bytes": tight_window,
                "full_bytes": tight_full,
                "full_bf16_logical_over_proposed_ratio": length * slot.bf16_token_bytes * full_layers / tight_full,
                "native_full_physical_over_proposed_ratio": native_full / tight_full,
                "total_with_unchanged_native_gdn_pool_bytes": tight_full + gdn_physical,
                "native_all_kv_over_proposed_with_native_gdn_ratio": native_kv / (tight_full + gdn_physical),
                "tight_gdn_bytes_preserving_all_native_states": gdn_tight,
                "total_with_tight_gdn_arena_bytes": tight_full + gdn_tight,
                "native_all_kv_over_proposed_tight_gdn_ratio": native_kv / (tight_full + gdn_tight),
                "native_logical_payload_over_proposed_tight_gdn_ratio": native_semantic / (tight_full + gdn_tight),
                "caveat": "Counterfactual standalone INT2/one-request-window bytes only. No exact cross-request prefix recovery, allocator/lifecycle implementation or NPU verification; excludes new checkpoint storage, indirection, alignment, graph arenas and fragmentation. Not an APC-compatible candidate.",
            },
        }
        if research_page is not None:
            # Preserve the previous none-only two-segment research ledger,
            # explicitly separate from today's default APC/page-snapshot route.
            active = min(requests, client_concurrency, target["max_num_seqs"])
            research_pages = ceil_div(length, research_page.block_size)
            fixed_sidecar = research_window.required_bytes * full_layers
            fixed_status = target["max_num_seqs"] * 4 * full_layers
            fixed_map = research_window.map_bytes
            native_full_cohort = active * native_full
            research_full_cohort = (active * research_pages * full_groups * pool_block_bytes
                                    + fixed_sidecar + fixed_status + fixed_map)
            none_gdn_cohort = active * gdn_groups * (1 + mtp) * pool_block_bytes
            row["research_request_windows_none_only"] = {
                "serving_available_under_current_default": False,
                "required_prefix_caching": False, "required_mamba_cache_mode": "none",
                "physical_tokens_per_page": research_page.block_size,
                "full_pages_per_request": research_pages,
                "cohort_active_requests": active,
                "sidecar_reserved_requests": target["max_num_seqs"],
                "fixed_sidecars_bytes_per_rank": fixed_sidecar,
                "fixed_window_status_bytes_per_rank": fixed_status,
                "shared_device_lease_map_bytes_per_rank": fixed_map,
                "pinned_host_lease_map_bytes_not_hbm": fixed_map,
                "cohort_full_bytes_including_all_reserved_sidecars": research_full_cohort,
                "native_over_research_full_ratio": native_full_cohort / research_full_cohort,
                "native_over_research_all_kv_ratio": (native_full_cohort + none_gdn_cohort) / (research_full_cohort + none_gdn_cohort),
                "checkpoint_bytes_for_apc": None,
                "caveat": "Historical none-only geometry; APC requires a new checkpoint/layout protocol and a new ledger. These ratios are not current-candidate acceptance.",
            }
        rows.append(row)
    variants = {}
    for name in ("baseline", "candidate"):
        effective = (research_variant_config(target, name) if name == "baseline"
                     else variant_config(target, name))
        # 2026-10-09: serving leaves the budget to the native engine. This
        # offline ledger cannot infer it from a configured capture list or
        # substitute 2048; historical reports retain their original numbers.
        resolved = effective.get("max_num_batched_tokens")
        capacity = (max(resolved, max(effective["compilation_config"]["cudagraph_capture_sizes"], default=0))
                    if resolved is not None else None)
        cluster = 16 if effective.get("experimental_mixed_cv", False) else 1
        known_workspace = None
        if cube_cores is not None and capacity is not None:
            workspace = WorkspaceGeometry(capacity, geometry.local_query_heads, geometry.local_kv_heads,
                geometry.head_dim, effective.get("attention_splits", 1), cube_cores, cluster,
                c4_member_pipeline=effective.get("experimental_c4_member_pipeline", False),
                bf16_history=effective.get("experimental_bf16_history", False))
            known_workspace = {"cv_bytes": workspace.cv_bytes, "total_bytes": workspace.total_bytes}
        variants[name] = {
            "token_capacity": capacity, "history_cluster_size": cluster,
            "token_capacity_source": "native_unresolved" if capacity is None else "explicit",
            "cube_cores": cube_cores, "workspace_per_rank": known_workspace,
            "scope": "One arena per unique device/geometry, shared across sequential FULL layers; already charged before native KV profiling.",
            "whole_prefill_limit_tokens": (capacity - effective["max_num_seqs"] * (mtp + 1)
                                            if effective.get("experimental_whole_prefill", False) else None),
        }
    end = rows[-1]
    capacities = {}
    for name, budget in (("native", native_kv_budget_bytes), ("oscar", oscar_kv_budget_bytes)):
        if budget is None:
            capacities[name] = None
            continue
        positive(f"{name}_kv_budget_bytes", budget)
        blocks = budget // pool_block_bytes
        per_request = end[f"{name}_pool_blocks_per_request"]
        reservation_packing = max(0, (blocks - 1) // per_request)
        # Align's previous-state handoff reservation is a phase peak. Do not
        # call packing by that peak an unconditional concurrency upper bound.
        minimum_live_blocks = per_request - (gdn_groups if prefix else 0)
        upper = max(0, (blocks - 1) // minimum_live_blocks)
        capacities[name] = {
            "supplied_profiled_kv_pool_budget_bytes": budget, "pool_blocks": blocks,
            "equal_full_length_requests_upper_bound": upper,
            "equal_full_length_requests_under_peak_reservation": reservation_packing,
            "upper_bound_basis": "Zero sharing, FULL pages plus at least one running GDN state and MTP reserve; phase peaks/previous-state handoff can lower attainable packing.",
            "capped_by_client_and_server": min(upper, client_concurrency, target["max_num_seqs"]),
            "warning": "Uses one null block, zero shared-prefix reuse, and native live GDN reservation; cached evictable prefix pages stay within this same allocated pool. Scheduler/admission/fragmentation may lower concurrency. Do not subtract workspace again from a profiled KV budget.",
        }
    return {
        "evidence_kind": "offline_source_derived_byte_accounting_not_device_measurement",
        "unit": "bytes per TP rank (MiB = bytes / 1048576)",
        "per_request_scope": "Allocated FULL page demand plus native maximum live GDN reservation; not measured whole HBM, not a sum of all cached prefix checkpoints.",
        "workload": {"prompt_tokens": prompt_tokens, "output_tokens": output_tokens,
                     "requests": requests, "client_concurrency": client_concurrency,
                     "note": "120 is the total cohort; it does not mean 120 simultaneously allocated requests."},
        "model_geometry": asdict(geometry), "target_tp": tp,
        "assumptions": ["Uniform FULL geometry and native BF16 K/V; GDN BF16 is explicit in tools/target_cli.py.",
                        f"Prefix caching={prefix}, native mamba_cache_mode={mode}; zero shared-prefix reuse assumed, not prefix caching disabled.",
                        "GDN align reservation is 2+MTP pages/layer, none is 1+MTP; evictable cached checkpoints are reported separately from non-evictable live demand.",
                        "MTP draft FULL layers use the same geometry and the native uniform-page grouping heuristic.",
                        "The prompt length is a client label unless server token usage independently confirms it."],
        "layout": {
            "production_storage": "physical_page_owned_exact_snapshots",
            "prefix_caching": prefix, "native_mamba_cache_mode": mode,
            "native_mamba_block_tokens": native_block if prefix else max_len,
            "native_mamba_mode_request": requested_mode,
            "dtype": {"full_current_and_window": "bfloat16", "history_codes": "int2",
                      "history_scale_zero": "float16", "gdn_conv": "bfloat16", "gdn_ssm": "bfloat16"},
            "conv_bytes": conv_bytes, "ssm_bytes": ssm_bytes, "native_padded_page_bytes": page_bytes,
            "native_full_block_tokens": native_block, "oscar_full_block_tokens": page.block_size,
            "int2_token_bytes": slot.token_bytes, "bf16_token_bytes": slot.bf16_token_bytes,
            "encoding_only_ratio": slot.bf16_token_bytes / slot.token_bytes,
            "int2_payload_per_page_bytes": page.payload_bytes,
            "snapshot_rows_per_physical_page": window.rows,
            "snapshot_bytes_per_physical_page": window.required_bytes,
            "snapshot_tag_bytes_per_page": window.rows * 8,
            "shared_pool_width": width, "full_groups": full_groups, "gdn_groups": gdn_groups,
            "pool_block_bytes": pool_block_bytes, "gdn_states_per_layer_per_request": gdn_state_blocks,
            "gdn_conv_ssm_payload_per_request_bytes": gdn_tight,
            "gdn_group_padding_slots": gdn_groups * width - geometry.gdn_layers,
            "one_null_block_bytes": pool_block_bytes,
        },
        "prefix_sharing_and_retention": {
            "shared_prefix_tokens_assumed": 0,
            "sharing_savings_bytes_claimed": 0,
            "actual_unique_prefix_blocks": None,
            "actual_gdn_cached_checkpoint_blocks": None,
            "cached_blocks_accounting": "Native BlockPool retains completed hashes in the same fixed allocated pool. Zero-refcount cached blocks are evictable and count as free for admission, not as an additional tensor allocation or guaranteed shared-prefix saving.",
            "scope": "No input/token-hash evidence was supplied; zero sharing is the comparison baseline. Cache residency/hit ratios and shared GDN checkpoint lifetimes require actual native pool metadata.",
        },
        "source_contracts": {
            "native_modes": "references/vllm/vllm/model_executor/models/config.py::MambaModelConfig.verify_and_update_config",
            "native_gdn_reservation": "references/vllm/vllm/v1/kv_cache_interface.py::MambaSpec.max_memory_usage_bytes",
            "native_page_geometry": "references/vllm-ascend/vllm_ascend/patch/platform/patch_mamba_config.py::verify_and_update_config",
            "native_pool": "references/vllm/vllm/v1/core/block_pool.py::BlockPool",
            "oscar_page_geometry": "oscar_ascend/layout.py::HybridPageLayout",
            "exact_snapshots": "oscar_ascend/lifecycle.py::SnapshotLayout",
            "research_none_layout": "oscar_ascend/dual_region_layout.py::DualRegionPageLayout",
        },
        "per_request": rows,
        "fixed_external_allocations": {
            "variants": variants,
            "persistent_rotation_bytes_per_rank": (3 * geometry.target_full_layers + 1) * geometry.head_dim ** 2 * 4,
            "rotation_basis": "Three FP32 matrices per target FULL layer, one shared identity matrix for draft layers (runtime.py/rotations.py).",
            "native_model_weights_graph_buffers_allocator_reserve_bytes": None,
            "native_and_operator_temporary_peak_bytes": None,
            "whole_hbm_compression_ratio": None,
        },
        "capacity_from_supplied_profiled_budgets": capacities,
        "cohort_kv_upper_all_at_full_length": {
            str(k): {"native_bytes_per_rank": k * end["native_total_kv_bytes"] + pool_block_bytes,
                     "oscar_bytes_per_rank": k * end["oscar_total_kv_bytes"] + pool_block_bytes}
            for k in sorted({12, 26, min(requests, client_concurrency)})
        },
        "unknowns": ["Current target model config and KVCacheSpec/allocator snapshot unless provided separately.",
                     "Actual native and OSCAR post-profile KV budgets unless supplied via CLI.",
                     "Current captured-graph allocations, native FIA temporaries, allocator fragmentation and physical HBM peaks.",
                     "Whether observed Running requests all hold their complete 64K histories at the same instant."],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/target.json")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-config", type=Path)
    source.add_argument("--reference-geometry", action="store_true",
                        help="Explicit conditional 16+1 FULL/48 GDN geometry; not current-target confirmation")
    parser.add_argument("--prompt-tokens", type=int, default=65100)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--requests", type=int, default=120)
    parser.add_argument("--client-concurrency", type=int, default=32)
    parser.add_argument("--cube-cores", type=int,
                        help="Explicit measured or hypothetical selected Cube count; absent means unknown scratch bytes")
    parser.add_argument("--native-kv-budget-bytes", type=int)
    parser.add_argument("--oscar-kv-budget-bytes", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    target_raw = args.config.read_bytes()
    target = json.loads(target_raw)
    geometry = reference_geometry() if args.reference_geometry else geometry_from_model(json.loads(args.model_config.read_bytes()), target)
    result = analyze(target, geometry, prompt_tokens=args.prompt_tokens, output_tokens=args.output_tokens,
                     requests=args.requests, client_concurrency=args.client_concurrency,
                     cube_cores=args.cube_cores, native_kv_budget_bytes=args.native_kv_budget_bytes,
                     oscar_kv_budget_bytes=args.oscar_kv_budget_bytes)
    result["input_provenance"] = {
        "target_config": str(args.config), "target_config_sha256": hashlib.sha256(target_raw).hexdigest(),
        "model_geometry_status": "conditional_reference_geometry_not_live_verified" if args.reference_geometry else "derived_from_supplied_model_config_not_runtime_verified",
        "model_config": str(args.model_config) if args.model_config else None,
        "model_config_sha256": hashlib.sha256(args.model_config.read_bytes()).hexdigest() if args.model_config else None,
        "cube_cores_status": "unknown" if args.cube_cores is None else "caller_supplied_not_live_verified",
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
        print(f"Wrote offline byte ledger: {args.output}")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
