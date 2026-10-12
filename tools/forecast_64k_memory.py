"""Conditional 64K dual-region memory forecast; no device or speed estimate.

档案 #127/#128: physical pages and metadata128 are different; preserve native
GDN state accounting. #129/#137/#140: Running and client wall are not bytes or
device timing. #151/#153/#154: isolated output and host arithmetic are not
device acceptance. #152/#155: capacity cannot establish throughput improvement.
Design: docs/compact_pool_64k_design.md section 11.8, 128-slot exact sidecar.
Reference: references/vllm/vllm/v1/core/block_pool.py:get_usage excludes null.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


MIB = 1 << 20
GIB = 1 << 30


@dataclass(frozen=True)
class ForecastInputs:
    usable_pool_blocks_min: int = 960
    usable_pool_blocks_max: int = 986
    shared_pool_width: int = 17
    physical_page_bytes: int = 817152
    old_full_blocks_per_request: int = 24
    new_full_blocks_per_request: int = 12
    native_full_blocks_at_output_end: int = 86
    gdn_blocks_per_request: int = 12
    sidecar_slots: int = 128
    sidecar_rows: int = 323
    sidecar_row_bytes: int = 1032
    old_workspace_bytes: int = 1259537664
    new_workspace_bytes: int = 923761920
    cohort_concurrency: int = 32
    prompt_tokens: int = 65100
    output_tokens: int = 512
    lookahead_tokens: int = 3
    bf16_kv_row_bytes: int = 1024

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in asdict(self).values()):
            raise ValueError("Every input must be a positive integer")
        if self.usable_pool_blocks_min > self.usable_pool_blocks_max:
            raise ValueError("Usable pool range is reversed")
        if self.sidecar_slots < self.cohort_concurrency:
            raise ValueError("Sidecar slots must cover the requested cohort")

    @property
    def pool_block_bytes(self) -> int:
        return self.shared_pool_width * self.physical_page_bytes

    @property
    def sidecar_per_slot_bytes(self) -> int:
        return self.sidecar_rows * self.sidecar_row_bytes * self.shared_pool_width

    @property
    def sidecar_bytes(self) -> int:
        return self.sidecar_slots * self.sidecar_per_slot_bytes

    @property
    def new_request_blocks(self) -> int:
        return self.new_full_blocks_per_request + self.gdn_blocks_per_request

    @property
    def request_metadata_bytes(self) -> int:
        # One int32 status per request per FULL layer and one shared int64
        # (slot,epoch) device map. Pinned CPU mirror is not charged as HBM.
        return self.sidecar_slots * (4 * self.shared_pool_width + 16)

    @property
    def net_known_nonkv_increase_bytes(self) -> int:
        return self.sidecar_bytes + self.request_metadata_bytes + self.new_workspace_bytes - self.old_workspace_bytes


def capacity_from_profiled_budget(old_profiled_budget: int, extra_peak_bytes: int,
                                  inputs: ForecastInputs) -> dict:
    """Charge the change in non-KV allocation exactly once against old budget."""
    if type(old_profiled_budget) is not int or old_profiled_budget < 0:
        raise ValueError("Old profiled budget must be a nonnegative integer")
    if type(extra_peak_bytes) is not int or extra_peak_bytes < 0:
        raise ValueError("Additional peak must be a nonnegative integer")
    pool_budget = old_profiled_budget - inputs.net_known_nonkv_increase_bytes - extra_peak_bytes
    configured_blocks = max(0, pool_budget // inputs.pool_block_bytes)
    usable_blocks = max(0, configured_blocks - 1)
    request_capacity = min(inputs.sidecar_slots, usable_blocks // inputs.new_request_blocks)
    return {
        "old_profiled_budget_bytes": old_profiled_budget,
        "new_pool_budget_bytes": pool_budget,
        "configured_num_gpu_blocks": configured_blocks,
        "usable_pool_blocks_excluding_null": usable_blocks,
        "full_length_request_capacity": request_capacity,
        "client_capped_residency_capacity": min(request_capacity, inputs.cohort_concurrency),
        "k32_fits_byte_budget": request_capacity >= inputs.cohort_concurrency,
        "null_block_fits": configured_blocks >= 1,
        "pool_rounding_unused_bytes": pool_budget % inputs.pool_block_bytes if pool_budget >= 0 else None,
    }


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator


def build_forecast(inputs: ForecastInputs, source: dict | None = None) -> dict:
    p = inputs.pool_block_bytes
    allocated_lo = (inputs.usable_pool_blocks_min + 1) * p
    allocated_hi = (inputs.usable_pool_blocks_max + 1) * p
    # If old num_blocks=floor(effective_profiled_budget/P), the unused residue
    # is an unknown integer in [0,P-1]. Do not claim the allocation is the budget.
    profiled_hi = allocated_hi + p - 1
    new_k_pool = (inputs.cohort_concurrency * inputs.new_request_blocks + 1) * p
    headroom = [allocated_lo - inputs.net_known_nonkv_increase_bytes - new_k_pool,
                profiled_hi - inputs.net_known_nonkv_increase_bytes - new_k_pool]
    sidecar_per_request_at_k = inputs.sidecar_bytes / inputs.cohort_concurrency
    native_full = inputs.native_full_blocks_at_output_end * p
    native_all = native_full + inputs.gdn_blocks_per_request * p
    logical_full = (inputs.prompt_tokens + inputs.output_tokens) * inputs.bf16_kv_row_bytes * inputs.shared_pool_width
    new_full_pool = inputs.new_full_blocks_per_request * p
    gdn_pool = inputs.gdn_blocks_per_request * p
    active_full = new_full_pool + inputs.sidecar_per_slot_bytes
    amortized_full = new_full_pool + sidecar_per_request_at_k
    scenarios = []
    for peak in (0, 512 * MIB, GIB, 2 * GIB):
        conservative = capacity_from_profiled_budget(allocated_lo, peak, inputs)
        upper_zero_residue = capacity_from_profiled_budget(allocated_hi, peak, inputs)
        upper_with_residue = capacity_from_profiled_budget(profiled_hi, peak, inputs)
        scenarios.append({
            "additional_unknown_nonkv_peak_bytes": peak,
            "additional_unknown_nonkv_peak_mib": peak / MIB,
            "lower_budget_case": conservative,
            "upper_pool_allocation_no_residue_case": upper_zero_residue,
            "upper_budget_with_possible_rounding_residue_case": upper_with_residue,
            "full_length_request_capacity_range": [conservative["full_length_request_capacity"],
                                                    upper_with_residue["full_length_request_capacity"]],
            "client_capped_capacity_range": [conservative["client_capped_residency_capacity"],
                                             upper_with_residue["client_capped_residency_capacity"]],
            "k32_fits_all_conditional_budget_cases": conservative["k32_fits_byte_budget"],
            "k32_remaining_extra_headroom_bytes_range": [h - peak for h in headroom],
        })
    return {
        "schema_version": 1,
        "status": "conditional_byte_forecast_not_device_measurement_or_speed_prediction",
        "unit": "bytes per TP rank; MiB=2^20 bytes; GiB=2^30 bytes",
        "archive_refs": [127, 128, 129, 137, 140, 151, 152, 153, 154, 155],
        "source": source or {},
        "inputs": asdict(inputs),
        "assumptions": [
            "Warmup R1/W0/3.7% inference usable960-986 applies to the old run: complete prompt resident, 36 used blocks, no other allocation. This is not a measured pool size.",
            "Reference geometry is correct: 17 FULL layers per rank, Hkv1/D256, BF16 row1024B; 48 GDN layers retain four states in three native groups, charged12 shared-pool blocks/request.",
            "Dual-region B5632 versus old B2816 yields12 versus24 FULL pages for65100+512+3; physical page817152B and shared width17 remain unchanged.",
            "Exactly128 sidecar slots are physically reserved at323*(1024+8)*17 bytes each, including int64 tags. All128 are charged even at clientK32.",
            "Old workspace1259537664B was already deducted in the old KV profiling; new workspace923761920B replaces it. Only new-minus-old workspace and the newly added sidecar are charged.",
            "Effective rank-limiting old KV budget is floored by the same block cost with no num_gpu_blocks_override or other capacity cap. Its unknown floor residue is bounded by one block minus one byte.",
            "All old non-KV costs other than the stated changes remain in the original profiled budget. Additional activation/graph/metadata/allocator peak relative to that run is unknown and represented only by explicit scenarios.",
            "These are full-length residency byte capacities. Scheduler/MTP/lifecycle compatibility, allocator fragmentation, native activation shape changes and actual tensor allocations require separate evidence.",
        ],
        "budget_ledger": {
            "workspace_reference_formula": {
                "scope": "WorkspaceGeometry.buffer_bytes for Hq6/Hkv1/D256, splits1,20Cube,C16; source-derived arithmetic, not measured allocation.",
                "formula": "4596*input_token_capacity + 30792*history_query_token_capacity + 99943680",
                "old_input_capacity": 32768,
                "old_history_query_capacity": 32768,
                "new_input_capacity": 66048,
                "new_history_query_capacity": 16896,
                "old_formula_bytes": 4596 * 32768 + 30792 * 32768 + 99943680,
                "new_formula_bytes": 4596 * 66048 + 30792 * 16896 + 99943680,
                "given_workspace_inputs_match_formula": (
                    inputs.old_workspace_bytes == 4596 * 32768 + 30792 * 32768 + 99943680
                    and inputs.new_workspace_bytes == 4596 * 66048 + 30792 * 16896 + 99943680),
            },
            "one_shared_pool_block_bytes": p,
            "one_null_block_bytes": p,
            "old_allocated_pool_bytes_range": [allocated_lo, allocated_hi],
            "old_effective_profiled_budget_bytes_range_including_floor_residue": [allocated_lo, profiled_hi],
            "old_unallocated_floor_residue_bytes_range": [0, p - 1],
            "old_full_length_request_blocks": inputs.old_full_blocks_per_request + inputs.gdn_blocks_per_request,
            "old_full_length_request_capacity_range": [
                inputs.usable_pool_blocks_min // (inputs.old_full_blocks_per_request + inputs.gdn_blocks_per_request),
                inputs.usable_pool_blocks_max // (inputs.old_full_blocks_per_request + inputs.gdn_blocks_per_request)],
            "new_full_length_request_blocks": inputs.new_request_blocks,
            "sidecar_per_slot_bytes": inputs.sidecar_per_slot_bytes,
            "sidecar_reserved_bytes": inputs.sidecar_bytes,
            "sidecar_reserved_mib": inputs.sidecar_bytes / MIB,
            "request_status_and_device_map_bytes": inputs.request_metadata_bytes,
            "workspace_reclaimed_bytes": inputs.old_workspace_bytes - inputs.new_workspace_bytes,
            "workspace_reclaimed_mib": (inputs.old_workspace_bytes - inputs.new_workspace_bytes) / MIB,
            "net_known_nonkv_increase_bytes": inputs.net_known_nonkv_increase_bytes,
            "net_known_nonkv_increase_mib": inputs.net_known_nonkv_increase_bytes / MIB,
            "formula": "new_pool_budget = old_effective_profiled_budget - (new_workspace - old_workspace + sidecar128 + request_status_and_device_map) - additional_unknown_nonkv_peak; new_usable=floor(new_pool_budget/P)-1",
            "no_double_counting": "Do not subtract the old workspace, unchanged rotations, or all128 sidecar slots again after applying this net change.",
        },
        "per_request_output_end_byte_comparisons": {
            "page_counts_also_cover_output_end_plus_mtp3": True,
            "native_full_physical_bytes": native_full,
            "native_full_logical_bf16_payload_bytes": logical_full,
            "native_all_kv_physical_bytes": native_all,
            "new_full_pool_physical_bytes": new_full_pool,
            "unchanged_gdn_pool_physical_bytes": gdn_pool,
            "full_active_one_slot": {
                "bytes": active_full, "mib": active_full / MIB,
                "native_full_physical_ratio": ratio(native_full, active_full),
                "native_full_logical_payload_ratio": ratio(logical_full, active_full),
                "scope": "Active FULL charge plus one exact ring. This is not the physically reserved128-slot sidecar amortization at K32.",
            },
            "full_reserved_128_slots_amortized_at_k32": {
                "bytes_per_request": amortized_full, "mib_per_request": amortized_full / MIB,
                "sidecar_bytes_per_request": sidecar_per_request_at_k,
                "native_full_physical_ratio": ratio(native_full, amortized_full),
                "native_full_logical_payload_ratio": ratio(logical_full, amortized_full),
            },
            "all_kv_active_one_slot": {
                "bytes": active_full + gdn_pool, "mib": (active_full + gdn_pool) / MIB,
                "native_all_kv_physical_ratio": ratio(native_all, active_full + gdn_pool),
            },
            "all_kv_reserved_128_slots_amortized_at_k32": {
                "bytes_per_request": amortized_full + gdn_pool,
                "mib_per_request": (amortized_full + gdn_pool) / MIB,
                "native_all_kv_physical_ratio": ratio(native_all, amortized_full + gdn_pool),
            },
            "scope": "Comparisons exclude shared null, unused pool reserve, workspace and other runtime buffers; complete cohort ledger below includes its one null. No whole-HBM ratio is known.",
        },
        "k32_cohort_ledger": {
            "new_pool_blocks_including_null": inputs.cohort_concurrency * inputs.new_request_blocks + 1,
            "new_pool_bytes_including_null": new_k_pool,
            "new_pool_plus_all128_sidecar_bytes": new_k_pool + inputs.sidecar_bytes,
            "new_pool_plus_sidecar_plus_workspace_bytes": new_k_pool + inputs.sidecar_bytes + inputs.new_workspace_bytes,
            "new_pool_sidecar_workspace_and_request_metadata_bytes": new_k_pool + inputs.sidecar_bytes + inputs.new_workspace_bytes + inputs.request_metadata_bytes,
            "new_pool_and_fixed_arenas_scope": "Excludes model weights, native activation/graph/rotations and other already-profiled unchanged memory; not total HBM.",
            "additional_unknown_peak_allowance_bytes_range": headroom,
            "additional_unknown_peak_allowance_mib_range": [h / MIB for h in headroom],
            "additional_unknown_peak_allowance_gib_range": [h / GIB for h in headroom],
            "upper_allowance_if_old_pool_has_zero_floor_residue_bytes": allocated_hi - inputs.net_known_nonkv_increase_bytes - new_k_pool,
            "allowance_semantics": "At or below the lower endpoint K32 fits every conditional budget; above the upper endpoint no conditional budget fits. Interior values depend on old pool size and its unknown residue. Extra metadata must consume this allowance too.",
        },
        "extra_peak_scenarios": scenarios,
        "unknowns": {
            "actual_old_and_new_profiled_kv_budget_bytes": None,
            "actual_native_activation_and_graph_peak_change_bytes": None,
            "actual_lease_metadata_and_allocator_overhead_bytes": None,
            "actual_npu_hbm_peak_bytes": None,
            "actual_graph_and_service_residency": None,
            "throughput_or_latency_prediction": None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path("reports/evidence_64k_120_20261009.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/forecast_64k_memory_20261009.json"))
    args = parser.parse_args()
    raw = args.evidence.read_bytes()
    evidence = json.loads(raw)
    warmup = evidence["conditional_single_request_warmup_pool_inference"]["oscar"]
    lower, upper = warmup["cases"][0]["usable_pool_blocks_range"]
    inputs = ForecastInputs(usable_pool_blocks_min=lower, usable_pool_blocks_max=upper)
    result = build_forecast(inputs, {
        "evidence_path": args.evidence.as_posix(),
        "evidence_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_log_sha256": evidence["source"]["sha256"],
        "warmup_engine_line": warmup["engine_line"],
        "warmup_kv_usage_pct": warmup["kv_usage_pct"],
        "design": "docs/compact_pool_64k_design.md section11.8",
        "workspace_values": "Current old/new WorkspaceGeometry byte ledger supplied for this task; no device profiling inferred.",
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": args.output.as_posix(), "status": result["status"],
                      "net_nonkv_increase_mib": result["budget_ledger"]["net_known_nonkv_increase_mib"],
                      "capacity_scenarios": [{"extra_peak_mib": r["additional_unknown_nonkv_peak_mib"],
                                              "capacity": r["full_length_request_capacity_range"]}
                                             for r in result["extra_peak_scenarios"]],
                      "k32_extra_peak_allowance_mib": result["k32_cohort_ledger"]["additional_unknown_peak_allowance_mib_range"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
