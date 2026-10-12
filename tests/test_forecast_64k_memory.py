"""档案 #127/#128/#137/#140/#151–155: conditional bytes are not NPU evidence.

Expected values are independently stated integer byte arithmetic from design
section11.8 and the supplied warmup bounds, including the null-block boundary.
"""

from dataclasses import replace

import pytest

from tools.forecast_64k_memory import (
    ForecastInputs,
    build_forecast,
    capacity_from_profiled_budget,
)


def test_sidecar_and_workspace_charge_only_the_net_change_once():
    result = build_forecast(ForecastInputs())
    ledger = result["budget_ledger"]
    assert ledger["workspace_reference_formula"]["old_formula_bytes"] == 1259537664
    assert ledger["workspace_reference_formula"]["new_formula_bytes"] == 923761920
    assert ledger["workspace_reference_formula"]["given_workspace_inputs_match_formula"]
    assert ledger["one_shared_pool_block_bytes"] == 13891584
    assert ledger["sidecar_per_slot_bytes"] == 5666712
    assert ledger["sidecar_reserved_bytes"] == 725339136
    assert ledger["workspace_reclaimed_bytes"] == 335775744
    assert ledger["net_known_nonkv_increase_bytes"] == 389574144
    first = result["extra_peak_scenarios"][0]["lower_budget_case"]
    assert first["old_profiled_budget_bytes"] == 13349812224
    assert first["new_pool_budget_bytes"] == 12960238080
    # The old workspace was already deducted in the original profiling.
    assert first["old_profiled_budget_bytes"] - first["new_pool_budget_bytes"] == 389574144


def test_all_additional_peak_scenarios_preserve_client_cap_and_rounding():
    result = build_forecast(ForecastInputs())
    scenarios = result["extra_peak_scenarios"]
    assert [r["additional_unknown_nonkv_peak_bytes"] for r in scenarios] == [0, 536870912, 1073741824, 2147483648]
    assert [r["full_length_request_capacity_range"] for r in scenarios] == [[38, 39], [37, 38], [35, 36], [32, 33]]
    assert [r["client_capped_capacity_range"] for r in scenarios] == [[32, 32]] * 4
    assert all(r["k32_fits_all_conditional_budget_cases"] for r in scenarios)
    assert [r["lower_budget_case"]["usable_pool_blocks_excluding_null"] for r in scenarios] == [931, 893, 854, 777]


def test_k32_extra_peak_boundary_is_inclusive_and_next_byte_fails():
    inputs = ForecastInputs()
    result = build_forecast(inputs)
    budgets = result["budget_ledger"]["old_effective_profiled_budget_bytes_range_including_floor_residue"]
    allowances = result["k32_cohort_ledger"]["additional_unknown_peak_allowance_bytes_range"]
    assert allowances == [2277609984, 2652682751]
    assert result["k32_cohort_ledger"]["new_pool_blocks_including_null"] == 769
    assert result["k32_cohort_ledger"]["new_pool_bytes_including_null"] == 10682628096
    for budget, allowance in zip(budgets, allowances):
        at = capacity_from_profiled_budget(budget, allowance, inputs)
        past = capacity_from_profiled_budget(budget, allowance + 1, inputs)
        assert at["configured_num_gpu_blocks"] == 769
        assert at["full_length_request_capacity"] == 32
        assert past["configured_num_gpu_blocks"] == 768
        assert past["full_length_request_capacity"] == 31


def test_unknown_old_profile_residue_is_separate_from_allocated_pool():
    result = build_forecast(ForecastInputs())
    ledger = result["budget_ledger"]
    assert ledger["old_allocated_pool_bytes_range"] == [13349812224, 13710993408]
    assert ledger["old_unallocated_floor_residue_bytes_range"] == [0, 13891583]
    assert ledger["old_effective_profiled_budget_bytes_range_including_floor_residue"] == [13349812224, 13724884991]
    first = result["extra_peak_scenarios"][0]
    assert first["upper_pool_allocation_no_residue_case"]["usable_pool_blocks_excluding_null"] == 957
    assert first["upper_budget_with_possible_rounding_residue_case"]["usable_pool_blocks_excluding_null"] == 958


def test_full_active_and_all128_reserved_amortization_are_distinct():
    comparisons = build_forecast(ForecastInputs())["per_request_output_end_byte_comparisons"]
    active = comparisons["full_active_one_slot"]
    reserved = comparisons["full_reserved_128_slots_amortized_at_k32"]
    assert active["bytes"] == 172365720
    assert reserved["bytes_per_request"] == 189365856
    assert reserved["sidecar_bytes_per_request"] == 22666848
    assert active["native_full_physical_ratio"] == pytest.approx(6.931054643579942)
    assert reserved["native_full_physical_ratio"] == pytest.approx(6.308825937448829)
    all_active = comparisons["all_kv_active_one_slot"]
    all_reserved = comparisons["all_kv_reserved_128_slots_amortized_at_k32"]
    assert all_active["bytes"] == 339064728
    assert all_reserved["bytes_per_request"] == 356064864
    assert all_active["native_all_kv_physical_ratio"] == pytest.approx(4.0150895082192095)
    assert all_reserved["native_all_kv_physical_ratio"] == pytest.approx(3.823391099886789)
    assert active["native_full_logical_payload_ratio"] == pytest.approx(6.626455051503281)
    assert reserved["native_full_logical_payload_ratio"] == pytest.approx(6.031571478228894)


def test_below_null_budget_reports_zero_capacity_without_negative_blocks():
    inputs = ForecastInputs()
    no_null = capacity_from_profiled_budget(inputs.net_known_nonkv_increase_bytes + inputs.pool_block_bytes - 1, 0, inputs)
    assert not no_null["null_block_fits"]
    assert no_null["usable_pool_blocks_excluding_null"] == 0
    assert no_null["full_length_request_capacity"] == 0
    exhausted = capacity_from_profiled_budget(0, 1, inputs)
    assert exhausted["new_pool_budget_bytes"] < 0
    assert exhausted["full_length_request_capacity"] == 0
    assert exhausted["pool_rounding_unused_bytes"] is None


@pytest.mark.parametrize("changes", [
    {"usable_pool_blocks_min": 987},
    {"sidecar_slots": 31},
    {"physical_page_bytes": 0},
    {"new_workspace_bytes": -1},
])
def test_invalid_assumptions_fail_explicitly(changes):
    with pytest.raises(ValueError):
        replace(ForecastInputs(), **changes)


def test_unknown_peaks_and_performance_remain_unmeasured():
    result = build_forecast(ForecastInputs())
    assert all(value is None for value in result["unknowns"].values())
    assert "not_device_measurement_or_speed_prediction" in result["status"]
