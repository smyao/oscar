"""Archive #170: original-space attention evidence, never a clip proxy gate."""
import torch
import pytest

from csrc.experiments.native_exact_current_20261028.counterexample import fitted_rotations, evaluate_case


@pytest.fixture(scope="module")
def rotations():
    torch.set_num_threads(1)
    return fitted_rotations()


def test_actual_recipe3_single_token_removes_only_extra_rotation_error(rotations):
    case = evaluate_case(1, rotations)
    assert case["original_space_equals_v_exact"]
    assert case["max_abs_old_vs_original_space"] > 0
    assert all(row["equal"] for row in case["writer_inputs"].values())


def test_production_store_producers_remain_equal_for_causal_multiple_queries(rotations):
    case = evaluate_case(33, rotations)
    assert all(row["equal"] and row["before_sha256"] == row["after_sha256"]
               for row in case["writer_inputs"].values())
