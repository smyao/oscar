"""Source/control-flow cost evidence only: no device timing or SDK execution."""
from collections import Counter

import pytest

from tools.experiments import batch_kernel_cost_20261019 as profile


@pytest.fixture(scope="module")
def counter(tmp_path_factory):
    return profile.Profiler(tmp_path_factory.mktemp("batch-kernel-count"))


@pytest.mark.parametrize("family", list(profile.FILES))
def test_actual_cube_tail_loop_and_fp32_inverse_rotation(counter, family):
    q = 1 if family == "mdl_q1" else 4
    counts = counter.cube(family, 0, 257, q)
    assert counts["qk_calls"] == counts["pv_calls"] == 2
    assert counts["qk_flops"] == counts["pv_flops"] == 2 * q * 6 * 256 * 256 * 2
    assert counts["rot_calls"] == 1
    assert counts["rot_flops"] == 2 * q * 6 * 256 * 256


def test_precise_short_tile_full_score_clear_is_included(counter):
    short = profile.one_task(counter, "mdl_q4", 2, 0, 4, 4)
    full = profile.one_task(counter, "mdl_q4", 2, 0, 256, 4)
    assert short["score_padding_zero_write"] == 32 * 256 * 4
    assert full["score_padding_zero_write"] == 0
    missing = Counter(short)
    del missing["score_padding_zero_write"]
    assert profile.data_bytes(short) - profile.data_bytes(missing) == 32768
    assert short["qk_flops"] == 2 * 24 * 16 * 256
    assert short["bf16_kv_publish"] == 64 * 256 * 4


def test_executed_unpack_counts_and_steady_tile_resources(counter):
    helpers = {str(h): counter.unpack(h) for h in (16, 32, 64)}
    for h in (16, 32, 64):
        assert helpers[str(h)]["Gather"] == 1 + h // 8
        assert helpers[str(h)]["ShiftRight"] == helpers[str(h)]["And"] == 16
    q1, q4 = profile.tile_resources(counter, helpers)
    assert q1["resources"]["temporary_gm_min_bytes"] == 568320
    assert q4["resources"]["temporary_gm_min_bytes"] == 667648
    assert q4["resources"]["persistent_source_bytes"] == 34832
    assert q1["resources"]["bf16_qk_pv_flops"] * 4 == q4["resources"]["bf16_qk_pv_flops"]
    assert q4["unpack_vector_API_invocations"]["Gather"] == 72
    # The history pipeline has its own protocol; count its extracted body.
    assert q4["counts"]["cube_wait_api"] > 0


def test_retained_later_q1_shape_drives_real_runtime_splits(counter):
    packed = profile.scenario(counter, 20, 1, 32768, 20, None)
    retained = profile.scenario(counter, 20, 1, 32768, 80, None)
    assert packed["splits"] == 20
    assert retained["splits"] == 5
    assert retained["descriptor_scan_rows"] == 80 * 3 * 5
    assert sum(retained["history_tasks_per_core"]) == 20 * 5
    assert retained["counts"]["rot_calls"] == 20 * 5


def test_step_eager_draft_extent_and_native_current_exclusion(counter):
    step = profile.full_layer_step(counter, 46, 32768, [196, 512])
    assert step["target_q4"]["padded_tokens"] == 196
    assert step["first_q4"]["padded_tokens"] == 184
    assert step["later_q1_retained_eager_first_n"]["padded_tokens"] == 184
    assert step["later_q1_retained_eager_first_n"]["splits"] == 3
    assert step["first_q4"]["per_request_source_counts"][2] == {}
    all_sources = profile.scenario(counter, 46, 4, 32768, 184, None)
    omitted = all_sources["per_request_source_counts"][2]
    for key in ("qk_flops", "pv_flops", "score_padding_zero_write", "q_payload_read"):
        assert all_sources["counts"][key] - step["first_q4"]["counts"][key] == omitted[key] * 46
    assert step["omitted_native_current_FIA"]["internal_bytes_flops_latency"] == "not modelled"


def test_counter_detects_mutated_actual_cube_stride(monkeypatch, tmp_path):
    source = profile.cpp_program()
    needle = "start+=kKvRows"
    assert needle in source
    # Deliberate source mutation: double the extracted actual tile stride.
    # A formula-only estimator would remain at two tiles and miss this.
    monkeypatch.setattr(profile, "cpp_program", lambda: source.replace(needle, "start+=2*kKvRows"))
    mutated = profile.Profiler(tmp_path)
    counts = mutated.cube("mdl_q4", 0, 257, 4)
    assert counts["qk_calls"] == counts["pv_calls"] == 1
    assert counts["qk_flops"] != 2 * 24 * 256 * 256 * 2
