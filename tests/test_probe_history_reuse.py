"""Archive #126/#129/#140-145/#148-152: fe0-versus-C4/q1 host contracts.

These tests validate probe inputs and fail-closed evidence, never claim NPU
accuracy, graph capture or a production performance improvement.
"""

import json
from types import SimpleNamespace

import pytest

from tools import probe_history_reuse as reuse
from oscar_ascend.ops.cv_dispatch import (FAST_CV_OP, FAST_WEIGHTED_CV_OP,
                                          select_source_splits)


def test_active_device_accepts_any_explicit_four_card_target(monkeypatch):
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    reuse._active_device({"devices": [4, 5, 6, 7], "soc_version": "ascend910b4"})
    assert reuse.os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "4,5,6,7"


@pytest.mark.parametrize("devices", (
    None, [4, 5, 6], [4, 5, 6, 6], [4, 5, 6, -1], [4, 5, 6, True],
    [4, 5, 6, []],
))
def test_active_device_rejects_invalid_explicit_targets(monkeypatch, devices):
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    with pytest.raises(reuse.HistoryReuseProbeError, match="four explicit unique"):
        reuse._active_device({"devices": devices, "soc_version": "ascend910b4"})


def test_active_device_rejects_soc_or_inherited_selection_mismatch(monkeypatch):
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    with pytest.raises(reuse.HistoryReuseProbeError, match="ascend910b4"):
        reuse._active_device({"devices": [4, 5, 6, 7], "soc_version": "ascend910b3"})
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")
    with pytest.raises(reuse.HistoryReuseProbeError, match="differs from target"):
        reuse._active_device({"devices": [4, 5, 6, 7], "soc_version": "ascend910b4"})


def test_small_fixture_is_deterministic_and_uses_disjoint_physical_pages():
    torch = pytest.importorskip("torch")
    shape = reuse.Shape("small", (4, 6), (17, 65), 64, 1, 1, False)
    first = reuse.make_fixture(torch, shape)
    second = reuse.make_fixture(torch, shape)
    assert first["input_sha256"] == second["input_sha256"]
    assert len(first["expected"]) == sum(shape.qlens)
    pages = [page for request in first["page_assignments"] for page in request]
    assert sorted(pages) == list(range(first["blocks"]))
    assert first["cpu"]["table"].shape[0] == len(shape.qlens)
    assert bool((first["cpu"]["slots"] >= 0).all())


def test_decode_measurement_uses_production_splits_not_old_s1():
    # #150 follow-up: N128 has seven global query tiles and needs S3 for
    # twenty Cubes in the current runtime. Other oracle shapes stay fixed.
    target = {"max_num_batched_tokens": 16384}
    shape = next(case for case in reuse.CASES if case.name == "decode32")
    assert shape.splits == 1
    measured = reuse.measurement_shape(shape, target, 20)
    assert measured.splits == 3 and measured.qlens == shape.qlens
    assert measured.contexts == shape.contexts
    assert reuse.measurement_shape(shape, target, 1).splits == 1
    for case in reuse.CASES:
        if case.name != "decode32":
            assert reuse.measurement_shape(case, target, 20) is case


def test_q1_shapes_keep_real_32_request_lengths_and_both_native_padding_sizes():
    assert len(reuse.Q1_CASES) == 2
    small, large = reuse.Q1_CASES
    assert small.qlens == large.qlens == (1,) * 32
    assert small.contexts == large.contexts == (20000, 23000, 27000, 30000) * 8
    assert (small.padded_tokens, small.splits) == (128, 3)
    assert (large.padded_tokens, large.splits) == (16384, 1)
    assert small.slot_context and large.slot_context
    assert reuse.baseline_workspace_per_core(small.dim) == reuse.baseline_workspace_per_core(large.dim)
    for shape in reuse.Q1_CASES:
        shape.validate()
        assert reuse.select_cv_op(4, shape.heads, shape.kv_heads,
                                  shape.padded_tokens, 1, q1_draft=True) == reuse.Q1_CV_OP
        assert reuse.select_cv_op(4, shape.heads, shape.kv_heads,
                                  shape.padded_tokens, 1) == reuse.FE0_CV_OP


def test_q1_padding_fixture_has_32_real_slots_and_96_negative_holes():
    torch = pytest.importorskip("torch")
    shape = reuse.Shape("q1_padding_contract", (1,) * 32, (17,) * 32,
                        64, 1, 3, False, padded_tokens=128, slot_context=True)
    fixture = reuse.make_fixture(torch, shape)
    assert fixture["tokens"] == 128 and fixture["actual_tokens"] == 32
    assert fixture["cpu"]["starts"].tolist() == list(range(33))
    assert fixture["cpu"]["slots"].shape == (128,)
    assert bool((fixture["cpu"]["slots"][:32] >= 0).all())
    assert fixture["cpu"]["slots"][32:].tolist() == [-1] * 96
    assert len(fixture["expected"]) == 32
    tasks = torch.zeros((128, 1, 3, 3, 16), dtype=torch.int64)
    tasks[32:, ..., 1] = -1
    positions = torch.tensor([17] * 32 + [-1] * 96, dtype=torch.int64)
    reuse._check_padding_tasks(torch, fixture, tasks, positions)
    tasks[40, 0, 0, 0, 1] = 0
    with pytest.raises(reuse.HistoryReuseProbeError, match="harmless qcount=-1"):
        reuse._check_padding_tasks(torch, fixture, tasks, positions)
    tasks[40, 0, 0, 0, 1] = -1
    positions[40] = 0
    with pytest.raises(reuse.HistoryReuseProbeError, match="slots/positions"):
        reuse._check_padding_tasks(torch, fixture, tasks, positions)


def test_invalid_live_and_dead_tail_corrupt_distinct_slots():
    torch = pytest.importorskip("torch")
    valid = reuse.make_fixture(torch, reuse.Shape("v", (85,), (511,), 64, 1, 1, False))
    live = reuse.make_fixture(torch, reuse.Shape("l", (85,), (511,), 64, 1, 1,
                                                 False, corrupt_metadata=True))
    dead = reuse.make_fixture(torch, reuse.Shape("d", (85,), (511,), 64, 1, 1,
                                                 False, corrupt_dead_tail=True))
    assert valid["input_sha256"] != live["input_sha256"] != dead["input_sha256"]
    assert torch.equal(valid["cpu"]["q"], live["cpu"]["q"])
    assert torch.equal(valid["cpu"]["q"], dead["cpu"]["q"])
    assert not torch.equal(valid["cpu"]["raw"], live["cpu"]["raw"])
    assert not torch.equal(valid["cpu"]["raw"], dead["cpu"]["raw"])


def test_mature_shared_faults_are_distinct_and_keep_full_group_geometry():
    torch = pytest.importorskip("torch")
    meta_shape, query_shape = reuse.CASES[-2:]
    assert meta_shape.qlens == query_shape.qlens == (168,)
    assert meta_shape.recent_tokens == query_shape.recent_tokens == 32
    meta = reuse.make_fixture(torch, meta_shape)
    query = reuse.make_fixture(torch, query_shape)
    assert not torch.equal(meta["cpu"]["raw"], query["cpu"]["raw"])
    assert torch.isnan(query["cpu"]["qr"][105, 0, 0])
    assert bool(torch.isfinite(meta["cpu"]["qr"]).all())
    assert meta["spec"].expect_clusters and query["spec"].expect_clusters


def test_unaligned_second_request_and_terminal_split_counterfactual():
    torch = pytest.importorskip("torch")
    shape = next(case for case in reuse.CASES
                 if case.name == "mixed_unaligned_mature_s3")
    fixture = reuse.make_fixture(torch, shape)
    assert shape.speed_gate is False and shape.required_clusters == 2
    assert shape.kv_heads == 2 and shape.splits == 3
    assert fixture["cpu"]["starts"].tolist() == [0, 3, 388]
    assert 3 % 21 != 0 and 3 % 84 != 0
    tasks = torch.zeros((388, 2, 3, 3, 16), dtype=torch.int64)
    for token in (3 + 252, 3 + 273, 3 + 294, 3 + 315):
        tasks[token, :, 0, 2, 1] = 21
        tasks[token, :, 0, 2, 3] = 534
        tasks[token, :, 0, 2, 4] = 769
        tasks[token, :, 0, 2, 8] = 769
        tasks[token, :, 0, 1, 1] = 21
        tasks[token, :, 0, 1, 3] = 299
        tasks[token, :, 0, 1, 4] = 534
        tasks[token, :, 0, 1, 8] = 769
    before_nonterminal = tasks[:, :, 0, 1].clone()
    assert reuse._disable_terminal_source0_split(tasks, shape) == 8
    assert torch.equal(tasks[:, :, 0, 1], before_nonterminal)
    assert all(int(tasks[token, head, 0, 2, 4]) == 768
               for token in (255, 276, 297, 318) for head in (0, 1))


def test_source0_leader_status_keeps_each_group_and_lane():
    torch = pytest.importorskip("torch")
    shape = reuse.Shape("status", (4,), (17,), 64, 1, 1, False)
    tasks = torch.zeros((4 * 3, 16), dtype=torch.int64)
    tasks[0, 1] = 4
    status = torch.zeros((4 * 3, 2), dtype=torch.int32)
    status[0] = torch.tensor((3, 2), dtype=torch.int32)
    summary = reuse._source0_leader_status({"spec": shape, "tokens": 4},
                                           {"tasks": tasks, "status": status})
    assert summary["leaders"] == 1
    assert summary["nonzero_leaders"] == [{"token": 0, "kv_head": 0,
        "split": 0, "qcount": 4, "status_aiv": [3, 2]}]


def test_candidate_argument_is_inserted_after_workspace_before_attributes():
    calls = []
    ops = SimpleNamespace(
        attention_cv_out=lambda *args: calls.append(("fe0", args)),
        attention_cv_cluster4_out=lambda *args: calls.append(("C4", args)),
        attention_cv_q1_out=lambda *args: calls.append(("q1", args)))
    tensors = {key: key for key in ("q", "qr", "ck", "cv", "rv", "raw",
                                    "table", "wk", "wv", "tags")}
    buffers = {key: key for key in ("tasks", "partial", "lse", "status", "workspace")}
    buffers["cluster_stats"] = "stats"
    fixture = {"spec": reuse.Shape("x", (1,), (17,), 64, 1, 1, False),
               "blocks": 2, "stride": 20480, "scale": 0.125, "tokens": 1}
    reuse._launch(ops, tensors, fixture, buffers, 2, candidate=False)
    reuse._launch(ops, tensors, fixture, buffers, 2, candidate=True)
    assert calls[0][0] == "fe0" and calls[1][0] == "C4"
    assert calls[1][1][15] == "stats"
    assert calls[1][1][:15] == calls[0][1][:15]
    assert calls[1][1][16:] == calls[0][1][15:]
    # #150: raw C4 remains exercised for accuracy; production q1/q4 routes
    # the same original INT2 symbol and exact ABI with the candidate workspace.
    route = reuse._launch(ops, tensors, fixture, buffers, 2, candidate=True,
                          production_route=True)
    assert route == reuse.FE0_CV_OP
    assert calls[2] == calls[0]
    q1_route = reuse._launch(ops, tensors, fixture, buffers, 2, candidate=True,
                             q1_schedule=True, production_route=True)
    assert q1_route == reuse.Q1_CV_OP
    assert calls[3][0] == "q1" and calls[3][1] == calls[0][1]


@pytest.mark.parametrize("heads,kv_heads,tokens,max_query_len,expected", [
    (6, 1, 128, 4, reuse.FE0_CV_OP),  # 32 q4 requests, not one q128
    (6, 1, 512, 1, reuse.FE0_CV_OP),  # graph padding cannot create a cluster
    (6, 1, 83, 83, reuse.FE0_CV_OP),
    (6, 1, 84, 84, reuse.CANDIDATE_OP),
    (12, 2, 388, 385, reuse.CANDIDATE_OP),
    (4, 1, 127, 127, reuse.FE0_CV_OP),
    (4, 1, 128, 128, reuse.CANDIDATE_OP),
    (6, 1, 512, None, reuse.CANDIDATE_OP),
])
def test_production_dispatch_requires_four_groups_in_one_request(heads, kv_heads,
                                                                tokens, max_query_len, expected):
    # #150: eligibility is a structural bound, never inferred from timing.
    assert reuse.select_cv_op(4, heads, kv_heads, tokens, max_query_len) == expected
    assert reuse.select_cv_op(1, heads, kv_heads, tokens, max_query_len) == reuse.FE0_CV_OP


def test_target_proven_q4_routes_weighted_only_in_candidate_geometry():
    common = dict(q1_draft=False, fast_unpack=True, weighted_q4=True, head_dim=256)
    assert reuse.select_cv_op(4, 6, 1, 128, 4, **common) == FAST_WEIGHTED_CV_OP
    # Do not broaden the single target proof to another q length, D, GQA,
    # padded capacity, baseline, or a candidate with weighted disabled.
    assert reuse.select_cv_op(4, 6, 1, 128, 3, **common) == FAST_CV_OP
    assert reuse.select_cv_op(4, 6, 1, 256, 4, **common) == FAST_CV_OP
    assert reuse.select_cv_op(4, 6, 1, 128, 4, **{**common, "head_dim": 128}) == FAST_CV_OP
    assert reuse.select_cv_op(4, 12, 2, 128, 4, **common) == FAST_CV_OP
    assert reuse.select_cv_op(4, 6, 1, 128, 4, fast_unpack=True,
                              weighted_q4=False, head_dim=256) == FAST_CV_OP
    assert select_source_splits(3, FAST_WEIGHTED_CV_OP,
                                tokens=128, weighted_q4_split2=True) == 2
    assert select_source_splits(3, FAST_CV_OP,
                                tokens=128, weighted_q4_split2=True) == 3
    assert select_source_splits(3, FAST_WEIGHTED_CV_OP,
                                tokens=128, weighted_q4_split2=False) == 3
    assert select_source_splits(20, FAST_WEIGHTED_CV_OP,
                                tokens=4, weighted_q4_split2=True) == 20


def test_cluster_counters_require_real_sharing_and_exact_owner_algebra():
    torch = pytest.importorskip("torch")
    active = torch.tensor([[2, 8, 1, 7, 21, 2, 8, 2], [0] * 8], dtype=torch.int64)
    result = reuse._check_cluster_stats(active, expect_clusters=True,
                                        expected_source0_leaders=9)
    assert result["eligible_clusters"] == 2
    with pytest.raises(reuse.HistoryReuseProbeError, match="leader ownership"):
        reuse._check_cluster_stats(active, expect_clusters=True,
                                   expected_source0_leaders=10)
    with pytest.raises(reuse.HistoryReuseProbeError, match="unexpectedly activated"):
        reuse._check_cluster_stats(active, expect_clusters=False)
    with pytest.raises(reuse.HistoryReuseProbeError, match="did not activate"):
        reuse._check_cluster_stats(torch.zeros_like(active), expect_clusters=True)
    solo = torch.tensor([[0, 0, 3, 0, 0, 4, 0, 0], [0] * 8], dtype=torch.int64)
    assert reuse._check_cluster_stats(solo, expect_clusters=False)["independent_leaders"] == 3
    with pytest.raises(reuse.HistoryReuseProbeError, match="no fe0 C1 history work"):
        reuse._check_cluster_stats(torch.zeros_like(active), expect_clusters=False)
    broken = active.clone()
    broken[0, 4] = 20
    with pytest.raises(reuse.HistoryReuseProbeError, match="inconsistent"):
        reuse._check_cluster_stats(broken, expect_clusters=True)


def test_cluster_counters_two_phase_attribution_is_global_only():
    torch = pytest.importorskip("torch")
    # Archive #149: pass one charges skips to each member tile's core, pass two
    # charges the cluster to the remapped bucket's core. Per-core skips need
    # not equal grouped leaders; the global identity must still hold.
    two_phase = torch.tensor([[1, 4, 0, 2, 6, 0, 0, 1],
                              [0, 0, 0, 0, 0, 0, 4, 0]], dtype=torch.int64)
    result = reuse._check_cluster_stats(two_phase, expect_clusters=True,
                                        expected_source0_leaders=4)
    assert result["eligible_clusters"] == 1
    assert result["grouped_leaders"] == 4
    assert result["original_schedule_skips"] == 4


def test_q1_schedule_gate_requires_both_speed_cases_and_changed_input_graph():
    rows = [{"case": shape.name, "q1_precision": "bitwise_passed",
             "frozen_oracle": "passed", "q1_performance": "passed",
             "production_operator": reuse.Q1_CV_OP, "q1_over_fe0": 0.99}
            for shape in reuse.Q1_CASES]
    graph = {"q1_graph_capture": "passed", "q1_graph_replay": "passed"}
    assert reuse._q1_gate(rows, graph)["q1_schedule_gate"] == "passed"
    with pytest.raises(reuse.HistoryReuseProbeError, match="missing a required"):
        reuse._q1_gate(rows[:1], graph)
    assert reuse._q1_gate(rows, None)["q1_schedule_gate"] == "failed"
    rows[1]["q1_over_fe0"] = 1.0001
    assert reuse._q1_gate(rows, graph)["q1_performance"] == "failed"
    assert reuse._q1_gate(rows, graph)["q1_schedule_gate"] == "failed"


def test_bitwise_comparison_distinguishes_signed_zero_and_nan_payload():
    torch = pytest.importorskip("torch")
    positive = torch.tensor([0.0, float("nan")], dtype=torch.float32)
    same = positive.clone()
    negative_zero = positive.clone()
    negative_zero[0] = -0.0
    assert reuse._bitwise_identical(torch, positive, same)
    assert not reuse._bitwise_identical(torch, positive, negative_zero)
    assert reuse._bitwise_mismatch_count(torch, positive, negative_zero) == 1


def test_fe0_self_noise_is_reported_and_blocks_activation(monkeypatch, tmp_path):
    monkeypatch.setattr(reuse, "probe", lambda *_: (_ for _ in ()).throw(
        reuse.BaselineNondeterministic("fe0 repeat partial mismatched_elements=2")))
    output = tmp_path / "repeatability.json"
    assert reuse.main(["--output", str(output)]) == 2
    report = json.loads(output.read_text())
    assert report["status"] == "needs_evidence"
    assert report["production_promotion"] == "blocked"
    assert report["candidate_evaluation_allowed"] is False
    assert "mismatched_elements=2" in report["error"]


def test_workspace_and_policy_stay_separate_from_production():
    assert reuse.BASELINE_COMMIT == "fe0e925e7ef78bfb64217a300031502fc4a7b7bc"
    assert reuse.candidate_workspace_per_core(256) == 1_839_104
    assert reuse.baseline_workspace_per_core(256) == 917_504
    assert all(case.name != "user_dataset" for case in reuse.CASES)
    assert set(reuse.assert_fe0_kernel_unchanged()) == {
        "csrc/kernels/attention_cv.cpp", "csrc/kernels/oscar_common.h"}


def test_failed_probe_reports_nonzero_and_never_claims_activation(monkeypatch, tmp_path):
    monkeypatch.setattr(reuse, "probe", lambda *_: (_ for _ in ()).throw(
        reuse.HistoryReuseProbeError("candidate mismatch")))
    output = tmp_path / "probe.json"
    assert reuse.main(["--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and report["production_promotion"] == "blocked"
    assert report["candidate_evaluation_allowed"] is False
