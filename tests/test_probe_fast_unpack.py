"""Archive #126/#129/#148-151/#150-P0: fast INT2 probe contracts, not NPU proof."""

import json
from types import SimpleNamespace

import pytest

from tools import probe_fast_unpack as fast


def _plan():
    return fast.case_plan({"max_num_batched_tokens": 16384}, 20)


def _rows():
    rows = []
    for shape, mode, timed in _plan():
        invalid = shape.corrupt_metadata or shape.corrupt_qr
        row = {"case": shape.name, "status": "passed", "mode": mode,
               "old_operator": fast._MODES[mode][0],
               "fast_operator": fast._MODES[mode][1],
               "precision": "bitwise_passed",
               "partial_lse_status": "bitwise_passed",
               "merged": "invalid_input_not_applicable" if invalid else "bitwise_passed",
               "fe0_repeatability": "bitwise_passed",
               "frozen_oracle": "invalid_input_not_applicable" if invalid else "passed",
               "shape": {"padding_tokens": (shape.padded_tokens or sum(shape.qlens))
                         - sum(shape.qlens)},
               "padding_task_gate": "passed",
               "terminal_split_counterfactual": None,
               "fp16_metadata_valid_bits": ([{"half_bits": "0x0001"}] *
                    len(fast._VALID_HALF_BITS)) if shape.name in fast._META_CASES else None,
               "fp16_metadata_invalid_and_dead_tail": ({"status": "passed",
                   "live_invalid_cases": [{"partial_lse_status": "bitwise_passed"}] *
                       len(fast._INVALID_HALF_BITS),
                   "dead_tail": {"fe0_old_fast_output_status": "bitwise_passed"}}
                    if shape.name in fast._META_CASES else None),
               "graph_capture": "passed" if shape.name in fast._GRAPHS else "not_run",
               "graph_replay": "passed" if shape.name in fast._GRAPHS else "not_run",
               "performance": {"gate": "passed", "fast_over_old": 0.9,
                               "warmup": 2, "repeats": 5,
                               "order": "alternating_AB_BA",
                               "old": {"device_event_ms": [10.0] * 5, "median_ms": 10.0},
                               "fast": {"device_event_ms": [9.0] * 5, "median_ms": 9.0}}
                              if timed else None}
        if shape.name == "mixed_unaligned_mature_s3":
            row["terminal_split_counterfactual"] = {
                "fe0_old_fast_partial_lse_status_merge": "bitwise_passed"}
        if shape.name == "q4_decode32_n128_s3":
            row["p0_weighted"] = {"operator": fast.P0_WEIGHTED,
                "predicted_kv256_per_core": [32] * 20,
                "performance": {"gate": "passed", "warmup": 2, "repeats": 5,
                    "order": "alternating_AB_BA", "weighted_over_fast": 0.8,
                    "fast": {"device_event_ms": [10.0] * 5, "median_ms": 10.0},
                    "weighted": {"device_event_ms": [8.0] * 5, "median_ms": 8.0}}}
        rows.append(row)
    return rows


def test_case_plan_keeps_original_nine_and_measures_five_distinct_modes():
    plan = _plan()
    assert len(plan) == 18
    assert {shape.name for shape, _, _ in plan[:9]} == {
        shape.name for shape in fast.reuse.CASES}
    assert all(mode == "c4" for _, mode, _ in plan[:9])
    assert {(shape.name, mode) for shape, mode, _ in plan[9:11]} == {
        ("base_frontier_d64", "base"), ("base_mixed_hkv2_d128", "base")}
    timed = {shape.name for shape, _, measure in plan if measure}
    assert timed == fast._TIMED
    by_name = {shape.name: (shape, mode) for shape, mode, _ in plan}
    q4, q4_mode = by_name["q4_decode32_n128_s3"]
    assert q4_mode == "base" and len(q4.qlens) == 32
    assert q4.qlens == (4,) * 32 and q4.splits == 3
    assert (q4.padded_tokens or sum(q4.qlens)) == 128
    for name, padded, splits in (("q1_decode32_n128_s3", 128, 3),
                                 ("q1_mixed32_n16384_s1", 16384, 1)):
        shape, mode = by_name[name]
        assert mode == "q1" and shape.qlens == (1,) * 32
        assert shape.padded_tokens == padded and shape.splits == splits
        assert shape.contexts == (20000, 23000, 27000, 30000) * 8
        assert shape.slot_context is True
    mixed, mode = by_name["mixed_q1_q4_long"]
    assert mode == "c4" and mixed.qlens == (1, 4, 385)
    assert sum(mixed.qlens) == 390 and mixed.splits == 2
    assert mixed.expect_clusters
    assert {(by_name[name][1], by_name[name][0].dim) for name in fast._META_CASES} == {
        ("base", 256), ("q1", 256), ("c4", 256)}


def test_p0_predictor_balances_real_kv256_width_and_ignores_dead_rows():
    torch = pytest.importorskip("torch")
    tasks = torch.zeros((44, 16), dtype=torch.int64)
    for task_id in range(20):
        tasks[task_id, 1] = 1; tasks[task_id, 3] = 64
        tasks[task_id, 4] = 64 + (task_id % 5 + 1) * 256
    for task_id in range(20, 40):
        tasks[task_id, 1] = 1; tasks[task_id, 4] = 1
    tasks[40, 1] = -1
    tasks[41, 1] = 1; tasks[41, 10] = 2
    tasks[43, 1] = 1; tasks[43, 3] = 9; tasks[43, 4] = 8
    loads = fast.predicted_kv256_weights(tasks, 20)
    assert sum(loads) == sum(task_id % 5 + 1 for task_id in range(20)) + 20
    assert max(loads) - min(loads) <= 4


def test_fast_symbols_preserve_old_abi_and_c4_stats_position():
    calls = []
    def op(label):
        return lambda *args: calls.append((label, args))
    ops = SimpleNamespace(**{name: op(name) for names in fast._MODES.values()
                             for name in names})
    tensors = {key: key for key in ("q", "qr", "ck", "cv", "rv", "raw",
                                    "table", "wk", "wv", "tags")}
    fixture = {"spec": fast.reuse.Shape("tiny", (1,), (17,), 64, 1, 1, False),
               "blocks": 2, "stride": 20480, "scale": 0.125}
    buffers = {key: key for key in ("tasks", "partial", "lse", "status", "workspace")}
    for mode in ("base", "q1", "c4"):
        buffers["cluster_stats"] = "stats" if mode == "c4" else None
        fast._launch(ops, tensors, fixture, buffers, 2, mode, fast=False)
        fast._launch(ops, tensors, fixture, buffers, 2, mode, fast=True)
        old, new = calls[-2:]
        assert old[0] != new[0]
        assert old[1] == new[1]
        assert len(old[1]) == (26 if mode == "c4" else 25)
        if mode == "c4":
            assert old[1][15] == new[1][15] == "stats"
        else:
            assert old[1][15] == fast.reuse.BLOCK_TOKENS


def test_special_half_fixture_recomputes_dense_oracle_from_modified_live_bytes():
    torch = pytest.importorskip("torch")
    shape = fast.reuse.Shape("meta_host", (4,), (641,), 64, 1, 1, False)
    fixture = fast.reuse.make_fixture(torch, shape)
    before_hash = fixture["input_sha256"]
    evidence = fast._inject_valid_metadata(torch, fixture)
    assert len(evidence) == len(fast._VALID_HALF_BITS)
    assert fixture["input_sha256"] != before_hash
    assert fast._read_half_bits(fixture, fixture["cpu"]["raw"], 64,
                                "k", "scale") == 0x0001
    assert fast._read_half_bits(fixture, fixture["cpu"]["raw"], 64,
                                "k", "zero") == 0x8000
    assert len(fixture["expected"]) == 4
    assert all(bool(torch.isfinite(value[0]).all()) and
               bool(torch.isfinite(value[1]).all())
               for value in fixture["expected"].values())
    invalid = fixture["cpu"]["raw"].clone()
    fast._write_half_bits(fixture, invalid, 64, "k", "scale", 0x7C00)
    fixture["cpu"]["raw"] = invalid
    with pytest.raises(ValueError, match="invalid fp16 quantizer metadata"):
        fast._recompute_oracle_after_metadata(torch, fixture)
    # C4's terminal shared source0 split starts beyond recent=256; the
    # special bytes must live in that grouped split, not only independent S0.
    c4 = fast.reuse.make_fixture(torch, fast.reuse.Shape(
        "meta_half_c4_d256", (385,), (641,), 64, 1, 2, True))
    fast._inject_valid_metadata(torch, c4)
    assert fast._read_half_bits(c4, c4["cpu"]["raw"], 400,
                                "k", "scale") == 0x0001
    assert fast._read_half_bits(c4, c4["cpu"]["raw"], 405,
                                "v", "zero") == 0x3C00


def test_live_bad_metadata_accepts_final_status_2_or_3_but_not_zero():
    torch = pytest.importorskip("torch")
    shape = fast.reuse.Shape("invalid_status", (1,), (641,), 64, 1, 1, False)
    tasks = torch.zeros((3, 16), dtype=torch.int64)
    tasks[0, 1] = 1
    statuses = torch.zeros((3, 2), dtype=torch.int32)
    statuses[0, 0] = 2  # Softmax non-finite can override prior unpack error 3.
    assert fast._live_source_error_codes(tasks, statuses, shape, 1) == [2]
    statuses[0, 0] = 3
    assert fast._live_source_error_codes(tasks, statuses, shape, 1) == [3]
    statuses[0, 0] = 0
    with pytest.raises(fast.FastUnpackProbeError, match="no final source0 error"):
        fast._live_source_error_codes(tasks, statuses, shape, 1)


def test_verdict_requires_every_precision_graph_and_ratio_gate():
    rows = _rows()
    assert fast.verdict(rows)["status"] == "passed"
    with pytest.raises(fast.FastUnpackProbeError, match="missing a required"):
        fast.verdict(rows[:-1])
    bad = _rows()
    next(row for row in bad if row["case"] == "q4_decode32_n128_s3")["graph_replay"] = "not_run"
    assert fast.verdict(bad)["graph_replay"] == "failed"
    bad = _rows()
    row = next(row for row in bad if row["case"] == "q1_mixed32_n16384_s1")
    row["performance"]["fast"]["device_event_ms"] = [10.0001] * 5
    row["performance"]["fast"]["median_ms"] = 10.0001
    row["performance"]["fast_over_old"] = 1.00001
    assert fast.verdict(bad)["performance"] == "failed"
    bad = _rows()
    next(row for row in bad if row["case"] == "q4_decode32_n128_s3")[
        "p0_weighted"]["performance"]["gate"] = "failed"
    assert fast.verdict(bad)["p0_performance"] == "failed"
    bad = _rows()
    next(row for row in bad if row["case"] == "mature_20k")["fast_operator"] = fast.reuse.CANDIDATE_OP
    assert fast.verdict(bad)["precision"] == "failed"
    bad = _rows()
    next(row for row in bad if row["case"] == "mixed_unaligned_mature_s3")["terminal_split_counterfactual"] = None
    assert fast.verdict(bad)["precision"] == "failed"
    bad = _rows()
    next(row for row in bad if row["case"] == "q1_decode32_n128_s3")["padding_task_gate"] = "missing"
    assert fast.verdict(bad)["precision"] == "failed"
    bad = _rows()
    next(row for row in bad if row["case"] == "meta_half_q1_d256")[
        "fp16_metadata_invalid_and_dead_tail"] = None
    assert fast.verdict(bad)["precision"] == "failed"


def test_failure_is_rc2_and_phase_state_filename_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(fast, "probe", lambda *_: (_ for _ in ()).throw(
        fast.FastUnpackProbeError("signed fast symbol missing")))
    output = tmp_path / "fast-unpack-report.json"
    assert fast.main(["--output", str(output)]) == 2
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert report["candidate_evaluation_allowed"] is False
    assert report["error"] == "signed fast symbol missing"
    with pytest.raises(SystemExit) as error:
        fast.main(["--output", str(tmp_path / "fast-unpack.json")])
    assert error.value.code == 2
