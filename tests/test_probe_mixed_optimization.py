"""Archive #126/#129/#143-151/D.4: short mixed gate host contracts."""

import json
from types import SimpleNamespace

import pytest

from tools import probe_mixed_optimization as opt


def test_seven_shape_plan_preserves_exact_65k_history_and_decode_cohort():
    cold, warm, long = opt.shape_plan()
    assert (cold.name, warm.name, long.name) == (
        "mixed_cold_first_chunk", "mixed_history_30k", "mixed_history_65k")
    assert long.qlens == (4,) * 31 + (16260,)
    assert long.contexts == (65100,) * 31 + (48840,)
    assert sum(long.qlens) == 16384
    assert long.contexts[-1] + long.qlens[-1] == 65100
    assert warm.contexts[-1] + warm.qlens[-1] == 30000
    assert long.splits == cold.splits == warm.splits == 1


def test_c16_workspace_mirrors_bounded_binding_contract():
    assert opt._c16_workspace_per_core(256) == 4_997_120
    assert opt._c16_workspace_per_core(128) < opt._c16_workspace_per_core(256)
    with pytest.raises(opt.MixedOptimizationProbeError, match="unsupported"):
        opt._c16_workspace_per_core(192)


def test_redirect_keeps_exact_abi_and_only_changes_candidate_symbol():
    calls = []

    def record(name):
        return lambda *args: calls.append((name, args))

    ops = SimpleNamespace(**{name: record(name) for name in (
        opt.OLD_BASE, opt.NEW_BALANCED, opt.OLD_C4, opt.NEW_C16)})
    tensors = {key: key for key in ("q", "qr", "ck", "cv", "rv", "raw",
                                     "table", "wk", "wv", "tags")}
    fixture = {"spec": opt.reuse.Shape("tiny", (4,), (641,), 64, 1, 1, False),
               "blocks": 2, "stride": 20480, "scale": 0.125}
    buffers = {key: key for key in ("tasks", "partial", "lse", "status", "workspace")}
    for is_mixed, expected_names in ((False, (opt.OLD_BASE, opt.NEW_BALANCED)),
                                     (True, (opt.OLD_C4, opt.NEW_C16))):
        buffers["cluster_stats"] = "stats" if is_mixed else None
        opt._launch_cv(ops, tensors, fixture, buffers, 20,
                       is_mixed=is_mixed, candidate=False)
        opt._launch_cv(ops, tensors, fixture, buffers, 20,
                       is_mixed=is_mixed, candidate=True)
        old, new = calls[-2:]
        assert (old[0], new[0]) == expected_names
        assert old[1] == new[1]
        assert len(old[1]) == (26 if is_mixed else 25)


def test_real_dispatch_keeps_compact_q4_same_op_and_balances_large_padding():
    from oscar_ascend.ops.cv_dispatch import select_cv_op

    def route(tokens, max_q):
        return select_cv_op(16, 6, 1, tokens, max_q,
                            fast_unpack=True, mixed_cv=True)

    assert route(128, 4) == opt.OLD_BASE
    assert route(512, 4) == opt.NEW_BALANCED
    assert route(16384, 4) == opt.NEW_BALANCED
    assert route(16384, 16260) == opt.NEW_C16


def test_speed_gate_is_strict_only_for_distinct_operators():
    old = [{"cv_ms": 10.0, "total_ms": 12.0}] * 5
    slow = [{"cv_ms": 10.01, "total_ms": 12.01}] * 5
    samples = {"old": old, "new": slow}
    distinct = opt._verdict(samples, opt.OLD_BASE, opt.NEW_BALANCED, 1.0)
    assert distinct["status"] == "failed"
    assert distinct["ratio_scope"] == "different_operators_strict_speed_gate"
    same = opt._verdict(samples, opt.OLD_BASE, opt.OLD_BASE, 1.0)
    assert same["status"] == "passed"
    assert same["ratio_scope"] == "identical_operator_repeatability"
    with pytest.raises(opt.MixedOptimizationProbeError, match=r"2\+5"):
        opt._verdict({"old": old[:-1], "new": slow}, opt.OLD_BASE,
                     opt.NEW_BALANCED, 1.0)


def test_c16_stats_allow_cross_core_skip_leader_ownership():
    torch = pytest.importorskip("torch")
    # First pass skipped 16 members on core1; second pass grouped all on
    # core0. #149 forbids imposing per-core skips==grouped.
    stats = torch.tensor([[1, 16, 0, 2, 30, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0, 16, 0]], dtype=torch.int64)
    row = opt._check_c16_stats(torch, {"cluster_stats": stats}, 16,
                               require_clusters=True)
    assert row["eligible_clusters"] == 1
    assert row["grouped_leaders"] == row["original_schedule_skips"] == 16
    stats[0, 4] = 3
    with pytest.raises(opt.MixedOptimizationProbeError, match="per-core"):
        opt._check_c16_stats(torch, {"cluster_stats": stats}, 16,
                             require_clusters=True)


def test_fault_selector_requires_a_complete_shared_c16_bucket():
    torch = pytest.importorskip("torch")
    spec = opt.reuse.Shape("graph", (1024,), (641,), 256, 1, 1,
                           True, recent_tokens=32)
    tasks = torch.zeros((1024 * 3, 16), dtype=torch.int64)
    for j in range(16):
        token = 336 + j * 21
        row = tasks[token * 3]
        row[0], row[1], row[3], row[4], row[5] = token, 21, 64, 641, 0
        row[8], row[9], row[10] = 641, 0, 0
    anchor, position = opt._fault_cluster(tasks, {"spec": spec, "tokens": 1024})
    assert (anchor, position) == (336, 80)
    tasks[357 * 3, 4] = 640
    with pytest.raises(opt.MixedOptimizationProbeError, match="complete shared"):
        opt._fault_cluster(tasks, {"spec": spec, "tokens": 1024})


def test_first_draft_runs_full_cv_then_merge_without_native_current(monkeypatch):
    torch = pytest.importorskip("torch")
    stages = []

    class Event:
        tick = 0

        def __init__(self, **_kwargs):
            pass

        def record(self, _stream):
            Event.tick += 1
            self.at = Event.tick

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return float(other.at - self.at)

    stream = object()
    monkeypatch.setattr(torch, "npu", SimpleNamespace(
        current_stream=lambda: stream, Event=Event), raising=False)
    monkeypatch.setattr(opt.reuse, "_poison", lambda *_: None)
    monkeypatch.setattr(opt.fast, "_launch", lambda *_args, **_kwargs:
                        stages.append("cv"))
    spec = opt.reuse.Shape("mtp_first_draft_warm30k", (1,), (17,),
                           64, 1, 1, False)
    fixture = {"spec": spec, "tokens": 1}
    cv = {"partial": torch.zeros((1, 6, 3, 64)),
          "lse": torch.zeros((1, 6, 3))}
    buffers = {"cv": cv, "output": torch.zeros((6, 64)),
               "output_lse": torch.zeros(6),
               "merge_status": torch.zeros(6, dtype=torch.int32)}
    ops = SimpleNamespace(merge_lse_out=lambda *_: stages.append("merge"))
    row = opt._run_first_draft_once(torch, ops, fixture, {}, buffers, 20)
    assert stages == ["cv", "merge"]
    assert row["cv_ms"] > 0 and row["total_ms"] > row["cv_ms"]
    assert row["suppress_ms"] == row["current_ms"] == 0


def test_failure_report_returns_rc2_and_cannot_overwrite_phase_state(monkeypatch, tmp_path):
    monkeypatch.setattr(opt, "probe", lambda *_: (_ for _ in ()).throw(
        opt.MixedOptimizationProbeError("candidate NPU oracle mismatch")))
    report = tmp_path / "mixed-optimization-report.json"
    assert opt.main(["--output", str(report)]) == 2
    parsed = json.loads(report.read_text())
    assert parsed["status"] == "failed"
    assert parsed["first_error"] == "candidate NPU oracle mismatch"
    with pytest.raises(SystemExit) as error:
        opt.main(["--output", str(tmp_path / "mixed-optimization.json")])
    assert error.value.code == 2
