"""Archive #126/#129/#143-153/D.4: B4 short gate host contracts only."""

import json
from types import SimpleNamespace

import pytest

from tools import probe_batched_history as b4


def test_only_three_required_hot_shapes_and_actual_old_routes():
    continuation, warm30, warm65 = b4.case_plan()
    assert (continuation.name, continuation.qlens, continuation.contexts) == (
        "long_continuation_20k", (4096,), (16000,))
    assert warm30.name == "mixed_history_30k"
    assert warm65.name == "mixed_history_65k"
    assert warm30.qlens == warm65.qlens == (4,) * 31 + (16260,)
    assert warm65.contexts == (65100,) * 31 + (48840,)
    assert b4._old_symbol(continuation, 4096) == b4.FAST_C4
    assert b4._old_symbol(warm30, 16384) == b4.FAST_C16
    assert b4._old_symbol(warm65, 16384) == b4.FAST_C16


def test_b4_workspace_and_proxy_preserve_c4_abi_including_stats():
    assert b4.b4_workspace_per_core(256) == 2_627_584
    calls = []

    def record(name):
        return lambda *args: calls.append((name, args))

    ops = SimpleNamespace(**{name: record(name) for name in
                             (b4.FAST_C4, b4.FAST_C16, b4.B4)})
    fixture = {"spec": b4.reuse.Shape("tiny", (4,), (641,), 64, 1, 1, True),
               "blocks": 2, "stride": 20480, "scale": 0.125}
    tensors = {key: key for key in ("q", "qr", "ck", "cv", "rv", "raw",
                                     "table", "wk", "wv", "tags")}
    buffers = {key: key for key in ("tasks", "partial", "lse", "status", "workspace")}
    buffers["cluster_stats"] = "stats"
    for selected in (b4.FAST_C4, b4.FAST_C16, b4.B4):
        b4.fast._launch(b4._proxy(ops, selected), tensors, fixture,
                        buffers, 20, "c4", fast=True)
    assert [name for name, _ in calls] == [b4.FAST_C4, b4.FAST_C16, b4.B4]
    assert calls[0][1] == calls[1][1] == calls[2][1]
    assert len(calls[0][1]) == 26 and calls[0][1][15] == "stats"


def test_strict_ratio_checks_both_cv_and_full_attention():
    old = [{"cv_ms": 10.0, "total_ms": 13.0}] * 5
    faster = [{"cv_ms": 9.0, "total_ms": 12.0}] * 5
    assert b4._verdict({"old": old, "new": faster}, b4.FAST_C16, 1.0)["status"] == "passed"
    slower_total = [{"cv_ms": 9.0, "total_ms": 13.001}] * 5
    assert b4._verdict({"old": old, "new": slower_total}, b4.FAST_C4, 1.0)["status"] == "failed"
    with pytest.raises(b4.BatchedHistoryProbeError, match=r"2\+5"):
        b4._verdict({"old": old, "new": faster[:-1]}, b4.FAST_C4, 1.0)


def test_failure_rc2_and_phase_filename_separation(monkeypatch, tmp_path):
    monkeypatch.setattr(b4, "probe", lambda *_: (_ for _ in ()).throw(
        b4.BatchedHistoryProbeError("B4 output changed bitwise")))
    output = tmp_path / "batched-history-report.json"
    assert b4.main(["--output", str(output)]) == 2
    assert json.loads(output.read_text())["first_error"] == "B4 output changed bitwise"
    with pytest.raises(SystemExit) as error:
        b4.main(["--output", str(tmp_path / "batched-history.json")])
    assert error.value.code == 2
