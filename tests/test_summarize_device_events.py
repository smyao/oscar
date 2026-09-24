"""Archive #70-#73/#129/#133-#135: device events stay diagnostic and per rank."""

import json

import pytest

from tools.summarize_device_events import (DeviceEventEvidenceError,
    compact_cv_shape_lines, compact_device_event_line, compact_device_stage_lines,
    summarize_device_events)


def _event(rank, phase, milliseconds, *, run_id="diag-1", tokens=16384,
           max_query_len=8192, max_seq_len=32768, draft_index=0, is_draft=False):
    stage = "draft" if is_draft else "prefill"
    return {"t": "oscar-device-event", "phase": phase, "rank": rank,
            "pid": 1000 + (rank if type(rank) is int else 0), "run_id": run_id, "device_ms": milliseconds,
            "host_ms": milliseconds + .1, "tokens": tokens,
            "max_query_len": max_query_len, "draft_index": draft_index,
            "max_seq_len": max_seq_len, "requests": 4, "source_splits": 2,
            "is_draft": is_draft, "dummy_origin": False, "stage": stage,
            "scope": "synchronized_npu_stream_interval", "device_evidence": "npu_event",
            "debug_synchronization": True}


def test_summarizes_each_tp_rank_without_fabricating_wall_or_graph_inner_phases(tmp_path):
    for rank in range(4):
        records = [_event(rank, "fia", 10 + rank),
                   _event(rank, "current_native_fia", 3 + rank),
                   _event(rank, "merge", 1 + rank),
                   _event(rank, "fia", 2, tokens=4, max_query_len=4,
                          draft_index=1, is_draft=True),
                   _event(rank, "graph_replay", 30 + rank, tokens=4, max_query_len=4)]
        (tmp_path / f"device-events-{rank}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in records) + "\n")
    output = tmp_path / "summary.json"
    report = summarize_device_events(tmp_path, run_id="diag-1", output=output)
    assert report["status"] == "observed" and report["event_count"] == 20
    assert report["phases"]["fia"]["critical_rank_sum_ms"] == 15
    assert report["phases"]["fia"]["rank_with_max_sum"] == 3
    assert report["ranks"]["0"]["phase_sum_ms"] == 46
    assert report["critical_rank_phase_sum_ms"] == 58  # max rank, never 4-rank sum
    assert report["graph_replay"]["inner_phases"] == "not_inferred_from_graph_replay"
    assert report["stages"]["prefill"]["critical_rank_phase_sum_ms"] == 23
    assert report["stages"]["draft"]["critical_rank_phase_sum_ms"] == 2
    assert report["stages"]["graph"]["critical_rank_phase_sum_ms"] == 33
    assert report["missing_graph_ranks"] == []
    assert any(row["phase"] == "fia" and row["is_draft"] is True and
               row["draft_index"] == 1 for row in report["shape_groups"])
    assert any(row["stage"] == "prefill" and row["max_seq_len"] == 32768 and
               row["requests"] == 4 and row["source_splits"] == 2 for row in report["shape_groups"])
    assert output.exists()
    assert "scope=synchronized_diagnostic_only" in compact_device_event_line(report, output=output)
    assert len(compact_device_stage_lines(report)) == 3
    assert len(compact_cv_shape_lines(report)) == 2


def test_missing_rank_and_wrong_run_id_never_count_as_complete(tmp_path):
    (tmp_path / "device-events-1.jsonl").write_text(
        json.dumps(_event(1, "fia", 3, run_id="old")) + "\n")
    report = summarize_device_events(tmp_path, run_id="diag-1")
    assert report["status"] == "needs_evidence"
    assert report["missing_ranks"] == [0, 1, 2, 3]


def test_rejects_unsynchronized_or_non_device_evidence(tmp_path):
    row = _event(0, "fia", 3)
    row["debug_synchronization"] = False
    (tmp_path / "device-events-0.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(DeviceEventEvidenceError, match="provenance"):
        summarize_device_events(tmp_path, run_id="diag-1")


def test_failed_event_or_unknown_rank_needs_evidence(tmp_path):
    failed = _event(0, "fia", 3)
    failed.update(failed=True, error="device exception", device_ms=None)
    unknown = _event(None, "merge", 2)
    (tmp_path / "device-events-0.jsonl").write_text(
        json.dumps(failed) + "\n" + json.dumps(unknown) + "\n")
    report = summarize_device_events(tmp_path, run_id="diag-1")
    assert report["status"] == "needs_evidence"
    assert len(report["incomplete_events"]) == 2
    assert report["missing_required_eager_phases_by_rank"]["0"]


def test_one_tp_rank_missing_current_phase_cannot_borrow_another_ranks_event(tmp_path):
    # #133/#134: all four ranks must complete the eager phase. A phase seen
    # globally is not evidence that its absent rank executed it.
    for rank in range(4):
        phases = ["fia", "merge", "graph_replay"] + (["current_native_fia"] if rank != 3 else [])
        (tmp_path / f"device-events-{rank}.jsonl").write_text(
            "\n".join(json.dumps(_event(rank, phase, 1)) for phase in phases) + "\n")
    report = summarize_device_events(tmp_path, run_id="diag-1")
    assert report["missing_ranks"] == []
    assert report["status"] == "needs_evidence"
    assert report["missing_required_eager_phases_by_rank"]["3"] == ["current_native_fia"]


def test_one_tp_rank_missing_whole_graph_replay_needs_evidence(tmp_path):
    for rank in range(4):
        phases = ["fia", "merge", "current_native_fia"] + (["graph_replay"] if rank != 3 else [])
        (tmp_path / f"device-events-{rank}.jsonl").write_text(
            "\n".join(json.dumps(_event(rank, phase, 1)) for phase in phases) + "\n")
    report = summarize_device_events(tmp_path, run_id="diag-1")
    assert report["status"] == "needs_evidence"
    assert report["missing_graph_ranks"] == [3]
    assert report["graph_replay"]["inner_phases"] == "not_inferred_from_graph_replay"


def test_cv_copy_lines_select_exact_shape_critical_rank_and_unknown_draft_context(tmp_path):
    for rank in range(4):
        records = [_event(rank, "fia", 10), _event(rank, "current_native_fia", 2),
                   _event(rank, "merge", 1), _event(rank, "graph_replay", 5),
                   _event(rank, "fia", 8 if rank == 3 else 4,
                          tokens=1024, max_query_len=1024, max_seq_len=1024),
                   _event(rank, "fia", 3, tokens=4, max_query_len=4,
                          max_seq_len=0, draft_index=1, is_draft=True)]
        (tmp_path / f"device-events-{rank}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in records) + "\n")
    report = summarize_device_events(tmp_path, run_id="diag-1")
    lines = compact_cv_shape_lines(report)
    assert len(lines) == 3
    assert "case=candidate_no_old_context" in lines[0]
    assert "rank=3" in lines[0] and "rank_sum_ms=8.00" in lines[0]
    assert "case=largest_other_prefill" in lines[1]
    assert "case=largest_draft" in lines[2]
    assert "max_seq_len=0" in lines[2] and "context=unknown_max_seq_zero" in lines[2]
