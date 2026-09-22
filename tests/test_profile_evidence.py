"""Archive #70-73/D.4: host durations cannot become device timing evidence.

Synthetic traces here test the parser only; they are tagged and cannot pass
real NPU attribution gates. Missing restore-byte measurements remain unknown.
"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from oscar_ascend import timing
from tools.summarize_profile import ProfileEvidenceError, main, summarize_trace


def _device(name, start, duration, **updates):
    return {"ph": "X", "name": name, "cat": "kernel", "pid": 9, "tid": 1,
            "ts": start, "dur": duration, **updates}


def _trace(*events):
    return {"oscar_test_fixture": True, "traceEvents": [
        {"ph": "M", "pid": 9, "name": "process_name", "args": {"name": "Ascend Hardware"}},
        *events]}


def test_disabled_timing_does_not_create_events_clock_or_import_torch(monkeypatch):
    monkeypatch.delenv("OSCAR_TIMING", raising=False)
    monkeypatch.delenv("OSCAR_PROFILER", raising=False)
    with patch("builtins.__import__", side_effect=AssertionError("unexpected import")), \
         patch.object(timing.time, "perf_counter_ns", side_effect=AssertionError("unexpected clock")):
        with timing.phase("prepare", tokens=128):
            pass
        with timing.profile_session():
            pass


def test_phase_rejects_tensor_like_metadata_and_reserved_fields(monkeypatch):
    monkeypatch.setenv("OSCAR_TIMING", "1")
    with pytest.raises(TypeError, match="host scalars"):
        timing.phase("prepare", tensor=object())
    with pytest.raises(ValueError, match="evidence"):
        timing.phase("prepare", device_ms=0)


def test_npu_absence_creates_not_run_without_fabricated_times(tmp_path, monkeypatch):
    config = tmp_path / "target.json"
    config.write_text(json.dumps({"devices": None}))
    monkeypatch.setenv("OSCAR_PROFILER", "1")
    monkeypatch.setenv("OSCAR_TARGET_CONFIG", str(config))
    with pytest.raises(timing.ProfilingUnavailable, match="selection"):
        with timing.profile_session(tmp_path / "trace", "test-worker"):
            raise AssertionError("must not execute workload without selected NPU")
    status = json.loads((tmp_path / "trace/test-worker-status.json").read_text())
    assert status["state"] == "not_run"
    assert status["device_measurements"] is None
    assert status["graph_mode_modified"] is False


def test_host_scope_and_cpu_launch_cannot_be_device_times():
    report = summarize_trace(_trace(
        {"ph": "X", "name": 'oscar::prepare::{}', "cat": "user_annotation",
         "pid": 1, "tid": 1, "ts": 0, "dur": 725000},
        {"ph": "X", "name": "oscar_attention_cv_kernel", "cat": "cpu_op",
         "pid": 1, "tid": 1, "ts": 0, "dur": 6500000},
    ))
    assert report["phases"]["prepare"]["host_scope_sum_ms"] == 725
    assert report["summary"]["device_ms"]["prepare"] is None
    assert report["summary"]["device_ms"]["fia"] is None
    assert report["summary"]["npu_kernel_count"] == 0


def test_overlapping_cube_vector_durations_do_not_double_count_busy_time():
    report = summarize_trace(_trace(
        _device("oscar_attention_cv_kernel_aic", 1000, 18700),
        _device("oscar_attention_cv_kernel_aiv", 1100, 18500, tid=2),
        _device("oscar_rotate_clip_store_kernel", 20000, 209000),
    ))
    assert report["summary"]["device_ms"]["fia"] == pytest.approx(18.7)
    assert report["phases"]["fia"]["device_kernel_sum_ms"] == pytest.approx(37.2)
    assert report["summary"]["device_ms"]["phase1_stores"] == 209
    assert report["status"] == "test_fixture"
    assert report["history_restore"]["full_history_restore_bytes"] is None


def test_explicit_flow_attributes_opaque_graph_kernel_to_record_scope():
    report = summarize_trace(_trace(
        {"ph": "X", "name": 'oscar::merge::{"layer":"model.0"}', "cat": "user_annotation",
         "pid": 1, "tid": 7, "ts": 0, "dur": 100},
        {"ph": "s", "name": "torch_to_npu", "cat": "async", "id": 42,
         "pid": 1, "tid": 7, "ts": 10},
        {"ph": "f", "name": "torch_to_npu", "cat": "async", "id": 42,
         "pid": 9, "tid": 1, "ts": 1000},
        _device("compiled_opaque_hash", 1000, 500),
    ))
    assert report["summary"]["device_ms"]["merge"] == 0.5
    assert report["phases"]["merge"]["attribution"] == ["explicit_trace_flow"]


def test_explicit_correlation_links_device_to_host_launch_scope():
    report = summarize_trace(_trace(
        {"ph": "X", "name": "oscar::rotate::{}", "pid": 1, "tid": 7, "ts": 0, "dur": 100},
        {"ph": "X", "name": "aclLaunch", "pid": 1, "tid": 7, "ts": 10, "dur": 20,
         "args": {"correlation_id": 5}},
        _device("opaque", 1000, 123, args={"correlation_id": 5}),
    ))
    assert report["summary"]["device_ms"]["rotate"] == pytest.approx(0.123)
    assert report["phases"]["rotate"]["attribution"] == ["explicit_correlation_id"]


def test_standalone_history_restore_is_flagged_but_no_bytes_are_invented():
    report = summarize_trace(_trace(_device("oscar_full_dequant_kv", 0, 6500000)))
    assert report["summary"]["device_ms"]["dequant"] == 6500
    assert report["history_restore"]["status"] == "forbidden_kernel_observed"
    assert report["history_restore"]["full_history_restore_bytes"] is None


def test_missing_trace_cli_is_not_run_and_synthetic_trace_cannot_pass_gate(tmp_path):
    output = tmp_path / "report.json"
    assert main(["--output", str(output)]) == 2
    report = json.loads(output.read_text())
    assert report["status"] == "not_run" and report["device_ms"] is None
    trace = tmp_path / "synthetic.json"
    trace.write_text(json.dumps(_trace(_device("oscar_attention_cv", 0, 10))))
    assert main([str(trace), "--output", str(output), "--require-complete"]) == 1
    assert json.loads(output.read_text())["phase_attribution_complete"] is False


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1])
def test_invalid_device_duration_is_rejected(bad):
    with pytest.raises(ProfileEvidenceError):
        summarize_trace(_trace(_device("oscar_attention_cv", 0, bad)))
