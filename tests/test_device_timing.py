"""Archive #70-73/#133/#142-143: event lifecycle contracts, not NPU evidence."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from oscar_ascend import device_timing, plugin, timing
from oscar_ascend.integration.dummy_context import native_dummy_run


@pytest.fixture
def events(tmp_path, monkeypatch):
    for name in ("OSCAR_TIMING", "OSCAR_PROFILER", "OSCAR_DEBUG_SYNC"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(device_timing, "_run", None)
    marker = tmp_path / "device-timing-control.json"
    monkeypatch.setenv("OSCAR_DEVICE_TIMING_CONTROL", str(marker))
    monkeypatch.setenv("OSCAR_TRACE_DIR", str(tmp_path))
    calls = []
    state = {"capturing": False, "fail": False}

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing
            calls.append("event")

        def record(self, stream):
            calls.append("record")

        def synchronize(self):
            calls.append("sync")
            if state["fail"]:
                raise RuntimeError("injected device error")

        def elapsed_time(self, other):
            calls.append("elapsed")
            return 12.5  # Explicit mock; never a hardware measurement.

    monkeypatch.setattr(torch, "npu", SimpleNamespace(Event=Event,
        is_current_stream_capturing=lambda: state["capturing"],
        current_stream=lambda: "same-stream"), raising=False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 2)
    return marker, calls, state


def records(marker):
    return [json.loads(line) for p in marker.parent.glob("device-events-*.jsonl")
            for line in p.read_text().splitlines()]


def arm(marker):
    marker.write_text(json.dumps({"enabled": True, "run_id": "diagnostic-fixture"}))
    device_timing.refresh()


def test_primary_batch_creates_no_events_and_phases_do_not_poll(events):
    marker, calls, _ = events
    device_timing.refresh()
    with patch.object(device_timing.Path, "read_text", side_effect=AssertionError("per-phase poll")), \
         patch.object(device_timing.time, "perf_counter_ns", side_effect=AssertionError("primary clock")):
        with timing.phase("fia", tokens=16384):
            pass
    assert calls == [] and records(marker) == []


def test_event_interval_written_after_completion_and_disarmed(events):
    marker, calls, _ = events
    arm(marker)
    with timing.phase("fia", tokens=4096, requests=4, stage="prefill", max_seq_len=23000):
        calls.append("launch")
    assert calls == ["event", "event", "record", "launch", "record", "sync", "elapsed"]
    row, = records(marker)
    assert row["device_ms"] == 12.5 and row["rank"] == 2
    assert row["stage"] == "prefill" and row["run_id"] == "diagnostic-fixture"
    assert row["device_evidence"] == "npu_event" and row["debug_synchronization"]
    marker.write_text('{"enabled":false,"run_id":"diagnostic-fixture"}')
    device_timing.refresh()
    assert not device_timing.active()


@pytest.mark.parametrize("dummy", [False, True])
def test_no_events_or_synchronization_inside_capture_or_dummy(events, dummy):
    marker, calls, state = events
    arm(marker)
    if dummy:
        with native_dummy_run(), timing.phase("fia", tokens=512):
            pass
    else:
        state["capturing"] = True
        with timing.phase("fia", tokens=512):
            pass
    assert calls == [] and records(marker) == []


def test_device_failure_is_retained_and_raised(events):
    marker, calls, state = events
    arm(marker)
    state["fail"] = True
    with pytest.raises(RuntimeError, match="injected device error"):
        with timing.phase("rotate", tokens=4, stage="draft"):
            pass
    row, = records(marker)
    assert row["failed"] and row["device_ms"] is None
    assert "injected" in row["error"] and "elapsed" not in calls


def test_graph_wrapper_arms_eager_and_measures_replay_outside_graph(events):
    marker, calls, state = events
    # Wrapper map keys are native immutable batch descriptors.
    key = (16, 4)
    context = SimpleNamespace(cudagraph_runtime_mode="eager", batch_descriptor=key)

    class Wrapper:
        runtime_mode = "full"
        concrete_aclgraph_entries = {key: SimpleNamespace(aclgraph=object())}

        def __call__(self):
            if context.cudagraph_runtime_mode == "eager":
                with timing.phase("fia", tokens=16, stage="prefill"):
                    calls.append("eager")
            else:
                calls.append("graph")
            return 7

    module = SimpleNamespace(ACLGraphWrapper=Wrapper, get_forward_context=lambda: context)
    try:
        plugin._patch_graph_evidence(module)
        marker.write_text('{"enabled":true,"run_id":"diagnostic-fixture"}')
        assert Wrapper()() == 7  # Only the outer wrapper refreshes control.
        assert records(marker)[0]["phase"] == "fia"
        context.cudagraph_runtime_mode = "full"
        assert Wrapper()() == 7
        assert records(marker)[1]["phase"] == "graph_replay"
        assert records(marker)[1]["graph_scope"] == "whole_model_graph"
    finally:
        plugin.unregister()
