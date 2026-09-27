# 档案 #70-73/#133/#140-145：外部负载低扰动采样不每相位同步；
# 同step残差只合并OSCAR区间，图/跨stream缺证据必须显式missing。
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from oscar_ascend import passive_timing


class FakeNpu:
    def __init__(self):
        self.clock = 0
        self.stream = object()
        self.capture = False
        self.complete = True
        self.events = []

    def current_stream(self):
        return self.stream

    def is_current_stream_capturing(self):
        return self.capture

    def Event(self, *, enable_timing):
        assert enable_timing
        owner = self
        class Event:
            def __init__(self):
                self.time = None
            def record(self, stream):
                assert stream is owner.stream
                owner.clock += 1
                self.time = owner.clock
            def query(self):
                return owner.complete
            def elapsed_time(self, other):
                return float(other.time-self.time)
        event = Event()
        self.events.append(event)
        return event


def _setup(monkeypatch, tmp_path):
    fake = FakeNpu()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        npu=fake, distributed=SimpleNamespace(is_initialized=lambda: False)))
    control = tmp_path / "control.json"
    control.write_text(json.dumps({"enabled": True, "run_id": "external-window-0", "variant": "baseline"}))
    monkeypatch.setenv("OSCAR_PASSIVE_TIMING_CONTROL", str(control))
    monkeypatch.setenv("OSCAR_TRACE_DIR", str(tmp_path))
    monkeypatch.setattr(passive_timing, "_last_check", 0.0)
    monkeypatch.setattr(passive_timing, "_control", None)
    monkeypatch.setattr(passive_timing, "_current_run_id", None)
    monkeypatch.setattr(passive_timing, "_serial", 0)
    passive_timing._pending.clear()
    passive_timing._tls.step = None
    for key in passive_timing._QUOTAS:
        passive_timing._seen[key] = passive_timing._taken[key] = 0
    scheduled = SimpleNamespace(
        num_scheduled_tokens={"new": 8, "decode": 2},
        scheduled_new_reqs=[SimpleNamespace(req_id="new")],
        scheduled_cached_reqs=SimpleNamespace(req_ids=["decode"], num_output_tokens=[5]),
        scheduled_spec_decode_tokens={"decode": [7, 8]},
        preempted_req_ids={"other"})
    return fake, scheduled


def _last(tmp_path):
    path = next(tmp_path.glob("passive-step-*.jsonl"))
    return [json.loads(line) for line in path.read_text().splitlines()][-1]


def test_same_step_union_residual_and_host_scheduler_counts(monkeypatch, tmp_path):
    fake, scheduled = _setup(monkeypatch, tmp_path)
    passive_timing.begin_step(scheduled)
    with passive_timing.phase("target_forward"):
        with passive_timing.phase("fia", tokens=10):
            pass
    passive_timing.end_step()
    record = _last(tmp_path)
    assert record["status"] == "measured"
    assert record["bucket"] == "mixed"
    assert record["scheduled_prompt_tokens"] == 8
    assert record["scheduled_decode_tokens"] == 2
    assert record["scheduled_draft_tokens"] == 2
    assert record["preempted_requests"] == 1
    assert record["step_device_ms"] == 5.0
    assert record["oscar_union_ms"] == 1.0
    assert record["residual_ms"] == 4.0
    assert all(not hasattr(event, "synchronize") for event in fake.events)


def test_graph_replay_and_cross_stream_are_explicitly_missing(monkeypatch, tmp_path):
    fake, scheduled = _setup(monkeypatch, tmp_path)
    passive_timing.begin_step(scheduled)
    with passive_timing.phase("graph_replay"):
        pass
    passive_timing.end_step()
    assert _last(tmp_path)["missing_reason"] == "graph_replay_has_no_python_oscar_phase_breakdown"
    # Reset sampler quota for the same bucket to permit a second selected step.
    passive_timing._taken["mixed"] = 0
    passive_timing._seen["mixed"] = 0
    passive_timing.begin_step(scheduled)
    with passive_timing.phase("target_forward"):
        with passive_timing.phase("fia"):
            pass
    fake.stream = object()
    passive_timing.end_step()
    record = _last(tmp_path)
    assert record["missing_reason"] == "step_stream_mismatch_or_capture"
    assert record["step_device_ms"] is None
    assert record["partial_envelopes"] == [{"phase": "target_forward", "duration_ms": 3.0,
        "attention_backend": "oscar", "attention_union_ms": 1.0,
        "oscar_union_ms": 1.0, "residual_ms": 2.0,
        "scope": "same_stream_partial_envelope"}]


def test_incomplete_last_sample_is_recorded_without_synchronizing(monkeypatch, tmp_path):
    fake, scheduled = _setup(monkeypatch, tmp_path)
    fake.complete = False
    passive_timing.begin_step(scheduled)
    passive_timing.end_step()
    assert _last(tmp_path)["status"] == "pending"
    passive_timing._flush(final=True)
    assert _last(tmp_path)["missing_reason"] == "step_event_incomplete_at_worker_exit"
    assert len(passive_timing._pending) == 1  # Handles remain owned while unfinished.


def test_full_pending_queue_skips_before_allocating_another_event(monkeypatch, tmp_path):
    fake, scheduled = _setup(monkeypatch, tmp_path)
    fake.complete = False
    for index in range(passive_timing._MAX_PENDING):
        passive_timing._pending.append({"begin": fake.Event(enable_timing=True),
            "end": fake.Event(enable_timing=True), "phases": [], "step_id": f"old-{index}",
            "run_id": "external-window-0", "variant": "baseline", "bucket": "mixed"})
    allocated = len(fake.events)
    passive_timing.begin_step(scheduled)
    assert len(fake.events) == allocated
    assert len(passive_timing._pending) == passive_timing._MAX_PENDING
    assert passive_timing._tls.step is None
    assert _last(tmp_path)["status"] == "skipped"


def test_end_capacity_race_keeps_unfinished_handles_and_finalize_never_resolves(monkeypatch, tmp_path):
    fake, scheduled = _setup(monkeypatch, tmp_path)
    fake.complete = False
    passive_timing.begin_step(scheduled)
    active = passive_timing._tls.step
    # Simulate a second runner thread filling the shared queue after begin.
    for index in range(passive_timing._MAX_PENDING):
        passive_timing._pending.append({"begin": fake.Event(enable_timing=True),
            "end": fake.Event(enable_timing=True), "phases": [], "step_id": f"other-{index}",
            "run_id": "external-window-0", "variant": "baseline", "bucket": "mixed"})
    passive_timing.end_step()
    assert len(passive_timing._pending) == passive_timing._MAX_PENDING + 1
    assert active in passive_timing._pending
    assert _last(tmp_path)["missing_reason"] == "bounded_pending_capacity_race"
    # Active shutdown writes only metadata; a resolver would read an unfinished event.
    passive_timing._tls.step = active
    monkeypatch.setattr(passive_timing, "_resolved", lambda *_args: (_ for _ in ()).throw(
        AssertionError("unfinished event must not be timed at exit")))
    passive_timing._finalize()
    lines = [json.loads(line) for line in next(tmp_path.glob("passive-step-*.jsonl")).read_text().splitlines()]
    assert any(row.get("missing_reason") == "worker_exit_with_active_step" for row in lines)
    passive_timing._tls.step = None


def test_unpaired_and_error_steps_keep_events_until_query_completes(monkeypatch, tmp_path):
    fake, scheduled = _setup(monkeypatch, tmp_path)
    fake.complete = False
    passive_timing.begin_step(scheduled)
    active = passive_timing._tls.step
    passive_timing.begin_step(scheduled)  # sample_tokens did not follow execute_model.
    assert active in passive_timing._pending
    assert not list(tmp_path.glob("passive-step-*.jsonl"))
    fake.complete = True
    passive_timing._flush()
    assert _last(tmp_path)["missing_reason"] == "next_step_started_before_sample_tokens"
    passive_timing._taken["mixed"] = passive_timing._seen["mixed"] = 0
    fake.complete = False
    passive_timing.begin_step(scheduled)
    active = passive_timing._tls.step
    passive_timing.end_step(RuntimeError("device failure"))
    assert active in passive_timing._pending
    fake.complete = True
    passive_timing._flush()
    assert _last(tmp_path)["missing_reason"].startswith("runner_error:")


def test_native_attention_union_excludes_nested_backend_and_never_defaults_to_step(monkeypatch, tmp_path):
    fake, scheduled = _setup(monkeypatch, tmp_path)
    control = tmp_path / "control.json"
    control.write_text(json.dumps({"enabled": True, "run_id": "native-window", "variant": "native"}))
    passive_timing.begin_step(scheduled)
    with passive_timing.phase("target_forward"):
        with passive_timing.phase("native_attention", backend="AscendC8AttentionBackendImpl"):
            with passive_timing.phase("native_attention", backend="AscendAttentionBackendImpl"):
                pass
    passive_timing.end_step()
    measured = _last(tmp_path)
    assert measured["status"] == "measured"
    assert measured["attention_backend"] == "native"
    assert measured["attention_union_ms"] == 3.0
    assert measured["oscar_union_ms"] is None
    assert measured["residual_ms"] == measured["step_device_ms"] - 3.0
    assert measured["native_attention_backends"] == [
        "AscendAttentionBackendImpl", "AscendC8AttentionBackendImpl"]
    passive_timing._taken["mixed"] = passive_timing._seen["mixed"] = 0
    passive_timing.begin_step(scheduled)
    with passive_timing.phase("target_forward"):
        pass
    passive_timing.end_step()
    missing = _last(tmp_path)
    assert missing["status"] == "missing"
    assert missing["missing_reason"] == "no_native_attention_phase_events"
    assert missing["residual_ms"] is None
