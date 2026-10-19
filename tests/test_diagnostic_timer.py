# Archive #133/#140/#143: bounded same-thread profiler lifecycle, no HTTP/NPU
# synchronization, mature native batch and warm-history coverage, honest gaps.
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace, ModuleType

import pytest

from oscar_ascend import diagnostic_timer as timer


def schedule(kind="decode", batch=4, computed=32768):
    ids = [f"r{index}" for index in range(batch)]
    return SimpleNamespace(
        num_scheduled_tokens={key: 4 if kind == "decode" else 16384 for key in ids},
        scheduled_new_reqs=[], scheduled_cached_reqs=SimpleNamespace(
            req_ids=ids, num_output_tokens=[8 if kind == "decode" else 0] * batch,
            num_computed_tokens=[computed] * batch),
        scheduled_spec_decode_tokens={key: [1, 2, 3] for key in ids} if kind == "decode" else {},
        preempted_req_ids=[])


class Scope:
    def __enter__(self):
        return self
    def __exit__(self, *_):
        return False


@pytest.fixture
def setup(monkeypatch, tmp_path):
    control = tmp_path / "control.json"
    settings = {"enabled": True, "run_id": "case", "variant": "candidate", "profile": True,
                "output_dir": str(tmp_path), "target_batch": 4, "min_decode_requests": 4}
    control.write_text(json.dumps(settings))
    monkeypatch.setenv("OSCAR_TIMER_CONTROL", str(control))
    fake_torch = ModuleType("torch")
    fake_torch.npu = SimpleNamespace(is_current_stream_capturing=lambda: False, is_available=lambda: True)
    fake_torch.distributed = SimpleNamespace(is_initialized=lambda: True, get_rank=lambda: 2)
    markers = []
    fake_profiler = ModuleType("torch.profiler")
    fake_profiler.record_function = lambda label: (markers.append(label) or Scope())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.profiler", fake_profiler)
    monkeypatch.setattr(timer, "_run", None)
    monkeypatch.setattr(timer, "_serial", 0)
    timer._tls.step = None
    profiles = []

    class Profiler:
        def __init__(self, directory, metric):
            self.directory, self.metric = directory, metric
            self.calls = []
            self.owner = threading.get_ident()
            self.state = "NONE"
        def record(self, value):
            assert self.owner == threading.get_ident()
            self.calls.append(value)
        def start(self):
            self.record("start")
            self.state = "RECORD_AND_SAVE"
        def step(self):
            self.record("step")
            assert self.state == "RECORD_AND_SAVE"
            self.state = "NONE"
            self.directory.mkdir(parents=True)
            (self.directory / "trace_view.json").write_text('{"traceEvents": []}')
        def stop(self):
            self.record("stop")
            assert self.state == "NONE", "#133: stop must not occur during RECORD"

    def factory(directory, metric, worker_name):
        result = Profiler(directory, metric)
        profiles.append(result)
        return result
    monkeypatch.setattr(timer, "_create_profiler", factory)
    yield SimpleNamespace(path=tmp_path, control=control, settings=settings, profiles=profiles,
                          markers=markers, torch=fake_torch)
    timer._tls.step = None
    timer._run = None


def rows(setup):
    return [json.loads(line) for path in setup.path.glob("timer-steps-*.jsonl")
            for line in path.read_text().splitlines()]


def manifests(setup):
    return [json.loads(path.read_text()) for path in setup.path.glob("windows/*/*/step*.json")]


def test_disabled_is_no_device_work(monkeypatch):
    monkeypatch.delenv("OSCAR_TIMER_CONTROL", raising=False)
    monkeypatch.setattr(timer, "_run", None)
    monkeypatch.setattr(timer, "_capture", lambda: pytest.fail("disabled imported device state"))
    timer.begin_step(schedule())
    with timer.phase("anything", tensor=object()):
        pass
    assert not timer.active()


def test_one_step_record_save_export_stop_same_thread_and_host_fields(setup):
    timer.begin_step(schedule())
    with timer.phase("target_forward"):
        with timer.phase("graph_replay", requests=4, tokens=16):
            pass
    timer.end_step()
    assert setup.profiles[0].calls == ["start", "step", "stop"]
    assert setup.markers == ["oscar_timer::target_forward", "oscar_timer::graph_replay"]
    record = rows(setup)[0]
    assert record["rank"] == 2 and record["graph_replays"] == 1
    assert record["context_min"] == 32772 and record["scheduled_decode_requests"] == 4
    assert record["scheduled_draft_tokens"] == 12
    assert record["device_ms"] is None and record["profile_status"] == "complete"
    assert record["host_step_includes_profile_setup"] is False
    assert record["host_step_includes_profile_export"] is False
    assert record["profile_start_ms"] >= 0 and record["profile_export_ms"] >= 0
    manifest = manifests(setup)[0]
    assert manifest["status"] == "complete" and manifest["profile_artifacts"] == ["trace_view.json"]


def test_requires_mature_batch_and_warm_history_then_bounded_four_windows(setup):
    for _ in range(4):
        timer.begin_step(schedule(batch=1))
        assert not timer.active()
        timer.begin_step(schedule("prefill", 1, 0))
        assert not timer.active()
    for kind in ("prefill", "prefill", "decode", "decode", "decode", "prefill"):
        timer.begin_step(schedule(kind, batch=4 if kind == "decode" else 1, computed=16384))
        timer.end_step()
    assert len(setup.profiles) == 4
    assert [item.metric for item in setup.profiles] == ["PipeUtilization", "Memory"] * 2
    assert len(rows(setup)) == 4
    summary = json.loads(next(setup.path.glob("windows/*/*/run-summary.json")).read_text())
    assert summary["status"] == "complete" and summary["skipped_cold_prefill_steps"] == 4


def test_unknown_cpu_history_is_explicit_missing_and_never_samples_first_chunk(setup):
    data = schedule("prefill", 1)
    del data.scheduled_cached_reqs.num_computed_tokens
    timer.begin_step(data)
    assert not timer.active()
    timer._finalize()
    summary = json.loads(next(setup.path.glob("windows/*/*/run-summary.json")).read_text())
    assert summary["early_history_unknown_steps"] == 1
    assert summary["status"] == "missing"
    assert summary["missing_reason"] == "mature_decode_not_observed"


def test_run_id_resets_quota_and_disarm_is_not_profiler_rpc(setup):
    for _ in range(3):
        timer.begin_step(schedule())
        timer.end_step()
    assert len(setup.profiles) == 2
    setup.settings["run_id"] = "next"
    setup.control.write_text(json.dumps(setup.settings))
    timer.begin_step(schedule())
    timer.end_step()
    assert len(setup.profiles) == 3
    setup.control.write_text('{"enabled": false}')
    timer.begin_step(schedule())
    assert not timer.active()
    assert all(item.calls == ["start", "step", "stop"] for item in setup.profiles)


def test_disarm_flushes_once_without_device_checks_or_repeated_summary_io(setup, monkeypatch):
    timer.begin_step(schedule())
    timer.end_step()
    setup.control.write_text('{"enabled": false}')
    calls = []
    monkeypatch.setattr(timer, "_summary", lambda final=False: calls.append(final))
    monkeypatch.setattr(timer, "_capture", lambda: pytest.fail("disarmed device query"))
    for _ in range(5):
        timer.begin_step(schedule())
        with timer.phase("native_gdn"):
            pass
    assert calls == [True]
    assert timer._run is None


def test_unsupported_metric_keeps_host_evidence_and_marks_gap(setup, monkeypatch):
    def unavailable(*_):
        raise timer.ProfilerUnsupported("AiCMetrics.Memory missing")
    monkeypatch.setattr(timer, "_create_profiler", unavailable)
    timer.begin_step(schedule())
    with timer.phase("native_attention"):
        pass
    timer.end_step()
    assert manifests(setup)[0]["status"] == "unsupported"
    assert rows(setup)[0]["profiled"] is False
    assert rows(setup)[0]["device_ms"] is None
    assert not setup.markers


def test_profiler_create_runtime_failure_records_first_error(setup, monkeypatch):
    def broken(*_):
        raise RuntimeError("CANN profiler init failed")
    monkeypatch.setattr(timer, "_create_profiler", broken)
    with pytest.raises(RuntimeError, match="CANN profiler init failed"):
        timer.begin_step(schedule())
    assert manifests(setup)[0]["status"] == "error"
    assert "profiler_create" in manifests(setup)[0]["error"]
    assert not timer.active()


def test_only_info_file_cannot_be_complete_profile(setup, monkeypatch):
    timer.begin_step(schedule())
    profile = setup.profiles[0]
    def step():
        profile.state = "NONE"
        profile.directory.mkdir(parents=True)
        (profile.directory / "profiler_info.json").write_text("{}")
    monkeypatch.setattr(profile, "step", step)
    timer.end_step()
    assert manifests(setup)[0]["missing_reason"] == "profiler_export_has_no_artifacts"


def test_runner_error_keeps_original_failure_and_exports_bounded_window(setup):
    timer.begin_step(schedule())
    timer.end_step(ValueError("actual model error"))
    assert manifests(setup)[0]["status"] == "error"
    assert "actual model error" in rows(setup)[0]["error"]
    assert setup.profiles[0].calls == ["start", "step", "stop"]


def test_export_failure_persists_manifest_and_does_not_pretend_success(setup, monkeypatch):
    timer.begin_step(schedule())
    monkeypatch.setattr(setup.profiles[0], "step", lambda: (_ for _ in ()).throw(RuntimeError("export failed")))
    with pytest.raises(RuntimeError, match="export failed"):
        timer.end_step()
    assert manifests(setup)[0]["status"] == "error"
    assert rows(setup)[0]["status"] == "error"
    assert "stop" not in setup.profiles[0].calls  # no unsafe RECORD stop after failed save


def test_dummy_and_capture_never_start_profile(setup, monkeypatch):
    monkeypatch.setattr(timer, "is_native_dummy_run", lambda: True)
    timer.begin_step(schedule())
    monkeypatch.setattr(timer, "is_native_dummy_run", lambda: False)
    setup.torch.npu.is_current_stream_capturing = lambda: True
    timer.begin_step(schedule())
    assert not setup.profiles and not timer.active()


def test_unexpected_dynamic_capture_cannot_be_complete_window(setup):
    timer.begin_step(schedule())
    timer.note_graph_capture()
    timer.end_step()
    assert manifests(setup)[0]["missing_reason"] == "unexpected_graph_capture_in_profiled_step"
    assert rows(setup)[0]["graph_capture_calls"] == 1


def test_scalar_only_fields_and_scope_limit(setup, monkeypatch):
    timer.begin_step(schedule())
    with pytest.raises(TypeError, match="host scalars"):
        timer.phase("bad", tensor=object())
    monkeypatch.setattr(timer, "_MAX_SCOPES", 1)
    with timer.phase("target_forward"):
        with timer.phase("nested"):
            pass
    timer.end_step()
    assert rows(setup)[0]["dropped_scopes"] == 1


def test_native_spawn_registration_is_observer_only_without_torch(setup):
    source = '''
import json, sys
from oscar_ascend import plugin
plugin.register()
print(json.dumps({"enabled": plugin.hook_status()["enabled"], "passive": plugin._passive_only,
                  "device_imports": [x for x in sys.modules if x.split('.')[0] in
                      {"torch", "torch_npu", "vllm", "vllm_ascend"}]}))
plugin.unregister()
'''
    env = dict(os.environ, OSCAR_ENABLED="0", OSCAR_TIMER_CONTROL=str(setup.control))
    result = subprocess.run([sys.executable, "-c", source], env=env, capture_output=True, text=True, check=True)
    record = json.loads(result.stdout)
    assert record == {"enabled": True, "passive": True, "device_imports": []}


def test_plugin_actual_runner_exec_to_sample_and_native_metadata_hooks(setup, monkeypatch):
    from oscar_ascend import plugin
    plugin.unregister()
    module = ModuleType("vllm_ascend.worker.model_runner_v1")
    class Runner:
        def _update_states(self):
            return "states"
        def _prepare_inputs(self):
            return "inputs"
        def _model_forward(self):
            return "forward"
        def propose_draft_token_ids(self):
            return "draft"
        def execute_model(self, scheduler_output):
            self._update_states()
            self._prepare_inputs()
            self._model_forward()
            return None
        def sample_tokens(self):
            return self.propose_draft_token_ids()
    module.NPUModelRunner = Runner
    monkeypatch.setenv("OSCAR_ENABLED", "0")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    original = Runner.execute_model
    try:
        plugin.register()
        runner = Runner()
        assert runner.execute_model(schedule()) is None
        assert timer.active() and setup.profiles[0].calls == ["start"]
        assert runner.sample_tokens() == "draft"
        assert not timer.active()
        record = rows(setup)[0]
        assert {item["phase"] for item in record["scopes"]} == {
            "host_update_states", "host_prepare_inputs", "target_forward", "sample_tokens", "draft_proposal"}
        assert not any("get_attn_backend" in name for name in plugin.hook_status()["patched"])
    finally:
        plugin.unregister()
    assert Runner.execute_model is original


def test_unmatched_execute_and_atexit_never_cross_thread_stop(setup):
    timer.begin_step(schedule())
    timer._finalize()
    assert setup.profiles[0].calls == ["start"]
    assert manifests(setup)[0]["status"] == "collecting"
    with pytest.raises(RuntimeError, match="next_execute_before_sample_tokens"):
        timer.begin_step(schedule())
    assert "next_execute_before_sample_tokens" in rows(setup)[0]["error"]
    assert not timer.active() and len(setup.profiles) == 1
    assert all(profile.calls == ["start", "step", "stop"] for profile in setup.profiles)


def test_native_metric_enum_and_single_step_schedule_contract(setup, monkeypatch):
    captured = {}
    api = SimpleNamespace(AiCMetrics=SimpleNamespace(PipeUtilization="pipe"),
                          ExportType=SimpleNamespace(Text="text"), ProfilerLevel=SimpleNamespace(Level1="level1"),
                          ProfilerActivity=SimpleNamespace(CPU="cpu", NPU="npu"),
                          _ExperimentalConfig=lambda **kwargs: kwargs,
                          schedule=lambda **kwargs: (captured.update(schedule=kwargs) or "schedule"),
                          tensorboard_trace_handler=lambda *args, **kwargs: (captured.update(handler=kwargs) or "handler"),
                          profile=lambda **kwargs: (captured.update(profile=kwargs) or "profiler"))
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(profiler=api))
    # The fixture factory is replaced here by the real, separately saved function.
    assert _REAL_CREATE(setup.path, "PipeUtilization", "worker") == "profiler"
    assert captured["schedule"] == {"wait": 0, "warmup": 0, "active": 1, "repeat": 1}
    assert captured["handler"]["async_mode"] is False
    with pytest.raises(timer.ProfilerUnsupported, match="AiCMetrics.Memory"):
        _REAL_CREATE(setup.path, "Memory", "worker")


_REAL_CREATE = timer._create_profiler
