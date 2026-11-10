"""Archive #27/#94/#125/#157/#163/#176: host plumbing, not NPU acceptance."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.probe_evidence import ProbeEvidence, configure, evaluate_mtp_events
from tools.probe_kv_offload_service import require_swap_evidence
from tools.target_cli import serve_argv
from tools.service_probe import ServiceProbeError

ROOT = Path(__file__).resolve().parents[1]


def test_stats_flag_is_explicit_and_default_argv_unchanged():
    config = json.loads((ROOT / "configs/target.json").read_text())
    before = serve_argv(config)
    assert "--disable-log-stats" not in before
    assert serve_argv(dict(config, disable_log_stats=False)) == before
    enabled = serve_argv(dict(config, disable_log_stats=True))
    enabled.remove("--disable-log-stats")
    assert enabled == before
    node = json.loads((ROOT / "configs/benchmark_node93.json").read_text())
    assert node["disable_log_stats"] is True
    assert "--disable-log-stats" in serve_argv(node)
    for bad in (0, 1, "true", "false", None):
        with pytest.raises(ValueError, match="explicit boolean"):
            serve_argv(dict(config, disable_log_stats=bad))


def test_zero_acceptance_and_invalid_slots_exceeding_schedule_are_not_effective_drafts():
    rows = [{"event": "mtp_verified", "scheduled_draft_slots": 3,
             "num_accepted_tokens": 0, "invalid_draft_slots": 9}]
    result = evaluate_mtp_events(rows)
    assert result["status"] == "passed"
    assert result["scheduled_draft_slots"] == 3 and result["invalid_draft_slots"] == 9
    assert result["accepted_tokens"] == 0 and "acceptance_rate" not in result
    assert evaluate_mtp_events([])["status"] == "failed"
    for field, value in (("scheduled_draft_slots", 0), ("scheduled_draft_slots", 4),
                         ("num_accepted_tokens", -1), ("num_accepted_tokens", 4),
                         ("invalid_draft_slots", -1), ("invalid_draft_slots", True)):
        assert evaluate_mtp_events([dict(rows[0], **{field: value})])["status"] == "failed"


def test_run_and_window_provenance_duplicates_and_stop(tmp_path):
    environment, lifecycle = {}, {}
    configure(environment, lifecycle, tmp_path)
    probe = ProbeEvidence(SimpleNamespace(lifecycle=lifecycle)).start()
    directory = Path(lifecycle["probe_evidence"]["directory"])
    row = {"schema": 1, "run_id": probe.run_id, "window_id": probe.window_id,
           "pid": 1, "sequence": 1, "event": "mtp_verified", "request_id": "req",
           "scheduled_draft_slots": 3, "num_accepted_tokens": 0, "invalid_draft_slots": 0}
    path = directory / "scheduler-1.jsonl"
    path.write_text(json.dumps(dict(row, window_id="old-window")) + "\n" + json.dumps(row) + "\n")
    assert probe.read() == [row]
    probe.stop()
    assert not (directory / "active.json").exists()
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicated"):
        probe.read()
    path.write_text(json.dumps(dict(row, run_id="foreign")) + "\n")
    with pytest.raises(ValueError, match="foreign"):
        probe.read()


def test_terminal_stop_caches_no_more_filesystem_polling(tmp_path, monkeypatch):
    from oscar_ascend import probe_evidence as observer
    environment, lifecycle = {}, {}
    configure(environment, lifecycle, tmp_path)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    probe = ProbeEvidence(SimpleNamespace(lifecycle=lifecycle)).start()
    assert observer._window() is not None
    probe.stop()
    assert observer._window() is None  # Observe terminal marker exactly once.
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: pytest.fail("formal traffic must not read files"))
    monkeypatch.setattr(Path, "exists", lambda *a, **k: pytest.fail("formal traffic must not stat files"))
    for _ in range(3):
        assert observer._window() is None


@pytest.mark.parametrize("probe_first", [False, True])
def test_probe_and_production_hook_registration_orders_restore_exact_original(tmp_path, monkeypatch, probe_first):
    import sys
    from types import ModuleType
    from oscar_ascend import plugin
    plugin.unregister()
    module = ModuleType("vllm.v1.core.sched.scheduler")
    class Scheduler:
        def _preempt_request(self, request, timestamp):
            return "original"
        def make_spec_decoding_stats(self, *args):
            return None
    module.Scheduler = Scheduler
    original = Scheduler._preempt_request
    def production_callback(module):
        plugin._patch(module.Scheduler, "_preempt_request", lambda original:
                      lambda *args, **kwargs: original(*args, **kwargs))
    monkeypatch.setattr(plugin, "_callbacks", {module.__name__: production_callback})
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("OSCAR_ENABLED", "1")
    monkeypatch.delenv("OSCAR_PROBE_EVIDENCE_DIR", raising=False)
    try:
        if not probe_first:
            plugin.register()
        monkeypatch.setenv("OSCAR_PROBE_EVIDENCE_DIR", str(tmp_path))
        monkeypatch.setenv("OSCAR_PROBE_RUN_ID", "registration")
        plugin.register()
        assert Scheduler._preempt_request is not original
        plugin.unregister()
        assert Scheduler._preempt_request is original
    finally:
        plugin.unregister()


def test_failed_probe_registration_removes_finder(tmp_path, monkeypatch):
    import sys
    from oscar_ascend import probe_evidence as observer
    observer.unregister()
    monkeypatch.setenv("OSCAR_PROBE_EVIDENCE_DIR", str(tmp_path))
    monkeypatch.setenv("OSCAR_PROBE_RUN_ID", "invalid-api")
    monkeypatch.setitem(sys.modules, "vllm.v1.core.sched.scheduler", SimpleNamespace(Scheduler=object))
    before = tuple(sys.meta_path)
    with pytest.raises(AttributeError):
        observer.register()
    assert tuple(sys.meta_path) == before


def test_cancelled_deferred_observer_does_not_install_ghost_hooks(tmp_path, monkeypatch):
    import sys
    from oscar_ascend import probe_evidence as observer
    from oscar_ascend.plugin import _AfterExecLoader
    monkeypatch.setenv("OSCAR_PROBE_EVIDENCE_DIR", str(tmp_path))
    monkeypatch.setenv("OSCAR_PROBE_RUN_ID", "deferred-cancel")
    monkeypatch.delitem(sys.modules, "vllm.v1.core.sched.scheduler", raising=False)
    observer.register()
    runner = observer._finder.callback_runner
    class Loader:
        def exec_module(self, module):
            module.loaded = True
    calls = []
    loader = _AfterExecLoader(Loader(), calls.append, runner)
    observer.unregister()
    observer.register()  # A different run cannot resurrect an old loader.
    try:
        module = SimpleNamespace()
        loader.exec_module(module)
        assert module.loaded and not calls and not observer._patches
    finally:
        observer.unregister()


def test_native_preemption_must_join_real_same_request_tp_swap():
    from test_offload_service_probe import swap_logs
    rows = [{"event": "preempted", "request_id": "req-a", "preemptions_before": 0,
             "preemptions_after": 1}]
    result = require_swap_evidence(swap_logs(), "", "", native_events=rows)
    assert result["completed_preemptions"] == 1 and "native_preemptions_delta" not in result
    with pytest.raises(ServiceProbeError, match="same-generation"):
        require_swap_evidence(swap_logs(), "", "", native_events=[dict(rows[0], request_id="other")])
    with pytest.raises(ServiceProbeError, match="completed native"):
        require_swap_evidence(swap_logs(), "", "", native_events=[])
    with pytest.raises(ValueError, match="exactly once"):
        require_swap_evidence(swap_logs(), "", "", native_events=[dict(rows[0], preemptions_after=0)])


@pytest.mark.parametrize("emit", [True, False])
def test_real_owned_http_probe_disabled_stats_uses_only_independent_evidence(tmp_path, emit):
    from test_service_workflow import FAKE_SERVER, configured, fake_command, invoke
    config, path = configured(tmp_path)
    config["disable_log_stats"] = True
    path.write_text(json.dumps(config))
    command = fake_command(tmp_path, config)
    # The subprocess HTTP fixture emits host evidence only inside the explicit
    # active window. Native method correctness has a separate AST integration.
    injection = """count += 1
            if %s:
                sys.path.insert(0, os.getcwd())
                from oscar_ascend.probe_evidence import _emit, _window
                _emit(_window(), 'mtp_verified', 'request-' + str(count),
                      scheduled_draft_slots=3, num_accepted_tokens=0, invalid_draft_slots=5)
""" % repr(emit)
    program = FAKE_SERVER.replace("count += 1\n", injection)
    program = program.replace("elif self.path == '/metrics':", "elif self.path == '/metrics':\n            raise AssertionError('disabled stats probe must not request counters')")
    Path(command[1]).write_text(program)
    report, _ = invoke(tmp_path, path, command, correctness_only=True)
    assert report["status"] == ("passed" if emit else "failed")
    assert report["mtp"]["source"] == "native_scheduler_probe_events"
    assert report["server"]["cleanup_complete"] is True
    assert not (Path(report["server"]["probe_evidence"]["directory"]) / "active.json").exists()
    assert not (tmp_path / "logs/metrics_before.prom").exists()


@pytest.mark.parametrize("native,explicit", [(False, False), (True, False), (True, True)])
def test_owned_server_requires_explicit_functional_scope_for_observer(tmp_path, monkeypatch, native, explicit):
    from test_service_workflow import configured, fake_command
    from tools.service_probe import managed_server
    config, path = configured(tmp_path)
    config["disable_log_stats"] = True
    path.write_text(json.dumps(config))
    command = fake_command(tmp_path, config)
    program = Path(command[1])
    env_path = tmp_path / "child-env.json"
    program.write_text("import os, json\nfrom pathlib import Path\n"
                      + f"Path({str(env_path)!r}).write_text(json.dumps({{k: v for k, v in os.environ.items() if k.startswith('OSCAR_PROBE_')}}))\n"
                      + program.read_text())
    monkeypatch.setenv("OSCAR_PROBE_EVIDENCE_DIR", "stale-directory")
    monkeypatch.setenv("OSCAR_PROBE_RUN_ID", "stale-run")
    lifecycle = {}
    with managed_server(config, path, log_dir=tmp_path / "logs", command=command,
                        lifecycle=lifecycle, native=native, native_probe_evidence=explicit):
        actual = json.loads(env_path.read_text())
        assert bool(actual) == explicit
        if explicit:
            assert actual["OSCAR_PROBE_RUN_ID"] != "stale-run"
    assert lifecycle["cleanup_complete"] is True


def test_evidence_stop_write_failure_still_cleans_owned_child_and_fails(tmp_path, monkeypatch):
    from test_service_workflow import configured, fake_command
    from tools import service_probe
    config, path = configured(tmp_path)
    config["disable_log_stats"] = True
    lifecycle = {}
    def fail_stop(info):
        raise OSError("injected terminal marker failure")
    monkeypatch.setattr(service_probe, "stop_run", fail_stop)
    with pytest.raises(ServiceProbeError, match="close native probe"):
        with service_probe.managed_server(config, path, log_dir=tmp_path / "logs",
                command=fake_command(tmp_path, config), lifecycle=lifecycle, native_probe_evidence=True):
            pass
    assert lifecycle["cleanup_complete"] is True and lifecycle["exit_code"] is not None
    assert lifecycle["status"] == "failed"
    assert "injected terminal" in lifecycle["evidence_stop_error"]


def test_stats_disabled_formal_service_outlives_probe_without_passive_metrics(tmp_path, monkeypatch):
    from test_service_workflow import configured, fake_command, invoke
    import benchmarks.passive
    config, path = configured(tmp_path)
    config["disable_log_stats"] = True
    path.write_text(json.dumps(config))
    command = fake_command(tmp_path, config, "stop")
    program = Path(command[1])
    program.write_text(program.read_text().replace("count += 1\n", """count += 1
            sys.path.insert(0, os.getcwd())
            from oscar_ascend.probe_evidence import _emit, _window
            _emit(_window(), 'mtp_verified', 'request-' + str(count),
                  scheduled_draft_slots=3, num_accepted_tokens=0, invalid_draft_slots=0)
"""))
    monkeypatch.setattr(benchmarks.passive, "observe", lambda *a, **k:
                        pytest.fail("disabled native statistics must not start passive metrics observer"))
    report, _ = invoke(tmp_path, path, command, serve=True)
    assert report["status"] == "stopped"
    assert report["mtp"]["status"] == "passed"
    assert report["external_observation"]["reason"] == "disabled_by_config"
    assert report["server"]["cleanup_complete"] is True
    directory = Path(report["server"]["probe_evidence"]["directory"])
    assert (directory / "closed.json").is_file() and not (directory / "active.json").exists()
