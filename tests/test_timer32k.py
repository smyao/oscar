# Archive #94/#95/#125/#129/#133/#136/#140: host-only runner contract tests.
# Fake transport/owned lifecycles are not NPU, graph or performance evidence.
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import types

import pytest

from tools import timer32k
from tools.phase import PhaseResult


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(char) for char in text]


def config_file(tmp_path):
    config = json.loads((timer32k.ROOT / "configs/target.json").read_text())
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"]}))
    config["model"] = str(model)
    path = tmp_path / "target.json"
    path.write_text(json.dumps(config))
    return path


@pytest.mark.parametrize("rear", [False, True])
def test_plan_has_six_paired_small_batches_and_is_read_only(tmp_path, monkeypatch, capsys, rear):
    config = config_file(tmp_path)
    root = tmp_path / "not-created"
    for name in ("load_tokenizer", "run_phase", "read_npu_resources", "managed_server"):
        monkeypatch.setattr(timer32k, name, lambda *a, **kw: pytest.fail("plan touched execution"))
    assert timer32k.main(["--config", str(config), "--log-dir", str(root), "--plan"] +
                        (["--rear-cards"] if rear else [])) == 0
    result = json.loads(capsys.readouterr().out)
    assert not root.exists()
    assert result["arms"] == [
        {"variant": "native", "concurrency": 4, "modes": ["control", "profile"]},
        {"variant": "candidate", "concurrency": 4, "modes": ["control", "profile"]},
        {"variant": "candidate", "concurrency": 8, "modes": ["control", "profile"]}]
    assert result["prompt_tokens"] == 32768 and result["output_tokens"] == 256
    assert result["devices"] == ([4, 5, 6, 7] if rear else [0, 1, 2, 3])
    assert result["port"] == (7878 if rear else 8989)
    argv = result["vllm_argv"]
    assert "--quantization" not in argv
    assert "--scheduler-cls" not in argv
    assert "--no-enable-prefix-caching" in argv
    assert argv[argv.index("--max-num-batched-tokens") + 1] == "16384"
    assert json.loads(argv[argv.index("--speculative_config") + 1])["num_speculative_tokens"] == 3
    assert [stage[0] for stage in result["stages"]] == list(timer32k.BUILD_PHASES)


def test_prompts_are_natural_exact_deterministic_and_subset_paired():
    rows = timer32k.build_rows(Tokenizer())
    assert len(rows) == 8
    assert rows == timer32k.build_rows(Tokenizer())
    assert all(len(row["token_ids"]) == 32768 for row in rows)
    assert len({row["id"] for row in rows}) == 8
    assert len({timer32k.canonical_sha256(row["token_ids"]) for row in rows}) == 8
    assert timer32k.prompt_manifest(rows[:4]) == timer32k.prompt_manifest(rows)[:4]


def completed_sample(rows, output_tokens):
    return {"status": "completed", "batch_elapsed_seconds": 3.0,
        "completed_requests": len(rows), "failed_requests": 0, "timeouts": 0,
        "requests": [{"request_id": row["id"], "status": "completed", "done_received_ms": 3000,
            "prompt_sha256": timer32k.canonical_sha256(row["token_ids"]),
            "usage": {"prompt_tokens": len(row["token_ids"]), "completion_tokens": output_tokens},
            "generated_token_ids": [7] * output_tokens, "ttft_ms": 2000, "tpot_ms": 4, "e2e_ms": 3000}
            for row in rows], "running_waiting_samples": [{"running": len(rows), "waiting": 0}],
        "latency_ms": {"ttft_ms": {"p50": 2000, "p95": 2000},
            "tpot_ms": {"p50": 4, "p95": 4}, "e2e_ms": {"p50": 3000, "p95": 3000}},
        "throughput_tps": {"prompt": sum(len(r["token_ids"]) for r in rows) / 3,
                           "generation": len(rows) * output_tokens / 3}}


def fake_metrics(url, *, timeout, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("vllm:prompt_tokens_total 100\nvllm:generation_tokens_total 100\nvllm:num_preemptions_total 0\n")
    return {"status": "observed", "path": str(path), "mtp": {
        "vllm:spec_decode_num_drafts": 1, "vllm:spec_decode_num_draft_tokens": 3,
        "vllm:spec_decode_num_accepted_tokens": 2}}


@pytest.mark.parametrize("broken", ["usage", "done", "ids", "duplicate"])
def test_completion_guard_rejects_partial_or_mismatched_evidence(broken):
    rows = timer32k.build_rows(Tokenizer(), 2)
    sample = completed_sample(rows, 16)
    if broken == "usage":
        sample["requests"][0]["usage"]["completion_tokens"] = 15
    elif broken == "done":
        sample["requests"][0].pop("done_received_ms")
    elif broken == "ids":
        sample["requests"][0]["prompt_sha256"] = "bad"
    else:
        sample["requests"][1] = dict(sample["requests"][0])
    assert timer32k._completion_check(sample, rows, 16)["status"] == "failed"


def test_failed_profile_disarms_and_persists_complete_failure(tmp_path, monkeypatch):
    rows = timer32k.build_rows(Tokenizer(), 4)
    control = tmp_path / "control.json"
    server = types.SimpleNamespace(base_url="http://example.invalid", log=tmp_path / "server.log", check_alive=lambda: None)
    monkeypatch.setattr(timer32k, "metrics_snapshot", fake_metrics)
    def fail(*args, **kwargs):
        selected = json.loads(control.read_text())
        assert selected["enabled"] and selected["profile"]
        assert selected["target_batch"] == selected["min_decode_requests"] == 4
        raise RuntimeError("injected transport failure")
    monkeypatch.setattr(timer32k, "run_batch", fail)
    with pytest.raises(RuntimeError, match="injected transport"):
        timer32k.run_one_batch(server, {"served_model_name": "qwen3.8"}, rows,
            root=tmp_path, variant="candidate", mode="profile", control=control,
            output_tokens=256, timeout=50, pair_hash="pair", run_prefix="run")
    assert json.loads(control.read_text())["enabled"] is False
    report = json.loads((tmp_path / "candidate/b4-profile/batch.json").read_text())
    assert report["status"] == "failed" and "injected transport" in report["error"]


def test_control_flush_retains_failed_usage_without_rearming_profiler(tmp_path, monkeypatch):
    control = tmp_path / "control.json"
    timer32k.atomic_json(control, {"enabled": True})
    server = types.SimpleNamespace(base_url="http://example.invalid", log=tmp_path / "server.log", check_alive=lambda: None)
    def incomplete(url, rows, **kwargs):
        assert json.loads(control.read_text())["enabled"] is False
        assert kwargs["output_tokens"] == 1 and kwargs["timeout"] == 120
        sample = completed_sample(rows, 1)
        sample["requests"][0]["usage"]["completion_tokens"] = 0
        return sample
    monkeypatch.setattr(timer32k, "run_batch", incomplete)
    report = {}
    with pytest.raises(timer32k.TimerRunError, match="flush request failed"):
        timer32k.flush_disarmed_control(server, {"served_model_name": "qwen3.8"}, list(range(256)),
            control=control, run_id="test", timeout=900, report=report)
    assert report["status"] == "failed" and report["sample"]["requests"][0]["usage"]["completion_tokens"] == 0


def setup_fake_execution(tmp_path, monkeypatch, *, failing_batch=None, build_rc=0, release_status="passed"):
    path = config_file(tmp_path)
    state = {"phases": [], "requests": [], "services": [], "release": []}
    def phase(name, command, **kwargs):
        assert "OSCAR_TIMER_CONTROL" not in kwargs["env"]
        state["phases"].append(name)
        log = kwargs["log_dir"] / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("injected build receipt\n")
        return PhaseResult(name, command, build_rc, 0, False, False, str(log), True)
    monkeypatch.setattr(timer32k, "run_phase", phase)
    monkeypatch.setattr(timer32k, "load_tokenizer", lambda *args: Tokenizer())
    monkeypatch.setattr(timer32k, "model_geometry", lambda *args: (["model.layers.3.self_attn.attn"], 256, "model-fingerprint"))
    monkeypatch.setattr(timer32k, "read_npu_resources", lambda *args, **kwargs: {"test": "resource"})
    def release(*args, **kwargs):
        state["release"].append(kwargs)
        return {"status": release_status}
    monkeypatch.setattr(timer32k, "wait_for_release", release)
    monkeypatch.setattr(timer32k, "metrics_snapshot", fake_metrics)
    @contextmanager
    def managed(config, config_path, *, log_dir, lifecycle, native, quiet):
        assert quiet is True
        selected = json.loads(Path(config_path).read_text())
        assert selected["experimental_aligned_math"] is True
        assert selected["enable_prefix_caching"] is False
        assert selected["max_num_batched_tokens"] == 16384
        assert selected["speculative_config"]["num_speculative_tokens"] == 3
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / "server.log"
        log.write_text("server output preserved\n")
        state["services"].append("native" if native else "candidate")
        lifecycle.update(status="healthy", cleanup_complete=False, log=str(log))
        try:
            yield types.SimpleNamespace(base_url="http://example.invalid", log=log, check_alive=lambda: None)
        finally:
            lifecycle.update(status="finished", cleanup_complete=True, exit_code=0)
    monkeypatch.setattr(timer32k, "managed_server", managed)
    def batch(url, rows, **kwargs):
        control = json.loads(Path(timer32k.os.environ["OSCAR_TIMER_CONTROL"]).read_text())
        state["requests"].append({"rows": rows, "control": control, "kwargs": kwargs})
        if failing_batch is not None and len(state["requests"]) == failing_batch:
            raise RuntimeError("injected request error")
        return completed_sample(rows, kwargs["output_tokens"])
    monkeypatch.setattr(timer32k, "run_batch", batch)
    fake_summary = types.ModuleType("tools.summarize_timer")
    fake_summary.summarize = lambda root: {"root": str(root), "status": "diagnostic"}
    fake_summary.summary_lines = lambda report: ["OSCAR_TIMER_BEGIN", "OSCAR_TIMER_END"]
    monkeypatch.setitem(sys.modules, "tools.summarize_timer", fake_summary)
    return path, state


def test_complete_runner_pairs_exact_prompts_keeps_controls_unarmed_and_cleans_both_services(tmp_path, monkeypatch, capsys):
    config, state = setup_fake_execution(tmp_path, monkeypatch)
    root = tmp_path / "logs"
    monkeypatch.setenv("OSCAR_TIMER_CONTROL", "unrelated-original-control")
    monkeypatch.setenv("OSCAR_DEBUG_SYNC", "1")
    monkeypatch.setenv("OSCAR_PROFILER", "1")
    monkeypatch.setenv("OSCAR_TIMING_STDERR", "1")
    assert timer32k.main(["--config", str(config), "--log-dir", str(root)]) == 0
    assert timer32k.os.environ["OSCAR_TIMER_CONTROL"] == "unrelated-original-control"
    assert timer32k.os.environ["OSCAR_DEBUG_SYNC"] == "1"
    assert timer32k.os.environ["OSCAR_PROFILER"] == "1"
    assert timer32k.os.environ["OSCAR_TIMING_STDERR"] == "1"
    assert state["phases"] == list(timer32k.BUILD_PHASES)
    assert state["services"] == ["native", "candidate"] and len(state["release"]) == 2
    assert len(state["requests"]) == 11  # two warmups, six batches, three disarmed flushes
    long = [entry for entry in state["requests"] if len(entry["rows"][0]["token_ids"]) == 32768]
    assert [len(entry["rows"]) for entry in long] == [4, 4, 4, 4, 8, 8]
    assert [entry["control"]["enabled"] for entry in long] == [False, True] * 3
    assert len({entry["kwargs"]["repeat"] for entry in state["requests"]}) == 11
    flushes = [entry for entry in state["requests"] if entry["rows"][0]["id"] == "timer32k-control-flush"]
    assert len(flushes) == 3
    assert all(entry["control"]["enabled"] is False and entry["kwargs"]["output_tokens"] == 1 for entry in flushes)
    assert long[0]["rows"] == long[1]["rows"] == long[2]["rows"] == long[3]["rows"] == long[4]["rows"][:4]
    manifest = json.loads((root / "run.json").read_text())
    batches = [batch for service in manifest["services"] for batch in service["batches"]]
    assert len({batch["pair_hash"] for batch in batches}) == 1
    assert all(batch["request_completion_check"]["status"] == "passed" for batch in batches)
    assert all(batch["control_flush"]["profile_enabled"] is False
               for batch in batches if batch["mode"] == "profile")
    assert manifest["status"] == "completed" and manifest["returncode"] == 0
    assert json.loads((root / "timer-control.json").read_text())["enabled"] is False
    output = capsys.readouterr().out
    assert output.count("OSCAR_TIMER_BEGIN") == 1 and len(output.splitlines()) < 50


@pytest.mark.parametrize("failure,expected_rc", [("build", 17), ("request", 1), ("release", 1)])
def test_failure_stops_next_service_and_keeps_true_rc_and_release(tmp_path, monkeypatch, failure, expected_rc):
    config, state = setup_fake_execution(tmp_path, monkeypatch,
        build_rc=17 if failure == "build" else 0,
        failing_batch=2 if failure == "request" else None,
        release_status="failed" if failure == "release" else "passed")
    root = tmp_path / "logs"
    assert timer32k.main(["--config", str(config), "--log-dir", str(root)]) == expected_rc
    manifest = json.loads((root / "run.json").read_text())
    assert manifest["status"] == "failed" and manifest["returncode"] == expected_rc
    assert state["services"] == ([] if failure == "build" else ["native"])
    assert len(state["release"]) == (0 if failure == "build" else 1)
    assert json.loads((root / "timer-control.json").read_text())["enabled"] is False
    if failure != "build":
        assert manifest["services"][0]["lifecycle"]["cleanup_complete"] is True


def test_existing_evidence_is_not_overwritten(tmp_path, monkeypatch):
    config = config_file(tmp_path)
    root = tmp_path / "logs"
    root.mkdir()
    (root / "run.json").write_text("original")
    with pytest.raises(SystemExit):
        timer32k.main(["--config", str(config), "--log-dir", str(root)])
    assert (root / "run.json").read_text() == "original"


@pytest.mark.parametrize("key,value", [("profiler_config", {"profiler": "torch"}),
                                       ("diagnostic_device_timing", True)])
def test_old_profilers_cannot_contaminate_control_measurements(tmp_path, key, value):
    config = config_file(tmp_path)
    data = json.loads(config.read_text())
    data[key] = value
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="rejects existing profiler"):
        timer32k.main(["--config", str(config), "--log-dir", str(tmp_path / "logs"), "--plan"])


def test_interrupt_closes_owned_process_before_waiting_for_blocked_client_thread(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import os
    import signal
    import subprocess
    import threading
    import time
    config, state = setup_fake_execution(tmp_path, monkeypatch)
    processes = []
    @contextmanager
    def managed(config, config_path, *, log_dir, lifecycle, native, quiet):
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / "server.log"
        log.write_text("owned process with a blocked client reader\n")
        process = subprocess.Popen([sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(30)"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(process)
        assert process.stdout.readline() == b"ready\n"
        lifecycle.update(cleanup_complete=False, log=str(log))
        try:
            yield types.SimpleNamespace(base_url="http://example.invalid", process=process, log=log,
                                         check_alive=lambda: None)
        finally:
            lifecycle["cleanup_complete"] = timer32k.cleanup_group(process, .2)
            process.stdout.close()
    monkeypatch.setattr(timer32k, "managed_server", managed)
    def blocked_batch(url, rows, **kwargs):
        assert timer32k.os.environ["OSCAR_PROFILER"] == "0"
        assert timer32k.os.environ["OSCAR_TIMING_STDERR"] == "0"
        if len(rows[0]["token_ids"]) == 128:
            return completed_sample(rows, kwargs["output_tokens"])
        # Models the existing HTTP ThreadPool context: exception unwinding
        # blocks on its reader until owned-server shutdown produces EOF.
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(processes[0].stdout.read)
            interrupter = threading.Timer(.05, lambda: os.kill(os.getpid(), signal.SIGINT))
            interrupter.start()
            try:
                future.result()
            finally:
                interrupter.cancel()
        pytest.fail("interrupt should escape the blocked request")
    monkeypatch.setattr(timer32k, "run_batch", blocked_batch)
    started = time.monotonic()
    root = tmp_path / "logs"
    assert timer32k.main(["--config", str(config), "--log-dir", str(root)]) == 130
    assert time.monotonic() - started < 5
    assert processes[0].poll() is not None
    assert len(state["release"]) == 1
    assert json.loads((root / "run.json").read_text())["status"] == "interrupted"


def test_real_summary_missing_device_evidence_keeps_client_results_and_returns_two(tmp_path, monkeypatch, capsys):
    import importlib
    real_summary = importlib.import_module("tools.summarize_timer")
    config, _ = setup_fake_execution(tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules, "tools.summarize_timer", real_summary)
    root = tmp_path / "logs"
    assert timer32k.main(["--config", str(config), "--log-dir", str(root)]) == 2
    report = json.loads((root / "timer-summary.json").read_text())
    assert report["evidence_status"] == "missing" and len(report["runs"]) == 6
    assert all(row["normal"]["completed_requests"] for row in report["runs"])
    output = capsys.readouterr().out
    assert "=== OSCAR_TIMER_SUMMARY_BEGIN ===" in output
    assert "=== OSCAR_TIMER_SUMMARY_END ===" in output
    assert len((root / "timer-summary.txt").read_text().splitlines()) <= 200


def test_managed_server_quiet_keeps_child_bytes_on_disk_without_terminal_spam(tmp_path, monkeypatch, capsys):
    from tools import service_probe
    config = json.loads((timer32k.ROOT / "configs/target.json").read_text())
    config.update(shutdown_timeout_seconds=1, service_startup_timeout_seconds=5)
    monkeypatch.setattr(service_probe, "_assert_port_free", lambda *args: None)
    marker = "CHILD-PAYLOAD-" * 100
    def health(*args, **kwargs):
        # A real child writes to the actual file descriptor. No network/NPU.
        path = tmp_path / "service" / "server.log"
        deadline = timer32k.time.monotonic() + 2
        while marker not in path.read_text() and timer32k.time.monotonic() < deadline:
            timer32k.time.sleep(.01)
        assert marker in path.read_text()
        return ""
    monkeypatch.setattr(service_probe, "_http", health)
    lifecycle = {}
    command = [sys.executable, "-c", f"import time; print({marker!r}, flush=True); time.sleep(30)"]
    with service_probe.managed_server(config, timer32k.ROOT / "configs/target.json",
            log_dir=tmp_path / "service", lifecycle=lifecycle, command=command, quiet=True):
        assert marker not in capsys.readouterr().out
    assert marker in Path(lifecycle["log"]).read_text()
    assert lifecycle["cleanup_complete"] is True
