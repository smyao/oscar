# Archive #94/#95/#125/#133/#143/#164: host orchestration and exact SSE contracts,
# not device completion, C16 precision or model-performance acceptance.
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import types

import pytest

from tools import c16_breakdown as runner
from tools.phase import PhaseResult


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        assert text and not add_special_tokens
        return [21, 33, 45]


def config_file(tmp_path):
    config = json.loads((runner.ROOT / "configs/target.json").read_text())
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"]}))
    config["model"] = str(model)
    path = tmp_path / "target.json"
    path.write_text(json.dumps(config))
    return path


def completed(rows, output_tokens):
    from benchmarks.compare import canonical_sha256
    return {"status": "completed", "completed_requests": len(rows), "failed_requests": 0, "timeouts": 0,
        "batch_elapsed_seconds": .1, "running_waiting_samples": [],
        "requests": [{"request_id": row["id"], "status": "completed", "done_received_ms": 100,
            "prompt_sha256": canonical_sha256(row["token_ids"]),
            "usage": {"prompt_tokens": len(row["token_ids"]), "completion_tokens": output_tokens},
            "generated_token_ids": [7] * output_tokens} for row in rows]}


@pytest.mark.parametrize("rear", [False, True])
def test_plan_preserves_native_settings_and_only_schedules_one_candidate_request(tmp_path, monkeypatch, capsys, rear):
    config = config_file(tmp_path)
    logs = tmp_path / "untouched"
    for name in ("run_phase", "load_tokenizer", "load_build_receipt", "managed_server", "read_npu_resources"):
        monkeypatch.setattr(runner, name, lambda *a, **k: pytest.fail("plan touched execution"))
    assert runner.main(["--config", str(config), "--log-dir", str(logs), "--plan"] +
                       (["--rear-cards"] if rear else [])) == 0
    plan = json.loads(capsys.readouterr().out)
    assert not logs.exists()
    assert [name for name, _ in plan["stages"]] == ["build-dependencies", "install-plugin", "build-ops",
                                                   "prepare-rotations", "build-c16-timer"]
    assert plan["services"] == [{"variant": "candidate",
        "warmup": {"requests": 1, "prompt_tokens": 128, "output_tokens": 16},
        "workload": {"requests": 1, "prompt_tokens": 32768, "output_tokens": 16}}]
    assert plan["devices"] == ([4, 5, 6, 7] if rear else [0, 1, 2, 3])
    assert plan["port"] == (7878 if rear else 8989)
    args = plan["vllm_argv"]
    assert "--quantization" not in args and "--scheduler-cls" not in args
    assert "--no-enable-prefix-caching" in args
    assert args[args.index("--dtype") + 1] == "bfloat16"
    assert args[args.index("--max-num-batched-tokens") + 1] == "16384"
    assert json.loads(args[args.index("--speculative_config") + 1])["num_speculative_tokens"] == 3
    assert json.loads(args[args.index("--compilation-config") + 1])["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert plan["capture_selection"]["max_calls_per_rank"] == 20


def fake_execution(tmp_path, monkeypatch, *, build_rc=0, failed_request=None, release="passed", summary="complete"):
    config = config_file(tmp_path)
    state = {"phases": [], "requests": [], "services": [], "released": []}
    def phase(name, command, **kwargs):
        state["phases"].append(name)
        for key in ("OSCAR_TIMER_CONTROL", "OSCAR_DEVICE_TIMING_CONTROL", "OSCAR_PASSIVE_TIMING_CONTROL",
                    "OSCAR_C16_BREAKDOWN_DIR", "OSCAR_C16_TIMER_LIBRARY", "OSCAR_C16_TIMER_MANIFEST"):
            assert key not in kwargs["env"]
        assert kwargs["env"]["OSCAR_PROFILER"] == kwargs["env"]["OSCAR_DEBUG_SYNC"] == "0"
        logfile = kwargs["log_dir"] / f"{name}.log"
        logfile.parent.mkdir(parents=True, exist_ok=True)
        logfile.write_text("test phase output\n")
        if name == "build-c16-timer" and not build_rc:
            output = Path(command[command.index("--output") + 1])
            library = output.parent / "liboscar_c16_timer.so"
            library.write_bytes(b"host fixture; not a loadable library")
            runner.atomic_json(output, {"status": "built", "library": str(library), "sha256": "fixture"})
        return PhaseResult(name, command, build_rc, 0, False, False, str(logfile), True)
    monkeypatch.setattr(runner, "run_phase", phase)
    monkeypatch.setattr(runner, "load_tokenizer", lambda *a: Tokenizer())
    monkeypatch.setattr(runner, "model_geometry", lambda *a: (["model.layers.3.self_attn.attn"], 256, "model-fingerprint"))
    monkeypatch.setattr(runner, "read_npu_resources", lambda *a, **k: {"test": "resource snapshot"})
    def release_resources(*args, **kwargs):
        state["released"].append(kwargs)
        return {"status": release}
    monkeypatch.setattr(runner, "wait_for_release", release_resources)
    def metrics(url, *, timeout, path):
        path.write_text("vllm:prompt_tokens_total 100\nvllm:generation_tokens_total 100\nvllm:num_preemptions_total 0\n")
        return {"status": "observed", "path": str(path), "mtp": {"vllm:spec_decode_num_drafts": 1}}
    monkeypatch.setattr(runner, "metrics_snapshot", metrics)
    @contextmanager
    def managed(config, config_path, *, log_dir, lifecycle, native, quiet):
        assert native is False and quiet is True
        state["services"].append("candidate")
        assert config["experimental_aligned_math"] is True
        assert Path(runner.os.environ["OSCAR_C16_BREAKDOWN_DIR"]).is_dir()
        receipt = json.loads(Path(runner.os.environ["OSCAR_C16_TIMER_MANIFEST"]).read_text())
        assert receipt["library"] == runner.os.environ["OSCAR_C16_TIMER_LIBRARY"]
        for key in ("OSCAR_TIMER_CONTROL", "OSCAR_DEVICE_TIMING_CONTROL", "OSCAR_PASSIVE_TIMING_CONTROL", "OSCAR_PROFILE_DIR"):
            assert key not in runner.os.environ
        assert runner.os.environ["OSCAR_PROFILER"] == runner.os.environ["OSCAR_TIMING_STDERR"] == "0"
        log_dir.mkdir(parents=True, exist_ok=True)
        logfile = log_dir / "server.log"
        logfile.write_text("owned candidate service fixture\n")
        lifecycle.update(cleanup_complete=False, status="healthy", log=str(logfile))
        try:
            yield types.SimpleNamespace(base_url="http://example.invalid", log=logfile, check_alive=lambda: None)
        finally:
            lifecycle.update(cleanup_complete=True, status="finished", exit_code=0)
    monkeypatch.setattr(runner, "managed_server", managed)
    def batch(url, rows, **kwargs):
        state["requests"].append({"rows": rows, **kwargs})
        if failed_request == len(state["requests"]):
            raise RuntimeError("injected request failure")
        return completed(rows, kwargs["output_tokens"])
    monkeypatch.setattr(runner, "run_batch", batch)
    module = types.ModuleType("tools.summarize_c16_breakdown")
    module.summarize = lambda root: {"status": summary, "root": str(root)}
    module.summary_lines = lambda report: ["C16_BREAKDOWN_BEGIN", "C16_BREAKDOWN_END"]
    monkeypatch.setitem(sys.modules, "tools.summarize_c16_breakdown", module)
    return config, state


def test_real_orchestration_uses_one_owned_candidate_lifecycle_and_restores_environment(tmp_path, monkeypatch, capsys):
    config, state = fake_execution(tmp_path, monkeypatch)
    for key in ("OSCAR_TIMER_CONTROL", "OSCAR_C16_BREAKDOWN_DIR", "OSCAR_C16_TIMER_LIBRARY", "OSCAR_C16_TIMER_MANIFEST"):
        monkeypatch.setenv(key, "old-unrelated-control")
    monkeypatch.setenv("OSCAR_PROFILER", "1")
    logs = tmp_path / "logs"
    assert runner.main(["--config", str(config), "--log-dir", str(logs)]) == 0
    assert state["services"] == ["candidate"] and len(state["released"]) == 1
    assert [len(item["rows"][0]["token_ids"]) for item in state["requests"]] == [128, 32768]
    assert [item["output_tokens"] for item in state["requests"]] == [16, 16]
    assert len({item["repeat"] for item in state["requests"]}) == 2
    manifest = json.loads((logs / "run.json").read_text())
    assert manifest["status"] == "completed" and manifest["expected_ranks"] == 4
    assert manifest["service"]["request"]["completion_check"]["status"] == "passed"
    assert len(manifest["services"]) == 1 and manifest["timer_build"]["status"] == "built"
    assert runner.os.environ["OSCAR_TIMER_CONTROL"] == "old-unrelated-control"
    assert runner.os.environ["OSCAR_C16_TIMER_MANIFEST"] == "old-unrelated-control"
    assert runner.os.environ["OSCAR_PROFILER"] == "1"
    output = capsys.readouterr().out
    assert output.count("C16_BREAKDOWN_BEGIN") == 1 and len(output.splitlines()) < 40


@pytest.mark.parametrize("failure,rc", [("build", 17), ("warmup", 1), ("request", 1), ("release", 1), ("missing", 2)])
def test_failure_retains_true_exit_cleanup_and_evidence(tmp_path, monkeypatch, failure, rc):
    config, state = fake_execution(tmp_path, monkeypatch, build_rc=17 if failure == "build" else 0,
        failed_request={"warmup": 1, "request": 2}.get(failure), release="failed" if failure == "release" else "passed",
        summary="missing" if failure == "missing" else "complete")
    logs = tmp_path / "logs"
    assert runner.main(["--config", str(config), "--log-dir", str(logs)]) == rc
    report = json.loads((logs / "run.json").read_text())
    assert report["returncode"] == rc
    assert state["services"] == ([] if failure == "build" else ["candidate"])
    assert len(state["released"]) == (0 if failure == "build" else 1)
    if failure != "build":
        assert report["service"]["lifecycle"]["cleanup_complete"] is True
    if failure == "warmup":
        assert len(state["requests"]) == 1


@pytest.mark.parametrize("value", [None, "relative.so", "/nonexistent/lib.so"])
def test_success_build_without_real_absolute_artifact_is_rejected(tmp_path, value):
    path = tmp_path / "receipt.json"
    runner.atomic_json(path, {"status": "built", "library": value})
    with pytest.raises(runner.TimerRunError, match="nonempty absolute"):
        runner.load_build_receipt(path)


@pytest.mark.parametrize("key,value", [("profiler_config", {"profiler": "torch"}), ("diagnostic_device_timing", True),
                                       ("enable_prefix_caching", True), ("enforce_eager", True)])
def test_no_legacy_profiler_or_changed_inference_mode(tmp_path, key, value):
    config = json.loads(config_file(tmp_path).read_text())
    config[key] = value
    with pytest.raises(ValueError):
        runner.validate_config(config, 16)


def test_actual_sse_warmup_and_one_32k_request_complete(tmp_path, monkeypatch):
    from benchmarks import mixed
    from tests.test_benchmark_mixed import sse_service
    config, state = fake_execution(tmp_path, monkeypatch)
    with sse_service() as (url, requests):
        monkeypatch.setattr(runner, "run_batch", lambda _url, rows, **kwargs: mixed.run_batch(url, rows, **kwargs))
        assert runner.main(["--config", str(config), "--log-dir", str(tmp_path / "logs")]) == 0
    assert len(requests) == 2 and [len(request["prompt"]) for request in requests] == [128, 32768]
    assert all(request["max_tokens"] == 16 for request in requests)
    assert state["services"] == ["candidate"] and len(state["released"]) == 1


def test_interrupt_releases_blocked_transport_before_executor_wait(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import os
    import signal
    import subprocess
    import threading
    import time
    config, state = fake_execution(tmp_path, monkeypatch)
    processes = []
    @contextmanager
    def managed(config, config_path, *, log_dir, lifecycle, native, quiet):
        log_dir.mkdir(parents=True, exist_ok=True)
        logfile = log_dir / "server.log"
        logfile.write_text("owned process transport fixture\n")
        process = subprocess.Popen([sys.executable, "-c", "import time; print('ready',flush=True); time.sleep(30)"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(process)
        assert process.stdout.readline() == b"ready\n"
        lifecycle["cleanup_complete"] = False
        try:
            yield types.SimpleNamespace(process=process, base_url="http://example.invalid", log=logfile,
                                         check_alive=lambda: None)
        finally:
            lifecycle["cleanup_complete"] = runner.cleanup_group(process, .2)
            process.stdout.close()
    monkeypatch.setattr(runner, "managed_server", managed)
    def blocked(url, rows, **kwargs):
        if len(rows[0]["token_ids"]) == 128:
            return completed(rows, kwargs["output_tokens"])
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(processes[0].stdout.read)
            interrupter = threading.Timer(.05, lambda: os.kill(os.getpid(), signal.SIGINT))
            interrupter.start()
            try:
                future.result()
            finally:
                interrupter.cancel()
        pytest.fail("interrupt did not escape the blocked transport")
    monkeypatch.setattr(runner, "run_batch", blocked)
    logs = tmp_path / "logs"
    started = time.monotonic()
    assert runner.main(["--config", str(config), "--log-dir", str(logs)]) == 130
    assert time.monotonic() - started < 5 and processes[0].poll() is not None
    assert len(state["released"]) == 1
    manifest = json.loads((logs / "run.json").read_text())
    assert manifest["status"] == "interrupted" and manifest["service"]["lifecycle"]["cleanup_complete"] is True


def test_nonempty_output_directory_is_not_overwritten(tmp_path):
    config = config_file(tmp_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "run.json").write_text("original evidence")
    with pytest.raises(SystemExit):
        runner.main(["--config", str(config), "--log-dir", str(logs)])
    assert (logs / "run.json").read_text() == "original evidence"


def test_real_summary_keeps_missing_device_evidence_nonzero_and_copyable(tmp_path, monkeypatch, capsys):
    import importlib
    real_summary = importlib.import_module("tools.summarize_c16_breakdown")
    config, _ = fake_execution(tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules, "tools.summarize_c16_breakdown", real_summary)
    logs = tmp_path / "logs"
    assert runner.main(["--config", str(config), "--log-dir", str(logs)]) == 2
    result = json.loads((logs / "c16-breakdown-summary.json").read_text())
    assert result["status"] == "missing"
    text = (logs / "c16-breakdown-summary.txt").read_text()
    assert len(text.splitlines()) <= 150 and len(text.encode()) <= 40 * 1024
    assert "BEGIN" in text and "END" in text
    assert "SUMMARY_FAILED" not in capsys.readouterr().err
