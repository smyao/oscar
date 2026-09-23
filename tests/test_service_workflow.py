# Archive G21/G22/G31/#27/#50-52/#68/#72/#75/#94-95/#117: real subprocess
# fault injection around the service supervisor. Fake HTTP/telemetry/memory
# exist only in these tests and never constitute NPU acceptance evidence.
import json
from pathlib import Path
from types import SimpleNamespace
import socket
import subprocess
import sys
import time

import pytest

from tools.npu_resources import compare_release, wait_for_release
from tools import service_probe
from tools.phase import cleanup_group, group_exists
from tools.service_probe import (
    ServiceProbeError, evaluate_mtp_metrics, evaluate_telemetry, exact_prompt,
    parse_mtp_metrics, read_telemetry, run_service,
)

ROOT = Path(__file__).resolve().parents[1]
TEST_DEVICES = [40, 41, 42, 43]


class Tokenizer:
    def encode(self, sentence, add_special_tokens):
        assert sentence and not add_special_tokens
        return [17, 32, 901, 7, 10]


def snapshot(devices=TEST_DEVICES, free=2_000_000_000):
    return {"devices": devices, "memory": [{"physical_device": physical,
        "logical_device": logical, "free_bytes": free, "total_bytes": 4_000_000_000}
        for logical, physical in enumerate(devices)]}


def configured(tmp_path):
    config = json.loads((ROOT / "configs/target.json").read_text())
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"num_hidden_layers": 4,
        "full_attention_interval": 4, "head_dim": 64}))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config.update(model=str(model), devices=TEST_DEVICES, host="127.0.0.1", port=port,
        phase_timeout_seconds=10, service_startup_timeout_seconds=3,
        service_request_timeout_seconds=2, shutdown_timeout_seconds=.1,
        resource_release_timeout_seconds=.1, resource_release_tolerance_bytes=1024,
        mtp_metrics_timeout_seconds=.1)
    path = tmp_path / "target.json"
    path.write_text(json.dumps(config))
    return config, path


def telemetry_records():
    records = []
    for rank in range(4):
        for layer in ("language_model.model.layers.3.self_attn.attn", "mtp.layers.0.self_attn.attn"):
            records.append({"event": "cache_layout", "rank": rank, "layer": layer,
                "gdn_reshape": "native", "physical_blocks": 8, "physical_block_tokens": 2816,
                "snapshot_bytes_per_page": 333336, "page_bytes": 801792})
            records.append({"event": "attention_dispatched", "rank": rank, "layer": layer,
                            "route": "ascendc_int2_cv", "capture_origin": False})
        records.append({"event": "graph_capture_return", "rank": rank, "descriptor": "batch=4", "mode": "CUDAGraphMode.FULL"})
        records.append({"event": "graph_replay_launch_return", "rank": rank, "descriptor": "batch=4", "mode": "CUDAGraphMode.FULL"})
    return records


FAKE_SERVER = r'''
import json, os, pathlib, signal, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
port, mode = int(sys.argv[1]), sys.argv[2]
records = json.loads(pathlib.Path(sys.argv[3]).read_text())
count = 0
lock = threading.Lock()
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            body = b''
        elif self.path == '/metrics':
            value = 0 if mode == 'no_mtp' else count
            body = ('vllm:spec_decode_num_drafts_total{model_name="qwen3.5"} %s\n'
                    'vllm:spec_decode_num_draft_tokens_total{model_name="qwen3.5"} %s\n'
                    'vllm:spec_decode_num_accepted_tokens_total{model_name="qwen3.5"} %s\n'
                    % (value, value*3, value*2)).encode().replace(b'\\n', b'\n')
        else:
            self.send_error(404); return
        self.send_response(200); self.send_header('Content-Length', str(len(body)));self.end_headers();self.wfile.write(body)
    def do_POST(self):
        global count
        data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        assert self.path == '/v1/completions'
        assert data['min_tokens'] == 16 and data['ignore_eos']
        if mode == 'hang_long' and len(data['prompt']) > 128:
            time.sleep(30)
        with lock:
            count += 1
            if mode != 'no_trace':
                trace = pathlib.Path(os.environ['OSCAR_TRACE_DIR']);trace.mkdir(exist_ok=True)
                (trace/'worker-1.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records).replace('\\n', '\n'))
        response = {'choices':[{'text':'Sixteen generated tokens from the test fixture.', 'finish_reason':'length'}],
                    'usage': {'prompt_tokens':len(data['prompt'])+(1 if mode=='wrong_usage' else 0), 'completion_tokens':16}}
        body=json.dumps(response).encode()
        self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers()
        if mode == 'trickle_long' and len(data['prompt']) > 128:
            for byte in body:
                self.wfile.write(bytes([byte]));self.wfile.flush();time.sleep(.05)
        else:
            self.wfile.write(body)
        if mode == 'stop':
            threading.Timer(.5, lambda:os.kill(os.getppid(),signal.SIGINT)).start()
ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()
'''


def fake_command(tmp_path, config, mode="good"):
    program = tmp_path / "fake_server.py"
    program.write_text(FAKE_SERVER)
    records = tmp_path / "fake_telemetry.json"
    records.write_text(json.dumps(telemetry_records()))
    return [sys.executable, str(program), str(config["port"]), mode, str(records)]


def invoke(tmp_path, config_path, command, *, serve=False, resources=None):
    observations = []
    def memory(config, **kwargs):
        observations.append(config["devices"])
        return snapshot(config["devices"]) if resources is None else resources(len(observations))
    report = run_service(config_path, output=tmp_path / "report.json", log_dir=tmp_path / "logs",
        command=command, serve=serve, tokenizer_factory=lambda model: Tokenizer(), resource_reader=memory)
    assert json.loads((tmp_path / "report.json").read_text()) == report
    assert report["status"] != "running"
    return report, observations


def test_exact_prompt_uses_local_encoded_ids_and_no_special_tokens():
    assert exact_prompt(Tokenizer(), 12) == [17,32,901,7,10,17,32,901,7,10,17,32]
    with pytest.raises(ValueError):
        exact_prompt(Tokenizer(), 0)


def test_worker_progress_distinguishes_no_debug_host_dispatch_and_device_checkpoint(tmp_path):
    server = SimpleNamespace(trace_dir=tmp_path, lifecycle={"debug_sync": False})
    initial = service_probe.worker_progress(server)
    assert initial == [{"state": "no_worker_progress_yet", "debug_sync": False,
                        "device_completion": "not_established"}]
    (tmp_path / "worker-1.jsonl").write_text(json.dumps({"event": "attention_dispatched",
        "pid": 1, "rank": 0, "layer": "model.layer", "tokens": 16384}) + "\n")
    host = service_probe.worker_progress(server)[0]
    assert host["state"] == "last_host_event_only" and host["tokens"] == 16384
    assert host["device_completion"] == "not_established"
    (tmp_path / "phase-1.json").write_text(json.dumps({"pid": 1, "rank": 0,
        "state": "waiting_for_device", "phase": "fia", "tokens": 16384}))
    checkpoint = service_probe.worker_progress(server)[0]
    assert checkpoint["state"] == "waiting_for_device" and checkpoint["phase"] == "fia"


def test_compact_workers_renders_one_short_chunk_per_rank():
    now = time.time()
    progress = [
        {"state": "last_host_event_only", "event": "attention_progress", "pid": 1, "rank": 0,
         "layer": "language_model.model.layers.51.self_attn.attn", "tokens": 16384,
         "max_seq_len": 49152, "wall_time": now - 3.2, "device_completion": "not_established"},
        {"state": "last_host_event_only", "event": "graph_replay_progress", "pid": 2, "rank": 1,
         "layer": None, "tokens": None, "max_seq_len": None, "wall_time": now - 0.4,
         "device_completion": "not_established"},
        {"state": "no_worker_progress_yet", "debug_sync": False, "device_completion": "not_established"},
    ]
    rendered = service_probe.compact_workers(progress)
    first, second, third = rendered.split("; ")
    assert first.startswith("r0:attention_progress layers.51 n=16384 kv=49152 age=")
    assert second.startswith("r1:graph_replay_progress - age=")
    assert json.loads(third)["state"] == "no_worker_progress_yet"


@pytest.mark.parametrize("mode", ["hang_long", "trickle_long"])
def test_long_request_has_wall_deadline_progress_and_reaped_client(tmp_path, monkeypatch, capsys, mode):
    config, path = configured(tmp_path)
    config["service_request_timeout_seconds"] = .8
    path.write_text(json.dumps(config))
    monkeypatch.setattr(service_probe, "REQUEST_HEARTBEAT_SECONDS", .05)
    started = time.monotonic()
    report, reads = invoke(tmp_path, path, fake_command(tmp_path, config, mode))
    assert time.monotonic() - started < 5
    assert report["status"] == "failed" and "serial-16384" in report["error"]
    assert report["server"]["cleanup_complete"] and len(reads) == 2
    assert [r["prompt_tokens"] for r in report["requests"]] == [128]
    states = [json.loads(p.read_text()) for p in (tmp_path / "logs/requests").glob("*/status.json")]
    failed = next(r for r in states if r["label"] == "serial-16384")
    assert failed["status"] == "failed" and failed["client_returncode"] is not None
    assert failed["elapsed_seconds"] < 2
    terminal = capsys.readouterr()
    assert "REQUEST_START serial-16384" in terminal.out
    assert "REQUEST_WAIT serial-16384" in terminal.out
    assert "client=waiting_for_http_response" in terminal.out
    assert "REQUEST_ERROR serial-16384" in terminal.err
    assert "CLEANUP_START" in terminal.out and "CLEANUP_END" in terminal.out


def test_healthy_run_records_realistic_load_performance(tmp_path):
    config, path = configured(tmp_path)
    report, reads = invoke(tmp_path, path, fake_command(tmp_path, config))
    performance = report["performance"]
    assert performance["status"] == "measured"
    assert performance["request_count"] == 32 and performance["completed"] == 32
    assert performance["failed"] == 0 and performance["timed_out"] == 0
    assert performance["scope"].startswith("realistic concurrent load")
    assert sorted(x["prompt_tokens"] for x in report["requests"]) == [128,128,16384,16384,32768,32768,50000,50000]
    assert (tmp_path / "logs/performance.json").is_file()


def test_parse_gauges_sums_named_series_and_rejects_bad_values():
    text = ('# comment\nvllm:prompt_tokens_total{model_name="q"} 120\n'
            'vllm:prompt_tokens_total{model_name="q"} 30\n'
            'vllm:num_requests_running 3\nvllm:other 9\n')
    values = service_probe.parse_gauges(text, {"vllm:prompt_tokens_total", "vllm:num_requests_running"})
    assert values == {"vllm:prompt_tokens_total": 150.0, "vllm:num_requests_running": 3.0}
    with pytest.raises(ServiceProbeError):
        service_probe.parse_gauges("vllm:prompt_tokens_total nan\n", {"vllm:prompt_tokens_total"})


def test_performance_probe_validates_config_and_supports_disable(tmp_path):
    server = SimpleNamespace(base_url="http://127.0.0.1:1")
    bad = {"performance_request_count": 200, "performance_timeout_seconds": 10}
    with pytest.raises(ValueError):
        service_probe.run_performance(server, bad, Tokenizer(), log_dir=tmp_path,
                                      deadline=time.monotonic() + 10)
    disabled = {"performance_request_count": 0}
    assert service_probe.run_performance(server, disabled, Tokenizer(), log_dir=tmp_path,
                                         deadline=time.monotonic() + 10)["status"] == "disabled"


def test_healthy_http_with_full_evidence_executes_all_lengths_and_mixed_batch(tmp_path):
    config, path = configured(tmp_path)
    report, reads = invoke(tmp_path, path, fake_command(tmp_path, config))
    assert report["status"] == "passed"
    assert sorted(x["prompt_tokens"] for x in report["requests"]) == [128,128,16384,16384,32768,32768,50000,50000]
    assert report["telemetry"]["status"] == report["mtp"]["status"] == "passed"
    assert report["resource_release"] == "passed" and len(reads) == 2
    assert report["server"]["cleanup_complete"] and not group_exists(report["server"]["pid"])
    assert report["quality"] == "not_run"
    assert report["performance"]["status"] == "measured"  # measurement, not acceptance


@pytest.mark.parametrize("mode, diagnostic", [("no_trace", "mandatory"), ("wrong_usage", "exactly 128"), ("no_mtp", "mandatory")])
def test_http_success_cannot_hide_missing_route_mtp_or_wrong_token_count(tmp_path, mode, diagnostic):
    config, path = configured(tmp_path)
    report, reads = invoke(tmp_path, path, fake_command(tmp_path, config, mode))
    assert report["status"] == "failed" and diagnostic in report["error"]
    assert len(reads) == 2 and report["server"]["cleanup_complete"]
    assert "SERVICE_RESULT" in Path(report["server"]["log"]).read_text()


def test_startup_failure_records_exit_and_never_kills_an_unrelated_group(tmp_path):
    _, path = configured(tmp_path)
    outsider = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True)
    try:
        report, reads = invoke(tmp_path, path, [sys.executable, "-c", "raise SystemExit(7)"])
        assert report["status"] == "failed" and "rc=7" in report["error"]
        assert report["server"]["exit_code"] == 7 and len(reads) == 2
        assert outsider.poll() is None
    finally:
        assert cleanup_group(outsider, .1)


def test_hung_startup_has_finite_deadline_and_nonempty_final_log(tmp_path):
    config, path = configured(tmp_path)
    config["service_startup_timeout_seconds"] = .15
    path.write_text(json.dumps(config))
    started = time.monotonic()
    report, reads = invoke(tmp_path, path, [sys.executable, "-c", "import time;time.sleep(30)"])
    assert time.monotonic() - started < 3
    assert report["status"] == "failed" and "TimeoutError" in report["error"]
    assert report["server"]["cleanup_complete"] and len(reads) == 2


def test_interrupted_startup_still_releases_its_server_and_records_outcome(tmp_path):
    _, path = configured(tmp_path)
    command = [sys.executable, "-c", "import os,signal,time;time.sleep(.2);os.kill(os.getppid(),signal.SIGTERM);time.sleep(30)"]
    report, reads = invoke(tmp_path, path, command)
    assert report["status"] == "interrupted"
    assert report["server"]["cleanup_complete"] and len(reads) == 2


def test_formal_serve_validates_one_real_request_then_stops_owned_group_on_sigint(tmp_path):
    config, path = configured(tmp_path)
    report, reads = invoke(tmp_path, path, fake_command(tmp_path, config, "stop"), serve=True)
    assert report["status"] == "stopped" and len(report["requests"]) == 1
    assert report["server"]["cleanup_complete"] and len(reads) == 2
    assert report["telemetry"]["status"] == "passed"


def test_missing_device_selection_fails_before_any_resource_or_server_launch(tmp_path):
    config, path = configured(tmp_path)
    config["devices"] = None
    path.write_text(json.dumps(config))
    report, reads = invoke(tmp_path, path, ["must-not-run"])
    assert report["status"] == "failed" and "physical NPU" in report["error"]
    assert not reads and "pid" not in report["server"]


def test_memory_not_released_turns_successful_requests_into_failed_probe(tmp_path):
    config, path = configured(tmp_path)
    report, _ = invoke(tmp_path, path, fake_command(tmp_path, config),
        resources=lambda count: snapshot(free=2_000_000_000 if count == 1 else 1_000_000_000))
    assert report["status"] == "failed" and report["resource_release"] == "failed"
    assert report["server"]["cleanup_complete"]


def test_per_rank_layer_graph_checks_do_not_reduce_to_http_200():
    records = telemetry_records()
    expected = ["model.layers.3.self_attn.attn"]
    assert evaluate_telemetry(records, tensor_parallel_size=4, expected_layers=expected)["status"] == "passed"
    broken = [r for r in records if not (r["rank"] == 3 and r["event"] == "graph_replay_launch_return")]
    assert "rank 3 graph replay" in " ".join(evaluate_telemetry(broken, tensor_parallel_size=4, expected_layers=expected)["errors"])
    broken = [r for r in records if not (r["rank"] == 1 and "mtp" in r.get("layer", "") and r["event"] == "attention_dispatched")]
    assert evaluate_telemetry(broken, tensor_parallel_size=4, expected_layers=expected)["status"] == "failed"


def test_malformed_trace_is_a_failure_with_source_location(tmp_path):
    (tmp_path / "worker-12.jsonl").write_text('{"event":')
    with pytest.raises(ServiceProbeError, match="worker-12.jsonl:1"):
        read_telemetry(tmp_path)


def test_native_mtp_counter_parsing_records_acceptance_without_quality_claim():
    values = parse_mtp_metrics('''# HELP ignored
vllm:spec_decode_num_drafts_total{model_name="qwen3.5"} 10
vllm:spec_decode_num_draft_tokens_total{model_name="qwen3.5"} 30
vllm:spec_decode_num_accepted_tokens_total{model_name="qwen3.5"} 21
vllm:spec_decode_num_drafts_created{model_name="qwen3.5"} 90000
''')
    result = evaluate_mtp_metrics({}, values)
    assert result["status"] == "passed" and result["acceptance_rate"] == .7
    assert len(values) == 3
    assert evaluate_mtp_metrics(values, values)["status"] == "failed"
    with pytest.raises(ServiceProbeError, match="invalid native MTP"):
        parse_mtp_metrics("vllm:spec_decode_num_drafts_total NaN")


def test_release_comparison_is_per_device_and_total_memory_cannot_change():
    before, after = snapshot(), snapshot()
    after["memory"][2]["free_bytes"] -= 1025
    assert compare_release(before, after, tolerance_bytes=1024)["status"] == "failed"
    after = snapshot()
    after["memory"][0]["total_bytes"] += 1
    assert compare_release(before, after)["status"] == "failed"


def test_resource_observer_failure_is_not_a_zero_or_an_infinite_retry(tmp_path):
    def broken(*args, **kwargs):
        raise RuntimeError("driver observation failed")
    result = wait_for_release({"devices": TEST_DEVICES}, snapshot(), log_dir=tmp_path, timeout=.1, reader=broken)
    assert result["status"] == "failed" and "driver observation failed" in result["reason"]
    assert result["attempts"] == []
