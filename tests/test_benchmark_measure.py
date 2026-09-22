"""Local HTTP fixtures exercise the measurement client, not NPU acceptance.

Archive #70-73/#94-95: distinguish receive/device timing, freeze repeats,
preserve partial/error evidence and never turn missing profiler data green.
"""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time

import pytest

from benchmarks.compare import compare_runs, read_json, DEFAULT_POLICY, SOFTWARE_FIELDS
from benchmarks.measure import local_url, measure, stream_request


ROOT = Path(__file__).resolve().parents[1]


class Tokenizer:
    def encode(self, text, add_special_tokens):
        assert text and not add_special_tokens
        return [21, 33, 45]


@contextmanager
def server(mode="good"):
    state = {"requests": [], "count": 0}
    lock = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            if self.path != "/metrics":
                self.send_error(404)
                return
            with lock:
                count = state["count"]
            raw = (f'vllm:spec_decode_num_drafts_total{{model_name="qwen3.5"}} {count}\n'
                   f'vllm:spec_decode_num_draft_tokens_total{{model_name="qwen3.5"}} {count * 3}\n'
                   f'vllm:spec_decode_num_accepted_tokens_total{{model_name="qwen3.5"}} {count * 2}\n').encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                state["requests"].append(data)
                state["count"] += 1
            if mode == "http_error":
                self.send_error(503, "fixture unavailable")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            def event(value):
                raw = value if isinstance(value, bytes) else json.dumps(value).encode()
                self.wfile.write(b"data: " + raw + b"\r\n\r\n")
                self.wfile.flush()
            try:
                if mode == "timeout":
                    time.sleep(.2)
                if mode == "malformed":
                    event(b"{bad json}")
                    return
                event({"choices": [{"index": 0, "text": "", "token_ids": [], "prompt_token_ids": data["prompt"]}]})
                count = data["max_tokens"]
                groups = [[100], list(range(101, 100 + count - 1)), [100 + count - 1]]
                for index, group in enumerate(groups):
                    if not group:
                        continue
                    time.sleep(.002)
                    choice = {"index": 0, "text": "test", "token_ids": group,
                              "finish_reason": "length" if index == len(groups)-1 else None}
                    if mode == "no_token_ids":
                        choice.pop("token_ids")
                    event({"choices": [choice]})
                if mode != "no_usage":
                    event({"choices": [], "usage": {"prompt_tokens": len(data["prompt"]) + (1 if mode == "wrong_usage" else 0),
                                                     "completion_tokens": count}})
                if mode != "no_done":
                    event(b"[DONE]")
            except (BrokenPipeError, ConnectionResetError):
                return
    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: http.serve_forever(poll_interval=.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}", state
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=1)


def payload():
    return {"model": "qwen3.5", "prompt": [21, 33, 45], "max_tokens": 4,
            "stream": True, "return_token_ids": True}


def setup_config(tmp_path):
    config = read_json(ROOT / "configs/target.json")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"num_hidden_layers": 4,
        "full_attention_interval": 4, "head_dim": 64}))
    config.update(model=str(model), devices=[40, 41, 42, 43])
    path = tmp_path / "target.json"
    path.write_text(json.dumps(config))
    identity = tmp_path / "identity.json"
    identity.write_text(json.dumps({"software": {key: "synthetic-http-fixture" for key in SOFTWARE_FIELDS},
                                    "npu_model": "synthetic, not a hardware observation"}))
    return path, identity


def test_stream_measures_first_actual_token_and_preserves_multi_token_burst():
    with server() as (url, _):
        result = stream_request(url, payload(), timeout=1, request_id="test")
    assert result["status"] == "completed"
    assert result["usage"] == {"prompt_tokens": 3, "completion_tokens": 4}
    assert 0 < result["ttft_ms"] < result["e2e_ms"]
    assert [b["token_count"] for b in result["bursts"]] == [1, 2, 1]
    assert result["token_arrivals_ms"][1] == result["token_arrivals_ms"][2]
    assert result["itl_ms"][1] == 0  # Actual shared receipt, never invented within-burst spacing.
    assert result["generated_token_ids"] == [100, 101, 102, 103]


@pytest.mark.parametrize("mode, fragment", [
    ("http_error", "HTTP 503"), ("malformed", "JSONDecodeError"),
    ("no_token_ids", "return_token_ids"), ("no_usage", "usage/token IDs"),
    ("wrong_usage", "usage/token IDs"), ("no_done", "[DONE]"),
])
def test_stream_errors_cannot_be_counted_as_completed(mode, fragment):
    with server(mode) as (url, _):
        result = stream_request(url, payload(), timeout=1, request_id="fault")
    assert result["status"] == "failed" and fragment in result["error"]
    assert result["e2e_ms"] > 0


def test_stream_timeout_is_bounded_and_preserves_partial_results():
    with server("timeout") as (url, _):
        start = time.perf_counter()
        result = stream_request(url, payload(), timeout=.03, request_id="timeout")
        assert time.perf_counter() - start < .15
    assert result["status"] == "timeout" and result["token_arrivals_ms"] == []


def test_two_warmups_five_repeats_fingerprints_and_missing_device_evidence(tmp_path):
    config, identity = setup_config(tmp_path)
    reports = []
    with server() as (url, state):
        for variant in ("native", "oscar"):
            destination = tmp_path / variant / "run.json"
            result = measure(config, variant=variant, output=destination, url=url, lengths=[8], concurrency=[2],
                output_tokens=4, timeout=1, identity=identity, tokenizer_factory=lambda model: Tokenizer())
            assert read_json(destination) == result
            assert result["status"] == "needs_evidence" and result["client_measurements_complete"]
            assert result["measurement_type"] == "http_client" and result["cases"] == []
            case = result["http_cases"][0]
            assert case["q_len"] is None
            assert len(case["warmup"]) == 2 and len(case["samples"]) == 5
            assert [sample["repeat"] for sample in case["samples"]] == list(range(5))
            assert all(sample["completed_requests"] == 2 and sample["phase_ms"] == {} for sample in case["samples"])
            assert all(sample["throughput_tps"]["generation"] > 0 for sample in case["samples"])
            assert case["native_metrics_delta"]["mtp_counter_delta"]["vllm:spec_decode_num_drafts"] == 14
            assert not result["matrix"]["full_acceptance_matrix_complete"]
            reports.append(result)
        assert len(state["requests"]) == 28
        assert len({request["cache_salt"] for request in state["requests"]}) == 28
        assert all(len(request["prompt"]) == 8 and request["min_tokens"] == 4 for request in state["requests"])
    assert reports[0]["pair"] == reports[1]["pair"]
    assert reports[0]["http_cases"][0]["input_sha256"] == reports[1]["http_cases"][0]["input_sha256"]
    compared = compare_runs(*reports, read_json(DEFAULT_POLICY))
    assert compared["status"] == "not_run"  # HTTP timing never manufactures hardware acceptance.


def test_request_failures_are_retained_across_all_five_repeats(tmp_path):
    config, identity = setup_config(tmp_path)
    with server("wrong_usage") as (url, _):
        result = measure(config, variant="oscar", output=tmp_path / "bad.json", url=url, lengths=[8], concurrency=[1],
            output_tokens=4, timeout=1, identity=identity, tokenizer_factory=lambda model: Tokenizer())
    case = result["http_cases"][0]
    assert result["status"] == "failed" and not result["client_measurements_complete"]
    assert len(case["samples"]) == 5
    assert all(sample["failed_requests"] == 1 and sample["completed_requests"] == 0 for sample in case["samples"])


def test_context_limit_is_recorded_without_silently_shortening_input(tmp_path):
    config, _ = setup_config(tmp_path)
    with server() as (url, state):
        result = measure(config, variant="native", output=tmp_path / "boundary.json", url=url,
            lengths=[262144], concurrency=[1], output_tokens=32, timeout=1, tokenizer_factory=lambda model: Tokenizer())
    case = result["http_cases"][0]
    assert case["input_tokens"] == 262144 and case["status"] == "not_run"
    assert case["samples"] == [] and not state["requests"]
    assert not result["client_measurements_complete"]


def test_default_lengths_are_explicit_standard_scope_and_keep_full_policy_matrix(tmp_path):
    config, identity = setup_config(tmp_path)
    with server() as (url, _):
        result = measure(config, variant="native", output=tmp_path / "standard.json", url=url,
            concurrency=[1], output_tokens=4, timeout=2, identity=identity, tokenizer_factory=lambda model: Tokenizer())
    assert result["default_workload_scope"] == "required_standard_inputs"
    assert [case["input_tokens"] for case in result["http_cases"]] == [16384, 32768, 50000]
    assert result["matrix"]["required_input_lengths"] == [16384,32768,50000,100000,262144]
    assert not result["matrix"]["full_acceptance_matrix_complete"]
    assert result["client_measurements_complete"]


def test_missing_local_model_is_a_persisted_failure_not_a_download(tmp_path):
    config, _ = setup_config(tmp_path)
    data = read_json(config)
    data["model"] = str(tmp_path / "does-not-exist")
    config.write_text(json.dumps(data))
    result = measure(config, variant="oscar", output=tmp_path / "missing" / "run.json", lengths=[8], concurrency=[1])
    assert result["status"] == "failed" and "FileNotFoundError" in result["error"]
    assert read_json(tmp_path / "missing" / "run.json")["status"] == "failed"


@pytest.mark.parametrize("url", ["http://example.com", "ftp://localhost", "http://localhost/v1", "http://user:password@localhost"])
def test_client_does_not_redirect_workload_to_an_unrequested_remote_server(url):
    with pytest.raises(ValueError):
        local_url(url)
