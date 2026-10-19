# Archive #70-73/#133/#143: actual local SSE bookkeeping only, not NPU timing.
# Single-token completions have no token-to-token interval; absence is not 0.
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest

from benchmarks import mixed


@contextmanager
def sse_service():
    requests = []
    lock = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_GET(self):
            assert self.path == "/metrics"
            body = b"vllm:num_requests_running 1\nvllm:num_requests_waiting 0\n"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            assert self.path == "/v1/completions"
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                requests.append(payload)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            ids = list(range(100, 100 + payload["max_tokens"]))
            events = [
                {"choices": [{"index": 0, "text": "test", "token_ids": ids,
                              "prompt_token_ids": payload["prompt"], "finish_reason": "length"}]},
                {"choices": [], "usage": {"prompt_tokens": len(payload["prompt"]),
                                           "completion_tokens": len(ids)}}]
            body = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
            self.wfile.write(body + b"data: [DONE]\n\n")
            self.wfile.flush()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


@pytest.mark.parametrize("count", [1, 3])
def test_real_sse_one_token_batch_has_null_tpot_and_zero_samples(count):
    rows = [{"id": f"one-token-{i}", "token_ids": [21, 33, i]} for i in range(count)]
    with sse_service() as (url, requests):
        result = mixed.run_batch(url, rows, model="qwen3.8", output_tokens=1,
            timeout=2, repeat="one-token", sample_interval=.01)
    assert result["status"] == "completed" and result["completed_requests"] == count
    assert result["failed_requests"] == result["timeouts"] == 0
    assert result["latency_ms"]["tpot_ms"] == {"p50": None, "p95": None, "sample_count": 0}
    assert result["latency_ms"]["ttft_ms"]["sample_count"] == count
    assert result["latency_ms"]["e2e_ms"]["sample_count"] == count
    assert all(row["tpot_ms"] is None and row["done_received_ms"] is not None for row in result["requests"])
    assert all(row["usage"]["completion_tokens"] == len(row["generated_token_ids"]) == 1
               for row in result["requests"])
    assert result["throughput_tps"]["generation"] > 0
    assert len(requests) == count and len({p["cache_salt"] for p in requests}) == count
    json.dumps(result, allow_nan=False)


def test_real_batch_mixed_missing_latencies_uses_only_available_samples(monkeypatch):
    actual_stream = mixed.stream_request
    def partial_observation(*args, **kwargs):
        result = actual_stream(*args, **kwargs)
        assert result["status"] == "completed"
        # Keep real HTTP/DONE/usage completion, remove selected optional
        # observations to exercise each statistic's own availability mask.
        if kwargs["request_id"] == "row-0":
            result["tpot_ms"] = None
            result.pop("ttft_ms")
        elif kwargs["request_id"] == "row-1":
            result.pop("tpot_ms")
            result["e2e_ms"] = None
        return result
    monkeypatch.setattr(mixed, "stream_request", partial_observation)
    rows = [{"id": f"row-{i}", "token_ids": [11, i]} for i in range(3)]
    with sse_service() as (url, _):
        result = mixed.run_batch(url, rows, model="qwen3.8", output_tokens=2,
            timeout=2, repeat="missing-latency", sample_interval=.01)
    assert result["status"] == "completed"
    assert result["latency_ms"]["ttft_ms"]["sample_count"] == 2
    assert result["latency_ms"]["e2e_ms"]["sample_count"] == 2
    # Two tokens arrived in the same actual SSE event: 0 is measured here,
    # unlike the undefined one-token TPOT above.
    assert result["latency_ms"]["tpot_ms"] == {"p50": 0.0, "p95": 0.0, "sample_count": 1}
    assert result["completed_requests"] == 3 and result["throughput_tps"]["generation"] > 0
