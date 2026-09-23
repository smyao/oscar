"""CPU HTTP fixtures verify diagnostic bookkeeping, not NPU performance.

Archive #70-#73/#130-#139: external and synthetic workloads stay distinct;
an observer must not invent request lengths or device timing from /metrics.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time

from benchmarks.passive import observe


def test_passive_observer_captures_only_external_metrics_window(tmp_path):
    state = {"running": 0, "waiting": 0, "prompt": 0, "generation": 0, "posts": 0}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            assert self.path == "/metrics"
            with lock:
                snapshot = dict(state)
            body = (f"vllm:num_requests_running {snapshot['running']}\n"
                    f"vllm:num_requests_waiting {snapshot['waiting']}\n"
                    f"vllm:prompt_tokens_total {snapshot['prompt']}\n"
                    f"vllm:generation_tokens_total {snapshot['generation']}\n").encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            with lock:
                state["posts"] += 1
            self.send_error(500)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    stop, ready = threading.Event(), threading.Event()
    result = {}
    output = tmp_path / "external.json"
    url = f"http://127.0.0.1:{server.server_port}"
    thread = threading.Thread(target=lambda: result.update(report=observe(url,
        output=output, stop=stop, ready=ready, variant="oscar",
        sample_interval=.02, idle_seconds=.06, max_seconds=2)), daemon=True)
    try:
        thread.start()
        assert ready.wait(1)
        with lock:
            state["running"] = 4
            state["waiting"] = 28
        time.sleep(.1)
        with lock:
            state.update(running=0, waiting=0, prompt=800000, generation=512)
        time.sleep(.2)
        stop.set()
        thread.join(2)
        assert not thread.is_alive()
        report = result["report"]
        assert report["status"] == "observed"
        assert len(report["windows"]) == 1
        window = report["windows"][0]
        assert window["status"] == "observed"
        assert window["peak_observed_inflight_lower_bound"] == 32
        assert window["counter_delta"]["vllm:prompt_tokens_total"] == 800000
        assert window["throughput_tps"]["prompt"] > 0
        assert window["prompt_length_distribution"] == "unobserved_without_client_artifact"
        assert state["posts"] == 0
        assert json.loads(output.read_text()) == report
        assert Path(window["metrics_before"]["path"]).exists()
        assert Path(window["metrics_after"]["path"]).exists()
    finally:
        stop.set()
        server.shutdown()
        server.server_close()
        server_thread.join(1)
