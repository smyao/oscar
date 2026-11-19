"""#94/#125/#163: real localhost HTTP routing, stream bytes and error propagation.

Both upstreams are synthetic HTTP servers; no model, NPU or performance claim.
"""
from contextlib import contextmanager
from copy import deepcopy
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Event, Thread

import pytest

from tools.pd_proxy import Router, prefill_payload, decode_payload, endpoint


TICKET = {"do_remote_prefill": True, "do_remote_decode": False,
          "oscar_pd": {"schema": "oscar-pd-v1", "nonce": "test"}}


@contextmanager
def running(server):
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class Upstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, body))
        self.send_response(self.server.status)
        self.send_header("Content-Type", self.server.content_type)
        self.send_header("Content-Length", str(len(self.server.payload)))
        self.end_headers()
        self.wfile.write(self.server.payload)


def upstream(payload, status=200, content_type="application/json"):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    server.requests, server.payload, server.status, server.content_type = [], payload, status, content_type
    return server


@pytest.mark.parametrize("path,stream", [("/v1/completions", False), ("/v1/chat/completions", True)])
def test_real_http_handoff_preserves_decode_body_and_sse(path, stream):
    p = upstream(json.dumps({"choices": [{"text": "discard me"}], "kv_transfer_params": TICKET}).encode())
    expected = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\ndata: [DONE]\n\n' if stream else b'{"choices":[{"text":"hello"}]}'
    d = upstream(expected, content_type="text/event-stream" if stream else "application/json")
    body = {"model": "qwen3.8", "max_tokens": 16, "temperature": 0, "seed": 7, "stream": stream}
    body.update({"messages": [{"role": "user", "content": "hello"}]} if stream else {"prompt": [1, 2, 3]})
    with running(p) as purl, running(d) as durl, running(Router(("127.0.0.1", 0), purl, durl, 3)) as url:
        client = http.client.HTTPConnection("127.0.0.1", endpoint(url).port, timeout=5)
        client.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        response = client.getresponse()
        assert response.status == 200 and response.read() == expected
        client.close()
    assert p.requests == [(path, prefill_payload(body))]
    assert d.requests == [(path, dict(body, kv_transfer_params=TICKET))]


@pytest.mark.parametrize("status,payload,expected", [(503, b'{"error":"P failed"}', 503), (200, b'{"choices":[]}', 502)])
def test_prefill_failure_never_runs_decode(status, payload, expected):
    p, d = upstream(payload, status), upstream(b'{}')
    with running(p) as purl, running(d) as durl, running(Router(("127.0.0.1", 0), purl, durl, 3)) as url:
        client = http.client.HTTPConnection("127.0.0.1", endpoint(url).port, timeout=5)
        client.request("POST", "/v1/completions", json.dumps({"prompt": "hello", "model": "qwen3.8"}))
        response = client.getresponse()
        assert response.status == expected
        assert response.read()
        client.close()
    assert not d.requests


def test_prefill_sampling_is_independent_without_changing_prompt_or_decode_options():
    body = {"messages": [{"role": "user", "content": "hi"}], "tools": [{"name": "x"}],
            "stop": ["."], "max_completion_tokens": 100, "stream": True, "stream_options": {"include_usage": True},
            "response_format": {"type": "json_object"}}
    original = deepcopy(body)
    p = prefill_payload(body)
    assert p["tools"] == body["tools"] and p["messages"] == body["messages"]
    assert p["max_completion_tokens"] == p["max_tokens"] == 1
    assert p["stop"] == [] and p["ignore_eos"] is True and not p["stream"]
    assert "response_format" not in p and "stream_options" not in p
    assert decode_payload(body, {"kv_transfer_params": TICKET}) == dict(body, kv_transfer_params=TICKET)
    assert body == original


@pytest.mark.parametrize("body", [{"n": 2}, {"prompt": [[1, 2]]}, {"kv_transfer_params": {}},
                                  {"prompt_logprobs": 0}, {"messages": [{"content": [{"type": "image_url"}]}]}])
def test_unsupported_contracts_fail_before_upstream(body):
    with pytest.raises(ValueError):
        prefill_payload(body)


def test_disconnect_during_prefill_closes_upstream_and_never_starts_decode():
    entered, cancelled = Event(), Event()
    class WaitingPrefill(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.connection.settimeout(5)
            entered.set()
            if self.rfile.read(1) == b"":
                cancelled.set()
    p = ThreadingHTTPServer(("127.0.0.1", 0), WaitingPrefill)
    d = upstream(b'{}')
    with running(p) as purl, running(d) as durl, running(Router(("127.0.0.1", 0), purl, durl, 3)) as url:
        client = http.client.HTTPConnection("127.0.0.1", endpoint(url).port, timeout=5)
        client.request("POST", "/v1/completions", json.dumps({"prompt": "hello", "model": "qwen3.8"}))
        assert entered.wait(3)
        client.close()
        assert cancelled.wait(3)
    assert not d.requests
