"""Small OpenAI-compatible OSCAR P/D router (one P and one D endpoint).

Archive #94/#125: upstream errors remain visible; never retry on a fused engine.
Archive #157/#163: D receives the unchanged prompt and sampling options; P's
dummy sample is discarded. P computes N-1 through the external scheduler hook.
Reference: pinned Ascend disaggregated_prefill_v1 proxy request/response fields.
Uses only the standard library. SSE bytes pass through without retokenization.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import select
import socket
from threading import Event, Lock, Thread
from urllib.parse import urlsplit
import uuid


def prefill_payload(body):
    if not isinstance(body, dict):
        raise ValueError("request must be a JSON object")
    if body.get("n", 1) != 1 or body.get("best_of", 1) != 1:
        raise ValueError("OSCAR PD router currently supports n=best_of=1")
    if "kv_transfer_params" in body:
        raise ValueError("the PD router owns kv_transfer_params")
    if body.get("prompt_logprobs") is not None:
        raise ValueError("full prompt logprobs are unavailable across a KV handoff")
    prompt = body.get("prompt")
    if isinstance(prompt, list) and (not prompt or any(type(t) is not int or t < 0 for t in prompt)):
        raise ValueError("completion prompt must be one string or one token-ID list, not a batch")
    for message in body.get("messages", []):
        content = message.get("content")
        if isinstance(content, list) and any(part.get("type") != "text" for part in content):
            raise ValueError("OSCAR PD currently supports text chat content only")
    result = deepcopy(body)
    result.update(stream=False, max_tokens=1, min_tokens=1, ignore_eos=True,
                  stop=[], stop_token_ids=[], echo=False,
                  kv_transfer_params={"do_remote_decode": True, "do_remote_prefill": False})
    if "max_completion_tokens" in result:
        result["max_completion_tokens"] = 1
    # These constrain generated output only; preserve tools/messages/template
    # arguments, which can change tokenization and must agree on P and D.
    for key in ("stream_options", "structured_outputs", "response_format", "logprobs",
                "top_logprobs", "allowed_token_ids"):
        result.pop(key, None)
    return result


def decode_payload(body, prefill_response):
    params = prefill_response.get("kv_transfer_params")
    if (not isinstance(params, dict) or params.get("do_remote_prefill") is not True
            or params.get("do_remote_decode") is not False
            or not isinstance(params.get("oscar_pd"), dict)
            or params["oscar_pd"].get("schema") != "oscar-pd-v1"):
        raise ValueError("prefill endpoint did not return an OSCAR PD handoff")
    return dict(body, kv_transfer_params=params)


def endpoint(value):
    parsed = urlsplit(value)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in ("", "/")):
        raise ValueError("endpoint must be an http(s) origin without credentials or a path")
    return parsed


class Router(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, prefill, decode, timeout=1800):
        self.prefill, self.decode = endpoint(prefill), endpoint(decode)
        if self.prefill == self.decode:
            raise ValueError("P and D must be separate service endpoints")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive")
        self.upstream_timeout = timeout
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def reply(self, status, body, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def error(self, status, message):
        self.reply(status, json.dumps({"error": {"message": message, "type": "oscar_pd_error"}}).encode())

    def upstream(self, target, path, body=None):
        cls = http.client.HTTPSConnection if target.scheme == "https" else http.client.HTTPConnection
        conn = cls(target.hostname, target.port, timeout=self.server.upstream_timeout)
        conn.connect()
        with self._lock:
            if self._done.is_set():
                conn.close()
                raise ConnectionAbortedError("client disconnected")
            self._connection = conn
            self._upstream_socket = conn.sock
        headers = {"Content-Type": "application/json", "X-Request-Id": "oscar-pd-" + uuid.uuid4().hex}
        if self.headers.get("Authorization"):
            headers["Authorization"] = self.headers["Authorization"]
        conn.request("POST" if body is not None else "GET", path,
                     body=json.dumps(body).encode() if body is not None else None, headers=headers)
        return conn, conn.getresponse()

    def watch_disconnect(self):
        while not self._done.wait(0.1):
            try:
                readable, _, _ = select.select([self.connection], [], [], 0)
                if readable and not self.connection.recv(1, socket.MSG_PEEK):
                    self._done.set()
                    with self._lock:
                        if self._upstream_socket is not None:
                            self._upstream_socket.shutdown(socket.SHUT_RDWR)
                    return
            except OSError:
                return

    def do_POST(self):
        if self.path not in ("/v1/completions", "/v1/chat/completions"):
            self.error(404, "unsupported PD endpoint")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16 * 1024 * 1024 or self.headers.get("Transfer-Encoding"):
                raise ValueError("request requires Content-Length between 1 and 16 MiB")
            self.connection.settimeout(self.server.upstream_timeout)
            body = json.loads(self.rfile.read(length))
            prefill = prefill_payload(body)
        except (ValueError, TypeError, AttributeError) as exc:
            self.error(400, str(exc))
            return
        self._done, self._lock, self._connection, self._upstream_socket = Event(), Lock(), None, None
        watcher = Thread(target=self.watch_disconnect, daemon=True)
        watcher.start()
        headers_sent = False
        conn = None
        try:
            conn, response = self.upstream(self.server.prefill, self.path, prefill)
            data = response.read()
            if response.status != 200:
                self.reply(response.status, data)
                return
            decoded = decode_payload(body, json.loads(data))
            conn.close()
            conn, response = self.upstream(self.server.decode, self.path, decoded)
            self.send_response(response.status)
            self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
            self.send_header("Connection", "close")
            self.end_headers()
            headers_sent = True
            while data := response.read1(65536):
                self.wfile.write(data)
                self.wfile.flush()
        except (OSError, http.client.HTTPException, ValueError) as exc:
            self.log_error("PD request failed: %s", exc)
            if not headers_sent and not self._done.is_set():
                self.error(502, str(exc))
        finally:
            self._done.set()
            if conn is not None:
                conn.close()
            if self._connection is not None:
                self._connection.close()
            watcher.join(timeout=1)
            self.close_connection = True

    def do_GET(self):
        if self.path not in ("/health", "/v1/models"):
            self.error(404, "unsupported PD endpoint")
            return
        self._done, self._lock, self._connection, self._upstream_socket = Event(), Lock(), None, None
        try:
            targets = (self.server.prefill, self.server.decode) if self.path == "/health" else (self.server.decode,)
            for target in targets:
                conn, response = self.upstream(target, self.path)
                data, status = response.read(), response.status
                conn.close()
                if status != 200:
                    break
            self.reply(status, data)
        except (OSError, http.client.HTTPException) as exc:
            self.error(502, str(exc))
        finally:
            if self._connection is not None:
                self._connection.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8989)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args(argv)
    with Router((args.host, args.port), args.prefill_url, args.decode_url, args.timeout) as server:
        print(f"[oscar] PD_PROXY http://{args.host}:{server.server_port} P={args.prefill_url} D={args.decode_url}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
