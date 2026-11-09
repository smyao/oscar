"""Bounded scheduler-to-worker RPC for checksum-verified commit manifests.

Archive #34/#36/#37--49/#127/#128/#157/#163/#171: a checkpoint is published
only after all TP/group payloads complete. Immutable payload generations are
selected by a checksummed manifest; a torn concurrent manifest is an error,
never a mixed-generation cache hit. No SDK atomic overwrite/CAS is assumed.

Official Ascend/memcache doc/memcache_python_api.md defines host put/get;
init_bm=False clients cannot read/write data. The scheduler therefore routes
these operations to the rank-zero worker's initialized store over local IPC.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
from threading import Event, Lock, Thread
import time
import traceback
from typing import Any, Mapping

from .transport import MAX_HOST_OBJECT_BYTES, MissingObjectsError, StrictMemcacheTransport, TransferError


MANIFEST_SCHEMA = "oscar-kv-committed-generation-v1"
MAX_RPC_BYTES = 2 * MAX_HOST_OBJECT_BYTES


class ManifestIntegrityError(TransferError):
    pass


@dataclass(frozen=True)
class _SwapScope:
    namespace: str
    session: str
    world_size: int
    geometry: tuple[dict[str, Any], ...]
    max_tokens: int


_SWAP_FIELDS = {"request_id", "generation", "version", "epoch", "committed_tokens", "base_key"}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _decode_json(payload: bytes) -> Any:
    return json.loads(payload, object_pairs_hook=_unique_pairs, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON constant {value}")))


def encode_manifest(manifest: Mapping[str, Any]) -> bytes:
    if not isinstance(manifest, Mapping) or any(not isinstance(key, str) for key in manifest):
        raise ValueError("manifest must be a JSON object with string keys")
    body = dict(manifest)
    body_bytes = _canonical(body)
    payload = _canonical({"schema": MANIFEST_SCHEMA, "body": body, "checksum": hashlib.sha256(body_bytes).hexdigest()})
    if len(payload) > MAX_HOST_OBJECT_BYTES:
        raise ValueError("manifest exceeds the bounded metadata size")
    return payload


def decode_manifest(payload: bytes) -> dict[str, Any]:
    try:
        if not isinstance(payload, bytes) or not 0 < len(payload) <= MAX_HOST_OBJECT_BYTES:
            raise ValueError("invalid manifest byte length")
        record = _decode_json(payload)
        if (not isinstance(record, dict) or set(record) != {"schema", "body", "checksum"}
                or record["schema"] != MANIFEST_SCHEMA or not isinstance(record["body"], dict)
                or not isinstance(record["checksum"], str)):
            raise ValueError("manifest schema differs")
        actual = hashlib.sha256(_canonical(record["body"])).hexdigest()
        if not hmac.compare_digest(record["checksum"], actual):
            raise ValueError("manifest checksum differs")
        return record["body"]
    except (ValueError, TypeError, UnicodeError, OverflowError) as exc:
        raise ManifestIntegrityError(f"invalid/torn checkpoint manifest: {exc}") from exc


def rpc_endpoint(vllm_config) -> str:
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import get_zmq_rpc_path_lookup
    engine_id = vllm_config.kv_transfer_config.engine_id
    if not isinstance(engine_id, str) or not engine_id:
        raise ValueError("offload RPC requires the native shared engine_id")
    engine = hashlib.sha256(engine_id.encode()).hexdigest()[:24]
    return get_zmq_rpc_path_lookup(vllm_config) + "-oscar-" + engine


def _timeout_ms(value: float) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("metadata timeout_seconds must be finite and positive")
    return max(1, math.ceil(value * 1000))


def _key(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value.encode()) >= 256:
        raise ValueError("invalid metadata object key")
    return value


def _unbase64(value: Any) -> bytes:
    if not isinstance(value, str) or len(value) > MAX_RPC_BYTES:
        raise ManifestIntegrityError("invalid metadata payload encoding")
    try:
        return base64.b64decode(value, validate=True)
    except ValueError as exc:
        raise ManifestIntegrityError("invalid base64 metadata payload") from exc


class MetadataServer:
    """Single rank-zero worker proxy; all SDK operations remain serialized."""

    def __init__(self, transport: StrictMemcacheTransport, endpoint: str, *, timeout_seconds: float = 60.0):
        if transport.scheduler:
            raise ValueError("metadata server needs an init_bm=True worker transport")
        if not isinstance(endpoint, str) or not endpoint.startswith("ipc://"):
            raise ValueError("metadata RPC must use a local IPC endpoint")
        self.transport, self.endpoint = transport, endpoint
        self.timeout_seconds = _timeout_ms(timeout_seconds) / 1000
        self._stop, self._ready = Event(), Event()
        self._thread: Thread | None = None
        self._startup_error: Exception | None = None
        self._swap_scope: _SwapScope | None = None
        self._scope_lock = Lock()

    def configure_swap_scope(self, namespace: str, session: str, world_size: int,
                             geometry, *, max_tokens: int | None = None) -> None:
        """Bind deletion authority from the native worker path, never RPC.

        The real scheduler session arrives in connector execution metadata.
        Once bound, neither another session nor another physical layout can
        expand this server's deletion scope.
        """
        max_tokens = (1 << 20) if max_tokens is None else max_tokens
        if (not isinstance(namespace, str) or not namespace
                or not isinstance(session, str) or not session or "/" in session
                or len(session) > 128 or type(world_size) is not int or world_size <= 0
                or type(max_tokens) is not int or max_tokens <= 0 or max_tokens > (1 << 24)):
            raise ValueError("invalid private swap deletion scope")
        layers = tuple(_decode_json(_canonical(tuple(geometry))))
        if not layers:
            raise ValueError("swap deletion scope requires physical cache geometry")
        # Validate geometry and bound future key derivation without generating
        # all possible keys up front.
        count = 0
        for layer in layers:
            family = layer.get("cache_family")
            if family == "oscar_full":
                block = layer.get("storage_block_size")
                if type(block) is not int or block <= 0:
                    raise ValueError("invalid FULL block size in swap deletion scope")
                count += (max_tokens + block - 1) // block
            elif family == "gdn":
                components = layer.get("components")
                if not isinstance(components, list) or not components:
                    raise ValueError("invalid GDN components in swap deletion scope")
                count += len(components)
            else:
                raise ValueError("unsupported cache family in swap deletion scope")
        if count * world_size > (1 << 20):
            raise ValueError("swap deletion scope exceeds the bounded object count")
        scope = _SwapScope(namespace, session, world_size, layers, max_tokens)
        with self._scope_lock:
            if self._swap_scope is not None and self._swap_scope != scope:
                raise TransferError("private swap deletion scope cannot be rebound")
            self._swap_scope = scope

    def remove_swap(self, fields: Any) -> tuple[int, int]:
        """Delete only keys derived from this engine's completed private swap.

        The scheduler owns completion barriers: it issues this only after all
        save acknowledgements for an abandoned snapshot, or after all ranks'
        restore acknowledgements. The RPC cannot supply an arbitrary key list.
        Duplicate cleanup is safe because the versioned private key is never
        reused; only the SDK's exact object-not-found code is tolerated.
        """
        from .plans import PayloadPlan, expected_keys, snapshot_key

        with self._scope_lock:
            scope = self._swap_scope
        if scope is None:
            raise TransferError("private swap deletion scope is not configured")
        if not isinstance(fields, dict) or set(fields) != _SWAP_FIELDS:
            raise ValueError("private swap deletion requires only a structured plan identity")
        request_id = fields["request_id"]
        if not isinstance(request_id, str) or not request_id or len(request_id.encode()) > 8192:
            raise ValueError("invalid private swap request identity")
        for name, minimum in (("generation", 1), ("version", 1), ("epoch", 0), ("committed_tokens", 1)):
            if type(fields[name]) is not int or fields[name] < minimum:
                raise ValueError(f"invalid private swap {name}")
        if fields["committed_tokens"] > scope.max_tokens:
            raise ValueError("private swap token count exceeds the configured model bound")
        if not isinstance(fields["base_key"], str):
            raise ValueError("invalid private swap base key")
        plan = PayloadPlan(block_ids=(), **fields)
        if plan.base_key != snapshot_key(scope.session, plan, plan.committed_tokens):
            raise TransferError("private swap plan is outside the current session/identity")
        keys = expected_keys(scope.namespace, scope.world_size, plan, scope.geometry)
        removed = sum(self.transport.host_remove_batch(keys, missing_ok=True))
        return removed, len(keys) - removed

    def _manifest_keys(self, key: str, body: dict[str, Any]) -> tuple[str, ...]:
        values = body.get("keys")
        if (body.get("index_key") != key or not isinstance(body.get("namespace"), str)
                or not body["namespace"] or type(body.get("committed_tokens")) is not int
                or body["committed_tokens"] <= 0 or not isinstance(body.get("base_key"), str)
                or not body["base_key"].startswith("prefix/data/")
                or not isinstance(values, list) or not values):
            raise ManifestIntegrityError("prefix manifest has an incomplete publication identity")
        keys = tuple(_key(value) for value in values)
        if len(set(keys)) != len(keys) or key in keys:
            raise ManifestIntegrityError("prefix manifest has duplicate/self-referential payload keys")
        return keys

    def publish(self, key: str, payload: bytes) -> None:
        """Retain a complete first writer; replace a stale index once at most.

        remove+put is not CAS. A concurrent replacement can produce a short
        index miss, but never mutates the immutable payloads that a previously
        verified reader selected. Every winner is independently checked.
        """
        incoming = decode_manifest(payload)
        keys = self._manifest_keys(key, incoming)
        if not all(self.transport.exists(keys)):
            raise TransferError("cannot publish a prefix before every payload object exists")
        for attempt in range(2):
            if self.transport.host_put(key, payload, allow_duplicate=True):
                return
            try:
                current = decode_manifest(self.transport.host_get(key))
            except MissingObjectsError:
                current = None
            if current is not None:
                current_keys = self._manifest_keys(key, current)
                if any(current[name] != incoming[name] for name in ("namespace", "index_key", "committed_tokens")):
                    raise ManifestIntegrityError("existing prefix manifest belongs to another identity")
                if all(self.transport.exists(current_keys)):
                    return  # Immutable SDK key: retain the complete first writer.
            if attempt:
                raise TransferError("prefix publication lost two bounded races to an incomplete manifest")
            if current is not None:
                self.transport.host_remove(key, missing_ok=True)

    def dispatch(self, request: Any) -> dict[str, Any]:
        """Validate both RPC and manifest data before invoking host SDK I/O."""
        try:
            if not isinstance(request, dict):
                raise ValueError("metadata RPC request must be an object")
            operation = request.get("op")
            if operation == "remove_swap" and set(request) == {"op", "plan"}:
                removed, missing = self.remove_swap(request["plan"])
                return {"status": "ok", "removed": removed, "missing": missing}
            key = _key(request.get("key"))
            if operation == "get" and set(request) == {"op", "key"}:
                payload = self.transport.host_get(key)
                decode_manifest(payload)
                return {"status": "ok", "payload": base64.b64encode(payload).decode("ascii")}
            if operation == "put" and set(request) == {"op", "key", "payload"}:
                payload = _unbase64(request["payload"])
                decode_manifest(payload)
                self.publish(key, payload)
                return {"status": "ok"}
            raise ValueError("unsupported metadata RPC operation or fields")
        except MissingObjectsError:
            return {"status": "missing"}
        except Exception as exc:
            traceback.print_exc()
            return {"status": "error", "error_type": type(exc).__name__, "error": str(exc)}

    def start(self) -> MetadataServer:
        if self._thread is not None:
            raise RuntimeError("metadata server has already been started")
        self._thread = Thread(target=self._serve, name="oscar-manifest-rpc", daemon=True)
        self._thread.start()
        if not self._ready.wait(self.timeout_seconds):
            self._stop.set()
            raise TransferError("metadata RPC server startup exceeded its deadline")
        if self._startup_error is not None:
            raise TransferError(f"metadata RPC server startup failed: {self._startup_error}") from self._startup_error
        return self

    def _serve(self) -> None:
        context = socket = None
        try:
            import zmq
            if self.transport.backend is not None:
                self.transport.backend.set_device()
            context = zmq.Context()
            socket = context.socket(zmq.REP)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.SNDTIMEO, _timeout_ms(self.timeout_seconds))
            socket.setsockopt(zmq.MAXMSGSIZE, MAX_RPC_BYTES)
            socket.bind(self.endpoint)
            self._ready.set()
            while not self._stop.is_set():
                if not socket.poll(100, zmq.POLLIN):
                    continue
                payload = socket.recv()
                try:
                    if len(payload) > MAX_RPC_BYTES:
                        raise ValueError("metadata RPC message exceeds its bounded size")
                    request = _decode_json(payload)
                    response = self.dispatch(request)
                except Exception as exc:
                    response = {"status": "error", "error_type": type(exc).__name__, "error": str(exc)}
                socket.send(_canonical(response))
        except Exception as exc:
            traceback.print_exc()
            self._startup_error = exc
            self._ready.set()
        finally:
            if socket is not None:
                socket.close(linger=0)
            if context is not None:
                context.term()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=min(1.0, self.timeout_seconds))


class MetadataClient:
    """Bounded request/response client; never reads data via scheduler SDK."""

    def __init__(self, endpoint: str, *, timeout_seconds: float = 60.0):
        if not isinstance(endpoint, str) or not endpoint.startswith("ipc://"):
            raise ValueError("metadata RPC must use a local IPC endpoint")
        self.endpoint = endpoint
        self.timeout_seconds = _timeout_ms(timeout_seconds) / 1000
        self._lock = Lock()
        self._closed = False

    def _rpc(self, request: dict[str, Any]) -> dict[str, Any]:
        import zmq
        payload = _canonical(request)
        if len(payload) > MAX_RPC_BYTES:
            raise ValueError("metadata RPC message exceeds its bounded size")
        with self._lock:
            if self._closed:
                raise TransferError("metadata RPC client is closed")
            context = zmq.Context()
            socket = context.socket(zmq.REQ)
            deadline = time.monotonic() + self.timeout_seconds
            try:
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.IMMEDIATE, 1)
                socket.setsockopt(zmq.MAXMSGSIZE, MAX_RPC_BYTES)
                socket.setsockopt(zmq.SNDTIMEO, _timeout_ms(self.timeout_seconds))
                socket.connect(self.endpoint)
                socket.send(payload)
                socket.setsockopt(zmq.RCVTIMEO, _timeout_ms(max(0.001, deadline - time.monotonic())))
                data = socket.recv()
                if len(data) > MAX_RPC_BYTES:
                    raise TransferError("metadata RPC response exceeds its bounded size")
                response = _decode_json(data)
                if not isinstance(response, dict) or response.get("status") not in {"ok", "missing", "error"}:
                    raise TransferError("invalid metadata RPC response")
                if response["status"] == "error":
                    raise TransferError(f"metadata RPC {response.get('error_type')}: {response.get('error')}")
                return response
            except zmq.Again as exc:
                raise TransferError("metadata RPC exceeded its bounded deadline") from exc
            except (ValueError, UnicodeError) as exc:
                raise TransferError(f"invalid metadata RPC response: {exc}") from exc
            finally:
                socket.close(linger=0)
                context.term()

    def get(self, key: str) -> dict[str, Any] | None:
        response = self._rpc({"op": "get", "key": _key(key)})
        if response["status"] == "missing":
            return None
        if set(response) != {"status", "payload"}:
            raise TransferError("metadata get response is incomplete")
        return decode_manifest(_unbase64(response["payload"]))

    def put(self, key: str, manifest: Mapping[str, Any]) -> None:
        payload = encode_manifest(manifest)
        response = self._rpc({"op": "put", "key": _key(key), "payload": base64.b64encode(payload).decode("ascii")})
        if response != {"status": "ok"}:
            raise TransferError("metadata publication was not acknowledged")

    def remove_swap(self, plan) -> int:
        """Retire a private PayloadPlan; never accept caller-selected keys."""
        from .plans import PayloadPlan
        if not isinstance(plan, PayloadPlan) or plan.page_hashes:
            raise ValueError("private cleanup requires a non-prefix PayloadPlan")
        fields = {name: getattr(plan, name) for name in _SWAP_FIELDS}
        response = self._rpc({"op": "remove_swap", "plan": fields})
        if (set(response) != {"status", "removed", "missing"} or response["status"] != "ok"
                or type(response["removed"]) is not int or response["removed"] < 0
                or type(response["missing"]) is not int or response["missing"] < 0):
            raise TransferError("private swap cleanup was not acknowledged")
        return response["removed"]

    def close(self) -> None:
        with self._lock:
            self._closed = True
