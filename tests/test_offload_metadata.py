"""Host manifest integrity/real local IPC tests; no NPU or SDK claim.

Archive #37--49/#127/#128/#157/#163/#171/#183: never publish or consume a mixed
generation checkpoint, and never convert a metadata transport error to hit.
"""

import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from threading import Barrier, Event, current_thread
import time
from types import SimpleNamespace

import pytest

from oscar_ascend.offload import metadata as metadata_module
from oscar_ascend.offload.metadata import (
    ManifestIntegrityError,
    MetadataClient,
    MetadataServer,
    decode_manifest,
    encode_manifest,
    rpc_endpoint,
)
from oscar_ascend.offload.transport import MAX_HOST_OBJECT_BYTES, MissingObjectsError, StrictMemcacheTransport, TransferError
from oscar_ascend.offload.plans import PayloadPlan, expected_keys, snapshot_key
from test_offload_native_bridge import bridge_module  # noqa: F401


class MemoryStore:
    def __init__(self):
        self.objects = {}
        self.put_status = 0
        self.get_override = None
        self.immutable = False
        self.closed = False

    def put(self, key, payload):
        if self.immutable and key in self.objects:
            return -3006
        self.objects[key] = payload
        return self.put_status

    def get(self, key):
        if self.get_override is not None:
            return self.get_override
        return self.objects.get(key, b"")

    def batch_is_exist(self, keys):
        return [int(key in self.objects) for key in keys]

    def remove(self, key):
        if key not in self.objects:
            return -3007
        del self.objects[key]
        return 0

    def remove_batch(self, keys):
        return [self.remove(key) for key in keys]

    def close(self):
        self.closed = True


def body(generation="a"):
    return {"namespace": "math+pt", "base_key": "prefix/data/" + generation,
            "committed_tokens": 4096, "index_key": "prefix",
            "keys": [generation + "/full", generation + "/gdn"]}


@pytest.fixture
def decoded_payloads(monkeypatch):
    calls = []
    original = metadata_module.decode_manifest

    def decode(payload):
        calls.append(payload)
        return original(payload)

    monkeypatch.setattr(metadata_module, "decode_manifest", decode)
    return calls


@pytest.fixture
def rpc():
    # Unix socket paths have a short platform limit; pytest's macOS default
    # temp root can exceed it before the endpoint filename is appended.
    with TemporaryDirectory(prefix="oscar-rpc-", dir="/tmp") as directory:
        store = MemoryStore()
        for generation in ("a", "first", "second", "0", "1", "old", "new"):
            store.objects.update({key: b"payload" for key in body(generation)["keys"]})
        transport = StrictMemcacheTransport(store)
        endpoint = "ipc://" + directory + "/manifest"
        server = MetadataServer(transport, endpoint, timeout_seconds=1).start()
        client = MetadataClient(endpoint, timeout_seconds=1)
        try:
            yield store, transport, server, client
        finally:
            client.close()
            server.close()
            assert not server._thread.is_alive()
            transport.close()


def test_manifest_roundtrip_and_key_order_are_canonical():
    value = body()
    assert decode_manifest(encode_manifest(value)) == value
    assert encode_manifest(dict(reversed(tuple(value.items())))) == encode_manifest(value)


@pytest.mark.parametrize("mutation", ["checksum", "truncated", "mixed_generation", "duplicate", "nan"])
def test_torn_or_ambiguous_manifests_never_decode_as_hit(mutation):
    encoded = encode_manifest(body("old"))
    if mutation == "checksum":
        record = json.loads(encoded)
        record["checksum"] = "0" * 64
        encoded = json.dumps(record).encode()
    elif mutation == "truncated":
        encoded = encoded[:-3]
    elif mutation == "mixed_generation":
        record = json.loads(encoded)
        record["body"]["base_key"] = "prefix/data/new"
        encoded = json.dumps(record).encode()
    elif mutation == "duplicate":
        encoded = encoded.replace(b'"body":{', b'"body":{},"body":{')
    else:
        encoded = encoded.replace(b'4096', b'NaN')
    with pytest.raises(ManifestIntegrityError):
        decode_manifest(encoded)


def test_real_local_rpc_put_get_overwrite_and_missing(rpc):
    _, _, _, client = rpc
    assert client.get("prefix") is None
    client.put("prefix", body("first"))
    assert client.get("prefix") == body("first")
    client.put("prefix", body("second"))
    assert client.get("prefix") == body("second")


def test_repeated_missing_indexes_skip_sdk_get_and_later_publication_is_visible(rpc, capsys):
    store, transport, _, client = rpc
    original_get = store.get
    get_calls = []
    def sdk_get(key):
        get_calls.append(key)
        if key not in store.objects:
            print("SDK Get ERROR: object not found")
        return original_get(key)
    store.get = sdk_get
    for index in range(32):
        assert client.get(f"missing-prefix-{index}") is None
    assert client.get("prefix") is None
    assert get_calls == [], "ordinary cold prefix lookups must not call SDK Get"
    assert "SDK Get ERROR" not in capsys.readouterr().out
    assert not transport.failed
    # A cold miss is not cached: the same key must be readable after publish.
    client.put("prefix", body("first"))
    assert client.get("prefix") == body("first")
    assert get_calls == ["prefix"]
    assert not transport.failed


@pytest.mark.parametrize("failure", [RuntimeError("existence SDK failure"), [2]])
def test_host_lookup_existence_errors_are_not_missing_and_never_call_get(rpc, failure):
    store, transport, _, client = rpc
    get_calls = []
    def exists(keys):
        if isinstance(failure, Exception):
            raise failure
        return failure
    store.batch_is_exist = exists
    store.get = lambda key: get_calls.append(key) or b""
    with pytest.raises(TransferError, match="batch_is_exist"):
        client.get("prefix")
    assert get_calls == []
    assert transport.failed


def test_host_lookup_evicted_between_exists_and_get_is_a_nonfatal_miss(rpc):
    store, transport, _, client = rpc
    store.objects["prefix"] = encode_manifest(body())
    original_exists = store.batch_is_exist
    calls = []
    def exists(keys):
        calls.append(("exists", tuple(keys)))
        return original_exists(keys)
    def evicted_get(key):
        calls.append(("get", key))
        del store.objects[key]
        return b""
    store.batch_is_exist, store.get = exists, evicted_get
    assert client.get("prefix") is None
    assert calls == [("exists", ("prefix",)), ("get", "prefix"), ("exists", ("prefix",))]
    assert not transport.failed


def test_host_lookup_get_exception_is_explicit_not_a_cache_miss(rpc):
    store, transport, _, client = rpc
    store.objects["prefix"] = encode_manifest(body())
    error = RuntimeError("SDK Get transport failed")
    def failed_get(key):
        raise error
    store.get = failed_get
    with pytest.raises(TransferError, match="SDK Get transport failed"):
        client.get("prefix")
    assert transport.failed
    assert transport._failure.__cause__ is error


def test_real_rpc_checksum_error_is_explicit_and_not_missing(rpc):
    store, _, _, client = rpc
    value = json.loads(encode_manifest(body("old")))
    value["body"]["base_key"] = "prefix/data/concurrent-new"
    store.objects["prefix"] = json.dumps(value).encode()
    with pytest.raises(TransferError, match="ManifestIntegrityError"):
        client.get("prefix")


def test_real_rpc_existing_but_unreadable_host_object_fails(rpc):
    store, transport, _, client = rpc
    store.objects["prefix"] = encode_manifest(body())
    store.get_override = b""
    with pytest.raises(TransferError, match="empty bytes for an existing"):
        client.get("prefix")
    assert transport.failed


def test_malformed_put_does_not_call_host_sdk(rpc):
    store, _, server, _ = rpc
    before = dict(store.objects)
    response = server.dispatch({"op": "put", "key": "prefix", "payload": base64.b64encode(b"partial").decode()})
    assert response["status"] == "error"
    assert store.objects == before


@pytest.mark.parametrize("mutation", ["checksum", "duplicate-json-key"])
def test_rpc_put_rejects_corrupt_manifest_before_any_sdk_call(rpc, decoded_payloads, mutation):
    store, _, _, client = rpc
    payload = encode_manifest(body("first"))
    if mutation == "checksum":
        record = json.loads(payload)
        record["checksum"] = "0" * 64
        payload = json.dumps(record).encode()
    else:
        payload = payload.replace(b'"body":{', b'"body":{},"body":{')
    before = dict(store.objects)
    sdk_calls = []
    original_exists = store.batch_is_exist
    original_put = store.put
    store.batch_is_exist = lambda keys: (sdk_calls.append("exists"), original_exists(keys))[1]
    store.put = lambda key, value: (sdk_calls.append("put"), original_put(key, value))[1]
    with pytest.raises(TransferError, match="ManifestIntegrityError"):
        client._rpc({"op": "put", "key": "prefix", "payload": base64.b64encode(payload).decode()})
    assert decoded_payloads == [payload]
    assert sdk_calls == [] and store.objects == before


def test_multiple_rpc_clients_receive_one_verified_generation(rpc):
    _, _, server, client = rpc
    clients = [MetadataClient(server.endpoint, timeout_seconds=1) for _ in range(2)]
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda item: item[1].put("prefix", body(str(item[0]))), enumerate(clients)))
        result = client.get("prefix")
        assert result in (body("0"), body("1"))
    finally:
        for item in clients:
            item.close()


def test_missing_rpc_server_has_bounded_failure():
    with TemporaryDirectory(prefix="oscar-rpc-", dir="/tmp") as directory:
        client = MetadataClient("ipc://" + directory + "/absent", timeout_seconds=0.04)
        started = time.monotonic()
        with pytest.raises(TransferError, match="deadline"):
            client.get("prefix")
        assert time.monotonic() - started < 0.5
        client.close()


def test_rpc_exists_batches_cold_indexes_without_get_or_negative_cache(rpc):
    store, transport, _, client = rpc
    calls = []
    original_exists = store.batch_is_exist
    original_get = store.get
    def counted_exists(keys):
        calls.append(tuple(keys))
        return original_exists(keys)
    store.batch_is_exist = counted_exists
    store.get = lambda key: pytest.fail("existence queries must not call SDK Get")
    keys = tuple(f"cold-index-{index}" for index in range(256))
    assert client.exists(keys) == (False,) * 256
    assert calls == [keys]
    assert not transport.failed
    # Existence is a fresh query, not a negative cache or a manifest proof.
    manifest = body("first")
    manifest["index_key"] = keys[0]
    client.put(keys[0], manifest)
    calls.clear()
    assert client.exists(keys) == (True,) + (False,) * 255
    assert calls == [keys]
    assert not transport.failed
    store.get = original_get
    assert client.get(keys[0]) == manifest


def test_rpc_exists_chunks_wire_requests_at_256_and_preserves_order(rpc):
    store, _, server, client = rpc
    keys = tuple(f"index-{index}" for index in range(257))
    store.objects.update({keys[index]: b"not-a-manifest" for index in (0, 255, 256)})
    requests = []
    original_dispatch = server.dispatch
    def dispatch(request):
        requests.append(request)
        return original_dispatch(request)
    server.dispatch = dispatch
    result = client.exists(keys)
    assert result == tuple(index in (0, 255, 256) for index in range(257))
    assert requests == [{"op": "exists", "keys": list(keys[:256])},
                        {"op": "exists", "keys": list(keys[256:])}]
    assert client.exists(()) == ()
    assert len(requests) == 2


@pytest.mark.parametrize("keys", ["prefix", b"prefix", ["prefix", "prefix"], [True], [""], ["x" * 256]])
def test_rpc_exists_rejects_invalid_client_keys_before_sdk(rpc, keys):
    store, _, _, client = rpc
    store.batch_is_exist = lambda keys: pytest.fail("invalid keys must not reach the SDK")
    with pytest.raises((ValueError, TypeError)):
        client.exists(keys)


@pytest.mark.parametrize("keys", [[], ["same", "same"], [False], "prefix",
                                 [f"key-{index}" for index in range(257)]])
def test_rpc_exists_rejects_invalid_wire_batch_before_sdk(rpc, keys):
    store, _, server, _ = rpc
    store.batch_is_exist = lambda keys: pytest.fail("invalid wire batch must not reach the SDK")
    assert server.dispatch({"op": "exists", "keys": keys})["status"] == "error"


@pytest.mark.parametrize("failure", [RuntimeError("exists SDK failed"), [2]])
def test_rpc_exists_preserves_sdk_errors_instead_of_returning_misses(rpc, failure):
    store, transport, _, client = rpc
    def invalid_exists(keys):
        if isinstance(failure, Exception):
            raise failure
        return failure
    store.batch_is_exist = invalid_exists
    with pytest.raises(TransferError, match="batch_is_exist"):
        client.exists(["prefix"])
    assert transport.failed


@pytest.mark.parametrize("response", [
    {"status": "ok", "exists": []},
    {"status": "ok", "exists": [True, False]},
    {"status": "ok", "exists": [1]},
    {"status": "ok", "exists": "false"},
    {"status": "ok"},
    {"status": "missing"},
    {"status": "ok", "exists": [False], "unexpected": True},
])
def test_rpc_exists_rejects_malformed_responses(rpc, response):
    _, _, server, client = rpc
    server.dispatch = lambda request: response
    with pytest.raises(TransferError):
        client.exists(["prefix"])


@pytest.fixture
def zmq_connections(monkeypatch, rpc):
    """Observe real IPC resources and socket thread ownership, not private fields."""
    import zmq
    original_context = zmq.Context
    contexts, sockets = [], []
    class SocketObserver:
        def __init__(self, socket):
            self.socket = socket
            self.owner = current_thread()
        def __getattr__(self, name):
            return getattr(self.socket, name)
        def connect(self, *args, **kwargs):
            assert current_thread() is self.owner
            return self.socket.connect(*args, **kwargs)
        def send(self, *args, **kwargs):
            assert current_thread() is self.owner, "REQ socket crossed calling threads"
            return self.socket.send(*args, **kwargs)
        def recv(self, *args, **kwargs):
            assert current_thread() is self.owner, "REQ socket crossed calling threads"
            return self.socket.recv(*args, **kwargs)
    class ObservedContext(original_context):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            contexts.append(self)
        def socket(self, socket_type, *args, **kwargs):
            socket = super().socket(socket_type, *args, **kwargs)
            if socket_type == zmq.REQ:
                observer = SocketObserver(socket)
                sockets.append(observer)
                return observer
            return socket
    monkeypatch.setattr(zmq, "Context", ObservedContext)
    return SimpleNamespace(contexts=contexts, sockets=sockets)


def test_metadata_client_reuses_same_thread_resources_and_closes_idempotently(rpc, zmq_connections):
    _, _, server, _ = rpc
    tracked = zmq_connections
    client = MetadataClient(server.endpoint, timeout_seconds=1)
    try:
        for _ in range(3):
            assert client.get("prefix") is None
            assert client.exists(["first/full", "missing"]) == (True, False)
        assert len(tracked.contexts) == len(tracked.sockets) == 1
        assert not tracked.contexts[0].closed and not tracked.sockets[0].closed
    finally:
        client.close()
    client.close()
    assert all(context.closed for context in tracked.contexts)
    assert all(socket.closed for socket in tracked.sockets)
    for operation in (lambda: client.get("prefix"), lambda: client.exists(["prefix"]),
                      lambda: client.exists(())):
        with pytest.raises(TransferError, match="closed"):
            operation()
    assert len(tracked.contexts) == len(tracked.sockets) == 1


def test_one_metadata_client_uses_separate_reusable_sockets_for_calling_threads(rpc, zmq_connections):
    _, _, server, _ = rpc
    tracked = zmq_connections
    client = MetadataClient(server.endpoint, timeout_seconds=1)
    barrier = Barrier(2)
    def query(index):
        barrier.wait(timeout=2)
        for _ in range(3):
            assert client.exists([f"{index}/full", f"missing-{index}"]) == (True, False)
        return current_thread()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            owners = list(pool.map(query, (0, 1)))
        assert len(tracked.contexts) == 1
        assert len(tracked.sockets) == 2
        assert {socket.owner for socket in tracked.sockets} == set(owners)
    finally:
        client.close()
    assert all(context.closed for context in tracked.contexts)
    assert all(socket.closed for socket in tracked.sockets)


@pytest.mark.parametrize("bad_response", [
    {"status": "unrecognized"},
    {"status": "error", "error_type": "FixtureError", "error": "failed exchange"},
])
def test_failed_exchange_discards_connection_without_retry(rpc, zmq_connections, bad_response):
    _, _, server, _ = rpc
    tracked = zmq_connections
    original_dispatch = server.dispatch
    calls = []
    def dispatch(request):
        calls.append(request)
        return bad_response if len(calls) == 1 else original_dispatch(request)
    server.dispatch = dispatch
    client = MetadataClient(server.endpoint, timeout_seconds=1)
    try:
        with pytest.raises(TransferError):
            client.get("first-query")
        assert len(calls) == 1
        assert len(tracked.sockets) == 1 and tracked.sockets[0].closed
        assert client.get("second-query") is None
        assert [request["key"] for request in calls] == ["first-query", "second-query"]
        assert len(tracked.sockets) == 2
    finally:
        client.close()
    assert all(context.closed for context in tracked.contexts)
    assert all(socket.closed for socket in tracked.sockets)


def test_timed_out_exchange_discards_socket_and_late_reply_cannot_answer_next_request(rpc, zmq_connections):
    store, _, server, _ = rpc
    tracked = zmq_connections
    late_entered, release_late = Event(), Event()
    calls = []
    old, new = body("old"), body("new")
    old["index_key"], new["index_key"] = "late", "next"
    store.objects.update(late=encode_manifest(old), next=encode_manifest(new))
    original_dispatch = server.dispatch
    def dispatch(request):
        calls.append(request)
        if request.get("key") == "late":
            late_entered.set()
            assert release_late.wait(timeout=2), "test failed to release the first reply"
        return original_dispatch(request)
    server.dispatch = dispatch
    client = MetadataClient(server.endpoint, timeout_seconds=0.05)
    try:
        with pytest.raises(TransferError, match="deadline"):
            client.get("late")
        assert late_entered.is_set()
        assert len(calls) == 1  # No automatic retry of the failed exchange.
        assert len(tracked.sockets) == 1 and tracked.sockets[0].closed
        release_late.set()
        client.timeout_seconds = 1
        assert client.get("next") == new
        assert [request["key"] for request in calls] == ["late", "next"]
        assert len(tracked.sockets) == 2
    finally:
        release_late.set()
        client.close()
    assert all(context.closed for context in tracked.contexts)
    assert all(socket.closed for socket in tracked.sockets)


def test_sdk_host_contract_is_strict_and_scheduler_cannot_do_data_io():
    store = MemoryStore()
    transport = StrictMemcacheTransport(store)
    with pytest.raises(MissingObjectsError):
        transport.host_get("missing")
    transport.host_put("manifest", encode_manifest(body()))
    assert decode_manifest(transport.host_get("manifest")) == body()
    store.put_status = None
    with pytest.raises(TransferError, match="invalid/failed"):
        transport.host_put("failed", b"bytes")
    scheduler = StrictMemcacheTransport(store, scheduler=True)
    with pytest.raises(TransferError, match="init_bm=False"):
        scheduler.host_get("manifest")
    with pytest.raises(TransferError, match="init_bm=False"):
        scheduler.host_put("manifest", b"bytes")


def test_rpc_endpoints_are_shared_within_engine_and_isolated_between_engines(monkeypatch):
    module = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler"
    monkeypatch.setitem(sys.modules, module, SimpleNamespace(get_zmq_rpc_path_lookup=lambda _: "ipc:///tmp/lookup"))
    def config(engine):
        return SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id=engine))
    assert rpc_endpoint(config("a")) == rpc_endpoint(config("a"))
    assert rpc_endpoint(config("a")) != rpc_endpoint(config("b"))


def test_immutable_sdk_retains_complete_first_writer_without_error(rpc, decoded_payloads):
    store, transport, _, client = rpc
    store.immutable = True
    client.put("prefix", body("first"))
    assert decoded_payloads == [encode_manifest(body("first"))]
    decoded_payloads.clear()
    client.put("prefix", body("second"))
    # Validate the incoming bytes once and independently verify the winner.
    assert decoded_payloads == [encode_manifest(body("second")), encode_manifest(body("first"))]
    assert client.get("prefix") == body("first")
    assert not transport.failed


def test_immutable_sdk_replaces_stale_index_with_complete_new_generation(rpc):
    store, _, _, client = rpc
    store.immutable = True
    client.put("prefix", body("first"))
    del store.objects["first/gdn"]
    client.put("prefix", body("second"))
    assert client.get("prefix") == body("second")


def test_concurrent_replacement_winner_is_verified_instead_of_overwritten(rpc, decoded_payloads):
    store, transport, _, client = rpc
    store.immutable = True
    client.put("prefix", body("first"))
    del store.objects["first/gdn"]
    remove = store.remove
    def racing_remove(key):
        result = remove(key)
        # Another engine wins the gap between remove and the bounded retry.
        store.objects[key] = encode_manifest(body("new"))
        return result
    store.remove = racing_remove
    decoded_payloads.clear()
    client.put("prefix", body("second"))
    assert decoded_payloads == [encode_manifest(body(name)) for name in ("second", "first", "new")]
    assert client.get("prefix") == body("new")
    assert not transport.failed


def test_repeated_incomplete_replacement_has_bounded_explicit_failure(rpc):
    store, _, _, client = rpc
    store.immutable = True
    client.put("prefix", body("first"))
    del store.objects["first/gdn"]
    put = store.put
    attempts = []
    def counted_put(key, payload):
        attempts.append(key)
        return put(key, payload)
    store.put = counted_put
    def racing_remove(key):
        # A racing writer exposes another stale generation, never complete.
        store.objects[key] = encode_manifest(body("first"))
        return 0
    store.remove = racing_remove
    with pytest.raises(TransferError, match="two bounded races"):
        client.put("prefix", body("second"))
    assert attempts == ["prefix", "prefix"]


def test_manifest_is_not_published_until_every_payload_exists(rpc):
    store, _, _, client = rpc
    del store.objects["first/gdn"]
    with pytest.raises(TransferError, match="before every payload"):
        client.put("prefix", body("first"))
    assert "prefix" not in store.objects


@pytest.mark.parametrize("outcome", ["complete", "missing-payload", "sdk-error"])
def test_connector_all_tp_receipts_publish_through_rpc_with_one_payload_check(
        rpc, bridge_module, outcome):
    store, transport, _, client = rpc
    sender = bridge_module.OscarAscendStoreConnector.__new__(bridge_module.OscarAscendStoreConnector)
    sender.namespace = "joint-model-geometry"
    sender.world_size = 2
    sender.geometry = (
        {"layer_name": "full", "cache_family": "oscar_full", "storage_block_size": 4},
        {"layer_name": "gdn", "cache_family": "gdn", "components": [{}, {}]},
    )
    sender._scheduled_loads = {}
    sender._prefix_publications = {}
    sender._prefix_view = "shared"
    # Both paths point at the same SDK counter: a duplicated scheduler-side
    # check cannot hide behind a separate fake transport.
    sender.transport = transport
    sender.metadata_client = client
    plan = PayloadPlan("request", 1, 0, 0, 8, (), "prefix/data/checkpoint")
    keys = expected_keys(sender.namespace, sender.world_size, plan, sender.geometry)
    store.objects.update({key: b"payload" for key in keys})
    if outcome == "missing-payload":
        del store.objects[keys[-1]]
    calls = []
    original_exists = store.batch_is_exist

    def exists(batch):
        calls.append(tuple(batch))
        if outcome == "sdk-error":
            raise RuntimeError("SDK unavailable")
        return original_exists(batch)

    store.batch_is_exist = exists

    def output(rank):
        receipt = bridge_module.PrefixReceipt("prefix", plan, rank, sender._prefix_view)
        metadata = bridge_module.OscarOffloadWorkerMetadata(prefixes=(receipt,))
        return SimpleNamespace(kv_connector_worker_meta=metadata)

    sender.update_connector_output(output(0))
    assert calls == [] and "prefix" not in store.objects
    if outcome == "complete":
        sender.update_connector_output(output(1))
        manifest = decode_manifest(store.objects["prefix"])
        assert manifest["keys"] == list(keys)
        assert manifest["namespace"] == sender.namespace
        assert not sender._prefix_publications
    else:
        message = "before every payload" if outcome == "missing-payload" else "SDK unavailable"
        with pytest.raises(TransferError, match=message):
            sender.update_connector_output(output(1))
        assert "prefix" not in store.objects
    assert calls == [keys]


def test_duplicate_manifest_with_foreign_identity_is_never_retained(rpc):
    store, _, _, client = rpc
    store.immutable = True
    previous = body("first")
    previous["namespace"] = "wrong-model"
    store.objects["prefix"] = encode_manifest(previous)
    with pytest.raises(TransferError, match="another identity"):
        client.put("prefix", body("second"))


def test_keys_obey_sdk_utf8_byte_limit():
    transport = StrictMemcacheTransport(MemoryStore())
    with pytest.raises(ValueError, match="256"):
        transport.host_put("a" * 256, b"manifest")
    with pytest.raises(ValueError, match="256"):
        transport.exists(["界" * 86])


def test_host_known_duplicate_and_known_missing_codes_are_opt_in():
    store = MemoryStore()
    transport = StrictMemcacheTransport(store)
    store.immutable = True
    assert transport.host_put("key", b"first")
    assert not transport.host_put("key", b"second", allow_duplicate=True)
    assert transport.host_get("key") == b"first"
    assert transport.host_remove("key")
    assert not transport.host_remove("key")
    assert not transport.failed


def swap_geometry():
    return ({"layer_name": "full", "cache_family": "oscar_full", "storage_block_size": 128,
             "components": [{}, {}, {}, {}]},
            {"layer_name": "gdn", "cache_family": "gdn", "storage_block_size": 128,
             "components": [{}, {}]})


def swap_plan(session="session", *, version=1):
    plan = PayloadPlan("request", 1, version, 0, 256, (), "")
    return replace(plan, base_key=snapshot_key(session, plan, plan.committed_tokens))


def configure_swap_fixture(rpc):
    store, _, server, _ = rpc
    geometry = swap_geometry()
    server.configure_swap_scope("ns", "session", 2, geometry, max_tokens=1024)
    plan = swap_plan()
    keys = expected_keys("ns", 2, plan, geometry)
    store.objects.update({key: b"private payload" for key in keys})
    return plan, keys


def test_private_cleanup_real_rpc_derives_all_rank_keys_and_is_idempotent(rpc):
    store, _, _, client = rpc
    plan, keys = configure_swap_fixture(rpc)
    unrelated = dict(store.objects)
    for key in keys:
        unrelated.pop(key)
    assert client.remove_swap(plan) == len(keys)
    assert store.objects == unrelated
    assert client.remove_swap(plan) == 0


def test_private_cleanup_rejects_foreign_session_and_does_not_touch_storage(rpc):
    store, _, _, client = rpc
    configure_swap_fixture(rpc)
    before = dict(store.objects)
    with pytest.raises(TransferError, match="outside the current session"):
        client.remove_swap(swap_plan("other-engine"))
    assert store.objects == before


@pytest.mark.parametrize("mutation", ["prefix", "version", "length", "identity", "arbitrary_keys"])
def test_private_cleanup_rejects_expanded_or_forged_authority(rpc, mutation):
    store, _, server, client = rpc
    plan, _ = configure_swap_fixture(rpc)
    before = dict(store.objects)
    if mutation == "prefix":
        with pytest.raises(ValueError, match="non-prefix"):
            client.remove_swap(replace(plan, page_hashes=(("full", ("native-hash",)),)))
    elif mutation == "arbitrary_keys":
        response = server.dispatch({"op": "remove_swap", "plan": {"keys": list(store.objects)}})
        assert response["status"] == "error"
    else:
        changes = {"version": {"version": 0}, "length": {"committed_tokens": 1025},
                   "identity": {"request_id": "another-request"}}[mutation]
        with pytest.raises(TransferError):
            client.remove_swap(replace(plan, **changes))
    assert store.objects == before


def test_private_cleanup_stale_version_cannot_delete_new_generation(rpc):
    store, _, _, client = rpc
    old_plan, old_keys = configure_swap_fixture(rpc)
    new_plan = swap_plan(version=2)
    new_keys = expected_keys("ns", 2, new_plan, swap_geometry())
    store.objects.update({key: b"new generation" for key in new_keys})
    client.remove_swap(old_plan)
    client.remove_swap(old_plan)
    assert all(key not in store.objects for key in old_keys)
    assert all(store.objects[key] == b"new generation" for key in new_keys)


def test_private_cleanup_missing_race_is_safe_but_real_delete_error_fails(rpc):
    store, transport, _, client = rpc
    plan, keys = configure_swap_fixture(rpc)
    del store.objects[keys[0]]
    assert client.remove_swap(plan) == len(keys) - 1
    store.objects.update({key: b"payload" for key in keys})
    store.remove_batch = lambda _: [-3104] * len(keys)
    with pytest.raises(TransferError, match="failed object statuses"):
        client.remove_swap(plan)
    assert transport.failed
    assert all(key in store.objects for key in keys)


def test_private_cleanup_scope_cannot_be_configured_through_rpc_or_rebound(rpc):
    _, _, server, client = rpc
    with pytest.raises(TransferError, match="not configured"):
        client.remove_swap(swap_plan())
    configure_swap_fixture(rpc)
    server.configure_swap_scope("ns", "session", 2, swap_geometry(), max_tokens=1024)
    with pytest.raises(TransferError, match="cannot be rebound"):
        server.configure_swap_scope("ns", "another-session", 2, swap_geometry(), max_tokens=1024)
    assert server.dispatch({"op": "configure_swap_scope", "session": "attack"})["status"] == "error"


@pytest.mark.parametrize("physical_block", [768, 2304])
def test_target_maximum_length_complete_manifest_roundtrips_over_real_ipc(rpc, physical_block):
    # Archive #122 records target FULL layers 3,7,...,63 (16), with one
    # additional MTP FULL layer at serving time; the other 48 are GDN. Use
    # both supported physical geometries, never metadata128 block counts.
    config = json.loads((Path(__file__).parents[1] / "configs" / "target.json").read_text())
    world = config["tensor_parallel_size"]
    max_tokens = config["max_model_len"]
    layers = []
    for layer in range(64):
        full = layer % 4 == 3
        layers.append({"layer_name": f"model.layers.{layer}.self_attn",
                       "cache_family": "oscar_full" if full else "gdn",
                       "storage_block_size": physical_block if full else 768,
                       "components": [{}] * (4 if full else 2)})
    layers.append({"layer_name": "mtp.layers.0.self_attn", "cache_family": "oscar_full",
                   "storage_block_size": physical_block, "components": [{}, {}, {}, {}]})
    tokens = (max_tokens - 1) // physical_block * physical_block
    plan = PayloadPlan("maximum-prefix", 1, 0, 0, tokens, (), "prefix/data/maximum-generation")
    keys = expected_keys("target-model-namespace", world, plan, layers)
    manifest = {"namespace": "target-model-namespace", "committed_tokens": tokens,
                "base_key": plan.base_key, "index_key": "prefix", "keys": list(keys)}
    encoded = encode_manifest(manifest)
    assert len(keys) == world * (17 * (tokens // physical_block) + 48 * 2)
    if physical_block == 768:
        assert len(encoded) > 1024 * 1024  # Reproduces the former hard failure.
    assert len(encoded) < MAX_HOST_OBJECT_BYTES
    assert decode_manifest(encoded) == manifest
    store, _, server, _ = rpc
    store.objects.update({key: b"device-payload-is-not-needed-for-metadata-test" for key in keys})
    client = MetadataClient(server.endpoint, timeout_seconds=5)
    try:
        client.put("prefix", manifest)
        restored = client.get("prefix")
        assert restored == manifest
        assert restored["keys"] == list(keys)
    finally:
        client.close()


def test_manifest_size_remains_bounded():
    with pytest.raises(ValueError, match="bounded metadata size"):
        encode_manifest({"padding": "x" * MAX_HOST_OBJECT_BYTES})
