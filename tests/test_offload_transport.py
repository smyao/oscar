"""Host SDK-contract tests, not NPU/DMA evidence.

Archive #34/#36/#37--49/#127/#128/#157/#163/#171: fixed buffers, complete
physical/window/GDN bytes, all-participant completion, and strict errors.
"""

from dataclasses import dataclass
import json
import pickle
import signal
import subprocess
import sys
import textwrap
from threading import Event, Lock, Thread
from types import SimpleNamespace

import pytest

from oscar_ascend.offload.transport import (
    ByteSegment,
    MissingObjectsError,
    ObjectSizeError,
    StrictMemcacheTransport,
    TransferError,
    TransferObject,
)
from oscar_ascend.offload.worker import (
    CheckpointDescriptor,
    CheckpointReceipt,
    CheckpointWorker,
    require_joint_receipts,
)


BASE = 2 * 1024 * 1024


@dataclass
class KeyInfo:
    nbytes: int

    def size(self):
        return self.nbytes


class FakeStore:
    def __init__(self):
        self.memory = bytearray(range(256)) * 16
        self.objects = {}
        self.calls = []
        self.overrides = {}

    def result(self, method, default):
        value = self.overrides.get(method, default)
        if isinstance(value, Exception):
            raise value
        return value

    def register_buffer(self, address, length):
        self.calls.append(("register", address, length))
        return self.result("register", 0)

    def unregister_buffer(self, address, length):
        self.calls.append(("unregister", address, length))
        return self.result("unregister", 0)

    def close(self):
        self.calls.append(("close",))

    def remove_batch(self, keys):
        self.calls.append(("remove_batch", tuple(keys)))
        result = []
        for key in keys:
            result.append(0 if key in self.objects else -3007)
            self.objects.pop(key, None)
        return self.result("remove_batch", result)

    def batch_is_exist(self, keys):
        self.calls.append(("exists", tuple(keys)))
        return self.result("exists", [int(key in self.objects) for key in keys])

    def batch_get_key_info(self, keys):
        self.calls.append(("info", tuple(keys)))
        return self.result("info", [KeyInfo(len(self.objects.get(key, b""))) for key in keys])

    def batch_put_from_layers(self, keys, addresses, lengths, direction):
        assert direction == 0
        self.calls.append(("put", tuple(keys), addresses, lengths))
        for key, addrs, sizes in zip(keys, addresses, lengths):
            self.objects[key] = b"".join(bytes(self.memory[addr - BASE : addr - BASE + size]) for addr, size in zip(addrs, sizes))
        return self.result("put", [0] * len(keys))

    def batch_get_into_layers(self, keys, addresses, lengths, direction):
        assert direction == 1
        self.calls.append(("get", tuple(keys), addresses, lengths))
        for key, addrs, sizes in zip(keys, addresses, lengths):
            offset = 0
            for addr, size in zip(addrs, sizes):
                self.memory[addr - BASE : addr - BASE + size] = self.objects[key][offset : offset + size]
                offset += size
        return self.result("get", [0] * len(keys))


def setup_transport(**kwargs):
    store = FakeStore()
    transport = StrictMemcacheTransport(store, **kwargs)
    transport.register_buffers([ByteSegment(BASE, len(store.memory))])
    return store, transport


def obj(key="key", offset=0, length=16):
    return TransferObject(key, (ByteSegment(BASE + offset, length),))


class InlineSerialWatchdog:
    """Host callbacks only; assert the production SDK-call serialization."""

    def __init__(self):
        self.guard = Lock()
        self.close_calls = 0

    def run(self, operation, function):
        assert self.guard.acquire(blocking=False), f"concurrent SDK call: {operation}"
        try:
            return function()
        finally:
            self.guard.release()

    def close(self):
        self.close_calls += 1


class ThreadCall:
    def __init__(self, function):
        self.result = None
        self.errors = []

        def run():
            try:
                self.result = function()
            except Exception as exc:
                self.errors.append(exc)

        self.thread = Thread(target=run, daemon=True)
        self.thread.start()

    def join(self):
        self.thread.join(timeout=5)
        assert not self.thread.is_alive(), "host callback did not finish"


def pause_between_batches(monkeypatch, transport, keys):
    """Pause outside the SDK call, without relying on lock scheduling fairness."""
    entered, release = Event(), Event()
    original = transport._batches

    def gated(objects):
        target = tuple(item.key for item in objects) == tuple(keys)
        for index, batch in enumerate(original(objects)):
            yield batch
            if target and index == 0:
                entered.set()
                assert release.wait(5), "host test did not release the next batch"

    monkeypatch.setattr(transport, "_batches", gated)
    return entered, release


def observe_close_wait(monkeypatch, transport, *, count=1):
    waiting = Event()
    original = transport._read_condition.wait
    calls = []

    def observed_wait(*args, **kwargs):
        calls.append(True)
        if len(calls) == count:
            waiting.set()
        return original(*args, **kwargs)

    monkeypatch.setattr(transport._read_condition, "wait", observed_wait)
    return waiting


def test_full_components_and_gdn_roundtrip_without_token_proportional_copy():
    store, transport = setup_transport()
    # Variable-size packed, wk, wv, tags and GDN state.  Every complete
    # component is copied; the byte gap between components is untouched.
    segments = tuple(ByteSegment(BASE + offset, length) for offset, length in [(0, 13), (32, 7), (64, 9), (96, 8), (128, 33)])
    item = TransferObject("ns/generation/rank0/full+gdn", segments)
    expected = b"".join(store.memory[s.address - BASE : s.address - BASE + s.length] for s in segments)
    receipt = transport.put([item])
    assert receipt.nbytes == 70
    assert store.objects[item.key] == expected
    store.memory[:] = b"\xff" * len(store.memory)
    transport.get([item])
    assert b"".join(store.memory[s.address - BASE : s.address - BASE + s.length] for s in segments) == expected
    assert store.memory[13:32] == b"\xff" * 19


@pytest.mark.parametrize("status", [None, [], [0, 0], [-1], [1], [False], ["0"]])
def test_put_never_swallows_absent_partial_or_failed_status(status):
    store, transport = setup_transport()
    store.overrides["put"] = status
    with pytest.raises(TransferError):
        transport.put([obj()])
    assert transport.failed
    previous = len(store.calls)
    with pytest.raises(TransferError, match="earlier SDK failure"):
        transport.put([obj("second")])
    assert len(store.calls) == previous


def test_exception_has_no_retry_and_is_not_reported_as_miss():
    store, transport = setup_transport()
    store.overrides["exists"] = TimeoutError("bounded SDK timeout")
    with pytest.raises(TransferError, match="TimeoutError"):
        transport.exists(["key"])
    assert sum(call[0] == "exists" for call in store.calls) == 1


@pytest.mark.parametrize("status", [None, [], [2], [-1], [True]])
def test_exists_validates_every_status_and_does_not_treat_errors_as_hits(status):
    store, transport = setup_transport()
    store.overrides["exists"] = status
    with pytest.raises(TransferError):
        transport.exists(["key"])


def test_missing_is_distinct_and_does_not_modify_destination_or_poison():
    store, transport = setup_transport()
    before = bytes(store.memory)
    with pytest.raises(MissingObjectsError) as exc:
        transport.get([obj("missing")])
    assert exc.value.keys == ("missing",)
    assert not transport.failed
    assert bytes(store.memory) == before
    assert all(call[0] != "get" for call in store.calls)
    assert transport._active_reads == 0
    assert transport._read_owner is None
    transport.put([obj("present")])
    transport.get([obj("present")])


def test_size_mismatch_fails_before_any_dma():
    store, transport = setup_transport()
    store.objects["key"] = b"wrong"
    before = bytes(store.memory)
    with pytest.raises(ObjectSizeError):
        transport.get([obj()])
    assert bytes(store.memory) == before
    assert all(call[0] != "get" for call in store.calls)
    assert transport.failed


def test_present_payload_checks_sizes_once_without_existence_queries():
    store, transport = setup_transport(max_batch_objects=2, max_batch_bytes=24)
    objects = [obj(str(i), offset=i * 16, length=8) for i in range(5)]
    for item in objects:
        store.objects[item.key] = item.key.encode() * 8
    store.calls.clear()
    try:
        receipt = transport.get(objects)
        assert receipt.keys == tuple(str(i) for i in range(5))
        assert [call[0] for call in store.calls] == ["info"] * 3 + ["get"] * 3
        assert [call[1] for call in store.calls[:3]] == [("0", "1"), ("2", "3"), ("4",)]
        for i in range(5):
            assert bytes(store.memory[i * 16:i * 16 + 8]) == str(i).encode() * 8
    finally:
        transport.close()


def test_metadata_batching_does_not_inherit_payload_byte_limit():
    store, transport = setup_transport(max_batch_objects=2, max_batch_bytes=8)
    objects = [obj(str(i), offset=i * 16, length=8) for i in range(3)]
    for item in objects:
        store.objects[item.key] = b"abcdefgh"
    store.calls.clear()
    try:
        transport.get(objects)
        assert [call[0] for call in store.calls] == ["info"] * 2 + ["get"] * 3
    finally:
        transport.close()


def test_zero_sizes_probe_only_ambiguous_keys_and_preserve_missing():
    store, transport = setup_transport(max_batch_objects=2)
    store.objects["present"] = b"x" * 16
    store.calls.clear()
    before = bytes(store.memory)
    try:
        with pytest.raises(MissingObjectsError) as error:
            transport.get([obj("present"), obj("missing", offset=32)])
        assert error.value.keys == ("missing",)
        assert store.calls == [("info", ("present", "missing")), ("exists", ("missing",))]
        assert bytes(store.memory) == before and not transport.failed
    finally:
        transport.close()


def test_zero_size_for_existing_object_is_failure_not_miss():
    store, transport = setup_transport()
    store.objects["key"] = b"x" * 16
    store.overrides["info"] = [KeyInfo(0)]
    before = bytes(store.memory)
    try:
        with pytest.raises(ObjectSizeError, match="zero|empty|size"):
            transport.get([obj()])
        assert transport.failed
        assert bytes(store.memory) == before
        assert not any(call[0] == "get" for call in store.calls)
    finally:
        transport.close()


def test_zero_size_probe_error_stays_error():
    store, transport = setup_transport()
    cause = TimeoutError("existence unavailable")
    store.overrides["exists"] = cause
    try:
        with pytest.raises(TransferError, match="existence unavailable") as error:
            transport.get([obj("missing")])
        assert error.value.__cause__ is cause
        assert transport.failed and not any(call[0] == "get" for call in store.calls)
    finally:
        transport.close()


@pytest.mark.parametrize("info", [None, [], [KeyInfo(16), KeyInfo(16)], [object()],
                                   [KeyInfo(True)], [KeyInfo(-1)], [KeyInfo("16")]])
def test_invalid_key_info_is_never_a_miss_or_permission_to_copy(info):
    store, transport = setup_transport()
    store.objects["key"] = b"x" * 16
    store.overrides["info"] = info
    try:
        with pytest.raises(TransferError):
            transport.get([obj()])
        assert transport.failed
        assert not any(call[0] in {"exists", "get"} for call in store.calls)
    finally:
        transport.close()


def test_late_batch_bad_size_prevents_all_payload_dma():
    store, transport = setup_transport(max_batch_objects=2)
    objects = [obj(str(i), offset=i * 32) for i in range(3)]
    store.objects.update({"0": b"a" * 16, "1": b"b" * 16, "2": b"wrong"})
    before = bytes(store.memory)
    try:
        with pytest.raises(ObjectSizeError):
            transport.get(objects)
        assert sum(call[0] == "info" for call in store.calls) == 2
        assert not any(call[0] == "get" for call in store.calls)
        assert bytes(store.memory) == before
        assert transport._active_reads == 0
        assert transport._read_owner is None
    finally:
        transport.close()


def test_eviction_after_size_check_is_failed_read_not_completion():
    store, transport = setup_transport()
    store.objects["key"] = b"x" * 16
    info = store.batch_get_key_info

    def evict_after_info(keys):
        result = info(keys)
        store.objects.clear()
        return result

    store.batch_get_key_info = evict_after_info
    try:
        with pytest.raises(TransferError) as error:
            transport.get([obj()])
        assert isinstance(error.value.__cause__, KeyError)
        assert transport.failed
    finally:
        transport.close()


def test_transfer_object_size_cache_preserves_constructor_and_serialization(monkeypatch):
    from dataclasses import asdict, fields, replace
    from oscar_ascend.offload import transport as module

    item = TransferObject("key", (ByteSegment(BASE, 7), ByteSegment(BASE + 16, 9)))
    assert tuple(field.name for field in fields(item)) == ("key", "segments")
    assert set(asdict(item)) == {"key", "segments"}
    assert item.__getstate__() == {"key": item.key, "segments": item.segments}
    restored = pickle.loads(pickle.dumps(item))
    assert restored == item and restored.nbytes == 16
    changed = replace(item, segments=(ByteSegment(BASE, 3),))
    assert changed.nbytes == 3

    def unexpected_sum(*args, **kwargs):
        pytest.fail("frozen object size was recomputed")

    monkeypatch.setattr(module, "sum", unexpected_sum, raising=False)
    assert item.nbytes == restored.nbytes == 16
    assert changed.nbytes == 3


def test_contiguous_window_components_coalesce_only_in_sdk_vectors():
    store, transport = setup_transport()
    segments = tuple(ByteSegment(BASE + start, size) for start, size in
                     ((0, 13), (64, 14), (78, 18), (96, 8)))
    item = TransferObject("packed+window-k+window-v+tags", segments)
    expected = bytes(store.memory[:13] + store.memory[64:104])
    try:
        transport.put([item])
        assert store.objects[item.key] == expected
        put = next(call for call in store.calls if call[0] == "put")
        assert put[2:4] == ([[BASE, BASE + 64]], [[13, 40]])
        assert item.segments == segments and len(item.segments) == 4
        store.memory[:] = b"\xff" * len(store.memory)
        transport.get([item])
        get = next(call for call in store.calls if call[0] == "get")
        assert get[2:4] == put[2:4]
        assert bytes(store.memory[:13] + store.memory[64:104]) == expected
        assert store.memory[13:64] == b"\xff" * 51
        assert store.memory[104:] == b"\xff" * (len(store.memory) - 104)
    finally:
        transport.close()


def test_coalescing_does_not_cross_adjacent_registration_owners():
    store = FakeStore()
    transport = StrictMemcacheTransport(store, registration_alignment=1)
    transport.register_buffers([(BASE + 16, 32), (BASE, 16)])
    item = TransferObject("key", (ByteSegment(BASE, 16), ByteSegment(BASE + 16, 7),
                                  ByteSegment(BASE + 23, 9)))
    try:
        transport.put([item])
        put = next(call for call in store.calls if call[0] == "put")
        assert put[2:4] == ([[BASE, BASE + 16]], [[16, 16]])
        expected = bytes(store.memory[:32])
        store.memory[:] = b"\xff" * len(store.memory)
        transport.get([item])
        assert bytes(store.memory[:32]) == expected
        assert store.memory[32:] == b"\xff" * (len(store.memory) - 32)
    finally:
        transport.close()


@pytest.mark.parametrize("parts,expected", [
    (((0, 7), (12, 3), (15, 2)), ((0, 7), (12, 5))),
    (((8, 3), (0, 8)), ((8, 3), (0, 8))),
    (((0, 8), (4, 8)), ((0, 8), (4, 8))),
    (((0, 7), (7, 2)), ((0, 9),)),
])
def test_coalescing_preserves_payload_order_gaps_overlap_and_short_tails(parts, expected):
    store, transport = setup_transport()
    item = TransferObject("key", tuple(ByteSegment(BASE + start, size) for start, size in parts))
    payload = b"".join(bytes(store.memory[start:start + size]) for start, size in parts)
    try:
        transport.put([item])
        put = next(call for call in store.calls if call[0] == "put")
        assert put[2] == [[BASE + start for start, _ in expected]]
        assert put[3] == [[size for _, size in expected]]
        assert store.objects["key"] == payload
        store.memory[:] = b"\xff" * len(store.memory)
        transport.get([item])
        assert b"".join(bytes(store.memory[start:start + size]) for start, size in parts) == payload
        touched = {index for start, size in parts for index in range(start, start + size)}
        assert all(value == 255 for index, value in enumerate(store.memory) if index not in touched)
    finally:
        transport.close()


def test_contiguous_objects_are_never_coalesced_across_keys():
    store, transport = setup_transport()
    try:
        transport.put([obj("one", length=8), obj("two", offset=8, length=8)])
        put = next(call for call in store.calls if call[0] == "put")
        assert put[1:4] == (("one", "two"), [[BASE], [BASE + 8]], [[8], [8]])
        assert store.objects["one"] == bytes(store.memory[:8])
        assert store.objects["two"] == bytes(store.memory[8:16])
    finally:
        transport.close()


def test_registration_index_refreshes_for_new_owner_and_partial_cleanup():
    store = FakeStore()
    transport = StrictMemcacheTransport(store, registration_alignment=1)
    transport.register_buffers([(BASE + 128, 16)])
    transport.register_buffers([(BASE, 16)])
    try:
        transport.put([obj("new-owner")])
        assert store.objects["new-owner"] == bytes(store.memory[:16])
        assert transport._region_starts == (BASE, BASE + 128)
        unregister = store.unregister_buffer

        def fail_one(address, length):
            unregister(address, length)
            return -7 if address == BASE + 128 else 0

        store.unregister_buffer = fail_one
        with pytest.raises(TransferError, match="cleanup failed"):
            transport.close()
        assert transport.registered_regions == (ByteSegment(BASE + 128, 16),)
        assert transport._region_starts == (BASE + 128,)
        assert transport._region_index == transport.registered_regions
        with pytest.raises(TransferError, match="closed"):
            transport.get([obj("new-owner")])
    finally:
        transport.close()


def test_load_failure_is_not_completion_even_if_sdk_modified_bytes():
    store, transport = setup_transport()
    transport.put([obj()])
    store.overrides["get"] = [-7]
    with pytest.raises(TransferError, match="failed object statuses"):
        transport.get([obj()])
    assert transport.failed


@pytest.mark.parametrize("result", [None, False, -1])
def test_registration_requires_explicit_integer_success(result):
    store = FakeStore()
    store.overrides["register"] = result
    transport = StrictMemcacheTransport(store)
    owner = object()
    with pytest.raises(TransferError):
        transport.register_buffers([(BASE, 1024)], owners=[owner])
    assert transport.registered_regions == ()
    assert transport._owners == [owner]


def test_registration_uses_owning_aligned_region_and_deduplicates_aliases():
    store, transport = setup_transport()
    transport.register_buffers([SimpleNamespace(address=BASE, length=32)])
    assert sum(call[0] == "register" for call in store.calls) == 1
    with pytest.raises(ValueError, match="aligned"):
        transport.register_buffers([(BASE + 1, 32)])
    with pytest.raises(ValueError, match="unregistered"):
        transport.put([obj("bad", offset=len(store.memory) - 4, length=8)])


@pytest.mark.parametrize("offset,length,valid", [
    (0, 32, True), (31, 1, True), (32, 32, True), (128, 32, True),
    (-1, 1, False), (0, 33, False), (31, 2, False), (63, 2, False),
    (64, 1, False), (127, 2, False), (159, 2, False), (160, 1, False),
])
def test_dma_span_checks_preserve_gaps_and_adjacent_owner_boundaries(offset, length, valid):
    store = FakeStore()
    transport = StrictMemcacheTransport(store, registration_alignment=1)
    try:
        # Registration order is not address order; two owners are adjacent.
        transport.register_buffers([(BASE + 128, 32), (BASE, 32), (BASE + 32, 32)])
        item = obj("span", offset=offset, length=length)
        if valid:
            transport.put([item])
            assert store.objects["span"] == bytes(store.memory[offset:offset + length])
        else:
            with pytest.raises(ValueError, match="unregistered"):
                transport.put([item])
            assert not any(call[0] == "put" for call in store.calls)
    finally:
        transport.close()


def test_bounded_batches_keep_arbitrary_objects_whole():
    store, transport = setup_transport(max_batch_objects=2, max_batch_bytes=24)
    objects = [obj(str(i), offset=i * 16, length=8) for i in range(5)]
    transport.put(objects)
    calls = [call for call in store.calls if call[0] == "put"]
    assert [call[1] for call in calls] == [("0", "1"), ("2", "3"), ("4",)]
    with pytest.raises(ValueError, match="exceeds max_batch_bytes"):
        transport.put([obj("too_big", length=25)])


def test_metadata_client_has_no_buffer_or_dma_path():
    store = FakeStore()
    transport = StrictMemcacheTransport(store, scheduler=True)
    assert transport.exists(["missing"]) == (False,)
    with pytest.raises(TransferError, match="scheduler"):
        transport.register_buffers([(BASE, 32)])
    with pytest.raises(TransferError, match="scheduler"):
        transport.put([obj()])


def test_backend_factory_initializes_then_uses_underlying_strict_sdk():
    store = FakeStore()
    order = []
    backend = SimpleNamespace(store=store, ensure_initialized=lambda: order.append("init"))
    transport = StrictMemcacheTransport.from_backend(backend)
    assert order == ["init"]
    assert transport.store is store and transport.backend is backend


def test_native_factory_uses_metadata_only_scheduler_and_explicit_worker_rank(monkeypatch):
    calls = []
    class Backend:
        def __init__(self, parallel_config, **kwargs):
            calls.append((parallel_config, kwargs))
            self.store = FakeStore()
        def ensure_initialized(self):
            calls.append("initialized")
        @classmethod
        def create_scheduler_client(cls, parallel_config):
            return cls(parallel_config, local_rank=0, init_bm=False)
    module_name = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend"
    monkeypatch.setitem(sys.modules, module_name, SimpleNamespace(MemcacheBackend=Backend))
    config = SimpleNamespace(parallel_config="parallel")
    scheduler = StrictMemcacheTransport.from_vllm_config(config, scheduler=True)
    worker = StrictMemcacheTransport.from_vllm_config(config, local_rank=3)
    assert calls == [("parallel", {"local_rank": 0, "init_bm": False}), "initialized", ("parallel", {"local_rank": 3, "lazy_init": False}), "initialized"]
    assert scheduler.scheduler and not worker.scheduler


def descriptor():
    return CheckpointDescriptor("schema+pt+model", "immutable-save-generation", {0: ("r0/full", "r0/gdn"), 1: ("r1/full", "r1/gdn")})


def test_joint_completion_requires_every_participant_and_object():
    store, transport = setup_transport()
    checkpoint = descriptor()
    workers = [CheckpointWorker(transport, participant=rank, synchronize=lambda: None) for rank in range(2)]
    receipts = []
    for rank, worker in enumerate(workers):
        objects = [obj(key, offset=rank * 64 + index * 16) for index, key in enumerate(checkpoint.keys_by_participant[rank])]
        receipts.append(worker.save_checkpoint(checkpoint, objects))
    with pytest.raises(TransferError, match="every TP/group"):
        workers[0].verify_joint_checkpoint(checkpoint, receipts[:1])
    workers[0].verify_joint_checkpoint(checkpoint, receipts)
    del store.objects["r1/gdn"]
    with pytest.raises(MissingObjectsError):
        workers[0].verify_joint_checkpoint(checkpoint, receipts)


def test_duplicate_and_stale_receipts_cannot_publish():
    checkpoint = descriptor()
    r0 = CheckpointReceipt(checkpoint.identity, 0, checkpoint.keys_by_participant[0], 32)
    with pytest.raises(TransferError, match="duplicate"):
        require_joint_receipts(checkpoint, [r0, r0])
    r1 = CheckpointReceipt("previous-generation", 1, checkpoint.keys_by_participant[1], 32)
    with pytest.raises(TransferError, match="stale"):
        require_joint_receipts(checkpoint, [r0, r1])


def test_producer_completion_precedes_save_and_load_validation_precedes_graph():
    store, transport = setup_transport()
    checkpoint = CheckpointDescriptor("ns", "g1", {0: ("full", "gdn")})
    objects = [obj("full"), obj("gdn", offset=32)]
    order = []
    worker = CheckpointWorker(transport, participant=0, synchronize=lambda: order.append("synchronize"))
    original_put = store.batch_put_from_layers
    original_get = store.batch_get_into_layers
    store.batch_put_from_layers = lambda *args: (order.append("put"), original_put(*args))[1]
    store.batch_get_into_layers = lambda *args: (order.append("get"), original_get(*args))[1]
    receipt = worker.save_checkpoint(checkpoint, objects)
    assert order == ["synchronize", "put"]
    order.clear()
    worker.load_checkpoint(
        checkpoint,
        objects,
        verify_publication=lambda value: (order.append("publication"), worker.verify_joint_checkpoint(value, [receipt])),
        after_load=lambda: order.append("validate-state"),
    )
    order.append("graph")
    assert order == ["publication", "synchronize", "get", "validate-state", "graph"]


def test_no_transfer_or_receipt_when_producer_or_publication_fails():
    store, transport = setup_transport()
    checkpoint = CheckpointDescriptor("ns", "g1", {0: ("full",)})
    def fail():
        raise RuntimeError("incomplete state")
    worker = CheckpointWorker(transport, participant=0, synchronize=fail)
    with pytest.raises(RuntimeError, match="incomplete"):
        worker.save_checkpoint(checkpoint, [obj("full")])
    with pytest.raises(RuntimeError, match="incomplete"):
        worker.load_checkpoint(checkpoint, [obj("full")], verify_publication=lambda _: fail())
    assert all(call[0] not in ("get", "put") for call in store.calls)


def test_no_partial_local_component_or_rank_restore():
    store, transport = setup_transport()
    checkpoint = descriptor()
    worker = CheckpointWorker(transport, participant=0, synchronize=lambda: None)
    with pytest.raises(TransferError, match="cover"):
        worker.save_checkpoint(checkpoint, [obj("r0/full")])
    store.objects.update({"r0/full": b"a" * 16, "r0/gdn": b"b" * 16, "r1/full": b"c" * 16})
    with pytest.raises(MissingObjectsError) as exc:
        worker.load_checkpoint(checkpoint, [obj("r0/full"), obj("r0/gdn", offset=32)], verify_publication=lambda _: None)
    assert exc.value.keys == ("r1/gdn",)
    assert all(call[0] != "get" for call in store.calls)


def test_sdk_deadline_kills_worker_even_when_c_call_holds_python_gil():
    script = textwrap.dedent("""
        import ctypes
        from oscar_ascend.offload.transport import StrictMemcacheTransport
        class BlockingStore:
            def batch_is_exist(self, keys):
                ctypes.PyDLL(None).sleep(3)
                raise AssertionError("SDK timeout returned to the caller")
        StrictMemcacheTransport(BlockingStore(), timeout_seconds=0.08).exists(["key"])
        raise AssertionError("worker continued after an in-flight DMA timeout")
    """)
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=5)
    assert completed.returncode == -signal.SIGKILL
    assert "FATAL: OSCAR memcache batch_is_exist" in completed.stderr
    assert "in-flight DMA buffers" in completed.stderr


def test_device_probe_reports_failure_with_no_npu_and_keeps_real_exit_code(tmp_path, monkeypatch, capsys):
    from tools import probe_kv_offload
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    output = tmp_path / "probe.json"
    assert probe_kv_offload.main(["--output-json", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert "requires a real NPU" in report["error"]
    assert "Traceback" in capsys.readouterr().err


def test_close_unregisters_before_releasing_owners_and_closes_sdk():
    store = FakeStore()
    transport = StrictMemcacheTransport(store)
    owner = object()
    transport.register_buffers([(BASE, 32)], owners=[owner])
    unregister = store.unregister_buffer
    def checked_unregister(address, length):
        assert transport._owners == [owner]
        return unregister(address, length)
    store.unregister_buffer = checked_unregister
    transport.close()
    assert transport._owners == []
    assert [call[0] for call in store.calls] == ["register", "unregister", "close"]
    with pytest.raises(TransferError, match="closed"):
        transport.exists(["key"])


def test_close_reports_unregister_failure_keeps_owner_and_still_closes_sdk():
    store, transport = setup_transport()
    owner = object()
    transport._owners.append(owner)
    store.overrides["unregister"] = -7
    with pytest.raises(TransferError, match="cleanup failed"):
        transport.close()
    assert transport._owners == [owner]
    assert store.calls[-1] == ("close",)


def test_cleanup_can_run_after_poisoned_sdk_call():
    store, transport = setup_transport()
    store.overrides["put"] = [-1]
    with pytest.raises(TransferError):
        transport.put([obj()])
    transport.close()
    assert [call[0] for call in store.calls][-2:] == ["unregister", "close"]


def test_private_partial_put_rollback_keeps_original_poison_and_is_idempotent():
    store, transport = setup_transport()
    store.overrides["put"] = [0, -1]
    objects = [obj("private/one"), obj("private/two", offset=32)]
    with pytest.raises(TransferError):
        transport.put(objects)
    original_failure = transport._failure
    assert transport.rollback_private_write([item.key for item in objects]) == (True, True)
    assert transport.rollback_private_write([item.key for item in objects]) == (False, False)
    assert transport._failure is original_failure
    assert transport.failed
    with pytest.raises(TransferError, match="earlier SDK failure"):
        transport.exists(["private/one"])


def test_private_rollback_cannot_expand_last_write_scope_or_mask_failure():
    store, transport = setup_transport()
    store.overrides["put"] = [-1]
    with pytest.raises(TransferError):
        transport.put([obj("private")])
    original_failure = transport._failure
    with pytest.raises(TransferError, match="outside the most recent"):
        transport.rollback_private_write(["another-engine"])
    store.overrides["remove_batch"] = [-3104]
    with pytest.raises(TransferError, match="failed object statuses"):
        transport.rollback_private_write(["private"])
    assert transport._failure is original_failure
    assert transport.failed


@pytest.mark.parametrize("rank", [0, 1])
def test_probe_remote_cleanup_waits_all_readers_and_removes_only_owned_objects(rank):
    from tools.probe_kv_offload import _cleanup_remote_objects
    store, transport = setup_transport()
    prefix = "isolated-probe"
    own = [f"{prefix}/r{rank}/full", f"{prefix}/r{rank}/gdn",
           f"{prefix}/incomplete/r{rank}/full", f"{prefix}/host/r{rank}"]
    if rank == 0:
        own.append(f"{prefix}/manifest")
    protected = {f"{prefix}/r{1-rank}/full": b"other-rank", "another-run/object": b"unrelated"}
    if rank != 0:
        protected[f"{prefix}/manifest"] = b"rank-zero-owned"
    store.objects.update(protected)
    store.objects.update({key: b"probe" for key in own})
    order = []
    remove = store.remove_batch
    def checked_remove(keys):
        assert order == ["all-rank-read-complete"]
        order.append("delete")
        return remove(keys)
    store.remove_batch = checked_remove
    receipt = _cleanup_remote_objects(
        transport, prefix=prefix, rank=rank, world_size=2,
        barrier=lambda: order.append("all-rank-read-complete"),
    )
    assert receipt == {"remote_object_cleanup": "passed", "remote_object_cleanup_count": len(own)}
    assert store.objects == protected
    assert order == ["all-rank-read-complete", "delete"]
    transport.close()


@pytest.mark.parametrize("failure", ["missing", "delete-error", "still-exists"])
def test_probe_remote_cleanup_cannot_report_success_after_failed_deletion(failure):
    from tools.probe_kv_offload import _cleanup_remote_objects
    store, transport = setup_transport()
    prefix = "isolated-probe"
    keys = [f"{prefix}/r0/full", f"{prefix}/r0/gdn",
            f"{prefix}/incomplete/r0/full", f"{prefix}/host/r0", f"{prefix}/manifest"]
    store.objects.update({key: b"probe" for key in keys})
    if failure == "missing":
        del store.objects[keys[0]]
    elif failure == "delete-error":
        store.overrides["remove_batch"] = [-1] * len(keys)
    else:
        store.overrides["exists"] = [1] * len(keys)
    with pytest.raises((TransferError, RuntimeError)):
        _cleanup_remote_objects(transport, prefix=prefix, rank=0, world_size=1)
    transport.close()


def test_get_allows_metadata_and_put_between_serial_sdk_batches(monkeypatch):
    watchdog = InlineSerialWatchdog()
    store, transport = setup_transport(max_batch_objects=1, _watchdog=watchdog)
    items = [obj("read-a"), obj("read-b", offset=32)]
    store.objects.update({"read-a": b"a" * 16, "read-b": b"b" * 16})
    expected_saved = bytes(store.memory[64:80])
    entered, release = pause_between_batches(monkeypatch, transport, [item.key for item in items])
    reader = ThreadCall(lambda: transport.get(items))
    control = None
    try:
        assert entered.wait(5)
        assert transport._active_reads == 1
        # This finishes while the read is paused. No timing/fairness claim:
        # the gate explicitly makes the point between data batches available.
        control = ThreadCall(lambda: (transport.exists(["read-a"]), transport.put([obj("saved", offset=64)])))
        control.join()
        assert control.errors == []
        assert control.result[0] == (True,)
        assert store.objects["saved"] == expected_saved
        assert [call[0] for call in store.calls] == ["register", "info", "info", "get", "exists", "put"]
    finally:
        release.set()
        reader.join()
        if control is not None:
            control.join()
        transport.close()
    assert reader.errors == []
    assert reader.result.keys == ("read-a", "read-b")
    assert store.memory[:16] == b"a" * 16
    assert store.memory[32:48] == b"b" * 16
    assert transport._active_reads == 0
    assert transport._read_owner is None
    assert watchdog.close_calls == 1


@pytest.mark.parametrize("closer_count", [1, 2])
def test_close_waits_for_read_lease_and_rejects_new_reads_and_registration(monkeypatch, closer_count):
    watchdog = InlineSerialWatchdog()
    store, transport = setup_transport(max_batch_objects=1, _watchdog=watchdog)
    items = [obj("a"), obj("b", offset=32)]
    store.objects.update({"a": b"a" * 16, "b": b"b" * 16})
    entered, release = pause_between_batches(monkeypatch, transport, ["a", "b"])
    waiting = observe_close_wait(monkeypatch, transport, count=closer_count)
    reader = ThreadCall(lambda: transport.get(items))
    closers, queued = [], None
    try:
        assert entered.wait(5)
        closers = [ThreadCall(transport.close) for _ in range(closer_count)]
        assert waiting.wait(5)
        with transport._lock:
            assert transport._closing
            assert transport._active_reads == 1
            assert not transport._closed
            assert not any(call[0] in ("unregister", "close") for call in store.calls)
        owners = list(transport._owners)
        with pytest.raises(TransferError, match="closing"):
            transport.register_buffers([(BASE * 2, 16)], owners=[object()])
        assert transport._owners == owners
        queued = ThreadCall(lambda: transport.get([obj("a")]))
    finally:
        release.set()
        reader.join()
        for closer in closers:
            closer.join()
        if queued is not None:
            queued.join()
        transport.close()
    assert reader.errors == []
    assert all(not closer.errors for closer in closers)
    assert len(queued.errors) == 1
    assert isinstance(queued.errors[0], TransferError)
    assert "clos" in str(queued.errors[0])
    assert [call[0] for call in store.calls] == ["register", "info", "info", "get", "get", "unregister", "close"]
    assert watchdog.close_calls == 1
    assert transport._active_reads == 0


@pytest.mark.parametrize("reentrant_operation", ["close", "get"])
def test_read_thread_reentrant_close_or_get_fails_without_deadlock(monkeypatch, reentrant_operation):
    store, transport = setup_transport(_watchdog=InlineSerialWatchdog())
    store.objects["a"] = b"a" * 16

    def reenter(*args):
        if reentrant_operation == "close":
            transport.close()
        else:
            transport.get([obj("a")])
        pytest.fail("reentrant operation unexpectedly returned")

    monkeypatch.setattr(store, "batch_get_into_layers", reenter)
    reader = ThreadCall(lambda: transport.get([obj("a")]))
    reader.join()
    assert len(reader.errors) == 1
    assert isinstance(reader.errors[0], TransferError)
    assert "active read thread" in str(reader.errors[0])
    assert isinstance(reader.errors[0].__cause__, TransferError)
    assert not transport._closing
    assert transport.failed
    assert transport._active_reads == 0
    assert transport._read_owner is None
    assert not any(call[0] == "unregister" for call in store.calls)
    transport.close()


@pytest.mark.parametrize("failure", ["second_get", "interleaved_metadata"])
def test_failed_late_read_releases_lease_before_close_and_never_returns_receipt(monkeypatch, failure):
    store, transport = setup_transport(max_batch_objects=1, _watchdog=InlineSerialWatchdog())
    items = [obj("a"), obj("b", offset=32)]
    store.objects.update({"a": b"a" * 16, "b": b"b" * 16})
    entered, release = pause_between_batches(monkeypatch, transport, ["a", "b"])
    waiting = observe_close_wait(monkeypatch, transport)
    reader = ThreadCall(lambda: transport.get(items))
    closer = None
    try:
        assert entered.wait(5)
        if failure == "second_get":
            store.overrides["get"] = [-1]
        else:
            store.overrides["exists"] = [-1]
            with pytest.raises(TransferError, match="invalid/failed"):
                transport.exists(["a"])
        closer = ThreadCall(transport.close)
        assert waiting.wait(5)
        assert not any(call[0] == "unregister" for call in store.calls)
    finally:
        release.set()
        reader.join()
        if closer is not None:
            closer.join()
        transport.close()
    assert reader.result is None
    assert len(reader.errors) == 1
    assert isinstance(reader.errors[0], TransferError)
    assert transport.failed
    assert transport._active_reads == 0
    assert transport._read_owner is None
    assert closer.errors == []
    assert [call[0] for call in store.calls][-2:] == ["unregister", "close"]
    assert sum(call[0] == "get" for call in store.calls) == (2 if failure == "second_get" else 1)


def test_two_gets_are_serialized_for_their_entire_reads(monkeypatch):
    store, transport = setup_transport(max_batch_objects=1, _watchdog=InlineSerialWatchdog())
    store.objects.update({key: key.encode() * 16 for key in ("a", "b", "c")})
    entered, release = pause_between_batches(monkeypatch, transport, ["a", "b"])
    reader = ThreadCall(lambda: transport.get([obj("a"), obj("b", offset=32)]))
    second = None
    try:
        assert entered.wait(5)
        assert not transport._read_lock.acquire(blocking=False)
        second = ThreadCall(lambda: transport.get([obj("c", offset=64)]))
    finally:
        release.set()
        reader.join()
        if second is not None:
            second.join()
        transport.close()
    assert reader.errors == second.errors == []
    assert [call[:2] for call in store.calls if call[0] in ("info", "get")] == [
        ("info", ("a",)), ("info", ("b",)), ("get", ("a",)), ("get", ("b",)),
        ("info", ("c",)), ("get", ("c",)),
    ]


def test_put_keeps_operation_lock_across_batches(monkeypatch):
    store, transport = setup_transport(max_batch_objects=1, _watchdog=InlineSerialWatchdog())
    entered, release = pause_between_batches(monkeypatch, transport, ["a", "b"])
    writer = ThreadCall(lambda: transport.put([obj("a"), obj("b", offset=32)]))
    try:
        assert entered.wait(5)
        acquired = transport._lock.acquire(blocking=False)
        if acquired:
            transport._lock.release()
        assert not acquired
        assert transport._active_reads == 0
    finally:
        release.set()
        writer.join()
        transport.close()
    assert writer.errors == []
