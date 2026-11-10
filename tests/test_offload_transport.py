"""Host SDK-contract tests, not NPU/DMA evidence.

Archive #34/#36/#37--49/#127/#128/#157/#163/#171: fixed buffers, complete
physical/window/GDN bytes, all-participant completion, and strict errors.
"""

from dataclasses import dataclass
import json
import signal
import subprocess
import sys
import textwrap
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
