"""Archive #37-49/#127/#128/#157/#163/#171: complete host offload bridge.

Actual connector/scheduler/transport methods move real CPU tensor bytes through
an SDK-shaped memory store. Pinned native preemption and accepted-state copy
functions execute unchanged; this is not real memcache/NPU/graph acceptance.
User-supplied 2026-11-10 patch: complete-state read/publication queries use flag 0.
"""
from dataclasses import dataclass
import ctypes
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest
import torch

from oscar_ascend.offload.layout import CacheComponent, LayerCacheExport
from oscar_ascend.offload.plans import object_key, expected_keys
from oscar_ascend.offload.scheduler import OffloadSchedulerState
from oscar_ascend.offload.state import WorkerCheckpointTracker
from oscar_ascend.offload.transport import ByteSegment, StrictMemcacheTransport
from test_runtime_native import definitions


class PointerStore:
    def __init__(self):
        self.objects = {}
        self.put_calls = 0
        self.exists_flags = []

    def register_buffer(self, address, length):
        return 0

    def batch_is_exist(self, keys, flags):
        assert type(flags) is int and flags == 0
        self.exists_flags.append(flags)
        return [int(key in self.objects) for key in keys]

    def batch_get_key_info(self, keys):
        return [NS(size=lambda key=key: len(self.objects[key])) for key in keys]

    def batch_put_from_layers(self, keys, addresses, lengths, direction):
        assert direction == 0
        self.put_calls += 1
        for key, addrs, sizes in zip(keys, addresses, lengths):
            self.objects[key] = b"".join(ctypes.string_at(addr, size) for addr, size in zip(addrs, sizes))
        return [0] * len(keys)

    def batch_get_into_layers(self, keys, addresses, lengths, direction):
        assert direction == 1
        for key, addrs, sizes in zip(keys, addresses, lengths):
            position = 0
            for addr, size in zip(addrs, sizes):
                ctypes.memmove(addr, self.objects[key][position:position+size], size)
                position += size
        return [0] * len(keys)

    def remove_batch(self, keys):
        result = []
        for key in keys:
            result.append(0 if key in self.objects else -3007)
            self.objects.pop(key, None)
        return result


@pytest.fixture
def bridge_module(monkeypatch):
    base = ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.base")
    class Base:
        def _get_connector_metadata(self):
            return self.metadata
    base.KVConnectorBase_V1 = Base
    for name in ("KVConnectorMetadata", "KVConnectorWorkerMetadata", "KVConnectorHandshakeMetadata", "SupportsHMA"):
        setattr(base, name, type(name, (), {}))
    base.KVConnectorRole = NS(SCHEDULER="scheduler", WORKER="worker")
    monkeypatch.setitem(sys.modules, base.__name__, base)
    name = "oscar_ascend.offload.connector"
    path = Path(__file__).resolve().parents[1] / "oscar_ascend/offload/connector.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    class Event:
        def record(self):
            pass
        def synchronize(self):
            pass
    monkeypatch.setattr(torch.cpu, "Event", Event, raising=False)
    return module


def runner_and_exports(monkeypatch, mode):
    funcs = definitions("references/vllm/vllm/model_executor/layers/mamba/mamba_utils.py",
                        "bridge_native_copy", {"MambaCopySpec", "get_conv_copy_spec", "get_temporal_copy_spec"},
                        {"torch": torch, "dataclass": dataclass,
                         "is_conv_state_dim_first": lambda: False}, monkeypatch)
    full_tensors = (torch.arange(16*8, dtype=torch.int32).reshape(16, 2, 4),
                    torch.arange(16*6, dtype=torch.float32).reshape(16, 3, 2),
                    torch.arange(16*6, dtype=torch.float32).reshape(16, 3, 2)+1000,
                    torch.arange(16*3, dtype=torch.int64).reshape(16, 3))
    conv = torch.arange(16*12, dtype=torch.float32).reshape(16, 6, 2)+2000
    temporal = torch.arange(16*4, dtype=torch.float32).reshape(16, 2, 2)+3000
    components = tuple(CacheComponent.from_tensor(name, tensor) for name, tensor in
                       zip(("packed", "window_k", "window_v", "tags"), full_tensors))
    full = LayerCacheExport("full", "oscar_full", 4, components, full_tensors[0])
    gdn = LayerCacheExport("gdn", "gdn", 2 if mode == "align" else 100,
                           (CacheComponent.from_tensor("conv", conv), CacheComponent.from_tensor("ssm", temporal)),
                           (conv, temporal), {"mamba_cache_mode": mode})
    group = NS(layer_names=["gdn"], kv_cache_spec=NS(block_size=gdn.storage_block_size))
    runner = NS(model=NS(get_mamba_state_copy_func=lambda: (funcs.get_conv_copy_spec, funcs.get_temporal_copy_spec)),
                compilation_config=NS(static_forward_context={"gdn": NS(kv_cache=(conv, temporal))}),
                cache_config=NS(mamba_cache_mode=mode), device=torch.device("cpu"),
                input_batch=NS(req_ids=["r"], req_id_to_index={"r": 0},
                               num_accepted_tokens_cpu=np.array([1]), prev_req_id_to_index={}),
                num_accepted_tokens=NS(gpu=torch.ones(1, dtype=torch.int32), np=np.ones(1, dtype=np.int32)),
                num_computed_tokens=torch.tensor([6]), mamba_state_idx={"r": 4},
                requests={})
    monkeypatch.setattr("oscar_ascend.offload.state._mamba_groups", lambda runner: [(1, group)])
    return runner, {"full": full, "gdn": gdn}, (full_tensors, conv, temporal), group


def connector_pair(bridge_module, monkeypatch, mode):
    store = PointerStore()
    manifests = {}
    runner, exports, tensors, group = runner_and_exports(monkeypatch, mode)
    geometry = tuple(exports[name].geometry() for name in sorted(exports))
    def connector(scheduler_role):
        instance = bridge_module.OscarAscendStoreConnector.__new__(bridge_module.OscarAscendStoreConnector)
        instance.world_size = 1
        instance.rank = 0
        instance.session = "test-session"
        instance.namespace = "test-exact-PT-and-layout"
        instance.geometry = geometry
        instance.exports = exports
        instance.runner = runner
        instance.scheduler_state = None
        instance._restores = []
        instance._prefix_matches = {}
        instance._acks = []
        instance._prefix_acks = []
        instance._prefix_publications = {}
        instance._prefix_view = "shared"
        instance._swap_payloads = {}
        instance._load_acks = []
        instance._transfer_session = None
        instance._prefix_producer_event = None
        instance._record_producer_event = lambda: NS(synchronize=lambda: None)
        instance.metadata_server = None
        def remove_swap(plan):
            keys = expected_keys(instance.namespace, instance.world_size, plan, geometry)
            for key in keys:
                store.objects.pop(key, None)
            return len(keys)
        instance.metadata_client = NS(get=manifests.get,
                                      put=lambda key, value: manifests.__setitem__(key, value),
                                      remove_swap=remove_swap)
        instance._group_for_layer = {"full": 0, "gdn": 1}
        instance.tracker = WorkerCheckpointTracker()
        instance.transport = StrictMemcacheTransport(store, scheduler=scheduler_role,
            registration_alignment=1, _watchdog=NS(run=lambda operation, call: call()))
        instance._synchronize = lambda: None
        instance._zero_tail = torch.zeros(128, dtype=torch.uint8)
        if not scheduler_role:
            owned = (*tensors[0], *tensors[1:], instance._zero_tail)
            instance.transport.register_buffers(
                [ByteSegment(t.data_ptr(), t.numel()*t.element_size()) for t in owned], owners=owned)
        return instance
    return connector(True), connector(False), runner, tensors, group, store


def scheduler_for(connector, mode, tokens=7, prompt=3):
    req = NS(request_id="r", num_computed_tokens=6, num_prompt_tokens=prompt,
             all_token_ids=list(range(tokens)), spec_token_ids=[91, 92, 93],
             status="running", num_preemptions=0,
             block_hashes=[bytes([i])*8 for i in range(1, 20)])
    pages = {"r": ((9, 10, 11), (1, 2, 3, 4, 5, 6, 7, 8))}
    scheduler = NS(requests={"r": req}, parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1),
                   cache_config=NS(mamba_cache_mode=mode), connector=connector,
                   kv_cache_manager=NS(get_blocks=lambda rid: pages[rid],
                       free=lambda req: pages.pop(req.request_id), coordinator=NS(hash_block_size=2)),
                   encoder_cache_manager=NS(free=lambda req: None), _inflight_prefills=NS(discard=lambda req: None),
                   log_stats=False, waiting=NS(prepend_request=lambda req: None))
    connector.bind_scheduler(scheduler)
    return scheduler, req, pages


def scheduled(tokens=4, spec=(91, 92, 93)):
    return NS(num_scheduled_tokens={"r": tokens}, scheduled_spec_decode_tokens={"r": list(spec)} if spec else {},
              finished_req_ids=set(), preempted_req_ids=set())


def native_postprocess(monkeypatch, runner, group, ids, accepted, *, base=6, count=4, spec=3):
    """Execute pinned CPU spelling of exactly the native fused-copy decisions."""
    native = definitions("references/vllm-ascend/vllm_ascend/patch/worker/patch_mamba_utils.py",
                         "bridge_native_postprocess",
                         {"_tensor_view_from_data_ptr", "_postprocess_mamba_align_gpu_cpu_fallback"},
                         {"torch": torch}, monkeypatch)
    cpu = torch.tensor([accepted], dtype=torch.int32)
    runner.input_batch.num_accepted_tokens_cpu = cpu.numpy()
    runner.input_batch.block_table = [None, NS(get_numpy_array=lambda: np.array([ids]))]
    context = NS(mamba_state_idx_buf=NS(np=np.array([runner.mamba_state_idx["r"]])),
                 num_scheduled_tokens_buf=NS(np=np.array([count])),
                 num_computed_tokens_buf=NS(np=np.array([base])),
                 num_draft_tokens_buf=NS(np=np.array([spec])), block_size=group.kv_cache_spec.block_size,
                 mamba_group_ids=[1])
    native._postprocess_mamba_align_gpu_cpu_fallback(
        bufs=NS(postprocess_align=context), num_reqs=1,
        num_accepted_tokens_gpu=torch.tensor([accepted], dtype=torch.int32),
        num_accepted_tokens_cpu_tensor=cpu, input_batch=runner.input_batch,
        kv_cache_config=NS(kv_cache_groups=[None, group]),
        forward_context=runner.compilation_config.static_forward_context,
        mamba_state_copy_funcs=runner.model.get_mamba_state_copy_func())
    context.num_accepted_tokens_out = cpu
    runner._get_mamba_bufs = lambda: NS(postprocess_align=context)


@pytest.mark.parametrize("mode", ["none", "align"])
@pytest.mark.parametrize("accepted", [1, 2, 3, 4])
def test_active_native_preemption_to_ack_to_remapped_resume(bridge_module, monkeypatch, mode, accepted):
    sender, worker, runner, tensors, group, store = connector_pair(bridge_module, monkeypatch, mode)
    scheduler, req, pages = scheduler_for(sender, mode)
    out = scheduled()
    out.kv_connector_metadata = sender.build_connector_meta(out)
    source_ids = pages["r"]
    runner.requests["r"] = NS(block_ids=source_ids, prev_num_draft_len=3)
    runner.num_accepted_tokens.gpu[0] = accepted
    if mode == "align":
        native_postprocess(monkeypatch, runner, group, source_ids[1], accepted)
    normalized = int(runner.input_batch.num_accepted_tokens_cpu[0]) if mode == "align" else accepted
    running = runner.mamba_state_idx["r"] if mode == "align" else 0
    conv_expected = tensors[1][source_ids[1][running], normalized-1:].numpy().tobytes()
    conv_expected += bytes(worker.exports["gdn"].components[0].page_bytes - len(conv_expected))
    temporal_expected = tensors[2][source_ids[1][running+normalized-1]].numpy().tobytes()
    worker.observe_and_save(runner, None, out)
    # Native optimistic scheduler state is ahead until its queued output lands.
    req.num_computed_tokens = 10
    snap = sender.scheduler_state.before_preempt(req)
    native = definitions("references/vllm/vllm/v1/core/sched/scheduler.py", "bridge_native_preempt", set(),
                         {"RequestStatus": NS(RUNNING="running", PREEMPTED="preempted")}, monkeypatch,
                         methods=[("Scheduler", "_preempt_request")])
    native._preempt_request(scheduler, req, 0)
    assert req.num_computed_tokens == 0 and "r" not in pages
    empty = NS(num_scheduled_tokens={}, scheduled_spec_decode_tokens={})
    preemption = sender.build_connector_meta(empty)
    worker.handle_preemptions(preemption)
    committed = 6 + accepted
    saved_full = [b"".join(ctypes.string_at(s.address, s.length)
                   for s in worker.exports["full"].page_spans((source_ids[0][i],)))
                  for i in range((committed+3)//4)]
    # Every source page can now be cleared for another request in THIS step.
    for tensor in (*tensors[0], *tensors[1:]):
        tensor.zero_()
    sender.update_connector_output(NS(kv_connector_worker_meta=worker.build_connector_worker_meta()))
    assert sender.get_num_new_matched_tokens(req, 0) == (None, False)
    req.all_token_ids.extend(range(7, committed+1))
    sender.scheduler_state.after_output(out)
    assert sender.get_num_new_matched_tokens(req, 0) == (committed, False)
    destination = ((0, 12, 13), (0, 1, 2, 3, 10, 11, 12, 13))
    pages["r"] = destination
    sender.update_state_after_alloc(req, destination, committed)
    req.num_computed_tokens = committed
    resume = scheduled(1, ())
    resume.kv_connector_metadata = sender.build_connector_meta(resume)
    runner.requests["r"].block_ids = destination
    runner.input_batch.num_accepted_tokens_cpu[:] = 4
    worker.after_native_update_states(runner, resume)
    worker.metadata = resume.kv_connector_metadata
    worker.start_load_kv(None)
    assert runner.input_batch.num_accepted_tokens_cpu[0] == 1
    for i, expected in enumerate(saved_full):
        assert b"".join(ctypes.string_at(s.address, s.length)
                        for s in worker.exports["full"].page_spans((destination[0][i],))) == expected
    # The GDN checkpoint is stored canonically at accepted slot zero, with the
    # conv future tail zero-padded. The next native step will read that state.
    logical = 0 if mode == "none" else (committed-1)//2
    for component, expected in zip(worker.exports["gdn"].components, (conv_expected, temporal_expected)):
        span = component.span(destination[1][logical])
        assert ctypes.string_at(span.address, span.length) == expected
    assert store.put_calls > 0 and snap.identity[0] == "r"
    assert store.exists_flags and set(store.exists_flags) == {0}


def test_chunked_prefix_publish_evict_match_allocate_load(bridge_module, monkeypatch):
    sender, worker, runner, tensors, group, store = connector_pair(bridge_module, monkeypatch, "align")
    scheduler, req, pages = scheduler_for(sender, "align", tokens=12, prompt=12)
    scheduler._oscar_prefix_checkpoint_policy = NS(alignment_tokens=4, terminal_position=lambda n: 8)
    req.num_computed_tokens = 4
    runner.num_computed_tokens[0] = 4
    runner.mamba_state_idx["r"] = 3
    runner.requests["r"] = NS(block_ids=pages["r"], prev_num_draft_len=0)
    out = scheduled(4, ())
    out.kv_connector_metadata = sender.build_connector_meta(out)
    assert len(out.kv_connector_metadata.prefixes) == 1
    sender.world_size = worker.world_size = 2
    native_postprocess(monkeypatch, runner, group, pages["r"][1], 1, base=4, count=4, spec=0)
    worker.metadata = out.kv_connector_metadata
    worker.wait_for_save()
    worker.observe_and_save(runner, None, out)
    # Complete byte objects alone cannot advertise a hit. Publication waits
    # for receipts from every TP participant and a checked immutable manifest.
    assert sender.get_num_new_matched_tokens(req, 0) == (0, False)
    sender.update_connector_output(NS(kv_connector_worker_meta=worker.build_connector_worker_meta()))
    assert sender.get_num_new_matched_tokens(req, 0) == (0, False)
    worker.rank = 1
    worker.wait_for_save()
    worker.observe_and_save(runner, None, out)
    sender.update_connector_output(NS(kv_connector_worker_meta=worker.build_connector_worker_meta()))
    expected = {key: value for key, value in store.objects.items()}
    assert expected
    for tensor in (*tensors[0], *tensors[1:]):
        tensor.zero_()
    # Native local LRU has evicted all pages; the same prompt arrives again.
    sender.request_finished_all_groups(req, pages.pop("r"))
    replacement = NS(**vars(req))
    replacement.num_computed_tokens = 0
    scheduler.requests["r"] = replacement
    assert sender.get_num_new_matched_tokens(replacement, 0) == (8, False)
    destination = ((12, 13, 14), (0, 1, 2, 10, 11, 12, 13))
    pages["r"] = destination
    sender.update_state_after_alloc(replacement, destination, 8)
    replacement.num_computed_tokens = 8
    resume = scheduled(4, ())
    resume.kv_connector_metadata = sender.build_connector_meta(resume)
    worker.metadata = resume.kv_connector_metadata
    worker.start_load_kv(None)
    for obj in worker._objects(resume.kv_connector_metadata.restores[0], loading=True):
        assert b"".join(ctypes.string_at(s.address, s.length) for s in obj.segments) == expected[obj.key]
    # A missing GDN component cannot become a partial FULL-only cache hit.
    gdn_index = next(i for i, layer in enumerate(worker.geometry) if layer["cache_family"] == "gdn")
    store.objects.pop(object_key(worker.namespace, worker.rank,
        resume.kv_connector_metadata.restores[0], gdn_index, "gdn", component=0))
    again = NS(**vars(req))
    again.request_id = "again"
    scheduler.requests[again.request_id] = again
    assert sender.get_num_new_matched_tokens(again, 0) == (0, False)


@pytest.mark.parametrize("mode", ["none", "align"])
def test_streaming_branch_metadata_precedes_same_step_preemption(bridge_module, monkeypatch, mode):
    sender, worker, runner, _, group, store = connector_pair(bridge_module, monkeypatch, mode)
    scheduler, req, pages = scheduler_for(sender, mode, tokens=9)
    out = scheduled()
    out.kv_connector_metadata = sender.build_connector_meta(out)
    runner.requests["r"] = NS(block_ids=pages["r"], prev_num_draft_len=3,
                              prompt_token_ids=[0, 1, 2], output_token_ids=[3, 4, 5, -1, -1, -1])
    runner.num_accepted_tokens.gpu[0] = 2
    if mode == "align":
        native_postprocess(monkeypatch, runner, group, pages["r"][1], 2)
    worker.observe_and_save(runner, None, out)
    sender.scheduler_state.after_output(out)
    old_generation = worker.tracker._records["r"].generation
    req.num_computed_tokens = 8
    req.prompt_token_ids = [0, 1, 2]
    req._all_token_ids = req.all_token_ids
    req._output_token_ids = list(range(3, 9))
    req.update_block_hashes = lambda: None
    req.status = "stream_wait"
    scheduler.num_waiting_for_streaming_input = 1
    native = definitions("references/vllm/vllm/v1/core/sched/scheduler.py", "bridge_native_branch", set(),
        {"RequestStatus": NS(WAITING_FOR_STREAMING_REQ="stream_wait", WAITING="waiting",
                             RUNNING="running", PREEMPTED="preempted")}, monkeypatch,
        methods=[("Scheduler", "_update_request_as_session"), ("Scheduler", "_preempt_request")])
    sender.scheduler_state.branch(req)
    native._update_request_as_session(scheduler, req,
        NS(mm_features=[], prompt_token_ids=[101, 102], arrival_time=1.0, sampling_params=None))
    assert req.all_token_ids == [*range(8), 101, 102]
    req.status = "running"
    snapshot = sender.scheduler_state.before_preempt(req)
    native._preempt_request(scheduler, req, 0)
    empty = NS(num_scheduled_tokens={}, scheduled_spec_decode_tokens={})
    metadata = sender.build_connector_meta(empty)
    assert len(metadata.branches) == 1
    assert snapshot.generation > old_generation
    worker.handle_preemptions(metadata)
    sender.update_connector_output(NS(kv_connector_worker_meta=worker.build_connector_worker_meta()))
    assert sender.get_num_new_matched_tokens(req, 0) == (8, False)
    assert req.all_token_ids[8] == 101  # new input resumes after retained prefix
    assert store.objects  # actual synchronous byte-store happened under new generation
