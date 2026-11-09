"""Archive #37-49/#127/#128/#157/#163/#171: native completion boundaries.

Execute pinned mixin, TP aggregator, empty scheduler-output processing and
EngineCore.step bodies unchanged. These are local host integration tests;
they do not launch vLLM/NPU or claim device/graph/service acceptance.
"""
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import copy
from dataclasses import dataclass, field
import logging
from types import MethodType, SimpleNamespace as NS

import pytest

from test_offload_native_bridge import (
    bridge_module, connector_pair, scheduler_for, scheduled, native_postprocess,
)
from test_offload_cleanup import saved_tp_swap
from test_runtime_native import definitions


def native_completion(monkeypatch, worker):
    outputs = definitions("references/vllm/vllm/v1/outputs.py", "offload_native_outputs",
        {"KVConnectorOutput", "ModelRunnerOutput"},
        {"dataclass": dataclass, "field": field, "copy": copy}, monkeypatch)
    outputs.EMPTY_MODEL_RUNNER_OUTPUT = outputs.ModelRunnerOutput([], {})
    worker.bind_connector_metadata = lambda metadata: setattr(worker, "metadata", metadata)
    worker.clear_connector_metadata = lambda: setattr(worker, "metadata", None)
    worker.get_block_ids_with_load_errors = lambda: set()
    worker.get_kv_connector_stats = lambda: None
    worker.get_kv_connector_kv_cache_events = lambda: None
    native = definitions("references/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py",
        "offload_native_mixin", {"KVConnectorModelRunnerMixin"},
        {"contextmanager": contextmanager, "nullcontext": nullcontext,
         "KVConnectorBase": type(worker), "get_kv_transfer_group": lambda: worker,
         "has_kv_transfer_group": lambda: True, "get_forward_context": lambda: None,
         "set_forward_context": lambda *args: nullcontext(),
         "KVConnectorOutput": outputs.KVConnectorOutput,
         "ModelRunnerOutput": outputs.ModelRunnerOutput}, monkeypatch)
    aggregator = definitions("references/vllm/vllm/distributed/kv_transfer/kv_connector/utils.py",
        "offload_native_aggregator", {"KVOutputAggregator"},
        {"KVConnectorOutput": outputs.KVConnectorOutput,
         "logger": logging.getLogger(__name__)}, monkeypatch)
    return native.KVConnectorModelRunnerMixin, aggregator.KVOutputAggregator, outputs


def native_empty_engine_step(monkeypatch, sender, scheduler_output, worker_output):
    scheduler = sender.scheduler_state.scheduler
    native = definitions("references/vllm/vllm/v1/core/sched/scheduler.py", "offload_empty_scheduler", set(),
        {"defaultdict": defaultdict}, monkeypatch,
        methods=[("Scheduler", "update_from_output"),
                 ("Scheduler", "_update_from_kv_xfer_finished"),
                 ("Scheduler", "has_finished_requests")])
    interface = definitions("references/vllm/vllm/v1/core/sched/interface.py", "offload_scheduler_interface", set(),
        {}, monkeypatch, methods=[("SchedulerInterface", "has_requests")])
    scheduler.has_finished_requests = MethodType(native.has_finished_requests, scheduler)
    scheduler.has_unfinished_requests = lambda: bool(scheduler.requests)
    scheduler.has_requests = MethodType(interface.has_requests, scheduler)
    scheduler._update_from_kv_xfer_finished = MethodType(native._update_from_kv_xfer_finished, scheduler)
    scheduler.update_from_output = MethodType(native.update_from_output, scheduler)
    scheduler.perf_metrics = None
    scheduler.finished_req_ids_dict = {}
    scheduler.make_stats = lambda *args: None
    scheduler.kv_cache_manager.take_events = lambda: None
    sender.take_events = lambda: None
    scheduler.get_grammar_bitmask = lambda output: None
    scheduler.schedule = lambda: scheduler_output
    core = definitions("references/vllm/vllm/v1/engine/core.py", "offload_empty_engine", set(),
        {}, monkeypatch, methods=[("EngineCore", "step")])
    engine = NS(scheduler=scheduler,
        model_executor=NS(execute_model=lambda output, non_block: NS(result=lambda: worker_output)),
        log_error_detail=lambda output: nullcontext(),
        log_iteration_details=lambda output: nullcontext(), _process_aborts_queue=lambda: None)
    return core.step(engine)


def test_no_forward_tp_metadata_survives_real_engine_and_scheduler_empty_step(bridge_module, monkeypatch):
    sender, worker, runner, scheduler, request, pages, snapshot, receipts, store = saved_tp_swap(bridge_module, monkeypatch)
    mixin, aggregator, outputs = native_completion(monkeypatch, worker)
    empty = NS(num_scheduled_tokens={}, scheduled_spec_decode_tokens={}, finished_req_ids=set(),
               total_num_scheduled_tokens=0)
    empty.kv_connector_metadata = sender.build_connector_meta(empty)
    results = []
    for rank, metadata in enumerate(receipts):
        worker.rank = rank
        worker._acks.extend(metadata.acknowledgements)
        results.append(mixin.kv_connector_no_forward(empty, None))
    combined = aggregator(2).aggregate(results)
    assert len(combined.kv_connector_output.kv_connector_worker_meta.acknowledgements) == 2
    assert native_empty_engine_step(monkeypatch, sender, empty, combined) == ({}, False)
    assert sender.get_num_new_matched_tokens(request, 0) == (8, False)


def test_postprocess_prefix_receipt_flushes_in_final_finished_request_empty_step(bridge_module, monkeypatch):
    sender, worker, runner, tensors, group, store = connector_pair(bridge_module, monkeypatch, "align")
    scheduler, request, pages = scheduler_for(sender, "align", tokens=12, prompt=12)
    scheduler._oscar_prefix_checkpoint_policy = NS(alignment_tokens=4, terminal_position=lambda n: 8)
    request.num_computed_tokens = 4
    runner.num_computed_tokens[0] = 4
    runner.mamba_state_idx["r"] = 3
    runner.requests["r"] = NS(block_ids=pages["r"], prev_num_draft_len=0)
    output = scheduled(4, ())
    output.kv_connector_metadata = sender.build_connector_meta(output)
    mixin, aggregator, outputs = native_completion(monkeypatch, worker)
    ordering = []
    # A queued MTP producer is deliberately invisible to the postprocess
    # stream until its own event is waited. Reading earlier saves stale bytes.
    def producer_complete():
        ordering.append("draft_event_wait")
        tensors[0][0].fill_(177)
    def producer_event():
        ordering.append("draft_event_record")
        return NS(synchronize=producer_complete)
    worker._record_producer_event = producer_event
    worker._synchronize = lambda: ordering.append("postprocess_stream_sync")
    original_put = worker.transport.put
    def checked_put(objects):
        ordering.append("store_put")
        assert ordering == ["target_forward", "draft_forward", "draft_event_record",
                            "gdn_postprocess", "draft_event_wait", "postprocess_stream_sync", "store_put"]
        return original_put(objects)
    monkeypatch.setattr(worker.transport, "put", checked_put)
    # The real target forward context builds metadata BEFORE native MTP
    # postprocessing. Prefix bytes must not be advertised in that earlier frame.
    with mixin.maybe_get_kv_connector_output(output, defer_finalize=True) as early:
        ordering.append("target_forward")
    assert not early.kv_connector_worker_meta.prefixes
    assert worker._prefix_producer_event is None
    ordering.append("draft_forward")
    mixin.finalize_kv_connector()
    native_postprocess(monkeypatch, runner, group, pages["r"][1], 1, base=4, count=4, spec=0)
    ordering.append("gdn_postprocess")
    worker.observe_and_save(runner, None, output)
    assert worker._prefix_acks and store.objects
    sender.request_finished_all_groups(request, pages["r"])
    scheduler.requests.clear()
    scheduler.finished_req_ids = {"r"}
    # Native has_requests includes finished IDs even when no runnable request
    # exists, so this last metadata frame is executed and returned to scheduler.
    empty = NS(num_scheduled_tokens={}, scheduled_spec_decode_tokens={}, finished_req_ids={"r"},
               total_num_scheduled_tokens=0)
    empty.kv_connector_metadata = sender.build_connector_meta(empty)
    result = mixin.kv_connector_no_forward(empty, None)
    assert len(result.kv_connector_output.kv_connector_worker_meta.prefixes) == 1
    assert not worker._prefix_acks and "r" not in worker.tracker._records
    assert native_empty_engine_step(monkeypatch, sender, empty, aggregator(1).aggregate([result])) == ({}, False)
    replacement = NS(**vars(request))
    replacement.num_computed_tokens = 0
    scheduler.requests["r"] = replacement
    assert sender.get_num_new_matched_tokens(replacement, 0) == (8, False)


def test_normal_decode_finalization_does_not_add_a_producer_event_or_sync(bridge_module, monkeypatch):
    sender, worker, runner, tensors, group, store = connector_pair(bridge_module, monkeypatch, "none")
    scheduler_for(sender, "none", tokens=10)
    output = scheduled()
    output.kv_connector_metadata = sender.build_connector_meta(output)
    mixin, aggregator, outputs = native_completion(monkeypatch, worker)
    def unexpected():
        pytest.fail("ordinary decode must not create/wait a prefix producer fence")
    worker._record_producer_event = unexpected
    worker._synchronize = unexpected
    with mixin.maybe_get_kv_connector_output(output, defer_finalize=True):
        pass
    mixin.finalize_kv_connector()
    assert worker._prefix_producer_event is None


def test_native_tp_handshake_matches_every_registered_rank(bridge_module, monkeypatch):
    sender, worker, runner, tensors, group, store = connector_pair(bridge_module, monkeypatch, "none")
    sender.world_size = worker.world_size = 2
    rank = [0]
    native = definitions("references/vllm-ascend/vllm_ascend/worker/worker.py", "offload_native_handshake", set(),
        {"has_kv_transfer_group": lambda: True, "get_kv_transfer_group": lambda: worker,
         "get_tp_group": lambda: NS(rank_in_group=rank[0]),
         "get_pp_group": lambda: NS(rank_in_group=0),
         "get_pcp_group": lambda: NS(world_size=1)}, monkeypatch,
        methods=[("NPUWorker", "get_kv_connector_handshake_metadata")])
    metadata = {}
    for value in range(2):
        rank[0] = worker.rank = value
        metadata.update(native.get_kv_connector_handshake_metadata(None))
    sender.set_xfer_handshake_metadata_pp_aware(metadata)
    assert sender.namespace == worker.namespace and sender.geometry == worker.geometry
