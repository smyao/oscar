"""Archive #37-49/#127/#128/#157/#163/#171: preserve native async ownership.

Run pinned native scheduler/block-pool methods on host objects. These checks
cover streaming and reset policy, not NPU transfer or performance acceptance.
"""
from collections import deque
from dataclasses import dataclass
import enum
import logging
import time
from types import SimpleNamespace as NS

import pytest

from oscar_ascend.offload.hooks import resetting_cache
from test_offload_lifecycle import setup_scheduler
from test_runtime_native import definitions


@pytest.fixture
def native(monkeypatch):
    request = definitions(
        "references/vllm/vllm/v1/request.py", "async_policy_native_request",
        {"RequestStatus", "StreamingUpdate"}, {"enum": enum, "dataclass": dataclass}, monkeypatch)
    scheduler = definitions(
        "references/vllm/vllm/v1/core/sched/scheduler.py", "async_policy_native_scheduler", set(),
        {"RequestStatus": request.RequestStatus, "StreamingUpdate": request.StreamingUpdate,
         "time": time}, monkeypatch,
        methods=[("Scheduler", "add_request"), ("Scheduler", "reset_prefix_cache")])
    return NS(request=request, scheduler=scheduler)


def test_native_streaming_update_during_remote_load_only_queues_without_branch(native):
    scheduler, state, request, pages, frees = setup_scheduler(world_size=1)
    request.status = native.request.RequestStatus.WAITING_FOR_REMOTE_KVS
    request.streaming_queue = deque()
    generation, epoch = state.observe_request(request), state.epoch
    before_tokens = list(request.all_token_ids)
    before_pages = pages[request.request_id]
    branches = []

    def branch(session, update):
        branches.append((session, update))
        state.branch(session)

    scheduler._update_request_as_session = branch
    incoming = NS(request_id=request.request_id, resumable=True, mm_features=[],
                  prompt_token_ids=[1001, 1002], max_tokens=8, arrival_time=1.25,
                  sampling_params=object())
    assert native.scheduler.add_request(scheduler, incoming) is None
    update, = request.streaming_queue
    assert isinstance(update, native.request.StreamingUpdate)
    assert update.prompt_token_ids == incoming.prompt_token_ids
    assert update.sampling_params is incoming.sampling_params
    assert branches == []
    assert scheduler.requests[request.request_id] is request
    assert request.status == native.request.RequestStatus.WAITING_FOR_REMOTE_KVS
    assert request.all_token_ids == before_tokens
    assert pages[request.request_id] == before_pages and frees == []
    assert state.epoch == epoch and state.observe_request(request) == generation
    assert state.drain_branches() == ()


def test_native_forced_reset_keeps_remote_pages_and_offload_generation_on_failure(native, monkeypatch):
    blocks = definitions(
        "references/vllm/vllm/v1/core/kv_cache_utils.py", "async_policy_native_blocks",
        {"KVCacheBlock", "FreeKVCacheBlockQueue"}, {"dataclass": dataclass}, monkeypatch)
    pools = definitions(
        "references/vllm/vllm/v1/core/block_pool.py", "async_policy_native_pool",
        {"BlockPool", "BlockHashToBlockMap"},
        {"KVCacheBlock": blocks.KVCacheBlock, "FreeKVCacheBlockQueue": blocks.FreeKVCacheBlockQueue,
         "logger": logging.getLogger("async-native-reset")}, monkeypatch)
    manager = definitions(
        "references/vllm/vllm/v1/core/kv_cache_manager.py", "async_policy_native_manager", set(), {},
        monkeypatch, methods=[("KVCacheManager", "reset_prefix_cache")])
    pool = pools.BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=4)
    destinations = pool.get_new_blocks(2)
    before_free = pool.get_num_free_blocks()
    scheduler, state, request, pages, frees = setup_scheduler(world_size=1)
    request.status = native.request.RequestStatus.WAITING_FOR_REMOTE_KVS
    pages[request.request_id] = (tuple(block.block_id for block in destinations), ())
    scheduler.running = []
    scheduler.prev_step_scheduled_req_ids = {"previous-request"}
    scheduler.connector = NS(_oscar_offload=True, scheduler_state=state)
    scheduler.kv_cache_manager.block_pool = pool
    scheduler.kv_cache_manager.log_stats = False
    scheduler.kv_cache_manager.reset_prefix_cache = lambda: manager.reset_prefix_cache(scheduler.kv_cache_manager)
    generation, epoch = state.observe_request(request), state.epoch

    with pytest.raises(RuntimeError, match="remote KV transfer"):
        with resetting_cache(scheduler, reset_running_requests=True) as complete:
            assert state.suppress_preemption
            complete(native.scheduler.reset_prefix_cache(scheduler, reset_running_requests=True))

    assert state.epoch == epoch and state.observe_request(request) == generation
    assert not state.suppress_preemption
    assert request.status == native.request.RequestStatus.WAITING_FOR_REMOTE_KVS
    assert pages[request.request_id][0] == tuple(block.block_id for block in destinations)
    assert [block.ref_cnt for block in destinations] == [1, 1]
    assert pool.get_num_free_blocks() == before_free and frees == []
    # Failure does not mean native rolled back all its mutations.
    assert scheduler.prev_step_scheduled_req_ids == set()
