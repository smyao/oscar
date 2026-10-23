"""Archive #37-49/#127/#128/#148: real host commit/free, not device evidence.

AST-loaded native Scheduler/AsyncScheduler, KVCacheManager, hybrid commit,
Mamba/FULL managers and BlockPool execute together. Only requests, sampled
token transport, configuration initialization and event envelopes are fixtures.
No GDN payload, NPU, graph, model quality, throughput or pressure is simulated.
"""
from collections import defaultdict
import contextlib
from dataclasses import dataclass, replace
from types import ModuleType, SimpleNamespace as NS
from typing import overload
import sys

import pytest

from oscar_ascend import plugin
from test_bounded_whole_prefill_native import FixtureBase, Request, STATUS, native_class
from test_prefix_caching import _coordinator_with_hashing
from test_runtime_native import definitions, native_groups, native_specs


class LifecycleRequest(Request):
    def __init__(self, key, prompt=32768, max_tokens=64, changed_after=None):
        super().__init__(key, prompt=prompt)
        self.max_tokens = max_tokens
        self.skip_reading_prefix_cache = False
        self.block_hashes = [i.to_bytes(8, 'big') for i in range((prompt + max_tokens + 16) // 768)]
        if changed_after is not None:
            for i in range(changed_after // 768, len(self.block_hashes)):
                self.block_hashes[i] += key.encode()
        self._output_token_ids = self.output_token_ids = []
        self.pooling_params = None
        self.sampling_params = NS(min_tokens=0, eos_token_id=-999, stop_token_ids=[],
                                  repetition_detection=None, num_logprobs=None)
        self.resumable = False
        self.client_index = 0
        self.stop_reason = self.trace_headers = self.num_nans_in_logits = None

    def append_output_token_ids(self, token):
        self._output_token_ids.append(token)
        self._all_token_ids.append(token)
        self.num_tokens += 1
        self.num_output_tokens += 1

    def is_finished(self):
        return self.status.startswith('finished')

    def get_finished_reason(self):
        return self.status if self.is_finished() else None

    def take_events(self):
        return []

    def take_prefill_stats(self):
        return None


@pytest.fixture
def lifecycle(monkeypatch):
    plugin.unregister()
    native = native_specs.__wrapped__(monkeypatch)
    interface, ours, _ = native
    convert = ours.transform_native_specs

    def actual_geometry(specs):
        specs = {name: replace(spec, page_size_padded=817152,
                    **({'shapes': ((6, 2560), (12, 128, 128))}
                       if isinstance(spec, interface.MambaSpec) else {}))
                 for name, spec in specs.items()}
        return convert(specs, metadata_bytes=4)

    monkeypatch.setattr(ours, 'transform_native_specs', actual_geometry)
    grouped = native_groups.__wrapped__(native, monkeypatch, NS(param=True))
    cache = _coordinator_with_hashing(grouped, monkeypatch)
    managers = sys.modules['vllm.v1.core.single_type_kv_cache_manager']
    cross = definitions('references/vllm/vllm/v1/core/single_type_kv_cache_manager.py',
        'prefix_lifecycle_cross_type', {'CrossAttentionManager'},
        {'SingleTypeKVCacheManager': managers.SingleTypeKVCacheManager}, monkeypatch)
    methods = ('get_num_blocks_to_allocate', 'allocate_new_computed_blocks',
               'allocate_new_blocks', 'get_num_common_prefix_blocks',
               'remove_skipped_blocks', 'get_blocks', 'new_step_starts')
    inherited = definitions('references/vllm/vllm/v1/core/kv_cache_coordinator.py',
        'prefix_lifecycle_coordinator', set(), {'CrossAttentionManager': cross.CrossAttentionManager},
        monkeypatch, methods=[('KVCacheCoordinator', name) for name in methods])
    for name in methods:
        setattr(type(cache), name, getattr(inherited, name))
    # The production lease imports native hash helpers. Export the identical
    # AST-loaded functions already used by the real pool and hash adapter.
    utils = ModuleType('vllm.v1.core.kv_cache_utils')
    utils.BlockHashListWithBlockSize = grouped[3].BlockHashListWithBlockSize
    utils.make_block_hash_with_group_id = type(cache.block_pool).__init__.__globals__['make_block_hash_with_group_id']
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    kv_types = definitions('references/vllm/vllm/v1/core/kv_cache_manager.py',
        'prefix_lifecycle_kv_manager', {'KVCacheBlocks', 'KVCacheManager'},
        {'dataclass': dataclass, 'overload': overload,
         'get_kv_cache_coordinator': lambda **kwargs: cache,
         'get_kv_cache_spec_kind': lambda spec: NS(value='host_event_metadata'),
         'get_kv_cache_spec_sliding_window': lambda spec: None}, monkeypatch)
    kv = kv_types.KVCacheManager(cache.kv_cache_config, 262144, 2304, 768,
        max_num_batched_tokens=16384, use_eagle=True)
    for name in ('FINISHED_STOPPED', 'FINISHED_LENGTH_CAPPED', 'FINISHED_ABORTED'):
        monkeypatch.setattr(STATUS, name, name.lower(), raising=False)
    monkeypatch.setattr(STATUS, 'is_finished', lambda status: status.startswith('finished'), raising=False)
    helper = definitions('references/vllm/vllm/v1/core/sched/utils.py',
        'prefix_lifecycle_stop', {'check_stop', 'remove_all'},
        {'RequestStatus': STATUS, 'contextlib': contextlib}, monkeypatch)
    scheduler_methods = ('update_from_output', '_update_request_with_output',
                         '_handle_stopped_request', '_free_request', '_free_blocks',
                         '_connector_finished', 'finish_requests', 'make_stats',
                         '_mamba_block_aligned_split')
    bodies = definitions('references/vllm/vllm/v1/core/sched/scheduler.py',
        'prefix_lifecycle_scheduler', set(),
        {'RequestStatus': STATUS, 'check_stop': helper.check_stop,
         'remove_all': helper.remove_all, 'defaultdict': defaultdict,
         'EngineCoreOutput': NS, 'EngineCoreOutputs': NS}, monkeypatch,
        methods=[('Scheduler', name) for name in scheduler_methods])
    cls = native_class()
    base = cls.__mro__[1]
    for name in scheduler_methods:
        setattr(base, name, getattr(bodies, name))

    def initialized(self):
        FixtureBase.__init__(self)
        self.cache_config = grouped[4].cache_config
        self.block_size = cache.scheduler_block_size
        self.kv_cache_manager = kv
        self.kv_cache_config = cache.kv_cache_config
        self.need_mamba_block_aligned_split = True
        self.max_num_scheduled_tokens = 16384
        self.perf_metrics = self.finished_req_ids_dict = None
        self.structured_output_manager = NS(should_advance=lambda request: False)
        self.make_spec_decoding_stats = lambda *args, **kwargs: None

    base.__init__ = initialized
    plugin._patch_prefix_checkpoint_scheduler(NS(Scheduler=base))
    scheduler = cls()
    leases = cache._oscar_prefix_checkpoint_leases

    def add(request):
        scheduler.waiting.add_request(request)
        scheduler.requests[request.request_id] = request
        return request

    def complete(frame, accepted=3):
        keys = list(frame.num_scheduled_tokens)
        sampled = []
        for key in keys:
            request = scheduler.requests.get(key)
            if request is None or request.is_prefill_chunk:
                sampled.append([])
            else:
                drafts = frame.scheduled_spec_decode_tokens.get(key, ())
                sampled.append([777] * (1 + min(accepted, len(drafts))))
        frame.num_invalid_spec_tokens = 0
        runner = NS(sampled_token_ids=sampled, logprobs=None, prompt_logprobs_dict={},
            pooler_output=None, num_nans_in_logits=None, kv_connector_output=None,
            cudagraph_stats=None, routed_experts=None, req_ids=keys,
            req_id_to_index={key: i for i, key in enumerate(keys)})
        return scheduler.update_from_output(frame, runner)

    def prefill(request):
        endpoints = []
        while request.num_computed_tokens < request.num_prompt_tokens:
            frame = scheduler.schedule()
            endpoints.append(request.num_computed_tokens)
            complete(frame)
        return endpoints

    yield NS(scheduler=scheduler, kv=kv, cache=cache, leases=leases, add=add,
             complete=complete, prefill=prefill)
    plugin.unregister()


@pytest.mark.parametrize('accepted', [0, 3])
def test_real_async_commit_decode_finish_and_repeat_lookup(lifecycle, accepted):
    h = lifecycle
    request = h.add(LifecycleRequest('seed', max_tokens=33))
    assert h.prefill(request) == [16128, 29952, 31488, 32768]
    blocks = h.leases.blocks_by_request['seed']
    assert len(blocks) == 3 and all(block.ref_cnt >= 1 for block in blocks)
    assert [m.block_size for m in h.cache.single_type_managers] == [2304, 768, 768, 768]
    while not request.is_finished():
        frames = [h.scheduler.schedule(), h.scheduler.schedule()]
        assert request.num_output_placeholders == 8
        for frame in frames:
            h.complete(frame, accepted)
        if not request.is_finished():
            assert h.leases.blocks_by_request['seed'] == blocks
    assert request.num_output_tokens == 33
    assert not h.leases.blocks_by_request
    assert all(block.ref_cnt == 0 for block in blocks)
    repeat = LifecycleRequest('repeat')
    assert h.kv.get_computed_blocks(repeat)[1] == 29952


@pytest.mark.parametrize('operation', ['abort', 'preempt', 'reset'])
def test_native_release_paths_drop_leases_and_pending_outputs_cannot_repin(lifecycle, operation):
    h = lifecycle
    request = h.add(LifecycleRequest('active'))
    h.prefill(request)
    blocks = h.leases.blocks_by_request['active']
    pending = h.scheduler.schedule()
    assert request.num_output_placeholders == 4
    if operation == 'abort':
        h.scheduler.finish_requests('active', STATUS.FINISHED_ABORTED)
    elif operation == 'preempt':
        h.scheduler.running.remove(request)
        h.scheduler._preempt_request(request, 0.)
    else:
        assert h.scheduler.reset_prefix_cache() is False
        assert h.scheduler.reset_prefix_cache(reset_running_requests=True) is True
    assert not h.leases.blocks_by_request
    assert all(block.ref_cnt == 0 for block in blocks)
    h.complete(pending)
    assert not h.leases.blocks_by_request
    if operation == 'reset':
        assert h.kv.get_computed_blocks(LifecycleRequest('repeat'))[1] == 0


@pytest.mark.parametrize('changed_after,hit', [(None, 29952), (18432, 16128), (16128, 0)])
def test_real_full_and_partial_prefix_admission_acquires_only_own_terminal(lifecycle, changed_after, hit):
    h = lifecycle
    seed = h.add(LifecycleRequest('seed', max_tokens=1))
    h.prefill(seed)
    fork = LifecycleRequest('fork', changed_after=changed_after)
    # Native EAGLE requires one extra FULL page after the returned hit.
    # A common prefix ending at 16128 has no earlier retained GDN anchor;
    # retaining only the terminal must not manufacture a partial checkpoint.
    assert h.kv.get_computed_blocks(fork)[1] == hit
    h.add(fork)
    frame = h.scheduler.schedule()
    assert frame.scheduled_new_reqs[0].num_computed_tokens == hit
    h.complete(frame)
    h.prefill(fork)
    assert len(h.leases.blocks_by_request['fork']) == 3
    h.scheduler.finish_requests('fork', STATUS.FINISHED_ABORTED)
    assert not h.leases.blocks_by_request


def test_unregister_releases_live_leases_before_restoring_native_hooks(lifecycle):
    h = lifecycle
    request = h.add(LifecycleRequest('active'))
    h.prefill(request)
    blocks = h.leases.blocks_by_request['active']
    assert h.leases in plugin._prefix_checkpoint_leases
    before = [block.ref_cnt for block in blocks]
    plugin.unregister()
    assert not h.leases.blocks_by_request
    assert [block.ref_cnt for block in blocks] == [count - 1 for count in before]
    h.kv.free(request)
    assert all(block.ref_cnt == 0 for block in blocks)


def test_last_gdn_hash_mismatch_cannot_acquire_a_partial_lease(lifecycle, monkeypatch):
    h = lifecycle
    request = h.add(LifecycleRequest('forming'))
    h.complete(h.scheduler.schedule())
    frame = h.scheduler.schedule()
    assert request.num_computed_tokens == 29952
    blocks = h.leases.blocks_by_request['forming']
    h.leases.release('forming')
    # At this exact scheduling boundary each native request table still owns
    # its just-published terminal. Corrupt only the final group's identity.
    assert all(manager.req_to_blocks['forming'][38] is block
               for manager, block in zip(h.leases.managers, blocks))
    monkeypatch.setattr(blocks[-1], '_block_hash', blocks[-1].block_hash + b'wrong-group')
    before = [block.ref_cnt for block in blocks]
    touches = []
    original_touch = h.cache.block_pool.touch

    def tracked_touch(selected):
        touches.append(tuple(selected))
        return original_touch(selected)

    monkeypatch.setattr(h.cache.block_pool, 'touch', tracked_touch)
    h.leases.observe(request, request.num_computed_tokens)
    assert touches == [] and not h.leases.blocks_by_request
    assert [block.ref_cnt for block in blocks] == before
    assert frame.num_scheduled_tokens['forming'] == 13824


def test_two_active_prefix_forks_own_independent_leases_on_the_same_gdn_pages(lifecycle):
    h = lifecycle
    seed = h.add(LifecycleRequest('seed'))
    h.prefill(seed)
    shared = h.leases.blocks_by_request['seed']
    fork = h.add(LifecycleRequest('fork'))
    h.prefill(fork)
    assert h.leases.blocks_by_request['fork'] == shared
    before = [block.ref_cnt for block in shared]
    assert all(count >= 2 for count in before)
    # Repeated async publication must not add another reference for this owner.
    h.leases.observe(fork, fork.num_computed_tokens)
    assert [block.ref_cnt for block in shared] == before
    h.scheduler.finish_requests('seed', STATUS.FINISHED_ABORTED)
    assert [block.ref_cnt for block in shared] == [count - 1 for count in before]
    assert set(h.leases.blocks_by_request) == {'fork'}
    h.scheduler.finish_requests('fork', STATUS.FINISHED_ABORTED)
    assert all(block.ref_cnt == 0 for block in shared)
    assert not h.leases.blocks_by_request
