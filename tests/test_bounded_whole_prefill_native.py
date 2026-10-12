"""Archive #127-129/#137/#140/#142/#151-155: native host scheduling contracts.

Execute the pinned Scheduler schedule/reset/preempt/update methods and actual
AsyncScheduler class bodies. Only external allocator/transport/model objects
are fixtures. No scheduler loop is copied, and no device evidence is claimed.
"""
import ast
from abc import ABC, abstractmethod
from collections import deque
from contextlib import nullcontext
from enum import Enum
import heapq
import itertools
import json
from pathlib import Path
from types import SimpleNamespace as S
import time

import pytest

from test_bounded_whole_prefill import load, target

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "references/vllm/vllm/v1/core/sched"
STATUS = S(RUNNING="running", WAITING="waiting", PREEMPTED="preempted",
           WAITING_FOR_REMOTE_KVS="remote", WAITING_FOR_STREAMING_REQ="streaming")


class Request:
    def __init__(self, key, prompt=65100, computed=0, *, output=0, status=None):
        self.request_id = key
        self.num_prompt_tokens = prompt
        self.num_tokens = prompt + output
        self.num_computed_tokens = computed
        self.num_output_tokens = output
        self.num_output_placeholders = 0
        self.spec_token_ids = []
        self.next_decode_eligible_step = 0
        self.max_tokens = 512
        self.has_encoder_inputs = False
        self.mm_features = []
        self.prefill_stats = None
        self.status = status or (STATUS.RUNNING if computed else STATUS.WAITING)
        self.num_preemptions = 0
        self.async_tokens_to_discard = 0
        self.use_structured_output = False
        self.is_prefill_chunk = computed < prompt
        self.lora_request = None
        # Token payloads do not participate in these host admission tests.
        self.all_token_ids = self._all_token_ids = []
    @property
    def num_tokens_with_spec(self): return self.num_tokens + len(self.spec_token_ids)


def native_queues():
    path = NATIVE / "request_queue.py"
    tree = ast.parse(path.read_text())
    tree.body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                 *[node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]]
    ns = {"ABC": ABC, "abstractmethod": abstractmethod, "deque": deque,
          "Enum": Enum, "heapq": heapq, "Request": Request}
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), ns)
    return ns


QUEUES = native_queues()
Queue, POLICY = QUEUES["FCFSRequestQueue"], QUEUES["SchedulingPolicy"]


class Blocks:
    def get_block_ids(self, **_kwargs): return ([1],)


class KV:
    empty_kv_cache_blocks = Blocks()
    def __init__(self):
        self.fail_once = set()
        self.freed = []
        self.allocations = []
    def new_step_starts(self): pass
    def allocate_slots(self, request, rows, **kwargs):
        self.allocations.append((request.request_id, rows, kwargs))
        if request.request_id in self.fail_once:
            self.fail_once.remove(request.request_id)
            return None
        return Blocks()
    def get_computed_blocks(self, request): return Blocks(), 0
    def get_blocks(self, key): return Blocks()
    def get_num_common_prefix_blocks(self, key): return [0]
    def free(self, request): self.freed.append(request.request_id)
    def reset_prefix_cache(self): return True


class FixtureBase:
    def __init__(self, running=(), waiting=(), *, connector=None, ec_connector=None):
        self.running, self.waiting = list(running), Queue(waiting)
        self.skipped_waiting = Queue()
        self.requests = {req.request_id: req for req in (*running, *waiting)}
        self.scheduler_config = S(enable_chunked_prefill=True, long_prefill_token_threshold=0,
                                  async_scheduling=True)
        self.cache_config = S(enable_prefix_caching=False, mamba_cache_mode="none")
        self.parallel_config = S(pipeline_parallel_size=1)
        self.max_num_scheduled_tokens, self.max_num_running_reqs = 66048, 128
        self.max_model_len, self.num_spec_tokens = 262144, 3
        self.num_lookahead_tokens = 3
        self.current_step = 0
        self._pause_state = "unpaused"
        self.max_num_encoder_input_tokens = 0
        self.need_mamba_block_aligned_split = False  # Native none, not align.
        self.use_eagle = True
        self.is_encoder_decoder = False
        self.use_v2_model_runner = False
        self.use_pp = False
        self.lora_config = None
        self.connector, self.ec_connector = connector, ec_connector
        self.kv_cache_manager = KV()
        self.encoder_cache_manager = S(free=lambda req: None, get_freed_mm_hashes=lambda: [])
        self.kv_cache_config = S(kv_cache_groups=[object()])
        self.scheduler_reserve_full_isl = False
        self.policy = POLICY.FCFS
        self.log_stats = False
        self._inflight_prefills = set()
        self.prev_step_scheduled_req_ids = set()
        self.needs_kv_cache_zeroing = False
        self.finished_req_ids = set()
        self.enable_return_routed_experts = False
    def _is_blocked_waiting_status(self, status): return False


def native_class():
    source = ast.parse((NATIVE / "scheduler.py").read_text())
    original = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    methods = {"schedule", "_preempt_request", "_update_after_schedule", "_make_cached_request_data",
               "_select_waiting_queue_for_scheduling", "reset_prefix_cache"}
    selected = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    assert {n.name for n in selected} == methods
    scheduler = ast.ClassDef(name="Scheduler", bases=[ast.Name(id="FixtureBase", ctx=ast.Load())],
                             keywords=[], body=selected, decorator_list=[])
    async_source = ast.parse((NATIVE / "async_scheduler.py").read_text())
    async_cls = next(n for n in async_source.body if isinstance(n, ast.ClassDef) and n.name == "AsyncScheduler")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                             scheduler, async_cls], type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {"FixtureBase": FixtureBase, "time": time, "itertools": itertools,
          "PauseState": S(PAUSED_ALL="paused", UNPAUSED="unpaused"),
          "RequestStatus": STATUS, "SchedulingPolicy": POLICY,
          "create_request_queue": QUEUES["create_request_queue"],
          "record_function_or_nullcontext": lambda _label: nullcontext(),
          "NewRequestData": S(from_request=lambda req, blocks, *_args: S(
              req_id=req.request_id, num_computed_tokens=req.num_computed_tokens, block_ids=blocks)),
          "CachedRequestData": lambda **kwargs: S(**kwargs),
          "SchedulerOutput": lambda **kwargs: S(**kwargs, pending_structured_output_tokens=False,
                                                has_structured_output_requests=False)}
    exec(compile(module, str(NATIVE / "scheduler.py"), "exec"), ns)
    return ns["AsyncScheduler"]


@pytest.fixture
def scheduler_factory(monkeypatch, tmp_path):
    path = tmp_path / "effective-target.json"
    path.write_text(json.dumps(target()))
    monkeypatch.setenv("OSCAR_TARGET_CONFIG", str(path))
    cls = load(native_class())["BoundedWholePromptAsyncScheduler"]
    return cls


def history_rows(scheduler, output):
    cached = output.scheduled_cached_reqs
    contexts = dict(zip(cached.req_ids, cached.num_computed_tokens))
    contexts.update((req.req_id, req.num_computed_tokens) for req in output.scheduled_new_reqs)
    return sum(count for key, count in output.num_scheduled_tokens.items() if contexts[key] > 0)


def test_real_native_async_admits65100_with_31_q4_and_preserves_placeholders(scheduler_factory):
    running = [Request(f"decode{i}", computed=65100, output=1) for i in range(31)]
    for req in running: req.spec_token_ids = [-1]*3
    fresh = Request("fresh")
    scheduler = scheduler_factory(running, [fresh])
    output = scheduler.schedule()
    assert output.total_num_scheduled_tokens == 65100 + 31*4
    assert output.num_scheduled_tokens["fresh"] == 65100
    assert history_rows(scheduler, output) == 124
    assert all(req.num_output_placeholders == 4 for req in running)
    assert fresh.num_output_placeholders == 1
    assert fresh.spec_token_ids == [-1]*3 and fresh.num_computed_tokens == 65100
    assert scheduler.max_num_scheduled_tokens == 66048
    # Native speculative placeholders can grow across outstanding async steps,
    # but the matching computed-token advance keeps each next query at q4.
    second = scheduler.schedule()
    assert second.total_num_scheduled_tokens == history_rows(scheduler, second) == 128
    assert set(second.num_scheduled_tokens.values()) == {4}


def test_multiple_long_requests_first_fresh_step_then_total_history_bound(scheduler_factory):
    requests = [Request(f"long{i}", prompt=100000) for i in range(4)]
    scheduler = scheduler_factory(waiting=requests)
    first = scheduler.schedule()
    assert first.total_num_scheduled_tokens == 65536
    assert history_rows(scheduler, first) == 0
    assert set(first.num_scheduled_tokens.values()) == {16384}
    second = scheduler.schedule()
    assert second.total_num_scheduled_tokens == 16896
    assert history_rows(scheduler, second) == 16896
    assert second.num_scheduled_tokens == {"long0": 16384, "long1": 512}
    # The smaller step remains a native async step, not an isolated clipping helper.
    assert requests[0].num_computed_tokens == 32768
    assert requests[1].num_computed_tokens == 16896
    assert not second.scheduled_spec_decode_tokens


def test_historical_step_keeps_large_fresh_request_queued_without_partial_admission(scheduler_factory):
    long = Request("history", prompt=100000, computed=16384)
    fresh = Request("fresh")
    scheduler = scheduler_factory([long], [fresh])
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"history": 16384}
    assert history_rows(scheduler, output) <= 16896
    assert scheduler.waiting.peek_request() is fresh and fresh.num_computed_tokens == 0


def test_existing_decode_and_multiple_continuations_share_one_history_budget(scheduler_factory):
    decode = [Request(f"d{i}", computed=65100, output=1) for i in range(31)]
    for request in decode: request.spec_token_ids = [-1]*3
    partial = [Request(f"p{i}", prompt=100000, computed=16384) for i in range(2)]
    scheduler = scheduler_factory([*decode, *partial], [Request("fresh")])
    output = scheduler.schedule()
    assert output.total_num_scheduled_tokens == history_rows(scheduler, output) == 16896
    assert output.num_scheduled_tokens["p0"] == 16384
    assert output.num_scheduled_tokens["p1"] == 388
    assert scheduler.waiting.peek_request().request_id == "fresh"


@pytest.mark.parametrize("length,first_rows", [(65100, 65100), (65536, 65536), (65537, 16384)])
def test_actual_native_admission_boundary(scheduler_factory, length, first_rows):
    scheduler = scheduler_factory(waiting=[Request("boundary", prompt=length)])
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"boundary": first_rows}
    assert history_rows(scheduler, output) == 0


def test_native_allocation_preemption_then_zero_context_resume(scheduler_factory):
    req = Request("resume", computed=65100, output=1)
    scheduler = scheduler_factory([req])
    scheduler.kv_cache_manager.fail_once.add(req.request_id)
    first = scheduler.schedule()
    assert first.preempted_req_ids == {"resume"} and first.total_num_scheduled_tokens == 0
    assert req.status == STATUS.PREEMPTED and req.num_computed_tokens == 0
    second = scheduler.schedule()
    assert second.scheduled_cached_reqs.resumed_req_ids == {"resume"}
    assert second.scheduled_cached_reqs.num_computed_tokens == [0]
    assert second.num_scheduled_tokens == {"resume": 65101}
    assert history_rows(scheduler, second) == 0 and req.num_preemptions == 1


def test_real_force_reset_notifies_same_step_resumption_and_drops_stale_async_frame(scheduler_factory):
    req = Request("reset", computed=65100, output=1)
    req.num_output_placeholders = 4
    scheduler = scheduler_factory([req])
    scheduler.prev_step_scheduled_req_ids.add("reset")
    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    assert req.num_output_placeholders == 0 and req.async_tokens_to_discard == 4
    assert not scheduler.prev_step_scheduled_req_ids
    output = scheduler.schedule()
    assert output.preempted_req_ids == {"reset"}
    assert output.scheduled_cached_reqs.resumed_req_ids == {"reset"}
    assert output.scheduled_cached_reqs.num_computed_tokens == [0]
    assert scheduler._update_request_with_output(req, [123]) == ([], False)
    assert req.async_tokens_to_discard == 3


@pytest.mark.parametrize("field", ["connector", "ec_connector"])
def test_unqualified_external_connector_is_rejected(scheduler_factory, field):
    with pytest.raises(ValueError, match="external"):
        scheduler_factory(**{field: object()})


def test_positive_context_waiting_stream_is_not_assumed_fresh(scheduler_factory):
    request = Request("stream", prompt=100000, computed=20000, status=STATUS.WAITING)
    scheduler = scheduler_factory(waiting=[request, Request("stream2", prompt=100000,
                                                         computed=20000, status=STATUS.WAITING)])
    with pytest.raises(ValueError, match="waiting.*history"):
        scheduler.schedule()
    assert scheduler.kv_cache_manager.allocations == []
    assert request.num_computed_tokens == 20000


def _method(path, cls_name, name):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    method.decorator_list = []
    return method


def test_actual_ascend_dummy_gate_and_native_capture_warmup_stay_within_history():
    path = ROOT / "references/vllm-ascend/vllm_ascend/worker/model_runner_v1.py"
    method = _method(path, "NPUModelRunner", "_should_build_dummy_attn_metadata")
    modes = S(NONE="none", FULL="full")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                             method], type_ignores=[])
    ns = {"CUDAGraphMode": modes}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    gate = ns[method.name]
    assert not gate(None, is_profile=True, cudagraph_runtime_mode=modes.NONE)
    assert gate(None, force_attention=True, is_profile=True, cudagraph_runtime_mode=modes.NONE)
    gpu = ROOT / "references/vllm/vllm/v1/worker/gpu_model_runner.py"
    method = _method(gpu, "GPUModelRunner", "_warmup_and_capture")
    module.body[-1] = method
    exec(compile(ast.fix_missing_locations(module), str(gpu), "exec"), ns)
    calls = []
    runner = S(compilation_config=S(cudagraph_num_of_warmups=1),
               _dummy_run=lambda tokens, **kwargs: calls.append((tokens, kwargs)))
    for size in target()["compilation_config"]["cudagraph_capture_sizes"]:
        ns[method.name](runner, S(num_tokens=size, uniform=True, num_active_loras=0), modes.FULL)
    assert all(tokens <= 16896 for tokens, _ in calls)
    assert any(kwargs.get("force_attention") for _, kwargs in calls)


@pytest.mark.parametrize("additional", [
    {"SLO_limits_for_dynamic_batch": 10},
    {"profiling_chunk_config": {"enabled": True}},
])
def test_dynamic_scheduler_replacement_is_not_qualified(additional):
    cfg = target(); cfg["additional_config"] = additional
    with pytest.raises(ValueError, match="dynamic"):
        load(object)["validate_configuration"](cfg,
            S(enable_prefix_caching=False, mamba_cache_mode="none"),
            input_capacity=66048, max_sequences=128, speculative_tokens=3)
