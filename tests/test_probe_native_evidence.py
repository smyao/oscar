"""Archive #27/#94/#125/#157/#163/#176: execute native scheduler evidence seams.

Selected native methods are compiled unchanged from the read-only references;
ancillary managers are host fixtures. No logger, NPU or SDK is initialized.
OSCAR_REFERENCE_ROOT may identify the read-only references directory explicitly.
"""
import ast
from collections import defaultdict
import copy
from dataclasses import dataclass, field
import enum
import functools
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

from oscar_ascend import probe_evidence
from tools.probe_evidence import evaluate_mtp_events, preempted_requests

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = Path(os.environ.get("OSCAR_REFERENCE_ROOT", ROOT / "references"))


def native_definitions(module, relative_path, selected):
    path = REFERENCE_ROOT / relative_path
    tree = ast.parse(path.read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in selected:
            node = copy.deepcopy(node)
            methods = selected[node.name]
            if methods is not None:
                assert isinstance(node, ast.ClassDef)
                node.body = [method for method in node.body
                             if isinstance(method, ast.FunctionDef) and method.name in methods]
                assert {method.name for method in node.body} == set(methods)
            nodes.append(node)
    assert {node.name for node in nodes} == set(selected)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    code = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(code, str(path), "exec"), module.__dict__)


@pytest.fixture
def native(monkeypatch, tmp_path):
    module = ModuleType("probe_native_evidence_fixture")
    module.__dict__.update(enum=enum, dataclass=dataclass, field=field,
        defaultdict=defaultdict, SchedulerInterface=object,
        EngineCoreOutput=NS, EngineCoreOutputs=NS, _FINISHED_REASON_MAP={})
    monkeypatch.setitem(sys.modules, module.__name__, module)
    native_definitions(module, "vllm/vllm/v1/request.py", {
        "RequestStatus": None,
        "Request": {"append_output_token_ids", "update_block_hashes", "num_tokens", "num_output_tokens",
                    "is_finished", "get_finished_reason", "take_events", "take_prefill_stats"}})
    native_definitions(module, "vllm/vllm/v1/spec_decode/metrics.py", {"SpecDecodingStats": None})
    native_definitions(module, "vllm/vllm/v1/core/sched/utils.py", {"check_stop": None})
    native_definitions(module, "vllm/vllm/v1/core/sched/scheduler.py", {
        "Scheduler": {"make_spec_decoding_stats", "_preempt_request", "update_from_output",
                      "_update_request_with_output", "make_stats"}})
    native_definitions(module, "vllm/vllm/v1/core/sched/async_scheduler.py", {
        "AsyncScheduler": {"_update_request_with_output"}})
    monkeypatch.setenv("OSCAR_PROBE_EVIDENCE_DIR", str(tmp_path))
    monkeypatch.setenv("OSCAR_PROBE_RUN_ID", "native-test-run")
    (tmp_path / "active.json").write_text(json.dumps({"run_id": "native-test-run", "window_id": "window-1"}))
    try:
        yield module
    finally:
        probe_evidence.unregister()


def records(tmp_path):
    rows = [json.loads(line) for path in tmp_path.glob("scheduler-*.jsonl")
            for line in path.read_text().splitlines()]
    assert all(row["schema"] == 1 and row["run_id"] == "native-test-run"
               and row["window_id"] == "window-1" and row["pid"] == os.getpid() for row in rows)
    assert len({row["sequence"] for row in rows}) == len(rows)
    return rows


@pytest.mark.parametrize("accepted", [0, 1, 2, 3])
@pytest.mark.parametrize("invalid", [0, 2, 7])
def test_native_stats_disabled_keeps_raw_slots_and_none_return(native, tmp_path, accepted, invalid):
    probe_evidence.install(native)
    scheduler = native.Scheduler()
    scheduler.log_stats = False
    # The false early return must not create stats or apply invalid-slot subtraction.
    assert scheduler.make_spec_decoding_stats(None, 3, accepted, {"request": invalid}, "request") is None
    assert not hasattr(scheduler, "num_spec_tokens")
    row, = records(tmp_path)
    assert (row["scheduled_draft_slots"], row["num_accepted_tokens"], row["invalid_draft_slots"]) == (3, accepted, invalid)
    assert evaluate_mtp_events([row])["status"] == "passed"


def test_native_stats_enabled_retains_original_object_and_arithmetic(native, tmp_path):
    probe_evidence.install(native)
    scheduler = native.Scheduler()
    scheduler.log_stats, scheduler.num_spec_tokens = True, 3
    stats = native.SpecDecodingStats.new(3)
    assert scheduler.make_spec_decoding_stats(stats, 3, 1, {"request": 2}, "request") is stats
    assert stats.num_draft_tokens == stats.num_accepted_tokens == 1
    row, = records(tmp_path)
    assert row["scheduled_draft_slots"] == 3 and row["invalid_draft_slots"] == 2


def test_native_stats_exception_does_not_emit_success(native, tmp_path):
    probe_evidence.install(native)
    scheduler = native.Scheduler()
    scheduler.log_stats, scheduler.num_spec_tokens = True, 3
    with pytest.raises(AssertionError):
        scheduler.make_spec_decoding_stats(None, 3, 4, None, "request")
    assert not records(tmp_path)


def test_zero_draft_native_early_return_is_not_mtp_evidence(native, tmp_path):
    probe_evidence.install(native)
    scheduler = native.Scheduler()
    scheduler.log_stats = False
    assert scheduler.make_spec_decoding_stats(None, 0, 0, None, "request") is None
    assert not records(tmp_path)


def async_fixture(native):
    scheduler = native.AsyncScheduler()
    request = native.Request()
    request.__dict__.update(request_id="request", status=native.RequestStatus.RUNNING,
        num_computed_tokens=104, num_output_placeholders=4, num_preemptions=5,
        spec_token_ids=[11, 12, 13], _output_token_ids=[], _all_token_ids=list(range(101)),
        _block_hasher=None, async_tokens_to_discard=0, has_encoder_inputs=False,
        pooling_params=None, sampling_params=NS(min_tokens=0, eos_token_id=-1, stop_token_ids=[],
            repetition_detection=None, num_logprobs=None), max_tokens=1000,
        client_index=0, stop_reason=None, events=[], prefill_stats=None, trace_headers=None,
        num_nans_in_logits=0)
    request.output_token_ids = request._output_token_ids
    actions = []
    scheduler.__dict__.update(log_stats=False, requests={"request": request},
        num_spec_tokens=3, perf_metrics=None, connector=None, max_model_len=2048,
        structured_output_manager=NS(should_advance=lambda req: False),
        enable_return_routed_experts=False, finished_req_ids_dict={},
        kv_cache_manager=NS(take_events=lambda: None,
            cache_blocks=lambda req, count: actions.append(("cache", req.request_id, count)),
            free=lambda req: actions.append(("free-kv", req.request_id))),
        encoder_cache_manager=NS(free=lambda req: actions.append(("free-encoder", req.request_id))),
        _inflight_prefills={request}, waiting=NS(prepend_request=lambda req: actions.append(("wait", req.request_id))))
    return scheduler, request, actions


@pytest.mark.parametrize("accepted", [0, 1, 2, 3])
def test_real_async_update_records_sampler_count_and_preserves_native_state(native, tmp_path, accepted):
    probe_evidence.install(native)
    scheduler, request, actions = async_fixture(native)
    generated = list(range(201, 202 + accepted))
    scheduled = NS(num_scheduled_tokens={"request": 4},
                   scheduled_spec_decode_tokens={"request": [11] * accepted + [-1] * (3 - accepted)},
                   num_invalid_spec_tokens={"request": 3 - accepted})
    output = NS(sampled_token_ids=[generated], logprobs=None, prompt_logprobs_dict={},
                pooler_output=None, num_nans_in_logits=None, kv_connector_output=None,
                cudagraph_stats=None, routed_experts=None, req_id_to_index={"request": 0})
    result = scheduler.update_from_output(scheduled, output)
    assert result[0].outputs[0].new_token_ids == generated
    assert request._output_token_ids == generated and request.num_output_placeholders == 0
    assert request.num_computed_tokens == 101 + accepted
    assert actions == [("cache", "request", 101 + accepted)]
    assert not hasattr(result[0], "scheduler_stats")
    row, = records(tmp_path)
    assert row["num_accepted_tokens"] == len(generated) - 1 == accepted
    assert row["scheduled_draft_slots"] == 3


@pytest.mark.parametrize("dummy", [False, True])
def test_real_native_empty_or_dummy_output_has_no_mtp_event(native, tmp_path, dummy):
    probe_evidence.install(native)
    scheduler, request, actions = async_fixture(native)
    scheduled = NS(num_scheduled_tokens={} if dummy else {"request": 4},
        scheduled_spec_decode_tokens={} if dummy else {"request": [11, 12, 13]}, num_invalid_spec_tokens={})
    output = NS(sampled_token_ids=[], logprobs=None, prompt_logprobs_dict={}, pooler_output=None,
        num_nans_in_logits=None, kv_connector_output=None, cudagraph_stats=None,
        routed_experts=None, req_id_to_index={"request": 0})
    assert scheduler.update_from_output(scheduled, output) == {}
    assert not records(tmp_path) and not actions
    assert request.num_computed_tokens == 104 and request.num_output_placeholders == 4


def test_native_preemption_evidence_follows_free_reset_and_waiting(native, tmp_path):
    probe_evidence.install(native)
    scheduler, request, actions = async_fixture(native)
    assert scheduler._preempt_request(request, 1.25) is None
    assert actions == [("free-kv", "request"), ("free-encoder", "request"), ("wait", "request")]
    assert request not in scheduler._inflight_prefills
    assert request.status == native.RequestStatus.PREEMPTED
    assert request.num_computed_tokens == 0 and request.spec_token_ids == []
    assert request.events == []  # Native log_stats stays disabled.
    row, = records(tmp_path)
    assert row["preemptions_before"] == 5 and row["preemptions_after"] == request.num_preemptions == 6
    assert preempted_requests([row]) == ["request"]


def test_native_preemption_failure_after_increment_is_not_success(native, tmp_path):
    probe_evidence.install(native)
    scheduler, request, actions = async_fixture(native)
    error = RuntimeError("waiting insertion failed")
    def failed_wait(req):
        raise error
    scheduler.waiting.prepend_request = failed_wait
    with pytest.raises(RuntimeError) as caught:
        scheduler._preempt_request(request, 1.25)
    assert caught.value is error and request.num_preemptions == 6
    assert not records(tmp_path)


def test_probe_patch_composes_and_uninstalls_without_removing_existing_wrapper(native, tmp_path):
    original = native.Scheduler.make_spec_decoding_stats
    calls = []
    @functools.wraps(original)
    def existing(*args, **kwargs):
        calls.append("existing")
        return original(*args, **kwargs)
    native.Scheduler.make_spec_decoding_stats = existing
    probe_evidence.install(native)
    wrapped = native.Scheduler.make_spec_decoding_stats
    probe_evidence.install(native)
    assert native.Scheduler.make_spec_decoding_stats is wrapped
    scheduler = native.Scheduler()
    scheduler.log_stats = False
    assert scheduler.make_spec_decoding_stats(None, 3, 0, None, "request") is None
    assert calls == ["existing"] and len(records(tmp_path)) == 1
    probe_evidence.unregister()
    assert native.Scheduler.make_spec_decoding_stats is existing


@pytest.mark.parametrize("missing", ["directory", "run_id", "window"])
def test_without_explicit_active_window_native_result_is_unchanged(native, tmp_path, monkeypatch, missing):
    if missing == "window":
        (tmp_path / "active.json").unlink()
    else:
        monkeypatch.delenv("OSCAR_PROBE_EVIDENCE_DIR" if missing == "directory" else "OSCAR_PROBE_RUN_ID")
    probe_evidence.install(native)
    scheduler = native.Scheduler()
    scheduler.log_stats = False
    assert scheduler.make_spec_decoding_stats(None, 3, 0, None, "request") is None
    assert not records(tmp_path)
