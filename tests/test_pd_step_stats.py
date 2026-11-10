"""Archive #157/#163/#171/#182/#183: completed PD steps retain native meaning.

Execute read-only native optimistic/async schedule updates on host data. A
warm final prefill is not decode, and KV-only acknowledgements are not model
steps. These counters provide neither device timing nor quality evidence.
"""
import json
from types import SimpleNamespace as NS

import pytest

from oscar_ascend import plugin
from oscar_ascend.integration import step_stats
from oscar_ascend.offload import hooks
from test_runtime_native import definitions


MARKER = "[oscar] PD_STEP_STATS "
FIELDS = ("total_steps", "pure_decode_steps", "pure_prefill_steps", "mixed_steps", "excluded_steps")


class Request:
    def __init__(self, computed, prompt, rid="r"):
        self.request_id = rid
        self.num_computed_tokens = computed
        self.num_prompt_tokens = prompt
        self.num_tokens = max(computed + 1, prompt)
        self.num_output_placeholders = 0
        self.use_structured_output = False
        self.spec_token_ids = []


def output(tokens, *, drafts=None):
    return NS(num_scheduled_tokens=dict(tokens), scheduled_spec_decode_tokens=drafts or {},
              total_num_scheduled_tokens=sum(tokens.values()), scheduled_new_reqs=[],
              has_structured_output_requests=False, pending_structured_output_tokens=False)


def model(ids=("r",), *, invalid=()):
    return NS(req_ids=list(ids), kv_connector_output=NS(invalid_block_ids=set(invalid)))


def state(requests=None):
    request = Request(100, 100)
    return NS(requests={"r": request} if requests is None else requests)


def counters(scheduler):
    stats = scheduler._oscar_pd_step_stats
    return tuple(getattr(stats, name) for name in FIELDS)


def complete(scheduler, tokens=None, *, ids=None):
    tokens = {"r": 1} if tokens is None else tokens
    planned = output(tokens)
    step_stats.capture_step_kind(scheduler, planned)
    step_stats.record_completed_step(scheduler, planned, model(tokens if ids is None else ids))
    return planned


def reports(capsys):
    return [json.loads(line[len(MARKER):]) for line in capsys.readouterr().out.splitlines()
            if line.startswith(MARKER)]


@pytest.fixture(autouse=True)
def clean_plugin():
    plugin.unregister()
    yield
    plugin.unregister()


@pytest.mark.parametrize("computed,prompt,count,kind", [
    (0, 100, 32, "pure_prefill"),
    (96, 100, 4, "pure_prefill"),
    (99, 100, 1, "pure_prefill"),
    (2304, 2305, 1, "pure_prefill"),
    (100, 100, 1, "pure_decode"),
    (105, 100, 4, "pure_decode"),
])
def test_phase_uses_pre_schedule_position_not_query_width(computed, prompt, count, kind):
    request = Request(computed, prompt)
    scheduler = state({"r": request})
    planned = output({"r": count}, drafts={"r": [91, 92, 93]} if count == 4 else {})
    original_request, original_output = vars(request).copy(), vars(planned).copy()
    step_stats.capture_step_kind(scheduler, planned)
    assert planned._oscar_step_kind == kind
    assert vars(request) == original_request
    assert {key: value for key, value in vars(planned).items() if key != "_oscar_step_kind"} == original_output
    assert isinstance(planned._oscar_step_kind, str)


def test_mixed_batch_and_zero_token_entries_keep_prefill_visible():
    scheduler = state({"p": Request(99, 100, "p"), "d": Request(105, 100, "d")})
    planned = output({"p": 1, "d": 4, "ignored-missing": 0})
    step_stats.capture_step_kind(scheduler, planned)
    assert planned._oscar_step_kind == "mixed"
    step_stats.record_completed_step(scheduler, planned, model(("d", "p")))
    assert counters(scheduler) == (1, 0, 0, 1, 0)


@pytest.mark.parametrize("async_mode", [False, True])
def test_native_optimistic_progress_and_async_outputs_retain_independent_phases(monkeypatch, async_mode):
    native = definitions("references/vllm/vllm/v1/core/sched/scheduler.py", "pd_native_progress", set(), {},
        monkeypatch, methods=[("Scheduler", "_update_after_schedule")])
    base = type("Scheduler", (), {"_update_after_schedule": native._update_after_schedule})
    plugin._patch_prefix_cache_logging(NS(Scheduler=base))
    cls = base
    if async_mode:
        cls = definitions("references/vllm/vllm/v1/core/sched/async_scheduler.py", "pd_native_async",
            {"AsyncScheduler"}, {"Scheduler": base}, monkeypatch).AsyncScheduler
    scheduler = cls.__new__(cls)
    request = Request(99, 100)
    scheduler.requests = {"r": request}
    scheduler._inflight_prefills = {request}
    scheduler.enable_return_routed_experts = False
    scheduler.finished_req_ids = {"previous"}
    scheduler._spec_token_placeholders = [-1, -1, -1]
    scheduler.use_v2_model_runner = False
    first = output({"r": 1})
    scheduler._update_after_schedule(first)
    assert request.num_computed_tokens == 100 and not request.is_prefill_chunk
    assert first._oscar_step_kind == "pure_prefill"
    second = output({"r": 4}, drafts={"r": [91, 92, 93]})
    scheduler._update_after_schedule(second)
    assert request.num_computed_tokens == 104
    assert second._oscar_step_kind == "pure_decode"
    # A later optimistic schedule must not relabel the earlier output.
    step_stats.record_completed_step(scheduler, second, model())
    step_stats.record_completed_step(scheduler, first, model())
    assert counters(scheduler) == (2, 1, 1, 0, 0)


@pytest.mark.parametrize("problem", ["unknown", "empty_model", "missing_model", "wrong_ids",
    "partial_ids", "extra_ids", "duplicate_ids"])
def test_unverifiable_or_failed_output_is_excluded_not_decode(problem):
    scheduler = state({"r": Request(100, 100), "second": Request(100, 100, "second")})
    planned = output({"r": 1, "second": 4})
    step_stats.capture_step_kind(scheduler, planned)
    result = model(("r", "second"))
    if problem == "unknown":
        planned._oscar_step_kind = "unknown"
    elif problem == "empty_model":
        result.req_ids = []
    elif problem == "missing_model":
        result = None
    elif problem == "wrong_ids":
        result.req_ids = ["other"]
    elif problem == "partial_ids":
        result.req_ids = ["r"]
    elif problem == "extra_ids":
        result.req_ids.append("other")
    elif problem == "duplicate_ids":
        result.req_ids.append("r")
    step_stats.record_completed_step(scheduler, planned, result)
    assert counters(scheduler) == (0, 0, 0, 0, 1)
    step_stats.record_completed_step(scheduler, planned, result)
    assert counters(scheduler) == (0, 0, 0, 0, 1)


def test_unrelated_async_kv_error_does_not_erase_completed_model_step():
    scheduler = state()
    planned = output({"r": 4}, drafts={"r": [91, 92, 93]})
    step_stats.capture_step_kind(scheduler, planned)
    # Native _handle_invalid_blocks also reports requests from prior async
    # receives; this field alone does not establish current forward failure.
    step_stats.record_completed_step(scheduler, planned, model(invalid=(9,)))
    assert counters(scheduler) == (1, 1, 0, 0, 0)


def test_missing_request_phase_is_unknown_without_rewriting_native_output():
    scheduler = state({})
    planned = output({"missing": 1})
    step_stats.capture_step_kind(scheduler, planned)
    assert planned._oscar_step_kind == "unknown"
    step_stats.record_completed_step(scheduler, planned, model(("missing",)))
    assert counters(scheduler) == (0, 0, 0, 0, 1)


def test_empty_private_swap_receipt_and_duplicate_completion_do_not_add_steps(capsys):
    scheduler = state()
    planned = complete(scheduler)
    step_stats.record_completed_step(scheduler, planned, model())
    empty = output({})
    empty.kv_connector_metadata = NS(preemptions=(object(),), restores=(object(),))
    step_stats.capture_step_kind(scheduler, empty)
    assert empty._oscar_step_kind == "empty"
    for _ in range(4):
        step_stats.record_completed_step(scheduler, empty,
            NS(req_ids=[], kv_connector_output=NS(invalid_block_ids=set(),
                kv_connector_worker_meta=NS(acknowledgements=(object(),)))))
    assert counters(scheduler) == (1, 1, 0, 0, 0)
    assert reports(capsys) == []


def test_decode_resumed_from_private_swap_counts_as_one_model_step():
    scheduler = state()
    planned = output({"r": 4}, drafts={"r": [91, 92, 93]})
    planned.kv_connector_metadata = NS(restores=(NS(version=7),))
    step_stats.capture_step_kind(scheduler, planned)
    step_stats.record_completed_step(scheduler, planned, model())
    assert counters(scheduler) == (1, 1, 0, 0, 0)


def test_lifetime_ratio_and_periodic_final_flush_without_empty_spam(capsys):
    scheduler = state()
    for index in range(100):
        if index < 50:
            scheduler.requests = {"r": Request(100, 100)}
            complete(scheduler)
        elif index < 75:
            scheduler.requests = {"r": Request(99, 100)}
            complete(scheduler)
        else:
            scheduler.requests = {"r": Request(99, 100), "d": Request(101, 100, "d")}
            complete(scheduler, {"r": 1, "d": 4})
        if index < 99:
            assert reports(capsys) == []
    entry, = reports(capsys)
    assert tuple(entry[key] for key in FIELDS) == (100, 50, 25, 25, 0)
    assert (entry["pure_decode_pct"], entry["pure_prefill_pct"], entry["mixed_pct"]) == (50, 25, 25)
    assert entry["scope"] == "engine_lifetime"
    assert entry["count_basis"] == "completed_classified_main_model_steps"
    rejected = output({"r": 1})
    step_stats.capture_step_kind(scheduler, rejected)
    step_stats.record_completed_step(scheduler, rejected, model(()))
    assert reports(capsys) == []  # Exclusion at total=100 is not another boundary.
    scheduler.requests = {"r": Request(100, 100)}
    final = output({"r": 1})
    step_stats.capture_step_kind(scheduler, final)
    scheduler.requests.clear()  # Native final completion frees the last request.
    step_stats.record_completed_step(scheduler, final, model())
    entry, = reports(capsys)
    assert tuple(entry[key] for key in FIELDS) == (101, 51, 25, 25, 1)
    assert entry["pure_decode_pct"] == pytest.approx(100 * 51 / 101, abs=0.01)
    for _ in range(3):
        empty = output({})
        step_stats.capture_step_kind(scheduler, empty)
        step_stats.record_completed_step(scheduler, empty, model(()))
    assert reports(capsys) == []


@pytest.mark.parametrize("failure", [None, "native", "offload"])
def test_output_hook_records_only_after_native_and_offload_succeed(monkeypatch, failure):
    calls, sentinel = [], object()
    def native(scheduler, planned, result):
        calls.append("native")
        if failure == "native":
            raise RuntimeError("native failed")
        return sentinel
    cls = type("Scheduler", (), {"update_from_output": native})
    module = NS(Scheduler=cls)
    def after(scheduler, planned):
        calls.append("offload")
        if failure == "offload":
            raise RuntimeError("offload failed")
    monkeypatch.setattr(hooks, "after_output", after)
    monkeypatch.setattr(step_stats, "record_completed_step", lambda *args: calls.append("stats"))
    plugin._patch_scheduler_output(module)
    installed = cls.update_from_output
    plugin._patch_scheduler_output(module)
    assert cls.update_from_output is installed
    scheduler = cls()
    planned, result = output({"r": 1}), model()
    if failure:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            scheduler.update_from_output(planned, result)
    else:
        assert scheduler.update_from_output(planned, result) is sentinel
    assert calls == ({None: ["native", "offload", "stats"],
                      "native": ["native"], "offload": ["native", "offload"]}[failure])
    plugin.unregister()
    assert cls.update_from_output is native
