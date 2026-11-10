"""Archive #171/#182/#183: per-request cache evidence must retain native meaning.

Execute the pinned PrefillStats.set and scheduler update method on host data;
these logs are not device-completion, model-quality, or performance evidence.
"""

from dataclasses import dataclass
import json
from types import SimpleNamespace as NS

import pytest

from oscar_ascend import plugin
from oscar_ascend.integration import prefix_stats
from test_runtime_native import definitions


MARKER = "[oscar] PREFIX_CACHE_REQUEST "
COUNTERS = ("prompt_tokens", "local_hit_tokens", "external_hit_tokens", "total_hit_tokens")
RATES = ("local_hit_rate_pct", "external_hit_rate_pct", "total_hit_rate_pct")


@pytest.fixture(autouse=True)
def clean_plugin():
    plugin.unregister()
    yield
    plugin.unregister()


@pytest.fixture
def native(monkeypatch):
    stats = definitions(
        "references/vllm/vllm/v1/metrics/stats.py", "prefix_log_native_stats",
        {"PrefillStats"}, {"dataclass": dataclass}, monkeypatch).PrefillStats
    update = definitions(
        "references/vllm/vllm/v1/core/sched/scheduler.py", "prefix_log_native_update",
        set(), {}, monkeypatch, methods=[("Scheduler", "_update_after_schedule")])._update_after_schedule
    def update_from_output(scheduler, scheduler_output, model_runner_output):
        return None
    scheduler_cls = type("NativeScheduler", (), {
        "_update_after_schedule": update,
        "update_from_output": update_from_output,
    })
    return NS(stats=stats, scheduler_cls=scheduler_cls, update=update)


class Request:
    def __init__(self, stats, *, rid="request-1", preemptions=0):
        self.request_id = rid
        self.prefill_stats = stats
        self.num_preemptions = preemptions
        self.num_computed_tokens = stats.num_cached_tokens if stats is not None else 0
        self.num_prompt_tokens = stats.num_prompt_tokens if stats is not None else 100
        self.num_tokens = self.num_prompt_tokens
        self.num_output_placeholders = 0
        self.use_structured_output = True
        self.is_prefill_chunk = True


def state(native, prompt=1000, local=0, external=0, *, enabled=True, rid="request-1"):
    stats = native.stats()
    stats.set(prompt, local, external)
    request = Request(stats, rid=rid)
    scheduler = native.scheduler_cls()
    scheduler.requests = {rid: request}
    scheduler.cache_config = NS(enable_prefix_caching=enabled)
    scheduler._inflight_prefills = {request}
    scheduler.enable_return_routed_experts = False
    scheduler.finished_req_ids = {"previous-finished-request"}
    return scheduler, request


def output(request, *, admitted=True, new=True, count=1):
    return NS(scheduled_new_reqs=[NS(req_id=request.request_id)] if admitted and new else [],
              num_scheduled_tokens={request.request_id: count} if admitted else {},
              has_structured_output_requests=False)


def records(capsys):
    text = capsys.readouterr().out
    return [json.loads(line[len(MARKER):]) for line in text.splitlines() if line.startswith(MARKER)]


@pytest.mark.parametrize("prompt,local,external,enabled,expected_rates", [
    (1000, 0, 0, True, (0.0, 0.0, 0.0)),
    (1000, 400, 0, True, (40.0, 0.0, 40.0)),
    (1000, 0, 600, True, (0.0, 60.0, 60.0)),
    (1000, 200, 500, True, (20.0, 50.0, 70.0)),
    (1000, 0, 0, False, (0.0, 0.0, 0.0)),
    (0, 0, 0, True, (0.0, 0.0, 0.0)),
], ids=["cold", "local", "external", "mixed", "pc-disabled", "empty-prompt"])
def test_native_stats_logged_after_native_update_without_changing_result(
        native, monkeypatch, capsys, prompt, local, external, enabled, expected_rates):
    scheduler, request = state(native, prompt, local, external, enabled=enabled)
    control, control_request = state(native, prompt, local, external, enabled=enabled)
    count = max(prompt - local - external, 0)
    planned, control_output = output(request, count=count), output(control_request, count=count)
    before_stats = vars(request.prefill_stats).copy()
    expected_result = native.update(control, control_output)
    observe = prefix_stats.log_scheduled_prefix_cache
    calls = []
    def after_native(current, scheduled):
        assert current.requests[request.request_id].num_computed_tokens == control_request.num_computed_tokens
        assert current.finished_req_ids == control.finished_req_ids == set()
        calls.append(scheduled)
        return observe(current, scheduled)
    monkeypatch.setattr(prefix_stats, "log_scheduled_prefix_cache", after_native)
    plugin._patch_prefix_cache_logging(NS(Scheduler=native.scheduler_cls))
    assert scheduler._update_after_schedule(planned) is expected_result
    assert calls == [planned]
    assert vars(request.prefill_stats) == before_stats == vars(control_request.prefill_stats)
    assert request.is_prefill_chunk == control_request.is_prefill_chunk
    assert (request in scheduler._inflight_prefills) == (control_request in control._inflight_prefills)
    assert {key: value for key, value in vars(planned).items()
            if not key.startswith("_oscar_")} == vars(control_output)
    # Phase is captured before native advances this final prefill chunk.
    assert planned._oscar_step_kind == ("pure_prefill" if count else "empty")
    entry, = records(capsys)
    assert entry["request_id"] == request.request_id and entry["phase"] == "initial_schedule"
    assert tuple(entry[key] for key in COUNTERS) == (prompt, local, external, local + external)
    assert tuple(entry[key] for key in RATES) == expected_rates
    assert entry["hit_rate_denominator"] == "prompt_tokens"


def test_unadmitted_attempts_chunks_and_same_request_do_not_duplicate_but_new_request_can(native, capsys):
    scheduler, request = state(native, local=200, external=500)
    log = prefix_stats.log_scheduled_prefix_cache
    for _ in range(2):
        log(scheduler, output(request, admitted=False))
    assert records(capsys) == []
    assert not getattr(request, "_oscar_prefix_cache_logged", False)
    log(scheduler, output(request))
    assert len(records(capsys)) == 1
    log(scheduler, output(request, new=False))
    log(scheduler, output(request))
    assert records(capsys) == []
    # Request IDs may be reused; the old object, not its string ID, owns the marker.
    replacement = Request(request.prefill_stats, rid=request.request_id)
    scheduler.requests[request.request_id] = replacement
    log(scheduler, output(replacement))
    assert [entry["request_id"] for entry in records(capsys)] == [request.request_id]


def test_preempted_resume_never_logs_private_swap_as_initial_external_prefix(native, capsys):
    scheduler, request = state(native, external=500)
    request.num_preemptions = 1
    # Model Runner V2 may present a resumed request in scheduled_new_reqs.
    prefix_stats.log_scheduled_prefix_cache(scheduler, output(request))
    assert records(capsys) == []
    assert not getattr(request, "_oscar_prefix_cache_logged", False)


@pytest.mark.parametrize("invalid", ["missing", "negative", "boolean", "exceeds-prompt"])
def test_missing_or_invalid_native_stats_are_unavailable_not_zero(native, capsys, invalid):
    scheduler, request = state(native, local=200, external=500)
    if invalid == "missing":
        request.prefill_stats = None
    elif invalid == "negative":
        request.prefill_stats.num_external_cached_tokens = -1
    elif invalid == "boolean":
        request.prefill_stats.num_local_cached_tokens = True
    else:
        request.prefill_stats.num_external_cached_tokens = 1000
    before = None if request.prefill_stats is None else vars(request.prefill_stats).copy()
    prefix_stats.log_scheduled_prefix_cache(scheduler, output(request))
    entry, = records(capsys)
    assert entry["status"] == "unavailable" and entry["request_id"] == request.request_id
    assert all(key not in entry for key in (*COUNTERS, *RATES))
    assert (None if request.prefill_stats is None else vars(request.prefill_stats)) == before


def test_logging_patch_preserves_return_value_is_idempotent_and_unloads(native, capsys):
    sentinel, calls = object(), []
    def original(scheduler, planned):
        calls.append(planned)
        return sentinel
    native.scheduler_cls._update_after_schedule = original
    scheduler, request = state(native)
    module = NS(Scheduler=native.scheduler_cls)
    plugin._patch_prefix_cache_logging(module)
    installed = native.scheduler_cls._update_after_schedule
    plugin._patch_prefix_cache_logging(module)
    assert native.scheduler_cls._update_after_schedule is installed
    planned = output(request)
    assert scheduler._update_after_schedule(planned) is sentinel
    assert calls == [planned] and len(records(capsys)) == 1
    plugin.unregister()
    assert native.scheduler_cls._update_after_schedule is original


def test_logging_registration_is_independent_of_offload(native, monkeypatch, capsys):
    scheduler, request = state(native, enabled=False)
    monkeypatch.setattr(plugin, "_patch_prefix_checkpoint_scheduler", lambda module: None)
    monkeypatch.setattr(plugin, "_patch_request_window_scheduler", lambda module: None)
    monkeypatch.setattr(plugin, "_offload_configured", lambda: False)
    plugin._patch_scheduler(NS(Scheduler=native.scheduler_cls))
    scheduler._update_after_schedule(output(request))
    entry, = records(capsys)
    assert entry["prompt_tokens"] == 1000 and entry["total_hit_tokens"] == 0
