"""One prefix-cache report per admitted request, using native host counters.

Archive #37-49/#157/#163/#182/#183: a lookup is not an admission, chunked
prefill and private swap are not new prefix hits, and no device state is read.
Native PrefillStats is set before allocation may fail. Read it only from the
successful schedule's new requests, without consuming or changing the stats.
"""
from __future__ import annotations

import json


def log_scheduled_prefix_cache(scheduler, scheduler_output) -> None:
    """Log initial matched tokens; this is not a DMA-completion certificate."""
    for entry in scheduler_output.scheduled_new_reqs:
        request = scheduler.requests[entry.req_id]
        # V2 also includes resumed requests in scheduled_new_reqs. Retain the
        # first admission report across preemption, reset and streaming updates.
        if (getattr(request, "_oscar_prefix_cache_logged", False)
                or request.num_preemptions > 0):
            continue
        stats = getattr(request, "prefill_stats", None)
        fields = ("num_prompt_tokens", "num_local_cached_tokens", "num_external_cached_tokens")
        values = tuple(getattr(stats, field, None) for field in fields)
        report = {"request_id": request.request_id, "phase": "initial_schedule"}
        if (all(type(value) is int and value >= 0 for value in values)
                and values[1] + values[2] <= values[0]):
            prompt, local, external = values
            total = local + external
            report.update(prompt_tokens=prompt, local_hit_tokens=local,
                          external_hit_tokens=external, total_hit_tokens=total,
                          hit_rate_denominator="prompt_tokens")
            for name, hits in (("local", local), ("external", external), ("total", total)):
                report[f"{name}_hit_rate_pct"] = round(100.0 * hits / prompt, 2) if prompt else 0.0
        else:
            # Logging must neither fabricate misses nor change execution when
            # a native version does not expose the expected host counters.
            report.update(status="unavailable", reason="native_prefill_stats_unavailable_or_invalid")
        request._oscar_prefix_cache_logged = True
        print("[oscar] PREFIX_CACHE_REQUEST " + json.dumps(report, separators=(",", ":")), flush=True)
