"""Host-only counts of completed PD-fused main-model steps.

Archive #157/#163: prefill is a request phase, never a query-width heuristic.
Capture before native optimistic advancement; count after native output and
offload completion bookkeeping. No clocks, device reads or per-rank totals.
"""
from __future__ import annotations

from dataclasses import dataclass
import json


@dataclass
class StepStats:
    total_steps: int = 0
    pure_decode_steps: int = 0
    pure_prefill_steps: int = 0
    mixed_steps: int = 0
    excluded_steps: int = 0
    last_reported: tuple[int, int] = (0, 0)


def capture_step_kind(scheduler, scheduler_output) -> None:
    """Capture the phase before computed tokens or async placeholders advance."""
    prefill = decode = False
    for rid, count in scheduler_output.num_scheduled_tokens.items():
        if count <= 0:
            continue
        request = scheduler.requests.get(rid)
        computed = getattr(request, "num_computed_tokens", None)
        prompt = getattr(request, "num_prompt_tokens", None)
        if (type(computed) is not int or type(prompt) is not int
                or computed < 0 or prompt < 0):
            scheduler_output._oscar_step_kind = "unknown"
            return
        if computed < prompt:
            prefill = True
        else:
            decode = True
    scheduler_output._oscar_step_kind = (
        "mixed" if prefill and decode else "pure_prefill" if prefill
        else "pure_decode" if decode else "empty")


def record_completed_step(scheduler, scheduler_output, model_runner_output) -> None:
    """Count a successful main-model output once, never individual TP/draft calls."""
    if getattr(scheduler_output, "_oscar_step_stats_recorded", False):
        return
    scheduler_output._oscar_step_stats_recorded = True
    expected = {rid for rid, count in scheduler_output.num_scheduled_tokens.items() if count > 0}
    stats = getattr(scheduler, "_oscar_pd_step_stats", None)
    advanced = False
    if expected:
        if stats is None:
            stats = scheduler._oscar_pd_step_stats = StepStats()
        actual = getattr(model_runner_output, "req_ids", ())
        kind = getattr(scheduler_output, "_oscar_step_kind", "unknown")
        # A KV error notification can concern another asynchronous transfer.
        # Successful main-model output remains an executed step in that case.
        if (actual is None or len(actual) != len(expected)
                or set(actual) != expected
                or kind not in ("pure_decode", "pure_prefill", "mixed")):
            # Do not turn missing/incomplete execution evidence into decode.
            stats.excluded_steps += 1
        else:
            stats.total_steps += 1
            field = f"{kind}_steps"
            setattr(stats, field, getattr(stats, field) + 1)
            advanced = True
    if stats is None:
        return
    snapshot = (stats.total_steps, stats.excluded_steps)
    if (snapshot == stats.last_reported
            or not ((advanced and stats.total_steps % 100 == 0) or not scheduler.requests)):
        return
    report = {"scope": "engine_lifetime", "count_basis": "completed_classified_main_model_steps",
              "total_steps": stats.total_steps, "pure_decode_steps": stats.pure_decode_steps,
              "pure_prefill_steps": stats.pure_prefill_steps, "mixed_steps": stats.mixed_steps,
              "excluded_steps": stats.excluded_steps}
    for kind in ("pure_decode", "pure_prefill", "mixed"):
        report[f"{kind}_pct"] = (round(100.0 * getattr(stats, f"{kind}_steps") / stats.total_steps, 2)
                                  if stats.total_steps else 0.0)
    print("[oscar] PD_STEP_STATS " + json.dumps(report, separators=(",", ":")), flush=True)
    stats.last_reported = snapshot
