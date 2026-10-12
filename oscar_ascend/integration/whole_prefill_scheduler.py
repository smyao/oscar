"""Candidate whole-prompt admission with native GDN checkpoint splitting.

Archive #127/#129/#137/#148/#155/#156: preserve native request ownership, async
placeholders, Mamba alignment and long-request chunking. References:
vllm/v1/core/sched/scheduler.py:403-410,566-700 and config/scheduler.py:168.
Only the scheduler's private configuration view changes during schedule();
the shared VllmConfig and native source tree are never mutated.
"""
from __future__ import annotations

import os

from vllm.v1.core.sched.async_scheduler import AsyncScheduler


class AdmissionView:
    """Read-only, per-call view of the native scheduling policy."""

    def __init__(self, original, *, budget: int, max_sequences: int,
                 speculative_tokens: int, legacy_chunk: int = 16384, observe: bool = False):
        if (type(budget) is not int or type(max_sequences) is not int or
                type(speculative_tokens) is not int or type(legacy_chunk) is not int
                or budget <= 0 or max_sequences <= 0 or speculative_tokens < 0
                or legacy_chunk <= 0):
            raise ValueError('whole-prefill requires positive token/sequence budgets')
        self.original = original
        self.legacy_chunk = min(legacy_chunk, budget)
        # A smaller scheduling budget is valid: retain native chunked progress
        # instead of rejecting it or changing the compiled token capacity.
        self.whole_limit = max(0, budget - max_sequences * (speculative_tokens + 1))
        self.waiting_request = None
        # Probe-only host metadata; no per-request evidence allocation or
        # device readback in unobserved production runs.
        self.observations = {} if observe else None

    def __getattr__(self, name):
        return getattr(self.original, name)

    def select(self, request):
        self.waiting_request = request

    @property
    def is_whole(self):
        request = self.waiting_request
        if request is None or getattr(request, 'has_encoder_inputs', False):
            return False
        total = request.num_tokens
        computed = getattr(request, 'num_computed_tokens', 0)
        # Native WAITING admission records a fresh cache hit in PrefillStats
        # BEFORE reading this policy, but commits num_computed_tokens only
        # AFTER allocating slots. A just-arrived request still has computed=0.
        if computed == 0 and getattr(request, 'num_preemptions', 0) == 0:
            stats = getattr(request, 'prefill_stats', None)
            computed = getattr(stats, 'num_cached_tokens', 0)
        if (type(total) is not int or total <= 0 or type(computed) is not int
                or computed < 0 or computed > total):
            raise ValueError('native request must expose valid total/computed token counts')
        # Prefix hits consume no prompt budget. Admission must classify the
        # remaining suffix, not the original request length.
        eligible = total - computed <= self.whole_limit
        if self.observations is not None:
            self.observations.setdefault(request.request_id, (total - computed, eligible))
        return eligible

    @property
    def enable_chunked_prefill(self):
        # Native WAITING admission breaks if a whole request does not fit the
        # remaining budget; it is retained at the same queue position.
        return False if self.is_whole else self.original.enable_chunked_prefill

    @property
    def long_prefill_token_threshold(self):
        if self.is_whole:
            return 0
        original = self.original.long_prefill_token_threshold
        return min(original, self.legacy_chunk) if original > 0 else self.legacy_chunk


class WholePromptAsyncScheduler(AsyncScheduler):
    """Experimental async scheduler; requires an explicit --scheduler-cls."""

    def schedule(self):
        original = self.scheduler_config
        if isinstance(original, AdmissionView):
            raise RuntimeError('reentrant whole-prefill scheduling is unsupported')
        policy = AdmissionView(original, budget=self.max_num_scheduled_tokens,
                               max_sequences=self.max_num_running_reqs,
                               speculative_tokens=self.num_spec_tokens,
                               observe=bool(os.environ.get('OSCAR_TRACE_DIR')))
        if policy.observations is not None:
            from oscar_ascend.telemetry import emit_once
            emit_once('whole_prefill_policy',
                      key=(self.max_num_scheduled_tokens, policy.whole_limit, policy.legacy_chunk),
                      scope='engine_core', requested=True,
                      max_num_scheduled_tokens=self.max_num_scheduled_tokens,
                      whole_limit=policy.whole_limit, legacy_chunk=policy.legacy_chunk,
                      admission_mode='whole_eligible' if policy.whole_limit > 0 else 'chunked_only',
                      reason='decode_reservation_exhausted' if policy.whole_limit == 0 else 'capacity_available')
        self.scheduler_config = policy
        try:
            output = super().schedule()
        finally:
            self.scheduler_config = original
        if policy.observations:
            # The native scheduler may defer admission or split at a GDN
            # checkpoint after classifying a request as whole-eligible.
            # Report the returned allocation, never eligibility as execution.
            counts = dict(whole_scheduled=0, eligible_partial=0, chunked_scheduled=0, deferred=0)
            for request_id, (remaining, eligible) in policy.observations.items():
                scheduled = output.num_scheduled_tokens.get(request_id, 0)
                kind = ('deferred' if scheduled == 0 else
                        'whole_scheduled' if eligible and scheduled == remaining else
                        'eligible_partial' if eligible else 'chunked_scheduled')
                counts[kind] += 1
            emit_once('whole_prefill_admission',
                      key=(self.max_num_scheduled_tokens, policy.whole_limit,
                           *(value > 0 for value in counts.values())),
                      scope='engine_core', evidence='native_scheduler_output_sample',
                      whole_limit=policy.whole_limit, **counts)
        return output

    def _select_waiting_queue_for_scheduling(self):
        queue = super()._select_waiting_queue_for_scheduling()
        policy = self.scheduler_config
        if isinstance(policy, AdmissionView):
            policy.select(queue.peek_request() if queue is not None else None)
        return queue
