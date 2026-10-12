"""Optional large-current admission with a bounded historical query budget.

Archive #127-129/#137/#140/#142/#145/#151-155: native async ownership, GDN
alignment, placeholders and queue order stay with the original scheduler.
The bound covers the WHOLE historical batch, not each request independently.
Current-only and compact later-MTP must be qualified before selecting this
class. It is not selected by the candidate preset yet.
"""
from __future__ import annotations
import json
import os
from pathlib import Path

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from oscar_ascend.request_windows import PreemptionNotificationMixin
from .whole_prefill_scheduler import AdmissionView

LEGACY_HISTORY_CHUNK = 16384


def required_history_capacity(max_sequences, speculative_tokens):
    if (type(max_sequences) is not int or max_sequences<=0 or
            type(speculative_tokens) is not int or speculative_tokens<0):
        raise ValueError('invalid request/speculative capacity')
    return LEGACY_HISTORY_CHUNK+max_sequences*(speculative_tokens+1)


def has_historical_prefill(request):
    """Pinned native resumed-prefill boundary (scheduler.py:311–315)."""
    computed=request.num_computed_tokens
    prompt=request.num_prompt_tokens
    tokens=request.num_tokens
    if any(type(v) is not int or v<0 for v in (computed,prompt,tokens)):
        raise ValueError('native request token counters are invalid')
    return 0<computed<max(prompt,tokens-1)


def validate_configuration(target, cache, *, input_capacity,max_sequences,speculative_tokens):
    if not isinstance(target,dict):raise ValueError('OSCAR target must be an object')
    # Pinned Ascend platform.py:713-725 replaces scheduler_cls for these
    # modes; worker.py:847 additionally forces real attention during profiling.
    # The configuration entrypoint must call this gate before platform setup,
    # because this class cannot protect a constructor that was replaced.
    additional=target.get('additional_config',{})
    profiling=additional.get('profiling_chunk_config',{}) if isinstance(additional,dict) else None
    if (not isinstance(additional,dict) or not isinstance(profiling,dict) or
            additional.get('SLO_limits_for_dynamic_batch',-1)!=-1 or
            profiling.get('enabled',False) is not False):
        raise ValueError('bounded whole-prefill does not qualify dynamic chunk/batch sizing')
    for key in ('experimental_current_only','experimental_first_mtp_current_fia',
                'experimental_compact_later_mtp'):
        if target.get(key) is not True:
            raise ValueError('bounded whole-prefill requires '+key)
    if getattr(cache,'enable_prefix_caching',None) is not False:
        raise ValueError('bounded whole-prefill requires explicit prefix caching disabled')
    if getattr(cache,'mamba_cache_mode',None)!='none':
        raise ValueError('bounded whole-prefill requires native Mamba mode none')
    history=target.get('history_token_capacity')
    required=required_history_capacity(max_sequences,speculative_tokens)
    if (type(input_capacity) is not int or target.get('max_num_batched_tokens')!=input_capacity
            or type(history) is not int or not required<=history<input_capacity):
        raise ValueError('scheduler and preallocated input/history capacities disagree')
    if input_capacity-max_sequences*(speculative_tokens+1)<65536:
        raise ValueError('whole-prefill capacity does not cover64K after decode reservation')
    compilation=target.get('compilation_config',{})
    if compilation.get('cudagraph_mode')!='FULL_DECODE_ONLY':
        raise ValueError('bounded history arena requires the qualified decode-only graph mode')
    sizes=compilation.get('cudagraph_capture_sizes',[])
    if not isinstance(sizes,list) or any(type(n) is not int or not 0<n<=history for n in sizes):
        raise ValueError('graph capture shape exceeds historical query arena')
    return history


class _BoundedPolicyScheduler(AsyncScheduler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        path=os.environ.get('OSCAR_TARGET_CONFIG')
        if not path:raise ValueError('bounded whole-prefill requires the effective OSCAR target config')
        target=json.loads(Path(path).read_text())
        if self.connector is not None or self.ec_connector is not None:
            raise ValueError('external KV/encoder-cache connectors are not qualified for bounded current-only')
        self._oscar_history_capacity=validate_configuration(target,self.cache_config,
            input_capacity=self.max_num_scheduled_tokens,max_sequences=self.max_num_running_reqs,
            speculative_tokens=self.num_spec_tokens)

    def schedule(self):
        original=self.scheduler_config
        if isinstance(original,AdmissionView):raise RuntimeError('reentrant bounded scheduling')
        # Native streaming sessions can re-enter WAITING with positive context
        # even when prefix caching and connectors are disabled. They have no
        # qualified fresh-suffix proof in this admission policy. Reject before
        # native RUNNING allocation/mutation rather than overfill the CV arena.
        for name in ('waiting','skipped_waiting'):
            if any(req.num_computed_tokens!=0 for req in getattr(self,name,())):
                raise ValueError('bounded whole-prefill waiting requests with history are not qualified')
        full_budget=self.max_num_scheduled_tokens
        # Waiting requests have zero history under the no-prefix/no-transfer
        # constructor qualification and the step guard above. Partial prefills may have
        # history; their ENTIRE step must fit the ordinary CV arena. A complete
        # waiting prompt that does not fit this smaller step remains queued.
        active_budget=(self._oscar_history_capacity if any(
            has_historical_prefill(req) for req in self.running) else full_budget)
        self.scheduler_config=AdmissionView(original,budget=full_budget,
            max_sequences=self.max_num_running_reqs,speculative_tokens=self.num_spec_tokens,
            legacy_chunk=LEGACY_HISTORY_CHUNK)
        self.max_num_scheduled_tokens=active_budget
        try:
            return super().schedule()
        finally:
            self.scheduler_config=original
            self.max_num_scheduled_tokens=full_budget

    def _select_waiting_queue_for_scheduling(self):
        queue=super()._select_waiting_queue_for_scheduling()
        policy=self.scheduler_config
        if isinstance(policy,AdmissionView):
            policy.select(queue.peek_request() if queue is not None else None)
        return queue


class BoundedWholePromptAsyncScheduler(PreemptionNotificationMixin,_BoundedPolicyScheduler):
    """Explicit class; forced-reset preemptions use the existing output field."""
