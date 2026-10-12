"""Archive #129/#137/#155: admission must preserve native ownership and bounds."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from tools.native_contract_sources import native_sources

ROOT=Path(__file__).resolve().parents[1]


def load_policy(parent):
    # Execute our implementation with a stand-in native class, without
    # importing/initializing vLLM/NPU on the local CPU host.
    path=ROOT/'oscar_ascend/integration/whole_prefill_scheduler.py'
    tree=ast.parse(path.read_text())
    tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and
               n.module=='vllm.v1.core.sched.async_scheduler')]
    namespace={'AsyncScheduler':parent}
    exec(compile(tree,str(path),'exec'),namespace)
    return namespace


def native_waiting_clip(config, request, budget, computed=0):
    # Execute the pinned native WAITING clipping lines, not a second copy of
    # our implementation. Break is represented by a surrounding one-item loop.
    source=native_sources().read_text('references/vllm/vllm/v1/core/sched/scheduler.py')
    tree=ast.parse(source)
    nodes=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(
        isinstance(t,ast.Name) and t.id=='num_new_tokens' for t in n.targets)]
    start=next(n.lineno for n in nodes if isinstance(n.value,ast.BinOp) and
               ast.unparse(n.value)=='request.num_tokens - num_computed_tokens')
    lines=source.splitlines()
    end=next(i for i in range(start,len(lines)) if lines[i].strip()=='assert num_new_tokens > 0')
    import textwrap
    fragment=textwrap.dedent('\n'.join(lines[start-1:end+1]))
    code='result = None\nfor once in [True]:\n'+textwrap.indent(fragment,'    ')+'\n    result = num_new_tokens\n'
    ns={'self':SimpleNamespace(scheduler_config=config),'request':request,
        'num_computed_tokens':computed,'token_budget':budget}
    exec(compile(code,'<pinned native waiting admission>','exec'),ns)
    return ns['result']


def test_native_clip_defers_second_short_prompt_but_chunks_long_requests():
    View=load_policy(object)['AdmissionView']
    original=SimpleNamespace(enable_chunked_prefill=True,long_prefill_token_threshold=0)
    policy=View(original,budget=32768,max_sequences=128,speculative_tokens=3)
    one=SimpleNamespace(num_tokens=20000,has_encoder_inputs=False)
    policy.select(one)
    assert native_waiting_clip(policy,one,32640)==20000
    # A second 23K request remains waiting instead of becoming a partial12K.
    two=SimpleNamespace(num_tokens=23000,has_encoder_inputs=False)
    policy.select(two)
    assert native_waiting_clip(policy,two,12640) is None
    long=SimpleNamespace(num_tokens=65000,has_encoder_inputs=False)
    policy.select(long)
    assert native_waiting_clip(policy,long,32640)==16384
    assert original.enable_chunked_prefill is True
    assert original.long_prefill_token_threshold==0


def test_decode_reservation_prevents_32768_edge_starvation():
    View=load_policy(object)['AdmissionView']
    original=SimpleNamespace(enable_chunked_prefill=True,long_prefill_token_threshold=0)
    policy=View(original,budget=32768,max_sequences=128,speculative_tokens=3)
    for length in (32257,32768,262144):
        request=SimpleNamespace(num_tokens=length,has_encoder_inputs=False)
        policy.select(request)
        assert native_waiting_clip(policy,request,32256)==16384


def test_prefix_hit_classifies_only_the_remaining_prompt_suffix():
    View=load_policy(object)['AdmissionView']
    original=SimpleNamespace(enable_chunked_prefill=True,long_prefill_token_threshold=0)
    policy=View(original,budget=32768,max_sequences=128,speculative_tokens=3)
    request=SimpleNamespace(num_tokens=50000,num_computed_tokens=24624,has_encoder_inputs=False)
    policy.select(request)
    assert policy.is_whole
    request.num_computed_tokens=0
    assert not policy.is_whole


@pytest.mark.parametrize('budget', [2304, 4096, 16384, 32768, 32769, 65536])
def test_smaller_and_unaligned_budgets_keep_native_admission_progress(budget):
    from oscar_ascend.integration.prefix_scheduler import PrefixCheckpointPolicy
    View=load_policy(object)['AdmissionView']
    original=SimpleNamespace(enable_chunked_prefill=True,long_prefill_token_threshold=0)
    view=View(original,budget=budget,max_sequences=128,speculative_tokens=3)
    alignment=PrefixCheckpointPolicy(2304, True)
    request=SimpleNamespace(num_tokens=50000,num_prompt_tokens=50000,num_computed_tokens=0)
    view.select(request)
    for step in range(100):
        if request.num_computed_tokens == request.num_tokens:
            break
        tokens=native_waiting_clip(view,request,budget,computed=request.num_computed_tokens)
        assert tokens is not None
        tokens=alignment.split(request,tokens)
        assert 0 < tokens <= budget
        request.num_computed_tokens += tokens
    assert request.num_computed_tokens == 50000
    assert original.long_prefill_token_threshold == 0


def test_decode_reservation_can_exhaust_whole_capacity_without_rejecting_chunking():
    View=load_policy(object)['AdmissionView']
    original=SimpleNamespace(enable_chunked_prefill=True,long_prefill_token_threshold=0)
    view=View(original,budget=2304,max_sequences=1024,speculative_tokens=3)
    view.select(SimpleNamespace(num_tokens=4736))
    assert view.whole_limit == 0 and not view.is_whole
    assert view.enable_chunked_prefill and view.long_prefill_token_threshold == 2304


def test_async_parent_still_owns_queue_and_config_restores_on_error():
    original=SimpleNamespace(enable_chunked_prefill=True,long_prefill_token_threshold=0)
    request=SimpleNamespace(num_tokens=30000,has_encoder_inputs=False)
    queue=SimpleNamespace(peek_request=lambda:request)
    class Parent:
        def _select_waiting_queue_for_scheduling(self):return queue
        def schedule(self):
            assert self.scheduler_config.long_prefill_token_threshold==16384
            assert self._select_waiting_queue_for_scheduling() is queue
            assert not self.scheduler_config.enable_chunked_prefill
            assert original.enable_chunked_prefill
            raise RuntimeError('native allocation failure')
    Scheduler=load_policy(Parent)['WholePromptAsyncScheduler']
    scheduler=Scheduler()
    scheduler.scheduler_config=original
    scheduler.max_num_scheduled_tokens=32768
    scheduler.max_num_running_reqs=128
    scheduler.num_spec_tokens=3
    with pytest.raises(RuntimeError,match='native allocation failure'):
        scheduler.schedule()
    assert scheduler.scheduler_config is original


def test_first_prefix_hit_is_known_before_native_request_counter_is_committed():
    View=load_policy(object)['AdmissionView']
    policy=View(SimpleNamespace(enable_chunked_prefill=True,long_prefill_token_threshold=0),
                budget=32768,max_sequences=128,speculative_tokens=3)
    request=SimpleNamespace(num_tokens=50000,num_computed_tokens=0,num_preemptions=0,
                            prefill_stats=SimpleNamespace(num_cached_tokens=23040))
    policy.select(request)
    assert native_waiting_clip(policy,request,32768,computed=23040)==26960
    assert native_waiting_clip(policy,request,20000,computed=23040) is None
    # A preempted request must not reuse stale first-prefill statistics.
    request.num_preemptions=1
    assert native_waiting_clip(policy,request,32768)==16384


def test_explicit_scheduler_class_reaches_real_server_cli_without_changing_defaults():
    import json
    from tools.target_cli import serve_argv
    target=json.loads((ROOT/'configs/target.json').read_text())
    assert '--scheduler-cls' not in serve_argv(target)
    path='oscar_ascend.integration.whole_prefill_scheduler.WholePromptAsyncScheduler'
    proposed={**target,'max_num_batched_tokens':32768,'scheduler_cls':path}
    args=serve_argv(proposed)
    assert args[args.index('--scheduler-cls')+1]==path
    assert args[args.index('--max-num-batched-tokens')+1]=='32768'
    with pytest.raises(ValueError,match='class path'):
        serve_argv({**target,'scheduler_cls':None})


@pytest.mark.parametrize('prefix', [True, False])
@pytest.mark.parametrize('budget,sequences,scheduled,expected,mode', [
    (32768,128,20000,'whole_scheduled','whole_eligible'),
    (32768,128,16128,'eligible_partial','whole_eligible'),
    (32768,128,0,'deferred','whole_eligible'),
    (2304,1024,2304,'chunked_scheduled','chunked_only'),
])
def test_observation_distinguishes_request_eligibility_from_native_output(
        monkeypatch, prefix, budget, sequences, scheduled, expected, mode):
    from oscar_ascend import telemetry
    events=[]
    monkeypatch.setenv('OSCAR_TRACE_DIR','test-observation-no-files')
    monkeypatch.setattr(telemetry,'emit_once',lambda event,**fields:events.append((event,fields)))
    request=SimpleNamespace(request_id='request',num_tokens=20000,num_computed_tokens=0)
    output=SimpleNamespace(num_scheduled_tokens={'request':scheduled} if scheduled else {})
    original=SimpleNamespace(enable_chunked_prefill=True,long_prefill_token_threshold=0)
    queue=SimpleNamespace(peek_request=lambda:request)
    class Parent:
        def _select_waiting_queue_for_scheduling(self):return queue
        def schedule(self):
            self._select_waiting_queue_for_scheduling()
            assert self.scheduler_config.is_whole == (mode=='whole_eligible')
            # Native schedule updates request counters before returning its
            # output. Observation must retain the pre-schedule suffix length.
            request.num_computed_tokens += scheduled
            return output
    Scheduler=load_policy(Parent)['WholePromptAsyncScheduler']
    scheduler=Scheduler()
    scheduler.scheduler_config=original
    scheduler.cache_config=SimpleNamespace(enable_prefix_caching=prefix)
    scheduler.max_num_scheduled_tokens=budget
    scheduler.max_num_running_reqs=sequences
    scheduler.num_spec_tokens=3
    assert scheduler.schedule() is output and scheduler.scheduler_config is original
    policy,actual=events
    assert policy[0]=='whole_prefill_policy' and policy[1]['admission_mode']==mode
    assert actual[0]=='whole_prefill_admission' and actual[1][expected]==1
    assert actual[1]['scope']=='engine_core'
    assert actual[1]['evidence']=='native_scheduler_output_sample'
    assert sum(actual[1][k] for k in ('whole_scheduled','eligible_partial','chunked_scheduled','deferred'))==1


def test_unobserved_scheduler_allocates_no_per_request_evidence(monkeypatch):
    from oscar_ascend import telemetry
    monkeypatch.delenv('OSCAR_TRACE_DIR',raising=False)
    def forbidden(*args,**kwargs):raise AssertionError('unexpected hot-path telemetry')
    monkeypatch.setattr(telemetry,'emit_once',forbidden)
    original=SimpleNamespace(enable_chunked_prefill=True,long_prefill_token_threshold=0)
    output=object()
    class Parent:
        def schedule(self):
            assert self.scheduler_config.observations is None
            return output
    Scheduler=load_policy(Parent)['WholePromptAsyncScheduler']
    scheduler=Scheduler()
    scheduler.scheduler_config=original
    scheduler.max_num_scheduled_tokens=32768
    scheduler.max_num_running_reqs=128
    scheduler.num_spec_tokens=3
    assert scheduler.schedule() is output and scheduler.scheduler_config is original
