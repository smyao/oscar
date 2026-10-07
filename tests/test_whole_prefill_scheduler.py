"""Archive #129/#137/#155: admission must preserve native ownership and bounds."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def native_waiting_clip(config, request, budget):
    # Execute the pinned native WAITING clipping lines, not a second copy of
    # our implementation. Break is represented by a surrounding one-item loop.
    tree=ast.parse((ROOT/'references/vllm/vllm/v1/core/sched/scheduler.py').read_text())
    nodes=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(
        isinstance(t,ast.Name) and t.id=='num_new_tokens' for t in n.targets)]
    start=next(n.lineno for n in nodes if isinstance(n.value,ast.BinOp) and
               ast.unparse(n.value)=='request.num_tokens - num_computed_tokens')
    lines=(ROOT/'references/vllm/vllm/v1/core/sched/scheduler.py').read_text().splitlines()
    end=next(i for i in range(start,len(lines)) if lines[i].strip()=='assert num_new_tokens > 0')
    import textwrap
    fragment=textwrap.dedent('\n'.join(lines[start-1:end+1]))
    code='result = None\nfor once in [True]:\n'+textwrap.indent(fragment,'    ')+'\n    result = num_new_tokens\n'
    ns={'self':SimpleNamespace(scheduler_config=config),'request':request,
        'num_computed_tokens':0,'token_budget':budget}
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
    with pytest.raises(ValueError):
        View(original,budget=16384,max_sequences=128,speculative_tokens=3)


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
