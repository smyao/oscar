"""Archive #129/#142/#155: complete-step bounds against native clipping rules."""
import ast
from pathlib import Path
from types import SimpleNamespace as S

import pytest
from test_whole_prefill_scheduler import load_policy,native_waiting_clip

ROOT=Path(__file__).resolve().parents[1]


def load(parent):
    path=ROOT/'oscar_ascend/integration/bounded_whole_prefill.py'
    tree=ast.parse(path.read_text())
    tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module in (
        'vllm.v1.core.sched.async_scheduler','whole_prefill_scheduler'))]
    ns={'AsyncScheduler':parent,'AdmissionView':load_policy(object)['AdmissionView']}
    exec(compile(tree,str(path),'exec'),ns)
    return ns


def target():
    return {'max_num_batched_tokens':66048,'history_token_capacity':16896,
        'experimental_current_only':True,'experimental_first_mtp_current_fia':True,
        'experimental_compact_later_mtp':True,
        'compilation_config':{'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[1,4,128,512]}}


def test_large_current_capacity_covers65100_plus_all_decode_reservations():
    ns=load(object);cfg=target()
    assert ns['validate_configuration'](cfg,S(enable_prefix_caching=False,mamba_cache_mode='none'),
        input_capacity=66048,max_sequences=128,speculative_tokens=3)==16896
    view=ns['AdmissionView'](S(enable_chunked_prefill=True,long_prefill_token_threshold=0),
        budget=66048,max_sequences=128,speculative_tokens=3)
    request=S(num_tokens=65100,has_encoder_inputs=False);view.select(request)
    assert view.whole_limit==65536
    assert native_waiting_clip(view,request,66048-128*4)==65100
    # A historical step is smaller; never turn the same whole prompt into a
    # partial chunk merely because there is a smaller remaining step budget.
    assert native_waiting_clip(view,request,16896) is None


@pytest.mark.parametrize('computed,prompt,tokens,expected',[
    (0,65100,65100,False),(16384,65100,65100,True),
    (49152,65100,65100,True),(65100,65100,65101,False),
    (65103,65100,65104,False),(100,65100,65612,True)])
def test_resumed_prefill_boundary(computed,prompt,tokens,expected):
    assert load(object)['has_historical_prefill'](S(num_computed_tokens=computed,
        num_prompt_tokens=prompt,num_tokens=tokens)) is expected


@pytest.mark.parametrize('historical',[False,True])
def test_native_parent_gets_total_step_budget_and_state_is_restored(historical):
    original=S(enable_chunked_prefill=True,long_prefill_token_threshold=0)
    req=S(num_computed_tokens=16384 if historical else 65100,num_prompt_tokens=65100,
          num_tokens=65100 if historical else 65101)
    class Parent:
        def schedule(self):
            assert self.max_num_scheduled_tokens==(16896 if historical else 66048)
            assert self.scheduler_config.whole_limit==65536
            assert self.scheduler_config.long_prefill_token_threshold==16384
            raise RuntimeError('native allocation failure')
    ns=load(Parent);cls=ns['BoundedWholePromptAsyncScheduler']
    obj=cls.__new__(cls);obj.scheduler_config=original;obj.max_num_scheduled_tokens=66048
    obj.max_num_running_reqs=128;obj.num_spec_tokens=3;obj.running=[req]
    obj._oscar_history_capacity=16896
    with pytest.raises(RuntimeError,match='native allocation failure'):obj.schedule()
    assert obj.scheduler_config is original and obj.max_num_scheduled_tokens==66048


@pytest.mark.parametrize('field,value',[
    ('history_token_capacity',16384),('experimental_current_only',False),
    ('experimental_first_mtp_current_fia',False),('experimental_compact_later_mtp',False)])
def test_missing_qualification_is_rejected(field,value):
    cfg=target();cfg[field]=value
    with pytest.raises(ValueError):load(object)['validate_configuration'](cfg,
        S(enable_prefix_caching=False,mamba_cache_mode='none'),input_capacity=66048,
        max_sequences=128,speculative_tokens=3)


def test_prefix_and_large_forced_capture_are_not_silently_bypassed():
    fn=load(object)['validate_configuration'];cfg=target()
    with pytest.raises(ValueError,match='prefix'):
        fn(cfg,S(enable_prefix_caching=True,mamba_cache_mode='none'),input_capacity=66048,
           max_sequences=128,speculative_tokens=3)
    cfg['compilation_config']['cudagraph_capture_sizes']=[66048]
    with pytest.raises(ValueError,match='capture'):
        fn(cfg,S(enable_prefix_caching=False,mamba_cache_mode='none'),input_capacity=66048,
           max_sequences=128,speculative_tokens=3)
