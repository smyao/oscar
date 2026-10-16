"""Archive #13/#126/#145/#148: independent declared BF16 math, not device proof."""
import math
import pytest
import torch
from tools.bf16_reference import attention_tiles,compare_results,round_bf16,fixture_tasks_reference,load_acceptance
from oscar_ascend.ops.reference import attention

def test_rne_ties_and_signed_values():
    value=torch.tensor([1+1/256,1+3/256,-1-1/256,-1-3/256])
    assert torch.equal(round_bf16(value),torch.tensor([1.,1+2/128,-1.,-1-2/128]))

def test_authorized_bf16_contract_preserves_old_counterexample_as_diagnostic():
    q=torch.zeros(1,1,256);q[0,0,0]=2048
    k=torch.zeros(2,1,256);k[:,0,0]=torch.tensor([1.,1.+3/1024])
    v=torch.zeros_like(k);v[1,0,0]=1
    new=attention_tiles(q,k,v);old=attention(q,k,v,causal=False)
    assert new.output[0,0,0].item()==.5
    assert abs(old.output[0,0,0].item()-.5926666259765625)<1e-6
    report=compare_results(new,new,old_fp32=old)
    assert report['passed'] and not report['old_fp32_diagnostic_only']['output']['passed']
    assert report['old_fp32_diagnostic_only']['used_as_BF16_gate'] is False

def test_p_rounding_is_per_online_tile_not_dense_global_softmax():
    q=torch.ones(1,1,1);k=torch.zeros(257,1,1);k[-1]=1
    v=torch.ones_like(k);v[-1]=0
    answer=attention_tiles(q,k,v,scale=1.)
    alpha=torch.exp(torch.tensor(-1.));den=256*alpha+1
    expected=256*alpha/den
    assert torch.equal(answer.output.flatten(),expected.view(1))
    # Rounding a globally rescaled P is a different operation and must not
    # become the oracle just because its dense expression is shorter.
    wrong=256*round_bf16(alpha)/den
    assert abs(answer.output.item()-wrong.item())>1e-3

def test_final_rv_rounds_both_operands_and_accumulates_fp32():
    q=torch.zeros(1,1,2);k=torch.zeros_like(q);v=torch.tensor([[[1.,2.]]])
    angle=.31;r=torch.tensor([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    answer=attention_tiles(q,k,v,value_rotation=r)
    expected=round_bf16(v)@round_bf16(r).T
    assert torch.equal(answer.output,expected)
    assert answer.output.dtype==torch.float32
    assert not torch.equal(answer.output,v@r.T)
    assert torch.equal(attention_tiles(q,k,v,value_rotation=r.bfloat16()).output,expected)
    # Even identity rounds the normalized source partial before its Cube dot.
    values=torch.zeros(3,1,2);values[0]=1
    identity=attention_tiles(q,torch.zeros(3,1,2),values,value_rotation=torch.eye(2))
    assert torch.equal(identity.output,round_bf16(torch.full((1,1,2),1/3)))

def test_empty_and_partial_visible_rows_have_original_semantics():
    q=torch.ones(2,6,4);k=torch.ones(17,1,4);v=torch.arange(68).view(17,1,4).float()
    mask=torch.zeros(2,17,dtype=torch.bool);mask[0,:3]=True
    answer=attention_tiles(q,k,v,visible=mask)
    assert torch.equal(answer.output[1],torch.zeros(6,4))
    assert torch.isneginf(answer.lse[1]).all()
    assert torch.equal(answer.output[0],v[:3].mean(0).expand(6,4))

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),float('-inf')])
def test_nonfinite_valid_operands_do_not_get_silently_sanitized(bad):
    q=torch.zeros(1,1,4);q[0,0,0]=bad
    with pytest.raises(ValueError,match='nonfinite'):
        attention_tiles(q,torch.zeros(1,1,4),torch.zeros(1,1,4))

def test_threshold_and_contract_predeclared():
    value=load_acceptance()
    assert value['fused_attention']['atol']==value['fused_attention']['rtol']==.005
    assert value['frozen_before_measurement']

def test_complete_fixture_task_sources_and_physical_pages():
    from tools.probe_history_reuse import Shape,make_fixture
    fixture=make_fixture(torch,Shape('bf16_oracle_three_sources',(1,),(65,),64,1,1,False,recent_tokens=1,padded_tokens=2))
    # Independent literal table from documented prepare ABI: history[64,65),
    # exact sink[0,64), current[65,66); one live owner, one padding token.
    tasks=torch.zeros(6,16,dtype=torch.int64)
    for source,(begin,end) in enumerate(((64,65),(0,64),(65,66))):
        tasks[source,:11]=torch.tensor([0,1,0,begin,end,0,source,source,65,0,0])
        tasks[3+source,:11]=torch.tensor([1,-1,0,0,0,0,source,source,0,0,0])
    result=fixture_tasks_reference(fixture,tasks)
    assert result['partial'].shape==(2,6,3,64)
    assert result['live_output_rows']==18 and result['all_task_slots']==6
    assert torch.equal(result['output'][1],torch.zeros(6,64))
    assert torch.isneginf(result['output_lse'][1]).all()
    assert torch.isfinite(result['output'][0]).all()
    with pytest.raises(ValueError,match='all live source'):
        bad=tasks.clone();bad[2,1]=0;fixture_tasks_reference(fixture,bad)

def test_three_splits_keep_empty_segment_owners_and_padding():
    from tools.probe_history_reuse import Shape,make_fixture
    fixture=make_fixture(torch,Shape('bf16_oracle_s3',(1,),(65,),64,1,3,False,recent_tokens=1,padded_tokens=2))
    tasks=torch.zeros(18,16,dtype=torch.int64)
    for source,(begin,end) in enumerate(((64,65),(0,64),(65,66))):
        width=(end-begin+2)//3
        for split in range(3):
            segment=source*3+split;lo=min(begin+width*split,end);hi=min(lo+width,end)
            tasks[segment,:11]=torch.tensor([0,1,0,lo,hi,0,segment,source,65,0,0])
            tasks[9+segment,:11]=torch.tensor([1,-1,0,0,0,0,segment,source,0,0,0])
    result=fixture_tasks_reference(fixture,tasks)
    assert result['all_task_slots']==18 and result['live_output_rows']==54
    assert torch.isfinite(result['output'][0]).all()
    assert torch.isneginf(result['lse'][0,:,1:3]).all()
    assert torch.equal(result['partial'][0,:,1:3],torch.zeros(6,2,64))
    from pathlib import Path
    producer=(Path(__file__).resolve().parents[1]/'csrc/kernels/attention_tasks.cpp').read_text()
    assert 'row.SetValue(6,kind*splits+split)' in producer
    assert 'row.SetValue(7,kind)' in producer
