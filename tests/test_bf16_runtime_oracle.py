"""Archive #126/#145/#148: lifecycle source ranges versus real prepare ABI."""
import hashlib
import torch
from tools.probe_history_reuse import Shape,make_fixture
from tools.bf16_reference import fixture_tasks_reference
from tools.bf16_runtime_oracle import sampled_fixture_expected


def test_sampled_runtime_oracle_matches_observed_complete_q4_task_table():
    fixture=make_fixture(torch,Shape('runtime_oracle_q4',(4,4),(641,701),256,1,1,False,padded_tokens=8))
    # Literal table matches the previously completed official CPU prepare
    # export at C641/701. The observed SHA binds every follower/owner field.
    tasks=torch.zeros(24,16,dtype=torch.int64)
    for token in range(8):
        request=token//4;context=(641,701)[request];begin=request*4
        for kind in range(3):
            count=4 if token%4==0 else 0
            first,last=((64,context+4-256),(0,319),(context,context+4))[kind] if count else (0,0)
            tasks[token*3+kind,:11]=torch.tensor([token,count,0,first,last,request,kind,kind,context,begin,0])
    import ctypes
    data=ctypes.string_at(tasks.data_ptr(),tasks.numel()*8)
    assert hashlib.sha256(data).hexdigest()=='3a7cc923b7947a11c03ec4174703daf1badc44432eb1a169665f6154a6129f72'
    whole=fixture_tasks_reference(fixture,tasks)
    sampled=sampled_fixture_expected(fixture,native_current=False)
    for i,(out,lse) in sampled.items():
        torch.testing.assert_close(out,whole['output'][i],atol=.005,rtol=.005)
        torch.testing.assert_close(lse,whole['output_lse'][i],atol=.005,rtol=.005)


def test_fresh_native_current_reuses_independent_dense_BF16_output_contract():
    fixture=make_fixture(torch,Shape('fresh_runtime_oracle',(4,17),(65,0),256,1,1,False))
    result=sampled_fixture_expected(fixture,native_current=(False,True))
    assert set(result)==set(fixture['expected'])
    for i in range(4,21):
        assert torch.equal(result[i][0],fixture['expected'][i][0].bfloat16().float())
        assert torch.equal(result[i][1],fixture['expected'][i][1])
