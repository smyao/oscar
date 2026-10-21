"""Archive #126/#145/#148/#154/#164: probe Q producer and compact arena ABI.

Host orchestration tests only. The real producer math/device verification is
separate; these guards prevent direct BF16 probes from using old FP32 inputs.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tools import probe_bf16_history as probe


@pytest.mark.parametrize('route,expected',[
    ('mdl',1212416),('mdl_q1',1212416),('c4',2887680),('c16',5259264),
    ('general',1179648),('balanced',1179648)])
def test_compact_probe_workspace_matches_public_cpp_contract(route,expected):
    assert probe.candidate_workspace_per_core(route)==expected
    source=(probe.ROOT/'csrc/include/oscar_bf16_history_launch.h').read_text()
    assert f'=={expected});' in source


@pytest.mark.parametrize('corrupt',[False,True])
def test_real_producer_is_called_and_candidate_retains_unrounded_fp32(monkeypatch,corrupt):
    case=probe.case_plan()[0]
    case=replace(case,shape=replace(case.shape,qlens=(1,),contexts=(20,),
        padded_tokens=4,corrupt_qr=corrupt,corrupt_qr_token=0))
    fixture=probe.make_fixture(torch,case)
    tensors={key:value.clone() for key,value in fixture['cpu'].items()}
    original=tensors['qr'].clone()
    old,new=[probe.allocate(torch,fixture,'cpu',1,case.route,candidate=value) for value in (False,True)]
    calls=[]
    def producer(q,rotation,out,status,hadamard,slots):
        calls.append((q,rotation,hadamard,slots))
        assert out.dtype==torch.float32 and out.shape==(4,6,256)
        # Deliberately distinct marker, not a mathematical producer oracle.
        out.zero_();out[:,:,:256].fill_(7);status.zero_()
    ops=SimpleNamespace(rotate_vector_init_out=producer)
    probe.produce_bf16_query(ops,tensors,fixture,new)
    assert len(calls)==1 and calls[0][0] is tensors['q']
    assert calls[0][1] is tensors['rk_transpose'] and calls[0][2] is True
    assert calls[0][3] is tensors['slots']
    assert torch.equal(tensors['qr'].view(torch.int32),original.view(torch.int32))
    if corrupt:
        assert torch.isnan(new['qr_residual'][0,0,0])
    observed=[]
    monkeypatch.setattr(probe.shared,'_launch',lambda o,name,inputs,*a:observed.append((name,inputs['qr'])))
    for name,buffers in zip(probe.ROUTES[case.route][:2],(old,new)):
        probe.launch(ops,name,tensors,fixture,buffers,1,1,merge=False)
    assert observed[0][1] is tensors['qr'] and observed[0][1].dtype==torch.float32
    assert observed[1][1] is new['qr_residual'] and observed[1][1].dtype==torch.float32
    assert len(calls)==1  # CV-only launches do not hide producer work in timing.


def test_unknown_route_cannot_guess_workspace_bytes():
    with pytest.raises(ValueError):probe.candidate_workspace_per_core('unknown')


@pytest.mark.parametrize('defect',[None,'operand','nonfinite','inactive_row','old_abi'])
def test_full_pipeline_query_comparison_preserves_meaningful_bitwise_domains(defect):
    from tools import bf16_history_pipeline as full
    old=torch.randn(3,6,256);old[2]=0
    new=old.clone();slots=torch.tensor([0,1,-1])
    if defect=='operand':new[0,0,0]=17
    elif defect=='nonfinite':new[0,0,0]=float('nan')
    elif defect=='inactive_row':old[2,0,0]=new[2,0,0]=1
    elif defect=='old_abi':new=torch.zeros(3,6,272,dtype=torch.bfloat16)
    if defect:
        with pytest.raises(RuntimeError):full.check_query_pair(torch,old,new,slots)
    else:full.check_query_pair(torch,old,new,slots)
