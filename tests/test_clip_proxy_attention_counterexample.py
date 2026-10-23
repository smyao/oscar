"""Archive #13-20/#148: lower calibration proxy is not attention accuracy.

This D256 construction exercises the actual recipe4 selector. It is a local
mathematical counterexample, not measured model activations or LongBench.
"""
import pytest
import torch
from oscar_ascend.clip_calibration import select_head_clip
from oscar_ascend.rotations import calibration_cache_identity


def quantized(keys, ratio, mode):
    # Independent scalar-form uniform codec, with explicit BF16 boundaries.
    threshold=keys.float().abs().kthvalue(min(int(ratio*256),255)+1,dim=-1).values[:,None]
    x=keys.float().clamp(-threshold,threshold)
    low=x.min(-1,keepdim=True).values;high=x.max(-1,keepdim=True).values
    scale=(high-low).clamp_min(1e-8)/3;zero=-low/scale
    code=(x/scale+zero+.5).floor().clamp(0,3)
    if mode==1:
        scale=scale.bfloat16().float();zero=zero.bfloat16().float()
        difference=(code-zero).bfloat16().float()
    else:difference=code-zero
    return (difference*scale).bfloat16().float()


@pytest.mark.parametrize('mode',[0,1])
def test_actual_proxy_selector_can_improve_all_four_scores_but_damage_attention(mode):
    row=torch.ones(256);row[:11]=torch.tensor([100.,-100.]*5+[100.])
    keys=torch.stack([row,-row]).bfloat16()
    profile=calibration_cache_identity({'rotation_variant':'perhead_center',
        'rotation_k_clip':'per_head_calibrated','mamba_cache_dtype':'bfloat16',
        'mamba_ssm_cache_dtype':'bfloat16'},'analytic-counterexample')
    ratio,proof=select_head_clip(keys,torch.eye(256,dtype=torch.float64),torch.eye(256),
                                 torch.zeros(256),profile['clip_calibration'])
    assert ratio==.94
    assert all(proof['selected_scores'][name]<proof['baseline_scores'][name]
               for name in proof['baseline_scores'])
    # norm(Q)=sqrt(D); covariance I is compatible with a population of
    # equally weighted sqrt(D)*coordinate queries. This is one such query.
    query=torch.zeros(256);query[0]=16
    values=torch.tensor([1.,-1.])
    def output(k):return (torch.softmax((k.float()@query)/16,0)*values).sum()
    expected=output(keys);baseline=output(quantized(keys,.96,mode))
    selected=output(quantized(keys,ratio,mode))
    assert float(expected)==float(baseline)==1.0
    torch.testing.assert_close(selected,torch.tanh(torch.tensor(1.)),atol=1e-7,rtol=0)
    assert float((selected-expected).abs())>.23
