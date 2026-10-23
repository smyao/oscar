"""Archive #13-25/#86/#97/#148: local clip proxy/codec, not model quality.

An independent codec reference checks the brute-force candidate scores.
Recipe4 must leave the previously fitted rotations, mean and V fields intact.
"""
from copy import deepcopy
import json
import math
import struct

import pytest
import torch

from oscar_ascend import clip_calibration as clip
from oscar_ascend.rotation_calibration import fit_checkpoint_pair, checkpoint_pair_artifact
from tools.official_reference import quantize_rows, dequantize_rows, rotate_bf16


def policy(baseline=.96):
    return {'policy':clip.POLICY,'criterion':clip.CRITERION,'baseline_ratio':baseline,
            'candidate_ratios':list(clip.CANDIDATE_RATIOS),'max_samples':1024,'dequant_modes':[0,1]}


def test_scores_and_selection_match_independent_bf16_codec_bruteforce():
    g=torch.Generator().manual_seed(14);n,d=43,64
    keys=(torch.randn(n,d,generator=g)+torch.linspace(-2,3,d)).bfloat16()
    queries=torch.randn(129,d,generator=g,dtype=torch.float64)
    covariance=queries.T@queries/len(queries)
    rotation=torch.tensor([[(-1.)**((i&j).bit_count()) for j in range(d)] for i in range(d)])/math.sqrt(d)
    mean=keys.double().mean(0).float()
    selected,audit=clip.select_head_clip(keys,covariance,rotation,mean,policy())
    centered=(keys-mean.bfloat16()).bfloat16()
    rotated=rotate_bf16(centered,rotation)
    r=rotation.bfloat16().double();cov_rot=r.T@covariance@r
    expected=[]
    for row in audit['candidates']:
        ratio=row['ratio'];codes,scale,zero=quantize_rows(rotated,ratio)
        scores={}
        for mode,label in ((0,'prefill'),(1,'decode')):
            error=dequantize_rows(codes,scale,zero,mode).double()-rotated.double()
            for suffix,delta in (('raw',error),('centered',error-error.mean(0))):
                score=torch.einsum('nd,de,ne->n',delta,cov_rot,delta).mean()/d
                scores[label+'_'+suffix]=float(score)
                assert row['scores'][label+'_'+suffix]==pytest.approx(float(score),rel=1e-12,abs=1e-14)
        expected.append((ratio,scores))
    baseline=next(scores for ratio,scores in expected if ratio==.96)
    eligible=[(ratio,scores) for ratio,scores in expected if all(scores[k]<=baseline[k] for k in clip.METRICS)]
    winner=min(eligible,key=lambda item:(max(item[1][k]/baseline[k] for k in clip.METRICS),item[0]!=.96,abs(item[0]-.96),item[0]))
    assert selected==winner[0]
    assert all(audit['selected_scores'][key]<=audit['baseline_scores'][key] for key in clip.METRICS)
    assert audit['source']=='local_predeclared_calibration_proxy'


def test_zero_baseline_and_abi_index_dedup_keep_baseline_without_dividing_by_zero(monkeypatch):
    seen=[]
    def scores(rows,cov,ratio):
        seen.append(ratio)
        return dict.fromkeys(clip.METRICS,0.0 if ratio==.96 else 1.0)
    monkeypatch.setattr(clip,'candidate_scores',scores)
    selected,audit=clip.select_head_clip(torch.ones(4,8),torch.eye(8).double(),torch.eye(8),torch.zeros(8),policy())
    assert selected==.96 and audit['selected_scores']==dict.fromkeys(clip.METRICS,0.)
    assert len(audit['candidates'])==len({row['index'] for row in audit['candidates']})==2
    assert len(seen)==2 and .96 in seen
    assert all(not row['eligible'] for row in audit['candidates'] if row['ratio']!=.96)
    json.dumps(audit,allow_nan=False)


def test_ties_preserve_actual_config_baseline_and_sampling_is_bounded(monkeypatch):
    seen=[];n,d=1500,8
    keys=torch.arange(n*d,dtype=torch.float32).reshape(n,d).bfloat16()
    def scores(rows,cov,ratio):
        seen.append(rows.clone())
        return dict.fromkeys(clip.METRICS,1.0)
    monkeypatch.setattr(clip,'candidate_scores',scores)
    chosen,audit=clip.select_head_clip(keys,torch.eye(d).double(),torch.eye(d),torch.zeros(d),policy(.955))
    assert chosen==.955 and audit['sample_count']==1024
    expected_ids=torch.arange(1024)*(n-1)//1023
    assert torch.equal(seen[0],keys[expected_ids])
    for row in audit['candidates']:
        fp32=struct.unpack('f',struct.pack('f',row['ratio']))[0]
        assert row['index']==min(int(fp32*d),d-1)


def test_significantly_indefinite_covariance_cannot_be_clamped_to_a_zero_score():
    rows=torch.tensor([[1.,2.,3.,7.,-1.,5.,9.,11.],[2.,-2.,1.,8.,4.,6.,-3.,10.]]).bfloat16()
    with pytest.raises(ValueError,match='nonnegative.*covariance'):
        clip.candidate_scores(rows,-torch.eye(8,dtype=torch.float64),.96)


def fitted(recipe4):
    from test_center_calibration import ranks,profile
    selected=profile()
    if recipe4:selected.update(recipe_version=4,rotation_k_clip='per_head_calibrated',clip_calibration=policy())
    states=fit_checkpoint_pair(ranks(),layer_names=['model.layers.3.self_attn.attn'],head_dim=8,
        model_path='/model',model_fingerprint='model',global_q_heads=8,global_kv_heads=4,
        tp_size=2,prompt_sha256='prompt',max_tokens=128,calibration_profile=selected,testing=True)
    return states,selected


def test_clip_fit_does_not_change_rotation_mean_or_v_bytes_and_binds_schema2():
    (old_k,old_v),_=fitted(False);(new_k,new_v),profile=fitted(True)
    for key in ('rotation','eigenvalues','k_mean'):
        assert torch.equal(old_k['layers'][3][key],new_k['layers'][3][key])
    for key in ('rotation','eigenvalues'):
        assert torch.equal(old_v['layers'][3][key],new_v['layers'][3][key])
    assert set(old_v['layers'][3])==set(new_v['layers'][3])
    entry=new_k['layers'][3]
    assert entry['k_clip_ratio'].dtype==torch.float64 and entry['k_clip_ratio'].shape==(4,)
    for head,row in enumerate(entry['clip_selection']['heads']):
        assert row['global_kv_head']==head and row['selected_ratio']==float(entry['k_clip_ratio'][head])
        assert all(row['selected_scores'][key]<=row['baseline_scores'][key] for key in clip.METRICS)
    artifact=checkpoint_pair_artifact(new_k,new_v,layer_names=['model.layers.3.self_attn.attn'],
        head_dim=8,model_fingerprint='model',calibration_profile=profile,testing=True)
    assert torch.equal(artifact['layers']['model.layers.3.self_attn.attn']['k_clip_ratio'],entry['k_clip_ratio'])
    assert artifact['layers']['model.layers.3.self_attn.attn']['clip_selection']==entry['clip_selection']


@pytest.mark.parametrize('change,match',[
    (lambda k,v:k['layers'][3]['k_clip_ratio'].__setitem__(0,.5),'decision'),
    (lambda k,v:k['layers'][3].pop('clip_selection'),'provenance'),
    (lambda k,v:v['layers'][3].update(k_clip_ratio=torch.ones(4,dtype=torch.float64)),'not permitted'),
])
def test_pair_rejects_ratio_provenance_drift_and_clip_fields_on_v(change,match):
    (k,v),profile=fitted(True);change(k,v)
    with pytest.raises(ValueError,match=match):
        checkpoint_pair_artifact(k,v,layer_names=['model.layers.3.self_attn.attn'],head_dim=8,
            model_fingerprint='model',calibration_profile=profile,testing=True)


def test_changed_local_selection_contract_cannot_be_silently_accepted():
    changed=deepcopy(policy());changed['criterion']='centered_only'
    with pytest.raises(ValueError,match='frozen local'):
        clip.validate_policy(changed)
