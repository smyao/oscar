"""Per-head constants must preserve global-head/TP identity and mean binding."""
import copy

import pytest
import torch

from oscar_ascend.rotations import (build_hadamard_artifact, validate_artifact,
    prepare_device_rotations, calibration_cache_identity)

NAME = 'model.layers.3.self_attn.attn'
OPTIONS = dict(layer_names=[NAME], head_dim=64, model_fingerprint='model', device='cpu', testing=True)


def artifact():
    value = build_hadamard_artifact(**OPTIONS)
    value.update(format_version=2, source_grouping='layer_kv_head', global_kv_heads=4,
                 rotation_variant='perhead_center', objective='qqt_sst_r_h_pbr')
    base = value['layers'][NAME]['Rk']
    value['layers'][NAME].update(Rk=torch.stack([base.roll(i,0) for i in range(4)]),
        Rv=torch.stack([base.roll(i,1) for i in range(4)]),
        k_mean=torch.arange(256,dtype=torch.float32).reshape(4,64))
    return value


def test_global_per_head_constants_are_selected_by_tp_rank():
    valid = validate_artifact(artifact(), **OPTIONS)
    for rank in range(4):
        selected = prepare_device_rotations(valid, device='cpu', testing=True, tp_rank=rank, tp_size=4)[NAME]
        assert torch.equal(selected.key_transposed.T, valid['layers'][NAME]['Rk'][rank])
        assert torch.equal(selected.inverse_value_transposed, valid['layers'][NAME]['Rv'][rank])
        assert torch.equal(selected.key_mean, valid['layers'][NAME]['k_mean'][rank])
        assert selected.key_mean.is_contiguous()
    with pytest.raises(ValueError, match='one KV head'):
        prepare_device_rotations(valid, device='cpu', testing=True, tp_rank=0, tp_size=2)


@pytest.mark.parametrize('fault',['missing_mean','nan_mean','wrong_heads','bad_matrix'])
def test_center_artifact_rejects_incomplete_or_mismatched_constants(fault):
    value=artifact(); layer=value['layers'][NAME]
    if fault=='missing_mean':del layer['k_mean']
    if fault=='nan_mean':layer['k_mean'][0,0]=float('nan')
    if fault=='wrong_heads':value['global_kv_heads']=3
    if fault=='bad_matrix':layer['Rk'][2,0,0]=5
    with pytest.raises(ValueError):validate_artifact(value, **OPTIONS)


def test_center_has_new_cache_recipe_without_relabelling_old_v1():
    config=dict(mamba_cache_dtype='bfloat16',mamba_ssm_cache_dtype='bfloat16')
    before=copy.deepcopy(config)
    old=calibration_cache_identity(config,'model')
    new=calibration_cache_identity({**config,'rotation_variant':'perhead_center'},'model')
    assert config==before and old['recipe_version']==2 and 'rotation_variant' not in old
    assert new['recipe_version']==3 and new['rotation_variant']=='perhead_center'
    assert {k:v for k,v in new.items() if k not in ('recipe_version','rotation_variant')} == {k:v for k,v in old.items() if k!='recipe_version'}
