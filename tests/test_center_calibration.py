"""Archive G19-G24/#23-25/#86/#97: source-derived per-head center fitting.

Official fit_oscar2_variants.py center: M_q trace/D normalization; each head's
S_v is independently decomposed without that normalization. CPU only.
"""
from copy import deepcopy
import math

import pytest
import torch

from oscar_ascend.rotation_calibration import (
    OFFICIAL_COMMIT, OfflineLayerAccumulator, fit_checkpoint_pair,
    checkpoint_pair_artifact, publish_checkpoint_pair, verified_pair_files,
)

NAME = 'model.layers.3.self_attn.attn'


def profile(center=True):
    result = {'official_commit': OFFICIAL_COMMIT, 'model_fingerprint': 'model',
              'recipe_version': 3 if center else 2, 'max_tokens': 128,
              'model_execution': {'dtype': 'bfloat16', 'mamba_cache_dtype': 'bfloat16',
                                  'mamba_ssm_cache_dtype': 'bfloat16'}}
    if center: result['rotation_variant'] = 'perhead_center'
    return result


def ranks():
    result = []
    for rank in range(2):
        g = torch.Generator().manual_seed(211 + rank)
        q = torch.randn(37, 4, 8, generator=g).bfloat16()
        k = (torch.randn(37, 2, 8, generator=g) + rank * 3).bfloat16()
        k[:, 1] = (k[:, 1].float() + 7).bfloat16()
        v = (torch.randn(37, 2, 8, generator=g) * (rank + 2)).bfloat16()
        acc = OfflineLayerAccumulator(head_dim=8, local_kv_heads=2, local_q_heads=4,
                                      token_budget=37, testing=True)
        acc.observe(q, k, v)
        result.append({'rank': rank, 'tp_size': 2, 'global_q_heads': 8,
                       'global_kv_heads': 4, 'prompt_sha256': 'prompt',
                       'model_fingerprint': 'model', 'layers': {3: acc.export_payload()}})
    return result


def fit(data=None, center=True):
    return fit_checkpoint_pair(ranks() if data is None else data, layer_names=[NAME],
        head_dim=8, model_path='/model', model_fingerprint='model', global_q_heads=8,
        global_kv_heads=4, tp_size=2, prompt_sha256='prompt', max_tokens=128,
        calibration_profile=profile(center), testing=True)


def independent_basis(covariance, normalize):
    cov = (covariance + covariance.T) / 2
    if normalize: cov = cov / (torch.trace(cov) / len(cov))
    vals, vecs = torch.linalg.eigh((cov + cov.T) / 2)
    dim = len(cov)
    h = torch.tensor([[(-1.) ** ((i & j).bit_count()) for j in range(dim)]
                      for i in range(dim)], dtype=torch.float64) / math.sqrt(dim)
    br = torch.tensor([int(f'{i:03b}'[::-1], 2) for i in range(dim)])
    perm = torch.empty(dim, dtype=torch.long); perm[br] = torch.argsort(vals, descending=True)
    return (vecs @ h)[:, perm].float(), vals.float()


def test_perhead_center_matches_independent_q_and_sst_equations_without_head_averaging():
    data = ranks()
    key_state, value_state = fit(list(reversed(data)))
    assert key_state['format_version'] == value_state['format_version'] == 2
    assert key_state['source_grouping'] == value_state['source_grouping'] == 'layer_kv_head'
    assert key_state['global_kv_heads'] == value_state['global_kv_heads'] == 4
    for rank in data:
        entry = rank['layers'][3]
        for head in range(2):
            global_head = rank['rank'] * 2 + head
            mq = entry['q_gram'][head] / (37 * 2)
            keys, values = entry['k'][:, head].double(), entry['v'][:, head].double()
            energy = ((keys @ mq) * keys).sum(-1)
            sv = values.T @ (values * energy[:, None]) / energy.sum().clamp_min(1e-12)
            for state, covariance, normalize in ((key_state, mq, True), (value_state, sv, False)):
                rotation, eigenvalues = independent_basis(covariance, normalize)
                torch.testing.assert_close(state['layers'][3]['rotation'][global_head], rotation, atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(state['layers'][3]['eigenvalues'][global_head], eigenvalues, atol=1e-6, rtol=1e-6)
            assert torch.equal(key_state['layers'][3]['k_mean'][global_head], keys.mean(0).float())
    assert key_state['layers'][3]['rotation'].shape == (4, 8, 8)
    assert not torch.equal(key_state['layers'][3]['rotation'][0], key_state['layers'][3]['rotation'][1])
    assert not torch.equal(key_state['layers'][3]['k_mean'][0], key_state['layers'][3]['k_mean'][3])
    assert not torch.allclose(value_state['layers'][3]['eigenvalues'].sum(-1), torch.full((4,), 8.))


def test_old_profile_keeps_v1_shared_geometry_and_no_center_fields():
    k, v = fit(center=False)
    for state in (k, v):
        assert state['format_version'] == 1 and state['source_grouping'] == 'layer'
        assert state['layers'][3]['rotation'].shape == (8, 8)
        assert 'global_kv_heads' not in state and 'k_mean' not in state['layers'][3]
        assert state['calibration']['calibration_profile'] == profile(False)


def artifact(k, v):
    return checkpoint_pair_artifact(k, v, layer_names=[NAME], head_dim=8,
        model_fingerprint='model', calibration_profile=profile(), testing=True)


def test_schema2_pair_publication_preserves_global_head_order_and_mean(tmp_path):
    k, v = fit(); combined = artifact(k, v)
    assert combined['format_version'] == 2 and combined['source_grouping'] == 'layer_kv_head'
    assert combined['global_kv_heads'] == 4 and combined['rotation_variant'] == 'perhead_center'
    assert torch.equal(combined['layers'][NAME]['k_mean'], k['layers'][3]['k_mean'])
    publish_checkpoint_pair(k, v, tmp_path, artifact=combined)
    _, _, loaded = verified_pair_files(tmp_path)
    assert torch.equal(loaded['layers'][NAME]['Rk'], combined['layers'][NAME]['Rk'])
    assert torch.equal(loaded['layers'][NAME]['k_mean'], combined['layers'][NAME]['k_mean'])


@pytest.mark.parametrize('change,match', [
    (lambda k,v:k['layers'][3].pop('k_mean'), 'mean'),
    (lambda k,v:k['layers'][3].update(k_mean=torch.zeros(8)), 'mean'),
    (lambda k,v:k['layers'][3]['k_mean'].fill_(float('nan')), 'mean'),
    (lambda k,v:k['layers'][3]['rotation'][1].mul_(2), 'orthogonal'),
    (lambda k,v:v.update(global_kv_heads=2), 'global_kv_heads'),
    (lambda k,v:k.update(format_version=1), 'schema'),
    (lambda k,v:k['layers'][3].update(q_rotation=torch.eye(8)), 'companion'),
])
def test_malformed_or_unsupported_center_pair_fails_closed(change,match):
    k,v=fit(); change(k,v)
    with pytest.raises(ValueError,match=match): artifact(k,v)


def test_zero_query_metric_cannot_silently_use_identity_or_shared_rotation():
    data=ranks();data[0]['layers'][3]['q_gram'][0].zero_()
    with pytest.raises(ValueError,match='positive finite trace'):fit(data)


def test_center_requires_disjoint_kv_head_shards():
    with pytest.raises(ValueError,match='disjoint.*replicated'):
        fit_checkpoint_pair([{'rank':i} for i in range(4)],layer_names=[NAME],head_dim=8,
            model_path='/model',model_fingerprint='model',global_q_heads=8,global_kv_heads=2,
            tp_size=4,prompt_sha256='prompt',max_tokens=128,calibration_profile=profile(),testing=True)
