"""Archive #86/#97/#165/#166: automatic per-head center PT migration.

All captures are small CPU test payloads. No target/model/NPU execution or
quality claim. Old cache bytes and original capture profiles remain intact.
"""
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from oscar_ascend import rotations
from oscar_ascend.rotation_calibration import (
    checkpoint_pair_artifact, fit_checkpoint_pair, publish_checkpoint_pair,
    verified_pair_files,
)
from tools import install_serve, observe_serve
from tools.prepare_rotations import prepare
from tools.serving_variants import variant_config, runtime_feature_summary
from tools.target_cli import serve_argv
from test_official_rotation_calibration import model, rank_states, NAME


@pytest.fixture
def environment(tmp_path, monkeypatch):
    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    old = model(model_dir)
    new = {**old, 'rotation_variant': 'perhead_center'}
    _, _, fingerprint = rotations.checkpoint_identity(model_dir)
    original_path = rotations.rotation_artifact_path
    monkeypatch.delenv('OSCAR_ROTATION_ARTIFACT', raising=False)
    monkeypatch.setattr(rotations, 'rotation_artifact_path',
                        lambda config, fp: original_path(config, fp, root=tmp_path))
    return old, new, fingerprint


def capture_rows(config, fingerprint, directory):
    directory.mkdir(parents=True, exist_ok=True)
    paths, hashes = [], {}
    for payload in rank_states(fingerprint, n=13):
        path = directory / f"rank{payload['rank']}.pt"
        torch.save(payload, path)
        paths.append(str(path))
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'status': 'native_post_rope_capture_complete', 'model_fingerprint': fingerprint,
            'layer_ids': [3], 'prompt_sha256': 'prompt', 'token_count': 13,
            'rank_paths': paths, 'rank_sha256': hashes, 'tp_size': 2,
            'global_q_heads': 4, 'global_kv_heads': 2,
            'calibration_profile': rotations.calibration_cache_identity(config, fingerprint)}


def publish_old(config, fingerprint, *, with_capture=True):
    path = rotations.rotation_artifact_path(config, fingerprint)
    capture = capture_rows(config, fingerprint, path.parent / 'capture-retained')
    profile = rotations.calibration_cache_identity(config, fingerprint)
    ranks = [torch.load(p, weights_only=True, map_location='cpu') for p in capture['rank_paths']]
    k, v = fit_checkpoint_pair(ranks, layer_names=[NAME], head_dim=8, model_path=config['model'],
        model_fingerprint=fingerprint, global_q_heads=4, global_kv_heads=2, tp_size=2,
        prompt_sha256='prompt', max_tokens=profile['max_tokens'], calibration_profile=profile, testing=True)
    artifact = checkpoint_pair_artifact(k, v, layer_names=[NAME], head_dim=8,
        model_fingerprint=fingerprint, calibration_profile=profile, testing=True)
    if with_capture:
        artifact['capture_manifest'] = capture
    publish_checkpoint_pair(k, v, path.parent, artifact=artifact)
    return path, capture


def frozen_files(path):
    return {p: p.read_bytes() for p in path.parent.rglob('*') if p.suffix in {'.pt', '.json'}}


def forbid_capture(*args):
    pytest.fail('authenticated retained samples must avoid another model forward')


def test_candidate_enables_center_baseline_native_restore_shared_without_argv_changes():
    base = json.loads((install_serve.ROOT / 'configs/target.json').read_text())
    candidate = variant_config(base, 'candidate')
    assert candidate['rotation_variant'] == 'perhead_center'
    assert 'rotation_k_clip' not in candidate
    assert 'rotation_variant' not in base
    # These are model/serving options; fitting recipe must not alter native argv.
    assert serve_argv(candidate) == serve_argv(variant_config(base, 'baseline'))
    for variant in ('baseline', 'native'):
        assert 'rotation_variant' not in variant_config(candidate, variant)
        assert 'rotation_k_clip' not in variant_config(candidate, variant)
    summary = runtime_feature_summary(candidate, 'candidate')
    assert summary['rotation_grouping'] == 'layer_kv_head'
    assert summary['key_centering'] is True and summary['calibration_recipe_version'] == 5
    assert summary['rotation_refinement'] == 'optr'
    assert summary['clip_policy'] == 'scalar'
    assert summary['head_clip_experiment_enabled'] is False
    assert summary['rotation_variant'] == 'perhead_center'
    data_free = variant_config({**base, 'rotation_method': 'hadamard'}, 'candidate')
    assert 'rotation_variant' not in data_free
    assert 'rotation_k_clip' not in data_free
    assert runtime_feature_summary(data_free, 'candidate')['calibration_recipe_version'] is None


@pytest.mark.parametrize('entrypoint', [install_serve, observe_serve])
def test_both_one_key_plans_show_the_selected_center_recipe(entrypoint, capsys):
    assert entrypoint.main(['--variant', 'candidate', '--plan']) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan['runtime_selection']['rotation_variant'] == 'perhead_center'
    assert plan['runtime_selection']['rotation_grouping'] == 'layer_kv_head'
    assert plan['runtime_selection']['key_centering'] is True
    assert plan['runtime_selection']['clip_policy'] == 'scalar'
    assert plan['runtime_selection']['head_clip_experiment_enabled'] is False
    assert plan['runtime_selection']['calibration_recipe_version'] == 5
    assert plan['runtime_selection']['max_num_batched_tokens'] == 16384


def test_native_budget_candidate_clears_residual_clip_and_keeps_recipe3(environment):
    _, center, fingerprint = environment
    center_path, _ = publish_old(center, fingerprint)
    experimental = {**center, 'rotation_k_clip': 'per_head_calibrated'}
    clip_path, _ = publish_old(experimental, fingerprint)
    before = {**frozen_files(center_path), **frozen_files(clip_path)}
    candidate = variant_config(experimental, 'candidate')
    assert candidate['use_native_token_budget'] is True
    assert candidate.get('rotation_refinement') is None
    assert 'rotation_k_clip' not in candidate
    assert candidate['rotation_variant'] == 'perhead_center'
    assert rotations.rotation_artifact_path(candidate, fingerprint) == center_path
    assert center_path != clip_path
    result = prepare(candidate, capture_runner=forbid_capture, testing=True)
    assert result['reused'] and result['recipe_version'] == 3
    assert Path(result['artifact']) == center_path
    loaded = rotations.load_active_artifact(candidate, device='cpu', testing=True)
    assert 'k_clip_ratio' not in loaded['layers'][NAME]
    assert all(path.read_bytes() == data for path, data in before.items())
    # Explicit research configuration retains the quarantined capability.
    research = variant_config(experimental, None)
    assert research['rotation_k_clip'] == 'per_head_calibrated'
    summary = runtime_feature_summary(research, None)
    assert summary['calibration_recipe_version'] == 4
    assert summary['head_clip_experiment_enabled'] is True


def test_fixed_candidate_uses_optr_with_both_optr_and_center_caches(environment, monkeypatch):
    """User 2026-10-28: existing recipe3 must not bypass the OptR candidate."""
    from oscar_ascend.optr_artifacts import publish_artifact
    from test_optr_integration import artifact as refined_artifact
    _, center, fingerprint = environment
    candidate = variant_config({**center, 'use_native_token_budget': False,
                                'max_num_batched_tokens': 16384,
                                'k_clip_ratio': .96, 'v_clip_ratio': .92}, 'candidate')
    seed = {k: v for k, v in candidate.items() if k != 'rotation_refinement'}
    center_path, _ = publish_old(seed, fingerprint)
    refined_path = rotations.rotation_artifact_path(candidate, fingerprint)
    publish_artifact(refined_artifact(candidate), refined_path)
    before = {**frozen_files(center_path), **frozen_files(refined_path)}
    stale = {**candidate, 'optr_calibration': {'steps': 80}}
    # New official default must not authenticate the old 16-step fit as done.
    assert rotations.rotation_artifact_path(
        {**candidate, 'optr_calibration': {'steps': 16}}, fingerprint) != refined_path
    selected = variant_config(stale, 'candidate')
    assert selected == stale and selected['oscar_dequant_policy'] == 'fp32_metadata'
    assert selected['k_clip_ratio'] == candidate['k_clip_ratio']
    assert selected['v_clip_ratio'] == candidate['v_clip_ratio']
    assert rotations.rotation_artifact_path(selected, fingerprint) == refined_path != center_path
    result = prepare(selected, capture_runner=forbid_capture, testing=True)
    assert result['reused'] and result['recipe_version'] == 5
    assert Path(result['artifact']) == refined_path and result['source'] == 'optr_authenticated_cache'
    loaded = rotations.load_active_artifact(selected, device='cpu', testing=True)
    assert loaded['calibration_profile']['recipe_version'] == 5
    # An explicit conflicting artifact is not silently ignored or relabelled.
    with pytest.raises(ValueError):
        prepare({**selected, 'rotation_artifact': str(center_path)}, testing=True)
    monkeypatch.setenv('OSCAR_ROTATION_ARTIFACT', str(center_path))
    with pytest.raises(ValueError, match='authenticated OptR manifest required'):
        rotations.load_active_artifact(selected, device='cpu', testing=True)
    assert all(path.read_bytes() == data for path, data in before.items())


def test_recipe2_retained_samples_refit_to_distinct_center_cache_and_then_reuse(environment):
    old, new, fingerprint = environment
    old_path, capture = publish_old(old, fingerprint)
    before = frozen_files(old_path)
    capture_before = copy.deepcopy(capture)
    first = prepare(new, capture_runner=forbid_capture, testing=True)
    path = Path(first['artifact'])
    assert path != old_path and first['capture_reused'] is True
    assert first['model_forward_executed'] is False
    assert first['source'] == 'official_qqt_sst_refit_retained_samples'
    _, _, artifact = verified_pair_files(path.parent)
    expected_profile = rotations.calibration_cache_identity(new, fingerprint)
    assert artifact['format_version'] == 2 and artifact['source_grouping'] == 'layer_kv_head'
    assert artifact['layers'][NAME]['Rk'].shape == artifact['layers'][NAME]['Rv'].shape == (2, 8, 8)
    assert artifact['layers'][NAME]['k_mean'].shape == (2, 8)
    assert artifact['capture_manifest'] == capture_before
    assert artifact['capture_manifest']['calibration_profile']['recipe_version'] == 2
    assert artifact['capture_fit_profile'] == artifact['calibration_profile'] == expected_profile
    assert artifact['refit_provenance']['source_artifact_sha256'] == hashlib.sha256(old_path.read_bytes()).hexdigest()
    assert all(p.read_bytes() == data for p, data in before.items())
    second = prepare(new, capture_runner=forbid_capture, testing=True)
    assert second['reused'] is True and second['recipe_version'] == 3
    loaded = rotations.load_active_artifact(new, device='cpu', testing=True)
    assert loaded['calibration_profile'] == expected_profile


@pytest.mark.parametrize('missing', ['artifact', 'manifest', 'rank_file', 'rank_hash'])
def test_unavailable_retained_samples_trigger_one_normal_new_recipe_capture(environment, missing):
    old, new, fingerprint = environment
    if missing != 'artifact':
        path, capture = publish_old(old, fingerprint, with_capture=missing != 'manifest')
        if missing == 'rank_file':
            Path(capture['rank_paths'][0]).unlink()
        elif missing == 'rank_hash':
            k, v, artifact = verified_pair_files(path.parent)
            artifact['capture_manifest'].pop('rank_sha256')
            publish_checkpoint_pair(k, v, path.parent, artifact=artifact)
        before = frozen_files(path)
    calls = []
    def capture(_config_path, directory):
        calls.append(directory)
        return capture_rows(new, fingerprint, directory)
    result = prepare(new, capture_runner=capture, testing=True)
    assert len(calls) == 1 and result['capture_reused'] is False
    assert result['model_forward_executed'] is True
    _, _, artifact = verified_pair_files(Path(result['artifact']).parent)
    assert artifact['capture_manifest']['calibration_profile']['recipe_version'] == 3
    assert 'refit_provenance' not in artifact
    assert prepare(new, capture_runner=forbid_capture, testing=True)['reused'] is True
    if missing != 'artifact':
        assert all(p.read_bytes() == data for p, data in before.items())


@pytest.mark.parametrize('corruption', ['rank_hash', 'rank_identity', 'old_pair_hash', 'source_profile'])
def test_corrupt_retained_provenance_fails_instead_of_silently_resampling(environment, corruption):
    old, new, fingerprint = environment
    path, capture = publish_old(old, fingerprint)
    if corruption == 'rank_hash':
        with Path(capture['rank_paths'][0]).open('ab') as stream:
            stream.write(b'changed')
    elif corruption == 'old_pair_hash':
        with path.open('ab') as stream:
            stream.write(b'changed')
    else:
        k, v, artifact = verified_pair_files(path.parent)
        if corruption == 'source_profile':
            artifact['capture_manifest']['calibration_profile']['model_execution']['quantization'] = 'ascend'
        else:
            rank_path = Path(capture['rank_paths'][0])
            payload = torch.load(rank_path, map_location='cpu', weights_only=True)
            payload['model_fingerprint'] = 'foreign-model'
            torch.save(payload, rank_path)
            artifact['capture_manifest']['rank_sha256'][str(rank_path)] = hashlib.sha256(rank_path.read_bytes()).hexdigest()
        publish_checkpoint_pair(k, v, path.parent, artifact=artifact)
    with pytest.raises((ValueError, RuntimeError)):
        prepare(new, capture_runner=forbid_capture, testing=True)
    assert not rotations.rotation_artifact_path(new, fingerprint).exists()


def test_explicit_old_artifact_is_not_relabelled_or_replaced(environment):
    old, new, fingerprint = environment
    path, _ = publish_old(old, fingerprint)
    before = frozen_files(path)
    with pytest.raises(ValueError):
        prepare(new, output=path, capture_runner=forbid_capture, testing=True)
    assert all(p.read_bytes() == data for p, data in before.items())


def test_new_matching_path_corruption_is_not_hidden_by_old_sample_refit(environment):
    old, new, fingerprint = environment
    publish_old(old, fingerprint)
    report = prepare(new, capture_runner=forbid_capture, testing=True)
    path = Path(report['artifact'])
    with path.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises((ValueError, RuntimeError)):
        prepare(new, capture_runner=forbid_capture, testing=True)
