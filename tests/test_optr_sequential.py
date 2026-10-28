"""Official proxy parity, guarded NPU selection, and unchanged deployment gate."""
import copy
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from oscar_ascend import optr_sequential as sequential
from oscar_ascend.optr_policy import optr_policy, legacy_capture_policy, legacy_capture_profile
from test_optr_calibration import trace


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def policy(**kwargs):
    return {**optr_policy({}), 'sink_tokens': 2, 'recent_tokens': 4, **kwargs}


def direct_loss(rotation, sample, mean, phase, p, rk=None):
    # Independent literal formulation: full output projection, per query head,
    # no Gram caching. Official FP32 proxy adapted only for captured gates/rows.
    def effective(value, r, clip):
        x = value @ r
        index = min(int(float(torch.tensor(clip)) * x.shape[-1]), x.shape[-1] - 1)
        threshold = x.detach().abs().sort(-1).values[:, index:index + 1]
        x = x.clamp(-threshold, threshold)
        low, high = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
        scale = (high - low).clamp_min(1e-8) / 3
        z = -low / scale
        a = x / scale + z
        dq = (a + ((a + .5).floor().clamp(0, 3) - a).detach() - z) * scale
        result = dq @ r.T
        exact = torch.arange(len(x)) < p['sink_tokens']
        exact |= torch.arange(len(x)) >= len(x) - p['recent_tokens']
        return torch.where(exact[:, None], value, result)
    k = sample['k'].float() - mean
    v = sample['v'].float()
    mask = torch.arange(len(k))[None] > sample['positions'][:, None]
    kh = effective(k, rotation if phase == 'k' else rk, p['k_clip'])
    dv = None if phase == 'k' else effective(v, rotation, p['v_clip']) - v
    vals = []
    for h in range(sample['q'].shape[1]):
        q = sample['q'][:, h].float()
        pq = ((q @ kh.T) / k.shape[-1] ** .5).masked_fill(mask, -torch.inf).softmax(-1)
        if phase == 'k':
            pt = ((q @ k.T) / k.shape[-1] ** .5).masked_fill(mask, -torch.inf).softmax(-1)
            kl = (pt * (pt.clamp_min(1e-12).log() - pq.clamp_min(1e-12).log())).sum(-1).mean()
            error = (pq - pt) @ v
        else:
            kl = 0.
            error = pq @ dv
        error = (error * sample['gate'][:, h].float()) @ sample['wo'][h].T
        vals.append(kl + error.square().mean() * (p['lambda_output'] if phase == 'k' else 1))
    return torch.stack(vals).mean()


@pytest.mark.parametrize('phase', ['k', 'v'])
@pytest.mark.parametrize('gate', [True, False])
def test_batched_gram_loss_and_gradient_match_direct_official_formula(phase, gate):
    s = trace(17, length=81, dim=8, queries=7, group=3)
    if not gate:
        s['gate'].fill_(1)
    mean = torch.linspace(-.2, .3, 8)
    p = policy()
    base = torch.linalg.qr(torch.randn(8, 8, generator=torch.Generator().manual_seed(14))).Q
    reference = base.clone().requires_grad_()
    expected = direct_loss(reference, s, mean, phase, p, base)
    expected.backward()
    row = sequential.prepare_samples([s], mean, p, 'cpu')[0]
    if phase == 'v':
        with torch.no_grad():
            row['selected_k_p'] = sequential._probabilities(row, sequential._effective(row['k'], base, row['exact'], p['k_clip']))
    rotation = base.clone().requires_grad_()
    actual = sequential.phase_loss(rotation, row, phase, p)
    actual.backward()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(rotation.grad, reference.grad, atol=2e-6, rtol=2e-5)


def test_v_steps_use_fixed_k_probabilities_without_qk_softmax(monkeypatch):
    s = trace(17)
    p = policy()
    row = sequential.prepare_samples([s], torch.zeros(8), p, 'cpu')[0]
    row['selected_k_p'] = sequential._probabilities(row, row['k'])
    def forbidden(*_a, **_kw):
        pytest.fail('V phase must not recompute K probabilities')
    monkeypatch.setattr(sequential, '_probabilities', forbidden)
    r = torch.eye(8, requires_grad=True)
    sequential.phase_loss(r, row, 'v', p).backward()
    assert torch.isfinite(r.grad).all()


def test_official_schedule_and_orthogonal_selection():
    p = policy(steps=22, eval_every=20)
    events = []
    base = torch.eye(8)
    rk, rv, decision = sequential.fit_head(base, base, torch.zeros(8), [trace(2, 'train')],
        [trace(3, 'held')], policy=p, progress=events.append)
    assert [e['step'] for e in events if e['event'] == 'validation_done'] == [1, 21, 22] * 2
    assert all('proxy_loss' in e and 'normalized_error' not in e for e in events if e['event'] == 'validation_done')
    assert len([e for e in events if e['event'] == 'fixed_k_probabilities_ready']) == 1
    for r in (rk, rv):
        torch.testing.assert_close(r.T @ r, torch.eye(8), atol=5e-5, rtol=0)
    for row in decision['phases']:
        assert row['selected_proxy_loss'] <= row['baseline_proxy_loss']
    assert torch.equal(base, torch.eye(8))
    assert decision['model_quality'] == 'not_measured'


def test_d256_gqa6_actual_fit_keeps_finite_orthogonal_constants():
    p = optr_policy({'optr_calibration': {'steps': 1}})
    base = torch.eye(256)
    rk, rv, decision = sequential.fit_head(base, base, torch.zeros(256),
        [trace(7, 'train', dim=256, group=6, length=337)],
        [trace(9, 'held', dim=256, group=6, length=337)], policy=p)
    for r in (rk, rv):
        assert r.dtype == torch.float32 and r.device.type == 'cpu' and torch.isfinite(r).all()
        torch.testing.assert_close(r.T @ r, base, atol=5e-4, rtol=0)
    assert decision['policy']['fitting_objective'] == p['fitting_objective']


@pytest.mark.parametrize('fault', ['nonfinite', 'overlap', 'nonorthogonal', 'output'])
def test_sequential_rejects_invalid_input_or_broken_progress(fault):
    train, held = trace(1, 'train'), trace(2, 'held')
    base = torch.eye(8)
    def progress(_):
        if fault == 'output':
            raise OSError('log broken')
    if fault == 'nonfinite': train['k'][0, 0] = float('nan')
    if fault == 'overlap': held['request_id'] = 'train'
    if fault == 'nonorthogonal': base[0, 0] = 2
    with pytest.raises((ValueError, OSError)):
        sequential.fit_head(base, base, torch.zeros(8), [train], [held], policy=policy(steps=1), progress=progress)


@pytest.mark.parametrize('devices', [[0, 1, 2, 3], [4, 5, 6, 7]])
def test_npu_selection_uses_this_run_visibility_and_fp32(monkeypatch, devices):
    calls = []
    monkeypatch.delenv('ASCEND_RT_VISIBLE_DEVICES', raising=False)
    monkeypatch.setitem(sys.modules, 'torch_npu', SimpleNamespace())
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(is_available=lambda: True,
        set_compile_mode=lambda **x: calls.append(x),
        set_option=lambda x: calls.append(x), set_device=lambda x: calls.append(x)), raising=False)
    assert sequential.fitting_device({'devices': devices}) == 'npu:0'
    import os
    assert os.environ['ASCEND_RT_VISIBLE_DEVICES'] == ','.join(map(str, devices))
    assert calls == [{'ACL_PRECISION_MODE': 'must_keep_origin_dtype', 'ALLOW_MATMUL_HF32': 'disable'},
                     {'jit_compile': False}, 0]


def test_missing_npu_and_conflicting_visibility_never_fall_back(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch_npu', SimpleNamespace())
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(is_available=lambda: False), raising=False)
    monkeypatch.delenv('ASCEND_RT_VISIBLE_DEVICES', raising=False)
    with pytest.raises(RuntimeError, match='requires.*Ascend'):
        sequential.fitting_device({'devices': [0, 1, 2, 3]})
    monkeypatch.setenv('ASCEND_RT_VISIBLE_DEVICES', '4,5,6,7')
    with pytest.raises(ValueError, match='visibility'):
        sequential.fitting_device({'devices': [0, 1, 2, 3]})
    assert sequential.fitting_device({}, testing=True) == 'cpu'


def test_final_replay_caches_histories_without_changing_results(monkeypatch):
    import oscar_ascend.optr_calibration as c
    sample = trace(4, length=541, dim=8, queries=9, group=6)
    sample['history_splits'] = torch.full((9,), 20)
    rk = torch.eye(8)
    rv = torch.eye(8).roll(1, -1)
    mean = torch.linspace(-.4, .3, 8)
    for partition in ('single_split', 'captured_runtime_frames'):
        p = c._policy({'query_batch_size': 2, 'sink_tokens': 2, 'recent_tokens': 4, 'replay_partition': partition})
        with torch.no_grad():
            expected = torch.cat([c._replay(rk, rv, mean, batch, p) for batch in c._batches(sample, 2)])
            calls = []
            original = c.quantize_dequantize
            def counted(x, clip):
                calls.append(x.shape)
                return original(x, clip)
            with monkeypatch.context() as patch:
                patch.setattr(c, 'quantize_dequantize', counted)
                actual = c.replay_sample(rk, rv, mean, sample, p)
        assert torch.equal(actual, expected)
        assert calls == [torch.Size([541, 8])] * 2  # previously ten complete histories


def test_legacy_trace_reuse_keeps_all_source_checks(tmp_path):
    from test_optr_traces import build_bundle, json_save
    from oscar_ascend.optr_traces import load_trace_manifest, load_layer_samples
    path, config, top = build_bundle(tmp_path)
    top['optr_policy'] = legacy_capture_policy(config)
    top['calibration_profile'] = legacy_capture_profile(top['calibration_profile'], config)
    json_save(path, top)
    manifest = load_trace_manifest(path, config, ['model.layers.3.self_attn.attn'], 4, 'toy', testing=True)
    assert load_layer_samples(manifest, 3)['train']
    changed = copy.deepcopy(top)
    changed['calibration_profile']['base_calibration_profile']['prompt_source']['sha256'] = '0' * 64
    json_save(path, changed)
    with pytest.raises(ValueError, match='profile'):
        load_trace_manifest(path, config, ['model.layers.3.self_attn.attn'], 4, 'toy', testing=True)


def test_old_artifact_and_fit_are_not_the_new_objective(tmp_path):
    from test_optr_integration import configuration, artifact
    from oscar_ascend.optr_artifacts import validate_refinement_metadata
    from oscar_ascend.optr_checkpoint import FitCheckpoints, fitting_implementation_digest
    config = configuration(tmp_path)
    value = artifact(config)
    expected = copy.deepcopy(value['calibration_profile'])
    value['calibration_profile'] = legacy_capture_profile(expected, config)
    value['refinement_certificate']['policy'] = legacy_capture_policy(config)
    with pytest.raises(ValueError, match='profile'):
        validate_refinement_metadata(value, expected_profile=expected)
    legacy_id = {'profile': value['calibration_profile'], 'fitting_implementation_sha256': 'old'}
    old = FitCheckpoints(tmp_path / 'fits', legacy_id)
    old.save('layer-3-head-0', {'Rk': torch.eye(8)})
    new = FitCheckpoints(tmp_path / 'fits', {'profile': expected, 'fitting_implementation_sha256': fitting_implementation_digest()})
    assert new.load('layer-3-head-0') is None


def test_native_trace_discovery_uses_only_exact_previous_profile_path(tmp_path):
    import hashlib
    from tools.prepare_optr_rotations import _existing_trace_manifest
    from test_optr_integration import configuration
    from oscar_ascend.rotations import calibration_cache_identity
    config = configuration(tmp_path)
    profile = calibration_cache_identity(config, 'toy')
    old = legacy_capture_profile(profile, config)
    digest = hashlib.sha256(json.dumps(old, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:20]
    path = tmp_path / ('optr_' + digest) / 'native-traces' / 'capture-manifest.json'
    path.parent.mkdir(parents=True)
    path.write_text('{}')
    destination = tmp_path / 'optr_new' / 'refined.pt'
    assert _existing_trace_manifest(destination, profile, config) == path
    other = copy.deepcopy(profile)
    other['model_fingerprint'] = 'different'
    assert _existing_trace_manifest(destination, other, config) != path
