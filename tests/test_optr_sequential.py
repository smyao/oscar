"""Independent joint native-target objective, guarded devices, deployment gate."""
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
    # Independent literal per-query formula: actual forward cut, BF16 writer
    # operands and full W_O projection (no Gram or cached split probabilities).
    def bf(x):
        return x + (x.bfloat16().float() - x).detach()
    def quant(x, clip):
        index = min(int(float(torch.tensor(clip)) * x.shape[-1]), x.shape[-1] - 1)
        threshold = x.detach().abs().sort(-1).values[:, index:index + 1]
        x = x.clamp(-threshold, threshold)
        low, high = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
        scale = (high - low).clamp_min(1e-8) / 3
        z = -low / scale
        a = x / scale + z
        return bf((a + ((a + .5).floor().clamp(0, 3) - a).detach() - z) * scale)
    kr = bf(rotation if phase == 'k' else rk)
    k = bf(bf(sample['k'].float() - bf(mean)) @ kr)
    kq = quant(k, p['k_clip'])
    if phase == 'v':
        vr = bf(rotation)
        v = bf(sample['v'].float() @ vr)
        vq = quant(v, p['v_clip'])
    vals = []
    group = sample['q'].shape[1]
    for i, position in enumerate(sample['positions'].tolist()):
        contributions, divergences = [], []
        for h in range(group):
            context = int(sample['contexts'][i])
            cut = min(context, max(p['sink_tokens'], context + 1 - p['recent_tokens']))
            exact = (torch.arange(len(k)) < p['sink_tokens']) | (torch.arange(len(k)) >= cut)
            key = torch.where(exact[:, None], k, kq)
            q = sample['q'][i, h].float()
            logits = key[:position + 1] @ bf(q @ kr) / k.shape[-1] ** .5
            pq = logits.softmax(-1)
            if phase == 'k':
                pt = ((sample['k'][:position + 1].float() @ q) / k.shape[-1] ** .5).softmax(-1)
                kl = (pt * (pt.clamp_min(1e-12).log() - pq.clamp_min(1e-12).log())).sum()
                actual = pq @ sample['v'][:position + 1].float()
                teacher = sample['teacher'][i, h].float()
            else:
                kl = 0.
                value = torch.where(exact[:, None], v, vq)
                actual = bf(bf(pq @ value[:position + 1]) @ vr.T)
                teacher = pq @ sample['v'][:position + 1].float()
            gate = sample['gate'][i, h].float()
            # Native gate rounds both operands separately, before subtraction.
            error = bf(bf(actual) * gate) - bf(bf(teacher) * gate)
            contributions.append(error @ sample['wo'][h].T)
            divergences.append(kl)
        joint_error = torch.stack(contributions).sum(0)
        output_loss = joint_error.square().mean() / group
        vals.append(sum(divergences) / group + output_loss * (p['lambda_output'] if phase == 'k' else 1))
    return torch.stack(vals).mean()


def stage_v(row, rk, p):
    with torch.no_grad():
        row['selected_k_p'] = sequential._key_probabilities(rk, row, p)
        row['selected_k_exact_p'] = row['selected_k_p'] * row['exact']
        row['selected_k_quant_p'] = row['selected_k_p'].masked_fill(row['exact'], 0)
        teacher = (row['selected_k_p'] @ row['v']).bfloat16().float()
        row['conditional_teacher_gated'] = (teacher * row['gate']).bfloat16().float()


def test_train_and_held_reuse_only_the_same_fixed_output_projection():
    train, held = trace(17, 'train'), trace(19, 'held')
    held['wo'] = train['wo']  # Production loader shares this model weight.
    weights = {}
    left = sequential.prepare_samples([train], torch.zeros(8), policy(), 'cpu', weights=weights)[0]
    right = sequential.prepare_samples([held], torch.zeros(8), policy(), 'cpu', weights=weights)[0]
    assert left['wo_joint'] is right['wo_joint'] and len(weights) == 1
    assert left['teacher_gated'] is not right['teacher_gated']
    other = trace(23, 'other')
    third = sequential.prepare_samples([other], torch.zeros(8), policy(), 'cpu', weights=weights)[0]
    assert third['wo_joint'] is not left['wo_joint'] and len(weights) == 2


@pytest.mark.parametrize('phase', ['k', 'v'])
@pytest.mark.parametrize('gate', [True, False])
@pytest.mark.parametrize('group', [1, 3])
def test_joint_loss_and_gradient_match_literal_native_target_formula(phase, gate, group):
    s = trace(17, length=81, dim=8, queries=7, group=group)
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
            stage_v(row, base, p)
    rotation = base.clone().requires_grad_()
    actual = sequential.phase_loss(rotation, row, phase, p)
    actual.backward()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(rotation.grad, reference.grad, atol=2e-6, rtol=2e-5)


def test_joint_projection_rejects_improvement_that_only_reduces_separate_head_errors():
    # Both GQA heads project to the same output. Their signs matter, even
    # though the old separate-head squared norms discard those signs.
    before = torch.tensor([[[1.]], [[-1.]]])
    after = torch.tensor([[[.5]], [[.5]]])
    row = {'gate': torch.ones_like(before), 'wo_joint': torch.ones(2, 1)}
    teacher = torch.zeros_like(before)
    old_before, old_after = before.square().mean(), after.square().mean()
    new_before = sequential._projected_mse(before, teacher, row)
    new_after = sequential._projected_mse(after, teacher, row)
    assert old_after < old_before
    assert new_after > new_before
    assert (float(old_before), float(old_after), float(new_before), float(new_after)) == (1., .25, 0., .5)


def test_gate_difference_uses_separately_rounded_native_products():
    actual = torch.tensor([[[1.0078125]]])
    teacher = torch.tensor([[[1.]]])
    gate = torch.tensor([[[.1]]]).bfloat16().float()
    row = {'gate': gate, 'wo_joint': torch.ones(1, 1)}
    rounded_teacher = (teacher * gate).bfloat16().float()
    expected_difference = (actual * gate).bfloat16().float() - rounded_teacher
    old_difference = (actual - teacher) * gate
    assert float(old_difference) == .000782012939453125
    assert float(expected_difference) == .0009765625
    actual_loss = sequential._projected_mse(actual, rounded_teacher, row)
    torch.testing.assert_close(actual_loss, expected_difference.square().mean(), atol=0, rtol=0)
    assert actual_loss != old_difference.square().mean()


def test_k_output_target_uses_captured_teacher_while_kl_target_is_unchanged():
    s = trace(13, length=81, dim=8, queries=4, group=3)
    p, mean, rotation = policy(), torch.zeros(8), torch.eye(8)
    row = sequential.prepare_samples([s], mean, p, 'cpu')[0]
    with torch.no_grad():
        actual = sequential._key_probabilities(rotation, row, p) @ row['v']
    matching = {**s, 'teacher': actual.transpose(0, 1).bfloat16()}
    changed = {**matching, 'teacher': (matching['teacher'].float() + 1).bfloat16()}
    a, b = sequential.prepare_samples([matching, changed], mean, p, 'cpu')
    assert torch.equal(a['teacher_p'], b['teacher_p'])
    assert torch.equal(a['teacher_logp'], b['teacher_logp'])
    kl_only = {**p, 'lambda_output': 0.}
    a_kl = sequential.phase_loss(rotation, a, 'k', kl_only)
    b_kl = sequential.phase_loss(rotation, b, 'k', kl_only)
    torch.testing.assert_close(a_kl, b_kl, atol=0, rtol=0)
    a_loss = sequential.phase_loss(rotation, a, 'k', p)
    b_loss = sequential.phase_loss(rotation, b, 'k', p)
    torch.testing.assert_close(a_loss, a_kl, atol=0, rtol=0)
    assert b_loss > a_loss


def test_joint_projection_stays_nonnegative_with_d256_gqa6_cancellation():
    generator = torch.Generator().manual_seed(6)
    weight = torch.randn(3, 256, 17, generator=generator).bfloat16().float()
    value = torch.randn(3, 2, 256, generator=generator).bfloat16().float()
    actual = torch.cat((value, -value), dim=0).requires_grad_()
    row = {'gate': torch.ones_like(actual),
           'wo_joint': torch.cat((weight, weight), dim=0).reshape(1536, 17)}
    loss = sequential._projected_mse(actual, torch.zeros_like(actual), row)
    assert torch.isfinite(loss) and 0 <= loss < 1e-7
    loss.backward()
    assert torch.isfinite(actual.grad).all()


@pytest.mark.parametrize('phase', ['k', 'v'])
def test_d256_gqa6_loss_and_gradient_match_independent_projection(phase):
    sample = trace(29, length=337, dim=256, queries=2, group=6)
    p, mean, base = optr_policy({}), torch.linspace(-.2, .3, 256), torch.eye(256)
    reference = base.clone().requires_grad_()
    expected = direct_loss(reference, sample, mean, phase, p, base)
    expected.backward()
    row = sequential.prepare_samples([sample], mean, p, 'cpu')[0]
    if phase == 'v':
        stage_v(row, base, p)
    rotation = base.clone().requires_grad_()
    actual = sequential.phase_loss(rotation, row, phase, p)
    actual.backward()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(rotation.grad, reference.grad, atol=2e-6, rtol=2e-5)


def test_v_steps_use_fixed_k_probabilities_without_qk_softmax(monkeypatch):
    s = trace(17)
    p = policy()
    row = sequential.prepare_samples([s], torch.zeros(8), p, 'cpu')[0]
    stage_v(row, torch.eye(8), p)
    def forbidden(*_a, **_kw):
        pytest.fail('V phase must not recompute K probabilities')
    monkeypatch.setattr(sequential, '_probabilities', forbidden)
    monkeypatch.setattr(sequential, '_key_probabilities', forbidden)
    r = torch.eye(8, requires_grad=True)
    sequential.phase_loss(r, row, 'v', p).backward()
    assert torch.isfinite(r.grad).all()


@pytest.mark.parametrize('phase', ['k', 'v'])
def test_masked_future_rows_cannot_change_visible_loss_or_gradient(phase):
    # Same captured forward, but the trace later accumulated more tokens.
    # The old global-tail mask changed which *visible* keys were quantized.
    short = trace(17, length=21, queries=1, group=2)
    short['contexts'].fill_(17)
    long = copy.deepcopy(short)
    extra = trace(29, length=16)
    long['k'] = torch.cat((short['k'], extra['k']))
    long['v'] = torch.cat((short['v'], extra['v']))
    p, mean = policy(), torch.linspace(-.2, .3, 8)
    base = torch.linalg.qr(torch.randn(8, 8, generator=torch.Generator().manual_seed(18))).Q
    values, gradients = [], []
    for sample in (short, long):
        row = sequential.prepare_samples([sample], mean, p, 'cpu')[0]
        if phase == 'v':
            stage_v(row, base, p)
        rotation = base.clone().requires_grad_()
        loss = sequential.phase_loss(rotation, row, phase, p)
        loss.backward()
        values.append(loss.detach()); gradients.append(rotation.grad)
    torch.testing.assert_close(values[0], values[1], atol=2e-7, rtol=2e-5)
    torch.testing.assert_close(gradients[0], gradients[1], atol=2e-6, rtol=2e-5)


def test_proxy_masks_share_forward_cut_and_keep_prefill_current_exact():
    sample = trace(11, length=25, queries=4)
    sample['positions'] = torch.tensor([3, 17, 20, 24])
    sample['contexts'] = torch.tensor([0, 17, 17, 24])
    sample['current_ends'] = torch.tensor([4, 21, 21, 25])
    row = sequential.prepare_samples([sample], torch.zeros(8), policy(), 'cpu')[0]
    assert bool(row['exact'][0].all())
    assert torch.equal(row['exact'][1], row['exact'][2])
    assert torch.where(~row['exact'][1])[0].tolist() == list(range(2, 14))
    assert torch.where(~row['exact'][3])[0].tolist() == list(range(2, 21))
    # The short/long case would fail even with a causal attention mask if the
    # exact tail were computed from len(sample['k']) instead of frame context.


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
@pytest.mark.parametrize('index', [0, 1, 2, 3])
def test_npu_selection_uses_this_run_visibility_and_fp32(monkeypatch, devices, index):
    calls = []
    monkeypatch.delenv('ASCEND_RT_VISIBLE_DEVICES', raising=False)
    monkeypatch.setitem(sys.modules, 'torch_npu', SimpleNamespace())
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(is_available=lambda: True,
        set_compile_mode=lambda **x: calls.append(x),
        set_option=lambda x: calls.append(x), set_device=lambda x: calls.append(x)), raising=False)
    assert sequential.fitting_device({'devices': devices}, logical_index=index) == f'npu:{index}'
    import os
    assert os.environ['ASCEND_RT_VISIBLE_DEVICES'] == ','.join(map(str, devices))
    assert calls == [index, {'ACL_PRECISION_MODE': 'must_keep_origin_dtype', 'ALLOW_MATMUL_HF32': 'disable'},
                     {'jit_compile': False}]


@pytest.mark.parametrize('index', [-1, 4, True, '1'])
def test_fitting_rejects_invalid_logical_devices_before_npu_import(index):
    with pytest.raises(ValueError, match='logical device'):
        sequential.fitting_device({'devices': [0, 1, 2, 3]}, logical_index=index)


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


@pytest.mark.parametrize('version', ['v1', 'v2'])
def test_native_trace_discovery_uses_only_exact_previous_profile_path(tmp_path, version):
    import hashlib
    from tools.prepare_optr_rotations import _existing_trace_manifest
    from test_optr_integration import configuration
    from oscar_ascend.rotations import calibration_cache_identity
    config = configuration(tmp_path)
    config['optr_calibration'] = {'capture_lengths': {'train': [4096],
        'validation': [4096, 32768, 65536], 'test': [4096, 32768, 65536]}}
    profile = calibration_cache_identity(config, 'toy')
    old = legacy_capture_profile(profile, config, version=version)
    digest = hashlib.sha256(json.dumps(old, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:20]
    path = tmp_path / ('optr_' + digest) / 'native-traces' / 'capture-manifest.json'
    path.parent.mkdir(parents=True)
    path.write_text('{}')
    destination = tmp_path / 'optr_new' / 'refined.pt'
    assert _existing_trace_manifest(destination, profile, config) == path
    other = copy.deepcopy(profile)
    other['model_fingerprint'] = 'different'
    assert _existing_trace_manifest(destination, other, config) != path
    # Normal v3 must acquire genuine long-context training, not reinterpret
    # the old validation/test samples as training or reuse the seven-request capture.
    config.pop('optr_calibration')
    assert _existing_trace_manifest(destination, calibration_cache_identity(config, 'toy'), config) != path
