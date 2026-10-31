"""Archive #170/#175/#176: parity with fixed, unmodified upstream math.

Reference function bodies below are from daniel-eai/
Output-Aware-INT2-KV-Cache-Quantization commit
66d46561b9bdf5d715f591e0f5845a4cbd927dee:
  rotation/kv_calib_utils.py, optr/optr_common.py, optr/calibrate_optr.py.
Only the two local restore_exact_window imports are omitted: that same
function is defined below. No /tmp checkout or external package is required.
Reference code is deliberately independent of production fitting helpers.
"""
from types import SimpleNamespace

import pytest
import torch

from oscar_ascend import optr_sequential as sequential
from oscar_ascend.optr_policy import optr_policy
from test_optr_calibration import trace

DEV = "cpu"


def quant_int2_ste(
    tensor: torch.Tensor, clip_ratio: float, group_size: int
) -> torch.Tensor:
    shape = tensor.shape
    if shape[-1] % group_size != 0:
        raise ValueError(
            f"last dim {shape[-1]} must be divisible by group_size={group_size}"
        )
    grouped = tensor.reshape(*shape[:-1], shape[-1] // group_size, group_size)
    if clip_ratio and clip_ratio > 0:
        index = min(max(int(clip_ratio * group_size), 0), group_size - 1)
        threshold = grouped.detach().abs().sort(dim=-1).values[..., index : index + 1]
        grouped = torch.minimum(torch.maximum(grouped, -threshold), threshold)
    row_min = grouped.amin(dim=-1, keepdim=True)
    row_max = grouped.amax(dim=-1, keepdim=True)
    scale = (row_max - row_min).clamp(min=1e-8) / 3.0
    zero = -row_min / scale
    unrounded = grouped / scale + zero
    quantized = (unrounded + 0.5).floor().clamp(0, 3)
    straight_through = unrounded + (quantized - unrounded).detach()
    return ((straight_through - zero) * scale).reshape(shape)

def rotate_headwise(tensor: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return torch.einsum(
        "lhd,hde->lhe", tensor.to(rotation.dtype), rotation
    ).contiguous()

def derotate_headwise(tensor: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return torch.einsum(
        "lhd,hde->lhe",
        tensor.to(rotation.dtype),
        rotation.transpose(-1, -2),
    ).contiguous()

def restore_exact_window(
    approx: torch.Tensor,
    reference: torch.Tensor,
    prefix: int,
    recent: int,
) -> torch.Tensor:
    output = approx.clone()
    length = output.shape[0]
    if prefix > 0:
        output[: min(prefix, length)] = reference[: min(prefix, length)]
    if recent > 0:
        output[max(0, length - recent) :] = reference[max(0, length - recent) :]
    return output

def head_layout(seq_q, kvh):
    qh = seq_q.shape[1]
    if qh % kvh != 0:
        raise ValueError(f"q_heads={qh} not divisible by kv_heads={kvh}")
    return qh // kvh

def k_eff_headwise(k, R_k, mu, clip_k, grp, prefix, recent):
    kc = k - mu[None]
    kr = rotate_headwise(kc, R_k)
    kq = derotate_headwise(quant_int2_ste(kr, clip_k, grp), R_k)
    return restore_exact_window(kq, kc, prefix, recent), kc

def v_eff_headwise(v, R_v, clip_v, grp, prefix, recent):
    vr = rotate_headwise(v, R_v)
    vq = derotate_headwise(quant_int2_ste(vr, clip_v, grp), R_v)
    return restore_exact_window(vq, v, prefix, recent)

def k_loss_by_head(seqs, R_k, mu, WO, clip_k, grp, prefix, recent, qwin, scale, lam):
    """Output-aware K loss per KV head: KL + lam * post-W_O attention-shift err."""
    kvh = R_k.shape[0]
    vals = [[] for _ in range(kvh)]
    for s in seqs:
        keff, kc = k_eff_headwise(s.k, R_k, mu, clip_k, grp, prefix, recent)
        L = keff.shape[0]
        g = head_layout(s.q, kvh)
        lo = max(1, L - qwin)
        m = torch.arange(L, device=DEV).unsqueeze(0) > torch.arange(
            lo, L, device=DEV
        ).unsqueeze(1)
        for h in range(kvh):
            kt = kc[:, h]
            ks = keff[:, h]
            vfp = s.v[:, h]
            hv = []
            for gi in range(g):
                j = h * g + gi
                qt = s.q[lo:L, j]
                Lt = ((qt @ kt.T) * scale).masked_fill(m, float("-inf"))
                Ls = ((qt @ ks.T) * scale).masked_fill(m, float("-inf"))
                pt = torch.softmax(Lt, -1)
                pq = torch.softmax(Ls, -1)
                kl = (
                    (pt * (pt.clamp_min(1e-12).log() - pq.clamp_min(1e-12).log()))
                    .sum(-1)
                    .mean()
                )
                # post-W_O output error from the attention-distribution change
                d_out = ((pq - pt) @ vfp) @ WO[j].T  # [qwin, hidden]
                hv.append(kl + lam * d_out.pow(2).mean())
            vals[h].append(torch.stack(hv).mean())
    return torch.stack([torch.stack(v).mean() for v in vals])

def v_loss_by_head(
    seqs, R_k, R_v, mu, WO, clip_k, clip_v, grp, prefix, recent, qwin, scale
):
    """Output-aware V loss per KV head: post-W_O value reconstruction under P_q."""
    kvh = R_v.shape[0]
    vals = [[] for _ in range(kvh)]
    for s in seqs:
        keff, _ = k_eff_headwise(s.k, R_k, mu, clip_k, grp, prefix, recent)
        veff = v_eff_headwise(s.v, R_v, clip_v, grp, prefix, recent)
        L = keff.shape[0]
        g = head_layout(s.q, kvh)
        lo = max(1, L - qwin)
        m = torch.arange(L, device=DEV).unsqueeze(0) > torch.arange(
            lo, L, device=DEV
        ).unsqueeze(1)
        for h in range(kvh):
            ks = keff[:, h]
            dv = veff[:, h] - s.v[:, h]  # value error [L, hd]
            hv = []
            for gi in range(g):
                j = h * g + gi
                qt = s.q[lo:L, j]
                Ls = ((qt @ ks.T) * scale).masked_fill(m, float("-inf"))
                pq = torch.softmax(Ls, -1)
                d_out = (pq @ dv) @ WO[j].T  # [qwin, hidden]
                hv.append(d_out.pow(2).mean())
            vals[h].append(torch.stack(hv).mean())
    return torch.stack([torch.stack(v).mean() for v in vals])

def learn_rotation(
    R_base, base_held, loss_fn_calib, loss_fn_held, steps, lr, margin, kvh, hd
):
    A = torch.zeros(kvh, hd, hd, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([A], lr=lr)
    best = base_held.detach().clone()
    bestR = R_base.detach().clone()
    for step in range(steps):
        opt.zero_grad()
        R = R_base @ torch.matrix_exp(A - A.transpose(-1, -2))
        loss = loss_fn_calib(R).mean()
        loss.backward()
        opt.step()
        if step % 20 == 0 or step == steps - 1:
            with torch.no_grad():
                R = R_base @ torch.matrix_exp(A - A.transpose(-1, -2))
                hb = loss_fn_held(R)
                imp = hb < best
                best = torch.where(imp, hb, best)
                bestR[imp] = R.detach()[imp]
    keep = best < base_held * (1 - margin)
    Rf = R_base.detach().clone()
    Rf[keep] = bestR[keep]
    return Rf, keep, best


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def inputs(seed, label, *, heads=2, dim=8, group=3, length=81, wo_scale=1.):
    """One actual last-token decode: exact window also equals upstream tail."""
    samples = [trace(seed + h, label, length=length, dim=dim, queries=1, group=group)
               for h in range(heads)]
    for sample in samples:
        sample['wo'].mul_(wo_scale)
    q = torch.zeros(length, heads * group, dim)
    q[-1] = torch.cat([sample['q'].float()[0] for sample in samples])
    sequence = SimpleNamespace(q=q,
        k=torch.stack([sample['k'].float() for sample in samples], dim=1),
        v=torch.stack([sample['v'].float() for sample in samples], dim=1))
    wo = torch.cat([sample['wo'] for sample in samples])
    return samples, sequence, wo


def options(**kwargs):
    return {**optr_policy({}), 'sink_tokens': 2, 'recent_tokens': 4, **kwargs}


def source_loss(sequences, rk, rv, mean, wo, phase, policy):
    arguments = (policy['k_clip'], rk.shape[-1], policy['sink_tokens'],
                 policy['recent_tokens'], 1, rk.shape[-1] ** -.5)
    if phase == 'k':
        return k_loss_by_head(sequences, rk, mean, wo, *arguments, policy['lambda_output'])
    return v_loss_by_head(sequences, rk, rv, mean, wo,
        policy['k_clip'], policy['v_clip'], rk.shape[-1], policy['sink_tokens'],
        policy['recent_tokens'], 1, rk.shape[-1] ** -.5)


@pytest.mark.parametrize('phase', ['k', 'v'])
@pytest.mark.parametrize('dim,group,length', [(8, 3, 81), (256, 6, 337)])
def test_loss_and_gradient_match_fixed_upstream_function_bodies(phase, dim, group, length):
    heads = 2
    samples, sequence, wo = inputs(83, 'sample', heads=heads, dim=dim, group=group, length=length)
    generator = torch.Generator().manual_seed(73)
    rk = torch.linalg.qr(torch.randn(heads, dim, dim, generator=generator)).Q.contiguous()
    rv = torch.linalg.qr(torch.randn(heads, dim, dim, generator=generator)).Q.contiguous()
    mean = torch.randn(heads, dim, generator=generator) * .2
    reference = (rk if phase == 'k' else rv).clone().requires_grad_()
    expected = source_loss([sequence], reference if phase == 'k' else rk,
                           reference if phase == 'v' else rv, mean, wo, phase, options())
    expected.sum().backward()
    for head, sample in enumerate(samples):
        row = sequential.prepare_samples([sample], mean[head], options(), 'cpu')[0]
        if phase == 'v':
            with torch.no_grad():
                row['selected_k_quant_p'] = sequential._key_probabilities(rk[head], row, options()).masked_fill(row['exact'], 0)
        actual_rotation = (rk if phase == 'k' else rv)[head].clone().requires_grad_()
        actual = sequential.phase_loss(actual_rotation, row, phase, options())
        actual.backward()
        torch.testing.assert_close(actual, expected[head], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(actual_rotation.grad, reference.grad[head], atol=2e-6, rtol=2e-5)


def test_independent_head_adam_matches_upstream_all_head_mean(monkeypatch):
    # Small W_O produces gradients where Adam epsilon matters: omitting the
    # outer KV-head mean changes updates even though Adam normalizes moments.
    heads, dim = 2, 8
    train, source_train, wo = inputs(91, 'train', heads=heads, wo_scale=1e-3)
    held, source_held, _ = inputs(97, 'held', heads=heads, wo_scale=1e-3)
    for h in range(heads):
        held[h]['wo'] = train[h]['wo']
    p = options(steps=3, eval_every=20)
    base = torch.eye(dim).expand(heads, dim, dim).clone()
    mean = torch.linspace(-.1, .2, heads * dim).reshape(heads, dim)
    snapshots = []
    original = torch.optim.Adam

    class RecordingAdam(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.saved = []
            snapshots.append(self.saved)

        def step(self, *args, **kwargs):
            result = super().step(*args, **kwargs)
            self.saved.append(self.param_groups[0]['params'][0].detach().clone())
            return result

    monkeypatch.setattr(torch.optim, 'Adam', RecordingAdam)
    k_train = lambda rotation: source_loss([source_train], rotation, base, mean, wo, 'k', p)
    k_held = lambda rotation: source_loss([source_held], rotation, base, mean, wo, 'k', p)
    source_rk, _, _ = learn_rotation(base, k_held(base), k_train, k_held, p['steps'], p['lr'], 0., heads, dim)
    v_train = lambda rotation: source_loss([source_train], source_rk, rotation, mean, wo, 'v', p)
    v_held = lambda rotation: source_loss([source_held], source_rk, rotation, mean, wo, 'v', p)
    source_rv, _, _ = learn_rotation(base, v_held(base), v_train, v_held, p['steps'], p['lr'], 0., heads, dim)
    for head in range(heads):
        rk, rv, decision = sequential.fit_head(base[head], base[head], mean[head],
            [train[head]], [held[head]], policy=p, global_kv_heads=heads)
        assert decision['optimizer_loss_divisor'] == heads
        torch.testing.assert_close(rk, source_rk[head], atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(rv, source_rv[head], atol=3e-5, rtol=3e-5)
        for phase in range(2):
            for expected, actual in zip(snapshots[phase], snapshots[2 + 2 * head + phase], strict=True):
                torch.testing.assert_close(actual, expected[head], atol=3e-6, rtol=3e-5)

