"""Execute only hashed upstream math functions; no model/download/runtime code."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import torch
from oscar_ascend import optr_sequential as local
from oscar_ascend.optr_policy import optr_policy

ROOT = Path('/private/tmp/optr-source-20261027')
# The archive can contain one enclosing repository directory.
files = list(ROOT.rglob('optr/calibrate_optr.py'))
assert len(files) == 1
root = files[0].parents[1]
pinned = {
    'optr/calibrate_optr.py': '21e4f991a57ad6cf370f195dd06ca7182c3fe41d9a46fbcbceec7d997cda31ae',
    'rotation/kv_calib_utils.py': '9b3ea32c1e3df73f94559eb2241711f2c6459f3283b8657a64470e47468785f0',
}
for name, expected in pinned.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected

def functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)

ns = {'torch': torch, 'DEV': 'cpu', 'head_layout': lambda q, h: q.shape[1] // h}
functions(root / 'rotation/kv_calib_utils.py', ['quant_int2_ste', 'rotate_headwise', 'derotate_headwise', 'restore_exact_window'], ns)
module = ModuleType('kv_calib_utils')
module.restore_exact_window = ns['restore_exact_window']
sys.modules['kv_calib_utils'] = module
functions(root / 'optr/calibrate_optr.py', ['k_eff_headwise', 'v_eff_headwise', 'k_loss_by_head', 'v_loss_by_head', 'learn_rotation'], ns)
torch.set_num_threads(1)
p = {**optr_policy({}), 'sink_tokens': 2, 'recent_tokens': 4, 'steps': 3, 'eval_every': 20}
dim, group, length, queries = 8, 3, 81, 7
mean = torch.linspace(-.2, .3, dim)
wo = torch.randn(group, 12, dim, generator=torch.Generator().manual_seed(84))
def sample(seed, request):
    rng = torch.Generator().manual_seed(seed)
    q = torch.randn(length, group, dim, generator=rng).bfloat16().float()
    k = torch.randn(length, 1, dim, generator=rng).bfloat16().float()
    v = torch.randn(length, 1, dim, generator=rng).bfloat16().float()
    seq = SimpleNamespace(q=q, k=k, v=v)
    trace = {'request_id': request, 'q': q[-queries:].bfloat16(), 'k': k[:, 0].bfloat16(),
        'v': v[:, 0].bfloat16(), 'gate': torch.ones(queries, group, dim).bfloat16(),
        'teacher': torch.zeros(queries, group, dim).bfloat16(), 'wo': wo,
        'positions': torch.arange(length - queries, length), 'contexts': torch.full((queries,), length - queries)}
    return seq, trace
train_seq, train = sample(19, 'train')
held_seq, held = sample(20, 'held')
base = torch.linalg.qr(torch.randn(dim, dim, generator=torch.Generator().manual_seed(91))).Q.contiguous()
scale = dim ** -.5
args = (mean[None], wo, p['k_clip'], dim, p['sink_tokens'], p['recent_tokens'], queries, scale, p['lambda_output'])
rows = {}
for phase in ('k', 'v'):
    ref = base[None].clone().requires_grad_()
    if phase == 'k':
        expected = ns['k_loss_by_head']([train_seq], ref, *args).mean()
    else:
        expected = ns['v_loss_by_head']([train_seq], base[None], ref, mean[None], wo,
            p['k_clip'], p['v_clip'], dim, p['sink_tokens'], p['recent_tokens'], queries, scale).mean()
    expected.backward()
    staged = local.prepare_samples([train], mean, p, 'cpu')[0]
    if phase == 'v':
        staged['selected_k_p'] = local._probabilities(staged, local._effective(staged['k'], base, staged['exact'], p['k_clip']))
    r = base.clone().requires_grad_()
    actual = local.phase_loss(r, staged, phase, p)
    actual.backward()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(r.grad, ref.grad[0], atol=2e-6, rtol=2e-5)
    rows[phase] = {'official_loss': float(expected.detach()), 'adapted_loss': float(actual.detach()),
                   'gradient_max_abs_difference': float((r.grad - ref.grad[0]).abs().max())}

# Compare complete sequential Adam/held-out selection to actual upstream loops.
def kloss(seq, r): return ns['k_loss_by_head']([seq], r, *args)
kbase = kloss(held_seq, base[None])
rk, _, _ = ns['learn_rotation'](base[None], kbase, lambda r: kloss(train_seq, r),
                               lambda r: kloss(held_seq, r), p['steps'], p['lr'], 0., 1, dim)
def vloss(seq, r):
    return ns['v_loss_by_head']([seq], rk, r, mean[None], wo, p['k_clip'], p['v_clip'], dim,
                                p['sink_tokens'], p['recent_tokens'], queries, scale)
vbase = vloss(held_seq, base[None])
rv, _, _ = ns['learn_rotation'](base[None], vbase, lambda r: vloss(train_seq, r),
                               lambda r: vloss(held_seq, r), p['steps'], p['lr'], 0., 1, dim)
lk, lv, _ = local.fit_head(base, base, mean, [train], [held], policy=p)
torch.testing.assert_close(lk, rk[0], atol=2e-5, rtol=2e-5)
torch.testing.assert_close(lv, rv[0], atol=2e-5, rtol=2e-5)
result = {'upstream_commit': '66d46561b9bdf5d715f591e0f5845a4cbd927dee', 'source_hashes': pinned,
    'scope': 'CPU official ungated proxy with group=head_dim and identical tail queries; not NPU or model validation',
    'loss_gradient_tolerance': {'atol': 2e-6, 'rtol': 2e-5},
    'rotation_tolerance': {'atol': 2e-5, 'rtol': 2e-5}, 'phases': rows,
    'k_matrix_max_abs_difference': float((lk-rk[0]).abs().max()),
    'v_matrix_max_abs_difference': float((lv-rv[0]).abs().max()), 'passed': True}
Path(__file__).with_name('upstream-parity.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
