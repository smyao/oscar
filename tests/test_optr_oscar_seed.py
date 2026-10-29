"""User-selected oscar_base source equivalence, bounded capture and provenance.

The reference below is an AST-unparsed snapshot of only the named pure
functions from pinned oscar_base. CI never imports that repository/engine or
requires the developer machine's absolute path. An optional source audit
compares this snapshot to the explicitly selected read-only checkout.
"""
import ast
import copy
import hashlib
import math
import os
from pathlib import Path

import pytest
import torch

from oscar_ascend.optr_oscar_seed import (
    BaseSeedAccumulator, SAMPLE_CAP, SOURCE_FILES, fit_seed_artifact,
    legacy_base_profile, seed_profile, shared_covariance, source_provenance,
    validate_seed_artifact, validate_seed_profile,
)
from oscar_ascend.rotations import calibration_cache_identity

NAME = "model.layers.3.self_attn.attn"

REFERENCE_SOURCE = r'''
def finalize_cov(model=None) -> dict:
    """worker 内读取钩子缓冲 → 旋转校准统计（无任何 model 前向）。

    返回 {layer_name: {"q": Σ_Q, "v": Σ_S, "count"}}：
      Σ_Q = (1/H_kv) Σ_h Q_g^T Q_g / n  —— K 旋转的 hessian（qqt 目标，论文
      compute_kv_rotation.py:93-108 / README:312）；
      Σ_S = (1/H_kv) Σ_h (V_h·√w)^T(V_h·√w)/n，w_i=(k_i^T Σ_Qh k_i) —— V 旋转的
      score-weighted（sst 目标，compute_kv_rotation.py:111-136）。
    计算在捕获设备（NPU/fp32）上完成；返回张量已 CPU（RPC 安全）。
    """
    out = {}
    for layer_name, d in _captures.items():
        if not d['k'] or not d['q']:
            continue
        q = torch.cat(d['q'], dim=0)
        k = torch.cat(d['k'], dim=0)
        v = torch.cat(d['v'], dim=0)
        if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
            print(f'[oscar-ascend][calib] ⚠️ {layer_name} 捕获非 3D（dim={q.dim()}/{k.dim()}/{v.dim()}），跳过')
            continue
        n_hq = q.shape[1]
        n_hk = k.shape[1]
        if n_hq % n_hk != 0 or n_hq < n_hk:
            print(f'[oscar-ascend][calib] ⚠️ {layer_name} 头数异常 Hq={n_hq}/Hk={n_hk}，跳过')
            continue
        g = n_hq // n_hk
        n = k.shape[0]

        def _sym(x: torch.Tensor) -> torch.Tensor:
            return (x + x.T) / 2
        dev = q.device
        cov_q = torch.zeros(q.shape[-1], q.shape[-1], device=dev, dtype=torch.float32)
        cov_s = torch.zeros(v.shape[-1], v.shape[-1], device=dev, dtype=torch.float32)
        for h in range(n_hk):
            qg = q[:, h * g:(h + 1) * g, :].float().reshape(-1, q.shape[-1])
            kh = k[:, h, :].float()
            vh = v[:, h, :].float()
            qtq = _sym(qg.t() @ qg / qg.shape[0])
            cov_q += qtq
            weights = (kh @ qtq * kh).sum(1)
            weights = weights / weights.sum().clamp(min=1e-12) * n
            vw = vh * weights.unsqueeze(1).sqrt()
            cov_s += _sym(vw.t() @ vw / n)
        cov_q = cov_q / n_hk
        cov_s = cov_s / n_hk
        out[layer_name] = {'q': {'cov': cov_q.detach().cpu(), 'count': n}, 'v': {'cov': cov_s.detach().cpu(), 'count': n}}
    return out

def _merge_rank_stats(per_rank: list[dict], D: int) -> dict[str, dict[str, dict]]:
    """跨 rank 加权合并协方差（每 rank 协方差已按 token 平均；按 count 加权）。"""
    merged: dict[str, dict[str, dict]] = {}
    for rank_stats in per_rank:
        for layer, st in rank_stats.items():
            for kind in ('q', 'v'):
                agg = merged.setdefault(layer, {}).setdefault(kind, {'cov_sum': torch.zeros(D, D), 'count': 0})
                cov = st[kind]['cov']
                c = st[kind]['count']
                agg['cov_sum'] = agg['cov_sum'] + cov * c
                agg['count'] += c
    return merged

def build_hadamard(n: int) -> torch.Tensor:
    """[n,n] 归一化 Hadamard（n 必须 2 的幂）。"""
    if n < 1 or n & n - 1:
        raise ValueError(f'Hadamard size must be a power of two, got {n}')
    if n == 1:
        return torch.ones(1, 1, dtype=torch.float64)
    h = build_hadamard(n // 2)
    return torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0) / math.sqrt(2)

def bit_reversal_perm(d: int) -> torch.Tensor:
    if d < 1 or d & d - 1:
        raise ValueError(f'Bit-reversal size must be a power of two, got {d}')
    bits = int(math.log2(d))
    return torch.tensor([int(bin(i)[2:].zfill(bits)[::-1], 2) for i in range(d)])

def make_br_perm_matrix(eigenvalues: torch.Tensor) -> torch.Tensor:
    """P_br：按特征值（降序）排序后位反转放置——高方差方向均匀交错。"""
    d = len(eigenvalues)
    sorted_idx = torch.argsort(eigenvalues, descending=True)
    br = bit_reversal_perm(d)
    perm = torch.zeros(d, dtype=torch.long)
    for i in range(d):
        perm[br[i]] = sorted_idx[i]
    return torch.eye(d, dtype=torch.float64)[:, perm]

def compose_rotation(rotation: torch.Tensor, eigvals: torch.Tensor, hadamard: torch.Tensor) -> torch.Tensor:
    """R · H · P_br（论文默认验证配方 r_h_pbr）。"""
    pbr = make_br_perm_matrix(eigvals)
    return rotation @ hadamard @ pbr

def _rotation_from_stats(stats: dict, D: int) -> tuple[torch.Tensor, torch.Tensor]:
    """hessian 协方差 → (R = U·H·P_br, eigenvalues)。

    一律在 **父进程 CPU fp64** 完成（离线一次性、256×256 小矩阵；真机 05:02 教训：
    torch-npu 不支持 float64 → 任何 NPU 侧 fp64 计算都可能触发 ERR01002，纯属风险）。
    """
    c = max(1, int(stats['count']))
    cov = stats['cov_sum'] / c
    eps = 1e-06 * torch.eye(D)
    cov64 = (cov.double() + cov.double().T) / 2
    try:
        evals, evecs = torch.linalg.eigh(cov64 + eps.double())
    except Exception as e:
        raise RuntimeError(f'eigh 失败（cov 含 NaN?）: {e}') from e
    H = build_hadamard(D)
    rot = compose_rotation(evecs, evals, H)
    err = (rot @ rot.T - torch.eye(D, dtype=torch.float64)).abs().max().item()
    if err > 1e-06:
        print(f'[gen-rotations] ⚠️ 组合旋转正交误差 {err:.2e}（eigh 精度；继续）')
    return (rot.float().contiguous(), evals.float().contiguous())
'''


def reference(q, k, v):
    namespace = {"torch": torch, "math": math,
                 "_captures": {NAME: {"q": [q.float()], "k": [k.float()], "v": [v.float()]}}}
    exec(compile(ast.parse(REFERENCE_SOURCE), "pinned_oscar_base_pure_functions", "exec"), namespace)
    return namespace, namespace["finalize_cov"]()[NAME]


def samples(seed=17, n=83, heads=2, group=3, dim=8):
    rng = torch.Generator().manual_seed(seed)
    q = torch.randn(n, heads * group, dim, generator=rng).bfloat16()
    k = (torch.randn(n, heads, dim, generator=rng) + torch.arange(heads)[None, :, None] * 2).bfloat16()
    v = torch.randn(n, heads, dim, generator=rng).bfloat16()
    return q, k, v


def profile(budget=1000):
    return seed_profile(calibration_cache_identity({"rotation_variant": "perhead_center",
        "rotation_calibration_tokens": budget, "mamba_cache_dtype": "bfloat16",
        "mamba_ssm_cache_dtype": "bfloat16"}, "model"))


def ranks(n=83, dim=8, heads=2):
    result, captured = [], []
    for rank in range(2):
        q, k, v = samples(seed=rank + 7, n=n, heads=heads, dim=dim)
        collector = BaseSeedAccumulator(head_dim=dim, local_q_heads=heads * 3,
            local_kv_heads=heads, token_budget=n, testing=True)
        collector.observe(q[:31], k[:31], v[:31])
        collector.observe(q[31:], k[31:], v[31:])
        result.append({"rank": rank, "tp_size": 2, "global_q_heads": heads * 6,
            "global_kv_heads": heads * 2, "prompt_sha256": "prompt",
            "model_fingerprint": "model", "layers": {3: collector.export_payload()}})
        captured.append((q[:SAMPLE_CAP], k[:SAMPLE_CAP], v[:SAMPLE_CAP]))
    return result, captured


def fit(data=None, n=83, dim=8, heads=2):
    return fit_seed_artifact(ranks(n, dim, heads)[0] if data is None else data, profile(),
        layer_names=[NAME], head_dim=dim, global_q_heads=heads * 6,
        global_kv_heads=heads * 2, tp_size=2, prompt_sha256="prompt", testing=True)


@pytest.mark.parametrize("n,heads,group,dim", [(29, 1, 6, 256), (83, 2, 3, 8), (512, 1, 1, 16)])
def test_fp32_covariance_exactly_matches_pinned_source_ast(n, heads, group, dim):
    q, k, v = samples(n=n, heads=heads, group=group, dim=dim)
    _, expected = reference(q, k, v)
    actual = shared_covariance(q, k, v)
    for key in ("q", "v"):
        assert actual[key]["count"] == expected[key]["count"]
        assert actual[key]["cov"].dtype == torch.float32
        assert torch.equal(actual[key]["cov"], expected[key]["cov"])


def test_fit_shared_zero_mean_matches_reference_merge_eigh_uhp():
    data, captured = ranks(n=515)
    reference_rows = []
    for q, k, v in captured:
        namespace, covariance = reference(q, k, v)
        reference_rows.append({NAME: covariance})
    merged = namespace["_merge_rank_stats"](reference_rows, 8)[NAME]
    actual = fit(data, n=515)
    layer = actual["layers"][NAME]
    for kind, key in (("q", "Rk"), ("v", "Rv")):
        expected, _ = namespace["_rotation_from_stats"](merged[kind], 8)
        assert torch.equal(layer[key], expected.expand(4, -1, -1))
    assert torch.count_nonzero(layer["k_mean"]) == 0
    assert actual["calibration"]["tokens"] == 515
    assert actual["calibration"]["statistics_tokens_per_rank"] == 512


def test_collector_owns_only_cap_prefix_and_releases_device_rows():
    q, k, v = samples(n=900, heads=1)
    collector = BaseSeedAccumulator(head_dim=8, local_q_heads=3,
        local_kv_heads=1, token_budget=900, testing=True)
    collector.observe(q[:400], k[:400], v[:400])
    owned = collector._rows["q"][0]
    assert owned.untyped_storage().nbytes() == 400 * 3 * 8 * 4
    owned_copy = owned.clone()
    q[:400].zero_()
    assert torch.equal(owned, owned_copy)
    collector.observe(q[400:700], k[400:700], v[400:700])
    assert collector._rows is None and collector._captured == 512 and collector.count == 700
    before = copy.deepcopy(collector._statistics)
    collector.observe(q[700:] * 100, k[700:] * 100, v[700:] * 100)
    result = collector.export_payload()
    assert result["count"] == 900 and result["statistics_count"] == 512
    for kind in ("q", "v"):
        assert torch.equal(before[kind]["cov"], result["oscar_base_seed"][kind]["cov"])
    assert "q_gram" not in result and "k" not in result and "v" not in result


def test_collector_rejects_cpu_production_and_incomplete_capture():
    q, k, v = samples(n=17)
    collector = BaseSeedAccumulator(head_dim=8, local_q_heads=6, local_kv_heads=2, token_budget=17)
    with pytest.raises(RuntimeError, match="NPU"):
        collector.observe(q, k, v)
    collector = BaseSeedAccumulator(head_dim=8, local_q_heads=6, local_kv_heads=2,
                                    token_budget=18, testing=True)
    collector.observe(q, k, v)
    with pytest.raises(ValueError, match="budget"):
        collector.export_payload()


def test_zero_energy_matches_reference_without_changing_equation():
    q, k, v = samples(n=17)
    k.zero_()
    _, expected = reference(q, k, v)
    actual = shared_covariance(q, k, v)
    assert torch.equal(actual["v"]["cov"], expected["v"]["cov"])
    assert torch.count_nonzero(actual["v"]["cov"]) == 0


def test_seed_profile_legacy_sampling_identity_round_trip_and_immutability():
    old = calibration_cache_identity({"rotation_variant": "perhead_center",
        "mamba_cache_dtype": "bfloat16", "mamba_ssm_cache_dtype": "bfloat16"}, "model")
    snapshot = copy.deepcopy(old)
    new = seed_profile(old)
    assert old == snapshot and legacy_base_profile(new) == old
    assert legacy_base_profile(old) == old and legacy_base_profile(old) is not old
    assert new["recipe_version"] == 6 and "official_commit" not in new
    assert new["source_provenance"] == source_provenance()
    assert validate_seed_profile(new, "model") == new
    with pytest.raises(ValueError, match="fingerprint"):
        validate_seed_profile(new, "other")


@pytest.mark.parametrize("change,match", [
    (lambda x: x.pop(), "complete unique"),
    (lambda x: x[1].update(prompt_sha256="other"), "identity"),
    (lambda x: x[1]["layers"].clear(), "coverage"),
    (lambda x: x[0]["layers"][3].pop("oscar_base_seed"), "covariance"),
    (lambda x: x[0]["layers"][3].update(statistics_count=82), "geometry"),
    (lambda x: x[0]["layers"][3].update(source_device="npu"), "production/test"),
    (lambda x: x[0]["layers"][3]["oscar_base_seed"]["q"].update(cov=torch.eye(8).double()), "covariance"),
])
def test_old_or_malformed_source_statistics_cannot_be_relabelled(change, match):
    data, _ = ranks()
    change(data)
    with pytest.raises(ValueError, match=match):
        fit(data)


@pytest.mark.parametrize("change,match", [
    (lambda x: x["layers"][NAME]["k_mean"].fill_(.1), "zero key mean"),
    (lambda x: x["layers"][NAME]["Rk"][1].mul_(-1), "identical shared"),
    (lambda x: x["layers"][NAME]["Rv"].mul_(2), "orthogonal"),
    (lambda x: x["source_provenance"].update(sample_cap=1024), "source/model"),
    (lambda x: x["calibration_profile"].update(recipe_version=3), "profile differs"),
    (lambda x: x.update(matrices_sha256="0" * 64), "digest"),
    (lambda x: x["calibration"].update(statistics_tokens_per_rank=100), "sampling"),
])
def test_seed_artifact_binds_shared_unmodified_source_identity(change, match):
    artifact = fit()
    expected = copy.deepcopy(artifact["calibration_profile"])
    change(artifact)
    with pytest.raises(ValueError, match=match):
        validate_seed_artifact(artifact, expected_profile=expected, testing=True)


def test_test_seed_rejected_by_production_loader():
    artifact = fit()
    with pytest.raises(ValueError, match="test-only"):
        validate_seed_artifact(artifact, expected_profile=profile())


def test_reference_snapshot_matches_explicitly_requested_local_reference_if_available():
    selected = os.environ.get("OSCAR_BASE_REFERENCE_ROOT")
    if not selected:
        pytest.skip("portable CI uses the pinned AST snapshot; explicit local source audit optional")
    root = Path(selected)
    expected = ast.parse(REFERENCE_SOURCE)
    names = {node.name for node in expected.body}
    observed = []
    for filename, digest in SOURCE_FILES.items():
        source = (root / filename).read_bytes()
        assert hashlib.sha256(source).hexdigest() == digest
        observed.extend(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name in names)
    assert ast.dump(ast.Module(body=observed, type_ignores=[])) == ast.dump(expected)
