"""Offline per-head K clipping under a predeclared local score proxy.

Archive G19-G24/#13-25/#86/#97/#148: use the same captured model/head rows,
finite FP64 moments, exact codec rounding and explicit evidence boundaries.
Official sources specify the per-head clip index/codec, not an automatic
selection objective. This policy is LOCAL and is not an official selector,
attention-loss measurement, LongBench acceptance or quality-recovery promise.
No model/NPU calls or test-oracle imports occur in this preparation module.
"""
from __future__ import annotations

import ctypes
import math
from collections.abc import Mapping

POLICY = 'centered_rotated_score_proxy_v1'
CRITERION = 'both_modes_raw_and_centered_nonworse_minimax_relative'
CANDIDATE_RATIOS = (.80, .85, .90, .92, .94, .96, .98, 1.0)
MAX_SAMPLES = 1024
METRICS = ('prefill_raw', 'prefill_centered', 'decode_raw', 'decode_centered')


def validate_policy(policy):
    if (not isinstance(policy, Mapping) or policy.get('policy') != POLICY or
            policy.get('criterion') != CRITERION or
            tuple(policy.get('candidate_ratios', ())) != CANDIDATE_RATIOS or
            policy.get('max_samples') != MAX_SAMPLES or
            tuple(policy.get('dequant_modes', ())) != (0, 1)):
        raise ValueError('per-head clip policy differs from the frozen local selection contract')
    baseline = policy.get('baseline_ratio')
    if (type(baseline) not in (int, float) or not math.isfinite(baseline) or
            not 0 < baseline <= 1):
        raise ValueError('clip baseline_ratio must be finite and in (0,1]')
    return float(baseline)


def clip_index(ratio, dim):
    """Match the writer's float32 scalar ABI before multiplying by D."""
    return min(int(ctypes.c_float(float(ratio)).value * dim), dim - 1)


def _bf16(value):
    import torch
    return value.to(torch.bfloat16).float()


def candidate_scores(rotated_keys_bf16, rotated_query_covariance_fp64, ratio):
    """K codec error in rotation space; two dequant modes, raw and demeaned.

    Raw error retains the history-versus-exact-source logit bias. Demeaned
    error measures variation across sampled keys. Both must improve or tie.
    """
    import torch
    x = rotated_keys_bf16.float()
    threshold = x.abs().sort(-1).values[:, clip_index(ratio, x.shape[-1]), None]
    clipped = x.clamp(-threshold, threshold)
    minimum, maximum = clipped.amin(-1, keepdim=True), clipped.amax(-1, keepdim=True)
    scale = (maximum - minimum).clamp_min(1e-8) / 3.0
    zero = -minimum / scale
    codes = (clipped / scale + zero + .5).clamp(0, 3).to(torch.uint8)
    scores = {}
    for mode in ('prefill', 'decode'):
        if mode == 'prefill':
            decoded = _bf16((codes.float() - zero) * scale)
        else:
            decoded = _bf16(_bf16(codes.float() - _bf16(zero)) * _bf16(scale))
        error = decoded.double() - x.double()
        for suffix, delta in (('raw', error), ('centered', error - error.mean(0))):
            energy = ((delta @ rotated_query_covariance_fp64) * delta).sum(-1).mean() / x.shape[-1]
            value = float(energy)
            if not math.isfinite(value) or value < 0:
                raise ValueError('per-head clip score must be finite and nonnegative; invalid covariance/statistics')
            scores[mode + '_' + suffix] = value
    return scores


def select_head_clip(key_rows, q_cov, rotation, k_mean, policy):
    """Return one ratio and an auditable four-term non-regression decision."""
    import torch
    baseline = validate_policy(policy)
    if (key_rows.ndim != 2 or not key_rows.shape[0] or
            q_cov.shape != (key_rows.shape[1], key_rows.shape[1]) or
            rotation.shape != q_cov.shape or k_mean.shape != (key_rows.shape[1],)):
        raise ValueError('per-head clip requires K[N,D], covariance/R[D,D] and mean[D]')
    if any(x.device.type != 'cpu' or not x.is_floating_point() or not bool(torch.isfinite(x).all())
           for x in (key_rows, q_cov, rotation, k_mean)):
        raise ValueError('offline clip inputs must be finite CPU tensors')
    dim = key_rows.shape[1]
    if dim < 2 or dim > 256 or dim & (dim - 1):
        raise ValueError('clip calibration requires power-of-two D<=256')
    count = min(key_rows.shape[0], MAX_SAMPLES)
    indices = torch.arange(count, device='cpu') * (key_rows.shape[0] - 1) // max(1, count - 1)
    r = rotation.float().bfloat16()
    centered = (key_rows[indices].bfloat16() - k_mean.float().bfloat16()).bfloat16()
    rotated = (centered.float() @ r.float()).bfloat16()
    covariance = (q_cov.double() + q_cov.double().T) / 2
    cov_rot = r.double().T @ covariance @ r.double()
    # Deduplicate by the actual ABI index; prefer the baseline ratio whenever
    # its index is shared, otherwise the representative nearest the baseline.
    ratios = {}
    for ratio in (*CANDIDATE_RATIOS, baseline):
        index = clip_index(ratio, dim)
        old = ratios.get(index)
        if old is None or (ratio != baseline, abs(ratio-baseline), ratio) < (old != baseline, abs(old-baseline), old):
            ratios[index] = ratio
    baseline_scores = candidate_scores(rotated, cov_rot, baseline)
    def checked(scores):
        if set(scores) != set(METRICS) or any(not math.isfinite(scores[k]) or scores[k] < 0 for k in METRICS):
            raise ValueError('clip candidate must supply four finite nonnegative scores')
        return scores
    checked(baseline_scores)
    candidates = []
    for index, ratio in sorted(ratios.items()):
        scores = baseline_scores if ratio == baseline else checked(candidate_scores(rotated, cov_rot, ratio))
        eligible = all(scores[key] <= baseline_scores[key] for key in METRICS)
        relatives = [(scores[key] / baseline_scores[key] if baseline_scores[key] > 0 else
                      0.0 if scores[key] == 0 else None) for key in METRICS]
        worst = max(relatives) if all(value is not None for value in relatives) else None
        candidates.append({'ratio': float(ratio), 'index': index, 'scores': scores,
                           'eligible': eligible, 'worst_relative': worst})
    selected = min((row for row in candidates if row['eligible']), key=lambda row:
                   (row['worst_relative'], row['ratio'] != baseline, abs(row['ratio']-baseline), row['ratio']))
    return selected['ratio'], {
        'policy': POLICY, 'criterion': CRITERION, 'source': 'local_predeclared_calibration_proxy',
        'official_scope': 'per-head clip index and uniform codec only; selection objective is local',
        'calibration_tokens': int(key_rows.shape[0]), 'sample_count': count,
        'sampling': 'equally_spaced_integer_indices_including_endpoints',
        'baseline_ratio': baseline, 'baseline_index': clip_index(baseline, dim),
        'baseline_scores': baseline_scores, 'selected_ratio': selected['ratio'],
        'selected_index': selected['index'], 'selected_scores': selected['scores'],
        'score_formula': 'mean(delta.T R_bf16.T M_q R_bf16 delta)/D; raw and per-dimension demeaned',
        'query_rounding': 'q_gram does not retain BF16(QR) query-side rounding',
        'quality_acceptance': 'not_established; calibration proxy is not attention loss or model score',
        'candidates': candidates}
