"""Archive #126/#145/#173: independent raw-space single-source expectation.

This test-only overlay leaves the frozen v4 reference untouched. It consumes
original fixture Q/K/V, never implementation outputs or reconstructed raw
projections. Nonzero history uses the original v4 sampled reference exactly.
"""
from pathlib import Path
import json

import torch

CONTRACT = 'oscar_native_exact_current_v1'
ACCEPTANCE = Path(__file__).resolve().parents[1] / 'configs/acceptance_native_exact_current.json'


def load_acceptance():
    value = json.loads(ACCEPTANCE.read_text())
    if (value.get('contract') != CONTRACT or value.get('policy') != 'native_exact'
            or value.get('frozen_before_measurement') is not True
            or value.get('fused_attention') != {'atol': .005, 'rtol': .005, 'per_element': True}):
        raise ValueError('native-exact current predeclared contract changed')
    return value


def raw_current_sample(query, key, value, scale):
    """FP64 dense causal-query reference with the native BF16 output boundary."""
    if any(x.device.type != 'cpu' or x.dtype != torch.bfloat16 for x in (query, key, value)):
        raise ValueError('offline original BF16 Q/K/V required')
    if (query.ndim != 3 or key.ndim != 3 or value.shape != key.shape or
            query.shape[0] != 1 or key.shape[0] < 1 or query.shape[2] != key.shape[2] or
            query.shape[1] % key.shape[1]):
        raise ValueError('single query and nonempty matching GQA keys required')
    if not all(bool(torch.isfinite(x).all()) for x in (query, key, value)):
        raise ValueError('native-exact numerical oracle requires finite original inputs')
    _, hq, dim = query.shape
    hk = key.shape[1]
    scores = torch.einsum('hgd,khd->hgk', query.double().reshape(hk, hq//hk, dim), key.double()) * scale
    lse = torch.logsumexp(scores, -1).reshape(hq).float()
    probability = torch.softmax(scores, -1)
    output = torch.einsum('hgk,khd->hgd', probability, value.double()).reshape(hq, dim).bfloat16().float()
    return output, lse


def sampled_fixture_expected(fixture, *, native_current, dequant_mode, indices=None):
    from .official_runtime_oracle import sampled_fixture_expected as v4_expected
    load_acceptance()
    spec = fixture['spec']
    if indices is None:
        indices = sorted(fixture['expected'])
    native = (tuple(native_current) if isinstance(native_current, (tuple, list))
              else (native_current,) * len(spec.contexts))
    if len(native) != len(spec.contexts) or any(type(x) is not bool for x in native):
        raise ValueError('explicit per-request native-current qualification required')
    exact, rotated = [], []
    offset = 0
    for length, context, enabled in zip(spec.qlens, spec.contexts, native):
        for index in indices:
            if offset <= index < offset + length:
                (exact if enabled and context == 0 else rotated).append((index, offset))
        offset += length
    if len(exact) + len(rotated) != len(indices):
        raise ValueError('sample lies outside the declared requests')
    result = (v4_expected(fixture, native_current=native_current, dequant_mode=dequant_mode,
                         indices=[i for i, _ in rotated]) if rotated else {})
    if exact:
        original = fixture.get('original_inputs')
        if not isinstance(original, dict) or not {'q', 'ck', 'cv'} <= set(original):
            raise ValueError('raw current oracle requires original projection inputs')
        for index, begin in exact:
            result[index] = raw_current_sample(original['q'][index:index+1],
                original['ck'][begin:index+1], original['cv'][begin:index+1], fixture['scale'])
    return result
