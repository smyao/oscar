"""Archive #13-20/#126/#145/#148/#164: TEST-ONLY sampled v4 lifecycle golden.

Inputs are independent fixture/cache snapshots, never candidate attention
outputs. Every source uses rotated BF16 Q/K/V and FP32 CV partials. The one
zero-history native-current result has its separate dense BF16-visible oracle;
no claim is made about native FIA's closed internal probability rounding.
D.4 permits this bounded offline diagnostic, not a serving history restore.
"""
from __future__ import annotations

import math
import torch

from .official_reference import (AttentionResult, attention_tiles, merge_and_inverse,
                                native_current_part, unpack_kv)


def _native_current_only(query, key, value):
    """Independent dense expectation for a single sampled causal query."""
    _, heads, dim = query.shape
    hk = key.shape[1]
    group = heads // hk
    q = query.float().reshape(hk, group, dim)
    scores = torch.einsum("hgd,khd->hgk", q, key.float()) * (dim ** -.5)
    if not torch.isfinite(scores).all():
        raise ValueError("native-current dense oracle has nonfinite scores")
    lse = torch.logsumexp(scores, -1).reshape(1, heads)
    probability = torch.softmax(scores, -1)
    out = torch.einsum("hgk,khd->hgd", probability, value.float()).reshape(1, heads, dim)
    return native_current_part(out.bfloat16(), lse)


def request_samples(q, qr, ck, cv, context, rv, history, precise, samples, *,
                    cached_quant_end, recent=256, sink=64, splits=1,
                    native_current=False, dequant_mode=None):
    if dequant_mode not in (0, 1):
        raise ValueError("explicit dequant_mode 0/1 required")
    if type(cached_quant_end) is not int or not min(sink, context) <= cached_quant_end <= context:
        raise ValueError("explicit valid fixed cached_quant_end required")
    if q.shape != qr.shape or not torch.equal(q.to(torch.bfloat16), qr.to(torch.bfloat16)):
        raise ValueError("all sampled v4 sources require one rotated query frame")
    length, hq, _ = qr.shape
    hk = ck.shape[1]
    qtile = 128 // (hq // hk)
    expected = {}
    for local in samples:
        begin = (local // qtile) * qtile
        count = min(qtile, length - begin)
        position = context + local
        if context == 0 and native_current:
            part = _native_current_only(qr[local:local+1], ck[:local+1], cv[:local+1])
            answer = merge_and_inverse([part], rv)
            expected[local] = answer.output[0].float(), answer.lse[0]
            continue
        # Current BF16 spans the producer group; only visibility is causal.
        bounds = ((min(sink, context), context),
                  (0, min(sink, context) + context - cached_quant_end),
                  (context, context + begin + count))
        parts = []
        for kind, (first, last) in enumerate(bounds):
            width = (last - first + splits - 1) // splits
            for split in range(splits):
                lo = min(last, first + split * width)
                hi = min(last, lo + width)
                candidate = torch.arange(lo, hi, dtype=torch.int64)
                if kind == 0:
                    key, value = history(candidate)
                    visible = (candidate >= sink) & (candidate < cached_quant_end) & (candidate <= position)
                elif kind == 1:
                    sink_count = min(sink, context)
                    positions = torch.where(candidate < sink_count, candidate,
                                            cached_quant_end + candidate - sink_count)
                    key, value = precise(positions)
                    visible = positions <= position
                else:
                    key, value = ck[candidate-context], cv[candidate-context]
                    visible = candidate <= position
                parts.append(attention_tiles(qr[local:local+1], key, value, visible=visible[None]))
        answer = merge_and_inverse(parts, rv)
        expected[local] = answer.output[0].float(), answer.lse[0]
    return expected


def sampled_fixture_expected(fixture, *, native_current, dequant_mode, indices=None):
    if fixture.get("cache_format") != "striped_rotated_fp32_v3":
        raise ValueError("official lifecycle requires rotated-window fixture format")
    if "cached_quant_ends" not in fixture:
        raise ValueError("v4 lifecycle oracle requires cached_quant_ends")
    data, spec = fixture["cpu"], fixture["spec"]
    dim, hk = spec.dim, spec.kv_heads
    if indices is None:
        indices = sorted(fixture["expected"])
    block_tokens = fixture["stride"] // (hk * (dim // 2 + 16))
    prefix = data["raw"].numel() - fixture["blocks"] * fixture["stride"]
    raw = data["raw"][prefix:].view(fixture["blocks"], block_tokens, hk, dim // 2 + 16)
    sink = fixture.get("sink_tokens", 64)
    ring = fixture.get("window_ring_tokens", spec.recent_tokens + fixture.get("speculative_tokens", 3))
    out, offset = {}, 0
    for request, (length, context, pages) in enumerate(zip(spec.qlens, spec.contexts, fixture["page_assignments"])):
        samples = [i-offset for i in indices if offset <= i < offset+length]
        if not samples:
            offset += length
            continue
        mode = dequant_mode[request] if isinstance(dequant_mode, (tuple, list)) else dequant_mode
        native = native_current[request] if isinstance(native_current, (tuple, list)) else native_current
        mapping = torch.tensor(pages, dtype=torch.int64)
        def history(positions):
            physical = mapping[positions // block_tokens]
            return unpack_kv(raw[physical, positions % block_tokens], dim, mode)
        def precise(positions):
            physical = mapping[positions // block_tokens]
            within = positions % block_tokens
            row = torch.where(positions < sink, positions, sink + within % ring)
            if not torch.equal(data["tags"][physical, row], within):
                raise ValueError("independent official lifecycle window tag mismatch")
            return data["wk"][physical, row], data["wv"][physical, row]
        answers = request_samples(data["q"][offset:offset+length], data["qr"][offset:offset+length],
            data["ck"][offset:offset+length], data["cv"][offset:offset+length], context,
            data["rv"], history, precise, samples, recent=spec.recent_tokens, sink=sink,
            splits=spec.splits, native_current=bool(native), dequant_mode=mode,
            cached_quant_end=fixture["cached_quant_ends"][request])
        out.update({offset+i: value for i, value in answers.items()})
        offset += length
    if set(out) != set(indices):
        raise RuntimeError("official lifecycle oracle lost a requested query")
    return out
