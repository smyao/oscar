"""Independent TEST-ONLY V1 uniform OSCAR reference, contract v4.

Archive #13-20/#126/#145/#148/#164: output, metadata and error domains are
separate; a CPU oracle pass is not NPU, graph or LongBench acceptance.
Startup D.4: this offline reconstruction is bounded by its supplied fixture;
production must never materialize full-history BF16 (old 6499.8-6655.1 ms).
Pinned upstream 38680852: quantized_kv_prefill.py rotation/inverse functions,
kv_quant_kernels.py affine quantizer, decode_attention.py FP32 stage-2 merge.
No old v3 oracle or production kernel output defines this numerical golden.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import torch

CONTRACT = "oscar_bf16_attention_v4"
CACHE_FORMAT = "striped_rotated_fp32_v3"
UPSTREAM_COMMIT = "38680852e52c10bd6b9e9a64ed9bc815e742859b"
ACCEPTANCE = Path(__file__).resolve().parents[1] / "configs/acceptance_official_bf16.json"


@dataclass(frozen=True)
class AttentionResult:
    output: torch.Tensor
    lse: torch.Tensor
    native_current: bool = False


def load_acceptance(path=ACCEPTANCE):
    value = json.loads(Path(path).read_text())
    if (value.get("contract") != CONTRACT or value.get("cache_format") != CACHE_FORMAT
            or value.get("frozen_before_measurement") is not True):
        raise ValueError("independent v4 premeasurement contract required")
    if value.get("fused_attention") != {"atol": .005, "rtol": .005, "per_element": True}:
        raise ValueError("v4 tolerance changed")
    if value.get("protocol") != {"status_and_metadata_exact": True,
            "padding_output_exact_zero": True, "padding_lse_exact_negative_infinity": True}:
        raise ValueError("v4 exact protocol changed")
    return value


def _finite(name, value):
    if value.device.type != "cpu":
        raise ValueError("official_reference is an offline CPU oracle, never a serving fallback")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(name + " must be finite")


def round_bf16(value):
    if not value.is_floating_point():
        raise ValueError("BF16 rounding requires floating input")
    _finite("operand before BF16 rounding", value)
    result = value.to(torch.bfloat16).float()
    _finite("operand after BF16 rounding", result)
    return result


def rotate_bf16(value, rotation):
    """A shared V1 matrix: BF16 operands, FP32 accumulation, BF16 output."""
    if rotation.ndim != 2 or rotation.shape[0] != rotation.shape[1] or value.shape[-1] != rotation.shape[0]:
        raise ValueError("V1 requires a shared square rotation; V2/per-head artifacts are outside this contract")
    return round_bf16(round_bf16(value) @ round_bf16(rotation)).to(torch.bfloat16)


def clip_rows(value, ratio):
    if value.dtype != torch.bfloat16 or value.shape[-1] not in (64, 128, 256):
        raise ValueError("rotated BF16 D64/128/256 rows required")
    if not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError("invalid clip ratio")
    x = value.float()
    _finite("rotated input", x)
    if ratio > 0:
        index = min(int(ratio * x.shape[-1]), x.shape[-1] - 1)
        threshold = x.abs().sort(-1).values[..., index, None]
        x = x.clamp(-threshold, threshold)
    return x


def quantize_rows(value, ratio=0.):
    """Uniform normalized-zero codec from the pinned official FP32 formula."""
    x = clip_rows(value, ratio)
    low, high = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
    scale = (high - low).clamp_min(1e-8) / 3.
    zero = -low / scale
    _finite("scale", scale)
    _finite("normalized zero", zero)
    codes = torch.clamp(x / scale + zero + .5, 0, 3).to(torch.uint8)
    return codes, scale.squeeze(-1), zero.squeeze(-1)


def dequantize_rows(codes, scale, zero, mode):
    if codes.dtype != torch.uint8 or bool((codes > 3).any()):
        raise ValueError("INT2 codes required")
    if (scale.dtype != torch.float32 or zero.dtype != torch.float32
            or scale.shape != codes.shape[:-1] or zero.shape != scale.shape):
        raise ValueError("one FP32 scale/normalized zero per row required")
    _finite("scale", scale)
    _finite("zero", zero)
    if bool((scale <= 0).any()):
        raise ValueError("scale must be positive")
    if mode == 0:
        return round_bf16((codes.float() - zero[..., None]) * scale[..., None])
    if mode == 1:
        s, z = round_bf16(scale), round_bf16(zero)
        if bool((s <= 0).any()):
            raise ValueError("decode scale underflows BF16")
        return round_bf16(round_bf16(codes.float() - z[..., None]) * s[..., None])
    raise ValueError("explicit dequant_mode 0(prefill) or 1(decode) required")


def _pack_codes(codes):
    dim = codes.shape[-1]
    planes = codes.long().reshape(*codes.shape[:-1], 8, dim // 8)
    words = sum(planes[..., plane, :] << (2 * plane) for plane in range(8))
    return torch.stack((words & 255, words >> 8), -1).to(torch.uint8).flatten(-2)


def pack_kv(key, value, k_clip=.96, v_clip=.92):
    if key.shape != value.shape:
        raise ValueError("matching rotated K/V rows required")
    k, ks, kz = quantize_rows(key, k_clip)
    v, vs, vz = quantize_rows(value, v_clip)
    metadata = torch.stack((ks, kz, vs, vz), -1).contiguous().view(torch.uint8)
    return torch.cat((_pack_codes(k), _pack_codes(v), metadata), -1)


def unpack_kv(packed, dim, mode):
    if packed.dtype != torch.uint8 or dim not in (64, 128, 256) or packed.shape[-1] != dim // 2 + 16:
        raise ValueError("striped_rotated_fp32_v3 slot shape/dtype required")
    metadata = packed[..., dim // 2:].contiguous().view(torch.float32)
    result = []
    for side in (0, 1):
        raw = packed[..., side * dim // 4:(side + 1) * dim // 4].long()
        words = raw[..., 0::2] | raw[..., 1::2] << 8
        codes = torch.stack([(words >> (2 * i)) & 3 for i in range(8)], -2).flatten(-2).to(torch.uint8)
        result.append(dequantize_rows(codes, metadata[..., side * 2], metadata[..., side * 2 + 1], mode))
    return tuple(result)


def rotated_attention_tiles(query, key, value, *, visible=None, scale=None, tile_size=256):
    """All sources share rotated Q/K/V; normalized partials stay FP32.

    KV256 is the declared Ascend organization, not CUDA's adaptive tile size.
    Each tile's PV probability is RNE BF16; denominator/max/alpha stay FP32.
    There is intentionally no value_rotation argument or per-partial inverse.
    """
    if query.ndim != 3 or key.ndim != 3 or key.shape != value.shape:
        raise ValueError("Q[Q,Hq,D] and matching K/V[K,Hkv,D] required")
    nq, hq, dim = query.shape
    nk, hk, kd = key.shape
    if dim != kd or min(hq, hk, dim) <= 0 or hq % hk or tile_size != 256:
        raise ValueError("invalid GQA/head dimension or noncontract tile size")
    if visible is None:
        visible = torch.ones((nq, nk), dtype=torch.bool)
    if visible.device.type != "cpu" or visible.dtype != torch.bool or visible.shape != (nq, nk):
        raise ValueError("explicit CPU bool [Q,K] visibility required")
    factor = dim ** -.5 if scale is None else scale
    if not isinstance(factor, (int, float)) or not math.isfinite(factor) or factor <= 0:
        raise ValueError("invalid attention scale")
    q, k, v = round_bf16(query), round_bf16(key), round_bf16(value)
    heads = torch.arange(hq) // (hq // hk)
    k, v = k[:, heads], v[:, heads]
    maximum = torch.full((nq, hq), -math.inf, dtype=torch.float32)
    denominator = torch.zeros_like(maximum)
    accumulator = torch.zeros((nq, hq, dim), dtype=torch.float32)
    for begin in range(0, nk, tile_size):
        end = min(begin + tile_size, nk)
        score = torch.einsum("qhd,khd->qhk", q, k[begin:end]) * factor
        _finite("QK scores before masking", score)
        score = score.masked_fill(~visible[:, None, begin:end], -math.inf)
        new_max = torch.maximum(maximum, score.amax(-1))
        safe_max = torch.where(torch.isfinite(new_max), new_max, torch.zeros_like(new_max))
        alpha = torch.where(torch.isfinite(maximum), torch.exp(maximum - safe_max), torch.zeros_like(maximum))
        probability = torch.exp(score - safe_max[..., None])
        denominator = denominator * alpha + probability.sum(-1)
        pv = torch.einsum("qhk,khd->qhd", round_bf16(probability), v[begin:end])
        accumulator = accumulator * alpha[..., None] + pv
        maximum = new_max
    live = denominator > 0
    safe_sum = torch.where(live, denominator, torch.ones_like(denominator))
    output = accumulator / safe_sum[..., None]
    lse = torch.where(live, maximum + torch.log(safe_sum), torch.full_like(maximum, -math.inf))
    _finite("rotated partial", output)
    return AttentionResult(output, lse)


attention_tiles = rotated_attention_tiles


def merge_rotated_parts(parts: Sequence[AttentionResult]):
    """Independent FP32 LSE weighted merge; no partial is narrowed or rotated."""
    if not parts:
        raise ValueError("at least one rotated partial required")
    if any(part.native_current for part in parts) and len(parts) != 1:
        raise ValueError("native BF16 current is permitted only as a single zero-history source")
    shape = parts[0].output.shape
    for part in parts:
        if part.output.shape != shape or part.lse.shape != shape[:-1]:
            raise ValueError("partial output/LSE shapes differ")
        if part.output.dtype != torch.float32 or part.lse.dtype != torch.float32:
            raise ValueError("normalized partial output and LSE must remain FP32")
        _finite("normalized partial", part.output)
        if part.lse.device.type != "cpu" or bool((torch.isnan(part.lse) | torch.isposinf(part.lse)).any()):
            raise ValueError("partial LSE must be finite or negative infinity")
        if bool((part.output[torch.isneginf(part.lse)] != 0).any()):
            raise ValueError("empty partial rows must contain exact zero")
    stacked_lse = torch.stack([part.lse for part in parts], dim=0)
    maximum = stacked_lse.amax(0)
    live = torch.isfinite(maximum)
    safe_max = torch.where(live, maximum, torch.zeros_like(maximum))
    weights = torch.exp(stacked_lse - safe_max)
    total = weights.sum(0)
    safe_total = torch.where(live, total, torch.ones_like(total))
    weighted = torch.stack([part.output for part in parts]) * weights[..., None]
    output = weighted.sum(0) / safe_total[..., None]
    lse = torch.where(live, maximum + torch.log(safe_total), torch.full_like(maximum, -math.inf))
    return AttentionResult(output, lse)


merge_parts = merge_rotated_parts


def native_current_part(output, lse):
    """Bridge an independently computed zero-history native FIA BF16 result.

    This describes its visible dtype only; the closed FIA internal probability
    rounding is not asserted to equal the KV256 or CUDA implementation.
    """
    if output.dtype != torch.bfloat16 or lse.dtype != torch.float32 or lse.shape != output.shape[:-1]:
        raise ValueError("native current requires rotated BF16 output and FP32 LSE")
    _finite("native rotated current", output)
    return AttentionResult(output.float(), lse, native_current=True)


def finalize_output(merged: AttentionResult, value_rotation):
    """One final BF16 inverse after the entire FP32 rotated-domain merge."""
    if merged.output.dtype != torch.float32 or merged.lse.dtype != torch.float32:
        raise ValueError("final inverse requires the FP32 merged result")
    output = rotate_bf16(merged.output.to(torch.bfloat16), value_rotation.T)
    if merged.lse.shape != output.shape[:-1] or bool((torch.isnan(merged.lse) | torch.isposinf(merged.lse)).any()):
        raise ValueError("merged LSE must match output and be finite or negative infinity")
    empty = torch.isneginf(merged.lse)
    if bool((merged.output[empty] != 0).any()):
        raise ValueError("empty merged rows must contain exact zero")
    output[empty] = 0
    return AttentionResult(output, merged.lse)


def merge_and_inverse(parts, value_rotation):
    merged = merge_rotated_parts(parts)
    return finalize_output(merged, value_rotation)


def error_metrics(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError("comparison shape mismatch")
    if bool((torch.isnan(expected) | torch.isposinf(expected)).any()):
        raise ValueError("reference must be finite or negative infinity")
    finite = torch.isfinite(expected)
    valid = (torch.equal(torch.isneginf(actual), torch.isneginf(expected))
             and bool(torch.isfinite(actual[finite]).all()))
    delta = (actual.float()[finite] - expected.float()[finite]).abs()
    threshold = .005 + .005 * expected.float()[finite].abs()
    return {"passed": bool(valid and (delta <= threshold).all()),
            "max_abs": float(delta.max()) if delta.numel() else 0.,
            "bad_elements": int((delta > threshold).sum()) if valid else int(actual.numel()),
            "atol": .005, "rtol": .005}


def exact_protocol(actual, expected):
    """No numeric tolerance is valid for metadata/status/padding contracts."""
    if set(actual) != set(expected):
        return False
    return all(a.dtype == expected[name].dtype and a.shape == expected[name].shape
               and torch.equal(a.contiguous().view(torch.uint8),
                               expected[name].contiguous().view(torch.uint8))
               for name, a in actual.items())


def compare_results(actual, expected):
    load_acceptance()
    output, lse = error_metrics(actual.output, expected.output), error_metrics(actual.lse, expected.lse)
    return {"contract": CONTRACT, "output": output, "lse": lse,
            "passed": output["passed"] and lse["passed"]}


def fixture_tasks_reference(fixture: dict, tasks: torch.Tensor, dequant_mode: int) -> dict:
    """Complete physical-page task oracle with an explicit persisted tier map.

    cached_quant_ends[request] is the fixed exclusive INT2-prefix end before
    this forward. It is supplied by the fixture, never inferred from q length,
    q position, or a SGLang allocator assumption. q/qr and all exact/current
    tensors are already rotated BF16. CV outputs remain rotated FP32.
    """
    if dequant_mode not in (0, 1):
        raise ValueError("explicit dequant_mode 0/1 required")
    if "cached_quant_ends" not in fixture:
        raise ValueError("v4 fixture requires explicit cached_quant_ends")
    cuts = fixture["cached_quant_ends"]
    if not isinstance(cuts, (list, tuple)) or any(type(cut) is not int or cut < 0 for cut in cuts):
        raise ValueError("cached_quant_ends must be nonnegative integer request boundaries")
    if fixture.get("cache_format", CACHE_FORMAT) != CACHE_FORMAT:
        raise ValueError("v4 fixture cannot interpret an old original-space window cache")
    data, spec = fixture["cpu"], fixture["spec"]
    q, qr = data["q"], data["qr"]
    n, hq, dim = q.shape
    hk, segments = spec.kv_heads, 3 * spec.splits
    if type(hk) is not int or hk <= 0 or type(spec.splits) is not int or spec.splits <= 0:
        raise ValueError("positive KV heads and split count required")
    group = hq // hk
    if q.dtype != torch.bfloat16 or qr.dtype != torch.bfloat16 or q.shape != qr.shape:
        raise ValueError("v4 fixture requires matching BF16 rotated q/qr")
    if data["slots"].shape != (n,) or data["slots"].dtype not in (torch.int32, torch.int64):
        raise ValueError("one integer slot per query required")
    for name in ("ck", "cv", "wk", "wv"):
        if data[name].dtype != torch.bfloat16 or data[name].device.type != "cpu":
            raise ValueError("v4 exact/current input must be rotated CPU BF16: " + name)
    if data["ck"].shape != (n, hk, dim) or data["cv"].shape != data["ck"].shape:
        raise ValueError("current rotated K/V geometry mismatch")
    if data["wk"].shape != data["wv"].shape or data["wk"].shape[-2:] != (hk, dim):
        raise ValueError("physical rotated-window K/V geometry mismatch")
    live_slots = data["slots"] >= 0
    if not torch.equal(q[live_slots], qr[live_slots]):
        raise ValueError("all v4 sources must use the same rotated query")
    if tasks.dtype != torch.int64 or tasks.device.type != "cpu" or tasks.shape != (n * hk * segments, 16):
        raise ValueError("complete actual CPU prepare task table required")
    if q.device.type != "cpu" or min(hq, hk, dim) <= 0 or hq % hk:
        raise ValueError("CPU fixture with valid GQA geometry required")
    sink = fixture.get("sink_tokens", 64)
    ring = fixture.get("window_ring_tokens", spec.recent_tokens + fixture.get("speculative_tokens", 3))
    if type(sink) is not int or sink < 0 or type(ring) is not int or ring <= 0:
        raise ValueError("invalid physical exact-window geometry")
    slot_bytes = dim // 2 + 16
    block_tokens = fixture["stride"] // (hk * slot_bytes)
    if block_tokens <= 0 or block_tokens % 128 or fixture["stride"] != block_tokens * hk * slot_bytes:
        raise ValueError("physical page and native metadata128 geometry differ")
    prefix = data["raw"].numel() - fixture["blocks"] * fixture["stride"]
    if prefix < 0:
        raise ValueError("raw allocation cannot contain physical pages")
    raw = data["raw"][prefix:].view(fixture["blocks"], block_tokens, hk, slot_bytes)
    partial = torch.zeros((n, hq, segments, dim), dtype=torch.float32)
    lse = torch.full((n, hq, segments), -math.inf, dtype=torch.float32)
    written = torch.zeros((n, hq, segments), dtype=torch.bool)

    def physical(request, positions):
        columns = positions // 128
        if not 0 <= request < data["table"].shape[0] or bool((columns < 0).any()) or bool((columns >= data["table"].shape[1]).any()):
            raise ValueError("request/native metadata column invalid")
        virtual = data["table"][request, columns].long()
        ratio = block_tokens // 128
        block, within = virtual // ratio, (virtual % ratio) * 128 + positions % 128
        if bool((virtual < 0).any()) or bool((block >= fixture["blocks"]).any()):
            raise ValueError("physical page identity invalid")
        return block, within

    for row in tasks.tolist():
        qb, count, head, begin, end, request, segment, kind, context, request_begin, error = row[:11]
        if error:
            raise ValueError("error tasks require exact-status qualification, not a numeric golden")
        if count <= 0:
            continue
        if (kind not in (0, 1, 2) or not 0 <= qb < qb + count <= n or not 0 <= head < hk
                or not 0 <= segment < segments or segment // spec.splits != kind
                or not 0 <= request < len(cuts) or not 0 <= begin <= end):
            raise ValueError("task owner/range/source invalid")
        cut = cuts[request]
        if not min(sink, context) <= cut <= context:
            raise ValueError("cached INT2 end must lie between sink and context")
        candidate = torch.arange(begin, end, dtype=torch.int64)
        query_positions = context + torch.arange(qb, qb + count) - request_begin
        head_slice = slice(head * group, (head + 1) * group)
        if kind == 0:
            block, within = physical(request, candidate)
            key, value = unpack_kv(raw[block, within, head:head + 1], dim, dequant_mode)
            visible = ((candidate[None] >= sink) & (candidate[None] < cut)
                       & (candidate[None] < context) & (candidate[None] <= query_positions[:, None]))
        elif kind == 1:
            sink_count = min(sink, context)
            positions = torch.where(candidate < sink_count, candidate, cut + candidate - sink_count)
            if bool((positions >= context).any()):
                raise ValueError("exact source candidates exceed the persisted cache map")
            block, within = physical(request, positions)
            window_row = torch.where(positions < sink, positions, sink + within % ring)
            if not torch.equal(data["tags"][block, window_row], within):
                raise ValueError("physical exact-window tag mismatch")
            key = data["wk"][block, window_row, head:head + 1]
            value = data["wv"][block, window_row, head:head + 1]
            visible = (((positions[None] < sink) | (positions[None] >= cut))
                       & (positions[None] <= query_positions[:, None]))
        else:
            positions = candidate
            tokens = request_begin + positions - context
            if bool((tokens < 0).any()) or bool((tokens >= n).any()) or bool((positions < context).any()):
                raise ValueError("current source points outside current BF16 tensors")
            key, value = data["ck"][tokens, head:head + 1], data["cv"][tokens, head:head + 1]
            visible = positions[None] <= query_positions[:, None]
        answer = attention_tiles(qr[qb:qb + count, head_slice], key, value,
                                 visible=visible, scale=fixture["scale"])
        if bool(written[qb:qb + count, head_slice, segment].any()):
            raise ValueError("duplicate source/split output owner")
        partial[qb:qb + count, head_slice, segment] = answer.output
        lse[qb:qb + count, head_slice, segment] = answer.lse
        written[qb:qb + count, head_slice, segment] = True
    if not bool(written[live_slots].all()):
        raise ValueError("all live source/split owners required; suppressed native-current needs a separate independent reference")
    merged = merge_parts([AttentionResult(partial[:, :, s], lse[:, :, s]) for s in range(segments)])
    final = finalize_output(merged, data["rv"])
    return {"partial": partial, "lse": lse, "output": final.output, "output_lse": final.lse,
            "merged_rotated": merged.output, "pre_inverse_bf16": merged.output.bfloat16(),
            "final_output": final.output, "contract": CONTRACT,
            "all_task_slots": int(tasks.shape[0]), "live_output_rows": int(written.sum())}
