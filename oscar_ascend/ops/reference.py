"""Test oracle for the pinned OSCAR PR; NEVER a production execution path.

Archive G26/G27, #13-#20: preserve fp16 metadata rounding and byte order.
Archive G30-G34, #6-#12: initialize every attention row, including empty rows.
Reference: PR 46774 / 57286d5d, triton_oscar_store.py:41-77,
oscar_attn.py:235-243, 594-616, 745-749. NPU parity is not established here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class QuantizedVector:
    packed: torch.Tensor
    scale: torch.Tensor
    zero: torch.Tensor
    head_dim: int


@dataclass(frozen=True)
class AttentionResult:
    output: torch.Tensor  # [query, query_head, dimension], fp32
    lse: torch.Tensor  # [query, query_head], natural logarithm


def _finite(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains non-finite values")


def rotate_clip(x: torch.Tensor, rotation: torch.Tensor, clip_ratio: float = 0.0) -> torch.Tensor:
    """PR fp32 rotation and per-vector absolute-percentile clipping."""
    if x.ndim < 1 or rotation.shape != (x.shape[-1], x.shape[-1]):
        raise ValueError("rotation must be square and match the final input dimension")
    if x.device != rotation.device:
        raise ValueError("input and rotation must use the same device")
    if not math.isfinite(clip_ratio) or not 0 <= clip_ratio <= 1:
        raise ValueError("clip_ratio must be finite and in [0, 1]")
    _finite("input", x)
    _finite("rotation", rotation)
    rotated = x.float() @ rotation.float()
    if clip_ratio > 0:
        threshold = torch.quantile(rotated.abs(), clip_ratio, dim=-1, keepdim=True)
        rotated = torch.clamp(rotated, -threshold, threshold)
    return rotated


def pack_int2(indices: torch.Tensor) -> torch.Tensor:
    """Pack the final dimension LSB-first; unused tail bits are zero."""
    if indices.ndim < 1 or indices.shape[-1] < 1:
        raise ValueError("indices need a non-empty final dimension")
    if indices.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("INT2 indices must have an integer dtype")
    if bool(((indices < 0) | (indices > 3)).any()):
        raise ValueError("INT2 indices must be in [0, 3]")
    dimension = indices.shape[-1]
    padded = torch.zeros(*indices.shape[:-1], 4 * ((dimension + 3) // 4),
                         dtype=torch.int32, device=indices.device)
    padded[..., :dimension] = indices.to(torch.int32)
    groups = padded.reshape(*indices.shape[:-1], -1, 4)
    return (groups[..., 0] | (groups[..., 1] << 2)
            | (groups[..., 2] << 4) | (groups[..., 3] << 6)).to(torch.uint8)


def unpack_int2(packed: torch.Tensor, head_dim: int) -> torch.Tensor:
    if head_dim < 1 or packed.ndim < 1 or packed.dtype != torch.uint8:
        raise ValueError("packed must be uint8 and head_dim must be positive")
    if packed.shape[-1] != (head_dim + 3) // 4:
        raise ValueError("packed byte count does not match head_dim")
    shifts = torch.arange(4, dtype=torch.int32, device=packed.device) * 2
    values = (packed.to(torch.int32).unsqueeze(-1) >> shifts) & 3
    return values.reshape(*packed.shape[:-1], -1)[..., :head_dim].to(torch.uint8)


def quantize_int2(x: torch.Tensor) -> QuantizedVector:
    """Match PR arithmetic; reject its undefined zero-scale metadata domain.

    1e-8 rounds to zero in fp16. No epsilon change is hidden here: a
    constant or sufficiently narrow vector fails explicitly instead.
    """
    if x.ndim < 1 or x.shape[-1] < 1 or not x.is_floating_point():
        raise ValueError("x must be floating point with a non-empty final dimension")
    _finite("quantization input", x)
    value = x.float()
    low = value.amin(dim=-1, keepdim=True)
    high = value.amax(dim=-1, keepdim=True)
    scale = ((high - low) / 3).clamp_min(1e-8).to(torch.float16)
    zero = low.to(torch.float16)
    if bool((~torch.isfinite(scale) | (scale <= 0) | ~torch.isfinite(zero)).any()):
        raise ValueError("PR metadata domain violation: fp16 scale must be finite and positive; zero finite")
    # Truncation after +0.5, NOT round-to-even.
    indices = ((value - zero.float()) / scale.float() + 0.5).to(torch.int32).clamp(0, 3)
    return QuantizedVector(pack_int2(indices), scale, zero, value.shape[-1])


def dequantize_int2(vector: QuantizedVector) -> torch.Tensor:
    expected = (*vector.packed.shape[:-1], 1)
    if vector.scale.shape != expected or vector.zero.shape != expected:
        raise ValueError("metadata must contain one scale and zero per vector")
    if vector.scale.dtype != torch.float16 or vector.zero.dtype != torch.float16:
        raise ValueError("metadata dtype must be fp16")
    if vector.scale.device != vector.packed.device or vector.zero.device != vector.packed.device:
        raise ValueError("packed data and metadata must use one device")
    if bool((~torch.isfinite(vector.scale) | (vector.scale <= 0) | ~torch.isfinite(vector.zero)).any()):
        raise ValueError("invalid fp16 quantizer metadata")
    return unpack_int2(vector.packed, vector.head_dim).float() * vector.scale.float() + vector.zero.float()


def _half_to_le_bytes(value: torch.Tensor) -> torch.Tensor:
    bits = value.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    return torch.cat((bits & 255, bits >> 8), dim=-1).to(torch.uint8)


def _le_bytes_to_half(value: torch.Tensor) -> torch.Tensor:
    bits = value[..., :1].to(torch.int32) | (value[..., 1:2].to(torch.int32) << 8)
    return bits.to(torch.int16).contiguous().view(torch.float16)


def encode_kv(key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Encode already rotated/clipped K,V as [K bytes, Ks,Kz,V bytes,Vs,Vz]."""
    if key.shape != value.shape or key.device != value.device:
        raise ValueError("key and value must share shape and device")
    k, v = quantize_int2(key), quantize_int2(value)
    return torch.cat((k.packed, _half_to_le_bytes(k.scale), _half_to_le_bytes(k.zero),
                      v.packed, _half_to_le_bytes(v.scale), _half_to_le_bytes(v.zero)), dim=-1)


def decode_kv(slots: torch.Tensor, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode to fp32 rotated space. Only test code may materialize history."""
    count = (head_dim + 3) // 4
    width = count + 4
    if slots.dtype != torch.uint8 or slots.shape[-1] != 2 * width:
        raise ValueError("KV slot byte geometry mismatch")
    outputs = []
    for start in (0, width):
        region = slots[..., start:start + width]
        outputs.append(dequantize_int2(QuantizedVector(
            region[..., :count], _le_bytes_to_half(region[..., count:count + 2]),
            _le_bytes_to_half(region[..., count + 2:count + 4]), head_dim)))
    return outputs[0], outputs[1]


def window_ranges(length: int, sink_tokens: int = 64, recent_tokens: int = 256) -> tuple[tuple[int, int], ...]:
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0
           for n in (length, sink_tokens, recent_tokens)):
        raise ValueError("window lengths must be nonnegative integers")
    sink = min(sink_tokens, length)
    recent = min(recent_tokens, length - sink)
    return (0, sink), (sink, length - recent), (length - recent, length)


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *,
              scale: float | None = None, causal: bool = True,
              query_positions: torch.Tensor | None = None,
              key_positions: torch.Tensor | None = None) -> AttentionResult:
    """Dense CPU/NPU oracle for prefill, decode or multi-query verification.

    Q is [Q,Hq,D], K/V are [K,Hkv,D]. Default query positions represent a
    suffix of K; explicit positions support noncontiguous segments and tails.
    """
    if query.ndim != 3 or key.ndim != 3 or key.shape != value.shape:
        raise ValueError("expected Q[Q,Hq,D] and matching K/V[K,Hkv,D]")
    nq, hq, dim = query.shape
    nk, hk, kd = key.shape
    if dim != kd or dim < 1 or hk < 1 or hq < 1 or hq % hk:
        raise ValueError("invalid attention dimension or GQA head ratio")
    if query.device != key.device or query.device != value.device:
        raise ValueError("attention tensors must share one device")
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        _finite(name, tensor)
    factor = dim ** -0.5 if scale is None else scale
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("attention scale must be finite and positive")
    if nk == 0:
        return AttentionResult(torch.zeros_like(query, dtype=torch.float32),
                               torch.full((nq, hq), -math.inf, device=query.device))
    head_map = torch.arange(hq, device=query.device) // (hq // hk)
    k = key.float().index_select(1, head_map)
    v = value.float().index_select(1, head_map)
    scores = torch.einsum("qhd,khd->qhk", query.float(), k) * factor
    _finite("attention scores before masking", scores)
    if causal:
        if query_positions is None:
            if nq > nk:
                raise ValueError("query positions required when Q exceeds K")
            query_positions = torch.arange(nk - nq, nk, device=query.device)
        if key_positions is None:
            key_positions = torch.arange(nk, device=query.device)
        if query_positions.shape != (nq,) or key_positions.shape != (nk,):
            raise ValueError("position vector shape mismatch")
        if query_positions.device != query.device or key_positions.device != query.device:
            raise ValueError("position vectors must use the attention device")
        scores = scores.masked_fill(key_positions[None, None, :] > query_positions[:, None, None], -math.inf)
    maximum = scores.amax(-1)
    live = torch.isfinite(maximum)
    safe_max = torch.where(live, maximum, torch.zeros_like(maximum))
    weights = torch.exp(scores - safe_max.unsqueeze(-1))
    denominator = weights.sum(-1)
    safe_den = torch.where(live, denominator, torch.ones_like(denominator))
    output = torch.einsum("qhk,khd->qhd", weights, v) / safe_den.unsqueeze(-1)
    lse = torch.where(live, safe_max + torch.log(safe_den), torch.full_like(maximum, -math.inf))
    return AttentionResult(output, lse)


def merge_attention(parts: list[AttentionResult] | tuple[AttentionResult, ...]) -> AttentionResult:
    """Merge normalized segment results using natural-log LSE, including empties."""
    if not parts:
        raise ValueError("at least one attention segment is required")
    shape, device = parts[0].output.shape, parts[0].output.device
    for part in parts:
        if part.output.shape != shape or part.lse.shape != shape[:-1]:
            raise ValueError("attention segment shapes differ")
        if part.output.device != device or part.lse.device != device:
            raise ValueError("attention segments must share one device")
        _finite("segment output", part.output)
        if bool((torch.isnan(part.lse) | torch.isposinf(part.lse)).any()):
            raise ValueError("segment LSE may be finite or -inf, never NaN/+inf")
    lse = torch.stack([part.lse.float() for part in parts])
    combined = torch.logsumexp(lse, dim=0)
    live = torch.isfinite(combined)
    safe = torch.where(live, combined, torch.zeros_like(combined))
    weights = torch.exp(lse - safe.unsqueeze(0))
    outputs = torch.stack([part.output.float() for part in parts])
    return AttentionResult((outputs * weights.unsqueeze(-1)).sum(0), combined)


def compressed_attention(query: torch.Tensor, slots: torch.Tensor,
                         key_rotation: torch.Tensor, value_rotation: torch.Tensor, *,
                         raw_key: torch.Tensor | None = None,
                         raw_value: torch.Tensor | None = None,
                         bf16_mask: torch.Tensor | None = None,
                         scale: float | None = None, causal: bool = True,
                         query_positions: torch.Tensor | None = None) -> AttentionResult:
    """Test-only full reconstruction oracle with explicit exact-window mask."""
    key, value = decode_kv(slots, query.shape[-1])
    key = key @ key_rotation.float().T
    value = value @ value_rotation.float().T
    if bf16_mask is not None:
        if raw_key is None or raw_value is None:
            raise ValueError("exact windows require raw K and V")
        if raw_key.shape != key.shape or raw_value.shape != value.shape:
            raise ValueError("raw window tensors must match decoded K/V")
        if bf16_mask.dtype != torch.bool or bf16_mask.shape != (key.shape[0],):
            raise ValueError("bf16_mask must be a bool vector over key tokens")
        mask = bf16_mask[:, None, None]
        key = torch.where(mask, raw_key.to(torch.bfloat16).float(), key)
        value = torch.where(mask, raw_value.to(torch.bfloat16).float(), value)
    elif raw_key is not None or raw_value is not None:
        raise ValueError("raw K/V require an explicit exact-window mask")
    return attention(query, key, value, scale=scale, causal=causal,
                     query_positions=query_positions)
