"""Archive #13-22/#126/#154: CPU-only fixture conversion, never serving code.

The two encodings retain exactly the same two-bit integers and FP16 metadata
bytes. Tests compare the new device reader/writer with the independent original
PR oracle, using this reversible conversion only to arrange test inputs.
"""
from __future__ import annotations


FORMAT = "oscar-int2-striped-v1-d256"


def _check(slots):
    import torch
    if (slots.device.type != "cpu" or slots.dtype != torch.uint8 or
            slots.ndim < 1 or slots.shape[-1] != 136):
        raise ValueError("striped fixture requires CPU uint8[...,136] D256 slots")


def canonical_slots_to_striped(slots):
    import torch
    _check(slots)
    out = torch.empty_like(slots)
    for old_begin, new_begin in ((0, 0), (68, 64)):
        packed = slots[..., old_begin:old_begin + 64].to(torch.int32)
        codes = ((packed[..., None] >> (2 * torch.arange(4))) & 3).flatten(-2)
        # word i holds dimensions i, 32+i, ..., 224+i, low bits first.
        grouped = codes.reshape(*codes.shape[:-1], 8, 32).transpose(-2, -1)
        words = (grouped << (2 * torch.arange(8))).sum(-1)
        out[..., new_begin:new_begin + 64:2] = words & 255
        out[..., new_begin + 1:new_begin + 64:2] = words >> 8
    out[..., 128:132] = slots[..., 64:68]
    out[..., 132:136] = slots[..., 132:136]
    return out


def striped_slots_to_canonical(slots):
    import torch
    _check(slots)
    out = torch.empty_like(slots)
    for new_begin, old_begin in ((0, 0), (64, 68)):
        raw = slots[..., new_begin:new_begin + 64].to(torch.int32)
        words = raw[..., ::2] | (raw[..., 1::2] << 8)
        grouped = (words[..., None] >> (2 * torch.arange(8))) & 3
        codes = grouped.transpose(-2, -1).reshape(*slots.shape[:-1], 256)
        canonical = (codes.reshape(*slots.shape[:-1], 64, 4) <<
                     (2 * torch.arange(4))).sum(-1)
        out[..., old_begin:old_begin + 64] = canonical
    out[..., 64:68] = slots[..., 128:132]
    out[..., 132:136] = slots[..., 132:136]
    return out


def raw_to_striped(fixture: dict, raw=None):
    """Change only compressed payload bytes, preserving prefix/page padding."""
    from .probe_history_reuse import BLOCK_TOKENS, PREFIX
    spec = fixture["spec"]
    if spec.dim != 256:
        raise ValueError("striped-v1 is explicitly limited to D256")
    source = fixture["cpu"]["raw"] if raw is None else raw
    out = source.clone()
    page_bytes = BLOCK_TOKENS * spec.kv_heads * 136
    # Bounded test chunks avoid a second history-sized tensor of integer codes.
    for page in range(fixture["blocks"]):
        begin = PREFIX + page * fixture["stride"]
        slots = source[begin:begin + page_bytes].view(-1, 136)
        out[begin:begin + page_bytes] = canonical_slots_to_striped(slots).flatten()
    return out
