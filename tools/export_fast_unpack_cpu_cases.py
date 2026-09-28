"""Offline exact INT2 fast-unpack cases for the official AscendC CPU debugger.

Archive #126/#129/#143/#148-151 and startup D.4: this exporter exercises
bounded history tiles and FP16 metadata boundaries against the independent
PR oracle. It never supplies production KV or target NPU performance evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BLOCK_TOKENS = 512
SINK = 4
RECENT = 32
SPECULATIVE = 3
PREFIX = 64
LIVE_POSITION = 100
DEAD_POSITION = 520


def _single_case(torch, dim: int, qlen: int, context: int):
    """Mirror test_cv_contracts._case without importing a test into this tool."""
    from oscar_ascend.ops.reference import attention, encode_kv, decode_kv

    if dim not in (64, 128, 256) or qlen <= 0 or not 0 <= context < 2 * BLOCK_TOKENS:
        raise ValueError("invalid fast-unpack CPU case")
    generator = torch.Generator().manual_seed(47 + dim + context)
    heads = 6
    kv_heads = 1
    query = torch.randn((qlen, heads, dim), generator=generator).to(torch.bfloat16)
    old_key = torch.randn((context, kv_heads, dim), generator=generator).to(torch.bfloat16)
    old_value = torch.randn((context, kv_heads, dim), generator=generator).to(torch.bfloat16)
    current_key = torch.randn((qlen, kv_heads, dim), generator=generator).to(torch.bfloat16)
    current_value = torch.randn((qlen, kv_heads, dim), generator=generator).to(torch.bfloat16)
    rotation_generator = torch.Generator().manual_seed(1900 + dim)
    rk = torch.linalg.qr(torch.randn(dim, dim, generator=rotation_generator)).Q.contiguous()
    rv = torch.linalg.qr(torch.randn(dim, dim, generator=rotation_generator)).Q.contiguous()
    packed = encode_kv(old_key.float() @ rk, old_value.float() @ rv)
    slot_bytes = dim // 2 + 8
    stride = BLOCK_TOKENS * slot_bytes
    raw = torch.full((PREFIX + 2 * stride,), 0xA5, dtype=torch.uint8)
    window_rows = SINK + RECENT + SPECULATIVE
    window_key = torch.full((2, window_rows, kv_heads, dim), float("nan"), dtype=torch.bfloat16)
    window_value = torch.full_like(window_key, float("nan"))
    tags = torch.full((2, window_rows), -1, dtype=torch.int64)
    table = torch.tensor([[4, 5, 6, 7, 0, 1, 2, 3]], dtype=torch.int32)
    for position in range(context):
        physical, inpage = 1 - position // BLOCK_TOKENS, position % BLOCK_TOKENS
        offset = PREFIX + physical * stride + inpage * slot_bytes
        raw[offset:offset + slot_bytes] = packed[position, 0]
        row = position if position < SINK else SINK + inpage % (RECENT + SPECULATIVE)
        window_key[physical, row] = old_key[position]
        window_value[physical, row] = old_value[position]
        tags[physical, row] = inpage
    tensors = {"q": query, "qr": query.float() @ rk, "ck": current_key,
               "cv": current_value, "rv": rv, "raw": raw, "table": table,
               "wk": window_key, "wv": window_value, "tags": tags,
               "starts": torch.tensor((0, qlen), dtype=torch.int32),
               "lens": torch.tensor((context + qlen,), dtype=torch.int32),
               "slots": torch.arange(context, context + qlen, dtype=torch.int64)}
    expected_output, expected_lse = _recompute_oracle(
        torch, tensors, old_key, old_value, rk, rv, qlen, context, validate_metadata=True)
    return {"tensors": tensors, "old_key": old_key, "old_value": old_value,
            "rk": rk, "rv": rv, "expected_output": expected_output,
            "expected_lse": expected_lse, "dim": dim, "qlen": qlen,
            "context": context, "stride": stride}


def _old_packed_rows(torch, tensors: dict, context: int, dim: int):
    slot_bytes = dim // 2 + 8
    stride = BLOCK_TOKENS * slot_bytes
    raw = tensors["raw"]
    rows = torch.empty((context, 1, slot_bytes), dtype=torch.uint8)
    for position in range(context):
        physical, inpage = 1 - position // BLOCK_TOKENS, position % BLOCK_TOKENS
        offset = PREFIX + physical * stride + inpage * slot_bytes
        rows[position, 0] = raw[offset:offset + slot_bytes]
    return rows


def _recompute_oracle(torch, tensors: dict, old_key, old_value, rk, rv,
                      qlen: int, context: int, *, validate_metadata: bool):
    from oscar_ascend.ops.reference import attention, decode_kv

    dim = tensors["q"].shape[-1]
    if context:
        packed = _old_packed_rows(torch, tensors, context, dim)
        rotated_key, rotated_value = decode_kv(packed, dim)
        restored_key = rotated_key @ rk.T
        restored_value = rotated_value @ rv.T
    else:
        restored_key, restored_value = old_key.float(), old_value.float()
    expected_output, expected_lse = [], []
    for token in range(qlen):
        old_positions = torch.arange(context)
        cutoff = min(context, max(SINK, context + token + 1 - RECENT))
        history = (old_positions >= SINK) & (old_positions < cutoff)
        selected_key = torch.where(history[:, None, None], restored_key, old_key.float())
        selected_value = torch.where(history[:, None, None], restored_value, old_value.float())
        keys = torch.cat((selected_key, tensors["ck"][:token + 1].float()))
        values = torch.cat((selected_value, tensors["cv"][:token + 1].float()))
        result = attention(tensors["q"][token:token + 1], keys, values,
                           scale=dim ** -0.5, causal=False)
        expected_output.append(result.output)
        expected_lse.append(result.lse)
    result_output = torch.cat(expected_output)
    result_lse = torch.cat(expected_lse)
    if validate_metadata and (not bool(torch.isfinite(result_output).all()) or
                              not bool(torch.isfinite(result_lse).all())):
        raise ValueError("legal fast-unpack metadata produced non-finite oracle")
    return result_output, result_lse


def _metadata_offset(dim: int, position: int, *, value: bool, zero: bool) -> int:
    if not 0 <= position < 2 * BLOCK_TOKENS:
        raise ValueError("metadata position is outside two physical pages")
    slot_bytes = dim // 2 + 8
    stride = BLOCK_TOKENS * slot_bytes
    physical, inpage = 1 - position // BLOCK_TOKENS, position % BLOCK_TOKENS
    return (PREFIX + physical * stride + inpage * slot_bytes +
            (dim // 4 + 4 if value else 0) + dim // 4 + (2 if zero else 0))


def _set_half_bits(tensors: dict, dim: int, position: int, *, value: bool,
                   zero: bool, bits: int) -> None:
    if not 0 <= bits <= 0xFFFF:
        raise ValueError("FP16 metadata bits are out of range")
    offset = _metadata_offset(dim, position, value=value, zero=zero)
    raw = tensors["raw"]
    raw[offset:offset + 2] = raw.new_tensor((bits & 255, bits >> 8))


def _write_case(torch, directory: Path, name: str, fixture: dict, *, mode: str,
                recompute: bool = False) -> dict:
    case = directory / name
    case.mkdir(parents=True, exist_ok=True)
    tensors = fixture["tensors"]
    if recompute:
        fixture["expected_output"], fixture["expected_lse"] = _recompute_oracle(
            torch, tensors, fixture["old_key"], fixture["old_value"], fixture["rk"],
            fixture["rv"], fixture["qlen"], fixture["context"], validate_metadata=True)
    for key, tensor in {**tensors, "expected_output": fixture["expected_output"],
                        "expected_lse": fixture["expected_lse"]}.items():
        (case / f"{key}.bin").write_bytes(bytes(tensor.contiguous().view(torch.uint8).flatten().tolist()))
    dim, qlen, context = fixture["dim"], fixture["qlen"], fixture["context"]
    # Existing official CPU harness shape: n hq hk D context sink recent spec
    # blockTokens blocks prefix stride cores requests splits.
    (case / "shape.txt").write_text(
        f"{qlen} 6 1 {dim} {context} {SINK} {RECENT} {SPECULATIVE} "
        f"{BLOCK_TOKENS} 2 {PREFIX} {fixture['stride']} 2 1 1\n")
    return {"op": mode, "path": str(case)}


def export_fast_unpack_cpu_cases(directory: str | Path) -> Path:
    import torch
    from .export_q1_cpu_cases import export_q1_cpu_cases

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cases = []
    for dim in (64, 128, 256):
        cases.append(_write_case(torch, directory, f"d{dim}_q4_c511", _single_case(
            torch, dim, 4, 511), mode="fast_fe0"))

    # One valid case carries all requested FP16 edge values at distinct live
    # history rows. An invalid bit pattern cannot be mixed with this oracle.
    legal = _single_case(torch, 64, 4, 511)
    for position, value, zero, bits in (
            (100, False, False, 0x0001),  # smallest positive subnormal K scale
            (101, True, False, 0x03FF),   # largest positive subnormal V scale
            (102, False, False, 0x0400), # smallest normal K scale
            (103, False, True, 0x8000),  # K zero=-0
            (104, True, True, 0x0000),   # V zero=+0
            (105, False, True, 0x0001)): # K zero=positive subnormal
        _set_half_bits(legal["tensors"], 64, position, value=value, zero=zero, bits=bits)
    cases.append(_write_case(torch, directory, "d64_q4_c511_legal_fp16_edges",
                             legal, mode="fast_fe0", recompute=True))

    for label, value, zero, bits in (
            ("scale_pos_zero", False, False, 0x0000),
            ("scale_neg_zero", False, False, 0x8000),
            ("scale_pos_inf", True, False, 0x7C00),
            ("scale_neg_inf", True, False, 0xFC00),
            ("scale_nan", False, False, 0x7E00),
            ("zero_pos_inf", False, True, 0x7C00),
            ("zero_neg_inf", True, True, 0xFC00),
            ("zero_nan", True, True, 0x7E00)):
        invalid = _single_case(torch, 64, 4, 511)
        _set_half_bits(invalid["tensors"], 64, LIVE_POSITION,
                       value=value, zero=zero, bits=bits)
        cases.append(_write_case(torch, directory, f"d64_q4_c511_{label}",
                                 invalid, mode="fast_fe0_error"))

    dead = _single_case(torch, 64, 4, 511)
    _set_half_bits(dead["tensors"], 64, DEAD_POSITION,
                   value=False, zero=False, bits=0x7E00)
    cases.append(_write_case(torch, directory, "d64_q4_c511_dead_tail_nan",
                             dead, mode="fast_fe0_dead", recompute=True))

    q1_dir = export_q1_cpu_cases(directory / "q1")
    q1_cases = json.loads((q1_dir / "cases.json").read_text())
    for item in q1_cases:
        if item["op"] == "q1_schedule" and ("q1_32_s3_d64" in item["path"] or
                                             "q1_32_s1_d64" in item["path"] or
                                             "q1_32_empty_s3_d64" in item["path"]):
            cases.append({"op": "fast_q1", "path": item["path"]})

    mature = _write_case(torch, directory, "d64_q168_c641",
                         _single_case(torch, 64, 168, 641), mode="fast_cluster4")
    cases.append(mature)
    cases.append({"op": "fast_cluster4_poison", "path": mature["path"]})
    # Positions 400..405 are live source0 history for mature groups after
    # q=21 and are read by C4's shared tile. The S1 CPU fixture exercises
    # grouped ownership; the target S>1 terminal-split case remains a distinct
    # real-NPU gate.
    mature_legal = _single_case(torch, 64, 168, 641)
    for position, value, zero, bits in (
            (400, False, False, 0x0001), (401, True, False, 0x03FF),
            (402, False, False, 0x0400), (403, False, True, 0x8000),
            (404, True, True, 0x0000), (405, False, True, 0x0001)):
        _set_half_bits(mature_legal["tensors"], 64, position,
                       value=value, zero=zero, bits=bits)
    cases.append(_write_case(torch, directory, "d64_q168_c641_legal_fp16_edges",
                             mature_legal, mode="fast_cluster4", recompute=True))
    mature_invalid = _single_case(torch, 64, 168, 641)
    _set_half_bits(mature_invalid["tensors"], 64, 400,
                   value=False, zero=False, bits=0x7E00)
    _set_half_bits(mature_invalid["tensors"], 64, 401,
                   value=True, zero=True, bits=0x7C00)
    cases.append(_write_case(torch, directory, "d64_q168_c641_invalid_live_scale",
                             mature_invalid, mode="fast_cluster4_error"))
    cases.append(_write_case(torch, directory, "d64_q4_c65",
                             _single_case(torch, 64, 4, 65), mode="fast_cluster4"))
    (directory / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(export_fast_unpack_cpu_cases(args.directory))
