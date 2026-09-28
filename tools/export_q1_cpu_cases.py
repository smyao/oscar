"""Export bounded q1 CV scheduling cases for the official AscendC CPU debugger.

Archive #126/#129/#148-150 and startup D.4: fe0 INT2 history, exact
sink/window/current sources and FP32 attention remain the numerical contract.
These CPU goldens exercise ownership and padding; they are not NPU speed or
model-quality evidence. Only this offline oracle materializes old BF16 KV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SINK = 4
RECENT = 32
SPECULATIVE = 3
BLOCK_TOKENS = 512
PREFIX = 64
SEED = 47


def _rotations(torch, dim: int):
    generator = torch.Generator().manual_seed(1900 + dim)
    key = torch.linalg.qr(torch.randn(dim, dim, generator=generator)).Q.contiguous()
    value = torch.linalg.qr(torch.randn(dim, dim, generator=generator)).Q.contiguous()
    return key, value


def _case(torch, *, name: str, contexts: tuple[int, ...], padded_tokens: int,
          dim: int, splits: int, cores: int) -> dict:
    from oscar_ascend.ops.reference import attention, decode_kv, encode_kv

    requests = len(contexts)
    if (not requests or requests > padded_tokens or len(set(contexts)) != requests or
            any(type(context) is not int or context < 0 or context >= BLOCK_TOKENS for context in contexts) or
            dim not in (64, 128, 256) or not 1 <= splits <= 32 or not 1 <= cores <= 32):
        raise ValueError(f"invalid q1 CPU fixture geometry: {name}")

    heads = 6
    kv_heads = 1
    slot_bytes = dim // 2 + 8
    stride = BLOCK_TOKENS * slot_bytes * kv_heads
    blocks = requests * 2
    row_count = SINK + RECENT + SPECULATIVE
    raw = torch.full((PREFIX + blocks * stride,), 0xA5, dtype=torch.uint8)
    raw_pages = raw[PREFIX:].view(blocks, BLOCK_TOKENS, kv_heads, slot_bytes)
    wk = torch.full((blocks, row_count, kv_heads, dim), float("nan"), dtype=torch.bfloat16)
    wv = torch.full_like(wk, float("nan"))
    tags = torch.full((blocks, row_count), -1, dtype=torch.int64)
    table = torch.empty((requests, 8), dtype=torch.int32)
    slots = torch.full((padded_tokens,), -1, dtype=torch.int64)
    expected_positions = torch.full((padded_tokens,), -1, dtype=torch.int64)
    q = torch.zeros((padded_tokens, heads, dim), dtype=torch.bfloat16)
    ck = torch.zeros((padded_tokens, kv_heads, dim), dtype=torch.bfloat16)
    cv = torch.zeros_like(ck)
    qr = torch.zeros((padded_tokens, heads, dim), dtype=torch.float32)
    expected_output = torch.zeros((padded_tokens, heads, dim), dtype=torch.float32)
    expected_lse = torch.full((padded_tokens, heads), -float("inf"), dtype=torch.float32)
    rk, rv = _rotations(torch, dim)

    for request, context in enumerate(contexts):
        generator = torch.Generator().manual_seed(SEED + dim + context)
        query = torch.randn((1, heads, dim), generator=generator).to(torch.bfloat16)
        old_k = torch.randn((context, kv_heads, dim), generator=generator).to(torch.bfloat16)
        old_v = torch.randn((context, kv_heads, dim), generator=generator).to(torch.bfloat16)
        current_k = torch.randn((1, kv_heads, dim), generator=generator).to(torch.bfloat16)
        current_v = torch.randn((1, kv_heads, dim), generator=generator).to(torch.bfloat16)
        physical = 2 * request + 1  # One permuted page plus one reserved page per request.
        table[request, :4] = physical * 4 + torch.arange(4, dtype=torch.int32)
        table[request, 4:] = (physical - 1) * 4 + torch.arange(4, dtype=torch.int32)
        if context:
            packed = encode_kv(old_k.float() @ rk, old_v.float() @ rv)
            restored_k, restored_v = decode_kv(packed, dim)
            raw_pages[physical, :context] = packed
            restored_k, restored_v = restored_k @ rk.T, restored_v @ rv.T
        else:
            restored_k, restored_v = old_k.float(), old_v.float()
        for position in range(context):
            row = position if position < SINK else SINK + position % (RECENT + SPECULATIVE)
            wk[physical, row] = old_k[position]
            wv[physical, row] = old_v[position]
            tags[physical, row] = position

        cut = context + 1 - RECENT
        old_positions = torch.arange(context)
        history = (old_positions >= SINK) & (old_positions < cut)
        selected_k = torch.where(history[:, None, None], restored_k, old_k.float())
        selected_v = torch.where(history[:, None, None], restored_v, old_v.float())
        keys = torch.cat((selected_k, current_k.float()))
        values = torch.cat((selected_v, current_v.float()))
        result = attention(query, keys, values)

        q[request] = query[0]
        qr[request] = query.float()[0] @ rk
        ck[request] = current_k[0]
        cv[request] = current_v[0]
        slots[request] = physical * BLOCK_TOKENS + context
        expected_positions[request] = context
        expected_output[request] = result.output[0]
        expected_lse[request] = result.lse[0]

    tensors = {"q": q, "qr": qr, "ck": ck, "cv": cv, "rv": rv, "raw": raw,
               "table": table, "wk": wk, "wv": wv, "tags": tags,
               "starts": torch.arange(requests + 1, dtype=torch.int32),
               "lens": torch.tensor([context + 1 for context in contexts], dtype=torch.int32),
               "slots": slots, "expected_positions": expected_positions,
               "expected_output": expected_output, "expected_lse": expected_lse}
    shape = (padded_tokens, heads, kv_heads, dim, 0, SINK, RECENT, SPECULATIVE,
             BLOCK_TOKENS, blocks, PREFIX, stride, cores, requests, splits)
    return {"name": name, "tensors": tensors, "shape": shape,
            "contexts": contexts, "physical_pages": tuple(2 * request + 1 for request in range(requests))}


def _bytes(torch, tensor) -> bytes:
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise ValueError("CPU golden must be contiguous and resident on CPU")
    # The official CPU-debug exporter uses raw uint8 storage. The configured
    # local development venv has no NumPy, so keep this bounded path portable.
    return bytes(tensor.view(torch.uint8).flatten().tolist())


def export_q1_cpu_cases(directory: str | Path) -> Path:
    """Write four small numerical cases plus a live-history metadata fault."""
    import torch

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    long_contexts = tuple(range(80, 112))
    definitions = (
        ("q1_32_s3_d64", long_contexts, 128, 64, 3, 20),
        ("q1_32_s1_d64", long_contexts, 128, 64, 1, 20),
        ("q1_32_empty_s3_d64", (0,) + tuple(range(80, 111)), 128, 64, 3, 20),
        ("q1_4_s1_d256", (80, 81, 82, 83), 8, 256, 1, 2),
    )
    cases: list[dict[str, str]] = []
    for name, contexts, padded, dim, splits, cores in definitions:
        fixture = _case(torch, name=name, contexts=contexts, padded_tokens=padded,
                        dim=dim, splits=splits, cores=cores)
        case = directory / name
        case.mkdir(parents=True, exist_ok=True)
        for key, value in fixture["tensors"].items():
            (case / f"{key}.bin").write_bytes(_bytes(torch, value))
        (case / "shape.txt").write_text(" ".join(map(str, fixture["shape"])) + "\n")
        cases.append({"op": "q1_schedule", "path": str(case)})
        if name == "q1_32_s3_d64":
            # Request 0, logical old position 20 belongs to the live INT2
            # history (sink=4, recent=32, context=80). Zero both K FP16 scale
            # bytes in the harness; fe0 and q1 must publish the same error.
            offset = PREFIX + fixture["physical_pages"][0] * fixture["shape"][11]
            offset += 20 * (dim // 2 + 8) + dim // 4
            (case / "bad_meta_offset.txt").write_text(f"{offset}\n")
            cases.append({"op": "q1_schedule_bad_meta", "path": str(case)})
    (directory / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(export_q1_cpu_cases(args.directory))
