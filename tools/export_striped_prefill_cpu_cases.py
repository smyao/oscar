"""Synthetic D256 striped reader goldens for official AscendC CPU-debug.

Archive #126/#129/#145/#148-154; startup D.4. The frozen independent
reference is sampled on long prefill queries, while old/new CV partial, LSE,
status and cluster counters are compared over every task. This never uses a
user dataset, serving route, native source edit, or target NPU.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path

from . import probe_history_reuse as reuse


def _bytes(tensor) -> bytes:
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise ValueError("golden tensor must be contiguous CPU storage")
    return ctypes.string_at(tensor.data_ptr(), tensor.numel() * tensor.element_size())


def _write_case(torch, directory: Path, spec: reuse.Shape, mode: str) -> dict:
    fixture = reuse.make_fixture(torch, spec)
    n, actual, hq, d = fixture["tokens"], fixture["actual_tokens"], spec.heads, spec.dim
    if d != 256 or spec.kv_heads != 1:
        raise ValueError("striped-v1 CPU golden requires D256/Hkv1")
    table = fixture["cpu"]["table"]
    columns = max(8, int(table.shape[1]))
    padded = torch.full((len(spec.qlens), columns), -1, dtype=torch.int32)
    padded[:, :table.shape[1]] = table
    output = torch.zeros((n, hq, d), dtype=torch.float32)
    lse = torch.full((n, hq), -float("inf"), dtype=torch.float32)
    samples = tuple(sorted(fixture["expected"]))
    if not samples or not all(0 <= token < actual for token in samples):
        raise ValueError("independent oracle omitted all live queries")
    for token in samples:
        output[token], lse[token] = fixture["expected"][token]
    positions = torch.full((n,), -1, dtype=torch.int64)
    offset = 0
    for qlen, context in zip(spec.qlens, spec.contexts):
        positions[offset:offset+qlen] = torch.arange(context, context+qlen)
        offset += qlen
    tensors = {**fixture["cpu"], "table": padded,
               "expected_output": output, "expected_lse": lse,
               "expected_positions": positions,
               "oracle_samples": torch.tensor(samples, dtype=torch.int64)}
    shape = (n, hq, spec.kv_heads, d, spec.contexts[0] if len(spec.contexts)==1 else 0,
             reuse.SINK, spec.recent_tokens, reuse.SPECULATIVE,
             reuse.BLOCK_TOKENS, fixture["blocks"], reuse.PREFIX, fixture["stride"],
             20, len(spec.qlens), spec.splits, columns)
    path = directory / spec.name
    path.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    for key, tensor in tensors.items():
        contents = _bytes(tensor)
        (path / f"{key}.bin").write_bytes(contents)
        if key not in ("expected_output", "expected_lse", "oracle_samples",
                       "expected_positions"):
            digest.update(key.encode()+b"\0"+contents)
    (path / "shape.txt").write_text(" ".join(map(str, shape))+"\n")
    (path / "input_sha256.txt").write_text(digest.hexdigest()+"\n")
    return {"name": spec.name, "mode": mode, "path": str(path),
            "actual_tokens": actual, "sampled_oracle_tokens": len(samples),
            "input_sha256": digest.hexdigest()}


def export(directory: Path) -> Path:
    import torch
    torch.set_num_threads(min(4, torch.get_num_threads()))
    directory.mkdir(parents=True, exist_ok=True)
    cases = (
        (reuse.Shape("q1_d256_c511_r256", (1,), (511,), 256, 1, 1, False,
                     padded_tokens=8, slot_context=True, recent_tokens=256), "q1"),
        (reuse.Shape("base_d256_q3q33", (3,33), (511,1025),256,1,3,False), "base"),
        (reuse.Shape("balanced_d256_q4_n128_s3", (4,)*4,(127,511,641,1025),
                     256,1,3,False,padded_tokens=128), "balanced"),
        (reuse.Shape("c4_d256_q168_c65_active", (168,),(65,),256,1,1,True,
                     recent_tokens=1), "cluster4"),
        (reuse.Shape("c4_d256_q385_c641", (385,),(641,),256,1,1,True,
                     recent_tokens=32), "cluster4"),
        (reuse.Shape("c4_d256_unaligned_q3q385", (3,385),(641,1025),
                     256,1,3,True,recent_tokens=32), "cluster4"),
        (reuse.Shape("c16_d256_q1024_c641", (1024,),(641,),256,1,1,True,
                     recent_tokens=32), "cluster16"),
        (reuse.Shape("c16_d256_q336_c65_active", (336,),(65,),256,1,1,True,
                     recent_tokens=1), "cluster16"),
    )
    rows = [_write_case(torch,directory,spec,mode) for spec,mode in cases]
    (directory/"cases.json").write_text(json.dumps(rows,indent=2)+"\n")
    return directory


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    print(export(parser.parse_args().directory))
