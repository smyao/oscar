"""Small official CPU-debug goldens for mixed q4 ownership and C16 sharing.

Archive #126/#129/#140/#143-151 and startup D.4: these offline PR-oracle
fixtures preserve bounded fused CV, exact windows, current-source suppression
and status; they do not prove target NPU speed, graph replay or full service.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import probe_history_reuse as reuse


ROOT = Path(__file__).resolve().parents[1]
TABLE_COLUMNS = 8  # Existing official CPU harness reads a fixed eight-column table.


def _packed_old_rows(torch, fixture: dict, request: int):
    spec: reuse.Shape = fixture["spec"]
    context, dim, hk = spec.contexts[request], spec.dim, spec.kv_heads
    slot_bytes = dim // 2 + 8
    raw = fixture["cpu"]["raw"][reuse.PREFIX:].view(
        fixture["blocks"], reuse.BLOCK_TOKENS, hk, slot_bytes)
    pages = fixture["page_assignments"][request]
    return torch.stack([raw[pages[position // reuse.BLOCK_TOKENS],
                            position % reuse.BLOCK_TOKENS]
                        for position in range(context)])


def _old_only_oracle(torch, fixture: dict, original: list[tuple]) -> tuple:
    """Exact sink/window plus INT2 history; source2 is empty in these cases."""
    from oscar_ascend.ops.reference import attention, decode_kv

    spec: reuse.Shape = fixture["spec"]
    n, h, d = fixture["tokens"], spec.heads, spec.dim
    output = torch.zeros((n, h, d), dtype=torch.float32)
    lse = torch.full((n, h), -float("inf"), dtype=torch.float32)
    rk = reuse._hadamard(torch, d)
    rv = fixture["cpu"]["rv"]
    start = 0
    for request, (qlen, context) in enumerate(zip(spec.qlens, spec.contexts)):
        old_k, old_v = original[request]
        if context:
            packed = _packed_old_rows(torch, fixture, request)
            rotated_k, rotated_v = decode_kv(packed, d)
            restored_k, restored_v = rotated_k @ rk.T, rotated_v @ rv.T
            chosen_k, chosen_v = old_k.float().clone(), old_v.float().clone()
            # For context65/sink64/recent1, position64 is source0 history
            # for every query. Retain general per-query boundaries below.
            for local in range(qlen):
                cut = min(context, max(reuse.SINK,
                                       context + local + 1 - spec.recent_tokens))
                selected_k, selected_v = chosen_k.clone(), chosen_v.clone()
                selected_k[reuse.SINK:cut] = restored_k[reuse.SINK:cut]
                selected_v[reuse.SINK:cut] = restored_v[reuse.SINK:cut]
                result = attention(fixture["cpu"]["q"][start + local:start + local + 1],
                                   selected_k, selected_v, scale=fixture["scale"],
                                   causal=False)
                output[start + local] = result.output[0]
                lse[start + local] = result.lse[0]
        start += qlen
    return output, lse


def _full_oracle(torch, fixture: dict) -> tuple:
    spec: reuse.Shape = fixture["spec"]
    n, h, d = fixture["tokens"], spec.heads, spec.dim
    output = torch.zeros((n, h, d), dtype=torch.float32)
    lse = torch.full((n, h), -float("inf"), dtype=torch.float32)
    for token, (row, row_lse) in fixture["expected"].items():
        output[token], lse[token] = row, row_lse
    if len(fixture["expected"]) != fixture["actual_tokens"]:
        raise ValueError("full q4 oracle omitted a live query")
    return output, lse


def _case(torch, name: str, spec: reuse.Shape, *, suppress_current: bool = False,
          invalid_scale: bool = False) -> dict:
    originals: list[tuple] = []

    def observe(request, _begin, _query, old_k, old_v, _ck, _cv):
        if request != len(originals):
            raise ValueError("request observer order changed")
        originals.append((old_k, old_v))

    fixture = reuse.make_fixture(torch, spec, on_request=observe)
    if fixture["tokens"] > 512:
        # The CPU harness must not expand these into a long performance matrix.
        raise ValueError("mixed CPU fixture exceeds bounded token count")
    if invalid_scale:
        if len(spec.qlens) != 1 or spec.contexts[0] <= reuse.SINK:
            raise ValueError("invalid C16 metadata fixture lacks live compressed history")
        physical = fixture["page_assignments"][0][reuse.SINK // reuse.BLOCK_TOKENS]
        slot_bytes = spec.dim // 2 + 8
        raw = fixture["cpu"]["raw"]
        offset = (reuse.PREFIX + physical * fixture["stride"] +
                  (reuse.SINK % reuse.BLOCK_TOKENS) * slot_bytes + spec.dim // 4)
        raw[offset:offset + 2] = torch.tensor((0, 0), dtype=torch.uint8)
    if suppress_current:
        output, lse = _old_only_oracle(torch, fixture, originals) if not invalid_scale else (
            torch.zeros((fixture["tokens"], spec.heads, spec.dim), dtype=torch.float32),
            torch.full((fixture["tokens"], spec.heads), -float("inf"), dtype=torch.float32))
    else:
        output, lse = _full_oracle(torch, fixture)
    if not invalid_scale and (not bool(torch.isfinite(output[:fixture["actual_tokens"]]).all()) or
                              (spec.contexts[0] > 0 and not bool(torch.isfinite(
                                  lse[:fixture["actual_tokens"]]).all()))):
        # Mixed unaligned starts with an empty old-context request, whose
        # source2-suppressed rows legitimately have -inf LSE.
        if not (suppress_current and len(spec.qlens) > 1 and spec.contexts[0] == 0):
            raise ValueError("valid CPU fixture produced non-finite live oracle")
    positions = torch.full((fixture["tokens"],), -1, dtype=torch.int64)
    offset = 0
    for context, qlen in zip(spec.contexts, spec.qlens):
        positions[offset:offset + qlen] = torch.arange(context, context + qlen)
        offset += qlen
    table = fixture["cpu"]["table"]
    if table.shape[1] > TABLE_COLUMNS:
        raise ValueError("official CPU fixture block table exceeds eight columns")
    padded_table = torch.full((len(spec.qlens), TABLE_COLUMNS), -1, dtype=torch.int32)
    padded_table[:, :table.shape[1]] = table
    tensors = {**fixture["cpu"], "table": padded_table,
               "expected_positions": positions, "expected_output": output,
               "expected_lse": lse}
    shape = (fixture["tokens"], spec.heads, spec.kv_heads, spec.dim,
             spec.contexts[0] if len(spec.qlens) == 1 else 0,
             reuse.SINK, spec.recent_tokens, reuse.SPECULATIVE,
             reuse.BLOCK_TOKENS, fixture["blocks"], reuse.PREFIX,
             fixture["stride"], 2 if suppress_current else 20,
             len(spec.qlens), spec.splits)
    return {"name": name, "shape": shape, "tensors": tensors,
            "suppress_current": suppress_current, "invalid_scale": invalid_scale}


def _write(torch, directory: Path, case: dict, mode: str) -> dict:
    path = directory / case["name"]
    path.mkdir(parents=True, exist_ok=True)
    for key, tensor in case["tensors"].items():
        (path / f"{key}.bin").write_bytes(bytes(tensor.contiguous().view(torch.uint8).flatten().tolist()))
    (path / "shape.txt").write_text(" ".join(map(str, case["shape"])) + "\n")
    return {"op": mode, "path": str(path)}


def export_mixed_cv_cpu_cases(directory: str | Path) -> Path:
    import torch

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cases = []
    contexts = tuple(range(160, 191))
    for splits in (1, 3):
        spec = reuse.Shape(f"balanced_q4_n128_s{splits}", (4,) * 31,
                           contexts, 64, 1, splits, False, padded_tokens=128,
                           recent_tokens=32)
        cases.append(_write(torch, directory,
                            _case(torch, spec.name, spec), "mixed_balanced"))
    for dim in (64, 256):
        spec = reuse.Shape(f"c16_d{dim}_q336_c65", (336,), (65,), dim, 1, 1,
                           True, recent_tokens=1)
        case = _case(torch, spec.name, spec, suppress_current=True)
        cases.append(_write(torch, directory, case, "mixed_cluster16"))
        if dim == 64:
            cases.append({"op": "mixed_cluster16_poison",
                          "path": str(directory / case["name"])})
    # D256's full active C16 CPU-debug simulation exceeds the fixed 120 s
    # bound (rc124); keep that recorded case intact. This short independent
    # C1 path covers the D256 template and workspace offsets only. Target NPU
    # must decide active D256 C16 precision and speed.
    d256_fallback = reuse.Shape("c16_d256_q4_c65", (4,), (65,), 256, 1, 1,
                                False, recent_tokens=1)
    cases.append(_write(torch, directory,
                        _case(torch, d256_fallback.name, d256_fallback,
                              suppress_current=True), "mixed_cluster16"))
    for name, qlens, contexts_ in (
            ("c16_d64_q335_c65", (335,), (65,)),
            ("c16_d64_unaligned_q3q336", (3, 336), (0, 65))):
        spec = reuse.Shape(name, qlens, contexts_, 64, 1, 1,
                           name.startswith("c16_d64_unaligned"), recent_tokens=1)
        cases.append(_write(torch, directory,
                            _case(torch, name, spec, suppress_current=True),
                            "mixed_cluster16"))
    bad = reuse.Shape("c16_d64_q336_bad_meta", (336,), (65,), 64, 1, 1,
                      True, recent_tokens=1)
    cases.append(_write(torch, directory,
                        _case(torch, bad.name, bad, suppress_current=True,
                              invalid_scale=True), "mixed_cluster16_error"))
    (directory / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(export_mixed_cv_cpu_cases(args.directory))
