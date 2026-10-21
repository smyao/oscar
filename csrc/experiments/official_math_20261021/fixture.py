"""Archive #13-20/#126/#145/#148/#164: independent v4 functional fixtures.

D.4 fixed-size CPU fixtures only; production keeps bounded fused INT2 reads.
Writer inputs, fixed-forward cuts and all goldens are prepared independently
of candidate output. No timing, NPU/model acceptance or old-oracle rewrite.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import torch

from tools.official_reference import CACHE_FORMAT, pack_kv, rotate_bf16, fixture_tasks_reference


def dump(path, value):
    path.write_bytes(bytes(value.contiguous().view(torch.uint8).flatten().tolist()))


def hadamard(dim):
    h = torch.ones(1, 1)
    while h.shape[0] < dim:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return (h / dim**.5).bfloat16()


def make_decode(folder, family, fault=False):
    folder.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(20261021)
    d, active, hk, context = 256, 4 if family == "q4" else 1, 2, 330
    n, hq, sink, recent, spec, block, prefix, splits = active + 1, 12, 64, 8, 3, 128, 32, 3
    total = context + active + 1
    blocks = (total + block - 1) // block
    stride, ring = block * hk * 144, recent + spec
    rotation = hadamard(d)
    rk = rotate_bf16(torch.randn(total, hk, d).bfloat16(), rotation)
    rv = rotate_bf16(torch.randn(total, hk, d).bfloat16(), rotation.flip(0).contiguous())
    rk[-1] = float("nan"); rv[-1] = float("nan")
    q = rotate_bf16(torch.randn(n, hq, d).bfloat16(), rotation)
    q[-1] = float("nan")
    qr = q.clone(); qr[-1] = 0
    raw = torch.full((prefix + blocks * stride,), 0x85, dtype=torch.uint8)
    expected_raw = raw.clone()
    expected_raw[prefix:prefix + (total - 1) * hk * 144] = pack_kv(rk[:-1], rv[:-1]).flatten()
    shape = (blocks, sink + ring, hk, d)
    wk = torch.full((blocks * (sink + ring) * hk * d * 2,), 0x85, dtype=torch.uint8).view(torch.bfloat16).reshape(shape)
    wv = wk.clone(); tags = torch.full(shape[:2], -1, dtype=torch.int64)
    expected_wk, expected_wv, expected_tags = wk.clone(), wv.clone(), tags.clone()
    for position in range(total - 1):
        page, within = divmod(position, block)
        row = position if position < sink else sink + within % ring
        expected_wk[page, row] = rk[position]
        expected_wv[page, row] = rv[position]
        expected_tags[page, row] = within
    slots = torch.arange(context, context + n, dtype=torch.int64); slots[-1] = -1
    positions = slots.clone()
    store_slots = torch.arange(total, dtype=torch.int64); store_slots[-1] = -1
    data = {"q": q, "qr": qr, "ck": rk[context:].clone(), "cv": rv[context:].clone(),
            "rv": rotation.flip(0).contiguous(), "raw": expected_raw,
            "wk": expected_wk, "wv": expected_wv, "tags": expected_tags,
            "table": torch.arange(blocks, dtype=torch.int32)[None],
            "starts": torch.tensor([0, active], dtype=torch.int32),
            "lens": torch.tensor([context + active], dtype=torch.int32), "slots": slots}
    fixture = {"cpu": data, "spec": SimpleNamespace(kv_heads=hk, splits=splits, recent_tokens=recent, speculative_tokens=spec),
               "stride": stride, "blocks": blocks, "scale": d**-.5,
               "cache_format": CACHE_FORMAT, "cached_quant_ends": [max(min(sink, context), context + 1 - recent)]}
    torch.save(fixture, folder / "fixture.pt")
    files = {**data, "raw_striped": raw, "wk": wk, "wv": wv, "tags": tags,
             "qr_bf16": q, "store_rk": rk, "store_rv": rv, "store_k": rk, "store_v": rv,
             "store_slots": store_slots, "store_positions": torch.arange(total),
             "store_expected_raw": expected_raw, "store_expected_wk": expected_wk,
             "store_expected_wv": expected_wv, "store_expected_tags": expected_tags,
             "expected_positions": positions,
             "required_status_checks": torch.tensor([[0, 0, 3]], dtype=torch.int64) if fault else torch.empty((0, 3), dtype=torch.int64)}
    for name, value in files.items():
        dump(folder / (name + ".bin"), value)
    (folder / "shape.txt").write_text(" ".join(map(str, (n, hq, hk, d, context, sink, recent, spec, block, blocks, prefix, stride, 2, 1, splits, blocks))) + "\n")
    (folder / "case_flags.txt").write_text(str(int(fault)) + "\n")
    (folder / "store_tokens.txt").write_text(str(total) + "\n")


def make_cluster(folder, family, fault=False, empty_history=False):
    from tools import probe_history_reuse as reuse
    from tools.aligned_fixture import make_fixture
    folder.mkdir(parents=True, exist_ok=False)
    active = 85 if family == "c4" else 337
    hk = 2 if family == "c4" else 1
    recent = 256 if empty_history else 16
    shape = reuse.Shape("official_" + family, (active,), (96,), 256, hk, 1, True,
                        recent_tokens=recent, padded_tokens=active + 1,
                        corrupt_metadata=fault, corrupt_position=80)
    fixture = make_fixture(torch, shape)
    data = fixture["cpu"]
    rk, rv = fixture["rotation_k"], data["rv"]
    data["q"] = data["qr"].clone()
    for name, matrix in (("ck", rk), ("cv", rv)):
        data[name] = rotate_bf16(data[name], matrix)
    valid = data["tags"] >= 0
    for name, matrix in (("wk", rk), ("wv", rv)):
        data[name][valid] = rotate_bf16(data[name][valid], matrix)
    fixture.update(cache_format=CACHE_FORMAT, cached_quant_ends=[max(64, 97 - recent)])
    positions = torch.full((active + 1,), -1, dtype=torch.int64)
    positions[:active] = torch.arange(96, 96 + active)
    checks = [((qb * hk + hk - 1) * 3, 1, 3) for qb in range(0, active, 21)] if fault else []
    data["expected_positions"] = positions
    data["required_status_checks"] = torch.tensor(checks, dtype=torch.int64).reshape(-1, 3)
    torch.save(fixture, folder / "fixture.pt")
    for name, value in data.items():
        dump(folder / (name + ".bin"), value)
    dump(folder / "raw_striped.bin", data["raw"])
    dims = (active + 1, 6 * hk, hk, 256, 96, 64, recent, 3, reuse.BLOCK_TOKENS,
            fixture["blocks"], reuse.PREFIX, fixture["stride"], 2, 1, 1, data["table"].shape[1])
    (folder / "shape.txt").write_text(" ".join(map(str, dims)) + "\n")
    (folder / "case_flags.txt").write_text(str(int(fault)) + "\n")
    (folder / "cluster_flags.txt").write_text(f"0 {hk} {80 if fault else -1} -1\n")


def golden(folder, output, mode):
    fixture = torch.load(folder / "fixture.pt", weights_only=False)
    tasks = torch.frombuffer(bytearray((output / "tasks.bin").read_bytes()), dtype=torch.int64).reshape(-1, 16).clone()
    if (folder / "case_flags.txt").read_text().strip() == "0":
        results = fixture_tasks_reference(fixture, tasks, mode)
        for key, field in (("partial", "partial"), ("lse", "lse"), ("output", "merged_rotated"), ("output_lse", "output_lse")):
            dump(output / ("official_expected_" + key + ".bin"), results[field])
        dump(output / "official_final_output.bin", results["final_output"])
    dump(output / "rv_bf16.bin", fixture["cpu"]["rv"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("make", "golden"))
    parser.add_argument("family", choices=("q4", "q1", "c4", "c16"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--fault", action="store_true")
    parser.add_argument("--empty-history", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", type=int, default=0)
    args = parser.parse_args(); torch.set_num_threads(1)
    if args.action == "make":
        if args.family in ("q4", "q1"):
            if args.empty_history: raise ValueError("empty-history boundary uses a mature cluster")
            make_decode(args.folder, args.family, args.fault)
        else:
            make_cluster(args.folder, args.family, args.fault, args.empty_history)
    else:
        golden(args.folder, args.output, args.mode)
