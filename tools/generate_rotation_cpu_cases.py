# Archive G27/#13-22/#37-49/#111: independent PR goldens for the identical
# AscendC rotate/clip/store kernel running in CANN CPU-debug, never production.
"""Generate BF16/dense/Hadamard/clip/tail cases for CANN's CPU debugger."""
import argparse
import json
from pathlib import Path

import torch

from oscar_ascend.ops.reference import encode_kv, rotate_clip
from oscar_ascend.rotations import build_hadamard_artifact


def _write(path, tensor):
    path.write_bytes(bytes(tensor.contiguous().view(torch.uint8).flatten().tolist()))


def generate(root):
    root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(33846774)
    cases = []
    for dim in (64, 128, 256):
        for hadamard in (False, True):
            if hadamard:
                artifact = build_hadamard_artifact(layer_names=["model.layers.3.self_attn.attn"],
                    head_dim=dim, model_fingerprint="cpu-debug-fixture", device="cpu", testing=True)
                rk = artifact["layers"]["model.layers.3.self_attn.attn"]["Rk"]
                rv = artifact["layers"]["model.layers.3.self_attn.attn"]["Rv"]
            else:
                rk = torch.linalg.qr(torch.randn(dim, dim, generator=generator)).Q
                rv = torch.linalg.qr(torch.randn(dim, dim, generator=generator)).Q
            n, heads, block, blocks, sink, recent = 13, 2, 128, 3, 2, 4
            key = torch.randn(n, heads, dim, generator=generator).bfloat16()
            value = torch.randn(n, heads, dim, generator=generator).bfloat16()
            rotate_dir = root / f"rotate_{dim}_{int(hadamard)}"
            rotate_dir.mkdir(exist_ok=True)
            for name, tensor in (("input", key), ("rotation", rk.T),
                                 ("expected_output", rotate_clip(key, rk)),
                                 ("expected_status", torch.zeros(n, heads, dtype=torch.int32))):
                _write(rotate_dir / f"{name}.bin", tensor)
            (rotate_dir / "shape.txt").write_text(f"{n*heads} {dim} 2 {int(hadamard)} {heads} 0\n")
            cases.append({"op": "rotate", "path": str(rotate_dir)})
            masked_dir = root / f"rotate_masked_{dim}_{int(hadamard)}"
            masked_dir.mkdir(exist_ok=True)
            mask = torch.tensor([0, 1, -1, 3, -1, -1, -1, -1, 8, -1, 10, -1, -1], dtype=torch.int64)
            masked_input = key.clone()
            masked_input[mask < 0] = torch.nan
            masked_input[10, 0] = torch.nan  # Active non-finite input MUST retain status 2.
            # Independent PR matmul equation, retaining deliberate NaNs;
            # the oracle's finite-domain guard is intentionally inapplicable
            # only to this status-error test. Negative slots have exact zero.
            masked_expected = masked_input.float() @ rk
            masked_expected[mask < 0] = 0
            masked_status = torch.zeros(n, heads, dtype=torch.int32)
            masked_status[10, 0] = 2
            for name, tensor in (("input", masked_input), ("rotation", rk.T), ("slots", mask),
                                 ("expected_output", masked_expected), ("expected_status", masked_status)):
                _write(masked_dir / f"{name}.bin", tensor)
            (masked_dir / "shape.txt").write_text(f"{n*heads} {dim} 2 {int(hadamard)} {heads} 1\n")
            cases.append({"op": "rotate", "path": str(masked_dir)})
            for clip in (0.0, 0.95):
                path = root / f"rotate_store_{dim}_{int(hadamard)}_{clip:.2f}"
                path.mkdir(exist_ok=True)
                slots = torch.tensor([-1, 0, 1, 2, 3, 4, 5, 6, 254, 255, 256, 257, 384], dtype=torch.int64)
                positions = torch.tensor([-1, 0, 1, 2, 3, 4, 5, 6, 126, 127, 128, 129, 384], dtype=torch.int64)
                offset = blocks * 256
                head_bytes = dim // 2 + 8
                stride = block * heads * head_bytes + 256
                window = sink + recent
                raw_stride = window * heads * dim + 32  # BF16 elements, guard gap.
                tag_stride = window + 4
                raw = torch.full((offset + blocks * stride,), 173, dtype=torch.uint8)
                raw_k = torch.full((blocks * raw_stride * 2,), 173, dtype=torch.uint8)
                raw_v = torch.full_like(raw_k, 173)
                tags = torch.full((blocks * tag_stride * 8,), 173, dtype=torch.uint8)
                status = torch.zeros(n, heads, dtype=torch.int32)
                packed = encode_kv(rotate_clip(key, rk, clip), rotate_clip(value, rv, clip))
                # Sequential oracle naturally gives last-write-wins recent
                # state. The parallel kernel must derive unique writers.
                for i, slot in enumerate(slots.tolist()):
                    if slot < 0:
                        continue
                    if slot >= blocks * block:
                        status[i] = 1
                        continue
                    page, in_page = divmod(slot, block)
                    target = offset + page * stride + in_page * heads * head_bytes
                    raw[target:target + heads * head_bytes] = packed[i].flatten()
                    # PR's logical sink and recent segments form a disjoint
                    # union. A sink token has only its sink snapshot.
                    locations = [int(positions[i])] if positions[i] < sink else [sink + in_page % recent]
                    for row in locations:
                        target = (page * raw_stride + row * heads * dim) * 2
                        length = heads * dim * 2
                        raw_k[target:target + length] = key[i].view(torch.uint8).flatten()
                        raw_v[target:target + length] = value[i].view(torch.uint8).flatten()
                        target = (page * tag_stride + row) * 8
                        tags[target:target + 8] = torch.tensor([in_page], dtype=torch.int64).view(torch.uint8)
                values = (("key", key), ("value", value), ("rk", rk.T), ("rv", rv.T),
                    ("slots", slots), ("positions", positions), ("expected_packed", raw),
                    ("expected_raw_key", raw_k), ("expected_raw_value", raw_v),
                    ("expected_tags", tags), ("expected_status", status))
                for name, tensor in values:
                    _write(path / f"{name}.bin", tensor)
                (path / "shape.txt").write_text(
                    f"{n} {heads} {dim} 2 {block} {blocks} {offset} {stride} "
                    f"{raw_stride} {raw_stride} {tag_stride} {sink} {recent} "
                    f"{clip} {clip} {int(hadamard)}\n")
                cases.append({"op": "rotate_store", "path": str(path)})
    # Numerical and address failures must publish a distinct status and leave
    # the corresponding cache bytes untouched, even when graph padding is NaN.
    path = root / "rotate_store_errors"
    path.mkdir(exist_ok=True)
    dim, n, heads, block, blocks, sink, recent = 64, 6, 2, 128, 1, 2, 4
    rotation = torch.eye(dim)
    key = torch.randn(n, heads, dim, generator=generator).bfloat16()
    value = torch.randn(n, heads, dim, generator=generator).bfloat16()
    key[0] = torch.nan
    key[1] = 0
    key[2] = torch.nan
    slots = torch.tensor([-1, 0, 1, 2, 128, 3], dtype=torch.int64)
    positions = torch.tensor([-1, 0, 1, -1, 128, 3], dtype=torch.int64)
    offset, stride = 256, block * heads * (dim // 2 + 8) + 256
    window = sink + recent
    raw_stride, tag_stride = window * heads * dim + 32, window + 4
    raw = torch.full((offset + blocks * stride,), 173, dtype=torch.uint8)
    raw_k = torch.full((blocks * raw_stride * 2,), 173, dtype=torch.uint8)
    raw_v = torch.full_like(raw_k, 173)
    tags = torch.full((blocks * tag_stride * 8,), 173, dtype=torch.uint8)
    valid = encode_kv(key[5].float(), value[5].float())
    address = offset + 3 * heads * (dim // 2 + 8)
    raw[address:address + valid.numel()] = valid.flatten()
    address = (sink + 3 % recent) * heads * dim * 2
    raw_k[address:address + heads * dim * 2] = key[5].view(torch.uint8).flatten()
    raw_v[address:address + heads * dim * 2] = value[5].view(torch.uint8).flatten()
    address = (sink + 3 % recent) * 8
    tags[address:address + 8] = torch.tensor([3], dtype=torch.int64).view(torch.uint8)
    status = torch.tensor([0, 3, 2, 4, 1, 0], dtype=torch.int32)[:, None].expand(n, heads)
    values = (("key", key), ("value", value), ("rk", rotation), ("rv", rotation),
        ("slots", slots), ("positions", positions), ("expected_packed", raw),
        ("expected_raw_key", raw_k), ("expected_raw_value", raw_v),
        ("expected_tags", tags), ("expected_status", status))
    for name, tensor in values:
        _write(path / f"{name}.bin", tensor)
    (path / "shape.txt").write_text(
        f"{n} {heads} {dim} 2 {block} {blocks} {offset} {stride} "
        f"{raw_stride} {raw_stride} {tag_stride} {sink} {recent} 0 0 0\n")
    cases.append({"op": "rotate_store", "path": str(path)})
    # Short sink-only input is essential: in a longer chunk a later recent
    # write could overwrite and conceal a redundant sink copy in the ring.
    path = root / "rotate_store_sink_only"
    path.mkdir(exist_ok=True)
    n, heads, dim, block, blocks, sink, recent = 2, 2, 64, 128, 1, 4, 4
    key = torch.randn(n, heads, dim, generator=generator).bfloat16()
    value = torch.randn(n, heads, dim, generator=generator).bfloat16()
    rotation = torch.eye(dim)
    slots = positions = torch.arange(n, dtype=torch.int64)
    offset, stride = 256, block * heads * (dim // 2 + 8) + 256
    window = sink + recent
    raw_stride, tag_stride = window * heads * dim + 32, window + 4
    raw = torch.full((offset + stride,), 173, dtype=torch.uint8)
    raw_k = torch.full((raw_stride * 2,), 173, dtype=torch.uint8)
    raw_v = torch.full_like(raw_k, 173)
    tags = torch.full((tag_stride * 8,), 173, dtype=torch.uint8)
    encoded = encode_kv(key.float(), value.float()).flatten()
    raw[offset:offset + encoded.numel()] = encoded
    raw_k[:key.numel() * 2] = key.view(torch.uint8).flatten()
    raw_v[:value.numel() * 2] = value.view(torch.uint8).flatten()
    tags[:n * 8] = slots.view(torch.uint8)
    for name, tensor in (("key", key), ("value", value), ("rk", rotation), ("rv", rotation),
            ("slots", slots), ("positions", positions), ("expected_packed", raw),
            ("expected_raw_key", raw_k), ("expected_raw_value", raw_v), ("expected_tags", tags),
            ("expected_status", torch.zeros(n, heads, dtype=torch.int32))):
        _write(path / f"{name}.bin", tensor)
    (path / "shape.txt").write_text(
        f"{n} {heads} {dim} 2 {block} {blocks} {offset} {stride} "
        f"{raw_stride} {raw_stride} {tag_stride} {sink} {recent} 0 0 0\n")
    cases.append({"op": "rotate_store", "path": str(path)})
    (root / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(generate(parser.parse_args().output), indent=2))
