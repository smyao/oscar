"""Independent CPU goldens for an isolated active-row rotate/store experiment.

Archive #86/#97: identity is an explicit test artifact, never a replacement for
calibrated rotations. #111: input dtype is serialized exactly. #126/#145:
tile/padding transitions need real async validation beyond these CPU fixtures.
#155: compare BF16 NaN poison and signed zeros as bytes, not float equality.
D.4 four questions: (1) phase1_stores; (2) old writes cost 208.6-216.0ms/16K
while full-history dequant cost 6499.8-6655.1ms/card; (3) bounded offline new
token fixtures only, independent PR rotate/quantize oracle; (4) no device time,
model quality or service gain is inferred from fixture generation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import torch

from oscar_ascend.ops import reference


SHAPE_FIELDS = (
    "tokens", "heads", "dim", "dtype", "block_tokens", "blocks", "offset",
    "page_stride", "key_stride", "value_stride", "tag_stride", "sink",
    "recent", "key_clip", "value_clip", "hadamard",
)
DTYPES = {"fp16": (torch.float16, 1), "bf16": (torch.bfloat16, 2)}
PACKED_POISON = 0xAD
RAW_POISON_BITS = 0x7FC5  # BF16 quiet NaN, deliberately not float-equal to itself.
TAG_POISON_BITS = 0xADADADADADADADAD


def _payload(tensor: torch.Tensor) -> bytes:
    return bytes(tensor.detach().cpu().contiguous().view(torch.uint8).flatten().tolist())


def _write(path: Path, tensor: torch.Tensor) -> dict:
    data = _payload(tensor)
    path.write_bytes(data)
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _pattern(size: int, values: tuple[int, ...]) -> torch.Tensor:
    if size % len(values):
        raise ValueError("poison pattern must divide buffer size")
    return torch.tensor(values, dtype=torch.uint8).repeat(size // len(values))


def _striped(canonical: torch.Tensor, dim: int) -> torch.Tensor:
    """Arrange PR's two-bit integers into the separately specified striped ABI.

    D64 and D256 use [K words, V words, Ks/Kz, Vs/Vz]; a little-endian
    uint16 word holds columns i, D/8+i, ..., 7*D/8+i, low code first.
    """
    count, half = dim // 4, dim // 4 + 4
    result = torch.empty(dim // 2 + 8, dtype=torch.uint8)
    for side in range(2):
        old, new = side * half, side * count
        codes = reference.unpack_int2(canonical[old:old + count], dim)
        groups = codes.to(torch.int32).reshape(8, dim // 8).T
        words = (groups << (2 * torch.arange(8))).sum(-1)
        result[new:new + count:2] = words & 255
        result[new + 1:new + count:2] = words >> 8
        result[dim // 2 + side * 4:dim // 2 + side * 4 + 4] = canonical[old + count:old + half]
        # The permutation must recover the independent PR codes exactly.
        recovered = ((words[:, None] >> (2 * torch.arange(8))) & 3).T.flatten()
        assert torch.equal(recovered.to(torch.uint8), codes)
    return result


def _side(raw: torch.Tensor, rotation: torch.Tensor, clip: float) -> tuple[int, torch.Tensor | None]:
    if not bool(torch.isfinite(raw).all()):
        return 2, None
    rotated = reference.rotate_clip(raw, rotation)
    # Store checks the pre-clip FP32 row reduction, not only each element.
    # Large positive finite BF16 values deliberately overflow this reduction.
    if not bool(torch.isfinite(rotated.sum())):
        return 2, None
    clipped = reference.rotate_clip(raw, rotation, clip) if clip else rotated
    try:
        reference.quantize_int2(clipped)
    except ValueError:
        return 3, None
    return 0, clipped


def _inputs(n: int, heads: int, dim: int, dtype_name: str, mode: str,
            block: int, blocks: int) -> tuple[torch.Tensor, ...]:
    dtype, _ = DTYPES[dtype_name]
    seed = 601006 + n * 31 + dim * 7 + heads * 3 + int(dtype_name == "bf16")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    key = torch.randn(n, heads, dim, generator=generator).to(dtype)
    value = (torch.randn(n, heads, dim, generator=generator) * 1.25).to(dtype)
    slots = torch.arange(n, dtype=torch.int64)
    positions = slots.clone()
    if mode == "signed_zero":
        key[..., 0], key[..., 1] = -0.0, 0.0
        value[..., 0], value[..., 1] = 0.0, -0.0
        # All nonpositive rows test zero signs in dense-I reductions; exact
        # raw windows must retain source signs even when rotated zero differs.
        key[1::2] = -key[1::2].abs()
        value[1::2] = -value[1::2].abs()
        key[0] = -0.0  # Degenerate metadata remains status 3, not success.
    elif mode == "faults":
        key[0] = torch.nan; slots[0] = -1; positions[0] = -1
        key[1] = torch.inf; slots[1] = blocks * block; positions[1] = -1
        key[2] = torch.nan; positions[2] = -1
        key[3, 0, 0] = torch.nan
        value[4, :, 1] = torch.inf
        key[5, :, 2] = -torch.inf
        key[6] = 0.0; value[6] = torch.nan  # K metadata 3 precedes V numeric 2.
        key[7] = torch.nan; value[7] = 0.0
        positions[8] = -1
        slots[9] = -7; value[9] = -torch.inf
        slots[10] = blocks * block + 17; key[10] = 0.0
        key[11, 0] = torch.nan
        value[12, heads - 1] = torch.nan
    elif mode in ("partial_padding", "suffix_padding", "all_padding",
                  "mtp_q1_padding", "mtp_q4_padding"):
        count = {"partial_padding": 3, "suffix_padding": 16, "all_padding": n,
                 "mtp_q1_padding": n - 1, "mtp_q4_padding": n - 4}[mode]
        slots[-count:] = -1; positions[-count:] = -1
        key[-count:] = torch.nan; value[-count:] = torch.inf
    elif mode == "ring_pages":
        slots += block - 21
        positions = slots.clone()
    elif mode == "large_finite":
        if dtype_name == "bf16":
            key[0] = 2.0 ** 126  # Finite BF16, guaranteed FP32 row-sum overflow.
            key[1, :, ::2] = 2.0 ** 40
            key[1, :, 1::2] = -(2.0 ** 40)  # Finite sum, FP16 metadata overflow.
        else:
            key[0] = 65504.0  # Finite sum; constant-vector FP16 scale becomes zero.
            key[1, :, ::2] = 65504.0
            key[1, :, 1::2] = -65504.0  # Full FP16 range remains representable.
            value[1, :, ::2] = 65504.0
            value[1, :, 1::2] = -65504.0
    elif mode in ("decode_q4_hadamard", "decode_q1_identity"):
        if (n, heads, dim, dtype_name, blocks) != (128, 1, 256, "bf16", 32):
            raise ValueError("decode fixtures require N128/Hkv1/D256/BF16/32 pages")
        query = 4 if mode == "decode_q4_hadamard" else 1
        slots.fill_(-1); positions.fill_(-1)
        for request in range(32):
            # Every request owns one distinct physical page and one short
            # contiguous interval. The first in-page slot is never 8-aligned.
            in_page = 17 + 8 * (request % 13)
            if in_page + query > block:
                raise ValueError("decode interval crosses a physical page")
            context = (128 + 11 * request) * block + in_page
            begin = request * query
            slots[begin:begin + query] = request * block + in_page + torch.arange(query)
            positions[begin:begin + query] = context + torch.arange(query)
        if query == 1:
            key[32:] = torch.nan; value[32:] = torch.inf
    elif mode not in ("normal", "clip"):
        raise ValueError(f"unknown mode {mode}")
    return key, value, slots, positions


def write_case(root: Path, *, n: int, heads: int, dim: int, dtype_name: str,
               mode: str, block: int) -> dict:
    _, dtype_code = DTYPES[dtype_name]
    blocks = 32 if mode.startswith("decode_") else 2
    offset, sink, recent = 13, 3, 5
    head_bytes, window = dim // 2 + 8, sink + recent
    page_stride = block * heads * head_bytes + 17
    key_stride = window * heads * dim + 16
    value_stride = window * heads * dim + 32
    tag_stride = window + 3
    k_clip, v_clip = (0.75, 0.95) if mode == "clip" else (0.0, 0.0)
    key, value, slots, positions = _inputs(n, heads, dim, dtype_name, mode, block, blocks)
    hadamard = mode == "decode_q4_hadamard"
    if hadamard:
        # Explicit Sylvester H/sqrt(D), a real non-identity FP32 matrix.
        # D256 normalization is exactly representable as 1/16.
        rotation = torch.ones((1, 1), dtype=torch.float32)
        while rotation.shape[0] < dim:
            rotation = torch.cat((torch.cat((rotation, rotation), 1),
                                  torch.cat((rotation, -rotation), 1)), 0)
        rotation = rotation * (dim ** -0.5)
    else:
        rotation = torch.eye(dim, dtype=torch.float32)
    status = torch.zeros((n, heads), dtype=torch.int32)
    packed = torch.full((offset + blocks * page_stride,), PACKED_POISON, dtype=torch.uint8)
    canonical_arena = packed.clone()
    raw_key = _pattern(blocks * key_stride * 2, (RAW_POISON_BITS & 255, RAW_POISON_BITS >> 8))
    raw_value = _pattern(blocks * value_stride * 2, (RAW_POISON_BITS & 255, RAW_POISON_BITS >> 8))
    tags = torch.full((blocks * tag_stride * 8,), PACKED_POISON, dtype=torch.uint8)
    initial = {"packed": packed.clone(), "raw_key": raw_key.clone(),
               "raw_value": raw_value.clone(), "tags": tags.clone(),
               "status": torch.full_like(status, 0x55555555)}
    published_rows, last_window_writers = 0, {}
    for token in range(n):
        slot, position = int(slots[token]), int(positions[token])
        # Address and position outcomes override numeric outcomes in the
        # original published status contract. These rows never write cache.
        if slot < 0:
            continue
        if slot >= blocks * block:
            status[token] = 1; continue
        if position < 0:
            status[token] = 4; continue
        page, in_page = divmod(slot, block)
        for head in range(heads):
            k_status, k = _side(key[token, head], rotation, k_clip)
            v_status, v = _side(value[token, head], rotation, v_clip)
            code = k_status or v_status
            status[token, head] = code
            if code:
                continue
            canonical = reference.encode_kv(k, v)
            encoded = _striped(canonical, dim)
            destination = offset + page * page_stride + (in_page * heads + head) * head_bytes
            packed[destination:destination + head_bytes] = encoded
            canonical_arena[destination:destination + head_bytes] = canonical
            published_rows += 1
            # Original native-interval ownership contract: a same-page
            # token+recent slot suppresses this raw writer, independently of
            # that later row's numerical status. Quantization above remains
            # the independent PR oracle. Do not use future status to invent
            # a different fallback writer after a failed row.
            keep_recent = position >= sink
            if keep_recent and token + recent < n:
                later = int(slots[token + recent])
                if later >= 0 and later // block == page and later == slot + recent:
                    keep_recent = False
            if position >= sink and not keep_recent:
                continue
            index = position if position < sink else sink + in_page % recent
            for source, target, stride in ((key, raw_key, key_stride),
                                            (value, raw_value, value_stride)):
                destination = (page * stride + (index * heads + head) * dim) * 2
                raw = source[token, head].to(torch.bfloat16).view(torch.uint8)
                target[destination:destination + dim * 2] = raw
            if head == 0:
                destination = (page * tag_stride + index) * 8
                tags[destination:destination + 8] = torch.tensor([in_page], dtype=torch.int64).view(torch.uint8)
            last_window_writers[f"{page}:{index}:{head}"] = token
    name = f"n{n}_h{heads}_d{dim}_{dtype_name}_b{block}_{mode}"
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    values = {"key": key, "value": value, "rk": rotation.T, "rv": rotation.T,
              "slots": slots, "positions": positions, "expected_packed": packed,
              "expected_canonical_packed": canonical_arena, "expected_raw_key": raw_key,
              "expected_raw_value": raw_value, "expected_tags": tags, "expected_status": status,
              **{"initial_" + name: tensor for name, tensor in initial.items()}}
    files = {name + ".bin": _write(path / (name + ".bin"), tensor) for name, tensor in values.items()}
    shape = (n, heads, dim, dtype_code, block, blocks, offset, page_stride,
             key_stride, value_stride, tag_stride, sink, recent, k_clip, v_clip, int(hadamard))
    (path / "shape.txt").write_text(" ".join(map(str, shape)) + "\n")
    metadata = {
        "name": name, "path": str(path), "mode": mode,
        "shape_fields": dict(zip(SHAPE_FIELDS, shape)), "input_dtype": dtype_name,
        "production_binding_geometry": block % 128 == 0,
        "production_launch_grid": min(32, max(1, (n * heads + 7) // 8)),
        "binding_evidence": "csrc/rotation_bindings.cpp Cores and RotateStoreStriped page geometry checks",
        "rotation": "explicit FP32 normalized Sylvester Hadamard, stored transposed" if hadamard
                    else "explicit FP32 identity, stored transposed (identity unchanged)",
        "poison": {"packed_byte": PACKED_POISON, "raw_bf16_bits": RAW_POISON_BITS,
                   "tag_u64_bits_hex": hex(TAG_POISON_BITS), "status_i32": 0x55555555,
                   "compare": "exact bytes including NaN payload, signed zero, prefix and page gaps"},
        "files": files, "status_histogram": dict(sorted(Counter(status.flatten().tolist()).items())),
        "published_rows": published_rows, "last_window_writers": last_window_writers,
        "raw_source": "unrotated input converted once to BF16, source signed zeros preserved",
        "oracle": "independent ops.reference FP32 matrix rotation/clip/quantize; separately specified striped permutation",
        "status_precedence": "negative slot=0; out-of-range=1; negative position=4; K numeric/metadata then V=2/3",
        "window_domain": "original token+recent same-page/slot+recent suppression; successful final ring owners",
        "cann": "not_run", "npu": "not_run", "graph": "not_run", "performance": "not_run",
    }
    if mode.startswith("decode_"):
        query = 4 if hadamard else 1
        metadata["decode"] = {"requests": 32, "query_rows_per_request": query,
                              "active_tokens": 32 * query, "padding_tokens": n - 32 * query,
                              "physical_pages": 32, "one_request_per_page": True,
                              "first_positions": [int(positions[r * query]) for r in range(32)],
                              "first_in_page": [int(slots[r * query]) % block for r in range(32)]}
        metadata["decode"]["geometry_scope"] = "production binding geometry" if block % 128 == 0 else "page-layout stress only"
    (path / "fixture.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def generate(root: Path) -> dict:
    if sys.byteorder != "little":
        raise RuntimeError("these CANN fixture files require a little-endian host")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    root.mkdir(parents=True, exist_ok=True)
    specs = [dict(n=n, heads=heads, dim=dim, dtype_name=dtype, mode="normal",
                  block=129 if n == 64 else 257)
             for n in (64, 128) for heads in (1, 2) for dim in (64, 256) for dtype in DTYPES]
    for dim in (64, 256):
        for dtype in DTYPES:
            for mode in ("clip", "signed_zero", "faults", "partial_padding", "suffix_padding",
                         "all_padding", "ring_pages", "large_finite"):
                specs.append(dict(n=64, heads=2, dim=dim, dtype_name=dtype, mode=mode,
                                  block=257 if mode == "ring_pages" else 129))
    for n in (64, 128):
        for mode in ("mtp_q1_padding", "mtp_q4_padding"):
            specs.append(dict(n=n, heads=2, dim=256, dtype_name="bf16", mode=mode,
                              block=129 if n == 64 else 257))
    for block in (129, 128):
        for mode in ("decode_q4_hadamard", "decode_q1_identity"):
            specs.append(dict(n=128, heads=1, dim=256, dtype_name="bf16", mode=mode, block=block))
    cases = [write_case(root, **spec) for spec in specs]
    manifest = {"status": "independent_cpu_fixtures_generated", "cases": cases,
                "case_count": len(cases), "shape_fields": SHAPE_FIELDS,
                "archive": ["86", "97", "111", "126", "145", "155", "startup D.4"],
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "reference_sha256": hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest(),
                "cann": "not_run", "npu": "not_run", "graph": "not_run", "performance": "not_run"}
    (root / "cases.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/active_identity_store_20261006"))
    result = generate(parser.parse_args().output)
    print(json.dumps({"status": result["status"], "cases": result["case_count"]}))
